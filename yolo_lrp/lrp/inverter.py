# trunk-ignore(black-py)
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch

from .relevance import LayerRelevance
from .rules import RULE_REGISTRY, ConvRule, LinearRule, PropRule

logger = logging.getLogger(__name__)

# What flows through invert(): a LayerRelevance at a top-level module_list
# entry, or a plain Tensor once a TopLevelPropFunc has scattered it (see
# block_rules.py's TopLevelPropFunc/NestedPropFunc split).
RelevanceLike = Union[LayerRelevance, torch.Tensor]

# Constrained TypeVar so invert()/__call__ are type-preserving (Tensor in
# -> Tensor out, LayerRelevance in -> LayerRelevance out) without a cast
# at every call site.
RelevanceT = TypeVar("RelevanceT", LayerRelevance, torch.Tensor)

# inv_funcs' real value type is block_rules.PropFunc, but that can't be
# imported here (circular import - see block_rules.py). Callable[...,
# RelevanceLike] is the loosest honest type from this side.
InvFunc = Callable[..., RelevanceLike]

# Layer types whose backward pass is the identity (f(x) = x) - invert()
# returns relevance unchanged for these rather than consulting a table.
IDENTITY_MAPPINGS = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.ReLU,
    torch.nn.modules.activation.ReLU,
    torch.nn.ELU,
    torch.nn.Flatten,
    torch.nn.Dropout,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.Softmax,
    torch.nn.LogSoftmax,
    torch.nn.Sigmoid,
    torch.nn.SiLU,
    torch.nn.Identity,  # YOLO26's Detect.dfl under default reg_max=1 ("DFL removal")
)


class Inverter:
    """
    Owns two per-layer-type backward-dispatch tables and the `invert()`
    method that consults them: `rules_by_layer_type` for primitive layers
    (Conv/Linear/MaxPool/Upsample, via `PropRule` - see rules.py) and
    `inv_funcs` for composite YOLO blocks (C3/C2f/SPPF/..., via prop_*
    functions - see block_rules.py). Forward-hook bookkeeping lives in
    fwd_hooks.py/YOLOLRP instead - this class only computes backward
    relevance.

    Attributes
    ----------

    linear_rule : LinearRule
        Propagation rule for linear layers.

    conv_rule : ConvRule
        Propagation rule for convolutional layers.

    rules_by_layer_type : Dict[Type[torch.nn.Module], PropRule]
        Backward dispatch table for primitive layers, resolved once in
        __init__ from RULE_REGISTRY.

    inv_funcs : Dict[Type[torch.nn.Module], InvFunc]
        Backward dispatch table for composite YOLO blocks, populated
        externally via `register_inv_func`.

    pass_not_implemented : bool
        Return relevance unchanged for an unhandled layer type instead of
        raising.

    prop_to : List[int]
        Layer indices Detect's cv3 heads propagate relevance back to - set
        externally by YOLOLRP, read by `block_rules.prop_Detect`.
    """

    def __init__(
        self,
        linear_rule: Optional[PropRule] = None,
        conv_rule: Optional[PropRule] = None,
        pass_not_implemented: bool = False,
    ) -> None:
        """
        Resolves `rules_by_layer_type` from RULE_REGISTRY, keyed to the
        given conv/linear rule instances where those are needed.

        Arguments
        ---------

        linear_rule : PropRule, optional
            Rule instance for Linear layers. Omitted means Linear layers
            are simply left out of `rules_by_layer_type`, not defaulted.

        conv_rule : PropRule, optional
            Rule instance for Conv1d/2d/3d layers. Same omission behavior
            as `linear_rule`.

        pass_not_implemented : bool
            See class docstring.

        Returns
        -------

            None
        """

        self.linear_rule = linear_rule
        self.conv_rule = conv_rule
        self.prop_to: List[int] = []

        # RULE_REGISTRY maps layer type -> PropRule subclass. ConvRule/
        # LinearRule need the caller's configured instance (kept, not
        # copied, so later mutation e.g. self.conv_rule.contrastive still
        # applies); every other rule class gets one shared default
        # instance. Resolved to a flat type -> instance dict once here
        # rather than on every invert() call.
        instances_by_class: Dict[Type[PropRule], PropRule] = {}
        for rule in (conv_rule, linear_rule):
            if rule is not None:
                instances_by_class[type(rule)] = rule

        self.rules_by_layer_type: Dict[Type[torch.nn.Module], PropRule] = {}
        for layer_type, rule_cls in RULE_REGISTRY.items():
            if (
                rule_cls in (ConvRule, LinearRule)
                and rule_cls not in instances_by_class
            ):
                continue  # deliberately omitted - see __init__'s docstring
            if rule_cls not in instances_by_class:
                instances_by_class[rule_cls] = rule_cls()
            self.rules_by_layer_type[layer_type] = instances_by_class[rule_cls]

        self.inv_funcs: Dict[Type[torch.nn.Module], InvFunc] = {}
        self.pass_not_implemented = pass_not_implemented

    def register_inv_func(
        self, module: Type[torch.nn.Module], inv_func: InvFunc
    ) -> None:
        """
        Registers the backward relevance-propagation function to dispatch
        to for a composite block type.

        Arguments
        ---------

        module : Type[torch.nn.Module]
            Module type to register `inv_func` against.

        inv_func : InvFunc
            Function computing that type's backward relevance propagation
            (see block_rules.PropFunc for the precise shape).

        Returns
        -------

            None
        """

        if module in self.inv_funcs:
            logger.warning("Replacing previous inverse registered for %s", module)

        self.inv_funcs[module] = inv_func

    def invert(
        self, layer: torch.nn.Module, relevance: RelevanceT, **kwargs: Any
    ) -> RelevanceT:
        """
        Computes the backward pass through `layer` for `relevance`, trying
        in order: the primitive-layer dispatch table, the composite-block
        dispatch table, then special cases neither table covers
        (Sequential recursion, identity layer types).

        Arguments
        ---------

        layer : torch.nn.Module
            Layer to propagate relevance through.

        relevance : LayerRelevance or torch.Tensor
            Incoming relevance from higher up in the network.

        Returns
        -------

        LayerRelevance or torch.Tensor
            Redistributed relevance for the lower layers, in whichever
            shape it came in as.

        Raises
        ------

        NotImplementedError
            If no dispatch table or special case covers `layer`'s type
            and `pass_not_implemented` is False.
        """

        rule = self.rules_by_layer_type.get(type(layer))
        if rule is not None:
            return rule(layer, relevance, **kwargs)

        if type(layer) in self.inv_funcs:
            # InvFunc can't express "returns the type it was given" - this
            # cast asserts the same type-preserving contract documented on
            # invert() itself.
            return cast(
                RelevanceT,
                self.inv_funcs[type(layer)](self, layer, relevance, **kwargs),
            )

        if isinstance(layer, torch.nn.modules.container.Sequential):
            # Sequential's __getitem__ stub doesn't type slices as
            # iterable - it is one at runtime.
            reversed_layers = cast(Iterable[torch.nn.Module], layer[::-1])
            for sub_layer in reversed_layers:
                relevance = self.invert(sub_layer, relevance)
            return relevance

        if type(layer) in IDENTITY_MAPPINGS:
            return relevance

        if self.pass_not_implemented:
            return relevance

        raise NotImplementedError(
            f"Relevance propagation not implemented for layer type {type(layer)}"
        )

    def __call__(
        self, layer: torch.nn.Module, relevance: RelevanceT, **kwargs: Any
    ) -> RelevanceT:
        """
        Alias for `invert`.

        Arguments
        ---------

        layer : torch.nn.Module
            Layer to propagate relevance through.

        relevance : LayerRelevance or torch.Tensor
            Incoming relevance.

        Returns
        -------

        LayerRelevance or torch.Tensor
            Redistributed relevance, in whichever shape it came in as.
        """

        return self.invert(layer, relevance, **kwargs)
