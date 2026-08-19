# trunk-ignore(black-py)
import logging
from typing import (
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

# What actually flows through invert()/__call__: a LayerRelevance at a
# top-level module_list entry, a plain Tensor once a TopLevelPropFunc has
# scattered it (see block_rules.py's own TopLevelPropFunc/NestedPropFunc
# split for the full reasoning) - both real, both live, never just Tensor
# alone despite Inverter's own earlier signature having claimed that.
RelevanceLike = Union[LayerRelevance, torch.Tensor]

# invert()/__call__ are type-preserving, not just "accepts either, returns
# either" - passing a Tensor always gets a Tensor back, passing a
# LayerRelevance always gets a LayerRelevance back (every dispatch path,
# PropRule.propagate()/prop_* functions alike, follows this). A
# constrained TypeVar (rather than the plain RelevanceLike Union) lets
# mypy carry that correlation through every call site, instead of forcing
# a cast back to the specific type at every single one.
RelevanceT = TypeVar("RelevanceT", LayerRelevance, torch.Tensor)

# inv_funcs' real value type is block_rules.PropFunc, but inverter.py
# can't import block_rules (block_rules imports Inverter under
# TYPE_CHECKING; the reverse import at runtime would be circular - see
# block_rules.py's own note on this). Callable[..., RelevanceLike] is the
# loosest honest description from this side: something invoked as
# `f(self, layer, relevance, **kwargs)` returning relevance-shaped data.
InvFunc = Callable[..., RelevanceLike]

# Layer types whose backward pass is genuinely the identity (f(x) = x),
# so `invert()` returns relevance unchanged rather than consulting either
# dispatch table.
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
    # torch.nn.Identity is genuinely identity (f(x) = x). It's what
    # YOLO26's Detect.dfl becomes under the default reg_max=1 ("DFL
    # removal").
    torch.nn.Identity,
)


class Inverter:
    """
    Owns the two real per-layer-type backward-dispatch tables and the
    single `invert()` method that consults them: `rules_by_layer_type`
    for primitive layers (Conv/Linear/MaxPool/Upsample, via `PropRule`
    instances - see rules.py) and `inv_funcs` for composite YOLO blocks
    (C3/C2f/SPPF/..., via prop_* functions - see block_rules.py). Forward-
    hook bookkeeping is a separate concern owned by fwd_hooks.py/YOLOLRP,
    not by this class - Inverter only ever computes backward relevance.

    Attributes
    ----------

    linear_rule : LinearRule
        Propagation rule to use for linear layers

    conv_rule : ConvRule
        Propagation rule for convolutional layers

    rules_by_layer_type : Dict[Type[torch.nn.Module], PropRule]
        Backward dispatch table for primitive layers, keyed directly by
        torch layer type (resolved once, in __init__, from
        RULE_REGISTRY's type -> PropRule-subclass mapping).

    inv_funcs : Dict[Type[torch.nn.Module], InvFunc]
        Backward dispatch table for composite YOLO blocks, keyed by exact
        module type. Populated externally via `register_inv_func`
        (YOLOLRP does this from `block_rules.PROP_REGISTRY` or an
        injected equivalent).

    pass_not_implemented : bool
        Return relevance unchanged for a layer type neither dispatch
        table (nor any of `invert`'s other special cases) covers, instead
        of raising.

    prop_to : List[int]
        Layer indices Detect's cv3 heads propagate relevance back to -
        not set here (Inverter has no notion of "Detect" at all); bolted
        on externally by `YOLOLRP.__init__` and read by
        `block_rules.prop_Detect`. Declared here, defaulting to empty,
        purely so it's a real attribute rather than an ad hoc one mypy
        (and readers) can't see coming.
    """

    def __init__(
        self,
        linear_rule: Optional[PropRule] = None,
        conv_rule: Optional[PropRule] = None,
        pass_not_implemented: bool = False,
    ) -> None:
        """
        Arguments
        ---------

        linear_rule : PropRule, optional
            Configured rule instance for Linear layers. See class
            docstring. If omitted, Linear layers are simply left out of
            `rules_by_layer_type` (not silently given a default instance -
            unlike MaxPool/Upsample, Conv/Linear rules carry meaningful
            per-model configuration, so an omission here is assumed
            deliberate).

        conv_rule : PropRule, optional
            Configured rule instance for Conv1d/2d/3d layers. See
            `linear_rule` above; same omission behavior.

        pass_not_implemented : bool
            See class docstring.

        Returns
        -------

            None
        """

        self.linear_rule = linear_rule
        self.conv_rule = conv_rule
        self.prop_to: List[int] = []

        # RULE_REGISTRY maps torch layer type -> PropRule *subclass*
        # (e.g. {Conv2d: ConvRule, MaxPool2d: MaxPoolRule, ...}), since
        # which *kind* of rule applies to a layer never changes. Which
        # *instance* to use does, for ConvRule/LinearRule specifically -
        # conv_rule/linear_rule carry per-model configuration
        # (power/eps/positive/contrastive) supplied by the caller (same
        # objects, not copies, so mutating e.g. self.conv_rule.contrastive
        # afterwards is still reflected here). Every other rule class in
        # RULE_REGISTRY (MaxPoolRule, UpsampleRule, ...) needs no such
        # configuration, so it gets exactly one shared default instance.
        # Resolved down to a flat layer-type -> instance dict once, here,
        # rather than re-resolving subclass -> instance on every single
        # invert() call.
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
            Function computing that type's backward relevance
            propagation (see block_rules.PropFunc, the more precisely-
            shaped type actual prop_* functions are declared with).

        Returns
        -------

            None
        """

        if module in self.inv_funcs:
            logger.warning("Replacing previous inverse registered for %s", module)

        self.inv_funcs[module] = inv_func

    def invert(
        self, layer: torch.nn.Module, relevance: RelevanceT, **kwargs
    ) -> RelevanceT:
        """
        Computes the backward pass for the incoming relevance for the
        specified layer, trying (in order): the primitive-layer dispatch
        table, the composite-block dispatch table, then a handful of
        special cases neither table needs to know about (Sequential
        recursion, genuinely-identity layer types).

        Arguments
        ---------

        layer : torch.nn.Module
            Layer to propagate relevance through.

        relevance : LayerRelevance or torch.Tensor
            Incoming relevance from higher up in the network - a
            LayerRelevance at a top-level module_list entry, a plain
            Tensor once a TopLevelPropFunc has scattered it (see
            block_rules.py's TopLevelPropFunc/NestedPropFunc split).

        Returns
        -------

        LayerRelevance or torch.Tensor
            Redistributed relevance going to the lower layers in the
            network, in whichever of the two shapes it came in as.

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
            # InvFunc can't express "returns whatever type it was given"
            # (that's block_rules.PropFunc's job, which this module can't
            # import - see InvFunc's own note) - this cast asserts the
            # same type-preserving contract invert() documents everywhere
            # else, for this one dynamic dispatch path.
            return cast(
                RelevanceT,
                self.inv_funcs[type(layer)](self, layer, relevance, **kwargs),
            )

        if isinstance(layer, torch.nn.modules.container.Sequential):
            # torch's own Sequential.__getitem__ stub doesn't resolve the
            # slice overload as iterable - it really is one at runtime.
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
        self, layer: torch.nn.Module, relevance: RelevanceT, **kwargs
    ) -> RelevanceT:
        """Wrapper for invert method"""
        return self.invert(layer, relevance, **kwargs)
