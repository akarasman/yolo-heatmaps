import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import torch

from ..lrp.fwd_hooks import DEFAULT_FWD_HOOKS, FwdHookFunc, _silent_pass
from ..lrp.inverter import Inverter
from ..lrp.relevance import LayerRelevance, RelevanceMessage, scale_key
from ..lrp.rules import ConvRule, LinearRule
from .block_rules import PROP_REGISTRY, PropFunc
from .fwd_hooks import FWD_HOOK_REGISTRY

logger = logging.getLogger(__name__)


class YOLOLRP:
    """
    Generate layerwise relevance propagation per-pixel explanation for
    classification result of a PyTorch Ultralytics YOLO model.

    (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140).

    This wraps an ultralytics `YOLO` model rather than being one itself: it owns no
    parameters of its own, only a reference to the wrapped model plus the LRP
    bookkeeping (registered propagation rules, forward hooks, cached relevance).
    See each method's own docstring for the actual public/private API surface.

    Generic across Ultralytics YOLO versions, not tied to one architecture:
    which block types map to which propagation/hook functions is injected
    via `prop_registry`/`fwd_hook_registry` (defaulting to YOLO26's own -
    see `block_rules.PROP_REGISTRY`/`fwd_hooks.FWD_HOOK_REGISTRY`) rather
    than hardcoded here, and the few genuinely structural assumptions about
    the detection head (`_get_prop_to`, `_get_cls_preds`) are broken out
    into their own overridable methods rather than inlined. A different
    version whose registry is a straightforward data swap doesn't need a
    subclass at all - just construct with different registries. One is
    only worth writing if a version's head is structurally different
    enough that `_get_prop_to`/`_get_cls_preds` need real overriding, not
    just different registry data.

    ATTENTION:
        Currently, generating heatmaps for a network only works if all
        layers that have to be inverted are specified explicitly
        and registered as a module. If for example,
        the functional max_poolnd is used, the inversion will not work.
    """

    def __init__(
        self,
        model: Any,
        prop_registry: Optional[Dict[Type[torch.nn.Module], PropFunc]] = None,
        fwd_hook_registry: Optional[Dict[Type[torch.nn.Module], FwdHookFunc]] = None,
        contrastive: bool = False,
        power: int = 1,
        positive: bool = True,
        eps: float = 1e-6,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Constructs the LRP explainer around an already-loaded ultralytics
        YOLO model, building the conv/linear propagation rules and
        registering the forward hooks and relevance-propagation functions
        for every module type `prop_registry`/`fwd_hook_registry` cover.

        Arguments
        ---------

        model : Any
            ultralytics YOLO model to wrap and explain (untyped: ultralytics
            ships no type stubs). `model.model.model` must be the underlying
            nn.Sequential of layers.

        prop_registry : dict, optional
            Module type -> backward relevance-propagation function. Defaults
            to YOLO26's own registry (`block_rules.PROP_REGISTRY`). Pass a
            different registry to target a different YOLO version whose
            block types are otherwise a drop-in data swap.

        fwd_hook_registry : dict, optional
            Module type -> forward hook function. Defaults to YOLO26's own
            registry (`fwd_hooks.FWD_HOOK_REGISTRY`).

        contrastive : bool
            Initial contrastive setting for the conv/linear propagation
            rules. Each `explain()` call overrides this with its own
            `contrastive` argument before propagating, so this only matters
            if the rules are ever used before the first `explain()` call.

        power : int
            Exponent applied to inputs/weights by the propagation rules.

        positive : bool
            Whether the propagation rules truncate negative activations to
            zero.

        eps : float
            Small constant added to denominators in the propagation rules
            to avoid division by zero.

        device : torch.device
            Initial device for the wrapped model and propagator state.

        Returns
        -------

            None
        """

        # `model` is an ultralytics.YOLO wrapper; ultralytics ships no type
        # stubs, so it's typed as Any rather than torch.nn.Module (its
        # `.model.model` is the actual nn.Sequential of layers).
        self.model = model
        self.device = device
        self.prediction: Optional[torch.Tensor] = None
        self.r_values: Optional[List[Any]] = None

        conv_rule = ConvRule(
            power=power, positive=positive, eps=eps, contrastive=contrastive
        )
        linear_rule = LinearRule(
            power=power, positive=positive, eps=eps, contrastive=contrastive
        )
        self.save_r_values = False

        # Initialize the 'Relevance Propagator' with the chosen rule.
        # This will be used to back-propagate the relevance values
        # through the layers.
        self.inverter = Inverter(
            linear_rule=linear_rule,
            conv_rule=conv_rule,
            pass_not_implemented=True,
        )

        # Detect's `from` indices (which head-branch layer numbers its cv3
        # heads propagate back to). Overridable - see _get_prop_to - since
        # this assumes an Ultralytics-parsed-yaml-config-shaped model.
        self.inverter.prop_to = self._get_prop_to()

        self._register_inv_funcs(
            prop_registry if prop_registry is not None else PROP_REGISTRY
        )

        # Forward-hook lookup table for this instance: block-specific
        # overrides (YOLO-version-specific, injectable) merged over the
        # architecture-independent primitive defaults. Owned here, not by
        # Inverter - attaching forward hooks to the live model is a
        # YOLOLRP concern, and Inverter's own job (invert()) never reads
        # this table.
        self._fwd_hooks: Dict[Type[torch.nn.Module], FwdHookFunc] = {
            **DEFAULT_FWD_HOOKS,
            **(
                fwd_hook_registry
                if fwd_hook_registry is not None
                else FWD_HOOK_REGISTRY
            ),
        }

        # Parsing the individual model layers
        self._hook_handles: List[Any] = []
        self.module_list: List[torch.nn.Module] = []
        self._attach_forward_hooks(self.model.model.model)
        self._index_module_list(self.model.model.model)

    def _get_prop_to(self) -> List[int]:
        """
        Returns the layer indices Detect's cv3 heads propagate relevance
        back to. Default reads them from the model's own parsed Ultralytics
        config instead of a hardcoded magic list, so it keeps working if
        Ultralytics reorders these layers within a version (YOLO26: [16,
        19, 22]; the old YOLOv8 architecture this library used to target
        had [15, 18, 21] here instead). Override for a version whose model
        object doesn't expose the same `model.model.yaml["head"]` shape.

        Arguments
        ---------

            None

        Returns
        -------

        List[int]
            One entry per detection scale.
        """

        return cast(List[int], self.model.model.yaml["head"][-1][0])

    def to(self, device: Union[str, torch.device]) -> "YOLOLRP":
        """
        Move the wrapped model and propagator state to the given device.

        Arguments
        ---------

        device : str or torch.device
            Device to move to, e.g. "cuda", "cuda:1", "cpu", or a torch.device.

        Returns
        -------

            YOLOLRP
                self, for chaining.
        """

        device = torch.device(device) if isinstance(device, str) else device
        self.device = device
        self.model.model.to(device)
        return self

    def _index_module_list(self, entry_point: torch.nn.Module) -> None:
        """
        Flattens the model's top-level children into self.module_list,
        tagging each with its registration number for later backward
        traversal.

        Arguments
        ---------

        entry_point : torch.nn.Module
            Module whose direct children should be indexed.

        Returns
        -------

            None
        """

        for mod in entry_point.children():
            setattr(mod, "reg_num", len(self.module_list))
            self.module_list.append(mod)

    def _remove_hooks(self) -> None:
        """
        Removes any forward/backward hooks previously attached by this
        instance.

        Arguments
        ---------

            None

        Returns
        -------

            None
        """

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _attach_forward_hooks(self, parent_module: torch.nn.Module) -> None:
        """
        Recursively registers the forward hooks that save input and output
        tensors for later computation of relevance distribution. Hook handles
        are tracked on `self._hook_handles` so a later `explain()` call can
        remove and re-attach them via the public torch API, instead of
        probing torch's private `_forward_hooks` bookkeeping to check for
        an existing registration.

        Arguments
        ---------

        parent_module : torch.nn.Module
            Model to register hooks for.

        Returns
        -------

            None
        """

        for mod in parent_module.children():

            if list(mod.children()):
                self._attach_forward_hooks(mod)

            self._hook_handles.append(
                mod.register_forward_hook(self._get_fwd_hook(mod))
            )

            # Special case for ReLU layer
            if isinstance(mod, torch.nn.ReLU) or isinstance(
                mod, torch.nn.modules.activation.ReLU
            ):
                self._hook_handles.append(
                    mod.register_full_backward_hook(self._relu_hook_function)
                )

    def _get_fwd_hook(self, mod: torch.nn.Module) -> FwdHookFunc:
        """
        Looks up the forward hook to attach for `mod` in this instance's
        merged table (block-specific overrides + primitive defaults - see
        `self._fwd_hooks` in `__init__`), falling back to a silent no-op
        for any module type neither covers.

        Arguments
        ---------

        mod : torch.nn.Module
            Module a forward hook is about to be attached to.

        Returns
        -------

        FwdHookFunc
            The registered hook for `type(mod)`, or a no-op hook if none
            is registered.
        """

        return self._fwd_hooks.get(type(mod), _silent_pass)

    def _register_inv_funcs(
        self, inv_funcs: Optional[Dict[Type[torch.nn.Module], PropFunc]] = None
    ) -> None:
        """
        Registers, per module type, the backward relevance-propagation
        function the Inverter should dispatch to for composite YOLO
        blocks.

        Arguments
        ---------

        inv_funcs : dict, optional
            Mapping of module type to the function that computes its
            backward relevance propagation.

        Returns
        -------

            None
        """

        for mod, inv_func in (inv_funcs or {}).items():
            self.inverter.register_inv_func(mod, inv_func)

    @staticmethod
    def _relu_hook_function(
        module: torch.nn.Module,
        grad_in: Tuple[torch.Tensor, ...],
        grad_out: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """
        If there is a negative gradient, change it to zero.

        Arguments
        ---------

        module : torch.nn.Module
            The ReLU module the hook fired on (unused, required by torch's
            backward hook signature).

        grad_in : Tuple[torch.Tensor, ...]
            Gradient(s) with respect to the module's input.

        grad_out : Tuple[torch.Tensor, ...]
            Gradient(s) with respect to the module's output (unused,
            required by torch's backward hook signature).

        Returns
        -------

        Tuple[torch.Tensor, ...]
            The clamped input gradient, as a single-element tuple.
        """

        return (torch.clamp(grad_in[0], min=0.0),)

    @staticmethod
    def _suppress_outliers(
        relevance: torch.Tensor, quantile: float = 0.98
    ) -> torch.Tensor:
        """
        Zero out the top `1 - quantile` fraction of relevance values (extreme
        outliers), keeping the rest unchanged.

        Arguments
        ---------

        relevance : torch.Tensor
            Relevance tensor to suppress outliers in.

        quantile : float
            Values at or above this quantile are zeroed out.

        Returns
        -------

        torch.Tensor
            The relevance tensor with extreme values zeroed out.
        """

        threshold = torch.quantile(relevance, quantile)
        return torch.where(
            relevance < threshold,
            relevance,
            torch.tensor(0.0, device=relevance.device),
        )

    def __call__(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """
        The explanation wrapper returns the same prediction as the
        original model, but wraps the model call method in the evaluate
        method to save the last prediction.

        Arguments
        ---------

        in_tensor : torch.Tensor
            Model input to pass through the pytorch model.

        Returns
        -------

        torch.Tensor
            Model output
        """

        return self.evaluate(in_tensor)

    def evaluate(self, in_tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Evaluates the model on a new input. The registered forward hooks
        will save all the data that is necessary to compute the relevance
        per neuron per layer.

        Arguments
        ---------

        in_tensor : torch.Tensor
            New input for which to predict an output.

        **kwargs : Any
            Forwarded to the wrapped model's own `__call__`.

        Returns
        -------

        torch.Tensor
            Model prediction
        """

        self.prediction = self.model.model(in_tensor.unsqueeze(0), **kwargs)
        return self.prediction

    def get_layer_relevance(self) -> Optional[List[Any]]:
        """
        Get relevance snapshots per layer in the network.

        Arguments
        ---------

            None

        Returns
        -------

        list
            list of relevance snapshots

        """

        if self.r_values is None:
            logger.warning(
                "No relevances have been calculated yet, returning None "
                "in get_layer_relevance."
            )
        return self.r_values

    def explain(
        self,
        frame: torch.Tensor,
        cls: Optional[Union[int, str]] = None,
        max_class_only: bool = True,
        contrastive: bool = False,
        primal_intensity: float = 0.5,
        dual_intensity: float = 0.5,
    ) -> torch.Tensor:
        """
        Method for generating an explanatort heatmap for the model with
        the LRP rule chosen at the initialization of the module.

        Arguments
        ---------

        frame : torch.Tensor
            Input frame for which to evaluate the LRP algorithm.

        cls : int or str, optional
            Index (or name) of the class for which the relevance distribution
            is to be analyzed. If None, the 'winning' class is used for
            indexing.

        max_class_only : bool
            Forwarded to the relevance initializer; whether to zero out all
            but the winning class's activations before seeding relevance.

        contrastive : bool
            Whether to compute contrastive (primal vs. dual) relevance
            instead of plain LRP. This is the single source of truth for
            contrastive-ness: it's applied to the propagation rules for the
            duration of this call, overriding whatever they were left at by
            a previous call or the constructor's own `contrastive` default.

        primal_intensity : float
            Weight applied to the primal (class of interest) relevance when
            combining primal and dual relevance. Only used if contrastive is True.

        dual_intensity : float
            Weight applied to the dual (contrasted-away) relevance when
            combining primal and dual relevance. Only used if contrastive is True.

        Returns
        -------

        torch.Tensor
            2D explanation map of relevance over the input frame.
        """

        self.r_values = None

        if isinstance(cls, str):
            cls = list(self.model.names.values()).index(cls)

        self._remove_hooks()
        self._attach_forward_hooks(self.model.model.model)

        return self._propagate_relevance(
            frame,
            cls,
            max_class_only,
            contrastive,
            primal_intensity,
            dual_intensity,
        )

    def _get_cls_preds(self, frame: torch.Tensor) -> List[torch.Tensor]:
        """
        Runs the model forward and returns Detect's per-scale classification
        predictions - the raw material `_initialize_relevance` seeds
        relevance from. Default assumes a Detect-style head exposed as
        `model.model.model[-1].cv3`, one branch per detection scale, each
        branch's final layer forward-hooked with a cached `.out_tensor`.
        Override for a version whose head isn't shaped this way (e.g. a
        different Detect variant, or one that doesn't expose `.cv3` the
        same way).

        Arguments
        ---------

        frame : torch.Tensor
            Input frame to run the model forward on.

        Returns
        -------

        List[torch.Tensor]
            One classification-branch prediction tensor per detection
            scale, sigmoid-activated.

        Raises
        ------

        RuntimeError
            If `Detect.cv3` is missing - almost always means the wrapped
            model was fused for standalone inference (e.g. by a prior
            `model(...)`/`model.predict(...)` call on the same `YOLO`
            instance, or the same instance passed to more than one
            `YOLOLRP`), which replaces `cv3` with None. Construct
            `YOLOLRP` from a freshly-loaded model that's never been used
            for a raw predict/call.
        """

        self.model.model.predict(frame.unsqueeze(0))
        cv3 = self.model.model.model[-1].cv3
        if cv3 is None:
            raise RuntimeError(
                "Detect.cv3 is None - the wrapped model appears to have "
                "been fused for standalone inference (e.g. by a prior "
                "model(...)/model.predict(...) call on the same YOLO "
                "instance). YOLOLRP needs the unfused module tree; "
                "reload the weights fresh (YOLO(...)) and construct "
                "YOLOLRP from that instance before calling predict/__call__ "
                "on it directly."
            )
        return [cv3_branch[-1].out_tensor.sigmoid() for cv3_branch in cv3]

    def _propagate_relevance(
        self,
        frame: torch.Tensor,
        cls: Optional[int],
        max_class_only: bool,
        contrastive: bool,
        primal_intensity: float,
        dual_intensity: float,
    ) -> torch.Tensor:
        """
        Runs the model forward, then the LRP backward pass layer by layer
        through the (already hook-populated) reversed module list, and
        combines the resulting per-class relevance into a single explanation
        map. Caches per-layer relevance snapshots on `self.r_values` as it goes.

        Arguments
        ---------

        frame : torch.Tensor
            Input frame for which to evaluate the LRP algorithm.

        cls : int, optional
            Index of the class of interest, already resolved from any class
            name by the caller. If None, the winning class is used.

        max_class_only : bool
            Forwarded to the relevance initializer.

        contrastive : bool
            Whether to propagate contrastive (primal vs. dual) relevance.
            Applied to the conv/linear propagation rules for this call, so
            they can't silently disagree with what the rest of the pipeline
            is doing.

        primal_intensity : float
            Forwarded to `_finalize_relevance`.

        dual_intensity : float
            Forwarded to `_finalize_relevance`.

        Returns
        -------

        torch.Tensor
            2D explanation map of relevance over the input frame.
        """

        # The conv/linear rules are shared, long-lived objects (built once in
        # __init__ and referenced by self.inverter), so their contrastive
        # flag has to be kept in sync with this call's own `contrastive`
        # argument every time - otherwise a later non-default explain() call
        # would silently propagate with a stale contrastive setting from
        # construction time or a previous call.
        # Inverter itself types these Optional (a bare Inverter can be built
        # without either), but __init__ above always constructs and passes
        # both - the asserts document that invariant for readers and mypy.
        assert self.inverter.conv_rule is not None
        assert self.inverter.linear_rule is not None
        self.inverter.conv_rule.contrastive = contrastive
        self.inverter.linear_rule.contrastive = contrastive

        with torch.no_grad():

            # We have to iterate through the model backwards.
            rev_model: List[torch.nn.Module] = self.module_list[::-1]

            cls_preds = self._get_cls_preds(frame)
            relevance: LayerRelevance = _initialize_relevance(
                cls_preds,
                cls=cls,
                max_class_only=max_class_only,
                contrastive=contrastive,
            )

            # List to save relevance distributions per layer
            self.r_values = [relevance]
            for layer in rev_model:
                # Compute layer specific backwards-propagation of relevance values
                # reg_num is set on every layer in module_list by
                # _index_module_list before this loop ever runs (see
                # setattr(mod, "reg_num", ...) above) - cast, not getattr
                # with a fallback, since pop_cache needs a real int here.
                relevance.pop_cache(cast(int, getattr(layer, "reg_num")))
                self.r_values.append(relevance)
                relevance = self.inverter(layer, relevance)

            if self.save_r_values:
                # LayerRelevance only does scatter/gather, cache, and
                # print - snapshotting to numpy for debugging is this
                # (sole) caller's job, reading the cache directly.
                own = relevance.cache.get(-1)
                self.r_values.append(
                    own.relevance.cpu().numpy()
                    if own is not None
                    else torch.tensor([]).numpy()
                )

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            lrp_out = relevance.scatter(-1)
            return self._finalize_relevance(
                lrp_out, contrastive, primal_intensity, dual_intensity
            )

    def _finalize_relevance(
        self,
        lrp_out: torch.Tensor,
        contrastive: bool,
        primal_intensity: float,
        dual_intensity: float,
    ) -> torch.Tensor:
        """
        Collapses the scattered per-channel relevance for the input layer
        into a single 2D explanation map. In contrastive mode, the primal
        (class of interest) and dual (contrasted-away) relevance are each
        outlier-suppressed before being combined; otherwise the primal
        relevance is summed across channels first and suppressed after.

        Arguments
        ---------

        lrp_out : torch.Tensor
            Scattered relevance at the input layer, batch dim 0 holding the
            primal relevance and (in contrastive mode) batch dim 1 holding
            the dual relevance.

        contrastive : bool
            Whether to combine primal and dual relevance, or just return the
            (outlier-suppressed) primal relevance on its own.

        primal_intensity : float
            Weight applied to the primal relevance. Only used if contrastive
            is True.

        dual_intensity : float
            Weight applied to the dual relevance. Only used if contrastive
            is True.

        Returns
        -------

        torch.Tensor
            2D explanation map.
        """

        if contrastive:
            lrp_p = self._suppress_outliers(lrp_out[[0]])
            lrp_d = self._suppress_outliers(lrp_out[[1]])
            return (primal_intensity * lrp_p - dual_intensity * lrp_d).sum(dim=1)[0]
        else:
            lrp_p = lrp_out[[0]].sum(dim=1)[0]
            return self._suppress_outliers(lrp_p)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """
        Evaluates model on a given input tensor.

        Arguments
        ---------

        in_tensor : torch.Tensor
            Model input tensor.

        Returns
        -------

        torch.Tensor
            Model output tensor.
        """

        return cast(torch.Tensor, self.model(in_tensor))

    def extra_repr(self) -> str:
        """
        Returns the wrapped model's own extra_repr(), for convenient
        introspection of what's being explained.

        Arguments
        ---------

            None

        Returns
        -------

        str
            The wrapped model's extra_repr() output.
        """

        return cast(str, self.model.extra_repr())


def _initialize_relevance(
    cls_preds: List[torch.Tensor],
    cls: Optional[int] = None,
    max_class_only: bool = False,
    contrastive: bool = False,
) -> LayerRelevance:
    """
    Builds the initial per-scale relevance that `YOLOLRP._propagate_relevance`
    seeds its backward pass from, out of the raw class-prediction tensors at
    each detection scale. Private to this module - only used internally by
    `YOLOLRP`, not part of the library's public API. A plain function,
    not a class: nothing here is ever reused across multiple calls (a fresh
    call is made once per `explain()`, with no state carried between calls),
    so there's nothing a class/`__call__` split would buy over just passing
    every argument directly.

    Arguments
    ---------

    cls_preds : List[torch.Tensor]
        One classification-branch prediction tensor per detection scale
        (Detect's `cv3` outputs, already sigmoid-activated).

    cls : int, optional
        Index of the class of interest. If None, all classes contribute
        (subject to `max_class_only`) and the 'winning' class per location
        effectively drives the relevance.

    max_class_only : bool
        Zero all output activations from classes that are not the max,
        before any class-of-interest filtering.

    contrastive : bool
        Whether to build a dual (primal vs. contrasted-away) relevance
        batch instead of a single one. Requires `cls` to be set.

    Returns
    -------

    LayerRelevance
        Initial relevance, one message per detection scale (keyed via
        `scale_key` - the scales have differing spatial shapes and can't
        be merged into a single message).
    """

    if contrastive:
        assert (
            cls is not None
        ), "Contrastive implementation of lrp requires target class specification"

    relevance = LayerRelevance(contrastive=contrastive)
    for j, cls_pred in enumerate(cls_preds):

        # Keep only max class outputs (the rest may be discarded as noise)
        if max_class_only:
            max_class, i = cls_pred.max(dim=1, keepdim=True)
            cls_pred = torch.zeros_like(cls_pred).scatter(1, i, max_class)

        # Filter out only class of interest
        if cls is None:
            relevance.gather(
                RelevanceMessage(from_=None, to=scale_key(j), relevance=cls_pred)
            )
            continue

        # Construct dual relevance
        if contrastive:
            dual = cls_pred.clone()
            dual[:, cls] = 0.0
            cls_pred = torch.cat([cls_pred, dual], dim=0)
        else:
            cls_pred[:, :cls] = 0.0
            cls_pred[:, cls + 1 :] = 0.0

        relevance.gather(
            RelevanceMessage(from_=None, to=scale_key(j), relevance=cls_pred)
        )

    return relevance
