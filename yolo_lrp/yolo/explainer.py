import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn.functional as F
from ultralytics.utils.nms import non_max_suppression

from ..lrp.fwd_hooks import DEFAULT_FWD_HOOKS, FwdHookFunc, _silent_pass
from ..lrp.inverter import Inverter
from ..lrp.relevance import LayerRelevance, RelevanceMessage, scale_key
from ..lrp.rules import ConvRule, LinearRule
from .block_rules import PROP_REGISTRY, PropFunc
from .fwd_hooks import FWD_HOOK_REGISTRY

logger = logging.getLogger(__name__)

# Background grid cells have every class at float noise (~1e-7), not
# truly zero - without this floor, max_class_only's argmax would still
# "win" there by a meaningless margin.
_MAX_CLASS_ONLY_FLOOR = 0.01


class YOLOLRP:
    """
    LRP/CRP explainer for an ultralytics YOLO detector
    (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140).

    Wraps a YOLO model rather than being one - owns no parameters, just
    the wrapped model plus LRP bookkeeping. Generic across YOLO versions:
    block-to-rule mappings come from `prop_registry`/`fwd_hook_registry`
    (default: YOLO26's own), and the structural assumptions about the
    detection head are isolated in `_get_prop_to`/`_get_cls_preds` for
    overriding.

    Diverges from the YOLO-specific CRP formulation `_initialize_relevance`
    follows in two ways: no object-confidence weighting (modern YOLO heads
    have no separate objectness score), and it adds its own extra levers
    (`seed_strategy`, `seed_dilate`, `_MAX_CLASS_ONLY_FLOOR`) the
    formulation doesn't have.

    ATTENTION: every layer needing inversion must be a registered module -
    a functional op used directly (e.g. `F.max_pool2d`) won't be caught.
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
        Builds the propagation rules and registers forward hooks +
        relevance rules for every module type the registries cover.

        Arguments
        ---------

        model : Any
            Ultralytics YOLO wrapper (untyped - no stubs); `model.model.model`
            must be the underlying nn.Sequential.

        prop_registry : dict, optional
            Module type -> propagation function. Defaults to YOLO26's own.

        fwd_hook_registry : dict, optional
            Module type -> forward hook function. Defaults to YOLO26's own.

        contrastive : bool
            Initial contrastive setting for the propagation rules.

        power : int
            Exponent applied to inputs/weights by the propagation rules.

        positive : bool
            Whether to truncate negative activations to zero.

        eps : float
            Stabilizing epsilon for the propagation rules.

        device : torch.device
            Initial device for the wrapped model and propagator state.

        Returns
        -------

            None
        """

        self.model = model
        self.device = device
        self.prediction: Optional[torch.Tensor] = None
        self.r_values: Optional[
            List[Tuple[Optional[torch.nn.Module], torch.Tensor]]
        ] = None

        conv_rule = ConvRule(
            power=power, positive=positive, eps=eps, contrastive=contrastive
        )
        linear_rule = LinearRule(
            power=power, positive=positive, eps=eps, contrastive=contrastive
        )
        # Off by default - real per-layer snapshotting isn't free.
        self.save_r_values = False

        self.inverter = Inverter(
            linear_rule=linear_rule,
            conv_rule=conv_rule,
            pass_not_implemented=True,
        )

        # Detect's cv3 heads propagate back to these layer indices -
        # overridable via _get_prop_to for a differently-shaped head.
        self.inverter.prop_to = self._get_prop_to()

        self._register_inv_funcs(
            prop_registry if prop_registry is not None else PROP_REGISTRY
        )

        self._fwd_hooks: Dict[Type[torch.nn.Module], FwdHookFunc] = {
            **DEFAULT_FWD_HOOKS,
            **(
                fwd_hook_registry
                if fwd_hook_registry is not None
                else FWD_HOOK_REGISTRY
            ),
        }

        self._hook_handles: List[Any] = []
        self.module_list: List[torch.nn.Module] = []
        self._attach_forward_hooks(self.model.model.model)
        self._index_module_list(self.model.model.model)

    def _get_prop_to(self) -> List[int]:
        """
        Layer indices Detect's cv3 heads propagate relevance back to,
        read from the model's own parsed config. Override for a
        differently-shaped head.

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
        Moves the wrapped model and propagator state to `device`.

        Arguments
        ---------

        device : str or torch.device
            Device to move to.

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
        Flattens `entry_point`'s children into `self.module_list`,
        tagging each with its registration number.

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
        Removes hooks previously attached by this instance.

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
        Recursively attaches forward hooks that cache each module's
        input/output for relevance propagation. ReLU also gets a
        backward hook clamping negative gradients.

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

            if isinstance(mod, torch.nn.ReLU) or isinstance(
                mod, torch.nn.modules.activation.ReLU
            ):
                self._hook_handles.append(
                    mod.register_full_backward_hook(self._relu_hook_function)
                )

    def _get_fwd_hook(self, mod: torch.nn.Module) -> FwdHookFunc:
        """
        Forward hook registered for `mod`'s type, or a no-op if none is.

        Arguments
        ---------

        mod : torch.nn.Module
            Module a forward hook is about to be attached to.

        Returns
        -------

        FwdHookFunc
            The registered hook, or a no-op.
        """

        return self._fwd_hooks.get(type(mod), _silent_pass)

    def _register_inv_funcs(
        self, inv_funcs: Optional[Dict[Type[torch.nn.Module], PropFunc]] = None
    ) -> None:
        """
        Registers each module type's backward relevance-propagation
        function with the inverter.

        Arguments
        ---------

        inv_funcs : dict, optional
            Module type -> propagation function.

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
        Clamps negative input gradients to zero.

        Arguments
        ---------

        module : torch.nn.Module
            Unused (required by torch's hook signature).

        grad_in : Tuple[torch.Tensor, ...]
            Gradient(s) w.r.t. the module's input.

        grad_out : Tuple[torch.Tensor, ...]
            Unused (required by torch's hook signature).

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
        Clips relevance above `quantile` down to that threshold instead
        of zeroing it - the top values are the model's strongest signal,
        not noise.

        Arguments
        ---------

        relevance : torch.Tensor
            Relevance tensor to clip.

        quantile : float
            Values above this quantile are clipped down to it.

        Returns
        -------

        torch.Tensor
            The clipped relevance tensor.
        """

        threshold = torch.quantile(relevance, quantile)
        return torch.clamp(relevance, max=threshold)

    def __call__(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """
        Alias for `evaluate`.

        Arguments
        ---------

        in_tensor : torch.Tensor
            Model input.

        Returns
        -------

        torch.Tensor
            Model output.
        """

        return self.evaluate(in_tensor)

    def evaluate(self, in_tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Runs the model forward, populating the forward-hook-cached
        tensors relevance propagation needs.

        Arguments
        ---------

        in_tensor : torch.Tensor
            New input to predict on.

        **kwargs : Any
            Forwarded to the wrapped model's own `__call__`.

        Returns
        -------

        torch.Tensor
            Model prediction.
        """

        self.prediction = self.model.model(in_tensor.unsqueeze(0), **kwargs)
        return self.prediction

    def get_layer_relevance(
        self,
    ) -> Optional[List[Tuple[Optional[torch.nn.Module], torch.Tensor]]]:
        """
        Per-layer relevance snapshots from the last `explain()` call, in
        backward order, each a real detached clone at that layer's own
        resolution. Final `(None, ...)` entry is the network's own input.
        `None` unless `explain()` ran with `save_r_values=True`.

        Arguments
        ---------

            None

        Returns
        -------

        list of (torch.nn.Module or None, torch.Tensor), optional
            One (layer, relevance) pair per layer, plus a final
            (None, relevance) entry for the network's own input.
        """

        if self.r_values is None:
            logger.warning(
                "No relevance snapshots available, returning None in "
                "get_layer_relevance - either explain() hasn't been "
                "called yet, or it was called with save_r_values=False."
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
        seed_strategy: str = "confidence",
        seed_threshold: float = 0.25,
        seed_gamma: float = 0.5,
        seed_dilate: int = 0,
        localize_bbox: bool = True,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> torch.Tensor:
        """
        Runs LRP/CRP and returns a 2D relevance map for `cls` (or the
        winning class, if None) over `frame`.

        Arguments
        ---------

        frame : torch.Tensor
            Input frame to explain.

        cls : int or str, optional
            Class to explain, by index or name. None uses the winning class.

        max_class_only : bool
            Zero out non-winning classes before seeding relevance.

        contrastive : bool
            Compute contrastive (CRP) relevance instead of plain LRP.

        primal_intensity : float
            Weight for primal relevance when combining contrastive output.

        dual_intensity : float
            Weight for dual relevance when combining contrastive output.

        seed_strategy : str
            How raw confidence becomes seed weight - "confidence",
            "binary", or "compressed" (see `_apply_seed_strategy`).

        seed_threshold : float
            Confidence floor for "binary"/"compressed"; ignored by "confidence".

        seed_gamma : float
            Compression exponent for "compressed"; ignored otherwise.

        seed_dilate : int
            Grid cells to grow the seeded region by (see `_dilate_seed`).
            0 is a no-op.

        localize_bbox : bool
            Gate relevance to the class's own post-NMS box before
            propagation (paper eq 7).

        bbox : Tuple[float, float, float, float], optional
            Explicit box to use instead of auto-detecting one.

        Returns
        -------

        torch.Tensor
            2D explanation map.
        """

        self.r_values = None

        if isinstance(cls, str):
            names = list(self.model.names.values())
            if cls not in names:
                raise ValueError(
                    f"Unknown class {cls!r} - not one of this model's "
                    f"classes: {names}"
                )
            cls = names.index(cls)

        self._remove_hooks()
        self._attach_forward_hooks(self.model.model.model)

        return self._propagate_relevance(
            frame,
            cls,
            max_class_only,
            contrastive,
            primal_intensity,
            dual_intensity,
            seed_strategy,
            seed_threshold,
            seed_gamma,
            seed_dilate,
            localize_bbox,
            bbox,
        )

    def _get_cls_preds(
        self, frame: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Runs the model forward, returning per-scale classification
        confidences plus this frame's post-NMS detections (for
        `localize_bbox`). NMS runs on the raw output directly via
        `ultralytics.utils.nms.non_max_suppression`, not through the
        wrapper's own `.predict()` - that fuses Conv+BatchNorm on first
        use and would break every hook this class relies on.

        Arguments
        ---------

        frame : torch.Tensor
            Input frame to run the model forward on.

        Returns
        -------

        Tuple[List[torch.Tensor], torch.Tensor]
            Per-scale classification confidences, and post-NMS
            detections as one (N, 6) tensor - (x1, y1, x2, y2, conf,
            cls) per row.

        Raises
        ------

        RuntimeError
            If Detect.cv3 is missing - the model was fused for
            standalone inference (e.g. a prior predict()/__call__ on the
            same instance).
        """

        raw_preds, _ = self.model.model.predict(frame.unsqueeze(0))
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
        cls_preds = [cv3_branch[-1].out_tensor.sigmoid() for cv3_branch in cv3]

        is_end2end = raw_preds.shape[-1] == 6
        detections = non_max_suppression(
            raw_preds, conf_thres=0.25, end2end=is_end2end
        )[0]
        return cls_preds, detections

    def _resolve_bbox(
        self, detections: torch.Tensor, cls: Optional[int]
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Highest-confidence post-NMS box for `cls` in `detections`.

        Arguments
        ---------

        detections : torch.Tensor
            Post-NMS detections, (N, 6) - (x1, y1, x2, y2, conf, cls) per row.

        cls : int, optional
            Class of interest. None always resolves to no box.

        Returns
        -------

        Tuple[float, float, float, float], optional
            (x1, y1, x2, y2), or None if not detected.
        """

        if cls is None or detections.numel() == 0:
            return None

        matches = detections[detections[:, 5].long() == cls]
        if matches.shape[0] == 0:
            return None

        best = matches[matches[:, 4].argmax()]
        return cast(Tuple[float, float, float, float], tuple(best[:4].tolist()))

    def _propagate_relevance(
        self,
        frame: torch.Tensor,
        cls: Optional[int],
        max_class_only: bool,
        contrastive: bool,
        primal_intensity: float,
        dual_intensity: float,
        seed_strategy: str = "confidence",
        seed_threshold: float = 0.25,
        seed_gamma: float = 0.5,
        seed_dilate: int = 0,
        localize_bbox: bool = True,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> torch.Tensor:
        """
        Runs the LRP backward pass through the reversed module list and
        combines primal/dual relevance into a final map. Snapshots
        per-layer relevance onto `self.r_values` if `save_r_values` is set.

        Arguments
        ---------

        frame : torch.Tensor
            Input frame to explain.

        cls : int, optional
            Class of interest, already resolved from any name. None uses
            the winning class.

        max_class_only : bool
            Forwarded to `_initialize_relevance`.

        contrastive : bool
            Whether to propagate contrastive relevance; also synced onto
            the shared conv/linear rules.

        primal_intensity : float
            Forwarded to `_finalize_relevance`.

        dual_intensity : float
            Forwarded to `_finalize_relevance`.

        seed_strategy : str
            Forwarded to `_initialize_relevance`.

        seed_threshold : float
            Forwarded to `_initialize_relevance`.

        seed_gamma : float
            Forwarded to `_initialize_relevance`.

        seed_dilate : int
            Forwarded to `_initialize_relevance`.

        localize_bbox : bool
            Whether to resolve and apply a bounding-box gate.

        bbox : Tuple[float, float, float, float], optional
            Explicit box overriding auto-detection.

        Returns
        -------

        torch.Tensor
            2D explanation map.
        """

        # Shared, long-lived rule objects - keep their contrastive flag
        # in sync with this call, not a stale one from construction or a
        # previous call.
        assert self.inverter.conv_rule is not None
        assert self.inverter.linear_rule is not None
        self.inverter.conv_rule.contrastive = contrastive
        self.inverter.linear_rule.contrastive = contrastive

        with torch.no_grad():

            rev_model: List[torch.nn.Module] = self.module_list[::-1]

            cls_preds, detections = self._get_cls_preds(frame)

            resolved_bbox = bbox
            if resolved_bbox is None and localize_bbox:
                resolved_bbox = self._resolve_bbox(detections, cls)
                if resolved_bbox is None and cls is not None:
                    logger.warning(
                        f"localize_bbox=True but class {cls} has no "
                        "post-NMS detection in this frame - proceeding "
                        "without spatial localization."
                    )

            relevance: LayerRelevance = _initialize_relevance(
                cls_preds,
                cls=cls,
                max_class_only=max_class_only,
                contrastive=contrastive,
                seed_strategy=seed_strategy,
                seed_threshold=seed_threshold,
                seed_gamma=seed_gamma,
                seed_dilate=seed_dilate,
                bbox=resolved_bbox,
                image_size=(frame.shape[-2], frame.shape[-1]),
            )

            self.r_values = [] if self.save_r_values else None
            for layer in rev_model:
                relevance.pop_cache(cast(int, getattr(layer, "reg_num")))
                if self.r_values is not None:
                    # Clone before it's consumed below - LayerRelevance
                    # mutates its cache in place, so this is a real
                    # snapshot, not a live reference.
                    own = relevance.scatter(which=-1, destroy=False)
                    self.r_values.append((layer, own.detach().clone()))
                relevance = self.inverter(layer, relevance)

            if self.r_values is not None:
                own_final = relevance.scatter(which=-1, destroy=False)
                self.r_values.append((None, own_final.detach().clone()))

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
        Collapses per-channel relevance at the input layer into a single 2D map.

        Arguments
        ---------

        lrp_out : torch.Tensor
            Scattered relevance at the input layer; batch dim 0 is
            primal, dim 1 (if contrastive) is dual.

        contrastive : bool
            Whether to combine primal and dual relevance.

        primal_intensity : float
            Weight for primal relevance.

        dual_intensity : float
            Weight for dual relevance.

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
        Evaluates the wrapped model on `in_tensor`.

        Arguments
        ---------

        in_tensor : torch.Tensor
            Model input.

        Returns
        -------

        torch.Tensor
            Model output.
        """

        return cast(torch.Tensor, self.model(in_tensor))

    def extra_repr(self) -> str:
        """
        Wrapped model's own `extra_repr()`.

        Arguments
        ---------

            None

        Returns
        -------

        str
            The wrapped model's extra_repr() output.
        """

        return cast(str, self.model.extra_repr())


def _apply_seed_strategy(
    cls_pred: torch.Tensor,
    strategy: str,
    threshold: float,
    gamma: float,
) -> torch.Tensor:
    """
    Reweights raw classification confidence into relevance seed weight -
    raw confidence is sharply peaked near an object's centroid (YOLO's
    training only supervises a few central anchors), so seeding with it
    directly carries that peak through LRP's conservation almost unchanged.

    Arguments
    ---------

    cls_pred : torch.Tensor
        Raw sigmoid confidence at one detection scale, all classes present.

    strategy : str
        "confidence" (raw, unchanged), "binary" (flat 1.0 above
        `threshold`), or "compressed" (`confidence ** gamma` above
        `threshold`).

    threshold : float
        Confidence floor for "binary"/"compressed"; ignored by "confidence".

    gamma : float
        Compression exponent for "compressed"; ignored otherwise.

    Returns
    -------

    torch.Tensor
        Reweighted confidence, same shape as `cls_pred`.

    Raises
    ------

    ValueError
        If `strategy` isn't recognized.
    """

    if strategy == "confidence":
        return cls_pred
    elif strategy == "binary":
        return (cls_pred > threshold).to(cls_pred.dtype)
    elif strategy == "compressed":
        return torch.where(
            cls_pred > threshold,
            cls_pred.pow(gamma),
            torch.zeros_like(cls_pred),
        )
    else:
        raise ValueError(
            f"Unknown seed_strategy {strategy!r} - expected one of "
            "'confidence', 'binary', 'compressed'."
        )


def _dilate_seed(cls_pred: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Grows the seeded region by `radius` grid cells via max-pooling -
    controls how many locations are active, not how much weight each gets.

    Arguments
    ---------

    cls_pred : torch.Tensor
        Seed confidence for one detection scale.

    radius : int
        Grid cells to grow by in every direction. 0 is a no-op.

    Returns
    -------

    torch.Tensor
        Dilated seed confidence, same shape as `cls_pred`.
    """

    if radius <= 0:
        return cls_pred
    return F.max_pool2d(cls_pred, kernel_size=2 * radius + 1, stride=1, padding=radius)


def _reduce_to_max_class(cls_pred: torch.Tensor) -> torch.Tensor:
    """
    Keeps only the winning class's confidence per location, zeroing the
    rest (paper eq 6) - only if the winner clears
    `_MAX_CLASS_ONLY_FLOOR`, since background cells sit at float noise
    and would otherwise "win" by a meaningless margin.

    Arguments
    ---------

    cls_pred : torch.Tensor
        Per-class confidence for one detection scale.

    Returns
    -------

    torch.Tensor
        Same shape, non-winning (or sub-floor) classes zeroed.
    """

    max_class, i = cls_pred.max(dim=1, keepdim=True)
    max_class = torch.where(
        max_class > _MAX_CLASS_ONLY_FLOOR, max_class, torch.zeros_like(max_class)
    )
    return torch.zeros_like(cls_pred).scatter(1, i, max_class)


def _second_best_class(cls_pred: torch.Tensor, cls: int) -> torch.Tensor:
    """
    Best *other* class's confidence per location, excluding `cls` (paper
    eq 10). Computed before `_reduce_to_max_class` runs, so a location
    where `cls` is the winner still surfaces its real runner-up.

    Arguments
    ---------

    cls_pred : torch.Tensor
        Per-class confidence, not yet through `_reduce_to_max_class`.

    cls : int
        Class to exclude from the search.

    Returns
    -------

    torch.Tensor
        Same shape, only the best non-`cls` class survives per location.
    """

    masked = cls_pred.clone()
    masked[:, cls] = -1.0  # confidence is always >= 0, so this always loses the max
    best_other, i = masked.max(dim=1, keepdim=True)
    best_other = torch.where(
        best_other > _MAX_CLASS_ONLY_FLOOR, best_other, torch.zeros_like(best_other)
    )
    return torch.zeros_like(cls_pred).scatter(1, i, best_other)


def _localize_to_bbox(
    cls_pred: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Zeros every grid cell whose anchor point falls outside `bbox` (paper
    eq 7) - a stand-in for the cell's decoded box, since this port never
    reads Detect's box-regression branch.

    Arguments
    ---------

    cls_pred : torch.Tensor
        Per-class confidence for one detection scale.

    bbox : Tuple[float, float, float, float]
        (x1, y1, x2, y2) in the frame's pixel coordinates.

    image_size : Tuple[int, int]
        (height, width) `bbox` is expressed in.

    Returns
    -------

    torch.Tensor
        Same shape, classes zeroed outside `bbox`.
    """

    _, _, h, w = cls_pred.shape
    stride_y = image_size[0] / h
    stride_x = image_size[1] / w
    x1, y1, x2, y2 = bbox

    ys = (
        torch.arange(h, dtype=cls_pred.dtype, device=cls_pred.device) + 0.5
    ) * stride_y
    xs = (
        torch.arange(w, dtype=cls_pred.dtype, device=cls_pred.device) + 0.5
    ) * stride_x
    inside = ((ys >= y1) & (ys <= y2))[:, None] & ((xs >= x1) & (xs <= x2))[None, :]

    return cls_pred * inside.to(cls_pred.dtype)


def _initialize_relevance(
    cls_preds: List[torch.Tensor],
    cls: Optional[int] = None,
    max_class_only: bool = False,
    contrastive: bool = False,
    seed_strategy: str = "confidence",
    seed_threshold: float = 0.25,
    seed_gamma: float = 0.5,
    seed_dilate: int = 0,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> LayerRelevance:
    """
    Builds the per-scale seed relevance `_propagate_relevance` starts its
    backward pass from.

    Arguments
    ---------

    cls_preds : List[torch.Tensor]
        Per-scale classification confidence (Detect's cv3 outputs,
        sigmoid-activated).

    cls : int, optional
        Class of interest. None: whichever class wins at each location
        drives relevance.

    max_class_only : bool
        Gate primal relevance via `_reduce_to_max_class`. Never applied to dual.

    contrastive : bool
        Build dual relevance via `_second_best_class` (paper eq 10).
        Requires `cls`.

    seed_strategy : str
        Forwarded to `_apply_seed_strategy`.

    seed_threshold : float
        Forwarded to `_apply_seed_strategy`.

    seed_gamma : float
        Forwarded to `_apply_seed_strategy`.

    seed_dilate : int
        Forwarded to `_dilate_seed`.

    bbox : Tuple[float, float, float, float], optional
        Forwarded to `_localize_to_bbox`. None skips localization.

    image_size : Tuple[int, int], optional
        Required if `bbox` is given.

    Returns
    -------

    LayerRelevance
        Initial relevance, one message per detection scale.
    """

    if contrastive:
        assert (
            cls is not None
        ), "Contrastive implementation of lrp requires target class specification"

    relevance = LayerRelevance(contrastive=contrastive)
    for j, cls_pred in enumerate(cls_preds):

        cls_pred = _apply_seed_strategy(
            cls_pred, seed_strategy, seed_threshold, seed_gamma
        )
        cls_pred = _dilate_seed(cls_pred, seed_dilate)

        if bbox is not None:
            assert image_size is not None, "bbox given without image_size"
            cls_pred = _localize_to_bbox(cls_pred, bbox, image_size)

        if cls is None:
            if max_class_only:
                cls_pred = _reduce_to_max_class(cls_pred)
            relevance.gather(
                RelevanceMessage(from_=None, to=scale_key(j), relevance=cls_pred)
            )
            continue

        # Dual (eq 10) must come from cls_pred before max_class_only
        # collapses it - see _second_best_class.
        dual = _second_best_class(cls_pred, cls) if contrastive else None

        primal = _reduce_to_max_class(cls_pred) if max_class_only else cls_pred.clone()
        primal[:, :cls] = 0.0
        primal[:, cls + 1 :] = 0.0

        cls_pred = torch.cat([primal, dual], dim=0) if dual is not None else primal

        relevance.gather(
            RelevanceMessage(from_=None, to=scale_key(j), relevance=cls_pred)
        )

    return relevance
