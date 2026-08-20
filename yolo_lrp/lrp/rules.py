# trunk-ignore(isort)
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union, cast

import torch
import torch.nn.functional as F
from torch.nn import (
    Conv1d,
    Conv2d,
    Conv3d,
    Linear,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    Upsample,
)

from .protocols import ConvLike, LinearLike, MaxPoolLike, UpsampleLike
from .relevance import LayerRelevance, RelevanceMessage

# __call__/propagate are type-preserving (LayerRelevance in -> LayerRelevance
# out, Tensor in -> Tensor out - see propagate's own isinstance branch),
# same reasoning as Inverter's own RelevanceT in inverter.py. Can't import
# that one instead: inverter.py imports RULE_REGISTRY/ConvRule/LinearRule/
# PropRule from this module, so the reverse import would be circular.
RelevanceT = TypeVar("RelevanceT", LayerRelevance, torch.Tensor)


class PropRule(ABC):
    """
    Strategy interface for primitive-layer relevance propagation rules
    (e.g. ConvRule, LinearRule - see `select_rule`/`RULE_REGISTRY` for how
    Inverter picks which rule handles a given layer type).

    Implements the LayerRelevance-vs-plain-tensor handling shared by every
    rule as a template method (`propagate`); concrete rules only implement
    `compute`, the actual per-layer-type propagation math.

    Attributes
    ----------

    power : int / float
        Exponent to apply to input / weights

    eps : float
        Small number added to denominator to avoid divide by zero

    positive : bool
        Truncate negative activations of prev layer to zero

    contrastive : bool
        Compute dual relevance for contrastive
    """

    def __init__(
        self,
        power: int = 1,
        eps: float = 1e-6,
        positive: bool = True,
        contrastive: bool = True,
    ) -> None:
        """
        Stores the shared propagation config (see class docstring).

        Arguments
        ---------

        power : int
            See class docstring.

        eps : float
            See class docstring.

        positive : bool
            See class docstring.

        contrastive : bool
            See class docstring.

        Returns
        -------

            None
        """

        self.power = power
        self.eps = eps
        self.positive = positive
        self.contrastive = contrastive

    def __call__(self, module: torch.nn.Module, relevance: RelevanceT) -> RelevanceT:
        """
        Alias for `propagate`.

        Arguments
        ---------

        module : torch.nn.Module
            Module through which relevance is propagated.

        relevance : LayerRelevance or torch.Tensor
            Incoming relevance from the layer above.

        Returns
        -------

        LayerRelevance or torch.Tensor
            Redistributed relevance, same type as input.
        """

        return self.propagate(module, relevance)

    def propagate(self, module: torch.nn.Module, relevance: RelevanceT) -> RelevanceT:
        """
        Propagates relevance through `module` and redistributes it to the
        layer below.

        ATTENTION: mutates `relevance` in place when it's a LayerRelevance -
        each call morphs it into the output relevance, discarding what came
        before. Use LayerRelevance's own `cache()` to snapshot per-layer
        relevance if you need it.

        Arguments
        ---------

        module : torch.nn.Module
            Module through which relevance is propagated.

        relevance : LayerRelevance or torch.Tensor
            Incoming relevance from the layer above.

        Returns
        -------

        LayerRelevance or torch.Tensor
            Redistributed relevance, same type as input.
        """

        # Unwrap LayerRelevance to a plain tensor for compute(), then
        # re-wrap; plain tensors pass through untouched.
        if isinstance(relevance, LayerRelevance):
            msg = relevance.scatter(-1)
            msg = self.compute(module, msg)
            relevance.gather(
                RelevanceMessage(
                    from_=getattr(module, "reg_num", None),
                    to=-1,
                    relevance=msg,
                )
            )
            return relevance

        return self.compute(module, relevance)

    @abstractmethod
    def compute(
        self, module: torch.nn.Module, relevance_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Redistributes a plain relevance tensor backward through `module`,
        implementing the actual layer-type-specific LRP math. Called by
        `propagate` - never call this directly with a LayerRelevance.

        Arguments
        ---------

        module : torch.nn.Module
            Module through which relevance is propagated.

        relevance_in : torch.Tensor
            Incoming relevance, already unwrapped to a plain tensor.

        Returns
        -------

        torch.Tensor
            Redistributed relevance.
        """

        raise NotImplementedError


class LinearRule(PropRule):
    """
    Epsilon-rule relevance propagation for Linear layers, per
    http://iphome.hhi.de/samek/pdf/MonXAI19.pdf.
    """

    def _get_fwd_step(self) -> Callable[..., torch.Tensor]:
        """
        Forward step for the 4-step procedure: duplicates the input under
        contrastive mode, then calls `F.linear`.

        Arguments
        ---------

            None

        Returns
        -------

        Callable[..., torch.Tensor]
            Forward function taking (in_tensor, w, **kwargs).
        """

        def linear_wrapper(
            in_tensor: torch.Tensor, w: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor:
            if self.contrastive:
                x = torch.cat([in_tensor] * 2, dim=0)
            else:
                x = in_tensor
            return F.linear(x, w, **kwargs)

        return linear_wrapper

    def _get_bwd_step(self) -> Callable[..., torch.Tensor]:
        """
        Backward (de-conv-equivalent) step for the 4-step procedure.

        Arguments
        ---------

            None

        Returns
        -------

        Callable[..., torch.Tensor]
            Backward function taking (relevance_in, w, **kwargs).
        """

        def linear_wrapper(
            relevance_in: torch.Tensor, w: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor:
            return F.linear(relevance_in, w.t(), **kwargs)

        return linear_wrapper

    def compute(
        self, module: torch.nn.Module, relevance_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Redistributes relevance through a Linear layer via the standard
        4-step epsilon-rule procedure. See `PropRule.compute`.

        Arguments
        ---------

        module : torch.nn.Module
            Linear layer through which relevance is propagated.

        relevance_in : torch.Tensor
            Incoming relevance, already unwrapped to a plain tensor.

        Returns
        -------

        torch.Tensor
            Redistributed relevance.
        """

        linear = cast(LinearLike, module)

        linear_fwd = self._get_fwd_step()
        linear_bwd = self._get_bwd_step()

        x = linear.in_tensor
        x = x.pow(self.power)
        w = linear.weight.pow(self.power)

        # Step 1: forward pass with modified weights
        z = linear_fwd(x, w, bias=None)
        z = z + torch.sign(z) * self.eps
        relevance_in[z == 0] = 0
        z[z == 0] = 1

        # Step 2: divide incoming relevance by activation
        s = relevance_in / z

        # Step 3: backward (de-conv-equivalent) pass
        c = linear_bwd(s, w, bias=None)

        # Step 4: multiply by input
        relevance_out = c * x

        return relevance_out


class ConvRule(PropRule):
    """
    Epsilon-rule relevance propagation for Conv1d/2d/3d layers, per
    http://iphome.hhi.de/samek/pdf/MonXAI19.pdf.
    """

    def _get_fwd_step(
        self, m: Union[Conv1d, Conv2d, Conv3d]
    ) -> Callable[..., torch.Tensor]:
        """
        Forward step for the 4-step procedure: duplicates the input under
        contrastive mode, then calls the matching `F.convNd`.

        Arguments
        ---------

        m : Conv1d, Conv2d, or Conv3d
            Layer instance whose dimensionality selects the conv op.

        Returns
        -------

        Callable[..., torch.Tensor]
            Forward function taking (in_tensor, **kwargs).

        Raises
        ------

        Exception
            If `m` isn't a Conv1d/Conv2d/Conv3d instance.
        """

        try:
            conv = {Conv1d: F.conv1d, Conv2d: F.conv2d, Conv3d: F.conv3d}[type(m)]
        except KeyError:
            raise Exception("Layer must be one of {}".format((Conv1d, Conv2d, Conv3d)))

        # Duplicates input for contrastive (primal+dual) propagation.
        def conv_wrapper(in_tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            if self.contrastive:
                x = torch.cat([in_tensor, in_tensor], dim=0)
            else:
                x = in_tensor
            return conv(x, **kwargs)

        return conv_wrapper

    def _get_bwd_step(
        self, m: Union[Conv1d, Conv2d, Conv3d]
    ) -> Callable[..., torch.Tensor]:
        """
        Backward (transpose-conv) step for the 4-step procedure.

        Arguments
        ---------

        m : Conv1d, Conv2d, or Conv3d
            Layer instance whose dimensionality selects the transpose-conv op.

        Returns
        -------

        Callable[..., torch.Tensor]
            Backward function taking (relevance_in, **kwargs).

        Raises
        ------

        Exception
            If `m` isn't a Conv1d/Conv2d/Conv3d instance.
        """

        try:
            inv_conv = {
                Conv1d: F.conv_transpose1d,
                Conv2d: F.conv_transpose2d,
                Conv3d: F.conv_transpose3d,
            }[type(m)]
        except KeyError:
            raise Exception("Layer must be one of {}".format((Conv1d, Conv2d, Conv3d)))

        def inv_conv_wrapper(relevance_in: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return inv_conv(relevance_in, **kwargs)

        return inv_conv_wrapper

    def compute(
        self, module: torch.nn.Module, relevance_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Redistributes relevance through a Conv layer via the standard
        4-step epsilon-rule procedure. See `PropRule.compute`.

        Arguments
        ---------

        module : torch.nn.Module
            Conv layer through which relevance is propagated.

        relevance_in : torch.Tensor
            Incoming relevance, already unwrapped to a plain tensor.

        Returns
        -------

        torch.Tensor
            Redistributed relevance.
        """

        conv = cast(ConvLike, module)
        # Only ever registered against real Conv1d/2d/3d types (see
        # RULE_REGISTRY), never a plain torch.nn.Module.
        conv_nd = cast(Union[Conv1d, Conv2d, Conv3d], module)

        relevance_in = torch.cat(
            [r.view_as(conv.out_tensor) for r in relevance_in], dim=0
        )
        conv_fwd = self._get_fwd_step(conv_nd)
        conv_bwd = self._get_bwd_step(conv_nd)

        with torch.no_grad():

            # x stays at its real (single-pass) batch size; conv_fwd/
            # conv_bwd are the only place that duplicate it for
            # contrastive mode. Duplicating here too would double it
            # again, crashing on real batch != 1 inputs.
            x = conv.in_tensor.clone()
            w = conv.weight.clone()
            if self.positive:
                x = x.clamp(min=0)
                w = w.clamp(min=0)
            x = x.pow(self.power)
            w = w.pow(self.power)

            # Step 1: forward pass with modified weights
            z = conv_fwd(
                x,
                weight=w,
                bias=None,
                stride=conv.stride,
                padding=conv.padding,
                groups=conv.groups,
            )
            z = z + torch.sign(z) * self.eps
            relevance_in[z == 0] = 0
            z[z == 0] = 1

            # Step 2: divide incoming relevance by activation
            s = relevance_in / z

            # Step 3: backward (transpose-conv) pass
            if conv.stride != (1, 1):

                # stride != 1 can leave the naive transpose-conv output
                # smaller than the real input; compute the exact
                # output_padding needed to match it back up.
                _, _, H, W = relevance_in.size()
                Hnew = (
                    (H - 1) * conv.stride[0]
                    - 2 * conv.padding[0]
                    + conv.dilation[0] * (conv.kernel_size[0] - 1)
                    + conv.output_padding[0]
                    + 1
                )
                Wnew = (
                    (W - 1) * conv.stride[1]
                    - 2 * conv.padding[1]
                    + conv.dilation[1] * (conv.kernel_size[1] - 1)
                    + conv.output_padding[1]
                    + 1
                )
                _, _, Hin, Win = x.size()

                cp = conv_bwd(
                    s,
                    weight=w,
                    bias=None,
                    padding=conv.padding,
                    output_padding=(Hin - Hnew, Win - Wnew),
                    stride=conv.stride,
                    dilation=conv.dilation,
                    groups=conv.groups,
                )
            else:
                cp = conv_bwd(
                    s,
                    weight=w,
                    bias=None,
                    padding=conv.padding,
                    stride=conv.stride,
                    groups=conv.groups,
                )

            # Step 4: multiply by input
            relevance_out = cp * x

            return relevance_out


def _winner_takes_all(
    relevance_in: torch.Tensor,
    in_shape: Tuple[int, ...],
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Sends each pooled output's relevance entirely to its argmax input
    position, accumulating (via `scatter_add_`, not overwriting) wherever
    overlapping windows share an argmax.

    Arguments
    ---------

    relevance_in : torch.Tensor
        Incoming relevance, shape (B, C, H, W) - pooled output's spatial size.

    in_shape : Tuple[int, ...]
        Module input shape, (1, C, Hin, Win) as cached by
        `_max_pool_nd_fwd_hook`.

    indices : torch.Tensor
        argmax indices from `F.max_pool2d(..., return_indices=True)`,
        shape (B, C, H, W), flat position in [0, Hin * Win) per PyTorch's
        MaxUnpool2d convention.

    Returns
    -------

    torch.Tensor
        Relevance redistributed to the lower layer, shape (B, C, Hin, Win).
    """

    B, C = relevance_in.shape[:2]
    _, _, Hin, Win = in_shape

    relevance_out = torch.zeros(
        B, C, Hin * Win, dtype=relevance_in.dtype, device=relevance_in.device
    )
    relevance_out.scatter_add_(
        2, indices.reshape(B, C, -1), relevance_in.reshape(B, C, -1)
    )

    return relevance_out.view(B, C, Hin, Win)


class MaxPoolRule(PropRule):
    """
    LRP rule for max-pooling layers (MaxPool1d/2d/3d). By default passes
    relevance straight through each window unchanged - pooling applies no
    learned weights, so there's nothing to redistribute. Pass `max=True`
    for exact winner-takes-all redistribution to each window's argmax
    instead. Requires `module.indices`/`in_shape`/`out_shape`, as cached
    by `_max_pool_nd_fwd_hook`.
    """

    def __init__(self, max: bool = False, **kwargs: Any) -> None:
        """
        Stores whether to use winner-takes-all redistribution.

        Arguments
        ---------

        max : bool
            See class docstring.

        **kwargs
            Forwarded to PropRule.__init__ (power/eps/positive/
            contrastive - unused by this rule, accepted for a uniform
            constructor shape across every PropRule subclass).

        Returns
        -------

            None
        """

        super().__init__(**kwargs)
        self.max = max

    def compute(
        self, module: torch.nn.Module, relevance_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Redistributes relevance through a max-pooling window. See
        `PropRule.compute`.

        Arguments
        ---------

        module : torch.nn.Module
            MaxPool layer through which relevance is propagated.

        relevance_in : torch.Tensor
            Incoming relevance, already unwrapped to a plain tensor.

        Returns
        -------

        torch.Tensor
            Redistributed relevance.
        """

        pool = cast(MaxPoolLike, module)

        relevance_in = torch.cat([r.view(pool.out_shape) for r in relevance_in], dim=0)

        if not self.max:
            return relevance_in

        indices = torch.cat([pool.indices] * relevance_in.size(0), dim=0)
        return _winner_takes_all(relevance_in, pool.in_shape, indices)


class UpsampleRule(PropRule):
    """
    LRP rule for Upsample layers. Only 'nearest' mode is invertible.
    Requires `module.in_dim`/`out_shape`, as cached by `_upsample_fwd_hook`.
    """

    def compute(
        self, module: torch.nn.Module, relevance_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Redistributes relevance through an Upsample layer via average
        pooling - the adjoint of nearest-neighbor upsampling. See
        `PropRule.compute`.

        Arguments
        ---------

        module : torch.nn.Module
            Upsample layer through which relevance is propagated.

        relevance_in : torch.Tensor
            Incoming relevance, already unwrapped to a plain tensor.

        Returns
        -------

        torch.Tensor
            Redistributed relevance.

        Raises
        ------

        NotImplementedError
            If `module.mode` isn't 'nearest'.
        """

        upsample = cast(UpsampleLike, module)

        invert_upsample = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[
            upsample.in_dim - 2
        ]

        if upsample.mode != "nearest":
            raise NotImplementedError("Upsample layer must be in 'nearest' mode ")

        relevance_in = torch.cat(
            [r.view(upsample.out_shape) for r in relevance_in], dim=0
        )

        ks: Union[int, Tuple[int, ...]]
        if isinstance(upsample.scale_factor, float):
            ks = int(upsample.scale_factor)
        elif isinstance(upsample.scale_factor, tuple):
            ks = tuple(int(s) for s in upsample.scale_factor)

        inverted = invert_upsample(relevance_in, kernel_size=ks, stride=ks)
        # Only valid when scale_factor is a float (ks: int); a tuple
        # scale_factor isn't exercised by any YOLO26 block.
        inverted *= ks**2  # type: ignore[operator]  # normalizing constant

        return inverted


# Maps primitive torch layer types to the PropRule subclass that handles
# them, by exact type() (e.g. DWConv needs its own entry despite
# subclassing Conv). A static type->class map, not type->instance: which
# *configured* instance to use (power/eps/positive/contrastive) is
# Inverter's job, not this table's. Go through `select_rule`, not this
# directly.
RULE_REGISTRY: Dict[Type[torch.nn.Module], Type[PropRule]] = {
    Conv1d: ConvRule,
    Conv2d: ConvRule,
    Conv3d: ConvRule,
    Linear: LinearRule,
    MaxPool1d: MaxPoolRule,
    MaxPool2d: MaxPoolRule,
    MaxPool3d: MaxPoolRule,
    Upsample: UpsampleRule,
}


def select_rule(layer: torch.nn.Module) -> Optional[Type[PropRule]]:
    """
    Looks up which PropRule subclass handles a given layer, hiding
    RULE_REGISTRY's shape from callers.

    Arguments
    ---------

    layer : torch.nn.Module
        Layer instance to find a rule for.

    Returns
    -------

    Type[PropRule], optional
        The PropRule subclass registered for `type(layer)`, or None if
        this layer type has no registered rule.
    """

    return RULE_REGISTRY.get(type(layer))
