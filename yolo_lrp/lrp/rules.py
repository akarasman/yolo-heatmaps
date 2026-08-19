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
        """Wrapper for propagate, makes rule object callable"""

        return self.propagate(module, relevance)

    def propagate(self, module: torch.nn.Module, relevance: RelevanceT) -> RelevanceT:
        """

        Propagate incoming relevance through module and redistribute it to lower layers.

        Arguments
        ---------

        module : torch.nn.Module
            Module through which relevance is propagated.

        relevance : LayerRelevance or torch.Tensor
            LayerRelevance tensor or simple tensor containing upper layer relevance.

        Returns
        -------

        LayerRelevance or torch.Tensor
            Re-distributed relevance. If input is tensor output will also
            be tensor, and if it is LayerRelevance output will also be
            LayerRelevance tensor.

        ATTENTION : Each time LayerRelevance is "propagated" through a
        module it essentialy "morphs" into the output relevance. This is
        to say that any previous relevance info is not saved and is
        discarded. If for the sake of visualizing the relevance
        distribution as it backpropagates through the network you need to
        keep a cache of each layer relevance you can do so by using the
        cache() method.

        """

        # Shared for every rule: unwrap LayerRelevance to a plain tensor for
        # compute(), then re-wrap; pass plain tensors through untouched.
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
    LinearRule(power : int = 1, eps : float = 1e-06, positive : bool = True,
               contrastive : bool =True,)

    Apply layerwise relevance propagation rule to linear layer. Implemented using
    the procedure described here :

    http://iphome.hhi.de/samek/pdf/MonXAI19.pdf
    """

    def _get_fwd_step(self) -> Callable[..., torch.Tensor]:
        """Dimension non-specific forward function"""

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
        """Dimension non-specific inverse function"""

        def linear_wrapper(
            relevance_in: torch.Tensor, w: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor:
            return F.linear(relevance_in, w.t(), **kwargs)

        return linear_wrapper

    def compute(
        self, module: torch.nn.Module, relevance_in: torch.Tensor
    ) -> torch.Tensor:
        """Implementation of 4-step relevance propagation procedure. See
        PropRule.compute."""

        linear = cast(LinearLike, module)

        linear_fwd = self._get_fwd_step()
        linear_bwd = self._get_bwd_step()

        # Pre-process : Apply power to weights/input
        x = linear.in_tensor
        x = x.pow(self.power)
        w = linear.weight.pow(self.power)

        # Compute forward activation with modified weights (step 1)
        z = linear_fwd(x, w, bias=None)
        z = z + torch.sign(z) * self.eps
        relevance_in[z == 0] = 0
        z[z == 0] = 1

        # Divide incoming relevance by activation (step 2)
        s = relevance_in / z

        # Compute gradient of s in terms of its input, which the equivalent of applying
        # the de-convolution operation (see PyTorch doc) (step 3)
        c = linear_bwd(s, w, bias=None)

        # Multiply by input (step 4)
        relevance_out = c * x

        return relevance_out


class ConvRule(PropRule):
    """

    ConvRule(power : int = 1, positive : bool = True, eps : float = 1e-6,
             contrastive : bool = True,)

    Apply layerwise relevance propagation rule to convolutional layers.
    Implemented efficiently using procedure described here :

    http://iphome.hhi.de/samek/pdf/MonXAI19.pdf
    """

    def _get_fwd_step(
        self, m: Union[Conv1d, Conv2d, Conv3d]
    ) -> Callable[..., torch.Tensor]:
        """Dimension non-specific forward function"""

        try:
            conv = {Conv1d: F.conv1d, Conv2d: F.conv2d, Conv3d: F.conv3d}[type(m)]
        except KeyError:
            raise Exception("Layer must be one of {}".format((Conv1d, Conv2d, Conv3d)))

        # The only function of the following wrapper is to duplicate the input
        # tensor. This is necessary for computing contrastive relevance propagation
        # efficiently.
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
        """Dimension non-specific inverse function"""

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
        """Implementation of 4-step relevance propagation procedure. See
        PropRule.compute."""

        conv = cast(ConvLike, module)
        # _get_fwd_step/_get_bwd_step only accept real Conv1d/2d/3d
        # instances (their own dict dispatches on exact type) - this
        # rule is only ever registered against those three real types
        # (see RULE_REGISTRY), never a plain torch.nn.Module.
        conv_nd = cast(Union[Conv1d, Conv2d, Conv3d], module)

        relevance_in = torch.cat(
            [r.view_as(conv.out_tensor) for r in relevance_in], dim=0
        )
        conv_fwd = self._get_fwd_step(conv_nd)
        conv_bwd = self._get_bwd_step(conv_nd)

        with torch.no_grad():

            # Pre-process : Apply power to weights/input, discard negative activations.
            # x stays at its real (single-forward-pass) batch size here -
            # conv_fwd/conv_bwd are the sole place that duplicate it for
            # the contrastive (primal+dual) case, same as LinearRule.
            # Pre-duplicating x here too (as this line used to) double-
            # doubles the batch under contrastive=True: conv_fwd would
            # duplicate an already-duplicated x, producing a batch twice
            # the size of relevance_in and crashing on real batch != 1
            # inputs. The final `cp * x` still broadcasts correctly
            # against cp's (possibly doubled) batch.
            x = conv.in_tensor.clone()
            w = conv.weight.clone()
            if self.positive:
                x = x.clamp(min=0)
                w = w.clamp(min=0)
            x = x.pow(self.power)
            w = w.pow(self.power)

            # Compute forward activation with modified weights (step 1)

            z = conv_fwd(
                x,
                weight=w,
                bias=None,
                stride=conv.stride,
                padding=conv.padding,
                groups=conv.groups,
            )  #
            z = z + torch.sign(z) * self.eps
            relevance_in[z == 0] = 0
            z[z == 0] = 1

            # Divide incoming relevance by activation (step 2)
            s = relevance_in / z

            # Compute gradient of s in terms of its input, which the
            # equivalent of applying the de-convolution operation (see
            # PyTorch doc) (step 3)
            if conv.stride != (1, 1):

                # When stride is not equal to 1 there may be some
                # discrepancy between the input and output sizes, the
                # following code fixes this by manually calculating the
                # new H, w dimensions and adding valid padding at
                # backward step
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

            # Multiply by input (step 4)
            relevance_out = cp * x

            return relevance_out


def _winner_takes_all(
    relevance_in: torch.Tensor,
    in_shape: Tuple[int, ...],
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Implements winner-takes-all redistribution of relevance for a max
    pooling layer: each pooled output's relevance goes entirely to the
    input position that was its argmax, accumulated (not overwritten) at
    positions two or more outputs map to - which happens whenever
    overlapping windows (stride < kernel_size) share an argmax.

    Vectorized via `scatter_add_` (one accumulate per (batch, channel)
    row, instead of a Python-level loop over every element).

    Arguments
    ---------

    relevance_in : torch.Tensor
        Incoming relevance from upper layers, shape (B, C, H, W) - the
        pooled output's own spatial size.

    in_shape : Tuple[int, ...]
        Shape of module input, (1, C, Hin, Win) as cached by
        _max_pool_nd_fwd_hook (leading dim always 1 - explain() only
        ever runs a single-image forward pass; B above may still be
        larger, e.g. doubled under contrastive relevance).

    indices : torch.Tensor
        argmax indices from F.max_pool2d(..., return_indices=True),
        shape (B, C, H, W), each value in [0, Hin * Win) - the flat
        position within its own (Hin, Win) input plane, per PyTorch's
        own MaxUnpool2d indexing convention.

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
    MaxPoolRule(max : bool = False, ...)

    LRP rule for max-pooling layers (MaxPool1d/2d/3d). By default, just
    passes relevance straight through each pooling window unchanged -
    pooling doesn't mix channels or apply learned weights the way
    conv/linear do, so there's no redistribution to compute. Pass
    `max=True` for exact winner-takes-all redistribution to the argmax
    position within each window instead.

    Requires `module.indices`/`module.in_shape`/`module.out_shape`, as
    cached by `_max_pool_nd_fwd_hook`.
    """

    def __init__(self, max: bool = False, **kwargs: Any) -> None:
        """
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
        """Redistributes relevance through a max-pooling window. See
        PropRule.compute."""

        pool = cast(MaxPoolLike, module)

        relevance_in = torch.cat([r.view(pool.out_shape) for r in relevance_in], dim=0)

        if not self.max:
            return relevance_in

        indices = torch.cat([pool.indices] * relevance_in.size(0), dim=0)
        return _winner_takes_all(relevance_in, pool.in_shape, indices)


class UpsampleRule(PropRule):
    """
    UpsampleRule(...)

    LRP rule for Upsample layers. ATTENTION: only 'nearest' upsampling is
    currently invertible.

    Requires `module.in_dim`/`module.out_shape`, as cached by
    `_upsample_fwd_hook`.
    """

    def compute(
        self, module: torch.nn.Module, relevance_in: torch.Tensor
    ) -> torch.Tensor:
        """Redistributes relevance through an Upsample layer via average pooling
        (the adjoint of nearest-neighbor upsampling). See PropRule.compute."""

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
        # Pre-existing: only valid when scale_factor was a float (ks: int)
        # - a non-uniform tuple scale_factor would already have failed at
        # runtime here, same as it does statically. Not exercised by any
        # YOLO26 block (every nn.Upsample in it uses a single scale_factor).
        inverted *= ks**2  # type: ignore[operator]  # Normalizing constant

        return inverted


# Maps primitive torch layer types to the PropRule subclass that handles
# them. Dispatch is by exact `type()`, matching every other dispatch table
# in this package (e.g. DWConv needs its own inv_func entry despite
# subclassing Conv). Deliberately a static type->class mapping rather than
# type->instance: which *kind* of rule applies to a layer never changes,
# but which *configured* instance to use does (power/eps/positive/
# contrastive are user-supplied per YOLOLRP, and contrastive gets
# mutated per explain() call) - Inverter owns that part separately. Not
# meant to be read directly; go through `select_rule`.
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
