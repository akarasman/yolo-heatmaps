from typing import Any, Dict, Optional, Type, cast

import torch
import torch.nn.functional as F
from ultralytics.nn.modules import block, conv

from ..lrp.fwd_hooks import DEFAULT_FWD_HOOKS, FwdHookFunc
from ..lrp.protocols import SPPFHookLike


def SPPF_fwd_hook(m: torch.nn.Module, in_tensor: Any, out_tensor: Any) -> None:
    """
    Caches per-stage max-pool indices for SPPF's `n` chained pooling
    calls - `m.m` is the same MaxPool2d instance reused `n` times, so
    torch's own hook on it would otherwise only keep the last call's
    indices. Not currently consumed by prop_SPPF's additive-merge rule;
    kept for parity and in case a future rule wants exact inversion.

    Arguments
    ---------

    m : torch.nn.Module
        The SPPF module.

    in_tensor : Any
        SPPF's input (1-tuple wrapping the input tensor, per torch's hook convention).

    out_tensor : Any
        Unused (required by torch's hook signature).

    Returns
    -------

        None
    """

    sppf = cast(SPPFHookLike, m)

    x = sppf.cv1(in_tensor[0])
    y = x
    indices = []
    for _ in range(getattr(sppf, "n", 3)):
        y, idx = F.max_pool2d(
            y,
            kernel_size=sppf.m.kernel_size,
            stride=sppf.m.stride,
            padding=sppf.m.padding,
            dilation=sppf.m.dilation,
            return_indices=True,
            ceil_mode=sppf.m.ceil_mode,
        )
        indices.append(idx)
    setattr(sppf, "indices", indices)


def Concat_fwd_hook(m: torch.nn.Module, in_tensors: Any, out_tensor: Any) -> None:
    """
    Caches per-input channel widths (along Concat's concat axis) and the
    output shape, so prop_Concat can split relevance back into the same
    slices later.

    Arguments
    ---------

    m : torch.nn.Module
        The Concat module.

    in_tensors : Any
        Concat's input (1-tuple wrapping the list of tensors being concatenated).

    out_tensor : Any
        Concat's output.

    Returns
    -------

        None
    """

    shapes = [in_tensor.shape[m.d] for in_tensor in in_tensors[0]]

    setattr(m, "in_shapes", shapes)
    setattr(m, "out_shape", out_tensor.shape)


# Maps YOLO26 block types to the hook that caches the extra shape/index
# bookkeeping their prop_* function (block_rules.py) needs. Only entries
# whose default hook behavior isn't enough get one; everything else falls
# through to DEFAULT_FWD_HOOKS. Access via `select_fwd_hook`, not directly.
FWD_HOOK_REGISTRY: Dict[Type[torch.nn.Module], FwdHookFunc] = {
    conv.Concat: Concat_fwd_hook,
    block.SPPF: SPPF_fwd_hook,
}


def select_fwd_hook(mod: torch.nn.Module) -> Optional[FwdHookFunc]:
    """
    Looks up the forward hook for a module: a block-specific override
    (FWD_HOOK_REGISTRY) if one exists, else the primitive-level default
    (lrp.fwd_hooks.DEFAULT_FWD_HOOKS).

    Arguments
    ---------

    mod : torch.nn.Module
        Module instance to find a forward hook for.

    Returns
    -------

    FwdHookFunc, optional
        The forward hook function registered for `type(mod)`, or None if
        this module type has no registered hook in either table.
    """

    return FWD_HOOK_REGISTRY.get(type(mod)) or DEFAULT_FWD_HOOKS.get(type(mod))
