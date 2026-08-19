from typing import Any, Callable, Dict, Type, cast

import torch
import torch.nn.functional as F

# Standardized signature for every forward hook in this package, whether
# it's a per-primitive default (DEFAULT_FWD_HOOKS, here) or a YOLO-block-
# specific override (yolo.fwd_hooks.FWD_HOOK_REGISTRY) - the forward-pass
# half of the rules in rules.py / prop_* functions in yolo/block_rules.py.
# Kept in its own module rather than alongside either of those: a forward
# hook's job is purely to cache shape/index bookkeeping during the
# forward pass - a distinct phase of the algorithm from the backward
# relevance math itself, even though each hook is written for, and
# consumed by, one specific rule/prop_* function. Nothing in rules.py
# calls these functions directly (only reads the attributes they cache),
# so there's no runtime dependency between this module and rules.py in
# any direction.
FwdHookFunc = Callable[[torch.nn.Module, Any, Any], None]


def _conv_nd_fwd_hook(m: torch.nn.Module, in_tensor: Any, out_tensor: Any) -> None:
    """
    Default n-dimensional convolution forward hook. Also used directly
    (not via a torch hook registration) by yolo.block_rules's
    prop_SPPF/prop_Bottleneck/prop_Attention/prop_PSABlock to fabricate a
    forward pass through a dummy summation conv.

    Arguments
    ---------

    m : torch.nn.Module
        The Conv1d/Conv2d/Conv3d module the hook fired on.

    in_tensor : tuple
        The module's input, as passed to the forward hook.

    out_tensor : torch.Tensor
        The module's output.

    Returns
    -------

        None
    """

    setattr(m, "in_tensor", in_tensor[0])
    setattr(m, "out_tensor", out_tensor)


def _max_pool_nd_fwd_hook(m: torch.nn.Module, in_tensor: Any, out_tensor: Any) -> None:
    """
    Default n-dimensional max pool forward hook.

    Arguments
    ---------

    m : torch.nn.Module
        The MaxPool1d/MaxPool2d/MaxPool3d module the hook fired on.

    in_tensor : tuple
        The module's input, as passed to the forward hook.

    out_tensor : torch.Tensor
        The module's output.

    Returns
    -------

        None
    """

    # Registered for MaxPool1d/2d/3d alike (see DEFAULT_FWD_HOOKS below),
    # but the pooling math here is hardcoded to 2d regardless - pre-
    # existing, not this pass's concern. torch.nn.MaxPool2d has real
    # stubs for all of these; the shared FwdHookFunc signature (m:
    # torch.nn.Module) just can't see them without narrowing first.
    pool = cast(torch.nn.MaxPool2d, m)
    _, indices = F.max_pool2d(
        in_tensor[0],
        kernel_size=pool.kernel_size,
        stride=pool.stride,
        padding=pool.padding,
        dilation=pool.dilation,
        return_indices=True,
        ceil_mode=pool.ceil_mode,
    )
    setattr(m, "indices", indices)
    setattr(m, "out_shape", out_tensor.size())
    setattr(m, "in_shape", in_tensor[0].size())


def _upsample_fwd_hook(m: torch.nn.Module, in_tensor: Any, out_tensor: Any) -> None:
    """
    Default up-sampling forward hook.

    Arguments
    ---------

    m : torch.nn.Module
        The Upsample module the hook fired on.

    in_tensor : tuple
        The module's input, as passed to the forward hook.

    out_tensor : torch.Tensor
        The module's output.

    Returns
    -------

        None
    """

    setattr(m, "in_dim", len(in_tensor[0].shape))
    setattr(m, "out_shape", out_tensor.shape)


def _linear_fwd_hook(m: torch.nn.Module, in_tensor: Any, out_tensor: Any) -> None:
    """
    Default Linear layer forward hook.

    Arguments
    ---------

    m : torch.nn.Module
        The Linear module the hook fired on.

    in_tensor : tuple
        The module's input, as passed to the forward hook.

    out_tensor : torch.Tensor
        The module's output.

    Returns
    -------

        None
    """

    setattr(m, "in_tensor", in_tensor[0])
    setattr(m, "out_shape", list(out_tensor.size()))


def _silent_pass(m: torch.nn.Module, in_tensor: Any, out_tensor: Any) -> None:
    """
    Silent forward hook that saves nothing.

    Arguments
    ---------

    m : torch.nn.Module
        The module the hook fired on (unused).

    in_tensor : tuple
        The module's input (unused).

    out_tensor : torch.Tensor
        The module's output (unused).

    Returns
    -------

        None
    """

    pass


# Architecture-independent forward hooks for standard torch layer
# primitives - always available regardless of YOLO version, unlike
# yolo.fwd_hooks.FWD_HOOK_REGISTRY (which is YOLO26-specific and
# injectable per version). Not meant to be read directly; go through
# yolo.fwd_hooks.select_fwd_hook.
DEFAULT_FWD_HOOKS: Dict[Type[torch.nn.Module], FwdHookFunc] = {
    torch.nn.MaxPool1d: _max_pool_nd_fwd_hook,
    torch.nn.MaxPool2d: _max_pool_nd_fwd_hook,
    torch.nn.MaxPool3d: _max_pool_nd_fwd_hook,
    torch.nn.Conv1d: _conv_nd_fwd_hook,
    torch.nn.Conv2d: _conv_nd_fwd_hook,
    torch.nn.Conv3d: _conv_nd_fwd_hook,
    torch.nn.Linear: _linear_fwd_hook,
    torch.nn.Upsample: _upsample_fwd_hook,
    torch.nn.BatchNorm1d: _silent_pass,
    torch.nn.BatchNorm2d: _conv_nd_fwd_hook,
    torch.nn.BatchNorm3d: _silent_pass,
    torch.nn.ReLU: _silent_pass,
    torch.nn.modules.activation.ReLU: _silent_pass,
    torch.nn.ELU: _silent_pass,
    torch.nn.Flatten: _silent_pass,
    torch.nn.Dropout: _silent_pass,
    torch.nn.Dropout2d: _silent_pass,
    torch.nn.Dropout3d: _silent_pass,
    torch.nn.Softmax: _silent_pass,
    torch.nn.LogSoftmax: _silent_pass,
    torch.nn.Sigmoid: _silent_pass,
    torch.nn.SiLU: _silent_pass,
    torch.nn.Identity: _silent_pass,
}
