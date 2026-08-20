from typing import Any, Callable, Dict, Type, cast

import torch
import torch.nn.functional as F

# Signature shared by every forward hook - per-primitive defaults here,
# YOLO-block overrides in yolo.fwd_hooks.FWD_HOOK_REGISTRY. Each hook just
# caches shape/index bookkeeping during the forward pass for its matching
# rule/prop_* function in rules.py / yolo/block_rules.py; no runtime
# dependency between this module and rules.py in either direction.
FwdHookFunc = Callable[[torch.nn.Module, Any, Any], None]


def _conv_nd_fwd_hook(m: torch.nn.Module, in_tensor: Any, out_tensor: Any) -> None:
    """
    Default n-D convolution forward hook. Also called directly (not via
    torch hook registration) by yolo.block_rules's prop_SPPF/
    prop_Bottleneck/prop_Attention/prop_PSABlock to fabricate a forward
    pass through a dummy summation conv.

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
    Default n-D max pool forward hook.

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

    # Hardcoded to 2d regardless of which MaxPoolNd fired (pre-existing);
    # cast just narrows so mypy can see MaxPool2d's attributes.
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


# Architecture-independent hooks for standard torch primitives, always
# available regardless of YOLO version (contrast yolo.fwd_hooks.
# FWD_HOOK_REGISTRY, which is YOLO26-specific/injectable). Access via
# yolo.fwd_hooks.select_fwd_hook, not directly.
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
