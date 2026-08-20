"""
Structural (PEP 544) types describing the shapes block_rules.py's prop_*
functions actually read off their `mod` parameter.

Those functions all stay typed `mod: torch.nn.Module` in their public
signature (required since `PROP_REGISTRY` stores them under one
`Dict[Type[torch.nn.Module], PropFunc]`, and `Callable` params are
contravariant). Since ultralytics ships no stubs for its real block
classes, attribute access on `mod` would otherwise resolve through
`Module.__getattr__`'s stubbed `Tensor | Module` return and produce a
wall of mypy noise. Fix: each function body does `mod = cast(XLike, mod)`
once at the top, narrowing the local type without touching the public
signature. These Protocols are what `X` is cast to - just the attributes
each function body actually reads, named to match (`HasCV1CV2` for
`.cv1`/`.cv2`, etc.).

Attributes passed onward to `Inverter.__call__`/`register_*` are typed
plain `torch.nn.Module`, or the real container type (`nn.ModuleList`/
`nn.Sequential`) where the ultralytics attribute genuinely is one.
Attributes only read locally (`.conv.in_tensor`, `.bn.out_tensor`) aren't
chased through nested Protocols - the read site casts the intermediate
expression to `HasInTensor`/`HasOutTensor` directly instead.
"""

from typing import List, Protocol, Sequence, Tuple, Union

import torch


class HasInTensor(Protocol):
    """A layer forward-hooked with a cached input tensor (see
    lrp.fwd_hooks._conv_nd_fwd_hook and friends)."""

    in_tensor: torch.Tensor


class HasOutTensor(Protocol):
    """A layer forward-hooked with a cached output tensor."""

    out_tensor: torch.Tensor


class HasConv(Protocol):
    """The `Conv`/`DWConv` wrapper (conv -> bn -> act); prop_Conv's `mod`,
    delegating straight to the inner Conv2d."""

    conv: torch.nn.Module


class HasCV1(Protocol):
    cv1: torch.nn.Module


class HasCV1CV2(HasCV1, Protocol):
    cv2: torch.nn.Module


class HasCV1CV2CV3(HasCV1CV2, Protocol):
    cv3: torch.nn.Module


class HasAdd(Protocol):
    """Optional residual-shortcut flag (SPPF, PSABlock), always read via
    `getattr(mod, "add", False)` - declared here only to document it."""

    add: bool


class SPPFLike(HasCV1CV2, HasAdd, Protocol):
    """`n`, like `add`, is read via `getattr(mod, "n", 3)` in prop_SPPF,
    not direct access - some checkpoints (e.g. YOLOv10's) predate both
    attributes."""

    n: int


class C3Like(HasCV1CV2CV3, Protocol):
    """C3/C3k - `self.m` is a real `nn.ModuleList`, iterated forward
    (never reversed or indexed) in prop_C3."""

    m: torch.nn.ModuleList


class C2fLike(HasCV1CV2, Protocol):
    """C2f/C3k2 - `self.m` is reversed and indexed (`mod.m[::-1]`) in
    prop_C2f."""

    c: int
    m: torch.nn.ModuleList


class C2PSALike(HasCV1CV2, Protocol):
    """C2PSA - `self.m` is a real `nn.Sequential` (of PSABlock), passed
    to Inverter as one unit in prop_C2PSA rather than iterated."""

    c: int
    m: torch.nn.Sequential


class DetectLike(Protocol):
    cv3: torch.nn.ModuleList


class ConcatLike(Protocol):
    in_shapes: List[int]
    d: int
    f: List[int]


class HasQKV(Protocol):
    qkv: torch.nn.Module


class AttentionLike(HasQKV, Protocol):
    proj: torch.nn.Module
    pe: torch.nn.Module
    num_heads: int
    key_dim: int
    head_dim: int
    scale: float


class HasAttn(Protocol):
    attn: torch.nn.Module


class PSABlockLike(HasAttn, HasAdd, Protocol):
    """`self.ffn` is a real `nn.Sequential` of Conv - indexed
    (`mod.ffn[0]`/`mod.ffn[-1]`) and passed to Inverter as one unit in
    prop_PSABlock."""

    ffn: torch.nn.Sequential


class DFLLike(Protocol):
    in_shape: Sequence[int]
    conv: torch.nn.Module


# --- YOLOv10-specific blocks (block.SCDown/PSA/CIB/RepVGGDW,
# head.v10Detect) ---
#
# v10Detect and C2fCIB need no new Protocol: both subclass Detect/C2f
# without changing the shape DetectLike/C2fLike already cover, and are
# registered directly against prop_Detect/prop_C2f.


class PSALike(HasCV1CV2, HasAttn, Protocol):
    """PSA - same split/attn/ffn/merge shape as C2PSA's cv1/cv2 plus a
    nested PSABlock, but attn/ffn are direct, always-residual submodules
    rather than wrapped in a PSABlock of their own."""

    c: int
    ffn: torch.nn.Sequential


class CIBLike(HasAdd, Protocol):
    """Compact Inverted Block - `self.cv1` is a real `nn.Sequential` (of
    Conv/RepVGGDW) wrapping an optional residual, the same shape as
    Bottleneck's cv1->cv2 pair collapsed into one Sequential."""

    cv1: torch.nn.Sequential


class RepVGGDWLike(Protocol):
    """RepVGGDW: `act(conv(x) + conv1(x))` - two parallel depthwise conv
    branches off the same input, summed."""

    conv: torch.nn.Module
    conv1: torch.nn.Module


# --- YOLOv9-specific blocks (block.AConv/RepNCSPELAN4/ELAN1/SPPELAN/
# RepCSP/RepBottleneck/RepConv) ---
#
# RepCSP and RepBottleneck need no new Protocol: both subclass C3/
# Bottleneck without overriding forward() (C3Like/HasCV1CV2 already cover
# them - only an inner submodule's type differs), registered directly
# against prop_C3/prop_Bottleneck.


class RepNCSPELAN4Like(Protocol):
    """RepNCSPELAN4 (and ELAN1, which subclasses it unchanged): `cv2`/
    `cv3` are a fixed two-stage chain (`y3 = cv3(cv2(y1))`), not an
    iterated ModuleList - prop_RepNCSPELAN4 unrolls it directly."""

    cv1: torch.nn.Module
    cv2: torch.nn.Module
    cv3: torch.nn.Module
    cv4: torch.nn.Module
    c: int


class SPPELANLike(Protocol):
    """SPPELAN - same n+1-way additive-merge shape as SPPF, with 3
    separately-constructed MaxPool2d branches instead of 1 reused one.
    Only cv1 (first branch) and cv5 (final merge) are read directly."""

    cv1: torch.nn.Module
    cv5: torch.nn.Module


class RepConvLike(Protocol):
    """RepConv: `act(conv1(x) + conv2(x) + id_out)` - `id_out` (a
    BatchNorm identity branch) is only non-zero when constructed with
    `bn=True`, which no supported model here does, so only conv1/conv2
    are handled."""

    conv1: torch.nn.Module
    conv2: torch.nn.Module


# AConv needs no new Protocol - `cv1(avg_pool2d(x, 2, 1, 0, False, True))`,
# and HasCV1 already covers `cv1`. The avg_pool2d is a fixed, non-learned
# anti-aliasing blur (cv1's own stride=2 does the real downsampling) but
# still shrinks H/W by 1, so prop_AConv inverts it via its own adjoint
# (conv_transpose2d, uniform 1/(k*k) weights) rather than skipping it -
# see prop_AConv's own docstring.


# --- rules.py: PropRule.compute() implementations ---
#
# These `module` params are primitive torch layers (Conv1d/2d/3d, Linear,
# MaxPool1d/2d/3d, Upsample) with real stubs, but compute()'s shared
# signature (`module: torch.nn.Module`) hides them the same way. Each
# Protocol below redeclares the real attributes compute() reads alongside
# the fwd_hook-cached ones, in one Protocol (a Protocol can't inherit
# from a concrete class like Conv2d, so two separate casts isn't an
# option).


class ConvLike(HasInTensor, HasOutTensor, Protocol):
    """nn.Conv1d/2d/3d, as ConvRule.compute reads it: in_tensor/out_tensor
    cached by _conv_nd_fwd_hook, plus real Conv attributes."""

    weight: torch.Tensor
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    kernel_size: Tuple[int, ...]
    output_padding: Tuple[int, ...]
    groups: int


class LinearLike(HasInTensor, Protocol):
    weight: torch.Tensor


class MaxPoolLike(Protocol):
    """Cached by _max_pool_nd_fwd_hook - none are real MaxPool1d/2d/3d
    attributes."""

    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    indices: torch.Tensor


class SPPFHookLike(HasCV1, Protocol):
    """As yolo.fwd_hooks.SPPF_fwd_hook reads it - `m` is the single real
    nn.MaxPool2d instance SPPF reuses across all `n` pooling stages."""

    m: torch.nn.MaxPool2d


class UpsampleLike(Protocol):
    """mode/scale_factor are real nn.Upsample attributes; in_dim/out_shape
    are cached by _upsample_fwd_hook."""

    in_dim: int
    out_shape: Tuple[int, ...]
    mode: str
    scale_factor: Union[float, Tuple[float, ...]]
