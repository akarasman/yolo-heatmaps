"""
Structural (PEP 544) types describing the shapes block_rules.py's prop_*
functions actually read off their `mod` parameter.

Every one of those functions is typed `mod: torch.nn.Module` in its public
signature - and has to stay that way, since `PROP_REGISTRY`/`inv_funcs`
holds them all as one `Dict[Type[torch.nn.Module], PropFunc]`, and
`Callable` parameter types are contravariant: a function narrower than
`Callable[[Inverter, torch.nn.Module, ...], ...]` can't be stored there.
There are also no type stubs for ultralytics' real block classes (C3,
Bottleneck, Attention, ...), so `mod.cv1`, `mod.qkv.conv.in_tensor`,
`mod.m[::-1]` all resolve through `torch.nn.Module.__getattr__`'s stubbed
`Tensor | Module` return and cascade into a wall of noise - not because
the code is wrong, but because mypy has no way to know these are
dynamically-dispatched, specifically-shaped YOLO blocks.

The fix used throughout block_rules.py: `mod = cast(XLike, mod)` once, at
the top of each function body (never in the public signature), narrowing
the *local* type for the rest of that function without touching the
Callable contract PROP_REGISTRY depends on. These Protocols are what `X`
is cast to - not full stubs for the real classes, just the attributes
each function's body actually reads, named to match (`HasCV1CV2` for a
block that reads `.cv1`/`.cv2`, etc.), built from a few shared pieces.

Attributes that get passed onward to `Inverter.__call__`/`register_*`
(e.g. `HasCV1.cv1`) are typed as plain `torch.nn.Module`, or - where the
real ultralytics attribute is genuinely a container (`self.m` on C2f/C3/
C2PSA) - the matching real torch container type (`nn.ModuleList`/
`nn.Sequential`), since both already have complete stubs and are
themselves real `nn.Module`s. Attributes that only ever get *read*
further within block_rules.py (`.conv.in_tensor`, `.bn.out_tensor`)
aren't chased through nested Protocols here; the specific line doing that
read casts the intermediate expression to `HasInTensor`/`HasOutTensor`
directly, since a Protocol needs no relationship to `torch.nn.Module` to
be a valid `cast()` target - only to be passed somewhere `torch.nn.Module`
is expected, which those intermediate `.conv`/`.bn` values never are.
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
    """The `Conv`/`DWConv` wrapper itself (conv -> bn -> act) - what
    prop_Conv's `mod` is, delegating straight to the inner Conv2d."""

    conv: torch.nn.Module


class HasCV1(Protocol):
    cv1: torch.nn.Module


class HasCV1CV2(HasCV1, Protocol):
    cv2: torch.nn.Module


class HasCV1CV2CV3(HasCV1CV2, Protocol):
    cv3: torch.nn.Module


class HasAdd(Protocol):
    """A block with YOLO26's optional residual-shortcut flag (SPPF,
    PSABlock). Every real read of it goes through `getattr(mod, "add",
    False)`, which mypy allows on any type regardless of whether it's
    declared here - this Protocol exists only to document the attribute
    for readers, not because anything requires it."""

    add: bool


class SPPFLike(HasCV1CV2, HasAdd, Protocol):
    """`n`, like `add` (see HasAdd), is read via `getattr(mod, "n", 3)` in
    prop_SPPF, not direct attribute access - some real checkpoints (e.g.
    YOLOv10's) predate both attributes and restore an SPPF instance whose
    __dict__ genuinely lacks them. Declared here only to document it."""

    n: int


class C3Like(HasCV1CV2CV3, Protocol):
    """C3/C3k - `self.m` is a real `nn.ModuleList`, iterated forward
    (never reversed or indexed) in prop_C3."""

    m: torch.nn.ModuleList


class C2fLike(HasCV1CV2, Protocol):
    """C2f/C3k2 - `self.m` is reversed and indexed (`mod.m[::-1]`) in
    prop_C2f, which `nn.ModuleList` supports directly."""

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
    prop_PSABlock, both of which `nn.Sequential` supports directly."""

    ffn: torch.nn.Sequential


class DFLLike(Protocol):
    in_shape: Sequence[int]
    conv: torch.nn.Module


# --- YOLOv10-specific blocks (block.SCDown/PSA/CIB/RepVGGDW,
# head.v10Detect) ---
#
# v10Detect and C2fCIB need no new Protocol at all: v10Detect subclasses
# Detect without changing cv3's shape (DetectLike already covers it), and
# C2fCIB subclasses C2f without overriding forward() (C2fLike already
# covers it) - both are registered directly against prop_Detect/prop_C2f,
# same as C3k2 already is.


class PSALike(HasCV1CV2, HasAttn, Protocol):
    """PSA - the same split/attn/ffn/merge shape as C2PSA's cv1/cv2 plus a
    nested PSABlock, but attn/ffn are direct, always-residual submodules
    here rather than wrapped in a PSABlock of their own."""

    c: int
    ffn: torch.nn.Sequential


class CIBLike(HasAdd, Protocol):
    """Compact Inverted Block - `self.cv1` is a real `nn.Sequential` (of
    Conv/RepVGGDW) wrapping an optional residual, the same shape as
    Bottleneck's cv1->cv2 pair collapsed into one Sequential."""

    cv1: torch.nn.Sequential


class RepVGGDWLike(Protocol):
    """RepVGGDW: `act(conv(x) + conv1(x))` - two parallel depthwise conv
    branches off the same input, summed (not a residual around a single
    branch)."""

    conv: torch.nn.Module
    conv1: torch.nn.Module


# --- YOLOv9-specific blocks (block.AConv/RepNCSPELAN4/ELAN1/SPPELAN/
# RepCSP/RepBottleneck/RepConv) ---
#
# RepCSP and RepBottleneck need no new Protocol at all: RepCSp subclasses
# C3 without overriding forward() (C3Like already covers it - only its
# own `self.m`'s element type, RepBottleneck, differs from plain
# Bottleneck/C3k), and RepBottleneck subclasses Bottleneck without
# overriding forward() either (HasCV1CV2 already covers it - only
# `self.cv1`'s type, RepConv, differs from plain Conv). Both are
# registered directly against prop_C3/prop_Bottleneck, same convention as
# C3k2 -> prop_C2f above.


class RepNCSPELAN4Like(Protocol):
    """RepNCSPELAN4 (and, by reuse, ELAN1 - subclasses it without
    overriding forward(), only what cv1-cv4 are made of): `cv2`/`cv3` are
    a fixed two-stage chain (`y3 = cv3(cv2(y1))`), not an iterated
    ModuleList like C2f's `self.m` - prop_RepNCSPELAN4 unrolls it
    directly instead of looping."""

    cv1: torch.nn.Module
    cv2: torch.nn.Module
    cv3: torch.nn.Module
    cv4: torch.nn.Module
    c: int


class SPPELANLike(Protocol):
    """SPPELAN - the same n+1-way additive-merge shape as SPPF, just with
    3 separately-constructed (rather than 1 shared, n-times-called)
    MaxPool2d branches; cv2/cv3/cv4 (the pooling branches) are never read
    directly, only cv1 (first branch) and cv5 (final merge)."""

    cv1: torch.nn.Module
    cv5: torch.nn.Module


class RepConvLike(Protocol):
    """RepConv: `act(conv1(x) + conv2(x) + id_out)` - `id_out` is a
    BatchNorm identity branch that's only ever non-zero when RepConv is
    constructed with `bn=True`, which nothing in this codebase's
    supported models does (confirmed against a real yolov9t.pt: `self.bn`
    is unset/None on every RepConv reached this way) - so, like
    RepVGGDW's own two-branch sum, only conv1/conv2 are handled."""

    conv1: torch.nn.Module
    conv2: torch.nn.Module


# AConv needs no new Protocol either - `cv1(avg_pool2d(x, 2, 1, 0, False,
# True))`, and HasCV1 already covers `cv1`. The avg_pool2d itself isn't
# transparent to relevance the way BatchNorm/SiLU are elsewhere in this
# module - it's a fixed, non-learned, stride-1 kernel=2 anti-aliasing
# blur (cv1's own stride=2 does the actual downsampling), but it *does*
# shrink H/W by 1, so prop_AConv inverts it via its own adjoint
# (conv_transpose2d with uniform 1/(k*k) weights) rather than skipping
# it - see prop_AConv's own docstring.


# --- rules.py: PropRule.compute() implementations ---
#
# Unlike block_rules.py's blocks, these `module` params really are
# primitive torch layers (Conv1d/2d/3d, Linear, MaxPool1d/2d/3d, Upsample)
# with complete, real torch stubs - but PropRule.compute()'s own signature
# is `module: torch.nn.Module` (shared across every rule, like PropFunc's
# `mod: torch.nn.Module` above), so those real stubs are never visible
# either. Rather than casting twice per read site (once to the real torch
# class for its real attributes, once to a Protocol for the fwd_hook-
# cached ones - Protocol can't inherit from a concrete class like Conv2d,
# same restriction as above), each of these redeclares the handful of
# real attributes the corresponding compute() body actually reads
# alongside the cached ones, as a single Protocol.


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
    """Cached by _max_pool_nd_fwd_hook - none of these are real
    MaxPool1d/2d/3d attributes."""

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
