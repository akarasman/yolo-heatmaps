from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, Union, cast

import torch
from ultralytics.nn.modules import block, conv, head

from ..lrp.fwd_hooks import _conv_nd_fwd_hook
from ..lrp.protocols import (
    AttentionLike,
    C2fLike,
    C2PSALike,
    C3Like,
    CIBLike,
    ConcatLike,
    DetectLike,
    DFLLike,
    HasConv,
    HasCV1,
    HasCV1CV2,
    HasInTensor,
    HasOutTensor,
    PSABlockLike,
    PSALike,
    RepConvLike,
    RepNCSPELAN4Like,
    RepVGGDWLike,
    SPPELANLike,
    SPPFLike,
)
from ..lrp.relevance import LayerRelevance, RelevanceMessage, scale_key
from ..lrp.utils import get_dummy_summation_conv_layer

if TYPE_CHECKING:
    # Not a runtime import: block_rules.py has no runtime dependency on
    # inverter.py at all (Inverter dispatches to prop_* functions via
    # inv_funcs, populated externally by YOLOLRP - it never imports
    # PROP_REGISTRY itself). Only used here for typing prop_* signatures.
    from ..lrp.inverter import Inverter

# prop_* functions come in two distinct shapes:
#
# - TopLevelPropFunc: always LayerRelevance in, LayerRelevance out. Used by
#   module types that are *only* ever a real Inverter.module_list entry
#   (C2f/C3k2, C2PSA, SPPF, Concat, Detect).
# - NestedPropFunc: always a plain tensor in, plain tensor out. Used by
#   module types that only ever appear *inside* another block's forward
#   (Bottleneck inside C3/C2f/C3k2's `self.m`; PSABlock inside C2PSA/C3k2's
#   `self.m`; DFL inside Detect) - by the time these run, the enclosing
#   TopLevelPropFunc has already scattered the real LayerRelevance to a
#   plain tensor once, at its own entry. Forcing these to accept/return
#   LayerRelevance too would mean wrapping/unwrapping one for every single
#   Bottleneck in every C3k2/C2f block, on every explain() call, for a
#   top-level dispatch that can never actually happen given how these
#   types are registered.
#
# Conv/DWConv (prop_Conv) and C3/C3k (prop_C3) are a third shape:
# genuinely dual-mode, branching internally on
# isinstance(relevance, LayerRelevance), because their PROP_REGISTRY
# entry is reached both ways in a real model - C3k in particular is
# *only* ever reached nested (inside C3k2's `self.m`, alongside plain
# Bottleneck), never top-level, even though its registered function
# (prop_C3) also has to keep working for top-level C3 dispatch too.
# Confirmed against a real yolo26n.pt: most of its C3k2 blocks' `self.m`
# contains C3k, not Bottleneck.
#
# Mirrors PropRule.propagate() (LayerRelevance-aware) vs. compute() (pure
# tensor) in rules.py - same split, same reason.
TopLevelPropFunc = Callable[
    ["Inverter", torch.nn.Module, LayerRelevance], LayerRelevance
]
NestedPropFunc = Callable[["Inverter", torch.nn.Module, torch.Tensor], torch.Tensor]
DualModePropFunc = Callable[
    ["Inverter", torch.nn.Module, Union[LayerRelevance, torch.Tensor]],
    Union[LayerRelevance, torch.Tensor],
]

# PROP_REGISTRY holds all three shapes (registration doesn't know in
# advance which role a type plays), so this is the type used wherever
# "any kind of prop function" is being stored/passed generically.
PropFunc = Union[TopLevelPropFunc, NestedPropFunc, DualModePropFunc]


def _split_additive_relevance(
    inverter: "Inverter",
    a: torch.Tensor,
    b: torch.Tensor,
    relevance: torch.Tensor,
):
    """
    Splits relevance for `a + b` back into relevance for `a` and relevance
    for `b`, via the dummy-summation-conv trick used throughout this
    module for residual/branch additions (get_dummy_summation_conv_layer):
    a 1x1 conv whose fixed weights literally reproduce `a + b` from
    `cat(a, b)`, so ConvRule's own LRP rule performs the actual
    (activation-magnitude-proportional) split, rather than a hand-rolled
    ratio computed here.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    a : torch.Tensor
        First addend (e.g. a residual/shortcut branch).

    b : torch.Tensor
        Second addend (e.g. the branch being shortcut around).

    relevance : torch.Tensor
        Incoming relevance for `a + b`.

    Returns
    -------

    Tuple[torch.Tensor, torch.Tensor]
        (relevance for `a`, relevance for `b`).
    """

    c = a.size(1)
    dummy_conv = get_dummy_summation_conv_layer(c)
    ab = torch.cat((a, b), dim=1)
    _conv_nd_fwd_hook(dummy_conv, [ab], a + b)
    split = inverter(dummy_conv, relevance)
    return split[:, :c, ...], split[:, c:, ...]


def prop_SPPF(
    inverter: "Inverter", mod: torch.nn.Module, relevance: LayerRelevance
) -> LayerRelevance:
    """
    Relevance propagation rule for SPPF (Spatial Pyramid Pooling - Fast).

    Generalizes over `mod.n` (number of chained max-pool stages - YOLOv8's
    SPPF always had 3, but it's now a config parameter) instead of assuming
    exactly 4 concatenated branches. Undoes cv2's channel-concat by summing
    all `n + 1` branch chunks together (cv1's own output plus one chunk per
    pooling stage) - this mirrors the max-pool cascade as a simple additive
    merge rather than an exact winner-takes-all inversion through each
    pooling stage (SPPF_fwd_hook's cached indices aren't consumed here).

    Also handles YOLO26's new `y + x` residual (`mod.add`, never present in
    YOLOv8's SPPF): relevance for it is split from the rest the same way
    prop_Bottleneck splits its own residual connection, via a dummy 1x1 conv
    that mirrors the `+` as a concat-then-sum.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The SPPF module being inverted.

    relevance : LayerRelevance
        Incoming relevance (a LayerRelevance - SPPF is always a top-level
        module_list entry, never called on a nested sub-module).

    Returns
    -------

    LayerRelevance
        Relevance re-gathered at SPPF's own input.
    """

    sppf = cast(SPPFLike, mod)

    if getattr(sppf, "add", False):
        msg = relevance.scatter(which=-1)
        c = msg.size(1)
        dummy_conv = get_dummy_summation_conv_layer(c)
        x = cast(HasInTensor, sppf.cv1.conv).in_tensor
        y = cast(HasOutTensor, sppf.cv2.conv).out_tensor
        xy = torch.concatenate((x, y), dim=1)
        _conv_nd_fwd_hook(dummy_conv, [xy], (x + y))
        msg = inverter(dummy_conv, msg)
        msg_residual = msg[:, :c, ...]
        msg = msg[:, c:, ...]
    else:
        msg_residual = None
        msg = relevance.scatter(which=-1)

    # getattr, not sppf.n directly: matches ultralytics' own forward()
    # (getattr(self, "n", 3)) - checkpoints pickled before `n`/`add` were
    # added to SPPF (e.g. YOLOv10's) restore an instance whose __dict__
    # genuinely lacks the attribute, not just an unset default.
    n = getattr(sppf, "n", 3)
    msg = inverter(sppf.cv2, msg)
    ch = msg.size(1) // (n + 1)
    rx = sum(msg[:, i * ch : (i + 1) * ch, ...] for i in range(n + 1))

    msg = inverter(sppf.cv1, rx)
    if msg_residual is not None:
        msg = msg + msg_residual

    relevance.gather(
        RelevanceMessage(from_=getattr(sppf, "reg_num", None), to=-1, relevance=msg)
    )

    return relevance


def prop_Concat(
    inverter: "Inverter", mod: torch.nn.Module, relevance: LayerRelevance
) -> LayerRelevance:
    """
    Relevance propagation rule for Concat: splits relevance along the same
    axis and slice widths (`mod.d`, `mod.in_shapes`, cached by
    Concat_fwd_hook) that were concatenated in the forward pass, and
    routes each slice to the layer it came from (`mod.f`).

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass (unused -
        Concat's relevance split needs no further propagation of its own).

    mod : torch.nn.Module
        The Concat module being inverted.

    relevance : LayerRelevance
        Incoming relevance (a LayerRelevance - Concat is always a
        top-level module_list entry).

    Returns
    -------

    LayerRelevance
        Relevance re-gathered at each of Concat's source layers.
    """

    concat = cast(ConcatLike, mod)
    slices = relevance.scatter(-1).split(concat.in_shapes, dim=concat.d)
    relevance.gather(
        *(
            RelevanceMessage(
                from_=getattr(concat, "reg_num", None), to=to, relevance=msg
            )
            for to, msg in zip(concat.f, slices)
        )
    )

    return relevance


def prop_Detect(
    inverter: "Inverter", mod: torch.nn.Module, relevance: LayerRelevance
) -> LayerRelevance:
    """
    Relevance propagation rule for the Detect head's classification branch
    (cv3 only - box regression's cv2 branch is never used, since explain()
    only ever seeds relevance from cv3's class predictions).

    Which head-branch layer each cv3[i] output should propagate back to is
    read from `inverter.prop_to` (set by YOLOLRP.__init__ via
    `_get_prop_to`, from the model's own parsed yaml config by default)
    rather than a hardcoded layer-index list, since that list is
    architecture-specific and silently wrong if Ultralytics ever reorders
    these layers again.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The Detect module being inverted.

    relevance : LayerRelevance
        Incoming relevance (a LayerRelevance - Detect is always a
        top-level module_list entry).

    Returns
    -------

    LayerRelevance
        Relevance re-gathered at each of Detect's cv3 input layers.
    """

    detect = cast(DetectLike, mod)
    relevance_out = []

    # One seed message per detection scale, keyed by scale_key(i) - see
    # _RelevanceInitializer, which seeds them that way since the scales
    # have differing spatial shapes and can't be merged into one message.
    for i, to in enumerate(inverter.prop_to):
        rel = relevance.scatter(which=scale_key(i))
        relevance_out.append(
            RelevanceMessage(
                from_=getattr(detect, "reg_num", None),
                to=to,
                relevance=inverter(detect.cv3[i], rel),
            )
        )

    relevance.gather(*relevance_out)
    return relevance


def prop_Conv(
    inverter: "Inverter",
    mod: torch.nn.Module,
    relevance: Union[LayerRelevance, torch.Tensor],
) -> Union[LayerRelevance, torch.Tensor]:
    """
    Relevance propagation rule for Conv (and, by reuse, DWConv - see
    PROP_REGISTRY): delegates straight to the inner nn.Conv2d, which
    ConvRule (a primitive-level rule, see rules.py) actually handles.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The Conv/DWConv wrapper module being inverted.

    relevance : LayerRelevance or torch.Tensor
        Incoming relevance, either a LayerRelevance (top-level dispatch)
        or a plain tensor (nested call from another prop_* function).

    Returns
    -------

    LayerRelevance or torch.Tensor
        Relevance re-gathered (or, for a plain tensor, simply computed) at
        the wrapped Conv2d's input.
    """

    conv = cast(HasConv, mod).conv
    # relevance's static type is a Union, but RelevanceT is a *constrained*
    # TypeVar - it can't bind to "either at once" from one call site, only
    # to whichever concrete member mypy has narrowed relevance to. Same
    # value, same call, just narrowed first so each branch binds cleanly.
    if isinstance(relevance, LayerRelevance):
        return inverter(conv, relevance)
    return inverter(conv, relevance)


def prop_C3(
    inverter: "Inverter",
    mod: torch.nn.Module,
    relevance: Union[LayerRelevance, torch.Tensor],
) -> Union[LayerRelevance, torch.Tensor]:
    """
    Relevance propagation rule for C3 (and, by reuse, C3k - see
    PROP_REGISTRY - since C3k subclasses C3 without changing its
    cv1/cv2/cv3/m shape, only what `self.m` contains).

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The C3/C3k module being inverted.

    relevance : LayerRelevance or torch.Tensor
        Incoming relevance, either a LayerRelevance (top-level dispatch -
        C3 itself is never actually top-level in YOLO26, but C3 is kept
        dual-mode for parity with C3k) or a plain tensor (the actual live
        case for C3k, which only ever appears nested inside C3k2's
        `self.m` - never a top-level module_list entry in this real
        architecture).

    Returns
    -------

    LayerRelevance or torch.Tensor
        Relevance re-gathered at the module's own input, if given a
        LayerRelevance; otherwise the plain tensor for that input.
    """

    c3 = cast(C3Like, mod)
    is_wrapped = isinstance(relevance, LayerRelevance)
    msg: torch.Tensor = (
        cast(LayerRelevance, relevance).scatter(which=-1)
        if is_wrapped
        else cast(torch.Tensor, relevance)
    )

    msg = inverter(c3.cv3, msg)

    c_ = msg.size(1)

    msg_cv1 = msg[:, : (c_ // 2), ...]
    msg_cv2 = msg[:, (c_ // 2) :, ...]

    for m1 in c3.m:
        msg_cv1 = inverter(m1, msg_cv1)

    msg = inverter(c3.cv1, msg_cv1) + inverter(c3.cv2, msg_cv2)

    if not is_wrapped:
        return msg

    cast(LayerRelevance, relevance).gather(
        RelevanceMessage(from_=getattr(c3, "reg_num", None), to=-1, relevance=msg)
    )

    return relevance


def prop_Bottleneck(
    inverter: "Inverter", mod: torch.nn.Module, relevance: torch.Tensor
) -> torch.Tensor:
    """
    Relevance propagation rule for Bottleneck's residual connection
    (`x + cv2(cv1(x))`). A NestedPropFunc, not a TopLevelPropFunc: never a
    top-level module_list entry (Bottleneck only ever appears nested
    inside C3/C3k/C2f/C3k2's `self.m`), so `relevance` here is always a
    plain tensor, not a LayerRelevance - unlike every top-level prop_*
    function, this one never scatters/gathers.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The Bottleneck module being inverted.

    relevance : torch.Tensor
        Incoming relevance.

    Returns
    -------

    torch.Tensor
        Relevance for Bottleneck's own input.
    """

    bneck = cast(HasCV1CV2, mod)
    msg = relevance
    c = msg.shape[1]

    dummy_conv = get_dummy_summation_conv_layer(c)
    x = cast(HasInTensor, bneck.cv1.conv).in_tensor

    y = cast(HasOutTensor, bneck.cv2.conv).out_tensor
    xy = torch.concatenate((x, y), dim=1)
    _conv_nd_fwd_hook(dummy_conv, [xy], (x + y))

    msg = inverter(dummy_conv, msg)
    msg_in = msg[:, :c, ...]
    msg_res = msg[:, c:, ...]
    # msg_res is relevance for y = cv2(cv1(x)) - cv2's output - so it must
    # be inverted through cv2 first (undoing the outer function), then
    # cv1, mirroring the reverse of the forward order (x -> cv1 -> cv2 ->
    # y). Inverting cv1 first fed it relevance shaped for cv2's output,
    # which is only shape-compatible by accident when cv1/cv2 preserve
    # channel count (e.g. a channel-reducing/expanding real Bottleneck
    # crashes instead).
    msg_res = inverter(bneck.cv2, msg_res)
    msg_res = inverter(bneck.cv1, msg_res)
    msg = msg_in + msg_res

    return msg


def prop_RepBottleneck(
    inverter: "Inverter", mod: torch.nn.Module, relevance: torch.Tensor
) -> torch.Tensor:
    """
    Relevance propagation rule for RepBottleneck: the same `x +
    cv2(cv1(x))` residual as prop_Bottleneck, but `cv1` is a RepConv (no
    single `.conv` submodule of its own - see RepConvLike), not a plain
    Conv wrapper, so the shared input `x` is read off `cv1.conv1`'s own
    `.conv.in_tensor` instead (`conv1` is one of RepConv's two internal
    branches, applied directly to `x` - see RepConvLike/
    _propagate_parallel_conv_sum - so its cached input tensor is the same
    `x`, regardless of which branch reads it).

    A NestedPropFunc, not a TopLevelPropFunc: RepBottleneck only ever
    appears nested inside RepCSP's `self.m`.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The RepBottleneck module being inverted.

    relevance : torch.Tensor
        Incoming relevance.

    Returns
    -------

    torch.Tensor
        Relevance for RepBottleneck's own input.
    """

    bneck = cast(HasCV1CV2, mod)
    msg = relevance
    c = msg.shape[1]

    dummy_conv = get_dummy_summation_conv_layer(c)
    cv1 = cast(RepConvLike, bneck.cv1)
    x = cast(HasInTensor, cv1.conv1.conv).in_tensor
    y = cast(HasOutTensor, bneck.cv2.conv).out_tensor
    xy = torch.concatenate((x, y), dim=1)
    _conv_nd_fwd_hook(dummy_conv, [xy], (x + y))

    msg = inverter(dummy_conv, msg)
    msg_in = msg[:, :c, ...]
    msg_res = msg[:, c:, ...]
    msg_res = inverter(bneck.cv2, msg_res)
    msg_res = inverter(bneck.cv1, msg_res)
    msg = msg_in + msg_res

    return msg


def prop_C2f(
    inverter: "Inverter", mod: torch.nn.Module, relevance: LayerRelevance
) -> LayerRelevance:
    """
    Relevance propagation rule for C2f (and, by reuse, C3k2 - see
    PROP_REGISTRY - since C3k2 subclasses C2f without overriding
    forward(), only its choice of `self.m` blocks).

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The C2f/C3k2 module being inverted.

    relevance : LayerRelevance
        Incoming relevance (a LayerRelevance - C2f/C3k2 is always a
        top-level module_list entry).

    Returns
    -------

    LayerRelevance
        Relevance re-gathered at the module's own input.
    """

    c2f = cast(C2fLike, mod)

    # Extract relevant tensors from the module
    msg = relevance.scatter(which=-1)
    msg_cv2 = inverter(c2f.cv2, msg)
    msg_m = list(msg_cv2.chunk(msg_cv2.size(1) // c2f.c, 1))

    # Relevance propagation through the bottleneck blocks (m)
    for i, m_block in enumerate(c2f.m[::-1]):
        msg_m[-(i + 2)] += inverter(m_block, msg_m[-(i + 1)])
    msg_cv1 = torch.cat(msg_m[:2], dim=1)
    msg = inverter(c2f.cv1, msg_cv1)
    relevance.gather(
        RelevanceMessage(from_=getattr(c2f, "reg_num", None), to=-1, relevance=msg)
    )
    return relevance


def prop_C2PSA(
    inverter: "Inverter", mod: torch.nn.Module, relevance: LayerRelevance
) -> LayerRelevance:
    """
    Relevance propagation rule for C2PSA (the attention-augmented CSP block
    YOLO26's backbone ends on - new vs. YOLOv8, which had no attention
    stage at all).

    Splits cv2's inverted relevance into the untouched `a` half and the
    attention-processed `b` half (the same channel split C2PSA's own
    forward uses, via `mod.c`), propagates `b` through `mod.m` (a
    Sequential of PSABlock, each handled by prop_PSABlock/prop_Attention),
    then inverts cv1 on the recombined result.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The C2PSA module being inverted.

    relevance : LayerRelevance
        Incoming relevance (a LayerRelevance - C2PSA is always a
        top-level module_list entry).

    Returns
    -------

    LayerRelevance
        Relevance re-gathered at C2PSA's own input.
    """

    c2psa = cast(C2PSALike, mod)
    msg = relevance.scatter(which=-1)
    msg = inverter(c2psa.cv2, msg)

    c = c2psa.c
    msg_a = msg[:, :c, ...]
    msg_b = msg[:, c:, ...]
    msg_b = inverter(c2psa.m, msg_b)

    msg = inverter(c2psa.cv1, torch.cat([msg_a, msg_b], dim=1))
    relevance.gather(
        RelevanceMessage(from_=getattr(c2psa, "reg_num", None), to=-1, relevance=msg)
    )
    return relevance


# Small, fixed epsilon for the attention relevance rules below - same role
# as PropRule's own `eps` (stabilizes division near zero), but these are
# free functions rather than PropRule instances, so there's no self.eps to
# read. Not exposed as a knob: nothing else in this module parameterizes
# prop_* functions either.
_ATTN_EPS = 1e-6


def _bilinear_relevance(
    P: torch.Tensor,
    Q: torch.Tensor,
    relevance_z: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    eps: float = _ATTN_EPS,
):
    """
    Splits relevance for a batched matrix product `z = P @ Q` back into
    relevance for each factor, via the AttnLRP-style bilinear rule
    (Achtibat et al. 2024): unlike a Conv/Linear layer, both `P` and `Q`
    are activations here (neither is a fixed learned weight), so each
    output element `z[...,m,n] = sum_k P[...,m,k] * Q[...,k,n]` is
    explained by *both* factors at once - the relevance for each is
    computed by treating the *other* factor as if it were the "weight" in
    an ordinary LRP-epsilon step, then swapping roles. Used twice by
    `prop_Attention`: once for `Z = V @ Aᵀ` (the attention-weighted sum of
    values) and once for `S = Qᵀ @ K` (the raw attention scores).

    Note this deliberately does *not* halve the relevance between `P` and
    `Q`: each of the two returned tensors independently carries the full
    incoming `relevance_z` total onward down its own path (mirroring how
    Q/K/V's separate paths get summed back together into a single `R_X`
    once they reconverge at the attention block's own input) - see
    `prop_Attention`'s docstring for why.

    Arguments
    ---------

    P : torch.Tensor
        First factor, shape (..., M, K).

    Q : torch.Tensor
        Second factor, shape (..., K, N).

    relevance_z : torch.Tensor
        Incoming relevance for `z = P @ Q`, shape (..., M, N).

    z : torch.Tensor, optional
        `P @ Q`, if already computed by the caller (avoids recomputing
        it). Computed here if not given.

    eps : float
        Stabilizer added to `z` before dividing by it, same role as
        PropRule's own `eps`.

    Returns
    -------

    Tuple[torch.Tensor, torch.Tensor]
        (relevance for `P`, relevance for `Q`).
    """

    if z is None:
        z = P @ Q

    z_stab = z + torch.sign(z) * eps
    z_stab[z_stab == 0] = 1
    s = relevance_z / z_stab

    relevance_P = P * (s @ Q.transpose(-2, -1))
    relevance_Q = Q * (P.transpose(-2, -1) @ s)

    return relevance_P, relevance_Q


def _softmax_relevance(
    scores: torch.Tensor,
    attn_weights: torch.Tensor,
    relevance_attn: torch.Tensor,
) -> torch.Tensor:
    """
    Relevance propagation rule for `attn_weights = softmax(scores)`,
    following AttnLRP's treatment of softmax as its own Taylor
    decomposition rather than an ordinary elementwise nonlinearity
    (unlike e.g. SiLU/ReLU elsewhere in this package, which are close
    enough to identity for LRP purposes to just pass relevance through
    unchanged via IDENTITY_MAPPINGS - softmax's normalization couples
    every output to every input along the softmax axis, so that
    shortcut doesn't hold here).

    Derived from softmax's own Jacobian, `d(attn_weights_i)/d(scores_k) =
    attn_weights_i * (delta_ik - attn_weights_k)`, via the standard LRP
    Deep Taylor identity `R_k = scores_k * sum_i (d(attn_weights_i)/d
    (scores_k)) * (R_i / attn_weights_i)`: the `attn_weights_i` factors
    cancel, leaving `R_scores = scores * (R_attn_weights - attn_weights *
    sum(R_attn_weights, axis=softmax_axis))`.

    Arguments
    ---------

    scores : torch.Tensor
        Pre-softmax attention scores.

    attn_weights : torch.Tensor
        Post-softmax attention weights (`softmax(scores, dim=-1)`).

    relevance_attn : torch.Tensor
        Incoming relevance for `attn_weights`.

    Returns
    -------

    torch.Tensor
        Relevance for `scores`.
    """

    total = relevance_attn.sum(dim=-1, keepdim=True)
    return scores * (relevance_attn - attn_weights * total)


def prop_Attention(
    inverter: "Inverter", mod: torch.nn.Module, relevance: torch.Tensor
) -> torch.Tensor:
    """
    AttnLRP-style (Achtibat et al. 2024) relevance propagation rule for
    Attention's single-head-per-group self-attention:

        qkv = qkv_conv(x);  q, k, v = split(qkv)
        scores = (q * scale)ᵀ @ k;  attn = softmax(scores)
        z = v @ attnᵀ
        out = proj(z.reshape(...) + pe(v.reshape(...)))

    Decomposes relevance back through every one of those steps -
    `proj`/`pe`/`qkv` via the existing ConvRule (they're plain, if
    unusually-shaped, convolutions), the `z + pe(...)` sum via the same
    dummy-summation-conv split used for residuals elsewhere in this
    module, and the two matrix products (`v @ attnᵀ` and `qᵀ @ k`) via
    `_bilinear_relevance`, with `_softmax_relevance` handling the
    softmax in between - rather than treating the whole block as an
    identity placeholder (the previous behavior; see git history).

    Like every other prop_* function inverting a Conv wrapper's own
    Conv2d directly (see prop_Conv), this treats BatchNorm as
    transparent to relevance - reading `.conv.out_tensor`/`.bn.out_tensor`
    for the *forward* magnitudes needed by the math below, but relying on
    the existing `inverter(mod.qkv, ...)`/`inverter(mod.pe, ...)`/
    `inverter(mod.proj, ...)` calls (which dispatch to prop_Conv ->
    ConvRule) for the actual backward step, exactly like every other
    Conv wrapper in this file.

    A NestedPropFunc, not a TopLevelPropFunc: Attention is only ever
    nested inside PSABlock, never a top-level module_list entry, so
    `relevance` here is always a plain tensor, matching PSABlock's own
    `prop_PSABlock`.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The Attention module being inverted.

    relevance : torch.Tensor
        Incoming relevance for Attention's own output.

    Returns
    -------

    torch.Tensor
        Relevance for Attention's own input `x`.
    """

    attn = cast(AttentionLike, mod)
    qkv = cast(HasOutTensor, attn.qkv.conv).out_tensor
    B, h, H, W = qkv.shape
    N = H * W
    num_heads, key_dim, head_dim = attn.num_heads, attn.key_dim, attn.head_dim

    q, k, v = qkv.view(B, num_heads, key_dim * 2 + head_dim, N).split(
        [key_dim, key_dim, head_dim], dim=2
    )
    q_scaled = q * attn.scale
    scores = q_scaled.transpose(-2, -1) @ k
    attn_weights = scores.softmax(dim=-1)
    z = v @ attn_weights.transpose(-2, -1)

    # relevance is for proj(z.reshape(B, C, H, W) + pe(v.reshape(B, C, H, W)))
    relevance_sum = inverter(attn.proj, relevance)

    z_reshaped = z.reshape(B, -1, H, W)
    pe_out = cast(HasOutTensor, attn.pe.conv).out_tensor
    relevance_z_reshaped, relevance_pe_out = _split_additive_relevance(
        inverter, z_reshaped, pe_out, relevance_sum
    )

    # -1 (not B): under contrastive=True, relevance carries a doubled
    # (primal+dual) batch that never applies to the real forward-cached
    # activations q/k/v/scores/attn_weights/z above (always the true,
    # single-image batch B) - ConvRule handles that mismatch internally
    # for inverter(mod.pe, ...)/inverter(mod.proj, ...)/inverter(mod.qkv,
    # ...) (see rules.py), and _bilinear_relevance/_softmax_relevance
    # broadcast over it, but these two `.view()` calls need the tensor's
    # own actual batch size, which may not be B.
    relevance_v_from_pe = inverter(attn.pe, relevance_pe_out).view(
        -1, num_heads, head_dim, N
    )
    relevance_z = relevance_z_reshaped.view(-1, num_heads, head_dim, N)

    # z = v @ attn_weightsᵀ
    relevance_v_from_attn, relevance_attn_t = _bilinear_relevance(
        v, attn_weights.transpose(-2, -1), relevance_z, z=z
    )
    relevance_attn_weights = relevance_attn_t.transpose(-2, -1)

    relevance_scores = _softmax_relevance(scores, attn_weights, relevance_attn_weights)

    # scores = q_scaledᵀ @ k
    relevance_q_scaled_t, relevance_k = _bilinear_relevance(
        q_scaled.transpose(-2, -1), k, relevance_scores, z=scores
    )
    # `scale` is a positive scalar multiply (q_scaled = q * mod.scale), so
    # relevance passes through it unchanged - same as any positive linear
    # rescaling under LRP-epsilon.
    relevance_q = relevance_q_scaled_t.transpose(-2, -1)

    relevance_v = relevance_v_from_attn + relevance_v_from_pe

    # Same reasoning as above: reshape to the real channel count `h` but
    # let batch be inferred, since it may be doubled relative to B.
    relevance_qkv = torch.cat([relevance_q, relevance_k, relevance_v], dim=2).reshape(
        -1, h, H, W
    )

    return inverter(attn.qkv, relevance_qkv)


def _propagate_attn_ffn_residuals(
    inverter: "Inverter",
    attn_mod: torch.nn.Module,
    ffn_mod: torch.nn.Sequential,
    msg: torch.Tensor,
) -> torch.Tensor:
    """
    Inverts the `x1 = x + attn(x); x2 = x1 + ffn(x1)` residual chain
    shared by PSABlock's `add=True` branch and PSA's own (always-
    residual) attn/ffn pair - undoing the outer sum (ffn's) first, then
    the inner one (attn's), each via `_split_additive_relevance`, mirroring
    prop_Bottleneck/prop_SPPF's own residual handling.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    attn_mod : torch.nn.Module
        The Attention submodule (dispatches to prop_Attention).

    ffn_mod : torch.nn.Sequential
        The feed-forward Sequential-of-Conv submodule.

    msg : torch.Tensor
        Incoming relevance for `x2` (the chain's own output).

    Returns
    -------

    torch.Tensor
        Relevance for `x` (the chain's own input).
    """

    ffn_in = cast(HasInTensor, ffn_mod[0].conv).in_tensor
    ffn_out = cast(HasOutTensor, ffn_mod[-1].bn).out_tensor
    msg_x1, msg_ffn_out = _split_additive_relevance(inverter, ffn_in, ffn_out, msg)
    msg = msg_x1 + inverter(ffn_mod, msg_ffn_out)

    attn = cast(AttentionLike, attn_mod)
    attn_in = cast(HasInTensor, attn.qkv.conv).in_tensor
    attn_out = cast(HasOutTensor, attn.proj.bn).out_tensor
    msg_x, msg_attn_out = _split_additive_relevance(inverter, attn_in, attn_out, msg)
    msg = msg_x + inverter(attn_mod, msg_attn_out)

    return msg


def prop_PSABlock(
    inverter: "Inverter", mod: torch.nn.Module, relevance: torch.Tensor
) -> torch.Tensor:
    """
    Relevance propagation rule for PSABlock:

        x1 = x + attn(x)   if mod.add else attn(x)
        x2 = x1 + ffn(x1)  if mod.add else ffn(x1)

    If `mod.add`, both residual sums are split via
    `_propagate_attn_ffn_residuals` (shared with prop_PSA, which always
    has both residuals); otherwise `ffn`/`attn` (a plain nn.Sequential of
    Conv - handled generically by Inverter.invert's own Sequential
    recursion - and prop_Attention, respectively) are inverted directly.

    A NestedPropFunc, not a TopLevelPropFunc: PSABlock is only ever
    nested inside C2PSA/C3k2's `self.m`, never a top-level module_list
    entry, so `relevance` here is always a plain tensor.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The PSABlock module being inverted.

    relevance : torch.Tensor
        Incoming relevance for PSABlock's own output.

    Returns
    -------

    torch.Tensor
        Relevance for PSABlock's own input.
    """

    psa = cast(PSABlockLike, mod)

    if getattr(psa, "add", False):
        return _propagate_attn_ffn_residuals(inverter, psa.attn, psa.ffn, relevance)

    msg = inverter(psa.ffn, relevance)
    return inverter(psa.attn, msg)


# --- YOLOv10-specific blocks ---
#
# v10Detect/C2fCIB need no new prop_* function at all - see PROP_REGISTRY
# below, where they're registered straight against prop_Detect/prop_C2f
# (same as C3k2 already is against prop_C2f).


def prop_SCDown(
    inverter: "Inverter", mod: torch.nn.Module, relevance: LayerRelevance
) -> LayerRelevance:
    """
    Relevance propagation rule for SCDown (Spatial-Channel decoupled
    downsampling): `cv2(cv1(x))`, two plain sequential Convs with no
    branching or residual - simpler than every other TopLevelPropFunc in
    this module.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The SCDown module being inverted.

    relevance : LayerRelevance
        Incoming relevance (a LayerRelevance - SCDown is always a
        top-level module_list entry).

    Returns
    -------

    LayerRelevance
        Relevance re-gathered at SCDown's own input.
    """

    scdown = cast(HasCV1CV2, mod)
    msg = relevance.scatter(which=-1)
    msg = inverter(scdown.cv2, msg)
    msg = inverter(scdown.cv1, msg)
    relevance.gather(
        RelevanceMessage(from_=getattr(scdown, "reg_num", None), to=-1, relevance=msg)
    )
    return relevance


def prop_PSA(
    inverter: "Inverter", mod: torch.nn.Module, relevance: LayerRelevance
) -> LayerRelevance:
    """
    Relevance propagation rule for PSA (Position-Sensitive Attention):

        a, b = cv1(x).split((c, c), dim=1)
        b = b + attn(b)
        b = b + ffn(b)
        return cv2(cat(a, b))

    The same split/attn/ffn/merge shape as prop_C2PSA's cv1/cv2 split
    plus a nested PSABlock's residuals, just fused into one module
    (`attn`/`ffn` are direct submodules here, always residual, rather
    than wrapped in a PSABlock of their own) - so the untouched `a` half
    is handled exactly like prop_C2PSA, and `b`'s two residual sums are
    handled by `_propagate_attn_ffn_residuals` (shared with
    prop_PSABlock's own `add=True` branch).

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The PSA module being inverted.

    relevance : LayerRelevance
        Incoming relevance (a LayerRelevance - PSA is always a top-level
        module_list entry).

    Returns
    -------

    LayerRelevance
        Relevance re-gathered at PSA's own input.
    """

    psa = cast(PSALike, mod)
    msg = relevance.scatter(which=-1)
    msg = inverter(psa.cv2, msg)

    c = psa.c
    msg_a = msg[:, :c, ...]
    msg_b = msg[:, c:, ...]
    msg_b = _propagate_attn_ffn_residuals(inverter, psa.attn, psa.ffn, msg_b)

    msg = inverter(psa.cv1, torch.cat([msg_a, msg_b], dim=1))
    relevance.gather(
        RelevanceMessage(from_=getattr(psa, "reg_num", None), to=-1, relevance=msg)
    )
    return relevance


def _propagate_parallel_conv_sum(
    inverter: "Inverter",
    branch_a: torch.nn.Module,
    branch_b: torch.nn.Module,
    relevance: torch.Tensor,
) -> torch.Tensor:
    """
    Inverts `act(branch_a(x) + branch_b(x))` - two parallel Conv-wrapper
    branches off the same input `x`, summed (unlike a residual, neither
    branch is `x` itself - both are convolutions of it) - shared by
    RepVGGDW (`conv`/`conv1`) and RepConv (`conv1`/`conv2`, when its
    optional identity-BN branch is unused; see RepConvLike's own note on
    why that's always true here). Splits relevance for the sum via
    `_split_additive_relevance` (same dummy-summation-conv trick as every
    other additive split in this module), then sums the two branches'
    independently-inverted relevance for `x` back together, since both
    genuinely explain the same input.

    Like prop_Conv and friends, treats the trailing `act` (SiLU) as
    transparent to relevance - never separately dispatched, since it's
    fused into the caller's own forward rather than a submodule Inverter
    ever sees on its own (same treatment IDENTITY_MAPPINGS gives SiLU
    everywhere else).

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    branch_a : torch.nn.Module
        First Conv-wrapper branch.

    branch_b : torch.nn.Module
        Second Conv-wrapper branch.

    relevance : torch.Tensor
        Incoming relevance for `branch_a(x) + branch_b(x)`.

    Returns
    -------

    torch.Tensor
        Relevance for the shared input `x`.
    """

    a = cast(HasOutTensor, branch_a.bn).out_tensor
    b = cast(HasOutTensor, branch_b.bn).out_tensor
    msg_a, msg_b = _split_additive_relevance(inverter, a, b, relevance)

    return inverter(branch_a, msg_a) + inverter(branch_b, msg_b)


def prop_RepVGGDW(
    inverter: "Inverter", mod: torch.nn.Module, relevance: torch.Tensor
) -> torch.Tensor:
    """
    Relevance propagation rule for RepVGGDW: `act(conv(x) + conv1(x))` -
    see `_propagate_parallel_conv_sum`.

    A NestedPropFunc, not a TopLevelPropFunc: RepVGGDW is only ever
    nested inside CIB's `cv1` Sequential, never a top-level module_list
    entry.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The RepVGGDW module being inverted.

    relevance : torch.Tensor
        Incoming relevance for RepVGGDW's own output.

    Returns
    -------

    torch.Tensor
        Relevance for RepVGGDW's own input `x`.
    """

    rep = cast(RepVGGDWLike, mod)
    return _propagate_parallel_conv_sum(inverter, rep.conv, rep.conv1, relevance)


def prop_RepConv(
    inverter: "Inverter", mod: torch.nn.Module, relevance: torch.Tensor
) -> torch.Tensor:
    """
    Relevance propagation rule for RepConv: `act(conv1(x) + conv2(x) +
    id_out)` - see `_propagate_parallel_conv_sum` (`id_out` is always 0
    here; see RepConvLike's own note).

    A NestedPropFunc, not a TopLevelPropFunc: RepConv is only ever
    nested inside RepBottleneck's `cv1`, itself only ever nested inside
    RepCSP's `self.m`, never a top-level module_list entry.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The RepConv module being inverted.

    relevance : torch.Tensor
        Incoming relevance for RepConv's own output.

    Returns
    -------

    torch.Tensor
        Relevance for RepConv's own input `x`.
    """

    rep = cast(RepConvLike, mod)
    return _propagate_parallel_conv_sum(inverter, rep.conv1, rep.conv2, relevance)


def prop_AConv(
    inverter: "Inverter", mod: torch.nn.Module, relevance: LayerRelevance
) -> LayerRelevance:
    """
    Relevance propagation rule for AConv: `cv1(avg_pool2d(x, 2, 1, 0,
    False, True))`. `cv1` is inverted normally (dispatches to prop_Conv
    -> ConvRule); the avg_pool2d before it - a fixed, non-learned,
    stride-1 kernel=2 anti-aliasing blur that genuinely shrinks H/W by 1
    (cv1's own stride=2 does the actual downsampling) - is inverted via
    its own adjoint, `conv_transpose2d` with uniform 1/(k*k) weights
    (mirroring how UpsampleRule in rules.py uses avg_pool2d as nearest-
    upsampling's adjoint, just in the opposite direction here), since
    treating it as literally transparent would leave a shape mismatch.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The AConv module being inverted.

    relevance : LayerRelevance
        Incoming relevance (a LayerRelevance - AConv is always a
        top-level module_list entry).

    Returns
    -------

    LayerRelevance
        Relevance re-gathered at AConv's own input.
    """

    aconv = cast(HasCV1, mod)
    msg = relevance.scatter(which=-1)
    msg = inverter(aconv.cv1, msg)

    c = msg.size(1)
    weight = torch.full((c, 1, 2, 2), 0.25, device=msg.device, dtype=msg.dtype)
    msg = torch.nn.functional.conv_transpose2d(msg, weight, stride=1, groups=c)

    relevance.gather(
        RelevanceMessage(from_=getattr(aconv, "reg_num", None), to=-1, relevance=msg)
    )
    return relevance


def prop_RepNCSPELAN4(
    inverter: "Inverter", mod: torch.nn.Module, relevance: LayerRelevance
) -> LayerRelevance:
    """
    Relevance propagation rule for RepNCSPELAN4 (and, by reuse, ELAN1 -
    see RepNCSPELAN4Like's own note):

        y0, y1 = cv1(x).chunk(2, 1)   # each width mod.c
        y2 = cv2(y1); y3 = cv3(y2)    # each width c4 (not its own attribute -
                                       # derived below from the merged width)
        return cv4(cat([y0, y1, y2, y3], 1))

    Mirrors prop_C2f's split/merge shape, but with exactly two fixed
    Sequential submodules (`cv2`, `cv3`) chained instead of an iterated
    ModuleList, so the chain is unrolled directly rather than looped.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The RepNCSPELAN4/ELAN1 module being inverted.

    relevance : LayerRelevance
        Incoming relevance (a LayerRelevance - RepNCSPELAN4/ELAN1 is
        always a top-level module_list entry).

    Returns
    -------

    LayerRelevance
        Relevance re-gathered at the module's own input.
    """

    elan = cast(RepNCSPELAN4Like, mod)
    msg = relevance.scatter(which=-1)
    msg_cv4 = inverter(elan.cv4, msg)

    c = elan.c
    c4 = (msg_cv4.size(1) - 2 * c) // 2
    msg_y0 = msg_cv4[:, :c, ...]
    msg_y1 = msg_cv4[:, c : 2 * c, ...]
    msg_y2 = msg_cv4[:, 2 * c : 2 * c + c4, ...]
    msg_y3 = msg_cv4[:, 2 * c + c4 :, ...]

    msg_y2 = msg_y2 + inverter(elan.cv3, msg_y3)
    msg_y1 = msg_y1 + inverter(elan.cv2, msg_y2)

    msg = inverter(elan.cv1, torch.cat([msg_y0, msg_y1], dim=1))
    relevance.gather(
        RelevanceMessage(from_=getattr(elan, "reg_num", None), to=-1, relevance=msg)
    )
    return relevance


def prop_SPPELAN(
    inverter: "Inverter", mod: torch.nn.Module, relevance: LayerRelevance
) -> LayerRelevance:
    """
    Relevance propagation rule for SPPELAN: the same n+1-way additive-
    merge shape as prop_SPPF (see its own docstring), just with 3
    separately-constructed MaxPool2d branches (`cv2`/`cv3`/`cv4`) chained
    off `cv1`'s output instead of 1 shared instance called `n` times -
    since neither is un-pooled per stage anyway (both treat the whole
    cascade as a single additive merge, skipping straight to `cv1`), the
    two are equivalent here; `cv2`/`cv3`/`cv4` are never read directly.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The SPPELAN module being inverted.

    relevance : LayerRelevance
        Incoming relevance (a LayerRelevance - SPPELAN is always a
        top-level module_list entry).

    Returns
    -------

    LayerRelevance
        Relevance re-gathered at SPPELAN's own input.
    """

    sppelan = cast(SPPELANLike, mod)
    msg = relevance.scatter(which=-1)
    msg = inverter(sppelan.cv5, msg)

    ch = msg.size(1) // 4
    rx = sum(msg[:, i * ch : (i + 1) * ch, ...] for i in range(4))

    msg = inverter(sppelan.cv1, rx)
    relevance.gather(
        RelevanceMessage(from_=getattr(sppelan, "reg_num", None), to=-1, relevance=msg)
    )
    return relevance


def prop_CIB(
    inverter: "Inverter", mod: torch.nn.Module, relevance: torch.Tensor
) -> torch.Tensor:
    """
    Relevance propagation rule for CIB (Compact Inverted Block):
    `x + cv1(x)` if `mod.add` else `cv1(x)`, where `cv1` is a 5-layer
    `nn.Sequential` (Conv/Conv/RepVGGDW-or-Conv/Conv/Conv). Same residual-
    split shape as prop_Bottleneck, collapsed to one Sequential instead of
    a separate cv1/cv2 pair - `inverter(cib.cv1, ...)` inverts the whole
    chain in one call via Inverter.invert's own Sequential recursion
    (which dispatches its middle layer to prop_RepVGGDW or prop_Conv,
    whichever it actually is, same as any other Sequential entry).

    A NestedPropFunc, not a TopLevelPropFunc: CIB is only ever nested
    inside C2fCIB's `self.m`, never a top-level module_list entry.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The CIB module being inverted.

    relevance : torch.Tensor
        Incoming relevance for CIB's own output.

    Returns
    -------

    torch.Tensor
        Relevance for CIB's own input.
    """

    cib = cast(CIBLike, mod)

    if not getattr(cib, "add", False):
        return inverter(cib.cv1, relevance)

    x = cast(HasInTensor, cib.cv1[0].conv).in_tensor
    y = cast(HasOutTensor, cib.cv1[-1].conv).out_tensor
    msg_in, msg_res = _split_additive_relevance(inverter, x, y, relevance)

    return msg_in + inverter(cib.cv1, msg_res)


def prop_DFL(
    inverter: "Inverter", mod: torch.nn.Module, relevance: torch.Tensor
) -> torch.Tensor:
    """
    Relevance propagation rule for DFL (Distribution Focal Loss decode).

    PLACEHOLDER / fallback only: YOLO26's default reg_max=1 makes
    `Detect.dfl` an nn.Identity (handled generically by IDENTITY_MAPPINGS
    in inverter.py), so this is never reached by a default YOLO26 model
    and is kept only for models explicitly configured with reg_max > 1. It
    also isn't exercised by explain() either way, since that only seeds
    relevance from Detect's cv3 (classification) branch, never cv2/dfl
    (box regression) - registered mainly so an unregistered-type crash
    can't happen if that ever changes.

    A NestedPropFunc, not a TopLevelPropFunc: DFL is only ever nested
    inside Detect, never a top-level module_list entry.

    Arguments
    ---------

    inverter : Inverter
        The Inverter instance orchestrating this backward pass.

    mod : torch.nn.Module
        The DFL module being inverted.

    relevance : torch.Tensor
        Incoming relevance tensor.

    Returns
    -------

    torch.Tensor
        Relevance propagated to DFL's input.
    """

    dfl = cast(DFLLike, mod)
    _, _, a = dfl.in_shape
    # Pre-existing, unreachable under YOLO26's default reg_max=1 (see this
    # function's own docstring) - args look swapped (a Tensor where
    # Inverter expects the layer) but that's not this pass's scope to
    # relitigate; silencing the typing complaint only.
    dfl_conv = cast(torch.nn.Conv2d, dfl.conv)
    relevance = inverter(
        relevance.unsqueeze(0), dfl_conv.weight.data  # type: ignore[arg-type]
    ).transpose(2, 1)
    relevance = torch.cat(
        [relevance[0, :, :, ai].flatten().unsqueeze(-1) for ai in range(a)],
        dim=-1,
    ).unsqueeze(0)

    return relevance


# Maps YOLO26 block/module types to the prop_* function that inverts them.
# Dispatch is by exact `type()`, matching RULE_REGISTRY in rules.py (and
# every other dispatch table in this package) - e.g. DWConv needs its own
# entry here despite subclassing Conv. Not meant to be read directly; go
# through `select_prop_rule`.
PROP_REGISTRY: Dict[Type[torch.nn.Module], PropFunc] = {
    block.C3: prop_C3,
    block.C3k: prop_C3,  # C3k subclasses C3, unchanged cv1/cv2/cv3/m shape
    block.C2f: prop_C2f,
    block.C3k2: prop_C2f,  # C3k2 subclasses C2f, no forward() override
    block.C2PSA: prop_C2PSA,
    block.PSABlock: prop_PSABlock,
    block.Attention: prop_Attention,
    block.Conv: prop_Conv,
    conv.DWConv: prop_Conv,  # cv3's depthwise branch; dispatch is by exact type
    head.Detect: prop_Detect,
    block.Bottleneck: prop_Bottleneck,
    conv.Concat: prop_Concat,
    block.SPPF: prop_SPPF,
    block.DFL: prop_DFL,
    # YOLOv10-specific - v10Detect subclasses Detect, C2fCIB subclasses
    # C2f, neither overriding forward() in a way that changes the shape
    # prop_Detect/prop_C2f already handle (same as C3k2 -> prop_C2f above).
    head.v10Detect: prop_Detect,
    block.C2fCIB: prop_C2f,
    block.SCDown: prop_SCDown,
    block.PSA: prop_PSA,
    block.CIB: prop_CIB,
    block.RepVGGDW: prop_RepVGGDW,
    # YOLOv9-specific - RepCSP subclasses C3 without overriding forward()
    # (same reasoning as v10Detect/C2fCIB above), but RepBottleneck needs
    # its own prop_* function despite also not overriding forward() - see
    # prop_RepBottleneck's own docstring (its cv1 is shaped differently
    # from plain Bottleneck's).
    block.RepCSP: prop_C3,
    block.RepBottleneck: prop_RepBottleneck,
    block.RepConv: prop_RepConv,
    block.AConv: prop_AConv,
    block.RepNCSPELAN4: prop_RepNCSPELAN4,
    block.ELAN1: prop_RepNCSPELAN4,  # subclasses RepNCSPELAN4, no forward() override
    block.SPPELAN: prop_SPPELAN,
}


def select_prop_rule(mod: torch.nn.Module) -> Optional[PropFunc]:
    """
    Looks up which prop_* function handles a given module, hiding
    PROP_REGISTRY's shape from callers.

    Arguments
    ---------

    mod : torch.nn.Module
        Module instance to find a propagation rule for.

    Returns
    -------

    PropFunc, optional
        The prop_* function registered for `type(mod)`, or None if this
        module type has no registered rule.
    """

    return PROP_REGISTRY.get(type(mod))
