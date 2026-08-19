from types import SimpleNamespace

import pytest
import torch
from ultralytics.nn.modules import block, conv, head

from src.lrp.fwd_hooks import _conv_nd_fwd_hook
from src.lrp.inverter import Inverter
from src.lrp.relevance import LayerRelevance, RelevanceMessage, scale_key
from src.lrp.rules import ConvRule
from src.lrp.utils import get_dummy_summation_conv_layer
from src.yolo.block_rules import (
    PROP_REGISTRY,
    _bilinear_relevance,
    _softmax_relevance,
    prop_Attention,
    prop_Bottleneck,
    prop_C2f,
    prop_C2PSA,
    prop_C3,
    prop_Concat,
    prop_Conv,
    prop_Detect,
    prop_DFL,
    prop_PSABlock,
    prop_SPPF,
    select_prop_rule,
)


class _IdentityInverter:
    """Test double standing in for Inverter: passes relevance through a
    sub-module unchanged. Isolates a prop_* function's own channel-
    routing/slicing arithmetic from real LRP math (already covered by
    test_rules.py's ConvRule/LinearRule tests) - same idea as
    test_rules.py's `_DoublingRule`."""

    def __init__(self, prop_to=None):
        self.prop_to = prop_to if prop_to is not None else []

    def __call__(self, mod, relevance):
        return relevance


# ---------------------------------------------------------------------------
# PROP_REGISTRY / select_prop_rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instance, expected_func",
    [
        (block.C3(4, 4), prop_C3),
        (block.C3k(4, 4), prop_C3),
        (block.C2f(4, 4), prop_C2f),
        (block.C3k2(4, 4), prop_C2f),
        (block.C2PSA(32, 32), prop_C2PSA),
        (block.PSABlock(32), prop_PSABlock),
        (block.Attention(32), prop_Attention),
        (block.Conv(4, 4), prop_Conv),
        (conv.DWConv(4, 4), prop_Conv),
        (head.Detect(nc=1, ch=(4,)), prop_Detect),
        (block.Bottleneck(4, 4), prop_Bottleneck),
        (conv.Concat(), prop_Concat),
        (block.SPPF(4, 4), prop_SPPF),
        (block.DFL(4), prop_DFL),
    ],
)
def test_select_prop_rule_resolves_registered_module_types(instance, expected_func):
    assert select_prop_rule(instance) is expected_func


def test_select_prop_rule_returns_none_for_unregistered_module_types():
    assert select_prop_rule(torch.nn.ReLU()) is None


def test_prop_registry_is_keyed_by_exact_type_not_subclass():
    # Dispatch is by exact type() everywhere in this package - pin that
    # PROP_REGISTRY follows the same rule (C3k/C3k2 get their own entries
    # precisely because subclass-matching isn't used).
    class SubC3(block.C3):
        pass

    assert select_prop_rule(SubC3(4, 4)) is None


# ---------------------------------------------------------------------------
# prop_C3 / prop_C2f / prop_C2PSA / prop_Detect / prop_Conv
# (channel-routing arithmetic, via _IdentityInverter)
# ---------------------------------------------------------------------------


def test_prop_c3_sums_the_cv1_and_cv2_channel_halves():
    seed = torch.arange(8.0).reshape(1, 8, 1, 1)
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=seed))
    mod = SimpleNamespace(
        cv1=None, cv2=None, cv3=None, m=[object(), object()], reg_num=3
    )

    result = prop_C3(_IdentityInverter(), mod, relevance)

    expected = seed[:, :4, ...] + seed[:, 4:, ...]
    assert result is relevance
    assert torch.equal(relevance.cache[-1].relevance, expected)
    assert relevance.cache[-1].from_ == 3


def test_prop_c2f_accumulates_bottleneck_chunks_in_reverse():
    # 4 chunks of width mod.c=2, distinct per-chunk values so an
    # off-by-one in the reverse-accumulation indexing would show up as a
    # different number rather than silently matching by symmetry.
    chunks = [torch.full((1, 2, 1, 1), v) for v in (1.0, 2.0, 4.0, 8.0)]
    seed = torch.cat(chunks, dim=1)
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=seed))
    mod = SimpleNamespace(cv1=None, cv2=None, m=[object(), object()], c=2, reg_num=7)

    result = prop_C2f(_IdentityInverter(), mod, relevance)

    # Hand-traced (and cross-checked by direct execution): chunk[2] +=
    # chunk[3], then chunk[1] += (updated) chunk[2] -> [1, 14, 12, 8],
    # cv1 sees only the first two chunks concatenated.
    expected = torch.cat([torch.full((1, 2, 1, 1), v) for v in (1.0, 14.0)], dim=1)
    assert torch.equal(relevance.cache[-1].relevance, expected)


def test_prop_c2psa_recombines_untouched_and_attention_halves():
    seed = torch.arange(6.0).reshape(1, 6, 1, 1)
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=seed))
    mod = SimpleNamespace(cv1=None, cv2=None, m=None, c=2, reg_num=9)

    result = prop_C2PSA(_IdentityInverter(), mod, relevance)

    # Splitting mod.c channels off then concatenating straight back
    # together, with every step a no-op under an identity inverter,
    # should reproduce the original seed exactly.
    assert torch.equal(relevance.cache[-1].relevance, seed)


def test_prop_detect_routes_each_scale_to_its_prop_to_target():
    scale0 = torch.full((1, 3, 1, 1), 1.0)
    scale1 = torch.full((1, 3, 1, 1), 2.0)
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=scale_key(0), relevance=scale0))
    relevance.gather(RelevanceMessage(from_=None, to=scale_key(1), relevance=scale1))
    mod = SimpleNamespace(cv3=[None, None], reg_num=11)

    result = prop_Detect(_IdentityInverter(prop_to=[7, 4]), mod, relevance)

    assert torch.equal(result.cache[7].relevance, scale0)
    assert torch.equal(result.cache[4].relevance, scale1)


def test_prop_conv_delegates_straight_to_the_inner_conv():
    relevance_in = torch.ones(1, 2, 2, 2)
    mod = SimpleNamespace(conv=object())

    result = prop_Conv(_IdentityInverter(), mod, relevance_in)

    assert result is relevance_in


# ---------------------------------------------------------------------------
# prop_Concat (real slicing, no inverter involved)
# ---------------------------------------------------------------------------


def test_prop_concat_splits_relevance_back_into_source_layer_slices():
    seed = torch.arange(5.0).reshape(1, 5, 1, 1)
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=seed))
    mod = SimpleNamespace(in_shapes=[2, 3], d=1, f=[10, 20], reg_num=2)

    result = prop_Concat(inverter=None, mod=mod, relevance=relevance)

    assert torch.equal(result.cache[10].relevance, seed[:, :2, ...])
    assert torch.equal(result.cache[20].relevance, seed[:, 2:, ...])
    assert -1 not in result.cache


# ---------------------------------------------------------------------------
# prop_Bottleneck / prop_SPPF (real Inverter + ConvRule, via real
# ultralytics modules - these two do enough of their own tensor surgery
# (the dummy-summation-conv residual split) to be worth checking for
# actual LRP conservation, not just shape/routing).
# ---------------------------------------------------------------------------


def _make_inverter() -> Inverter:
    conv_rule = ConvRule(power=1, eps=1e-9, positive=True, contrastive=False)
    inverter = Inverter(conv_rule=conv_rule)
    inverter.register_inv_func(block.Conv, prop_Conv)
    inverter.register_inv_func(block.Attention, prop_Attention)
    return inverter


def test_prop_bottleneck_conserves_relevance_for_a_real_module():
    torch.manual_seed(0)
    c = 4
    mod = block.Bottleneck(c, c, shortcut=True, g=1, k=(1, 1), e=1.0)
    mod.eval()
    x = torch.rand(1, c, 5, 5) + 0.1  # keep inputs positive: ConvRule clamps negatives

    with torch.no_grad():
        h = mod.cv1(x)
        y = mod.cv2(h)
        out = x + y if mod.add else y

    _conv_nd_fwd_hook(mod.cv1.conv, [x], h)
    _conv_nd_fwd_hook(mod.cv2.conv, [h], y)

    inverter = _make_inverter()
    result = prop_Bottleneck(inverter, mod, out.clone())

    assert result.shape == x.shape
    assert torch.allclose(result.sum(), out.sum(), rtol=0.05)


def test_prop_sppf_conserves_relevance_without_residual():
    torch.manual_seed(1)
    c = 4
    mod = block.SPPF(c, c, k=3, n=2)
    mod.eval()
    mod.add = False
    x = torch.rand(1, c, 6, 6) + 0.1

    with torch.no_grad():
        out = mod(x)

    _conv_nd_fwd_hook(mod.cv1.conv, [x], mod.cv1(x))
    y1 = mod.m(mod.cv1(x))
    y2 = mod.m(y1)
    cv2_in = torch.cat([mod.cv1(x), y1, y2], dim=1)
    _conv_nd_fwd_hook(mod.cv2.conv, [cv2_in], out)

    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=out.clone()))
    inverter = _make_inverter()

    result = prop_SPPF(inverter, mod, relevance)

    assert result.cache[-1].relevance.shape == x.shape
    assert torch.allclose(result.cache[-1].relevance.sum(), out.sum(), rtol=0.05)


def test_prop_sppf_conserves_relevance_with_residual():
    torch.manual_seed(2)
    c = 4
    mod = block.SPPF(c, c, k=3, n=2)
    mod.eval()
    mod.add = True
    x = torch.rand(1, c, 6, 6) + 0.1

    cv1_out = mod.cv1(x)
    y1 = mod.m(cv1_out)
    y2 = mod.m(y1)
    cv2_in = torch.cat([cv1_out, y1, y2], dim=1)
    cv2_out = mod.cv2(cv2_in)
    out = x + cv2_out

    _conv_nd_fwd_hook(mod.cv1.conv, [x], cv1_out)
    _conv_nd_fwd_hook(mod.cv2.conv, [cv2_in], cv2_out)

    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=out.clone()))
    inverter = _make_inverter()

    result = prop_SPPF(inverter, mod, relevance)

    assert result.cache[-1].relevance.shape == x.shape
    assert torch.allclose(result.cache[-1].relevance.sum(), out.sum(), rtol=0.05)


# ---------------------------------------------------------------------------
# _bilinear_relevance / _softmax_relevance (the AttnLRP-style primitives
# prop_Attention is built from) - verified against independent autograd
# computations of the same mathematical definitions, not just re-checking
# the closed-form matmul implementation against itself.
# ---------------------------------------------------------------------------


def test_bilinear_relevance_matches_an_autograd_cross_check():
    torch.manual_seed(10)
    P = torch.rand(2, 3, 4, requires_grad=True) + 0.1
    Q = torch.rand(2, 4, 5, requires_grad=True) + 0.1
    relevance_z = torch.randn(2, 3, 5)

    R_P, R_Q = _bilinear_relevance(P.detach(), Q.detach(), relevance_z)

    # Independent check: R_P should equal P * d(L)/dP where L = sum(z * s)
    # for s = (relevance_z / z).detach() treated as a fixed coefficient -
    # i.e. the same "gradient x input" identity _bilinear_relevance is
    # derived from, computed here via autograd instead of the closed-form
    # matmul expressions in the function itself.
    z = P @ Q
    s = (relevance_z / z).detach()
    L = (z * s).sum()
    grad_P, grad_Q = torch.autograd.grad(L, [P, Q])

    assert torch.allclose(R_P, P.detach() * grad_P, atol=1e-4)
    assert torch.allclose(R_Q, Q.detach() * grad_Q, atol=1e-4)


def test_softmax_relevance_matches_an_autograd_cross_check():
    torch.manual_seed(11)
    scores = torch.randn(2, 3, 5, requires_grad=True)
    relevance_attn = torch.randn(2, 3, 5)

    attn_weights = scores.softmax(dim=-1)
    R_scores = _softmax_relevance(
        scores.detach(), attn_weights.detach(), relevance_attn
    )

    coeff = (relevance_attn / attn_weights).detach()
    L = (scores.softmax(dim=-1) * coeff).sum()
    (grad_scores,) = torch.autograd.grad(L, [scores])

    assert torch.allclose(R_scores, scores.detach() * grad_scores, atol=1e-4)


def test_softmax_relevance_sums_to_zero_along_the_axis_when_scores_are_uniform():
    # When every score along the softmax axis is equal, softmax can't
    # discriminate between positions - the rule should reflect that by
    # not manufacturing any net relevance along that axis.
    scores = torch.full((2, 3, 4), 0.7)
    attn_weights = scores.softmax(dim=-1)
    relevance_attn = torch.randn(2, 3, 4)

    R_scores = _softmax_relevance(scores, attn_weights, relevance_attn)

    assert torch.allclose(R_scores.sum(dim=-1), torch.zeros(2, 3), atol=1e-5)


# ---------------------------------------------------------------------------
# prop_Attention / prop_PSABlock (real ultralytics modules, real forward
# pass, real Conv2d/BatchNorm2d forward hooks - these involve enough novel
# tensor surgery, on top of the primitives above, to be worth checking
# end-to-end against a real module rather than only unit-testing the math
# helpers in isolation).
# ---------------------------------------------------------------------------


def _attach_conv_bn_hooks(module: torch.nn.Module) -> None:
    """Registers the same forward hook real Conv2d/BatchNorm2d layers get
    from YOLOLRP._attach_forward_hooks (see DEFAULT_FWD_HOOKS in
    fwd_hooks.py), without needing a full YOLOLRP/model wrapper - enough
    for prop_Attention/prop_PSABlock, which only ever read
    .conv.in_tensor/.conv.out_tensor/.bn.out_tensor."""

    for sub in module.modules():
        if isinstance(sub, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
            sub.register_forward_hook(_conv_nd_fwd_hook)


def test_prop_attention_runs_end_to_end_for_a_real_module():
    torch.manual_seed(20)
    dim = 32
    mod = block.Attention(dim, num_heads=4, attn_ratio=0.5)
    mod.eval()
    _attach_conv_bn_hooks(mod)

    x = torch.rand(1, dim, 4, 4) + 0.1
    with torch.no_grad():
        out = mod(x)

    inverter = _make_inverter()
    result = prop_Attention(inverter, mod, out.clone())

    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    assert result.abs().sum() > 0


@pytest.mark.parametrize("shortcut", [True, False])
def test_prop_psablock_runs_end_to_end_for_a_real_module(shortcut):
    torch.manual_seed(21)
    c = 32
    mod = block.PSABlock(c, attn_ratio=0.5, num_heads=4, shortcut=shortcut)
    mod.eval()
    _attach_conv_bn_hooks(mod)

    x = torch.rand(1, c, 4, 4) + 0.1
    with torch.no_grad():
        out = mod(x)

    inverter = _make_inverter()
    result = prop_PSABlock(inverter, mod, out.clone())

    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    assert result.abs().sum() > 0
