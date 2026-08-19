from types import SimpleNamespace

import pytest
import torch
from torch.nn import Conv2d, Linear

from src.lrp.relevance import LayerRelevance, RelevanceMessage
from src.lrp.rules import (
    RULE_REGISTRY,
    ConvRule,
    LinearRule,
    MaxPoolRule,
    PropRule,
    UpsampleRule,
    select_rule,
)

# ---------------------------------------------------------------------------
# PropRule (abstract base / template method)
# ---------------------------------------------------------------------------


def test_prop_rule_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        PropRule()


class _DoublingRule(PropRule):
    """Minimal concrete PropRule, just to exercise propagate()'s shared
    LayerRelevance-vs-plain-tensor handling in isolation from any real
    layer-specific math."""

    def compute(self, module, relevance_in):
        return relevance_in * 2


def test_propagate_on_a_plain_tensor_delegates_to_compute():
    rule = _DoublingRule()
    result = rule.propagate(module=None, relevance=torch.tensor([1.0, 2.0]))
    assert torch.equal(result, torch.tensor([2.0, 4.0]))


def test_propagate_on_layer_relevance_unwraps_computes_and_regathers():
    rule = _DoublingRule()
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=torch.tensor([3.0])))

    result = rule.propagate(module=None, relevance=relevance)

    assert result is relevance
    assert torch.equal(relevance.cache[-1].relevance, torch.tensor([6.0]))


def test_call_is_a_wrapper_for_propagate():
    rule = _DoublingRule()
    result = rule(module=None, relevance=torch.tensor([5.0]))
    assert torch.equal(result, torch.tensor([10.0]))


# ---------------------------------------------------------------------------
# ConvRule
# ---------------------------------------------------------------------------


def _hooked_conv1x1(weight_value: float) -> Conv2d:
    """A 1x1, single in/out channel Conv2d with a known weight and cached
    in_tensor/out_tensor, as the real forward hooks (_conv_nd_fwd_hook)
    would populate."""

    conv = Conv2d(1, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(weight_value)
    x = torch.full((1, 1, 2, 2), 3.0)
    out = conv(x)
    conv.in_tensor = x
    conv.out_tensor = out
    return conv


def test_conv_rule_conserves_relevance_for_a_single_channel_1x1_conv():
    # With one input channel feeding one output channel through a 1x1
    # kernel, every output unit has exactly one contributing input, so
    # LRP-0 (power=1, positive=False) should conserve relevance exactly
    # (up to eps).
    conv = _hooked_conv1x1(weight_value=2.0)
    rule = ConvRule(power=1, eps=1e-9, positive=False, contrastive=False)

    relevance_in = torch.full_like(conv.out_tensor, 6.0)
    relevance_out = rule.compute(conv, relevance_in)

    assert relevance_out.shape == conv.in_tensor.shape
    assert torch.allclose(relevance_out.sum(), relevance_in.sum(), atol=1e-3)


def test_conv_rule_propagate_wraps_layer_relevance():
    conv = _hooked_conv1x1(weight_value=2.0)
    rule = ConvRule(power=1, eps=1e-9, positive=False, contrastive=False)

    relevance = LayerRelevance()
    relevance.gather(
        RelevanceMessage(
            from_=None, to=-1, relevance=torch.full_like(conv.out_tensor, 6.0)
        )
    )

    result = rule.propagate(conv, relevance)

    assert result is relevance
    assert relevance.cache[-1].relevance.shape == conv.in_tensor.shape


# ---------------------------------------------------------------------------
# LinearRule
# ---------------------------------------------------------------------------


def _hooked_linear(weight: torch.Tensor) -> Linear:
    """A Linear layer with known weights and a cached in_tensor, as the
    real forward hook (_linear_fwd_hook) would populate."""

    linear = Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        linear.weight.copy_(weight)
    x = torch.tensor([[1.0, 2.0]])
    linear.in_tensor = x
    return linear


def test_linear_rule_conserves_relevance_for_identity_weights():
    # Identity weights mean each output unit has exactly one contributing
    # input, same conservation argument as the 1x1 conv case above.
    linear = _hooked_linear(torch.eye(2))
    rule = LinearRule(power=1, eps=1e-9, positive=False, contrastive=False)

    relevance_in = torch.tensor([[3.0, 4.0]])
    relevance_out = rule.compute(linear, relevance_in.clone())

    assert relevance_out.shape == linear.in_tensor.shape
    assert torch.allclose(relevance_out.sum(), relevance_in.sum(), atol=1e-3)


# ---------------------------------------------------------------------------
# MaxPoolRule
# ---------------------------------------------------------------------------


def test_maxpool_rule_default_passes_relevance_through_unchanged():
    # No learned weights to redistribute through, unlike conv/linear - the
    # default behavior is a reshape, not a redistribution.
    module = SimpleNamespace(out_shape=(1, 1, 2, 2))
    rule = MaxPoolRule()

    relevance_in = torch.arange(4.0).reshape(1, 1, 2, 2)
    result = rule.compute(module, relevance_in)

    assert torch.equal(result, relevance_in)


def test_maxpool_rule_winner_takes_all_routes_relevance_to_the_argmax_and_conserves_it():
    x = torch.tensor([[[[1.0, 5.0], [3.0, 2.0]]]])  # max is 5.0 at flat index 1
    out, indices = torch.nn.functional.max_pool2d(x, kernel_size=2, return_indices=True)
    module = SimpleNamespace(out_shape=out.shape, in_shape=x.shape, indices=indices)
    rule = MaxPoolRule(max=True)

    relevance_in = torch.full_like(out, 7.0)
    result = rule.compute(module, relevance_in)

    assert result.shape == x.shape
    assert torch.allclose(result.sum(), relevance_in.sum())
    flat = result.flatten()
    assert flat[1].item() == pytest.approx(7.0)
    assert flat[[0, 2, 3]].sum().item() == pytest.approx(0.0)


def test_maxpool_rule_winner_takes_all_indexes_each_channel_independently():
    # Real spatial downsampling (4x4 -> 2x2, unlike the single-output
    # degenerate case above) across 2 channels - catches indexing that
    # accidentally uses the *output* plane size as each channel's stride
    # into the (larger) input-shaped result instead of the input plane
    # size, which the single-channel/single-output case above can't (its
    # input and output plane sizes coincide at 1, and there's only one
    # channel to misalign against).
    torch.manual_seed(0)
    x = torch.rand(1, 2, 4, 4)
    out, indices = torch.nn.functional.max_pool2d(
        x, kernel_size=2, stride=2, return_indices=True
    )
    module = SimpleNamespace(out_shape=out.shape, in_shape=x.shape, indices=indices)
    rule = MaxPoolRule(max=True)

    relevance_in = torch.rand(1, 2, 2, 2)
    result = rule.compute(module, relevance_in)

    assert result.shape == x.shape
    assert torch.allclose(result.sum(), relevance_in.sum())
    # Every unit of relevance should land exactly on its channel's own
    # argmax positions - nothing bleeds into the other channel.
    for c in range(2):
        expected = torch.zeros(4, 4).flatten()
        expected.scatter_add_(0, indices[0, c].flatten(), relevance_in[0, c].flatten())
        assert torch.allclose(result[0, c].flatten(), expected)


# ---------------------------------------------------------------------------
# UpsampleRule
# ---------------------------------------------------------------------------


def test_upsample_rule_conserves_relevance_via_the_average_pooling_inverse():
    module = SimpleNamespace(
        in_dim=4, mode="nearest", out_shape=(1, 1, 4, 4), scale_factor=2.0
    )
    rule = UpsampleRule()

    relevance_in = torch.ones(1, 1, 4, 4)
    result = rule.compute(module, relevance_in)

    assert result.shape == (1, 1, 2, 2)
    assert torch.allclose(result.sum(), relevance_in.sum())


def test_upsample_rule_rejects_non_nearest_mode():
    module = SimpleNamespace(
        in_dim=4, mode="bilinear", out_shape=(1, 1, 4, 4), scale_factor=2.0
    )
    rule = UpsampleRule()

    with pytest.raises(NotImplementedError):
        rule.compute(module, torch.ones(1, 1, 4, 4))


# ---------------------------------------------------------------------------
# RULE_REGISTRY / select_rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layer, expected_cls",
    [
        (torch.nn.Conv1d(1, 1, 1), ConvRule),
        (torch.nn.Conv2d(1, 1, 1), ConvRule),
        (torch.nn.Conv3d(1, 1, 1), ConvRule),
        (torch.nn.Linear(1, 1), LinearRule),
        (torch.nn.MaxPool1d(1), MaxPoolRule),
        (torch.nn.MaxPool2d(1), MaxPoolRule),
        (torch.nn.MaxPool3d(1), MaxPoolRule),
        (torch.nn.Upsample(scale_factor=2), UpsampleRule),
    ],
)
def test_select_rule_resolves_registered_layer_types(layer, expected_cls):
    assert select_rule(layer) is expected_cls


def test_select_rule_returns_none_for_unregistered_layer_types():
    assert select_rule(torch.nn.ReLU()) is None


def test_rule_registry_is_keyed_by_exact_type_not_subclass():
    # Dispatch is by exact type() everywhere in this package (e.g. DWConv
    # needs its own inv_func entry despite subclassing Conv) - this pins
    # that RULE_REGISTRY follows the same rule, not isinstance-style
    # subclass matching.
    class SubConv2d(torch.nn.Conv2d):
        pass

    assert select_rule(SubConv2d(1, 1, 1)) is None


def test_rule_registry_values_are_the_classes_select_rule_returns():
    assert set(RULE_REGISTRY.values()) == {
        ConvRule,
        LinearRule,
        MaxPoolRule,
        UpsampleRule,
    }
