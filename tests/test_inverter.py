import logging

import pytest
import torch
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

from src.lrp.inverter import Inverter
from src.lrp.rules import ConvRule, LinearRule

# ---------------------------------------------------------------------------
# __init__ / rules_by_layer_type
# ---------------------------------------------------------------------------


def test_supplied_conv_and_linear_rules_are_shared_across_their_dims():
    conv_rule = ConvRule(contrastive=False)
    linear_rule = LinearRule(contrastive=False)
    inverter = Inverter(conv_rule=conv_rule, linear_rule=linear_rule)

    assert inverter.rules_by_layer_type[Conv1d] is conv_rule
    assert inverter.rules_by_layer_type[Conv2d] is conv_rule
    assert inverter.rules_by_layer_type[Conv3d] is conv_rule
    assert inverter.rules_by_layer_type[Linear] is linear_rule


def test_omitted_conv_and_linear_rules_leave_their_types_out_of_the_table():
    inverter = Inverter()

    assert Conv1d not in inverter.rules_by_layer_type
    assert Conv2d not in inverter.rules_by_layer_type
    assert Conv3d not in inverter.rules_by_layer_type
    assert Linear not in inverter.rules_by_layer_type


def test_maxpool_and_upsample_get_default_instances_even_when_omitted():
    inverter = Inverter()

    assert MaxPool2d in inverter.rules_by_layer_type
    assert Upsample in inverter.rules_by_layer_type
    # One shared instance per rule class, not one per torch type.
    assert (
        inverter.rules_by_layer_type[MaxPool1d]
        is inverter.rules_by_layer_type[MaxPool2d]
        is inverter.rules_by_layer_type[MaxPool3d]
    )


# ---------------------------------------------------------------------------
# register_inv_func
# ---------------------------------------------------------------------------


class _FakeBlock(torch.nn.Module):
    pass


def test_register_inv_func_stores_by_module_type():
    inverter = Inverter()
    inv_func = lambda inv, layer, relevance: relevance

    inverter.register_inv_func(_FakeBlock, inv_func)

    assert inverter.inv_funcs[_FakeBlock] is inv_func


def test_register_inv_func_warns_on_replacing_an_existing_registration(caplog):
    inverter = Inverter()
    inverter.register_inv_func(_FakeBlock, lambda inv, layer, relevance: relevance)
    with caplog.at_level(logging.WARNING):
        inverter.register_inv_func(_FakeBlock, lambda inv, layer, relevance: relevance)

    assert "Replacing previous inverse" in caplog.text


# ---------------------------------------------------------------------------
# invert() dispatch
# ---------------------------------------------------------------------------


def _hooked_conv1x1(weight_value: float) -> Conv2d:
    """A 1x1, single in/out channel Conv2d with a known weight and cached
    in_tensor/out_tensor, as the real forward hook (_conv_nd_fwd_hook)
    would populate. Mirrors test_rules.py's helper of the same shape."""

    conv = Conv2d(1, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(weight_value)
    x = torch.full((1, 1, 2, 2), 3.0)
    conv.in_tensor = x
    conv.out_tensor = conv(x)
    return conv


def test_invert_dispatches_to_the_primitive_rule_table():
    conv = _hooked_conv1x1(weight_value=2.0)
    inverter = Inverter(
        conv_rule=ConvRule(power=1, eps=1e-9, positive=False, contrastive=False)
    )

    relevance_in = torch.full_like(conv.out_tensor, 6.0)
    relevance_out = inverter.invert(conv, relevance_in)

    assert relevance_out.shape == conv.in_tensor.shape
    assert torch.allclose(relevance_out.sum(), relevance_in.sum(), atol=1e-3)


def test_invert_dispatches_to_a_registered_inv_func_for_composite_blocks():
    inverter = Inverter()
    block = _FakeBlock()

    def inv_func(inv, layer, relevance, **kwargs):
        assert inv is inverter
        assert layer is block
        return relevance * 2

    inverter.register_inv_func(_FakeBlock, inv_func)

    result = inverter.invert(block, torch.tensor([3.0]))

    assert torch.equal(result, torch.tensor([6.0]))


def test_invert_recurses_through_sequential_in_reverse_order():
    inverter = Inverter()
    call_order = []

    class _First(torch.nn.Module):
        pass

    class _Second(torch.nn.Module):
        pass

    inverter.register_inv_func(
        _First,
        lambda inv, layer, relevance: call_order.append("first") or relevance,
    )
    inverter.register_inv_func(
        _Second,
        lambda inv, layer, relevance: call_order.append("second") or relevance,
    )

    seq = torch.nn.Sequential(_First(), _Second())
    inverter.invert(seq, torch.tensor([1.0]))

    # Sequential's forward runs First then Second - inversion must undo
    # that in reverse, so Second is inverted before First.
    assert call_order == ["second", "first"]


def test_invert_returns_relevance_unchanged_for_identity_mapped_types():
    inverter = Inverter()
    relevance_in = torch.tensor([1.0, -2.0, 3.0])

    result = inverter.invert(torch.nn.ReLU(), relevance_in)

    assert result is relevance_in


class _UnregisteredLayer(torch.nn.Module):
    pass


def test_invert_raises_for_an_unregistered_type_by_default():
    inverter = Inverter(pass_not_implemented=False)

    with pytest.raises(NotImplementedError):
        inverter.invert(_UnregisteredLayer(), torch.tensor([1.0]))


def test_invert_passes_through_for_an_unregistered_type_when_configured_to():
    inverter = Inverter(pass_not_implemented=True)
    relevance_in = torch.tensor([1.0])

    result = inverter.invert(_UnregisteredLayer(), relevance_in)

    assert result is relevance_in


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------


def test_call_is_a_wrapper_for_invert():
    inverter = Inverter()
    block = _FakeBlock()
    inverter.register_inv_func(_FakeBlock, lambda inv, layer, relevance: relevance * 3)

    result = inverter(block, torch.tensor([2.0]))

    assert torch.equal(result, torch.tensor([6.0]))
