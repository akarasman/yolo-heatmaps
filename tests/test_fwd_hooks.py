from types import SimpleNamespace

import pytest
import torch
from ultralytics.nn.modules import block, conv

from src.yolo.fwd_hooks import (
    FWD_HOOK_REGISTRY,
    Concat_fwd_hook,
    SPPF_fwd_hook,
    select_fwd_hook,
)

# ---------------------------------------------------------------------------
# Concat_fwd_hook
# ---------------------------------------------------------------------------


def test_concat_fwd_hook_caches_per_input_widths_along_the_concat_axis():
    m = SimpleNamespace(d=1)
    t1 = torch.zeros(1, 2, 4, 4)
    t2 = torch.zeros(1, 3, 4, 4)
    out = torch.cat([t1, t2], dim=1)

    Concat_fwd_hook(m, ([t1, t2],), out)

    assert m.in_shapes == [2, 3]
    assert m.out_shape == out.shape


def test_concat_fwd_hook_reads_the_concat_axis_off_the_module():
    # Splitting must follow whatever axis the module concatenates on, not
    # a hardcoded channel dim.
    m = SimpleNamespace(d=0)
    t1 = torch.zeros(2, 3)
    t2 = torch.zeros(5, 3)
    out = torch.cat([t1, t2], dim=0)

    Concat_fwd_hook(m, ([t1, t2],), out)

    assert m.in_shapes == [2, 5]


# ---------------------------------------------------------------------------
# SPPF_fwd_hook
# ---------------------------------------------------------------------------


def test_sppf_fwd_hook_caches_one_index_set_per_pooling_stage():
    n = 3
    m = SimpleNamespace(
        cv1=torch.nn.Identity(),
        m=torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        n=n,
    )
    x = torch.randn(1, 4, 8, 8)

    SPPF_fwd_hook(m, (x,), None)

    assert len(m.indices) == n
    for idx in m.indices:
        # stride=1, padding=1, kernel=3 keeps spatial dims unchanged
        assert idx.shape == x.shape


def test_sppf_fwd_hook_defaults_to_three_stages_when_n_is_unset():
    m = SimpleNamespace(
        cv1=torch.nn.Identity(),
        m=torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
    )
    x = torch.randn(1, 2, 4, 4)

    SPPF_fwd_hook(m, (x,), None)

    assert len(m.indices) == 3


def test_sppf_fwd_hook_chains_stages_on_progressively_pooled_input():
    # Each stage pools the *previous* stage's output, not the original
    # input every time - use a kernel/stride that actually shrinks the
    # map so a chaining bug (re-pooling `x` each time) would show up as
    # constant-shaped indices instead of shrinking ones.
    m = SimpleNamespace(
        cv1=torch.nn.Identity(),
        m=torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        n=2,
    )
    x = torch.randn(1, 1, 8, 8)

    SPPF_fwd_hook(m, (x,), None)

    assert m.indices[0].shape[-2:] == (4, 4)
    assert m.indices[1].shape[-2:] == (2, 2)


# ---------------------------------------------------------------------------
# FWD_HOOK_REGISTRY / select_fwd_hook
# ---------------------------------------------------------------------------


def test_select_fwd_hook_resolves_registered_module_types():
    assert select_fwd_hook(conv.Concat()) is Concat_fwd_hook
    assert select_fwd_hook(block.SPPF(4, 4)) is SPPF_fwd_hook


def test_select_fwd_hook_returns_none_for_unregistered_module_types():
    assert select_fwd_hook(block.Conv(4, 4)) is None


def test_fwd_hook_registry_values_are_the_functions_select_fwd_hook_returns():
    assert set(FWD_HOOK_REGISTRY.values()) == {Concat_fwd_hook, SPPF_fwd_hook}
