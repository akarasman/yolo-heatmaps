"""
Integration test against real, downloaded checkpoints for every YOLO
version this library supports (YOLOv8, YOLOv9, YOLO26, YOLOv10, YOLO11 -
see block_rules.PROP_REGISTRY's own YOLOv10-specific/YOLOv9-specific
sections) - the only file in this suite that needs real `ultralytics`
plus network access (to fetch each checkpoint on first run, ~5MB each,
cached in the repo root afterwards - see .gitignore's `*.pt`) rather than
the stubbed/synthetic modules every other test file uses. Skipped
automatically if either isn't available, so the rest of the suite (unit
tests only, see pyproject.toml) keeps working without them.

Every test below runs once per checkpoint (parametrized via `model`),
rather than being duplicated per version - the same explain()/registry
behavior is expected to hold across all of them.

Marked `integration` (registered in pyproject.toml) - deselect with
`pytest -m "not integration"` if you want the fast, network-free subset.
"""

import pytest
import torch

ultralytics = pytest.importorskip("ultralytics", reason="requires real ultralytics")

from yolo_lrp.lrp.inverter import IDENTITY_MAPPINGS
from yolo_lrp.lrp.rules import RULE_REGISTRY
from yolo_lrp.yolo.block_rules import PROP_REGISTRY
from yolo_lrp.yolo.explainer import YOLOLRP

pytestmark = pytest.mark.integration

# One checkpoint per supported YOLO version - see block_rules.py's
# "YOLOv10-specific"/"YOLOv9-specific" registry sections and this
# session's own port notes: YOLO26 is the primary target; YOLOv8 and
# YOLO11 needed zero new code (their block vocabularies are strict
# subsets of what's already registered); YOLOv10 and YOLOv9 each needed
# real new prop_* functions (SCDown/PSA/CIB/RepVGGDW/v10Detect for v10;
# AConv/RepNCSPELAN4/SPPELAN/RepCSP/RepBottleneck/RepConv for v9).
SUPPORTED_WEIGHTS = [
    "yolov8n.pt",
    "yolov9t.pt",
    "yolo26n.pt",
    "yolov10n.pt",
    "yolo11n.pt",
]


@pytest.fixture(scope="module", params=SUPPORTED_WEIGHTS)
def model(request):
    weights = request.param
    try:
        return ultralytics.YOLO(weights)
    except Exception as e:
        pytest.skip(f"could not load/download {weights}: {e}")


@pytest.fixture(scope="module")
def explainer(model):
    return YOLOLRP(model)


@pytest.fixture
def frame():
    torch.manual_seed(0)
    return torch.rand(3, 640, 640)


# ---------------------------------------------------------------------------
# explain() against the real model, across every mode it supports
# ---------------------------------------------------------------------------


def test_explain_produces_a_finite_2d_heatmap(explainer, frame):
    heatmap = explainer.explain(frame)

    assert heatmap.shape == (640, 640)
    assert torch.isfinite(heatmap).all()
    assert heatmap.sum() > 0


def test_explain_is_finite_for_max_class_only_false(explainer, frame):
    heatmap = explainer.explain(frame, max_class_only=False)

    assert heatmap.shape == (640, 640)
    assert torch.isfinite(heatmap).all()


def test_explain_is_finite_for_a_specific_class_by_index(explainer, frame):
    heatmap = explainer.explain(frame, cls=1)

    assert heatmap.shape == (640, 640)
    assert torch.isfinite(heatmap).all()


def test_explain_is_finite_for_a_specific_class_by_name(explainer, frame):
    name = next(iter(explainer.model.names.values()))

    heatmap = explainer.explain(frame, cls=name)

    assert heatmap.shape == (640, 640)
    assert torch.isfinite(heatmap).all()


def test_explain_contrastive_produces_a_finite_2d_heatmap(explainer, frame):
    # The path most likely to break: contrastive doubles relevance's
    # batch dimension everywhere, which is where the ConvRule
    # double-doubling bug and prop_Attention's hardcoded-batch .view()
    # bugs were both actually found this session.
    heatmap = explainer.explain(frame, cls=0, contrastive=True)

    assert heatmap.shape == (640, 640)
    assert torch.isfinite(heatmap).all()


def test_explain_can_be_called_repeatedly_on_the_same_explainer(explainer, frame):
    # explain() removes and re-attaches its own forward hooks every call
    # (see _remove_hooks/_attach_forward_hooks) - this pins that calling
    # it twice in a row doesn't leave stale hooks or cached state behind.
    first = explainer.explain(frame)
    second = explainer.explain(frame)

    assert torch.equal(first, second)


def test_to_moves_the_explainer_and_explain_still_works(explainer, frame):
    explainer.to("cpu")
    heatmap = explainer.explain(frame)

    assert torch.isfinite(heatmap).all()


def test_get_layer_relevance_returns_a_snapshot_per_layer_after_explain(
    explainer, frame
):
    explainer.explain(frame)

    r_values = explainer.get_layer_relevance()

    assert r_values is not None
    assert len(r_values) == len(explainer.module_list) + 1


# ---------------------------------------------------------------------------
# Registry coverage against every module type actually present in the
# real model - formalizes the check first done ad hoc in
# explore_yolo26.ipynb as a real regression test.
# ---------------------------------------------------------------------------


def test_registries_cover_every_module_type_in_the_real_model(explainer):
    # Container/wrapper types that are never passed to Inverter.invert()
    # directly: DetectionModel (the top-level model wrapper, never
    # walked itself), Sequential (handled by a dedicated isinstance
    # branch in Inverter.invert(), not a registry entry), ModuleList
    # (C3k2/C2f's `self.m` - iterated element-by-element by prop_C2f,
    # never passed as a unit).
    expected_uncovered = {
        type(explainer.model.model),
        torch.nn.Sequential,
        torch.nn.ModuleList,
    }

    seen_types = {type(m) for m in explainer.model.model.modules()}
    covered = set(PROP_REGISTRY) | set(RULE_REGISTRY) | set(IDENTITY_MAPPINGS)

    uncovered = {t for t in seen_types if t not in covered} - expected_uncovered

    assert uncovered == set(), f"module types with no registered rule: {uncovered}"
