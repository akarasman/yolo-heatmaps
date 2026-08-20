import torch

from yolo_lrp.lrp.relevance import scale_key
from yolo_lrp.yolo.explainer import (
    YOLOLRP,
    _initialize_relevance,
    _localize_to_bbox,
    _second_best_class,
)

# ---------------------------------------------------------------------------
# _suppress_outliers
# ---------------------------------------------------------------------------


def test_suppress_outliers_clips_extreme_values_instead_of_deleting_them():
    # 100 values 0..99 - the 98th percentile sits at 98, so 98 and 99 are
    # the "outliers" this is meant to tame.
    relevance = torch.arange(100, dtype=torch.float32)

    result = YOLOLRP._suppress_outliers(relevance, quantile=0.98)

    # Clipped down to the threshold, not deleted to zero - the whole
    # point: an explanation shouldn't erase its own strongest signal.
    assert result[98].item() == result.max().item()
    assert result[99].item() == result.max().item()
    assert result[99].item() != 0.0


def test_suppress_outliers_leaves_values_below_the_threshold_unchanged():
    relevance = torch.arange(100, dtype=torch.float32)

    result = YOLOLRP._suppress_outliers(relevance, quantile=0.98)

    assert torch.equal(result[:98], relevance[:98])


def test_suppress_outliers_handles_negative_values():
    # Real relevance can be negative (any model with an attention block -
    # see save_heatmap's own docstring) - suppression must not assume
    # non-negative input.
    relevance = torch.tensor([-10.0, -1.0, 0.0, 1.0, 100.0])

    result = YOLOLRP._suppress_outliers(relevance, quantile=0.6)

    assert result.min().item() == -10.0
    assert result.max().item() < 100.0


# ---------------------------------------------------------------------------
# _initialize_relevance's max_class_only noise-floor guard
# ---------------------------------------------------------------------------


def test_initialize_relevance_max_class_only_drops_noise_floor_winners():
    # Two grid locations, three classes, one detection scale. Location 0
    # is a real, confident detection (class 0 at 0.9). Location 1 is
    # background with no real object - every class near float-zero, with
    # class 1 "winning" only by a meaningless margin (2e-7 over 1e-7) -
    # the exact pattern a real explain() call produces for empty regions
    # (confirmed against a real image: every class under 1e-6 at 70% of
    # one scale's grid cells, with the nominal "winner" varying by pure
    # noise). Without a floor, max_class_only would keep that noise-floor
    # "winner" and seed relevance for whichever class it happened to be -
    # here, class 1 - from a location with no real signal at all.
    cls_pred = torch.tensor([[[[0.9, 1e-7]], [[0.05, 2e-7]], [[0.01, 1.5e-7]]]])

    relevance = _initialize_relevance(
        [cls_pred], cls=None, max_class_only=True, contrastive=False
    )
    seeded = relevance.scatter(which=scale_key(0))

    # Real detection survives unchanged.
    assert seeded[0, 0, 0, 0].item() == cls_pred[0, 0, 0, 0].item()
    # Noise-floor "winner" is dropped - the whole location is zeroed,
    # not attributed to class 1.
    assert seeded[0, :, 0, 1].sum().item() == 0.0


def test_initialize_relevance_max_class_only_keeps_a_real_low_confidence_winner():
    # A winner just above the floor should still survive - the guard is a
    # floor on meaningless noise, not a general high-confidence-only filter.
    cls_pred = torch.tensor([[[[0.02]], [[0.001]]]])  # shape (1, 2, 1, 1)

    relevance = _initialize_relevance(
        [cls_pred], cls=None, max_class_only=True, contrastive=False
    )
    seeded = relevance.scatter(which=scale_key(0))

    assert seeded[0, 0, 0, 0].item() == cls_pred[0, 0, 0, 0].item()


# ---------------------------------------------------------------------------
# _second_best_class (contrastive dual relevance, eq 10)
# ---------------------------------------------------------------------------


def test_second_best_class_excludes_only_the_named_class():
    # class 0 is the global winner (0.9); the real runner-up is class 1
    # (0.3), not class 2 (0.1).
    cls_pred = torch.tensor([[[[0.9]], [[0.3]], [[0.1]]]])  # (1, 3, 1, 1)

    dual = _second_best_class(cls_pred, cls=0)

    assert dual[0, 1, 0, 0].item() == cls_pred[0, 1, 0, 0].item()
    assert dual[0, 0, 0, 0].item() == 0.0
    assert dual[0, 2, 0, 0].item() == 0.0


def test_second_best_class_surfaces_the_runner_up_even_when_cls_is_the_winner():
    # The bug this exists to fix: if dual were derived from the already
    # max_class_only-collapsed primal tensor (which keeps only class 0,
    # zeroing everything else), excluding cls=0 from that would leave
    # nothing. Computed independently from the raw per-class field
    # instead, the real second-best (class 1, 0.3) survives.
    cls_pred = torch.tensor([[[[0.9]], [[0.3]], [[0.1]]]])  # (1, 3, 1, 1)

    dual = _second_best_class(cls_pred, cls=0)

    assert dual.sum().item() > 0.0
    assert dual[0, 1, 0, 0].item() == 0.30000001192092896 or abs(
        dual[0, 1, 0, 0].item() - 0.3
    ) < 1e-6


def test_initialize_relevance_contrastive_keeps_the_runner_up_at_the_winning_location():
    # End-to-end version of the above, through the real code path
    # (max_class_only=True, the default, is what originally triggered the
    # bug): explaining the class that IS the location's own winner must
    # still produce a real, nonzero dual from its actual runner-up.
    cls_pred = torch.tensor([[[[0.9]], [[0.3]], [[0.1]]]])  # (1, 3, 1, 1)

    relevance = _initialize_relevance(
        [cls_pred], cls=0, max_class_only=True, contrastive=True
    )
    seeded = relevance.scatter(which=scale_key(0))

    primal, dual = seeded[0], seeded[1]
    assert primal[0, 0, 0].item() == cls_pred[0, 0, 0, 0].item()
    assert dual.sum().item() > 0.0
    assert dual[1, 0, 0].item() > 0.0  # class 1, the real runner-up


# ---------------------------------------------------------------------------
# _localize_to_bbox (eq 7)
# ---------------------------------------------------------------------------


def test_localize_to_bbox_zeros_cells_outside_the_box():
    # 4x4 grid over a 64x64 image (stride 16). bbox covers only the top-left
    # cell's anchor point (8, 8); every other cell's anchor falls outside.
    cls_pred = torch.ones(1, 2, 4, 4)

    localized = _localize_to_bbox(cls_pred, bbox=(0, 0, 15, 15), image_size=(64, 64))

    assert localized[0, 0, 0, 0].item() == 1.0  # anchor (8, 8): inside
    assert localized.sum().item() == 2.0  # both classes at that one cell, nothing else


def test_localize_to_bbox_keeps_every_cell_when_bbox_covers_the_whole_image():
    cls_pred = torch.rand(1, 3, 4, 4)

    localized = _localize_to_bbox(cls_pred, bbox=(0, 0, 64, 64), image_size=(64, 64))

    assert torch.equal(localized, cls_pred)


def test_initialize_relevance_bbox_zeros_a_confident_winner_outside_the_box():
    # Two locations, one class. Location 0 (outside bbox) has the highest
    # confidence; location 1 (inside bbox) is real but weaker. Without
    # localization, max_class_only would keep location 0 regardless of the
    # box - the bbox gate must zero it before that reduction ever runs.
    cls_pred = torch.tensor([[[[0.9, 0.3]]]])  # (1, 1, 1, 2), stride = image_w / 2

    relevance = _initialize_relevance(
        [cls_pred],
        cls=0,
        max_class_only=True,
        contrastive=False,
        bbox=(4, 0, 8, 4),  # only column 1's anchor (x=6) falls inside
        image_size=(4, 8),
    )
    seeded = relevance.scatter(which=scale_key(0))

    assert seeded[0, 0, 0, 0].item() == 0.0
    assert seeded[0, 0, 0, 1].item() == cls_pred[0, 0, 0, 1].item()
