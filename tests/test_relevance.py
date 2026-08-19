import pytest
import torch

from src.lrp.relevance import LayerRelevance, RelevanceMessage, scale_key

# ---------------------------------------------------------------------------
# RelevanceMessage
# ---------------------------------------------------------------------------


def test_relevance_message_accepts_a_tensor():
    msg = RelevanceMessage(from_=None, to=-1, relevance=torch.ones(2, 2))
    assert torch.equal(msg.relevance, torch.ones(2, 2))


def test_relevance_message_rejects_non_tensor_payloads():
    with pytest.raises(TypeError):
        RelevanceMessage(from_=None, to=-1, relevance=[torch.ones(2, 2)])


# ---------------------------------------------------------------------------
# scale_key
# ---------------------------------------------------------------------------


def test_scale_key_is_distinct_from_own_input_and_real_layer_ids():
    keys = {scale_key(i) for i in range(3)}
    assert -1 not in keys
    assert all(k < 0 for k in keys)
    assert len(keys) == 3


def test_scale_key_is_stable_and_ordered():
    assert scale_key(0) == -2
    assert scale_key(1) == -3
    assert scale_key(2) == -4


# ---------------------------------------------------------------------------
# LayerRelevance construction
# ---------------------------------------------------------------------------


def test_layer_relevance_starts_empty():
    relevance = LayerRelevance()
    assert relevance.cache == {}


# ---------------------------------------------------------------------------
# gather
# ---------------------------------------------------------------------------


def test_gather_stores_a_new_target():
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=torch.ones(2)))
    assert torch.equal(relevance.cache[-1].relevance, torch.ones(2))


def test_gather_accumulates_repeat_contributions_to_the_same_target():
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=5, relevance=torch.ones(2)))
    relevance.gather(RelevanceMessage(from_=None, to=5, relevance=torch.ones(2) * 3))
    assert torch.equal(relevance.cache[5].relevance, torch.ones(2) * 4)


def test_gather_accumulates_own_input_the_same_way_as_any_other_target():
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=torch.ones(2)))
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=torch.ones(2) * 2))
    assert torch.equal(relevance.cache[-1].relevance, torch.ones(2) * 3)


def test_gather_accepts_multiple_messages_variadically():
    relevance = LayerRelevance()
    relevance.gather(
        RelevanceMessage(from_=None, to=1, relevance=torch.ones(1)),
        RelevanceMessage(from_=None, to=2, relevance=torch.ones(1) * 2),
    )
    assert relevance.cache[1].relevance.item() == 1
    assert relevance.cache[2].relevance.item() == 2


def test_gather_raises_a_helpful_error_on_shape_mismatch():
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=5, relevance=torch.ones(2)))
    with pytest.raises(RuntimeError, match="layer 5"):
        relevance.gather(RelevanceMessage(from_=None, to=5, relevance=torch.ones(3)))


# ---------------------------------------------------------------------------
# scatter
# ---------------------------------------------------------------------------


def test_scatter_own_input_defaults_to_empty_when_never_gathered():
    relevance = LayerRelevance()
    payload = relevance.scatter(which=-1)
    assert torch.equal(payload, torch.tensor([]))


def test_scatter_returns_and_destroys_by_default():
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=torch.ones(2)))
    payload = relevance.scatter(which=-1)
    assert torch.equal(payload, torch.ones(2))
    assert -1 not in relevance.cache


def test_scatter_can_leave_the_cache_entry_intact():
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=torch.ones(2)))
    relevance.scatter(which=-1, destroy=False)
    assert -1 in relevance.cache


def test_scatter_raises_for_a_missing_non_own_target():
    relevance = LayerRelevance()
    with pytest.raises(KeyError):
        relevance.scatter(which=7)


def test_scatter_returns_a_present_non_own_target():
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=7, relevance=torch.ones(2) * 5))
    assert torch.equal(relevance.scatter(which=7), torch.ones(2) * 5)


def test_scatter_bulk_returns_every_message_and_clears_the_cache():
    relevance = LayerRelevance()
    relevance.gather(
        RelevanceMessage(from_=None, to=-1, relevance=torch.ones(1)),
        RelevanceMessage(from_=None, to=3, relevance=torch.ones(1) * 2),
    )
    messages = relevance.scatter()
    assert {m.to for m in messages} == {-1, 3}
    assert relevance.cache == {}


# ---------------------------------------------------------------------------
# pop_cache
# ---------------------------------------------------------------------------


def test_pop_cache_is_a_noop_when_nothing_is_cached_for_that_layer():
    relevance = LayerRelevance()
    relevance.pop_cache(42)  # should not raise
    assert relevance.cache == {}


def test_pop_cache_merges_into_own_input():
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=9, relevance=torch.ones(2)))
    relevance.pop_cache(9)
    assert 9 not in relevance.cache
    assert torch.equal(relevance.cache[-1].relevance, torch.ones(2))


def test_pop_cache_accumulates_with_existing_own_input():
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=torch.ones(2)))
    relevance.gather(RelevanceMessage(from_=None, to=9, relevance=torch.ones(2) * 4))
    relevance.pop_cache(9)
    assert torch.equal(relevance.cache[-1].relevance, torch.ones(2) * 5)


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------


def test_str_on_a_fresh_layer_relevance_does_not_raise():
    assert "LayerRelevance" in str(LayerRelevance())


def test_str_reports_fractions_of_total_relevance():
    relevance = LayerRelevance()
    relevance.gather(RelevanceMessage(from_=None, to=-1, relevance=torch.tensor([3.0])))
    relevance.gather(RelevanceMessage(from_=None, to=2, relevance=torch.tensor([1.0])))
    text = str(relevance)
    assert "0.75" in text
    assert "(2, 0.25)" in text


def test_str_contrastive_uses_primal_dual_format():
    relevance = LayerRelevance(contrastive=True)
    relevance.gather(
        RelevanceMessage(from_=None, to=-1, relevance=torch.tensor([[2.0], [1.0]]))
    )
    assert "P:" in str(relevance) and "D:" in str(relevance)
