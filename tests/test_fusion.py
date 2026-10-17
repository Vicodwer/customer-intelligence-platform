import pytest

from src.rag.fusion import reciprocal_rank_fusion


def test_document_ranked_well_by_both_lists_wins():
    bm25 = ["a", "b", "c"]
    dense = ["b", "a", "d"]

    fused = reciprocal_rank_fusion([bm25, dense])

    # a and b are top-2 in both lists; c and d each appear in one list only.
    assert set(fused[:2]) == {"a", "b"}
    assert set(fused[2:]) == {"c", "d"}


def test_scores_match_hand_calculation():
    # x: rank 1 in list 1 and rank 3 in list 2 -> 1/61 + 1/63 = 0.032266
    # y: rank 2 in both lists                  -> 1/62 + 1/62 = 0.032258
    # Nearly tied; x is a hair ahead because 1/rank rewards a very high rank
    # slightly more than it penalises a lower one.
    assert 1 / 61 + 1 / 63 > 1 / 62 + 1 / 62

    fused = reciprocal_rank_fusion([["x", "y", "z"], ["w", "y", "x"]], k=60)

    assert fused.index("x") < fused.index("y")
    # z and w are each found by one list only (1/63 and 1/61), so they trail.
    assert fused[:2] == ["x", "y"]


def test_ranks_not_scores_decide_so_no_scaling_is_needed():
    # Fusion only sees ordered ids, so lists produced on very different
    # score scales combine without any rescaling.
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])

    assert fused == ["a", "b"]  # exact tie, broken by first appearance


def test_document_found_by_only_one_list_is_still_returned():
    fused = reciprocal_rank_fusion([["a"], ["b"]])

    assert set(fused) == {"a", "b"}


def test_depth_limits_the_result():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]], depth=2)

    assert len(fused) == 2


def test_repeat_inside_one_list_counts_once():
    once = reciprocal_rank_fusion([["a", "b"], ["b"]])
    repeated = reciprocal_rank_fusion([["a", "a", "a", "b"], ["b"]])

    assert once == repeated == ["b", "a"]


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_invalid_k_is_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)