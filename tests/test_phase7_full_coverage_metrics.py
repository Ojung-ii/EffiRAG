from effirag.phase7_eval_utils import coverage_stats


def test_coverage_partial_full_recall():
    gold = {"doc a::0", "doc b::1"}
    stage_one = {"doc a::0"}
    stage_all = {"doc a::0", "doc b::1"}
    stage_none = set()

    one = coverage_stats(gold, stage_one)
    assert one["partial_hit"] == 1.0
    assert one["full_coverage"] == 0.0
    assert one["gold_recall"] == 0.5

    all_ = coverage_stats(gold, stage_all)
    assert all_["partial_hit"] == 1.0
    assert all_["full_coverage"] == 1.0
    assert all_["gold_recall"] == 1.0

    none = coverage_stats(gold, stage_none)
    assert none["partial_hit"] == 0.0
    assert none["full_coverage"] == 0.0
    assert none["gold_recall"] == 0.0


def test_selected_subset_cannot_exceed_phase1_full():
    gold = {"doc a::0", "doc b::1"}
    phase1 = {"doc a::0", "doc b::1", "doc c::2"}
    selected = {"doc a::0"}
    p1 = coverage_stats(gold, phase1)
    sel = coverage_stats(gold, selected)
    assert sel["full_coverage"] <= p1["full_coverage"]

