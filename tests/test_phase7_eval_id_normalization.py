from effirag.phase7_eval_utils import canonical_support_key, normalize_evidence_id, normalize_title


def test_normalize_title_basic():
    assert normalize_title("  The Newcomers (Film)  ") == "the newcomers (film)"
    assert normalize_title("A   B\tC") == "a b c"


def test_normalize_evidence_id_variants():
    assert normalize_evidence_id("The Newcomers (film)::0") == "the newcomers (film)::0"
    assert normalize_evidence_id("The Newcomers (film)::sent=0") == "the newcomers (film)::0"
    assert normalize_evidence_id({"title": "The Newcomers (film)", "sent_idx": 0}) == "the newcomers (film)::0"
    assert canonical_support_key("The Newcomers (film)", 0) == "the newcomers (film)::0"

