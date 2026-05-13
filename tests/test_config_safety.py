import pytest

from effirag.config import RagConfig, audit_config_path_mode_consistency, dataclass_from_dict


def test_dataclass_from_dict_warns_on_unknown_keys_by_default():
    with pytest.warns(RuntimeWarning, match="Unknown config keys"):
        cfg = dataclass_from_dict(RagConfig, {"dataset": "hotpotqa", "typo_key_should_warn": 1})
    assert cfg.dataset == "hotpotqa"


def test_dataclass_from_dict_strict_unknown_keys_raises():
    with pytest.raises(ValueError, match="Unknown config keys"):
        dataclass_from_dict(
            RagConfig,
            {"dataset": "hotpotqa", "typo_key_should_raise": 1},
            strict_unknown_keys=True,
        )


def test_dataclass_from_dict_ignored_unknown_keys_skip_warning():
    cfg = dataclass_from_dict(
        RagConfig,
        {"dataset": "hotpotqa", "config": "configs/canonical/rag.yaml"},
        ignored_unknown_keys={"config"},
    )
    assert cfg.dataset == "hotpotqa"


def test_audit_config_path_mode_consistency_warns_for_canonical_path_with_baseline_mode():
    with pytest.warns(RuntimeWarning, match="canonical profile"):
        rows = audit_config_path_mode_consistency(
            "configs/canonical/rag_entity_first_chunk_grounded.yaml",
            "baseline",
            strict=False,
        )
    assert rows


def test_audit_config_path_mode_consistency_strict_raises():
    with pytest.raises(ValueError, match="canonical profile"):
        audit_config_path_mode_consistency(
            "configs/canonical/rag_entity_first_chunk_grounded.yaml",
            "baseline",
            strict=True,
        )
