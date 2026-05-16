from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_method_complexity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_method_complexity", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_branch_scan_risk_levels(tmp_path: Path):
    mod = _load_module()

    effirag_dir = tmp_path / "effirag"
    scripts_dir = tmp_path / "scripts"
    config_root = tmp_path / "configs" / "main_config" / "copy_span_instruction_unified"
    profile_dir = config_root / "profile_a"
    effirag_dir.mkdir(parents=True)
    scripts_dir.mkdir(parents=True)
    profile_dir.mkdir(parents=True)

    (effirag_dir / "method.py").write_text(
        "if dataset == 'hotpotqa':\n    pass\n",
        encoding="utf-8",
    )
    (scripts_dir / "runner.sh").write_text(
        "echo outputs/hotpotqa/logs\n",
        encoding="utf-8",
    )
    (profile_dir / "hotpotqa.yaml").write_text(
        "dataset_tag: hotpotqa\noutput_dir: outputs/x/hotpotqa\n",
        encoding="utf-8",
    )

    rows = mod.scan_dataset_branches(
        code_roots=[effirag_dir, scripts_dir],
        config_root=config_root,
        repo_root=tmp_path,
    )
    assert rows, "scan should produce at least one row"

    high_rows = [r for r in rows if str(r.get("risk_level")) == "high"]
    medium_rows = [r for r in rows if str(r.get("risk_level")) == "medium"]
    low_rows = [r for r in rows if str(r.get("risk_level")) == "low"]
    assert high_rows
    assert medium_rows
    assert low_rows


def test_profile_complexity_flag_aggregation(tmp_path: Path):
    mod = _load_module()
    config_root = tmp_path / "configs" / "main_config" / "copy_span_instruction_unified"
    a_dir = config_root / "profile_a"
    b_dir = config_root / "profile_b"
    a_dir.mkdir(parents=True)
    b_dir.mkdir(parents=True)

    (a_dir / "hotpotqa.yaml").write_text(
        "\n".join(
            [
                "gl_rcedr_enabled: true",
                "sentence_contract_render_enabled: false",
                "support_span_contract_enabled: true",
                "adaptive_support_span_enabled: false",
                "flag_x_enabled: true",
                "flag_y_enabled: false",
                "max_rendered_tokens: 360",
                "bridge_score_threshold: 0.35",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (b_dir / "hotpotqa.yaml").write_text(
        "\n".join(
            [
                "gl_rcedr_enabled: false",
                "sentence_contract_render_enabled: true",
                "support_span_contract_enabled: false",
                "adaptive_support_span_enabled: true",
                "max_prompt_tokens: 700",
                "coverage_gain_threshold: 0.05",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = mod.collect_profile_complexity(config_root)
    by_profile = {str(r.get("profile")): r for r in rows}

    assert "profile_a" in by_profile
    assert "profile_b" in by_profile

    row_a = by_profile["profile_a"]
    assert int(row_a["num_enabled_flags"]) >= 3
    assert int(row_a["num_disabled_flags"]) >= 1
    assert bool(row_a["uses_support_span"]) is True
    assert bool(row_a["uses_gl_rcedr"]) is True

    row_b = by_profile["profile_b"]
    assert bool(row_b["uses_sentence_contract"]) is True
    assert bool(row_b["uses_adaptive_span"]) is True
