#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ojungii/EffiRAG"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
LIMIT="${LIMIT:-100}"
OUT_ROOT="${OUT_ROOT:-outputs/phase7_evidence_flow/corridor_bq_experiment}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/logs"

echo "[corridor-bq] audit"
PYTHONPATH=. "${PYTHON}" scripts/audit_phase7_method.py \
  2>&1 | tee "${OUT_ROOT}/logs/audit_phase7_method.log"

DATASETS=("hotpotqa" "2wikimultihopqa")
PROFILES=("balanced_384_8:384:8" "legacy_512_10:512:10")
VARIANTS=(
  "baseline_a_minus_r:a_minus_r:false:false:false"
  "anchor_decay_a_plus_d_minus_r:a_plus_decay_minus_r:false:true:false"
  "corridor_a_plus_bq_minus_r:a_plus_bq_minus_r:true:false:false"
  "corridor_conditional_r_a_plus_bq_minus_rq:a_plus_bq_minus_rq:true:false:true"
)

for profile_row in "${PROFILES[@]}"; do
  IFS=":" read -r profile_name profile_tokens profile_atoms <<<"${profile_row}"

  for variant_row in "${VARIANTS[@]}"; do
    IFS=":" read -r variant_name objective_mode corridor_enabled decay_enabled cond_r_enabled <<<"${variant_row}"

    for ds in "${DATASETS[@]}"; do
      cfg="configs/main_config/phase7_evidence_flow/${ds}.yaml"
      out_dir="${OUT_ROOT}/${profile_name}/${variant_name}/${ds}"
      log_file="${OUT_ROOT}/logs/${profile_name}__${variant_name}__${ds}.log"
      mkdir -p "${out_dir}"

      echo "[corridor-bq] profile=${profile_name} variant=${variant_name} dataset=${ds} limit=${LIMIT} out=${out_dir}"
      PYTHONPATH=. "${PYTHON}" -m effirag.run_rag \
        --config "${cfg}" \
        --dataset "${ds}" \
        --limit "${LIMIT}" \
        --output-dir "${out_dir}" \
        --timestamp-output false \
        --phase7-enable-phase2-refinement false \
        --phase7-objective-mode "${objective_mode}" \
        --phase7-lambda-bridge 1.0 \
        --phase7-lambda-decay 1.0 \
        --phase7-lambda-bq 1.0 \
        --phase7-mu-redundancy 1.0 \
        --phase7-conditional-redundancy-enabled "${cond_r_enabled}" \
        --phase7-corridor-enabled "${corridor_enabled}" \
        --phase7-corridor-max-anchors 4 \
        --phase7-corridor-max-seeds 16 \
        --phase7-corridor-max-hops 3 \
        --phase7-corridor-max-paths-per-pair 1 \
        --phase7-corridor-degree-cap 100 \
        --phase7-corridor-max-pairs 64 \
        --phase7-anchor-decay-enabled "${decay_enabled}" \
        --phase7-anchor-decay-gamma 0.7 \
        --phase7-anchor-decay-max-hops 4 \
        --phase7-candidate-top-m 96 \
        --phase7-max-context-tokens "${profile_tokens}" \
        --phase7-max-selected-atoms "${profile_atoms}" \
        --phase7-diagnostics-enabled true \
        --phase7-diagnostics-max-examples-to-dump "${LIMIT}" \
        --phase7-diagnostics-dump-text true \
        --phase7-diagnostics-dump-context true \
        --phase7-diagnostics-dump-scores true \
        --phase7-diagnostics-fail-on-unit-mismatch true \
        >"${log_file}" 2>&1 &
    done
    wait

    for ds in "${DATASETS[@]}"; do
      out_dir="${OUT_ROOT}/${profile_name}/${variant_name}/${ds}"
      diag_jsonl="${out_dir}/phase7_diagnostics.jsonl"
      summary_json="${out_dir}/diagnostic_summary.json"
      if [[ -f "${diag_jsonl}" ]]; then
        if [[ "${ds}" == "2wikimultihopqa" ]]; then
          PYTHONPATH=. "${PYTHON}" scripts/analyze_phase7_diagnostics.py \
            --diag-jsonl "${diag_jsonl}" \
            --out "${summary_json}" \
            --compare-root "${OUT_ROOT}" \
            --compare-csv "${OUT_ROOT}/variant_comparison.csv" \
            --failure-md "${out_dir}/failure_cases.md" \
            --failure-top-k 20 \
            >>"${OUT_ROOT}/logs/analyze_${profile_name}__${variant_name}__${ds}.log" 2>&1
        else
          PYTHONPATH=. "${PYTHON}" scripts/analyze_phase7_diagnostics.py \
            --diag-jsonl "${diag_jsonl}" \
            --out "${summary_json}" \
            --compare-root "${OUT_ROOT}" \
            --compare-csv "${OUT_ROOT}/variant_comparison.csv" \
            >>"${OUT_ROOT}/logs/analyze_${profile_name}__${variant_name}__${ds}.log" 2>&1
        fi
      fi
    done
  done
done

PYTHONPATH=. "${PYTHON}" scripts/summarize_phase7_results.py --root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_phase7_results.log"

PYTHONPATH=. "${PYTHON}" scripts/analyze_phase7_failures.py \
  --root "${OUT_ROOT}" \
  --out "${OUT_ROOT}/failure_case_analysis.md" \
  2>&1 | tee "${OUT_ROOT}/logs/analyze_phase7_failures.log"

PYTHONPATH=. "${PYTHON}" scripts/summarize_phase7_corridor_bq_experiment.py \
  --root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/summarize_corridor_bq.log"

echo "[corridor-bq] done: ${OUT_ROOT}"
