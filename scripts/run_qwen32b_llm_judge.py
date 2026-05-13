#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]

JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "judge_id": {"type": "string"},
        "score": {"type": "integer", "enum": [0, 1, 2, 3]},
        "sufficient": {"type": "boolean"},
        "minimal_evidence_ids": {"type": "array", "items": {"type": "integer"}},
        "answer_surface_present": {"type": "boolean"},
        "equivalent_evidence": {"type": "boolean"},
        "insufficient_reason": {"type": ["string", "null"]},
        "error": {"type": ["string", "null"]},
    },
    "required": [
        "judge_id",
        "score",
        "sufficient",
        "minimal_evidence_ids",
        "answer_surface_present",
        "equivalent_evidence",
        "insufficient_reason",
        "error",
    ],
    "additionalProperties": False,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _strip(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = _strip(line)
            if not text:
                continue
            rows.append(dict(json.loads(text) or {}))
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            n += 1
    return int(n)


def _write_tsv(path: Path, headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def _iter_chunk_files(input_root: Path) -> List[Path]:
    files = []
    for fp in sorted(input_root.glob("*.jsonl")):
        if fp.name.endswith("_judged.jsonl"):
            continue
        files.append(fp)
    return files


def _strip_code_fence(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return raw


def _extract_first_json_object(text: str) -> str:
    s = _strip_code_fence(str(text or ""))
    if not s:
        raise ValueError("empty completion text")
    try:
        json.loads(s)
        return s
    except Exception:
        pass

    start = s.find("{")
    if start < 0:
        raise ValueError("no json object start '{' found")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                json.loads(candidate)
                return candidate
    raise ValueError("unable to extract valid json object from completion text")


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    txt = str(value).strip().lower()
    if txt in {"true", "1", "yes", "y"}:
        return True
    if txt in {"false", "0", "no", "n"}:
        return False
    return bool(default)


def _normalize_judge_payload(parsed: Mapping[str, Any], input_rec: Mapping[str, Any]) -> Dict[str, Any]:
    valid_ids = {int(x.get("id")) for x in list(input_rec.get("evidence", []) or []) if str(x.get("id", "")).isdigit()}

    raw_score = parsed.get("score")
    score = _safe_int(raw_score, -999)
    if score == 4:
        score = 3
    if score not in {0, 1, 2, 3}:
        raise ValueError(f"invalid score: {raw_score}")

    mids_raw = parsed.get("minimal_evidence_ids", [])
    if mids_raw is None:
        mids_raw = []
    if not isinstance(mids_raw, list):
        raise ValueError("minimal_evidence_ids must be a list")
    mids: List[int] = []
    for x in mids_raw:
        ii = _safe_int(x, -1)
        if ii in valid_ids:
            mids.append(ii)
    mids = sorted(set(mids))

    sufficient = bool(score >= 2)
    answer_surface_present = _coerce_bool(parsed.get("answer_surface_present"), False)
    equivalent_evidence = _coerce_bool(parsed.get("equivalent_evidence"), False)
    insuff = parsed.get("insufficient_reason")
    insufficient_reason = None if insuff is None else _strip(insuff)
    error_val = parsed.get("error")
    error_text = None if error_val is None else _strip(error_val)

    return {
        "judge_id": _strip(input_rec.get("judge_id")),
        "dataset": _strip(input_rec.get("dataset")),
        "qid": _strip(input_rec.get("qid")),
        "system_id": _strip(input_rec.get("system_id")),
        "score": int(score),
        "sufficient": bool(sufficient),
        "minimal_evidence_ids": mids,
        "answer_surface_present": bool(answer_surface_present),
        "equivalent_evidence": bool(equivalent_evidence),
        "insufficient_reason": insufficient_reason,
        "error": error_text,
    }


def _http_chat_completion(
    *,
    base_url: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout_sec: float,
) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if _strip(api_key):
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return {"status": int(resp.status), "body": body}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"status": int(exc.code), "body": body}
    except Exception as exc:
        return {"status": -1, "body": str(exc)}


def _extract_content_from_completion(resp_payload: Mapping[str, Any]) -> str:
    choices = list(resp_payload.get("choices", []) or [])
    if not choices:
        raise ValueError("missing choices in completion response")
    msg = dict(choices[0].get("message", {}) or {})
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if t is not None:
                    texts.append(str(t))
                elif "content" in item:
                    texts.append(str(item.get("content")))
            else:
                texts.append(str(item))
        return "".join(texts)
    return str(content)


def _read_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _build_prompt_renderer(raw_template: str) -> Tuple[str, str]:
    placeholders = [
        "{INPUT_RECORD_JSON}",
        "{{INPUT_RECORD_JSON}}",
        "<INPUT_RECORD_JSON>",
        "__INPUT_RECORD_JSON__",
    ]
    for ph in placeholders:
        if ph in raw_template:
            return raw_template, ph
    # Keep rubric untouched; append input placeholder for runtime insertion.
    template = (
        raw_template.rstrip()
        + "\n\nInput record JSON:\n{INPUT_RECORD_JSON}\n\nReturn JSON only."
    )
    return template, "{INPUT_RECORD_JSON}"


def _build_readme(round_dir: Path, manifest: Mapping[str, Any], validation_pass: Optional[bool]) -> None:
    text = f"""# Qwen32B LLM Judge Round

- generated_at_utc: {_utc_now_iso()}
- input round path: {manifest.get('input_round')}
- reused prompt path: {manifest.get('prompt_path')}
- judge model: {manifest.get('judge_model')}
- served model name: {manifest.get('served_model_name')}
- endpoint: {manifest.get('endpoint')}
- generation: temperature={manifest.get('temperature')}, top_p={manifest.get('top_p')}, max_tokens={manifest.get('max_tokens')}

## Outputs
- judged JSONL: `judge_outputs/`
- validation: `validation/judge_output_validation_report.json`, `validation/judge_output_validation_report.md`
- summaries: `summary/llm_judge_summary_blind.json`, `summary/llm_judge_summary_unblinded.json`
- logs: `logs/api_errors.jsonl`, `logs/parse_failures.jsonl`, `logs/retry_records.jsonl`

## Notes
- Codex-as-judge flow is deprecated in this round.
- This pipeline uses Qwen2.5-32B-AWQ served by vLLM as the only judge model.
- Existing judge input artifacts are reused and preserved.
- operational_instruction_for_chatgpt_or_codex.md is preserved for compatibility but deprecated for this pipeline.

## Validation
- validation_pass: {validation_pass}
"""
    (round_dir / "README.md").write_text(text, encoding="utf-8")


def _format_seconds(sec: float) -> str:
    s = max(0, int(sec))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{ss:02d}"
    return f"{m:02d}:{ss:02d}"


def _print_progress(
    *,
    chunk_name: str,
    chunk_done: int,
    chunk_total: int,
    global_done: int,
    global_total: int,
    error_count: int,
    start_ts: float,
) -> None:
    elapsed = max(0.001, time.time() - start_ts)
    speed = float(global_done) / elapsed if global_done > 0 else 0.0
    remain = max(0, global_total - global_done)
    eta = float(remain) / max(speed, 1e-9) if speed > 0 else 0.0
    pct = (100.0 * float(global_done) / float(global_total)) if global_total > 0 else 0.0
    line = (
        f"[judge] {chunk_name} chunk {chunk_done}/{chunk_total} | "
        f"global {global_done}/{global_total} ({pct:5.1f}%) | "
        f"err={error_count} | {speed:.2f} rec/s | ETA {_format_seconds(eta)}"
    )
    print("\r" + line, end="", flush=True)


class _CapabilityState:
    def __init__(self, guided_json_enabled: bool, response_format_enabled: bool) -> None:
        self._lock = Lock()
        self._guided_json_enabled = bool(guided_json_enabled)
        self._response_format_enabled = bool(response_format_enabled)

    def get(self) -> Tuple[bool, bool]:
        with self._lock:
            return self._guided_json_enabled, self._response_format_enabled

    def disable_guided_json(self) -> None:
        with self._lock:
            self._guided_json_enabled = False

    def disable_response_format(self) -> None:
        with self._lock:
            self._response_format_enabled = False

    def snapshot(self) -> Dict[str, bool]:
        with self._lock:
            return {
                "guided_json_enabled": bool(self._guided_json_enabled),
                "response_format_enabled": bool(self._response_format_enabled),
            }


def _is_capability_unsupported(raw_body: str, capability_key: str) -> bool:
    lowered = str(raw_body or "").lower()
    if capability_key not in lowered:
        return False
    hints = [
        "unknown",
        "unsupported",
        "invalid",
        "not allowed",
        "extra inputs are not permitted",
        "unexpected keyword",
        "unrecognized",
    ]
    return any(h in lowered for h in hints)


def _judge_one_record(
    *,
    record: Mapping[str, Any],
    chunk_name: str,
    prompt_template: str,
    placeholder: str,
    logs_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int,
    request_timeout: float,
    sleep_ms: int,
    caps: _CapabilityState,
) -> Tuple[Dict[str, Any], bool]:
    input_judge_id = _strip(record.get("judge_id"))
    rec_prompt = prompt_template.replace(placeholder, json.dumps(record, ensure_ascii=False))

    normalized: Optional[Dict[str, Any]] = None
    final_error: Optional[str] = None
    attempts = int(max_retries) + 1
    attempts_used = 0

    for attempt in range(attempts):
        attempts_used = attempt + 1
        if attempt > 0:
            _append_jsonl(
                logs_dir / "retry_records.jsonl",
                {
                    "timestamp_utc": _utc_now_iso(),
                    "chunk_file": chunk_name,
                    "judge_id": input_judge_id,
                    "attempt": attempt + 1,
                },
            )

        guided_enabled, response_enabled = caps.get()
        payload: Dict[str, Any] = {
            "model": _strip(model),
            "messages": [{"role": "user", "content": rec_prompt}],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if guided_enabled:
            payload["guided_json"] = JUDGE_SCHEMA
        if response_enabled:
            payload["response_format"] = {"type": "json_object"}

        resp = _http_chat_completion(
            base_url=_strip(base_url),
            api_key=_strip(api_key),
            payload=payload,
            timeout_sec=float(request_timeout),
        )
        status = _safe_int(resp.get("status"), -1)
        body = _strip(resp.get("body"))
        if status < 200 or status >= 300:
            if guided_enabled and _is_capability_unsupported(body, "guided_json"):
                caps.disable_guided_json()
                _append_jsonl(
                    logs_dir / "api_errors.jsonl",
                    {
                        "timestamp_utc": _utc_now_iso(),
                        "chunk_file": chunk_name,
                        "judge_id": input_judge_id,
                        "attempt": attempt + 1,
                        "status": status,
                        "error_type": "guided_json_unsupported_disabled",
                        "response_excerpt": body[:800],
                    },
                )
                continue
            if response_enabled and _is_capability_unsupported(body, "response_format"):
                caps.disable_response_format()
                _append_jsonl(
                    logs_dir / "api_errors.jsonl",
                    {
                        "timestamp_utc": _utc_now_iso(),
                        "chunk_file": chunk_name,
                        "judge_id": input_judge_id,
                        "attempt": attempt + 1,
                        "status": status,
                        "error_type": "response_format_unsupported_disabled",
                        "response_excerpt": body[:800],
                    },
                )
                continue

            final_error = f"api_error_status_{status}"
            _append_jsonl(
                logs_dir / "api_errors.jsonl",
                {
                    "timestamp_utc": _utc_now_iso(),
                    "chunk_file": chunk_name,
                    "judge_id": input_judge_id,
                    "attempt": attempt + 1,
                    "status": status,
                    "error_type": "api_error",
                    "response_excerpt": body[:1200],
                },
            )
            continue

        try:
            resp_json = dict(json.loads(body) or {})
            completion_text = _extract_content_from_completion(resp_json)
            parsed_text = _extract_first_json_object(completion_text)
            parsed_payload = dict(json.loads(parsed_text) or {})
            normalized = _normalize_judge_payload(parsed_payload, record)
            break
        except Exception as exc:
            final_error = f"parse_failure: {exc}"
            _append_jsonl(
                logs_dir / "parse_failures.jsonl",
                {
                    "timestamp_utc": _utc_now_iso(),
                    "chunk_file": chunk_name,
                    "judge_id": input_judge_id,
                    "attempt": attempt + 1,
                    "error": str(exc),
                    "response_excerpt": body[:1200],
                },
            )
            continue

    is_error = False
    if normalized is None:
        is_error = True
        normalized = {
            "judge_id": input_judge_id,
            "dataset": _strip(record.get("dataset")),
            "qid": _strip(record.get("qid")),
            "system_id": _strip(record.get("system_id")),
            "score": 0,
            "sufficient": False,
            "minimal_evidence_ids": [],
            "answer_surface_present": False,
            "equivalent_evidence": False,
            "insufficient_reason": None,
            "error": final_error or "unknown_error",
        }

    normalized["judge_model"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    normalized["served_model_name"] = _strip(model)
    normalized["temperature"] = float(temperature)
    normalized["top_p"] = float(top_p)
    normalized["max_tokens"] = int(max_tokens)
    normalized["base_url"] = _strip(base_url)
    normalized["request_attempts"] = int(attempts_used)

    if int(sleep_ms) > 0:
        time.sleep(float(sleep_ms) / 1000.0)

    return normalized, is_error


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Qwen2.5-32B-AWQ as LLM judge over prebuilt chunked inputs.")
    ap.add_argument("--input-root", required=True, help=".../judge_chunks")
    ap.add_argument("--prompt-path", required=True, help=".../prompts/llm_judge_rubric.md")
    ap.add_argument("--mapping-root", required=True, help=".../hidden_mapping")
    ap.add_argument("--output-root", required=True, help="outputs/llm_judge_qwen32b_round")
    ap.add_argument("--base-url", default="http://localhost:8012/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="qwen2.5-32b-awq-judge")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--request-timeout", type=float, default=120.0)
    ap.add_argument("--max-records", type=int, default=0, help="0 means all")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--round-id", default="", help="Optional round directory name")
    ap.add_argument("--use-guided-json", default="true")
    ap.add_argument("--use-response-format-json", default="false")
    ap.add_argument("--sleep-ms", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4, help="Parallel API requests per chunk")
    ap.add_argument("--show-progress", default="true", help="Print live CLI progress")
    ap.add_argument(
        "--progress-update-every",
        type=int,
        default=1,
        help="Update progress display every N completed records",
    )
    args = ap.parse_args()

    input_root = Path(args.input_root).resolve()
    prompt_path = Path(args.prompt_path).resolve()
    mapping_root = Path(args.mapping_root).resolve()
    output_parent = Path(args.output_root).resolve()
    round_dir = output_parent / (_strip(args.round_id) or _utc_stamp())

    judge_outputs_dir = round_dir / "judge_outputs"
    validation_dir = round_dir / "validation"
    summary_dir = round_dir / "summary"
    logs_dir = round_dir / "logs"
    prompts_dir = round_dir / "prompts"
    archive_dir = judge_outputs_dir / "archive"
    for d in [judge_outputs_dir, validation_dir, summary_dir, logs_dir, prompts_dir, archive_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for fp in [logs_dir / "api_errors.jsonl", logs_dir / "parse_failures.jsonl", logs_dir / "retry_records.jsonl"]:
        fp.touch(exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"prompt_path not found: {prompt_path}")
    if not mapping_root.exists():
        raise FileNotFoundError(f"mapping_root not found: {mapping_root}")

    use_guided_json = _as_bool(args.use_guided_json, True)
    use_response_format_json = _as_bool(args.use_response_format_json, False)
    caps = _CapabilityState(bool(use_guided_json), bool(use_response_format_json))
    concurrency = max(1, int(args.concurrency or 1))
    show_progress = _as_bool(args.show_progress, True)
    progress_update_every = max(1, int(args.progress_update_every or 1))

    raw_template = _read_prompt_template(prompt_path)
    prompt_template, placeholder = _build_prompt_renderer(raw_template)
    (prompts_dir / "qwen32b_judge_prompt.md").write_text(prompt_template, encoding="utf-8")

    chunk_files = _iter_chunk_files(input_root)
    failures: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    expected_ids_rows: List[Dict[str, Any]] = []

    total_processed = 0
    total_judged = 0
    total_error_records = 0
    remaining = int(args.max_records or 0)
    global_target_total = 0
    global_done = 0
    global_start_ts = time.time()

    # Pre-compute target size for progress percentage.
    planned_remaining = int(args.max_records or 0)
    for fp in chunk_files:
        if planned_remaining == 0:
            break
        output_name = fp.stem + "_judged.jsonl"
        out_path = judge_outputs_dir / output_name
        if out_path.exists() and not args.overwrite:
            continue
        n_rows = len(_read_jsonl(fp))
        if planned_remaining > 0:
            n_rows = min(n_rows, planned_remaining)
            planned_remaining -= n_rows
        global_target_total += n_rows

    if show_progress:
        print(
            f"[judge] start: chunks={len(chunk_files)}, target_records={global_target_total}, "
            f"concurrency={concurrency}, model={_strip(args.model)}",
            flush=True,
        )

    for chunk_idx, chunk_path in enumerate(chunk_files):
        chunk_name = chunk_path.name
        output_name = chunk_path.stem + "_judged.jsonl"
        out_path = judge_outputs_dir / output_name
        chunk_rows = _read_jsonl(chunk_path)
        input_count_full = len(chunk_rows)

        if remaining > 0:
            if remaining <= 0:
                chunk_rows = []
            else:
                chunk_rows = chunk_rows[:remaining]

        expected_count = len(chunk_rows)
        for rec in chunk_rows:
            expected_ids_rows.append(
                {
                    "judge_id": _strip(rec.get("judge_id")),
                    "dataset": _strip(rec.get("dataset")),
                    "qid": _strip(rec.get("qid")),
                    "system_id": _strip(rec.get("system_id")),
                    "chunk_file": chunk_name,
                }
            )

        if expected_count == 0:
            run_rows.append(
                {
                    "chunk_index": chunk_idx,
                    "chunk_file": chunk_name,
                    "output_file": output_name,
                    "status": "skipped_empty",
                    "input_count_full": input_count_full,
                    "expected_output_count": 0,
                    "written_count": 0,
                    "error_count": 0,
                    "elapsed_sec": 0.0,
                    "note": "",
                }
            )
            continue

        if out_path.exists() and not args.overwrite:
            run_rows.append(
                {
                    "chunk_index": chunk_idx,
                    "chunk_file": chunk_name,
                    "output_file": output_name,
                    "status": "skipped_existing",
                    "input_count_full": input_count_full,
                    "expected_output_count": expected_count,
                    "written_count": len(_read_jsonl(out_path)),
                    "error_count": 0,
                    "elapsed_sec": 0.0,
                    "note": "existing judged file kept",
                }
            )
            total_processed += expected_count
            if remaining > 0:
                remaining -= expected_count
                if remaining <= 0:
                    break
            continue

        if out_path.exists() and args.overwrite:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            archived = archive_dir / f"{out_path.stem}__{stamp}.jsonl"
            shutil.move(str(out_path), str(archived))

        if out_path.exists():
            out_path.unlink()

        start_t = time.time()
        wrote_count = 0
        err_count = 0

        rec_iter = iter(chunk_rows)
        futures: Dict[Future, Mapping[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for _ in range(min(concurrency, expected_count)):
                try:
                    rec = next(rec_iter)
                except StopIteration:
                    break
                fut = ex.submit(
                    _judge_one_record,
                    record=rec,
                    chunk_name=chunk_name,
                    prompt_template=prompt_template,
                    placeholder=placeholder,
                    logs_dir=logs_dir,
                    base_url=_strip(args.base_url),
                    api_key=_strip(args.api_key),
                    model=_strip(args.model),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_tokens=int(args.max_tokens),
                    max_retries=int(args.max_retries),
                    request_timeout=float(args.request_timeout),
                    sleep_ms=int(args.sleep_ms),
                    caps=caps,
                )
                futures[fut] = rec

            while futures:
                done, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    _ = futures.pop(fut, None)
                    try:
                        normalized, is_error = fut.result()
                    except Exception as exc:
                        is_error = True
                        normalized = {
                            "judge_id": "",
                            "dataset": "",
                            "qid": "",
                            "system_id": "",
                            "score": 0,
                            "sufficient": False,
                            "minimal_evidence_ids": [],
                            "answer_surface_present": False,
                            "equivalent_evidence": False,
                            "insufficient_reason": None,
                            "error": f"worker_exception: {exc}",
                            "judge_model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
                            "served_model_name": _strip(args.model),
                            "temperature": float(args.temperature),
                            "top_p": float(args.top_p),
                            "max_tokens": int(args.max_tokens),
                            "base_url": _strip(args.base_url),
                            "request_attempts": 0,
                        }

                    if is_error:
                        err_count += 1
                        total_error_records += 1
                    _append_jsonl(out_path, normalized)
                    wrote_count += 1
                    total_judged += 1
                    global_done += 1

                    if show_progress and (global_done % progress_update_every == 0 or global_done == global_target_total):
                        _print_progress(
                            chunk_name=chunk_name,
                            chunk_done=wrote_count,
                            chunk_total=expected_count,
                            global_done=global_done,
                            global_total=max(1, global_target_total),
                            error_count=total_error_records,
                            start_ts=global_start_ts,
                        )

                    try:
                        nxt = next(rec_iter)
                    except StopIteration:
                        nxt = None
                    if nxt is not None:
                        nfut = ex.submit(
                            _judge_one_record,
                            record=nxt,
                            chunk_name=chunk_name,
                            prompt_template=prompt_template,
                            placeholder=placeholder,
                            logs_dir=logs_dir,
                            base_url=_strip(args.base_url),
                            api_key=_strip(args.api_key),
                            model=_strip(args.model),
                            temperature=float(args.temperature),
                            top_p=float(args.top_p),
                            max_tokens=int(args.max_tokens),
                            max_retries=int(args.max_retries),
                            request_timeout=float(args.request_timeout),
                            sleep_ms=int(args.sleep_ms),
                            caps=caps,
                        )
                        futures[nfut] = nxt

        elapsed = float(time.time() - start_t)
        run_rows.append(
            {
                "chunk_index": chunk_idx,
                "chunk_file": chunk_name,
                "output_file": output_name,
                "status": "ok",
                "input_count_full": input_count_full,
                "expected_output_count": expected_count,
                "written_count": wrote_count,
                "error_count": err_count,
                "elapsed_sec": round(elapsed, 4),
                "note": "",
            }
        )

        if show_progress:
            # finalize one line per chunk for readability in logs/tmux scrollback.
            print(
                f"\n[judge] done chunk={chunk_name} wrote={wrote_count}/{expected_count} "
                f"errors={err_count} elapsed={_format_seconds(time.time() - start_t)}",
                flush=True,
            )
        total_processed += expected_count
        if remaining > 0:
            remaining -= expected_count
            if remaining <= 0:
                break

    expected_ids_path = round_dir / "expected_judge_ids.jsonl"
    _write_jsonl(expected_ids_path, expected_ids_rows)

    failures_payload = {"count": len(failures), "items": failures}
    _write_json(round_dir / "failures.json", failures_payload)

    run_headers = [
        "chunk_index",
        "chunk_file",
        "output_file",
        "status",
        "input_count_full",
        "expected_output_count",
        "written_count",
        "error_count",
        "elapsed_sec",
        "note",
    ]
    _write_tsv(round_dir / "run_records.tsv", run_headers, run_rows)

    input_round = str(input_root.parent.resolve()) if input_root.name == "judge_chunks" else str(input_root.resolve())
    parsed_base = urlparse(_strip(args.base_url))
    manifest = {
        "generated_at_utc": _utc_now_iso(),
        "judge_model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "served_model_name": _strip(args.model),
        "endpoint": _strip(args.base_url),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_tokens),
        "input_round": input_round,
        "input_root": str(input_root),
        "prompt_path": str(prompt_path),
        "mapping_root": str(mapping_root),
        "output_round": str(round_dir),
        "vllm_port": int(parsed_base.port or 0),
        "codex_as_judge": False,
        "max_retries": int(args.max_retries),
        "request_timeout": float(args.request_timeout),
        "use_guided_json_requested": bool(use_guided_json),
        "use_guided_json_effective": bool(caps.snapshot().get("guided_json_enabled", False)),
        "use_response_format_json_requested": bool(use_response_format_json),
        "use_response_format_json_effective": bool(caps.snapshot().get("response_format_enabled", False)),
        "concurrency": int(concurrency),
        "max_records": int(args.max_records),
        "overwrite": bool(args.overwrite),
        "counts": {
            "processed_input_records": int(total_processed),
            "judged_output_records": int(total_judged),
            "judge_error_records": int(total_error_records),
            "global_target_total": int(global_target_total),
            "chunk_files_total": len(chunk_files),
            "chunk_rows": run_rows,
        },
        "deprecated_notes": [
            "Codex-as-judge flow is deprecated.",
            "operational_instruction_for_chatgpt_or_codex.md is preserved but deprecated for this pipeline.",
        ],
    }
    _write_json(round_dir / "manifest.json", manifest)

    # Validation and aggregation.
    if show_progress:
        print("", flush=True)
    validation_pass = None
    try:
        validate_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_qwen32b_judge_outputs.py"),
            "--round-dir",
            str(round_dir),
        ]
        rc = 0
        rc = int(
            __import__("subprocess").run(validate_cmd, check=False, capture_output=False).returncode
        )
        validation_pass = rc == 0
    except Exception:
        validation_pass = False

    try:
        aggregate_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "aggregate_qwen32b_judge_outputs.py"),
            "--round-dir",
            str(round_dir),
        ]
        __import__("subprocess").run(aggregate_cmd, check=False, capture_output=False)
    except Exception:
        pass

    _build_readme(round_dir, manifest, validation_pass)
    if show_progress:
        print(
            f"[judge] finished: judged={total_judged}/{max(1, global_target_total)} "
            f"errors={total_error_records} elapsed={_format_seconds(time.time() - global_start_ts)}",
            flush=True,
        )
    print(json.dumps({"round_dir": str(round_dir), "validation_pass": validation_pass}, ensure_ascii=False))


if __name__ == "__main__":
    main()
