from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .eval_metrics import MetricRegistry, MetricSpec
from .utils import markdown_table


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _format_cell(value: Any, spec: Optional[MetricSpec]) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        txt = value.strip()
        return txt if txt else "-"
    if isinstance(value, bool):
        return "1" if value else "0"

    if spec is None:
        return str(value)
    try:
        num = float(value)
    except Exception:
        txt = str(value).strip()
        return txt if txt else "-"
    if spec.fmt:
        try:
            return format(num, spec.fmt)
        except Exception:
            return f"{num:.4f}"
    return f"{num:.4f}"


def _sort_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = [dict(r or {}) for r in list(records or [])]
    rows.sort(key=lambda x: (str(x.get("dataset", "")), str(x.get("variant", ""))))
    return rows


def group_markdown_tables(
    *,
    records: Sequence[Mapping[str, Any]],
    registry: MetricRegistry,
    leading_columns: Sequence[str] = ("dataset", "variant"),
) -> Dict[str, str]:
    rows = _sort_records(records)
    out: Dict[str, str] = {}
    for group in registry.groups():
        specs = list(registry.enabled_by_group(group))
        headers = list(leading_columns) + [spec.name for spec in specs]
        table_rows: List[List[str]] = []
        for row in rows:
            one: List[str] = []
            for key in leading_columns:
                one.append(_format_cell(row.get(key), None))
            for spec in specs:
                one.append(_format_cell(row.get(spec.name), spec))
            table_rows.append(one)
        out[group] = markdown_table(headers, table_rows)
    return out


def markdown_document_from_group_tables(tables: Mapping[str, str], title: str = "Context Efficiency Audit") -> str:
    lines: List[str] = [f"# {title}", ""]
    for group, table in dict(tables or {}).items():
        lines.append(f"## {group}")
        lines.append("")
        lines.append(str(table))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_markdown(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(text or ""), encoding="utf-8")


def write_lines(path: str, lines: Iterable[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join([str(x) for x in list(lines or [])]) + "\n", encoding="utf-8")


def build_group_table_markdown(
    *,
    records: Sequence[Mapping[str, Any]],
    registry: MetricRegistry,
    groups: Sequence[str],
    leading_columns: Sequence[str] = ("dataset", "variant"),
) -> str:
    rows = _sort_records(records)
    lines: List[str] = []
    for group in list(groups or []):
        specs = list(registry.enabled_by_group(group))
        if not specs:
            continue
        headers = list(leading_columns) + [spec.name for spec in specs]
        table_rows: List[List[str]] = []
        for row in rows:
            one: List[str] = []
            for key in leading_columns:
                one.append(_format_cell(row.get(key), None))
            for spec in specs:
                one.append(_format_cell(row.get(spec.name), spec))
            table_rows.append(one)
        lines.append(f"## {group}")
        lines.append("")
        lines.append(markdown_table(headers, table_rows))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_flat_markdown_table(
    *,
    records: Sequence[Mapping[str, Any]],
    headers: Sequence[str],
    registry: Optional[MetricRegistry] = None,
) -> str:
    rows = _sort_records(records)
    table_rows: List[List[str]] = []
    for row in rows:
        one: List[str] = []
        for h in list(headers or []):
            spec = registry.get(h) if registry is not None else None
            one.append(_format_cell(row.get(h), spec))
        table_rows.append(one)
    return markdown_table(list(headers or []), table_rows)
