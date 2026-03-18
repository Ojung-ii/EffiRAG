from __future__ import annotations

from .config import RetrievalConfig
from .registry import register_method
from .retrieval import run_graphrag_core
from .types import RetrievalResult, Sample


@register_method("naive_graphrag")
def run_naive_graphrag(sample: Sample, cfg: RetrievalConfig) -> RetrievalResult:
    # Same retrieval structure as EffiRAG, but without stable run-wise seed selection
    # and without corridor trimming.
    return run_graphrag_core(
        sample=sample,
        cfg=cfg,
        method_name="naive_graphrag",
        stable_seed_selection=False,
        enable_trim=False,
    )
