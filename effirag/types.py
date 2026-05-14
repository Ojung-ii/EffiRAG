from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ContextDocument:
    title: str
    sentences: List[str]


@dataclass
class Sample:
    qid: str
    question: str
    answer: str
    contexts: List[ContextDocument]
    supporting_facts: List[Tuple[str, int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnchorResult:
    anchor: str
    scores: Dict[str, float]
    top_candidates: List[str]
    sample_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    sample_id: str
    method: str
    anchors: List[str]
    seeds: List[str]
    selected_nodes: List[str]
    selected_sentence_ids: List[str]
    selected_sentences: List[str]
    candidate_sentence_ids: List[str] = field(default_factory=list)
    candidate_sentences: List[str] = field(default_factory=list)
    corridors: List[Dict[str, Any]] = field(default_factory=list)
    anchor_results: List[AnchorResult] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class RenderedContext:
    sample_id: str
    method: str
    text: str
    sentences: List[str]
    sentence_ids: List[str]
    truncated: bool
    render_mode: str = "flat"
    rendered_corridor_ids: List[str] = field(default_factory=list)
    truncated_corridor_count: int = 0
    truncated_sentence_count: int = 0
    retrieval_selected_sentence_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    sample_id: str
    generator: str
    model_name: str
    prediction: str
    raw_text: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    sample_id: str
    retrieval: RetrievalResult
    rendered: Optional[RenderedContext]
    generation: Optional[GenerationResult]
    metrics: Dict[str, float]
    efficiency: Dict[str, float]
