from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class TextChunk:
    chunk_id: str
    section: str
    text: str
    word_count: int


@dataclass
class TopicSummary:
    topic_id: int
    top_terms: list[str]
    representative_chunks: list[str]
    section_distribution: dict[str, int]


@dataclass
class EvaluationResult:
    selected_k: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    stability_ari_mean: float
    coherence_score: float
    distinctiveness_score: float
    candidate_scores: list[dict[str, float]] = field(default_factory=list)
