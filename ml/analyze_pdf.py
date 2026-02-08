from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .evaluation import evaluate_clustering, evaluate_coherence, evaluate_stability
from .pipeline import (
    build_analysis_text,
    build_tfidf,
    cluster_chunks,
    extract_text,
    segment_sections,
    summarize_topics,
)
from .types import EvaluationResult, TextChunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a PDF with topic clustering + evaluation.")
    parser.add_argument("--pdf-path", required=True, help="Input PDF path.")
    parser.add_argument("--output-dir", default="output/ml", help="Directory for generated artifacts.")
    parser.add_argument("--n-clusters", type=int, default=None, help="Fixed cluster count.")
    parser.add_argument("--k-min", type=int, default=2, help="Min cluster count for auto selection.")
    parser.add_argument("--k-max", type=int, default=8, help="Max cluster count for auto selection.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--min-chunk-words",
        type=int,
        default=120,
        help="Minimum words per chunk after merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = extract_text(str(pdf_path))
    full_text = build_analysis_text(pages)
    if len(full_text.strip()) < 400:
        raise SystemExit("Extracted text is too short. The PDF may be scanned or unreadable.")

    chunks = segment_sections(full_text, min_chunk_words=args.min_chunk_words)
    if not chunks:
        raise SystemExit("No chunks produced from the extracted text.")

    matrix, terms = build_tfidf(chunks)
    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise SystemExit("Insufficient text diversity for TF-IDF/clustering.")

    if args.n_clusters is not None:
        selected_k = max(1, min(args.n_clusters, len(chunks)))
        eval_result, labels, centroids, topic_summaries = _run_single_k(
            chunks,
            matrix,
            terms,
            selected_k,
            seed=args.seed,
            candidate_scores=[],
        )
    else:
        eval_result, labels, centroids, topic_summaries = _run_auto_k(
            chunks=chunks,
            matrix=matrix,
            terms=terms,
            k_min=args.k_min,
            k_max=args.k_max,
            seed=args.seed,
        )

    _write_artifacts(
        output_dir=output_dir,
        pdf_path=pdf_path,
        pages=pages,
        chunks=chunks,
        labels=labels,
        topic_summaries=topic_summaries,
        eval_result=eval_result,
    )

    print(f"Done. Outputs written to: {output_dir.resolve()}")


def _run_auto_k(
    chunks: list[TextChunk],
    matrix: np.ndarray,
    terms: list[str],
    k_min: int,
    k_max: int,
    seed: int,
) -> tuple[EvaluationResult, np.ndarray, np.ndarray, list]:
    n_chunks = len(chunks)
    low = max(2, min(k_min, n_chunks))
    high = max(low, min(k_max, n_chunks))

    candidates: list[dict[str, float]] = []
    per_k_outputs: dict[int, tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float]]] = {}
    for k in range(low, high + 1):
        labels, centroids = cluster_chunks(matrix, n_clusters=k, seed=seed)
        cluster_metrics = evaluate_clustering(matrix, labels)
        stability = evaluate_stability(chunks, k=k, seeds=[seed, seed + 1, seed + 2])
        record = {
            "k": float(k),
            "silhouette": cluster_metrics["silhouette"],
            "davies_bouldin": cluster_metrics["davies_bouldin"],
            "calinski_harabasz": cluster_metrics["calinski_harabasz"],
            "stability_ari_mean": stability["stability_ari_mean"],
        }
        candidates.append(record)
        per_k_outputs[k] = (labels, centroids, cluster_metrics, stability)

    selected_k = _choose_best_k(candidates)
    labels, centroids, cluster_metrics, stability = per_k_outputs[selected_k]
    topic_summaries = summarize_topics(chunks, labels, centroids, matrix, terms)
    topic_terms = {topic.topic_id: topic.top_terms for topic in topic_summaries}
    coherence = evaluate_coherence(chunks, labels, topic_terms)

    eval_result = EvaluationResult(
        selected_k=selected_k,
        silhouette=cluster_metrics["silhouette"],
        davies_bouldin=cluster_metrics["davies_bouldin"],
        calinski_harabasz=cluster_metrics["calinski_harabasz"],
        stability_ari_mean=stability["stability_ari_mean"],
        coherence_score=coherence["coherence_score"],
        distinctiveness_score=coherence["distinctiveness_score"],
        candidate_scores=candidates,
    )
    return eval_result, labels, centroids, topic_summaries


def _run_single_k(
    chunks: list[TextChunk],
    matrix: np.ndarray,
    terms: list[str],
    k: int,
    seed: int,
    candidate_scores: list[dict[str, float]],
) -> tuple[EvaluationResult, np.ndarray, np.ndarray, list]:
    labels, centroids = cluster_chunks(matrix, n_clusters=k, seed=seed)
    cluster_metrics = evaluate_clustering(matrix, labels)
    stability = evaluate_stability(chunks, k=k, seeds=[seed, seed + 1, seed + 2])
    topic_summaries = summarize_topics(chunks, labels, centroids, matrix, terms)
    topic_terms = {topic.topic_id: topic.top_terms for topic in topic_summaries}
    coherence = evaluate_coherence(chunks, labels, topic_terms)

    eval_result = EvaluationResult(
        selected_k=k,
        silhouette=cluster_metrics["silhouette"],
        davies_bouldin=cluster_metrics["davies_bouldin"],
        calinski_harabasz=cluster_metrics["calinski_harabasz"],
        stability_ari_mean=stability["stability_ari_mean"],
        coherence_score=coherence["coherence_score"],
        distinctiveness_score=coherence["distinctiveness_score"],
        candidate_scores=candidate_scores,
    )
    return eval_result, labels, centroids, topic_summaries


def _choose_best_k(candidates: list[dict[str, float]]) -> int:
    if not candidates:
        return 2

    def normalize(values: list[float], invert: bool = False) -> list[float]:
        vmin, vmax = min(values), max(values)
        if abs(vmax - vmin) < 1e-12:
            out = [0.5] * len(values)
        else:
            out = [(v - vmin) / (vmax - vmin) for v in values]
        if invert:
            out = [1.0 - x for x in out]
        return out

    sil = normalize([c["silhouette"] for c in candidates])
    dbi = normalize([c["davies_bouldin"] for c in candidates], invert=True)
    stability = normalize([c["stability_ari_mean"] for c in candidates])

    scored: list[tuple[int, float]] = []
    for i, c in enumerate(candidates):
        score = 0.45 * sil[i] + 0.35 * dbi[i] + 0.20 * stability[i]
        scored.append((int(c["k"]), score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def _write_artifacts(
    output_dir: Path,
    pdf_path: Path,
    pages: list,
    chunks: list[TextChunk],
    labels: np.ndarray,
    topic_summaries: list,
    eval_result: EvaluationResult,
) -> None:
    chunk_topic_rows = []
    for idx, chunk in enumerate(chunks):
        chunk_topic_rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "word_count": chunk.word_count,
                "topic_id": int(labels[idx]),
                "text": chunk.text,
            }
        )

    topic_term_rows = []
    for topic in topic_summaries:
        for rank, term in enumerate(topic.top_terms, start=1):
            topic_term_rows.append({"topic_id": topic.topic_id, "rank": rank, "term": term})

    analysis_payload = {
        "pdf_path": str(pdf_path),
        "pages": len(pages),
        "chunks": len(chunks),
        "topics": [
            {
                "topic_id": topic.topic_id,
                "top_terms": topic.top_terms,
                "representative_chunks": topic.representative_chunks,
                "section_distribution": topic.section_distribution,
            }
            for topic in topic_summaries
        ],
    }
    (output_dir / "analysis.json").write_text(json.dumps(analysis_payload, indent=2), encoding="utf-8")
    (output_dir / "evaluation.json").write_text(json.dumps(asdict(eval_result), indent=2), encoding="utf-8")

    _write_csv(output_dir / "chunk_topics.csv", chunk_topic_rows)
    _write_csv(output_dir / "topic_terms.csv", topic_term_rows)
    _write_report(output_dir / "report.md", pdf_path, analysis_payload, eval_result)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _quality_flags(eval_result: EvaluationResult) -> dict[str, str]:
    checks = {
        "silhouette": eval_result.silhouette >= 0.05,
        "davies_bouldin": eval_result.davies_bouldin <= 2.0,
        "stability_ari_mean": eval_result.stability_ari_mean >= 0.45,
        "distinctiveness_score": eval_result.distinctiveness_score >= 0.60,
    }
    return {k: ("PASS" if ok else "WARN") for k, ok in checks.items()}


def _write_report(path: Path, pdf_path: Path, analysis: dict, eval_result: EvaluationResult) -> None:
    flags = _quality_flags(eval_result)
    topic_lines = []
    for topic in analysis["topics"]:
        topic_lines.append(
            f"- Topic {topic['topic_id']}: {', '.join(topic['top_terms'][:8])} "
            f"(repr: {', '.join(topic['representative_chunks'])})"
        )

    report = f"""# PDF Topic Analysis Report

## Input
- PDF: `{pdf_path}`
- Pages: {analysis['pages']}
- Chunks: {analysis['chunks']}
- Selected clusters (k): {eval_result.selected_k}

## Topics
{chr(10).join(topic_lines) if topic_lines else '- No topics generated.'}

## Model Evaluation
- silhouette: {eval_result.silhouette:.4f} [{flags['silhouette']}]
- davies_bouldin: {eval_result.davies_bouldin:.4f} [{flags['davies_bouldin']}]
- calinski_harabasz: {eval_result.calinski_harabasz:.4f}
- stability_ari_mean: {eval_result.stability_ari_mean:.4f} [{flags['stability_ari_mean']}]
- coherence_score: {eval_result.coherence_score:.4f}
- distinctiveness_score: {eval_result.distinctiveness_score:.4f} [{flags['distinctiveness_score']}]
"""
    path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
