from __future__ import annotations

import unittest

import numpy as np

from ml.evaluation import evaluate_clustering, evaluate_coherence, evaluate_stability
from ml.pipeline import (
    _is_section_header,
    build_tfidf,
    cluster_chunks,
    segment_sections,
    summarize_topics,
)
from ml.types import TextChunk


class PipelineTests(unittest.TestCase):
    def test_section_header_detection(self) -> None:
        self.assertTrue(_is_section_header("1. Introduction"))
        self.assertTrue(_is_section_header("3.1. Data method and description"))
        self.assertFalse(_is_section_header("this is not a header"))

    def test_segmentation_produces_chunks(self) -> None:
        text = "\n\n".join(
            [
                "1. Introduction",
                "Digital government improves transparency and public services. " * 20,
                "2. Method",
                "This study applies text mining and production-network analysis. " * 20,
            ]
        )
        chunks = segment_sections(text, min_chunk_words=40, target_chunk_words=80)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any(chunk.section.startswith("1.") for chunk in chunks))

    def test_clustering_and_evaluation_metrics(self) -> None:
        chunks = [
            TextChunk("c1", "1. Intro", "digital government policy reform public services data platform", 9),
            TextChunk("c2", "1. Intro", "government platform digital services administrative reform", 7),
            TextChunk("c3", "2. Method", "model regression variable robustness empirical estimation", 7),
            TextChunk("c4", "2. Method", "estimation model robustness test variable controls", 7),
        ]
        matrix, terms = build_tfidf(chunks)
        labels, centroids = cluster_chunks(matrix, n_clusters=2, seed=42)
        metrics = evaluate_clustering(matrix, labels)
        self.assertIn("silhouette", metrics)
        self.assertIn("davies_bouldin", metrics)
        self.assertIn("calinski_harabasz", metrics)

        topics = summarize_topics(chunks, labels, centroids, matrix, terms)
        topic_terms = {topic.topic_id: topic.top_terms for topic in topics}
        coherence = evaluate_coherence(chunks, labels, topic_terms)
        self.assertIn("coherence_score", coherence)
        self.assertIn("distinctiveness_score", coherence)

        stability = evaluate_stability(chunks, k=2, seeds=[42, 43, 44])
        self.assertIn("stability_ari_mean", stability)
        self.assertGreaterEqual(stability["stability_ari_mean"], -1.0)
        self.assertLessEqual(stability["stability_ari_mean"], 1.0)

    def test_cluster_size_bounds(self) -> None:
        matrix = np.array([[1.0, 0.0], [0.9, 0.1]])
        labels, _ = cluster_chunks(matrix, n_clusters=10, seed=1)
        self.assertEqual(len(labels), 2)


if __name__ == "__main__":
    unittest.main()
