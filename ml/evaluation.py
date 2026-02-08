from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict

import numpy as np

from .pipeline import build_tfidf, cluster_chunks, _tokenize
from .types import TextChunk


def evaluate_clustering(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if len(labels) <= 1 or len(set(labels.tolist())) <= 1:
        return {"silhouette": 0.0, "davies_bouldin": 999.0, "calinski_harabasz": 0.0}
    return {
        "silhouette": float(_silhouette_score(matrix, labels)),
        "davies_bouldin": float(_davies_bouldin_score(matrix, labels)),
        "calinski_harabasz": float(_calinski_harabasz_score(matrix, labels)),
    }


def evaluate_stability(
    chunks: list[TextChunk],
    k: int,
    seeds: list[int],
) -> dict[str, float]:
    matrix, _ = build_tfidf(chunks)
    if matrix.shape[0] == 0:
        return {"stability_ari_mean": 0.0, "stability_ari_std": 0.0}
    if len(seeds) < 2:
        return {"stability_ari_mean": 1.0, "stability_ari_std": 0.0}

    all_labels: list[np.ndarray] = []
    for seed in seeds:
        labels, _ = cluster_chunks(matrix, n_clusters=k, seed=seed)
        all_labels.append(labels)

    scores: list[float] = []
    for a, b in itertools.combinations(all_labels, 2):
        scores.append(adjusted_rand_index(a, b))

    if not scores:
        return {"stability_ari_mean": 1.0, "stability_ari_std": 0.0}
    return {
        "stability_ari_mean": float(np.mean(scores)),
        "stability_ari_std": float(np.std(scores)),
    }


def evaluate_coherence(
    chunks: list[TextChunk],
    labels: np.ndarray,
    topic_terms: dict[int, list[str]],
) -> dict[str, float]:
    if len(chunks) == 0 or len(topic_terms) == 0:
        return {"coherence_score": 0.0, "distinctiveness_score": 0.0}

    tokens_by_chunk = [set(_tokenize(chunk.text)) for chunk in chunks]
    coherence_vals: list[float] = []
    term_sets: list[set[str]] = []

    for topic_id, terms in topic_terms.items():
        term_set = set(terms)
        term_sets.append(term_set)
        member_idx = np.where(labels == topic_id)[0]
        if len(member_idx) == 0:
            continue
        member_tokens = [tokens_by_chunk[i] for i in member_idx]

        term_scores: list[float] = []
        for t1, t2 in itertools.combinations(terms[:10], 2):
            d12 = sum(1 for token_set in member_tokens if t1 in token_set and t2 in token_set)
            d1 = sum(1 for token_set in member_tokens if t1 in token_set)
            if d1 == 0:
                continue
            term_scores.append(math.log((d12 + 1) / d1))
        if term_scores:
            coherence_vals.append(float(np.mean(term_scores)))

    coherence = float(np.mean(coherence_vals)) if coherence_vals else 0.0

    jaccard_scores: list[float] = []
    for a, b in itertools.combinations(term_sets, 2):
        union = len(a | b)
        if union == 0:
            continue
        jaccard_scores.append(len(a & b) / union)
    distinctiveness = 1.0 - (float(np.mean(jaccard_scores)) if jaccard_scores else 0.0)

    return {
        "coherence_score": coherence,
        "distinctiveness_score": max(0.0, min(1.0, distinctiveness)),
    }


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return 0.0
    n = len(labels_a)

    contingency: dict[tuple[int, int], int] = defaultdict(int)
    counts_a: Counter[int] = Counter(int(x) for x in labels_a.tolist())
    counts_b: Counter[int] = Counter(int(x) for x in labels_b.tolist())
    for la, lb in zip(labels_a.tolist(), labels_b.tolist()):
        contingency[(int(la), int(lb))] += 1

    sum_nij = sum(_comb2(v) for v in contingency.values())
    sum_ai = sum(_comb2(v) for v in counts_a.values())
    sum_bj = sum(_comb2(v) for v in counts_b.values())
    total = _comb2(n)
    if total == 0:
        return 0.0

    expected = (sum_ai * sum_bj) / total
    max_index = 0.5 * (sum_ai + sum_bj)
    denom = max_index - expected
    if denom == 0:
        return 0.0
    return float((sum_nij - expected) / denom)


def _comb2(n: int) -> float:
    return n * (n - 1) / 2.0


def _pairwise_cosine_distance(matrix: np.ndarray) -> np.ndarray:
    sim = np.clip(matrix @ matrix.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def _silhouette_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    dist = _pairwise_cosine_distance(matrix)
    n = len(labels)
    unique = sorted(set(labels.tolist()))
    sil = np.zeros(n, dtype=float)

    for i in range(n):
        own = labels[i]
        own_idx = np.where(labels == own)[0]
        if len(own_idx) <= 1:
            sil[i] = 0.0
            continue
        a = float(np.mean([dist[i, j] for j in own_idx if j != i]))

        b = float("inf")
        for other in unique:
            if other == own:
                continue
            other_idx = np.where(labels == other)[0]
            if len(other_idx) == 0:
                continue
            b = min(b, float(np.mean(dist[i, other_idx])))
        if not np.isfinite(b):
            sil[i] = 0.0
        else:
            sil[i] = (b - a) / max(a, b, 1e-12)
    return float(np.mean(sil))


def _davies_bouldin_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique = sorted(set(labels.tolist()))
    centroids: dict[int, np.ndarray] = {}
    scatters: dict[int, float] = {}

    for c in unique:
        members = matrix[labels == c]
        centroid = members.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[c] = centroid
        if len(members) == 0:
            scatters[c] = 0.0
        else:
            d = 1.0 - np.clip(members @ centroid, -1.0, 1.0)
            scatters[c] = float(np.mean(d))

    r_vals: list[float] = []
    for i in unique:
        worst = 0.0
        for j in unique:
            if i == j:
                continue
            m = 1.0 - float(np.clip(np.dot(centroids[i], centroids[j]), -1.0, 1.0))
            if m <= 1e-12:
                continue
            r = (scatters[i] + scatters[j]) / m
            if r > worst:
                worst = r
        r_vals.append(worst)
    return float(np.mean(r_vals)) if r_vals else 999.0


def _calinski_harabasz_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique = sorted(set(labels.tolist()))
    n_samples = matrix.shape[0]
    k = len(unique)
    if n_samples <= k or k <= 1:
        return 0.0

    overall = matrix.mean(axis=0)
    norm = np.linalg.norm(overall)
    if norm > 0:
        overall = overall / norm

    bss = 0.0
    wss = 0.0
    for c in unique:
        members = matrix[labels == c]
        if len(members) == 0:
            continue
        centroid = members.mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm
        bss += len(members) * float(np.sum((centroid - overall) ** 2))
        wss += float(np.sum((members - centroid) ** 2))
    if wss <= 1e-12:
        return 0.0
    return float((bss / (k - 1)) / (wss / (n_samples - k)))
