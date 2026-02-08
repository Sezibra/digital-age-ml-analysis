from __future__ import annotations

import math
import re
from collections import Counter

import fitz
import numpy as np

from .types import PageText, TextChunk, TopicSummary

HEADER_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9 ,&'/-]{2,120}$")
TOKEN_RE = re.compile(r"[a-z]{2,}")

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def extract_text(pdf_path: str) -> list[PageText]:
    doc = fitz.open(pdf_path)
    pages: list[PageText] = []
    for idx in range(doc.page_count):
        raw = doc[idx].get_text("text")
        cleaned = _normalize_page_text(raw)
        pages.append(PageText(page_number=idx + 1, text=cleaned))
    return pages


def segment_sections(text: str, min_chunk_words: int = 120, target_chunk_words: int = 220) -> list[TextChunk]:
    paragraphs = _text_to_paragraphs(text)
    chunks: list[TextChunk] = []
    current_section = "Unknown"
    buffer: list[str] = []
    buffer_words = 0
    chunk_idx = 1

    def flush() -> None:
        nonlocal buffer, buffer_words, chunk_idx
        if not buffer:
            return
        combined = " ".join(buffer).strip()
        words = _count_words(combined)
        if words == 0:
            buffer = []
            buffer_words = 0
            return
        chunks.append(
            TextChunk(
                chunk_id=f"chunk_{chunk_idx:03d}",
                section=current_section,
                text=combined,
                word_count=words,
            )
        )
        chunk_idx += 1
        buffer = []
        buffer_words = 0

    for para in paragraphs:
        if _is_section_header(para):
            flush()
            current_section = para
            continue

        para_words = _count_words(para)
        if para_words == 0:
            continue

        buffer.append(para)
        buffer_words += para_words
        if buffer_words >= target_chunk_words:
            flush()

    flush()

    merged: list[TextChunk] = []
    for chunk in chunks:
        if merged and chunk.word_count < min_chunk_words:
            prev = merged[-1]
            prev.text = f"{prev.text} {chunk.text}".strip()
            prev.word_count = _count_words(prev.text)
        else:
            merged.append(chunk)
    return merged


def build_tfidf(chunks: list[TextChunk], max_features: int = 3000) -> tuple[np.ndarray, list[str]]:
    tokenized = [_tokenize(chunk.text) for chunk in chunks]
    n_docs = len(tokenized)
    if n_docs == 0:
        return np.zeros((0, 0), dtype=float), []

    df = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    min_df = 2 if n_docs > 6 else 1
    terms = [t for t, freq in df.items() if freq >= min_df]
    if not terms:
        terms = [t for t, _ in df.most_common(max_features)]
    else:
        terms.sort(key=lambda t: (-df[t], t))
        terms = terms[:max_features]

    vocab = {term: i for i, term in enumerate(terms)}
    matrix = np.zeros((n_docs, len(terms)), dtype=float)

    for i, tokens in enumerate(tokenized):
        tf = Counter(t for t in tokens if t in vocab)
        if not tf:
            continue
        for term, freq in tf.items():
            matrix[i, vocab[term]] = float(freq)

    idf = np.zeros(len(terms), dtype=float)
    for term, idx in vocab.items():
        idf[idx] = math.log((1 + n_docs) / (1 + df[term])) + 1.0
    matrix = matrix * idf

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix = matrix / norms
    return matrix, terms


def cluster_chunks(
    matrix: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return np.zeros(0, dtype=int), np.zeros((0, 0), dtype=float)
    n_samples = matrix.shape[0]
    n_clusters = max(1, min(n_clusters, n_samples))
    rng = np.random.default_rng(seed)
    centroid_idx = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = matrix[centroid_idx].copy()

    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        sim = matrix @ centroids.T
        new_labels = np.argmax(sim, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for k in range(n_clusters):
            members = matrix[labels == k]
            if len(members) == 0:
                centroids[k] = matrix[rng.integers(0, n_samples)]
            else:
                centroid = members.mean(axis=0)
                norm = np.linalg.norm(centroid)
                centroids[k] = centroid if norm == 0 else centroid / norm

    return labels, centroids


def summarize_topics(
    chunks: list[TextChunk],
    labels: np.ndarray,
    centroids: np.ndarray,
    matrix: np.ndarray,
    terms: list[str],
    top_n_terms: int = 10,
    top_n_chunks: int = 3,
) -> list[TopicSummary]:
    summaries: list[TopicSummary] = []
    if len(chunks) == 0:
        return summaries

    unique_labels = sorted(set(int(x) for x in labels.tolist()))
    for label in unique_labels:
        member_idx = np.where(labels == label)[0]
        if len(member_idx) == 0:
            continue

        centroid = centroids[label]
        top_term_idx = np.argsort(-centroid)[:top_n_terms]
        top_terms = [terms[i] for i in top_term_idx if i < len(terms)]

        sims = matrix[member_idx] @ centroid
        order = np.argsort(-sims)[:top_n_chunks]
        repr_chunks = [chunks[member_idx[i]].chunk_id for i in order]

        section_dist: dict[str, int] = {}
        for i in member_idx:
            section = chunks[i].section
            section_dist[section] = section_dist.get(section, 0) + 1

        summaries.append(
            TopicSummary(
                topic_id=label,
                top_terms=top_terms,
                representative_chunks=repr_chunks,
                section_distribution=section_dist,
            )
        )
    return summaries


def build_analysis_text(pages: list[PageText]) -> str:
    return "\n\n".join(page.text for page in pages if page.text.strip())


def _normalize_page_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.replace("\r", "\n")
    return text.strip()


def _text_to_paragraphs(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _is_section_header(line: str) -> bool:
    return bool(HEADER_RE.match(line.strip()))


def _count_words(text: str) -> int:
    return len(TOKEN_RE.findall(text.lower()))


def _tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return [token for token in tokens if token not in STOPWORDS]
