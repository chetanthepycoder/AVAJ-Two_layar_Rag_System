from __future__ import annotations

import math
import re
from hashlib import sha256
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from config import AppSettings, get_settings
from ingestion import ChildChunk, IngestionPipeline


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class RetrievalCandidate:
    child: ChildChunk
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    fused_score: float = 0.0
    rerank_score: float = 0.0
    passed_threshold: bool = True
    retrieval_channels: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.child.id


class HybridSearch:
    def __init__(self, settings: Optional[AppSettings] = None, pipeline: Optional[IngestionPipeline] = None):
        self.settings = settings or get_settings()
        self.pipeline = pipeline or IngestionPipeline(self.settings)
        self._embedding_model = None
        self._chroma_collection = None
        self._rank_bm25 = None
        self._bm25_corpus: List[List[str]] = []
        self._children = self.pipeline.all_children()
        self._try_build_chroma()
        self._try_build_bm25()

    def refresh(self) -> None:
        self._children = self.pipeline.all_children()
        self._try_build_chroma()
        self._try_build_bm25()

    def search(
        self,
        queries: Sequence[str],
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        query_weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, object]:
        top_k = top_k or self.settings.top_k
        alpha = self.settings.hybrid_alpha if alpha is None else alpha
        weighted_queries = _normalise_weighted_queries(queries, query_weights)
        dense_rankings = self._dense_search(weighted_queries, top_k)
        sparse_rankings = self._sparse_search(weighted_queries, top_k)
        fused = self._rrf(dense_rankings, sparse_rankings, alpha)
        candidates = [candidate for candidate in fused[: max(top_k, 1)]]
        return {
            "candidates": candidates,
            "stats": {
                "dense_hits": len(dense_rankings),
                "sparse_hits": len(sparse_rankings),
                "fused_candidates": len(fused),
                "total_children": len(self._children),
                "query_weights": [weight for _query, weight in weighted_queries],
            },
        }

    def _dense_search(self, queries: Sequence[tuple[str, float]], top_k: int) -> List[RetrievalCandidate]:
        if not self._children:
            return []
        if self._chroma_collection is not None and self._embedding_model is not None:
            try:
                found: Dict[str, RetrievalCandidate] = {}
                query_text = _weighted_query_text(queries)
                query_embedding = self._embedding_model.encode([query_text], normalize_embeddings=True).tolist()[0]
                result = self._chroma_collection.query(query_embeddings=[query_embedding], n_results=top_k)
                ids = result.get("ids", [[]])[0]
                for rank, child_id in enumerate(ids, start=1):
                    child = next((item for item in self._children if item.id == child_id), None)
                    if child:
                        found[child_id] = RetrievalCandidate(child=child, dense_rank=rank, retrieval_channels=["dense"])
                return list(found.values())
            except Exception:
                pass
        return self._fallback_dense(queries, top_k)

    def _fallback_dense(self, queries: Sequence[tuple[str, float]], top_k: int) -> List[RetrievalCandidate]:
        query_tokens = _weighted_tokens(queries)
        scored = []
        for child in self._children:
            child_tokens = set(_tokenize(child.text))
            score = _weighted_jaccard(query_tokens, child_tokens)
            if score > 0:
                scored.append((score, child))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievalCandidate(child=child, dense_rank=rank, retrieval_channels=["dense"])
            for rank, (_, child) in enumerate(scored[:top_k], start=1)
        ]

    def _sparse_search(self, queries: Sequence[tuple[str, float]], top_k: int) -> List[RetrievalCandidate]:
        if not self._children:
            return []
        tokens = _weighted_query_tokens(queries)
        if self._rank_bm25 is not None:
            scores = self._rank_bm25.get_scores(tokens)
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        else:
            ranked = [(idx, _keyword_score(tokens, self._bm25_corpus[idx])) for idx in range(len(self._children))]
            ranked.sort(key=lambda item: item[1], reverse=True)
        candidates = []
        for rank, (idx, score) in enumerate(ranked[:top_k], start=1):
            if score <= 0:
                continue
            candidates.append(
                RetrievalCandidate(child=self._children[idx], sparse_rank=rank, retrieval_channels=["sparse"])
            )
        return candidates

    def _rrf(
        self,
        dense: Iterable[RetrievalCandidate],
        sparse: Iterable[RetrievalCandidate],
        alpha: float,
        k: int = 60,
    ) -> List[RetrievalCandidate]:
        merged: Dict[str, RetrievalCandidate] = {}
        for candidate in dense:
            current = merged.setdefault(candidate.id, candidate)
            current.dense_rank = candidate.dense_rank
            if "dense" not in current.retrieval_channels:
                current.retrieval_channels.append("dense")
            current.fused_score += alpha * (1 / (k + (candidate.dense_rank or k)))
        for candidate in sparse:
            current = merged.setdefault(candidate.id, candidate)
            current.sparse_rank = candidate.sparse_rank
            if "sparse" not in current.retrieval_channels:
                current.retrieval_channels.append("sparse")
            current.fused_score += (1 - alpha) * (1 / (k + (candidate.sparse_rank or k)))
        return sorted(merged.values(), key=lambda item: item.fused_score, reverse=True)

    def _try_build_bm25(self) -> None:
        self._bm25_corpus = [_tokenize(child.text) for child in self._children]
        try:
            from rank_bm25 import BM25Okapi

            self._rank_bm25 = BM25Okapi(self._bm25_corpus) if self._bm25_corpus else None
        except Exception:
            self._rank_bm25 = None

    def _try_build_chroma(self) -> None:
        self._chroma_collection = None
        self._embedding_model = None
        if not self.settings.enable_chroma or not self._children:
            return
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
            client = chromadb.PersistentClient(path=str(self.settings.data_dir / "chroma"))
            # Embedding dimensions and vector spaces vary by model. A model-specific collection
            # prevents old vectors from being mixed with vectors produced by a new model.
            collection = client.get_or_create_collection(_embedding_collection_name(self.settings.embedding_model))
            existing = set(collection.get(include=[])["ids"])
            missing = [child for child in self._children if child.id not in existing]
            if missing:
                embeddings = self._embedding_model.encode(
                    [child.text for child in missing],
                    normalize_embeddings=True,
                    batch_size=max(1, self.settings.embedding_batch_size),
                ).tolist()
                collection.add(
                    ids=[child.id for child in missing],
                    documents=[child.text for child in missing],
                    metadatas=[
                        {
                            "parent_id": child.parent_id,
                            "source_name": child.source_name,
                            "document_hash": child.document_hash,
                        }
                        for child in missing
                    ],
                    embeddings=embeddings,
                )
            self._chroma_collection = collection
        except Exception:
            self._chroma_collection = None
            self._embedding_model = None


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _embedding_collection_name(model_name: str) -> str:
    """Return a stable, Chroma-safe namespace for one embedding model."""
    digest = sha256(model_name.encode("utf-8")).hexdigest()[:12]
    return f"avaj_dense_{digest}"


def _normalise_weighted_queries(
    queries: Sequence[str], query_weights: Optional[Sequence[float]],
) -> list[tuple[str, float]]:
    weights = list(query_weights or [])
    return [
        (query, max(0.0, float(weights[index])) if index < len(weights) else 1.0)
        for index, query in enumerate(queries)
        if query and query.strip()
    ]


def _weighted_query_text(queries: Sequence[tuple[str, float]]) -> str:
    return " ".join(query for query, weight in queries for _ in range(max(1, round(weight))))


def _weighted_query_tokens(queries: Sequence[tuple[str, float]]) -> List[str]:
    return [token for query, weight in queries for token in _tokenize(query) for _ in range(max(1, round(weight)))]


def _weighted_tokens(queries: Sequence[tuple[str, float]]) -> Dict[str, float]:
    tokens: Dict[str, float] = {}
    for query, weight in queries:
        for token in _tokenize(query):
            tokens[token] = tokens.get(token, 0.0) + weight
    return tokens


def _weighted_jaccard(query_tokens: Dict[str, float], child_tokens: set[str]) -> float:
    if not query_tokens or not child_tokens:
        return 0.0
    overlap = sum(weight for token, weight in query_tokens.items() if token in child_tokens)
    union = sum(query_tokens.values()) + len(child_tokens - set(query_tokens))
    return overlap / union if union else 0.0


def _keyword_score(query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    frequencies: Dict[str, int] = {}
    for token in doc_tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    return sum(math.log(1 + frequencies.get(token, 0)) for token in query_tokens)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
