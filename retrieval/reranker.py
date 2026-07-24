from __future__ import annotations

from typing import List, Optional

from config import AppSettings, get_settings

from .hybrid_search import RetrievalCandidate, _tokenize


STOPWORDS = {
    "about",
    "and",
    "are",
    "for",
    "from",
    "give",
    "list",
    "me",
    "of",
    "out",
    "please",
    "show",
    "tell",
    "the",
    "to",
    "what",
    "which",
    "with",
}


class Reranker:
    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or get_settings()
        self._model = None
        if self.settings.enable_reranker:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.settings.reranker_model)
            except Exception:
                self._model = None

    def rerank(self, query: str, candidates: List[RetrievalCandidate], cutoff: Optional[float] = None) -> List[RetrievalCandidate]:
        cutoff = self.settings.rerank_cutoff if cutoff is None else cutoff
        if not candidates:
            return []
        if self._model is not None:
            try:
                pairs = [(query, candidate.child.text) for candidate in candidates]
                scores = self._model.predict(pairs)
                for candidate, score in zip(candidates, scores):
                    candidate.rerank_score = float(score)
                    candidate.passed_threshold = candidate.rerank_score >= cutoff
                return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)
            except Exception:
                pass
        query_tokens: set[str] = set()
        for token in _tokenize(query):
            if _is_content_token(token):
                query_tokens.update(_expanded_query_tokens(token))
        for candidate in candidates:
            child_tokens = set(_tokenize(candidate.child.text))
            overlap = len(query_tokens & child_tokens)
            density = overlap / max(len(query_tokens), 1)
            phrase_bonus = 1.0 if query.lower() in candidate.child.text.lower() else 0.0
            candidate.rerank_score = float(overlap + density + phrase_bonus) if overlap else -10.0
            candidate.passed_threshold = candidate.rerank_score >= max(cutoff, 0.01)
        return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)


def _is_content_token(token: str) -> bool:
    if token in STOPWORDS:
        return False
    return len(token) > 2 or token.isdigit() or token.lower() in {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}


def _expanded_query_tokens(token: str) -> set[str]:
    number_to_roman = {
        "1": "i",
        "2": "ii",
        "3": "iii",
        "4": "iv",
        "5": "v",
        "6": "vi",
        "7": "vii",
        "8": "viii",
        "9": "ix",
        "10": "x",
    }
    lowered = token.lower()
    expanded = {lowered}
    if lowered in number_to_roman:
        expanded.add(number_to_roman[lowered])
    return expanded
