from __future__ import annotations

import re
from typing import Dict, Iterator, List, Optional

from config import AppSettings, get_settings
from ingestion import IngestionPipeline, ParentChunk
from observability import AIWorkLog, StageTimer
from providers import GroqProvider, OllamaProvider
from retrieval import HybridSearch, QueryExpansionService, Reranker


class RAGEngine:
    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or get_settings()
        self.pipeline = IngestionPipeline(self.settings)
        self.ollama = OllamaProvider(self.settings)
        self.groq = GroqProvider(self.settings)
        self.expander = QueryExpansionService(self.settings, self.ollama)
        self.searcher = HybridSearch(self.settings, self.pipeline)
        self.reranker = Reranker(self.settings)

    def refresh(self) -> None:
        self.searcher.refresh()

    def ask(self, query: str) -> Iterator[Dict[str, object]]:
        log = AIWorkLog(query=query)

        with StageTimer(log, "layer1_ms"):
            layer1 = self.ollama.prompt_engineer(query, timeout=self.settings.layer1_timeout_seconds)
        log.layer1_output = layer1
        log.layer1_provider = str(layer1.get("provider", ""))
        if not log.layer1_provider.startswith("ollama"):
            log.add_fallback(f"Layer 1 used fallback: {log.layer1_provider}")

        if not layer1.get("is_npgc_related", True):
            yield {
                "type": "token",
                "token": (
                    "Main AVAJ hoon, National PG College ka official AI assistant. "
                    "Meri help sirf NPGC se related queries ke liye hai. "
                    "Kya aap NPGC ke baare mein kuch poochna chahte hain?"
                ),
            }
            log.safe_reasoning_summary = "Layer 1 classified the request as outside the NPGC domain."
            yield {"type": "log", "log": log.to_dict()}
            return

        with StageTimer(log, "query_expansion_ms"):
            expanded = self.expander.expand(query, layer1, log)
        optimized_query = str(layer1.get("optimized_query") or query)
        search_hints = {str(hint).strip().lower() for hint in layer1.get("search_hints", []) if str(hint).strip()}
        query_weights = [self.settings.search_hint_boost if item.lower() in search_hints else 1.0 for item in expanded]

        with StageTimer(log, "retrieval_ms"):
            search_result = self.searcher.search(
                expanded, self.settings.top_k, self.settings.hybrid_alpha, query_weights=query_weights
            )
            candidates = search_result["candidates"]  # type: ignore[assignment]
            log.retrieval_stats = search_result["stats"]  # type: ignore[assignment]
            log.fused_candidate_ids = [candidate.id for candidate in candidates]  # type: ignore[attr-defined]

        with StageTimer(log, "rerank_ms"):
            ranked = self.reranker.rerank(optimized_query, candidates, self.settings.rerank_cutoff)  # type: ignore[arg-type]
            passed = _select_reliable_candidates(ranked, self.settings.rerank_cutoff)
            log.rerank_ledger = [
                {
                    "child_id": candidate.id,
                    "source": candidate.child.source_name,
                    "dense_rank": candidate.dense_rank,
                    "sparse_rank": candidate.sparse_rank,
                    "fused_score": round(candidate.fused_score, 6),
                    "rerank_score": round(candidate.rerank_score, 4),
                    "passed": candidate in passed,
                }
                for candidate in ranked
            ]
        log.low_confidence_retrieval = not passed

        top_context_candidates = passed[:2]
        parents = self._parents_for_candidates(passed[: self.settings.max_context_parent_chunks])
        log.selected_parent_chunks = [
            {
                "parent_id": parent.id,
                "source": parent.source_name,
                "ordinal": parent.ordinal,
                "preview": parent.text[:500],
            }
            for parent in parents
        ]

        context_payloads = self._context_payloads(optimized_query, top_context_candidates)
        input_chars = sum(len(item["content"]) for item in context_payloads)
        with StageTimer(log, "layer2_ms"):
            layer2 = self.ollama.collect_content(
                original_query=query,
                optimized_query=optimized_query,
                intent=str(layer1.get("intent", "")),
                answer_format=str(layer1.get("answer_format", "direct")),
                domain=str(layer1.get("domain", "general")),
                rag_context=context_payloads,
                low_confidence=log.low_confidence_retrieval,
                timeout=self.settings.layer2_timeout_seconds,
            )
        log.layer2_output = layer2
        log.layer2_provider = str(layer2.get("provider", ""))
        human_context = str(layer2.get("structured_context", ""))
        log.evidence_extraction = {
            "input_characters": input_chars,
            "output_characters": len(human_context),
            "reduction_ratio": round(1 - (len(human_context) / max(input_chars, 1)), 4),
            "provider": log.layer2_provider,
            "selected_context_count": len(context_payloads),
            "preview": human_context[:1000],
        }
        if not log.layer2_provider.startswith("ollama"):
            log.add_fallback(f"Layer 2 used fallback: {log.layer2_provider}")

        generation_payload = _build_3layer_payload(query, layer1, layer2)
        with StageTimer(log, "layer3_ms"):
            answer_parts: List[str] = []
            for token in self._stream_answer(generation_payload, log):
                answer_parts.append(token)
                yield {"type": "token", "token": token}
            log.token_counts["output_characters"] = len("".join(answer_parts))
        log.safe_reasoning_summary = self._reasoning_summary(log)
        yield {"type": "log", "log": log.to_dict()}

    def _parents_for_candidates(self, candidates: list) -> List[ParentChunk]:
        parents: List[ParentChunk] = []
        seen = set()
        for candidate in candidates:
            if candidate.child.parent_id in seen:
                continue
            parent = self.pipeline.get_parent(candidate.child.parent_id)
            if parent:
                seen.add(parent.id)
                parents.append(parent)
        return parents

    def _context_payloads(self, query: str, candidates: list) -> list[dict]:
        payloads: list[dict] = []
        seen_parents = set()
        for candidate in candidates:
            parent_id = candidate.child.parent_id
            if parent_id in seen_parents:
                continue
            parent = self.pipeline.get_parent(parent_id)
            source_text = parent.text if parent else candidate.child.text
            content = _extract_relevant_evidence(query, source_text, min(3500, self.settings.max_context_characters))
            payloads.append(
                {
                    "rank": len(payloads) + 1,
                    "source": candidate.child.source_name,
                    "confidence": float(candidate.rerank_score),
                    "content": content,
                    "child_id": candidate.child.id,
                    "parent_id": parent_id,
                }
            )
            seen_parents.add(parent_id)
            if len(payloads) >= 2:
                break
        return payloads

    def _stream_answer(self, generation_payload: dict, log: AIWorkLog) -> Iterator[str]:
        try:
            yield from self.groq.stream_from_preprocessed(generation_payload)
            return
        except Exception as exc:
            log.add_fallback(str(exc))

        prompt = (
            "Answer using only this structured RAG payload. Include citations from source labels. "
            "If evidence is insufficient, say that clearly.\n\n"
            f"{generation_payload}"
        )
        try:
            yield from self.ollama.stream(prompt)
            return
        except Exception as exc:
            log.add_fallback(f"Ollama generation unavailable: {exc}")
        yield self._offline_answer(
            str(generation_payload.get("query", "")),
            str(generation_payload.get("human_readable_context", "")),
            bool(generation_payload.get("low_confidence_retrieval", False)),
        )

    @staticmethod
    def _offline_answer(query: str, evidence: str, low_confidence: bool) -> str:
        if low_confidence:
            return (
                f"I could not find sufficiently reliable information in the indexed documents regarding "
                f"{query}. The most relevant available evidence is:\n\n{evidence[:3000]}"
            )
        return (
            "I found relevant indexed evidence, but no generation provider is currently available. "
            "Here are the most relevant excerpts:\n\n"
            f"{evidence[:3000]}"
        )

    @staticmethod
    def _reasoning_summary(log: AIWorkLog) -> str:
        if log.low_confidence_retrieval:
            return "Retrieval returned no chunks above the configured confidence threshold."
        source_count = len(log.selected_parent_chunks)
        fallback_count = len(log.provider_fallback_events)
        return f"Answered from {source_count} selected parent chunk(s); provider fallback events: {fallback_count}."


def _build_3layer_payload(query: str, layer1: dict, layer2: dict) -> dict:
    """Combine the two Ollama layer outputs into the stable Groq input contract."""
    return {
        "original_query": query,
        "query": str(layer1.get("optimized_query") or query),
        "intent": str(layer1.get("intent") or ""),
        "query_intent": str(layer1.get("intent") or ""),
        "answer_format": str(layer1.get("answer_format") or "direct"),
        "domain": str(layer1.get("domain") or "general"),
        "language": str(layer1.get("language") or "hinglish"),
        "human_readable_context": str(layer2.get("structured_context") or ""),
        "retrieval_used": bool(layer2.get("retrieval_used", False)),
        "low_confidence_retrieval": str(layer2.get("coverage") or "none") == "none",
        "coverage": str(layer2.get("coverage") or "none"),
        "missing_info": str(layer2.get("missing_info") or ""),
        "groq_task": (
            "Generate the final AVAJ answer using only structured_context. Match answer_format and language. "
            "Cite source labels from context. If coverage is none, politely state that the information is unavailable."
        ),
    }


def _select_reliable_candidates(ranked: list, cutoff: float) -> list:
    passed = [candidate for candidate in ranked if candidate.passed_threshold]
    if not passed:
        return []
    top_score = max(candidate.rerank_score for candidate in passed)
   # if top_score <= 0:
    #    return []
    adaptive_floor = max(cutoff, 0.01)
    if top_score >= 2:
        adaptive_floor = max(adaptive_floor, top_score * 0.67)
    return [candidate for candidate in passed if candidate.rerank_score >= adaptive_floor]


def _extract_relevant_evidence(query: str, text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    terms = _content_terms(query)
    list_block = _extract_list_block(query, text, terms, limit)
    if list_block:
        return list_block
    ranked = _rank_text_units(text, terms)
    selected: list[str] = []
    total = 0
    for _score, index, unit in ranked:
        if total >= limit:
            break
        expanded = _with_neighbors(text, index, unit)
        if expanded in selected:
            continue
        if total + len(expanded) > limit and selected:
            continue
        selected.append(expanded)
        total += len(expanded) + 2
    if selected:
        return "\n".join(selected)[:limit].rsplit(" ", 1)[0]
    return _anchored_snippet(query, text, limit)


def _content_terms(query: str) -> set[str]:
    stopwords = {
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
    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9#.+-]+", query):
        lowered = token.lower()
        if lowered in stopwords:
            continue
        if len(lowered) > 2 or lowered.isdigit() or lowered in set(number_to_roman.values()):
            terms.add(lowered)
        if lowered in number_to_roman:
            terms.add(number_to_roman[lowered])
    return terms


def _rank_text_units(text: str, terms: set[str]) -> list[tuple[float, int, str]]:
    units = _split_units(text)
    ranked: list[tuple[float, int, str]] = []
    for index, unit in enumerate(units):
        unit_tokens = set(re.findall(r"[A-Za-z0-9#.+-]+", unit.lower()))
        hits = len(terms & unit_tokens)
        if hits == 0:
            continue
        density = hits / max(len(unit.split()), 1)
        structure_bonus = 0.5 if _looks_structured(unit) else 0.0
        ranked.append((hits + density + structure_bonus, index, unit))
    return sorted(ranked, key=lambda item: item[0], reverse=True)


def _extract_list_block(query: str, text: str, terms: set[str], limit: int) -> str:
    if not re.search(r"\b(list|subjects?|courses?|papers?|members?|items?|names?)\b", query, re.I):
        return ""
    units = _split_units(text)
    if len(units) < 3:
        return ""
    anchor_indexes = []
    for index, unit in enumerate(units):
        unit_tokens = set(re.findall(r"[A-Za-z0-9#.+-]+", unit.lower()))
        hits = len(terms & unit_tokens)
        if hits >= 2 or (hits >= 1 and _looks_structured(unit)):
            anchor_indexes.append(index)
    for anchor in anchor_indexes:
        block = _collect_following_block(units, anchor, terms, limit)
        if _block_has_list_value(block):
            return block[:limit].rsplit(" ", 1)[0]
    return ""


def _collect_following_block(units: list[str], anchor: int, terms: set[str], limit: int) -> str:
    selected: list[str] = []
    total = 0
    started = False
    for index in range(anchor, len(units)):
        unit = units[index]
        if index > anchor and _is_new_unrelated_section(unit, terms, started):
            break
        selected.append(unit)
        total += len(unit) + 1
        started = True
        if total >= limit:
            break
    return "\n".join(selected)


def _is_new_unrelated_section(unit: str, terms: set[str], started: bool) -> bool:
    if not started:
        return False
    unit_tokens = set(re.findall(r"[A-Za-z0-9#.+-]+", unit.lower()))
    if re.search(r"\bsemester\b", unit, re.I) and not (terms & unit_tokens):
        return True
    if re.search(r"\b(?:about|quick links|external links|contact us)\b", unit, re.I):
        return True
    return False


def _block_has_list_value(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    if len(lines) >= 4:
        return True
    return bool(re.search(r"\b[A-Z]{2,}\d{2,}\b|\|\s*[^|]+\s*\|", block))


def _split_units(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [unit.strip() for unit in re.split(r"(?<=[.!?])\s+|\s{2,}", text) if unit.strip()]


def _looks_structured(unit: str) -> bool:
    return "|" in unit or bool(re.search(r"\b(?:code|course|date|department|faculty|fee|name|paper)\b", unit, re.I))


def _with_neighbors(text: str, index: int, unit: str) -> str:
    units = _split_units(text)
    if len(units) <= 1:
        return unit
    start = max(0, index - 1)
    end = min(len(units), index + 2)
    return "\n".join(units[start:end])


def _anchored_snippet(query: str, text: str, limit: int) -> str:
    anchors = [word for word in re.findall(r"[A-Za-z0-9#.+-]{4,}", query)]
    lower_text = text.lower()
    for anchor in anchors:
        idx = lower_text.find(anchor.lower())
        if idx >= 0:
            start = max(0, idx - limit // 3)
            end = min(len(text), start + limit)
            return text[start:end].rsplit(" ", 1)[0]
    return text[:limit].rsplit(" ", 1)[0]
