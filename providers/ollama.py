from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Iterator, Mapping, Optional, Sequence

from config import AppSettings, get_settings


class OllamaProvider:
    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or get_settings()

    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.settings.ollama_host}/api/tags", timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def complete(
        self, prompt: str, timeout: Optional[int] = None, json_response: bool = False, model: Optional[str] = None,
    ) -> str:
        payload: dict[str, Any] = {"model": model or self.settings.ollama_model, "prompt": prompt, "stream": False}
        if json_response:
            payload["format"] = "json"
        request = urllib.request.Request(
            f"{self.settings.ollama_host}/api/generate", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.settings.request_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8")).get("response", "")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama unavailable at {self.settings.ollama_host}") from exc

    def stream(self, prompt: str) -> Iterator[str]:
        payload = {"model": self.settings.ollama_model, "prompt": prompt, "stream": True}
        request = urllib.request.Request(
            f"{self.settings.ollama_host}/api/generate", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                item = json.loads(raw_line.decode("utf-8"))
                if item.get("response"):
                    yield item["response"]
                if item.get("done"):
                    break

    def prompt_engineer(self, query: str, timeout: Optional[int] = None) -> dict[str, Any]:
        """Layer 1: optimize retrieval and classify the user request before search."""
        fallback = _layer1_fallback(query)
        if not self.is_available():
            fallback["provider"] = "layer1_offline_fallback"
            return fallback
        try:
            raw = self.complete(_build_layer1_prompt(query), timeout=timeout or self.settings.layer1_timeout_seconds,
                                json_response=True, model=self.settings.ollama_layer1_model)
            parsed = _parse_json_object(raw)
            if not parsed or not str(parsed.get("optimized_query", "")).strip():
                fallback.update(provider="layer1_parse_fallback", raw_output=raw)
                return fallback
            return _normalise_layer1(parsed, query, raw)
        except Exception as exc:
            fallback["provider"] = f"layer1_error_fallback: {exc}"
            return fallback

    def collect_content(
        self, original_query: str, optimized_query: str, intent: str, answer_format: str, domain: str,
        rag_context: Sequence[Mapping[str, Any]], low_confidence: bool = False, timeout: Optional[int] = None,
    ) -> dict[str, Any]:
        """Layer 2: extract and structure only retrieved evidence for final generation."""
        del original_query  # The optimized query and intent are the evidence-selection contract.
        contexts = [_normalize_context(item, index + 1) for index, item in enumerate(rag_context)]
        fallback = _layer2_fallback(contexts, low_confidence)
        if not self.is_available():
            fallback["provider"] = "layer2_offline_fallback"
            return fallback
        try:
            raw = self.complete(_build_layer2_prompt(optimized_query, intent, answer_format, domain, contexts, low_confidence),
                                timeout=timeout or self.settings.layer2_timeout_seconds, json_response=True)
            parsed = _parse_json_object(raw)
            if not parsed or "structured_context" not in parsed:
                fallback.update(provider="layer2_parse_fallback", raw_output=raw)
                return fallback
            return _normalise_layer2(parsed, contexts, raw)
        except Exception as exc:
            fallback["provider"] = f"layer2_error_fallback: {exc}"
            return fallback


def _build_layer1_prompt(query: str) -> str:
    return (
        "You are the Prompt Engineer for AVAJ, the AI assistant of National PG College (NPGC). Analyze the raw "
        "user query and output ONLY valid JSON with no markdown or preamble.\n\nRequired schema:\n"
        '{"optimized_query":"expanded search string under 20 words","search_hints":["3 to 6 official-document terms"],'
        '"intent":"one English sentence","answer_format":"listing|steps|direct|explanation|table",'
        '"domain":"academics|admission|faculty|fee|facility|event|general","is_npgc_related":true,'
        '"language":"hinglish|hindi|english"}\n\n'
        "Rules: expand abbreviations and remove filler in optimized_query; use listing for course lists, steps for "
        "procedures, direct for simple facts, and table only for supported comparisons. Set is_npgc_related false only "
        "when the request is clearly unrelated to NPGC. Never fabricate facts.\n\n"
        f"QUERY: {query}"
    )


def _build_layer2_prompt(
    optimized_query: str, intent: str, answer_format: str, domain: str, contexts: Sequence[Mapping[str, Any]],
    low_confidence: bool,
) -> str:
    return (
        "You are the Content Collector for the AVAJ RAG pipeline. Extract ONLY facts from RETRIEVED CONTEXT that "
        "directly answer the query. Output ONLY valid JSON, with no markdown or preamble.\n\nRequired schema:\n"
        '{"collected_facts":[{"rank":1,"source":"source_name","fact":"verbatim excerpt","confidence":0.0}],'
        '"structured_context":"human-readable evidence for the final LLM","coverage":"full|partial|none",'
        '"retrieval_used":true,"missing_info":"requested information absent from context","confidence_summary":0.0}\n\n'
        "Rules: facts must be verbatim excerpts, preserve source labels, and do not invent or infer. For no relevant "
        "context use coverage none, no facts, and retrieval_used false. Organize structured_context to suit answer_format.\n"
        f"LOW_CONFIDENCE_RETRIEVAL: {low_confidence}\nINTENT: {intent}\nANSWER_FORMAT: {answer_format}\n"
        f"DOMAIN: {domain}\nOPTIMIZED_QUERY: {optimized_query}\n\nRETRIEVED CONTEXT:\n"
        f"{json.dumps(list(contexts), ensure_ascii=False, separators=(',', ':'))}"
    )


def _layer1_fallback(query: str) -> dict[str, Any]:
    terms = [term for term in re.findall(r"[A-Za-z0-9#+.-]+", query) if len(term) > 2]
    return {"optimized_query": query, "search_hints": terms[:6], "intent": f"User asked: {query}",
            "answer_format": "direct", "domain": "general", "is_npgc_related": True, "language": "hinglish",
            "provider": "layer1_deterministic_fallback", "raw_output": ""}


def _layer2_fallback(contexts: Sequence[Mapping[str, Any]], low_confidence: bool) -> dict[str, Any]:
    lines = [f"[Source: {item.get('source', '?')} | Confidence: {float(item.get('confidence', 0) or 0):.2f}]\n"
             f"{item.get('content', '')}" for item in contexts]
    confidences = [float(item.get("confidence", 0) or 0) for item in contexts]
    return {
        "collected_facts": [{"rank": index + 1, "source": item.get("source", "?"),
                              "fact": str(item.get("content", ""))[:500],
                              "confidence": float(item.get("confidence", 0) or 0)}
                            for index, item in enumerate(contexts)],
        "structured_context": "\n\n".join(lines) if lines else "No retrieved context available.",
        "coverage": "none" if low_confidence or not contexts else "partial",
        "retrieval_used": bool(contexts) and not low_confidence,
        "missing_info": "" if contexts else "No relevant indexed information was retrieved.",
        "confidence_summary": sum(confidences) / len(confidences) if confidences else 0.0,
        "provider": "layer2_deterministic_fallback", "raw_output": "",
    }


def _normalise_layer1(parsed: Mapping[str, Any], query: str, raw: str) -> dict[str, Any]:
    formats = {"listing", "steps", "direct", "explanation", "table"}
    domains = {"academics", "admission", "faculty", "fee", "facility", "event", "general"}
    languages = {"hinglish", "hindi", "english"}
    hints = [str(item).strip() for item in parsed.get("search_hints", []) if str(item).strip()][:6]
    answer_format = str(parsed.get("answer_format", "")).lower()
    domain = str(parsed.get("domain", "")).lower()
    language = str(parsed.get("language", "")).lower()
    return {"optimized_query": str(parsed.get("optimized_query") or query).strip(), "search_hints": hints,
            "intent": str(parsed.get("intent") or "").strip(),
            "answer_format": answer_format if answer_format in formats else "direct",
            "domain": domain if domain in domains else "general",
            "is_npgc_related": bool(parsed.get("is_npgc_related", True)),
            "language": language if language in languages else "hinglish", "provider": "ollama_layer1", "raw_output": raw}


def _normalise_layer2(parsed: Mapping[str, Any], contexts: Sequence[Mapping[str, Any]], raw: str) -> dict[str, Any]:
    coverage = str(parsed.get("coverage", "partial")).lower()
    facts = parsed.get("collected_facts", [])
    return {"collected_facts": facts if isinstance(facts, list) else [],
            "structured_context": str(parsed.get("structured_context") or ""),
            "coverage": coverage if coverage in {"full", "partial", "none"} else "partial",
            "retrieval_used": bool(parsed.get("retrieval_used", bool(contexts))),
            "missing_info": str(parsed.get("missing_info") or ""),
            "confidence_summary": _coerce_float(parsed.get("confidence_summary"), 0.5),
            "provider": "ollama_layer2", "raw_output": raw}


def _normalize_context(item: Mapping[str, Any], rank: int) -> dict[str, Any]:
    return {"rank": int(item.get("rank", rank)), "source": str(item.get("source", "unknown")),
            "confidence": _coerce_float(item.get("confidence"), 0.0), "content": str(item.get("content", "")).strip()}


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start:end + 1])
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}
