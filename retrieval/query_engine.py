from __future__ import annotations

import re
from typing import Any, List, Optional

from config import AppSettings, get_settings
from observability import AIWorkLog
from providers.ollama import OllamaProvider


class QueryExpansionService:
    def __init__(self, settings: Optional[AppSettings] = None, ollama: Optional[OllamaProvider] = None):
        self.settings = settings or get_settings()
        self.ollama = ollama or OllamaProvider(self.settings)

    def expand(
        self,
        query: str,
        l1_payload: Optional[dict[str, Any]] = None,
        log: Optional[AIWorkLog] = None,
    ) -> List[str]:
        """Use Layer 1 terms when available, otherwise deterministically expand the query."""
        # Preserve the previous ``expand(query, log)`` call shape for integrations.
        if isinstance(l1_payload, AIWorkLog) and log is None:
            log = l1_payload
            l1_payload = None
        if l1_payload and str(l1_payload.get("provider", "")).startswith("ollama"):
            optimized = str(l1_payload.get("optimized_query") or query)
            hints = [str(hint) for hint in l1_payload.get("search_hints", []) if str(hint).strip()]
            unique = list(dict.fromkeys([optimized, query, *hints]))
            if log:
                log.expanded_queries = unique
            return unique

        return self._deterministic_expand(query, log)

    def _deterministic_expand(self, query: str, log: Optional[AIWorkLog] = None) -> List[str]:
        expansions = _fallback_query_variants(query)
        unique: list[str] = []
        for item in [query, *expansions]:
            if item and item.lower() not in {seen.lower() for seen in unique}:
                unique.append(item)
        if log:
            log.expanded_queries = unique
        return unique


def _fallback_query_variants(query: str) -> List[str]:
    keywords = [
        token
        for token in re.findall(r"[A-Za-z0-9#.+-]+", query)
        if len(token) > 2
        and token.lower()
        not in {
            "about",
            "give",
            "list",
            "please",
            "show",
            "tell",
            "what",
            "which",
        }
    ]
    variants: list[str] = []
    if len(keywords) >= 2:
        variants.append(" ".join(keywords))
    if len(keywords) >= 4:
        variants.append(" ".join(keywords[:4]))
    number_variants = _number_variants(query)
    variants.extend(number_variants)
    return variants


def _number_variants(query: str) -> List[str]:
    roman = {
        "1": "I",
        "2": "II",
        "3": "III",
        "4": "IV",
        "5": "V",
        "6": "VI",
        "7": "VII",
        "8": "VIII",
        "9": "IX",
        "10": "X",
    }
    variants: list[str] = []
    for number in re.findall(r"\b(?:[1-9]|10)\b", query):
        if number in roman:
            variants.append(re.sub(rf"\b{number}\b", roman[number], query, flags=re.IGNORECASE))
    return variants
