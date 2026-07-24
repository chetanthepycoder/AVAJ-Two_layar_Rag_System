from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class AIWorkLog:
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str = ""
    expanded_queries: List[str] = field(default_factory=list)
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)
    fused_candidate_ids: List[str] = field(default_factory=list)
    rerank_ledger: List[Dict[str, Any]] = field(default_factory=list)
    selected_parent_chunks: List[Dict[str, Any]] = field(default_factory=list)
    evidence_extraction: Dict[str, Any] = field(default_factory=dict)
    layer1_output: Dict[str, Any] = field(default_factory=dict)
    layer2_output: Dict[str, Any] = field(default_factory=dict)
    layer1_provider: str = ""
    layer2_provider: str = ""
    safe_reasoning_summary: str = ""
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    provider_fallback_events: List[str] = field(default_factory=list)
    token_counts: Dict[str, int] = field(default_factory=dict)
    low_confidence_retrieval: bool = False

    def add_latency(self, stage: str, elapsed_ms: float) -> None:
        self.performance_metrics[stage] = round(elapsed_ms, 2)

    def add_fallback(self, event: str) -> None:
        self.provider_fallback_events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StageTimer:
    def __init__(self, log: AIWorkLog, stage: str):
        self.log = log
        self.stage = stage
        self.started = 0.0

    def __enter__(self) -> "StageTimer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        elapsed_ms = (time.perf_counter() - self.started) * 1000
        self.log.add_latency(self.stage, elapsed_ms)
