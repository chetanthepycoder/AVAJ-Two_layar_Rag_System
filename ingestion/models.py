from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class DocumentRecord:
    source_name: str
    text: str
    sha256: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tables: List[str] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParentChunk:
    id: str
    document_hash: str
    source_name: str
    text: str
    ordinal: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChildChunk:
    id: str
    parent_id: str
    document_hash: str
    source_name: str
    text: str
    ordinal: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IndexEntry:
    document_hash: str
    source_name: str
    indexed_at: str
    parent_count: int
    child_count: int
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
