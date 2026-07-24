from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from config import AppSettings, get_settings

from .loaders import load_document
from .models import ChildChunk, DocumentRecord, IndexEntry, ParentChunk


class IngestionPipeline:
    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = (settings or get_settings()).ensure_dirs()
        self.ledger_path = self.settings.data_dir / "index_ledger.json"
        self.parents_path = self.settings.data_dir / "parents.json"
        self.children_path = self.settings.data_dir / "children.json"

    def ingest_path(self, path: Path, force: bool = False) -> IndexEntry:
        record = load_document(Path(path))
        existing = self.ledger().get(record.sha256)
        if existing and not force:
            existing["status"] = "duplicate_skipped"
            return IndexEntry(**existing)
        return self._index_record(record, force=force)

    def save_and_ingest_upload(self, filename: str, content: bytes, force: bool = False) -> IndexEntry:
        safe_name = Path(filename).name
        target = self.settings.upload_dir / safe_name
        target.write_bytes(content)
        return self.ingest_path(target, force=force)

    def _index_record(self, record: DocumentRecord, force: bool) -> IndexEntry:
        if force:
            self.delete_document(record.sha256)

        parent_chunks = self._make_parent_chunks(record)
        child_chunks = self._make_child_chunks(parent_chunks)

        parents = self.parents()
        children = self.children()
        for parent in parent_chunks:
            parents[parent.id] = parent.to_dict()
        for child in child_chunks:
            children[child.id] = child.to_dict()

        entry = IndexEntry(
            document_hash=record.sha256,
            source_name=record.source_name,
            indexed_at=datetime.now(timezone.utc).isoformat(),
            parent_count=len(parent_chunks),
            child_count=len(child_chunks),
            status="indexed",
            metadata=record.metadata | {"sections": record.sections, "table_count": len(record.tables)},
        )
        ledger = self.ledger()
        ledger[record.sha256] = entry.to_dict()
        self._write_json(self.ledger_path, ledger)
        self._write_json(self.parents_path, parents)
        self._write_json(self.children_path, children)
        return entry

    def delete_document(self, document_hash: str) -> bool:
        ledger = self.ledger()
        existed = document_hash in ledger
        ledger.pop(document_hash, None)
        parents = {k: v for k, v in self.parents().items() if v.get("document_hash") != document_hash}
        children = {k: v for k, v in self.children().items() if v.get("document_hash") != document_hash}
        self._write_json(self.ledger_path, ledger)
        self._write_json(self.parents_path, parents)
        self._write_json(self.children_path, children)
        chroma_dir = self.settings.data_dir / "chroma"
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir, ignore_errors=True)
        return existed

    def ledger(self) -> Dict[str, dict]:
        return self._read_json(self.ledger_path)

    def parents(self) -> Dict[str, dict]:
        return self._read_json(self.parents_path)

    def children(self) -> Dict[str, dict]:
        return self._read_json(self.children_path)

    def list_entries(self) -> List[IndexEntry]:
        return [IndexEntry(**item) for item in self.ledger().values()]

    def get_parent(self, parent_id: str) -> Optional[ParentChunk]:
        data = self.parents().get(parent_id)
        return ParentChunk(**data) if data else None

    def all_children(self) -> List[ChildChunk]:
        return [ChildChunk(**item) for item in self.children().values()]

    def _make_parent_chunks(self, record: DocumentRecord) -> List[ParentChunk]:
        chunks = _window_text(record.text, self.settings.parent_chunk_tokens, 0)
        if not chunks and record.text:
            chunks = [record.text]
        return [
            ParentChunk(
                id=f"{record.sha256}:p:{idx}",
                document_hash=record.sha256,
                source_name=record.source_name,
                text=chunk,
                ordinal=idx,
                metadata=record.metadata,
            )
            for idx, chunk in enumerate(chunks)
        ]

    def _make_child_chunks(self, parents: Iterable[ParentChunk]) -> List[ChildChunk]:
        children: List[ChildChunk] = []
        for parent in parents:
            chunks = _window_text(
                parent.text,
                self.settings.child_chunk_tokens,
                self.settings.child_overlap_tokens,
            )
            for idx, chunk in enumerate(chunks):
                children.append(
                    ChildChunk(
                        id=f"{parent.id}:c:{idx}",
                        parent_id=parent.id,
                        document_hash=parent.document_hash,
                        source_name=parent.source_name,
                        text=chunk,
                        ordinal=idx,
                        metadata=parent.metadata,
                    )
                )
        return children

    @staticmethod
    def _read_json(path: Path) -> Dict[str, dict]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, data: Dict[str, dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _window_words(words: List[str], size: int, overlap: int) -> List[str]:
    if not words:
        return []
    size = max(1, size)
    overlap = max(0, min(overlap, size - 1))
    step = size - overlap
    chunks: List[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + size >= len(words):
            break
    return chunks


def _window_text(text: str, size: int, overlap: int) -> List[str]:
    units = _split_structured_units(text)
    if not units:
        return []
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0
    for unit in units:
        unit_words = unit.split()
        if len(unit_words) > size:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_words = 0
            chunks.extend(_window_words(unit_words, size, overlap))
            continue
        if current and current_words + len(unit_words) > size:
            chunks.append("\n".join(current))
            overlap_units: List[str] = []
            overlap_words = 0
            for previous in reversed(current):
                count = len(previous.split())
                if overlap_words + count > overlap:
                    break
                overlap_units.insert(0, previous)
                overlap_words += count
            current = overlap_units
            current_words = overlap_words
        current.append(unit)
        current_words += len(unit_words)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_structured_units(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [unit.strip() for unit in re.split(r"(?<=[.!?])\s+|\s{2,}", text) if unit.strip()]
