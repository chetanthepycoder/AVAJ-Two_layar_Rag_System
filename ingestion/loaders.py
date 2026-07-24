from __future__ import annotations

import hashlib
import json
import re
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .models import DocumentRecord


SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".json", ".jsonl", ".pdf"}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def markdown_table(rows: Iterable[Iterable[Any]]) -> str:
    cleaned = [["" if cell is None else str(cell).strip() for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    normalized = [row + [""] * (width - len(row)) for row in cleaned]
    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:] or [[""] * width]

    def fmt(row: List[str]) -> str:
        return "| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |"

    return "\n".join([fmt(header), fmt(separator), *[fmt(row) for row in body]])


def _load_text(path: Path, content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="replace")


def _load_json(content: bytes, suffix: str) -> str:
    raw = _load_text(Path("document.json"), content)
    if suffix == ".jsonl":
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        items = json.loads(raw)
    return json.dumps(items, indent=2, ensure_ascii=False)


def _html_with_tables(content: bytes) -> Tuple[str, List[str], List[str]]:
    raw = _load_text(Path("document.html"), content)
    tables: List[str] = []
    sections: List[str] = []
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        for tag in soup.find_all(["nav", "header", "footer", "aside"]):
            tag.decompose()
        for tag in soup.find_all(attrs={"role": re.compile(r"navigation|banner|contentinfo", re.I)}):
            tag.decompose()
        for heading in soup.find_all(re.compile("^h[1-6]$")):
            text = heading.get_text(" ", strip=True)
            if text:
                sections.append(text)
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                rows.append([cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])])
            rendered = markdown_table(rows)
            if rendered:
                tables.append(rendered)
                table.replace_with(soup.new_string("\n" + rendered + "\n"))
        content_root = _best_html_content_root(soup)
        text = content_root.get_text("\n", strip=True)
        text = _normalize_structured_text(text)
        return text, tables, sections
    except Exception:
        table_pattern = re.compile(r"<table.*?</table>", flags=re.IGNORECASE | re.DOTALL)
        text = table_pattern.sub("\n[HTML table omitted: install beautifulsoup4 for table extraction]\n", raw)
        text = re.sub(r"<[^>]+>", " ", text)
        return unescape(re.sub(r"\s+", " ", text)), tables, sections


def _best_html_content_root(soup: Any) -> Any:
    candidates = []
    selectors = [
        "main",
        "article",
        "[role='main']",
        "[class*='content']",
        "[class*='main']",
        "[id*='content']",
        "[id*='main']",
    ]
    for selector in selectors:
        candidates.extend(soup.select(selector))
    candidates = [candidate for candidate in candidates if len(candidate.get_text(" ", strip=True)) > 200]
    if candidates:
        return max(candidates, key=lambda tag: len(tag.get_text(" ", strip=True)))
    return soup.body or soup


def _normalize_structured_text(text: str) -> str:
    lines = []
    previous = ""
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or line == previous:
            continue
        lines.append(line)
        previous = line
    return "\n".join(lines)


def _pdf_with_tables(path: Path) -> Tuple[str, List[str]]:
    tables: List[str] = []
    pages: List[str] = []
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                for table in page.extract_tables() or []:
                    rendered = markdown_table(table)
                    if rendered:
                        tables.append(rendered)
                        page_text += f"\n\nTable from page {page_number}:\n{rendered}\n"
                pages.append(page_text)
        return "\n\n".join(pages), tables
    except Exception:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages), tables
        except Exception as exc:
            raise RuntimeError(f"Could not parse PDF {path.name}: {exc}") from exc


def load_document(path: Path) -> DocumentRecord:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    content = path.read_bytes()
    digest = sha256_bytes(content)
    metadata: Dict[str, Any] = {"extension": suffix, "size_bytes": len(content)}
    tables: List[str] = []
    sections: List[str] = []

    if suffix in {".txt", ".md"}:
        text = _load_text(path, content)
    elif suffix in {".html", ".htm"}:
        text, tables, sections = _html_with_tables(content)
    elif suffix in {".json", ".jsonl"}:
        text = _load_json(content, suffix)
    elif suffix == ".pdf":
        text, tables = _pdf_with_tables(path)
    else:  # pragma: no cover - guarded above.
        raise ValueError(f"Unsupported file type: {suffix}")

    return DocumentRecord(
        source_name=path.name,
        text=text.strip(),
        sha256=digest,
        metadata=metadata,
        tables=tables,
        sections=sections,
    )
