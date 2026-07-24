from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from config import AppSettings, get_settings
from ingestion import IngestionPipeline
from rag_engine import RAGEngine

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None  # type: ignore[assignment]


def run_cli(settings: AppSettings | None = None) -> None:
    settings = settings or get_settings()
    if Console is None:
        _plain_cli(settings)
        return
    console = Console()
    pipeline = IngestionPipeline(settings)
    engine = RAGEngine(settings)
    console.print(Panel.fit("Enterprise Local RAG", subtitle="type /help for commands"))
    while True:
        try:
            command = console.input("[bold cyan]rag> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            return
        if not command:
            continue
        if command in {"/quit", "/exit"}:
            return
        if command == "/help":
            console.print("Commands: /ingest <path>, /docs, /ask <question>, /quit")
            continue
        if command.startswith("/ingest "):
            path = Path(command.removeprefix("/ingest ").strip('" '))
            with console.status(f"Ingesting {path}..."):
                entry = pipeline.ingest_path(path)
                engine.refresh()
            console.print(_entry_table([entry.to_dict()]))
            continue
        if command == "/docs":
            console.print(_entry_table([entry.to_dict() for entry in pipeline.list_entries()]))
            continue
        if command.startswith("/ask "):
            query = command.removeprefix("/ask ").strip()
            answer = []
            log: Dict[str, Any] = {}
            with console.status("Retrieving and generating..."):
                for event in engine.ask(query):
                    if event["type"] == "token":
                        answer.append(str(event["token"]))
                    elif event["type"] == "log":
                        log = event["log"]  # type: ignore[assignment]
            console.print(Panel(Markdown("".join(answer)), title="Answer"))
            _print_work_log(console, log)
            continue
        console.print("Unknown command. Type /help.")


def _plain_cli(settings: AppSettings) -> None:
    pipeline = IngestionPipeline(settings)
    engine = RAGEngine(settings)
    print("Enterprise Local RAG. Commands: /ingest <path>, /docs, /ask <question>, /quit")
    while True:
        command = input("rag> ").strip()
        if command in {"/quit", "/exit"}:
            return
        if command.startswith("/ingest "):
            entry = pipeline.ingest_path(Path(command.removeprefix("/ingest ").strip('" ')))
            engine.refresh()
            print(entry)
        elif command == "/docs":
            for entry in pipeline.list_entries():
                print(entry)
        elif command.startswith("/ask "):
            for event in engine.ask(command.removeprefix("/ask ").strip()):
                if event["type"] == "token":
                    print(event["token"], end="", flush=True)
            print()


def _entry_table(entries: Iterable[Dict[str, Any]]) -> Any:
    table = Table(title="Indexed Documents")
    for column in ["source_name", "status", "parent_count", "child_count", "document_hash"]:
        table.add_column(column)
    for entry in entries:
        table.add_row(
            str(entry.get("source_name", "")),
            str(entry.get("status", "")),
            str(entry.get("parent_count", "")),
            str(entry.get("child_count", "")),
            str(entry.get("document_hash", ""))[:12],
        )
    return table


def _print_work_log(console: Any, log: Dict[str, Any]) -> None:
    if not log:
        return
    metrics = Table(title="AI Work Log")
    metrics.add_column("Stage")
    metrics.add_column("Value")
    metrics.add_row("Query ID", str(log.get("query_id", "")))
    metrics.add_row("Expanded Queries", "\n".join(log.get("expanded_queries", [])))
    metrics.add_row("Low Confidence", str(log.get("low_confidence_retrieval", False)))
    metrics.add_row("Reasoning Summary", str(log.get("safe_reasoning_summary", "")))
    for key, value in log.get("performance_metrics", {}).items():
        metrics.add_row(key, f"{value} ms")
    console.print(metrics)

    ranked = Table(title="Reranking Ledger")
    for column in ["source", "rerank_score", "passed", "child_id"]:
        ranked.add_column(column)
    for item in log.get("rerank_ledger", [])[:20]:
        ranked.add_row(
            str(item.get("source", "")),
            str(item.get("rerank_score", "")),
            str(item.get("passed", "")),
            str(item.get("child_id", ""))[:24],
        )
    console.print(ranked)
