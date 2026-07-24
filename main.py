from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import get_settings
from ingestion import IngestionPipeline
from ui.cli import run_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise Local RAG")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("cli", help="Start the Rich terminal dashboard")
    subparsers.add_parser("web", help="Start the Streamlit web app")

    ingest = subparsers.add_parser("ingest", help="Index a document")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--force", action="store_true")

    ask = subparsers.add_parser("ask", help="Ask a question from the CLI")
    ask.add_argument("question")

    args = parser.parse_args()
    settings = get_settings()

    if args.command == "web":
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path("ui") / "app.py")], check=False)
    elif args.command == "ingest":
        entry = IngestionPipeline(settings).ingest_path(args.path, force=args.force)
        print(entry)
    elif args.command == "ask":
        from rag_engine import RAGEngine

        for event in RAGEngine(settings).ask(args.question):
            if event["type"] == "token":
                print(event["token"], end="", flush=True)
        print()
    else:
        run_cli(settings)


if __name__ == "__main__":
    main()
