from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

from config import get_settings
from ingestion import IngestionPipeline
from rag_engine import RAGEngine


def run() -> None:
    try:
        import streamlit as st
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Streamlit is not installed. Install requirements.txt first.") from exc

    base_settings = get_settings()
    st.set_page_config(page_title=base_settings.app_name, page_icon="RAG", layout="wide")
    st.title(base_settings.app_name)

    with st.sidebar:
        st.header("Configuration")
        groq_key = st.text_input("Groq API key", value=base_settings.groq_api_key or "", type="password")
        hf_token = st.text_input("Hugging Face token", value=base_settings.hf_token or "", type="password")
        groq_model = st.text_input("Groq model", value=base_settings.groq_model)
        ollama_model = st.text_input("Ollama model", value=base_settings.ollama_model)
        top_k = st.slider("Top-K retrieval", 10, 50, int(base_settings.top_k))
        alpha = st.slider("Hybrid semantic weight", 0.0, 1.0, float(base_settings.hybrid_alpha), 0.05)
        cutoff = st.slider("Reranker cutoff", -5.0, 5.0, float(base_settings.rerank_cutoff), 0.25)

    settings = base_settings.with_overrides(
        groq_api_key=groq_key or None,
        hf_token=hf_token or None,
        groq_model=groq_model,
        ollama_model=ollama_model,
        top_k=top_k,
        hybrid_alpha=alpha,
        rerank_cutoff=cutoff,
    )

    if "engine" not in st.session_state or st.session_state.get("settings_fingerprint") != str(settings.model_dump() if hasattr(settings, "model_dump") else settings.__dict__):
        st.session_state.engine = RAGEngine(settings)
        st.session_state.pipeline = IngestionPipeline(settings)
        st.session_state.settings_fingerprint = str(settings.model_dump() if hasattr(settings, "model_dump") else settings.__dict__)
        st.session_state.messages = st.session_state.get("messages", [])

    engine: RAGEngine = st.session_state.engine
    pipeline: IngestionPipeline = st.session_state.pipeline

    with st.sidebar:
        st.header("Provider Status")
        st.write(f"Groq configured: {'yes' if settings.groq_api_key else 'no'}")
        st.write(f"Ollama reachable: {'yes' if engine.ollama.is_available() else 'no'}")

    docs_tab, chat_tab = st.tabs(["Documents", "Chat"])

    with docs_tab:
        st.subheader("Document Management")
        uploads = st.file_uploader(
            "Upload documents",
            type=["txt", "md", "html", "htm", "json", "jsonl", "pdf"],
            accept_multiple_files=True,
        )
        force = st.checkbox("Force reindex uploaded duplicates")
        if uploads and st.button("Index uploaded documents", type="primary"):
            for upload in uploads:
                entry = pipeline.save_and_ingest_upload(upload.name, upload.getvalue(), force=force)
                st.success(f"{entry.source_name}: {entry.status}")
            engine.refresh()
        entries = [entry.to_dict() for entry in pipeline.list_entries()]
        st.dataframe(entries, use_container_width=True)
        if entries:
            selected_hash = st.selectbox("Document action target", [entry["document_hash"] for entry in entries])
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("Delete selected"):
                    pipeline.delete_document(selected_hash)
                    engine.refresh()
                    st.rerun()
            with action_cols[1]:
                st.caption("Reindex by uploading the document again with force reindex enabled.")

    with chat_tab:
        st.subheader("Ask the indexed documents")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("log"):
                    _render_work_log(st, message["log"])
        query = st.chat_input("Ask a question")
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                answer = ""
                work_log: Dict[str, Any] = {}
                for event in engine.ask(query):
                    if event["type"] == "token":
                        answer += str(event["token"])
                        placeholder.markdown(answer)
                    elif event["type"] == "log":
                        work_log = event["log"]  # type: ignore[assignment]
                _render_work_log(st, work_log)
            st.session_state.messages.append({"role": "assistant", "content": answer, "log": work_log})


def _render_work_log(st: Any, log: Dict[str, Any]) -> None:
    if not log:
        return
    with st.expander("AI Work Log", expanded=False):
        tabs = st.tabs(["Query Expansion", "Retrieval & Reranking", "Evidence Audit", "Performance"])
        with tabs[0]:
            st.write(log.get("expanded_queries", []))
            if log.get("provider_fallback_events"):
                st.warning("\n".join(log["provider_fallback_events"]))
        with tabs[1]:
            st.json(log.get("retrieval_stats", {}))
            st.dataframe(log.get("rerank_ledger", []), use_container_width=True)
        with tabs[2]:
            st.write(log.get("safe_reasoning_summary", ""))
            st.json(log.get("evidence_extraction", {}))
            st.dataframe(log.get("selected_parent_chunks", []), use_container_width=True)
        with tabs[3]:
            st.json(log.get("performance_metrics", {}))
            st.json(log.get("token_counts", {}))


if __name__ == "__main__":
    run()
