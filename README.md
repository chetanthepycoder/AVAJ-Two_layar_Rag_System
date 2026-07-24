# AVAJ — AI Assistant for National PG College

**AVAJ** (Advanced Voice Assistance Junction) is a local-first, production-grade RAG (Retrieval-Augmented Generation) system built specifically for **National Post Graduate College, Lucknow**. It answers student and faculty queries in natural **Hinglish** (Hindi + English in Roman script), sourcing every answer strictly from the college's official documents — never hallucinating facts.

> Built by **Abhishek** · SDE Intern · Vastu House Finance Company · BCA, University of Lucknow

---

## What is AVAJ?

AVAJ is a domain-locked AI assistant. Ask it anything about NPGC — admissions, courses, faculty, fees, exam schedules, departments, scholarships, rules — and it pulls the answer from indexed college documents, structures it through a three-layer AI pipeline, and responds in the natural Hinglish tone of an intelligent robotic assistant.

It works **offline by default**. Cloud providers (Groq) and local LLMs (Ollama) are optional accelerators — the system degrades gracefully when neither is available.

---

## How It Works

AVAJ processes every query through a strict **3-layer pipeline**:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 1 — Query Intelligence  (Ollama / fallback)  │
│  • Classifies whether query is NPGC-related         │
│  • Rewrites the query for better retrieval          │
│  • Identifies intent, domain, language, format      │
│  • Produces search hints for boosting               │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  HYBRID SEARCH  (BM25 sparse + ChromaDB dense)      │
│  • BM25 keyword search on child chunks              │
│  • BGE-M3 multilingual semantic vector search       │
│  • RRF (Reciprocal Rank Fusion) merges both lanes   │
│  • Search-hint boosting on high-priority terms      │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  RERANKER  (CrossEncoder / token-overlap fallback)  │
│  • Scores every candidate against the query         │
│  • Adaptive floor cutoff filters weak results       │
│  • Expands to parent chunks for full context        │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 2 — Evidence Collection  (Ollama / fallback) │
│  • Reads only grounded context passages             │
│  • Reports coverage: full / partial / none          │
│  • Structures evidence into clean human context     │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 3 — Final Answer Generation  (Groq stream)   │
│  • Streams the answer in natural Hinglish           │
│  • Cites source labels from evidence                │
│  • Falls back to Ollama → offline text if Groq down │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
                     Final Answer
```

### Document Ingestion

When you ingest a document, AVAJ:
1. Detects the file type (`.txt`, `.md`, `.html`, `.json`, `.jsonl`, `.pdf`)
2. Extracts clean text — including HTML table → Markdown conversion and PDF page extraction
3. Splits the text into **parent chunks** (~1400 tokens) and **child chunks** (~280 tokens with 50-token overlap)
4. SHA-256 hashes each document to skip duplicates automatically
5. Stores parents, children, and an index ledger as JSON on disk
6. Builds a ChromaDB vector collection for semantic search (if enabled)

### Supported File Types

| Format | Notes |
|--------|-------|
| `.pdf` | pdfplumber (with table extraction) → pypdf fallback |
| `.html` / `.htm` | BeautifulSoup, strips nav/ads, renders tables as Markdown |
| `.json` / `.jsonl` | Parsed and pretty-printed |
| `.md` / `.txt` | Plain text |

---

## Project Structure

```
AVAJ-main/
├── main.py                  # CLI entry point (ingest / ask / web / cli)
├── rag_engine.py            # Core 3-layer RAG orchestrator
├── requirements.txt
├── .env                     # All environment variables
│
├── config/
│   └── settings.py          # Pydantic settings, env var binding
│
├── ingestion/
│   ├── loaders.py           # File parsers (PDF, HTML, JSON, MD, TXT)
│   ├── pipeline.py          # Chunking, SHA dedup, JSON store
│   └── models.py            # DocumentRecord, ParentChunk, ChildChunk
│
├── retrieval/
│   ├── hybrid_search.py     # BM25 + ChromaDB dense + RRF fusion
│   ├── reranker.py          # CrossEncoder reranker (token-overlap fallback)
│   └── query_engine.py      # Query expansion service
│
├── providers/
│   ├── groq.py              # Groq streaming (Layer 3, AVAJ system prompt)
│   └── ollama.py            # Ollama (Layer 1 prompt-engineer, Layer 2 collect)
│
├── observability/
│   └── telemetry.py         # AIWorkLog, StageTimer (per-query debug log)
│
├── ui/
│   ├── app.py               # Streamlit web UI (chat + document manager)
│   └── cli.py               # Rich terminal dashboard
│
├── data/
│   ├── parents.json         # Indexed parent chunks
│   ├── children.json        # Indexed child chunks
│   ├── index_ledger.json    # Document SHA registry
│   └── uploads/             # Uploaded source documents
│
└── collection/              # Raw NPGC source files (PDFs, HTMLs, JSONs)
```

---

## Setup — Complete Guide

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or 3.12 recommended |
| pip | Latest |
| (Optional) Ollama | For local LLM layers 1 & 2 |
| (Optional) Groq account | For cloud-streamed Layer 3 answers |

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AVAJ.git
cd AVAJ
```

---

### Step 2 — Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `rich` | Terminal dashboard |
| `chromadb` | Vector store for semantic search |
| `sentence-transformers` | BGE-M3 embeddings + CrossEncoder reranker |
| `rank-bm25` | Keyword (sparse) search |
| `pdfplumber`, `pypdf` | PDF text and table extraction |
| `beautifulsoup4` | HTML parsing and table extraction |
| `groq` | Groq cloud LLM client |
| `ollama` | Ollama local LLM client |
| `pydantic`, `python-dotenv` | Config and env management |

---

### Step 4 — Configure Environment Variables

Copy the sample and fill in your keys:

```bash
cp .env .env.local
```

Or edit `.env` directly:

```env
# ── Provider Keys ──────────────────────────────────────────────────────────────
# Leave blank to run fully offline with BM25 + fallback generation
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here        # Only needed for gated HF models

# ── Ollama (Local LLM) ─────────────────────────────────────────────────────────
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b                     # Layer 2 model (evidence collection)
OLLAMA_LAYER1_MODEL=gemma2:2b               # Layer 1 model (query intelligence)

# ── Groq (Cloud LLM) ───────────────────────────────────────────────────────────
GROQ_MODEL=llama-3.3-70b-versatile

# ── Storage ────────────────────────────────────────────────────────────────────
RAG_DATA_DIR=data
RAG_UPLOAD_DIR=data/uploads

# ── Retrieval Tuning ───────────────────────────────────────────────────────────
RAG_TOP_K=8                                 # Candidates to retrieve before rerank
RAG_HYBRID_ALPHA=0.5                        # 0 = BM25 only, 1 = dense only
RAG_SEARCH_HINT_BOOST=1.5                   # Weight boost for Layer 1 search hints
RAG_RERANK_CUTOFF=-2.0                      # Min rerank score to pass
RAG_PARENT_CHUNK_TOKENS=1400
RAG_CHILD_CHUNK_TOKENS=280
RAG_CHILD_OVERLAP_TOKENS=50
RAG_MAX_CONTEXT_PARENT_CHUNKS=6
RAG_MAX_CONTEXT_CHARACTERS=9000

# ── Timeouts ───────────────────────────────────────────────────────────────────
LAYER1_TIMEOUT_SECONDS=120
LAYER2_TIMEOUT_SECONDS=120
RAG_TIMEOUT_SECONDS=120

# ── Embedding & Reranking ──────────────────────────────────────────────────────
# Set RAG_ENABLE_CHROMA=true after the BGE-M3 model has been downloaded
RAG_ENABLE_CHROMA=true
RAG_ENABLE_RERANKER=false                   # Set true after CrossEncoder is cached
EMBEDDING_MODEL=BAAI/bge-m3                 # Free multilingual model (EN/HI/Hinglish)
EMBEDDING_BATCH_SIZE=16
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
```

**Get a free Groq API key at:** https://console.groq.com

---

### Step 5 — (Optional) Set Up Ollama for Local LLMs

Install Ollama from https://ollama.com, then pull the models used by AVAJ:

```bash
ollama pull qwen2.5:3b       # Layer 2 — evidence collection
ollama pull gemma2:2b        # Layer 1 — query intelligence
```

Verify Ollama is running:
```bash
ollama list
```

> **Note:** AVAJ works without Ollama. When Ollama is unavailable, both Layer 1 and Layer 2 fall back to deterministic heuristics so the pipeline never crashes.

---

### Step 6 — Index Your Documents

Ingest any NPGC document (PDF, HTML, JSON, Markdown, TXT):

```bash
# Single file
python main.py ingest collection/College_brocher.pdf

# Force re-index an already indexed file
python main.py ingest collection/UG-BCA_NEP.pdf --force
```

The first ingest with `RAG_ENABLE_CHROMA=true` will download the `BAAI/bge-m3` model (~570 MB). This only happens once; the model is cached locally by SentenceTransformers.

---

### Step 7 — Run AVAJ

**Option A — Streamlit Web UI (recommended)**
```bash
python main.py web
```
Opens at `http://localhost:8501`. Features:
- Chat interface with streaming responses
- Document upload and management panel
- Live AI Work Log with retrieval stats, rerank ledger, and latency breakdown
- Sidebar sliders for Top-K, hybrid alpha, and rerank cutoff

**Option B — Rich Terminal Dashboard**
```bash
python main.py
```
Interactive terminal with commands:
```
/ingest <path>     Index a document
/docs              List all indexed documents
/ask <question>    Ask a question
/quit              Exit
```

**Option C — Single Question (non-interactive)**
```bash
python main.py ask "BCA ke liye admission process kya hai?"
```

---

### Step 8 — Verify the Setup

```bash
# Check what's indexed
python main.py ask "NPGC mein kaunse courses available hain?"
```

You should see a Hinglish response based on your indexed documents.

---

## Offline / Minimal Mode

AVAJ is designed to work without any API keys or Ollama:

| What's missing | What happens |
|----------------|--------------|
| No Groq key | Layer 3 falls back to Ollama streaming |
| No Ollama | Layers 1 & 2 use deterministic fallbacks; Layer 3 returns best indexed evidence as text |
| No ChromaDB / Chroma disabled | Semantic search uses Jaccard token overlap (no model download needed) |
| No reranker | Token-overlap scoring replaces CrossEncoder |

Set for instant offline start:
```env
RAG_ENABLE_CHROMA=false
RAG_ENABLE_RERANKER=false
```

---

## Key Features & Benefits

### 🔒 Domain-Locked — Zero Hallucination Risk
AVAJ refuses all queries unrelated to NPGC at the Layer 1 classification stage. It never fabricates information — if the answer is not in the indexed documents, it says so clearly.

### 🌐 Multilingual by Default
The default embedding model `BAAI/bge-m3` supports English, Hindi, and Hinglish natively. Queries and documents can freely mix scripts and languages.

### ⚡ Hybrid Search
Combines BM25 keyword search (exact term matching) with BGE-M3 semantic vector search (meaning-based matching), fused via Reciprocal Rank Fusion. Neither approach alone is as reliable as both together.

### 🧠 3-Layer AI Pipeline
Each layer has a deterministic fallback. The system is guaranteed to return an answer regardless of provider availability — from high-quality Groq streams down to offline indexed excerpts.

### 📊 Full Observability
Every query produces an `AIWorkLog` containing: expanded queries, retrieval stats (dense hits, sparse hits, fused candidates), rerank ledger with per-chunk scores, evidence audit, selected parent chunks, and per-stage latency in milliseconds. Visible in the Streamlit AI Work Log panel.

### 🗂️ Parent-Child Chunking
Documents are split into large parent chunks (for context) and small child chunks (for precise retrieval). Retrieval finds the most relevant child chunks, then expands back to parent chunks for richer context — balancing precision and completeness.

### 🔄 Automatic Deduplication
SHA-256 hashing prevents the same document from being indexed twice. Re-ingesting with `--force` cleanly replaces the old entry.

### 📁 Multiple Ingest Formats
PDFs with embedded tables, HTML college pages, JSON/JSONL data exports, Markdown notes, and plain text are all ingested through a unified pipeline.

### 🖥️ Two Interfaces
The Streamlit web UI is suitable for demos and daily use. The Rich terminal dashboard is suited for development and server environments with no GUI.

### 💾 No Database Required
All data (chunks, vectors, ledger) is stored as JSON files and a local ChromaDB directory. No PostgreSQL, Redis, or external services needed.

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key for Layer 3 generation |
| `HF_TOKEN` | — | HuggingFace token (only for gated models) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `llama3` | Model for Layer 2 (evidence collection) |
| `OLLAMA_LAYER1_MODEL` | same as `OLLAMA_MODEL` | Model for Layer 1 (query intelligence) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model for Layer 3 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | SentenceTransformer model for dense search |
| `EMBEDDING_BATCH_SIZE` | `16` | Batch size for embedding generation |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | CrossEncoder reranker model |
| `RAG_ENABLE_CHROMA` | `false` | Enable ChromaDB semantic search |
| `RAG_ENABLE_RERANKER` | `false` | Enable CrossEncoder reranker |
| `RAG_TOP_K` | `8` | Candidates retrieved before reranking |
| `RAG_HYBRID_ALPHA` | `0.5` | Dense/sparse weight balance (0=BM25, 1=dense) |
| `RAG_SEARCH_HINT_BOOST` | `1.5` | Weight multiplier for Layer 1 search hints |
| `RAG_RERANK_CUTOFF` | `-2.0` | Minimum rerank score to pass through |
| `RAG_PARENT_CHUNK_TOKENS` | `1400` | Token size of parent (context) chunks |
| `RAG_CHILD_CHUNK_TOKENS` | `280` | Token size of child (retrieval) chunks |
| `RAG_CHILD_OVERLAP_TOKENS` | `50` | Overlap between consecutive child chunks |
| `RAG_MAX_CONTEXT_PARENT_CHUNKS` | `6` | Max parent chunks sent to Layer 2 |
| `RAG_MAX_CONTEXT_CHARACTERS` | `9000` | Max characters of context per query |
| `LAYER1_TIMEOUT_SECONDS` | `15` | Timeout for Layer 1 Ollama call |
| `LAYER2_TIMEOUT_SECONDS` | `30` | Timeout for Layer 2 Ollama call |
| `RAG_DATA_DIR` | `data` | Directory for JSON stores and ChromaDB |
| `RAG_UPLOAD_DIR` | `data/uploads` | Directory for uploaded documents |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10 / 3.12 |
| Web UI | Streamlit |
| Terminal UI | Rich |
| Vector Store | ChromaDB |
| Embeddings | BAAI/bge-m3 (SentenceTransformers) |
| Keyword Search | BM25Okapi (rank-bm25) |
| Reranker | CrossEncoder ms-marco-MiniLM-L-12-v2 |
| Local LLM | Ollama (qwen2.5:3b, gemma2:2b) |
| Cloud LLM | Groq (llama-3.3-70b-versatile) |
| PDF Parsing | pdfplumber + pypdf |
| HTML Parsing | BeautifulSoup4 |
| Config | Pydantic v2 + python-dotenv |

---

## Troubleshooting

**`RAG_ENABLE_CHROMA=true` but no semantic search happening**
→ The BGE-M3 model needs to download on first run. This takes a few minutes and requires internet access. After it's cached, subsequent starts are instant.

**Ollama not connecting**
→ Run `ollama serve` in a separate terminal. Confirm with `curl http://localhost:11434/api/tags`.

**Groq generation failing**
→ Check your `GROQ_API_KEY` in `.env`. AVAJ will automatically fall back to Ollama or offline mode.

**PDF not extracting tables**
→ Ensure `pdfplumber` is installed (`pip install pdfplumber`). Some scanned PDFs have no text layer — only digitally-created PDFs extract cleanly.

**Old vectors after changing `EMBEDDING_MODEL`**
→ AVAJ uses a model-specific ChromaDB collection name derived from the model's SHA. Changing the model automatically creates a new collection and re-embeds all documents. No manual cleanup needed.

---

## License

This project was developed as part of an internship at **Vastu House Finance Company** and serves **National Post Graduate College, Lucknow**.

---

## Author

**Abhishek**
BCA Student — National Post Graduate College, University of Lucknow
SDE Intern — Vastu House Finance Company
