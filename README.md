# Enterprise Local RAG (AVAJ)

A local-first Retrieval-Augmented Generation system with ingestion, hybrid retrieval, reranking, provider fallback, Streamlit UI, Rich CLI, and an AI Work Log for debugging every query.

![GitHub License](https://img.shields.io/github/license/chetanthepycoder/AVAJ-Two_layar_Rag_System?style=plastic&color=green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-FF5A00?logo=ollama)
![Groq](https://img.shields.io/badge/Groq-%23FFFFFF.svg?logo=grogo&logoColor=black)

## Features

- **Local-first**: Designed to run entirely on local infrastructure with optional cloud fallback.
- **Two-layer RAG architecture**:
  - Layer 1: Query understanding and optimization using Ollama (local LLM).
  - Layer 2: Evidence-based answer generation with fallback to Groq (fast cloud LLM) for final synthesis.
- **Hybrid retrieval**: Combines dense and sparse search for robust retrieval.
- **Reranking**: Uses a cross-encoder to improve relevance of retrieved chunks.
- **Provider fallback**: Automatically falls back to alternative providers if the primary fails.
- **Observability**: Detailed AI Work Log for every query, showing query expansion, retrieval, reranking, and reasoning.
- **Multiple interfaces**:
  - Command-line interface (CLI) with Rich for interactive use.
  - Web interface built with Streamlit.
  - Python API for programmatic use.
- **Flexible configuration**: Environment variables for easy customization.
- **Multilingual support**: Uses BAAI/bge-m3 embedding model for English, Hindi, and Hinglish.
- **Document ingestion**: Supports various file formats for building your knowledge base.

## Architecture

AVAJ implements a three-layer answer flow to ensure accurate, grounded, and context-aware responses:

1. **Layer 1 (Query Understanding)**:
   - Uses Ollama (local LLM) to rewrite, optimize, and classify the user query.
   - Determines if the query is related to the knowledge domain (e.g., National PG College).
   - Extracts search hints and generates an optimized query for retrieval.

2. **Layer 2 (Retrieval and Reranking)**:
   - Expands the query using synonyms and related terms.
   - Performs hybrid search (dense + sparse) over the ingested documents.
   - Reranks the results using a cross-encoder model to prioritize the most relevant chunks.
   - Retrieves parent chunks for context preservation.

3. **Layer 3 (Answer Generation)**:
   - Uses Ollama to gather grounded facts from the retrieved context.
   - Uses Groq (or Ollama fallback) to generate the final answer, incorporating the original query, Layer 1 metadata, and Layer 2 structured context.
   - Each layer has deterministic fallbacks to ensure the system works offline or with limited resources.

![AVAJ Architecture](https://via.placeholder.com/800x400?text=AVAJ+Three-Layer+Rag+Architecture)

## Installation

### Prerequisites

- Python 3.10 or higher
- Git
- [Ollama](https://ollama.com/) (for local LLM embeddings and generation)
- [Groq](https://groq.com/) API key (optional, for faster inference)
- (Optional) GPU for faster embeddings (CUDA-enabled)

### Setup

1. **Clone the repository**:
   `ash
   git clone https://github.com/chetanprojects/AVAJ-Two-Layer_Rag-System.git
   cd AVAJ-Two-Layer_Rag-System
   `

2. **Create a virtual environment**:
   `ash
   python -m venv .venv
   # On Windows
   .\.venv\Scripts\Activate.ps1
   # On Linux/macOS
   source .venv/bin/activate
   `

3. **Install dependencies**:
   `ash
   pip install -r requirements.txt
   `

4. **Install Ollama** (if not already installed):
   - Download and install from [https://ollama.com/](https://ollama.com/)
   - Pull the required model (default is llama3):
     `ash
     ollama pull llama3
     `

5. **Set up environment variables** (optional but recommended):
   Create a .env file in the project root with the following variables:
   `	ext
   GROQ_API_KEY=your_groq_api_key_here
   HF_TOKEN=your_huggingface_token_here  # optional, for accessing gated models
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_MODEL=llama3
   OLLAMA_LAYER1_MODEL=llama3
   GROQ_MODEL=llama-3.3-70b-versatile
   LAYER1_TIMEOUT_SECONDS=15
   LAYER2_TIMEOUT_SECONDS=30
   RAG_DATA_DIR=data
   RAG_ENABLE_CHROMA=false
   RAG_ENABLE_RERANKER=false
   RAG_SEARCH_HINT_BOOST=1.5
   EMBEDDING_MODEL=BAAI/bge-m3
   EMBEDDING_BATCH_SIZE=16
   `
   - Set RAG_ENABLE_CHROMA=true and RAG_ENABLE_RERANKER=true to use local embedding and cross-encoder models (requires additional dependencies and model downloads).
   - Get a Groq API key from [https://console.groq.com/](https://console.groq.com/) for faster LLM inference.

6. **Ingest your documents**:
   `ash
   python main.py ingest path/to/your/document.pdf
   `
   Supported formats: PDF, DOCX, TXT, MD, and more (via Unstructured.io).

## Usage

### Command Line Interface (CLI)

Start the interactive CLI dashboard:
   `ash
   python main.py cli
   `

Commands:
   - /ingest <path>: Ingest a document or directory.
   - /docs: List all indexed documents.
   - /ask <question>: Ask a question to the RAG system.
   - /help: Show help.
   - /quit or /exit: Exit the CLI.

Example:
   `ash
   python main.py ingest sample_docs/faculty.md
   python main.py ask "Who is in the Computing department?"
   `

### Web Interface (Streamlit)

Launch the Streamlit web app:
   `ash
   python main.py web
   `
   Then open your browser to http://localhost:8501.

The web interface provides:
   - A chat interface for asking questions.
   - File upload for ingesting new documents.
   - An "AI Work Log" expander to inspect the internal reasoning steps for each answer.

### Python API

You can also use AVAJ as a library in your Python code:
   `python
   from rag_engine import RAGEngine
   from config import get_settings

   settings = get_settings()
   engine = RAGEngine(settings)

   for event in engine.ask("Your question here"):
       if event["type"] == "token":
           print(event["token"], end="", flush=True)
       elif event["type"] == "log":
           # Process the work log for debugging
           pass
   `

## How It Works

### Three-Layer Answer Flow

When you ask a question, AVAJ processes it through three distinct layers:

1. **Layer 1: Query Understanding**
   - The query is sent to Ollama (or a fallback provider) to:
     - Rewrite the query for clarity.
     - Classify the query domain (e.g., is it about the college?).
     - Extract search hints (keywords to boost in retrieval).
     - Determine the expected answer format (e.g., list, direct answer, summary).

2. **Layer 2: Retrieval and Evidence Collection**
   - The query is expanded with synonyms and related terms.
   - Hybrid search (combining vector similarity and keyword matching) retrieves candidate chunks.
   - A cross-encoder reranker scores the chunks for relevance.
   - The top chunks are passed to Layer 2 of the LLM to extract only grounded facts, reducing hallucination.

3. **Layer 3: Answer Generation**
   - The original query, Layer 1 metadata (intent, domain, format), and Layer 2 structured context are sent to Groq (or Ollama) to generate the final answer.
   - The answer is streamed back token-by-token for a responsive user experience.

### Fallback Mechanisms

- If Ollama is unavailable, Layer 1 and Layer 2 can fall back to Groq (if configured).
- If Groq is unavailable, the system can fall back to Ollama for the final answer.
- If both are unavailable, the system will use a rule-based response for Layer 1 classification and a simple template for Layer 3.

### Observability

Every interaction is logged in the AI Work Log, which includes:
   - Query expansion terms.
   - Retrieval statistics (dense/sparse scores, hybrid fusion).
   - Reranking ledger with scores and pass/fail status (passed/failed) for each candidate.
   - Reasoning summary from Layer 1.
   - Performance metrics for each stage.

This transparency allows users to debug and trust the system's outputs.

## Benefits

- **Privacy-first**: All data remains on your local machine unless you explicitly use external providers like Groq.
- **Cost-effective**: Uses free local models (Ollama) with optional paid API for speed.
- **Accurate**: The two-layer approach reduces hallucinations by grounding answers in retrieved evidence.
- **Flexible**: Swap between local and cloud providers based on availability and cost.
- **Transparent**: Full visibility into the AI's reasoning process via the Work Log.
- **Easy to use**: Simple CLI and intuitive web interface.
- **Scalable**: Designed to handle large document collections with efficient indexing and retrieval.

## Configuration

All configuration is done via environment variables or a .env file. Here's a breakdown:

| Variable | Description | Default |
|----------|-------------|---------|
| GROQ_API_KEY | API key for Groq (optional) | None |
| HF_TOKEN | Hugging Face token for gated models (optional) | None |
| OLLAMA_HOST | Host URL for Ollama | http://localhost:11434 |
| OLLAMA_MODEL | Model used for Ollama (general) | llama3 |
| OLLAMA_LAYER1_MODEL | Model used for Layer 1 (query understanding) | llama3 |
| GROQ_MODEL | Model used for Groq (final answer) | llama-3.3-70b-versatile |
| LAYER1_TIMEOUT_SECONDS | Timeout for Layer 1 Ollama calls | 15 |
| LAYER2_TIMEOUT_SECONDS | Timeout for Layer 2 Ollama calls | 30 |
| RAG_DATA_DIR | Directory for storing indexed data | data |
| RAG_UPLOAD_DIR | Directory for uploaded files (web UI) | data/uploads |
| RAG_ENABLE_CHROMA | Enable local Chroma vector store (requires chromadb and sentence-transformers) | alse |
| RAG_ENABLE_RERANKER | Enable local cross-encoder reranker (requires sentence-transformers and a reranker model) | alse |
| RAG_SEARCH_HINT_BOOST | Boost factor for search hints from Layer 1 | 1.5 |
| EMBEDDING_MODEL | SentenceTransformer model for embeddings | BAAI/bge-m3 |
| EMBEDDING_BATCH_SIZE | Batch size for embedding generation | 16 |

### Enabling Local Embeddings and Reranker

To use local embeddings and reranking (fully offline capable):

1. Install additional dependencies:
   `ash
   pip install chromadb sentence-transformers torch
   `
2. Set in .env:
   `	ext
   RAG_ENABLE_CHROMA=true
   RAG_ENABLE_RERANKER=true
   `
3. The system will automatically download the BAAI/bge-m3 embedding model and a cross-encoder model (default: cross-encoder/ms-marco-MiniLM-L-6-v2) on first run.

## Troubleshooting

- **Ollama not running**: Ensure Ollama is installed and the service is running. Run ollama serve in a separate terminal.
- **Groq API errors**: Check your GROQ_API_KEY and internet connection.
- **Slow first response**: The first run may take longer as models are downloaded and embedded.
- **No documents found**: Ensure you have ingested documents with python main.py ingest <path>.
- **CUDA errors**: If you don't have a GPU, the system will fall back to CPU. Ensure PyTorch is installed without CUDA if needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Ollama](https://ollama.com/) for making local LLMs accessible.
- [Groq](https://groq.com/) for fast inference.
- [Streamlit](https://streamlit.io/) for the web UI framework.
- [Rich](https://github.com/Textualize/rich) for beautiful CLI interfaces.
- [SentenceTransformers](https://www.sbert.net/) for embedding models.
- [Chroma](https://www.trychroma.com/) for vector storage.
- [Unstructured](https://unstructured-io.github.io/unstructured/) for document ingestion.
