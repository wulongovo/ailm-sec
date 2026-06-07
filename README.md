# AILLM-SEC

AI-Powered Security Analysis Assistant with RAG Knowledge Base

基于 RAG（检索增强生成）架构的企业级安全知识问答系统。

## Features

- **RAG Knowledge Base**: ChromaDB vector store + BGE-M3 embeddings
- **LLM Integration**: Ollama local inference / OpenAI API
- **Document Ingestion**: CVE data, Q&A pairs, markdown documents
- **Web UI**: Dark-themed chat interface
- **RESTful API**: FastAPI with auto-generated Swagger docs
- **Docker Deployment**: One-command deployment

## Quick Start

### Prerequisites

- Python 3.10+
- Ollama (for local LLM)

```bash
ollama pull qwen2.5
ollama pull bge-m3
```

### Install & Run

```bash
pip install -r requirements.txt
python main.py
# Open http://localhost:8000/api/ui
```

### Ingest Data

```bash
# Ingest Q&A training data
python scripts/ingest_data.py --qa /path/to/security_qa.jsonl

# Ingest CVE data
python scripts/ingest_data.py --cve /path/to/cve_data.jsonl

# Ingest markdown docs
python scripts/ingest_data.py --markdown /path/to/docs/
```

### Docker

```bash
docker-compose up -d
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | Health check |
| GET | /api/stats | System statistics |
| POST | /api/chat | Chat with AI assistant |
| POST | /api/ingest | Ingest documents |
| GET | /api/search?q=... | Search knowledge base |
| GET | /api/ui | Web interface |
| GET | /docs | Swagger API docs |

## Architecture

```
User Query
    |
    v
[FastAPI Backend]
    |
    +---> [ChromaDB Vector Store] -- retrieve top-5 docs
    |
    +---> [LLM Engine] -- generate answer with context
    |
    v
Response (answer + sources)
```

## Project Structure

```
ailm-sec/
  api/main.py           # FastAPI backend
  core/
    embeddings.py       # Embedding service (Ollama/OpenAI)
    rag.py              # RAG knowledge base (ChromaDB)
    llm.py              # LLM engine with RAG context
    ingestion.py        # Document ingestion pipeline
  frontend/index.html   # Web UI
  scripts/ingest_data.py
  tests/test_api.py
  Dockerfile
  docker-compose.yml
  .github/workflows/ci.yml
```

## License

MIT
