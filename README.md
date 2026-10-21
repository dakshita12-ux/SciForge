# SciForge

**Intelligent Scientific and Engineering Workflow Generation Using RAG and Multi-Agent AI**

## Overview

SciForge is a developing AI-powered system designed to help users translate scientific and engineering problems into computational workflows.

## Current Progress

### Stage 0 — Project Setup
- Python virtual environment
- Project structure
- Git initialization

### Stage 1 — Basic Backend
- FastAPI application
- GET and POST endpoints
- Path parameters
- Request validation
- Interactive API documentation

### Stage 2 — LLM Integration
- Groq API integration
- `openai/gpt-oss-120b`
- Environment-based API key configuration
- Scientific and engineering question answering

### Stage 3 — Retrieval-Augmented Generation
- Text-document ingestion
- Word-based chunking with overlap
- Sentence Transformer embeddings
- FAISS vector index
- Semantic retrieval of relevant document chunks
- Context-grounded LLM responses
- Source filenames and retrieved passages in API responses
- Knowledge-status and retrieval-testing endpoints

## Technology Stack

- Python
- FastAPI
- Uvicorn
- Groq API
- GPT-OSS 120B
- Sentence Transformers
- FAISS
- Pydantic
- python-dotenv

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   `python -m pip install -r requirements.txt`

3. Create a `.env` file in the project root and configure `GROQ_API_KEY`.
4. Add `.txt` knowledge documents to the `knowledge/` folder.
5. Build the vector index:

   `python -m backend.ingest`

6. Start the backend:

   `uvicorn backend.main:app --reload`

7. Open `http://127.0.0.1:8000/docs` to explore the APIs.

## Main Endpoints

- `GET /` — Welcome message
- `GET /about` — Project information
- `GET /health` — Backend health check
- `GET /knowledge/status` — Knowledge-index status
- `POST /knowledge/search` — Retrieve relevant document chunks
- `POST /ask` — Generate an answer using retrieved context

## Planned Features

- Multi-agent workflow planning
- Algorithm recommendation
- Code generation
- Controlled experiment execution
- Results analysis and report generation
- Frontend integration

## Development Status

SciForge is under active development. Its current RAG implementation retrieves text passages from a local knowledge base and supplies them as context to an LLM. It does not yet support automatic PDF ingestion, multi-agent orchestration, or automated experiment execution.

## Stage 4 — Intelligent Research Planning

Stage 4 introduces a research planning engine that transforms natural-language scientific and engineering problems into structured research plans.

### Features

- Retrieves relevant information from the existing FAISS-based RAG pipeline.
- Uses Groq with `openai/gpt-oss-120b` to analyze the problem.
- Identifies objectives, task types, inputs, outputs, assumptions, and missing information.
- Recommends approaches, evaluation metrics, implementation steps, and potential risks.
- Generates structured JSON output for future frontend integration.
- Attaches metadata for the documents retrieved by the knowledge retrieval system.
- Handles invalid requests, missing API keys, and model-response parsing errors.

### API Endpoint

`POST /plan`

Example request:

```json
{
  "problem": "Predict house prices using machine learning and recommend suitable algorithms and evaluation metrics."
}
```

### Testing

Start the development server:

```bash
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the endpoint.

### Limitations

The generated research plans are suggestions and require review. Stage 4 does not automatically execute experiments, train models, browse the web, or verify every generated claim against source documents.
