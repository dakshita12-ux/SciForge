# SciForge

**Intelligent Scientific and Engineering Workflow Generation Using RAG and Multi-Agent AI**

## Overview

SciForge is a project aimed at helping users translate scientific and engineering problems into computational workflows using AI.

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
- Model: `openai/gpt-oss-120b`
- Environment-based API key configuration
- Scientific and engineering question answering
- Basic API error handling

## Technology Stack

- Python
- FastAPI
- Uvicorn
- Groq API
- GPT-OSS 120B
- Pydantic
- python-dotenv

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Create a `.env` file in the project root and configure `GROQ_API_KEY`.
4. Start the backend:

   `uvicorn backend.main:app --reload`

5. Open `http://127.0.0.1:8000/docs` to explore the API.

## Planned Features

- Retrieval-Augmented Generation (RAG)
- Multi-agent workflow planning
- Algorithm recommendation
- Code generation
- Controlled experiment execution
- Results analysis and report generation

## Development Status

This project is under active development. The current implementation provides a basic FastAPI backend connected to an LLM through Groq. RAG, multi-agent orchestration, and automated scientific workflow execution are planned for later stages.
