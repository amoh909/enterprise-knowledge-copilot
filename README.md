# Enterprise Knowledge Copilot

Production-oriented RAG backend for answering questions over
enterprise documents using an LLM and vector retrieval.

## Overview

Enterprise Knowledge Copilot is a FastAPI-based retrieval-augmented
generation system that allows users to query an organization's
internal knowledge base.

The system retrieves relevant document content before generating
an answer, reducing reliance on the model's parametric knowledge
and helping prevent unsupported answers.

## Architecture

Documents
    ↓
Document ingestion
    ↓
Vector store
    ↓
Semantic retrieval
    ↓
LLM
    ↓
Grounded response
    ↓
REST API

## Features

- Document ingestion
- Vector-based retrieval
- Retrieval-augmented generation
- LLM API integration
- Grounded question answering
- Hallucination-aware prompting
- REST API
- OpenAPI/Swagger documentation
- Dockerized deployment
- Lightweight evaluation set

## Technology

- Python
- FastAPI
- Groq API (OpenAI GPT-OSS Open-Weight Reasoning Models)
- Local Document File Loader (Memory-efficient text block index)
- Docker & Docker Compose
- Pydantic
- Uvicorn

## Example

POST /ask

{
  "question": "How many annual leave days do employees receive?"
}

Response:

{
  "answer": "Full-time employees receive 20 days of paid annual leave."
}

## Running locally

### Prerequisites
- Docker Desktop installed and running
- A Groq API Key (Get one for free at [://groq.com](https://://groq.com))

### Setup & Deployment
1. **Clone the repository** and navigate to the project root.
2. **Configure Environment Variables:** Create a `.env` file in the root directory and add your Groq key:
   ```env
   GROQ_API_KEY=your_actual_free_groq_api_key_here
   ```
3. **Populate Knowledge Base:** Place your enterprise text (`.txt` or `.md`) files inside the local `documents/` directory.
4. **Spin up the Docker Container:**
   ```bash
   docker compose up --build
   ```
5. **Explore the API:** Once running, open your browser and navigate to `http://localhost:8000/docs` to interact with the Swagger UI panel.

## Evaluation Pipeline

The project includes an automated test runner script to evaluate RAG alignment and verify retrieval accuracy against ground-truth datasets.

To run the evaluation suite locally:
1. Ensure your local virtual environment is active.
2. Install the local development-specific testing requirements:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Execute the automated test runner (while your Docker API container is actively running):
   ```bash
   python evaluation/run_eval.py
   ```
This will sequentially feed test cases from `evaluation/test_questions.json` directly into your endpoint and output a side-by-side assertion analysis comparing expected facts with live engine outputs.

## Limitations

This project uses synthetic documents for demonstration purposes.
It is a minimal reference implementation rather than a production
enterprise deployment.

## Future Improvements

- PostgreSQL-backed conversation history
- Authentication and authorization
- Streaming responses
- Retrieval evaluation metrics
- Hybrid keyword/vector retrieval
- Document-level access control
- Observability and tracing
- Asynchronous ingestion