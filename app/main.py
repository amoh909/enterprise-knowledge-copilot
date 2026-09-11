import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.rag import answer_question, build_index
from app.schemas import QuestionRequest, QuestionResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Verifying local vector database status...")
    index_path = os.path.join("data", "index.pkl")
    
    if not os.path.exists(index_path):
        print(f"[STARTUP] {index_path} is missing! Triggering automated semantic ingestion pipeline...")
        try:
            build_index()
            print("[STARTUP] Knowledge base vector index built and cached successfully!")
        except Exception as e:
            print(f"[STARTUP] CRITICAL ERROR: Ingestion failed during application boot: {e}")
    else:
        print("[STARTUP] Cached vector database located and verified. Ready for user queries.")
        
    yield
    print("[SHUTDOWN] Tearing down server contexts...")

app = FastAPI(
    title="Enterprise Knowledge Copilot",
    description="RAG-powered enterprise knowledge assistant",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    result = answer_question(request.question)

    return QuestionResponse(
        answer=result["answer"],
        sources=result["sources"],
    )
