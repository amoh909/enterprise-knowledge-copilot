from fastapi import FastAPI
from app.rag import answer_question
from app.schemas import QuestionRequest, QuestionResponse

app = FastAPI(
    title="Enterprise Knowledge Copilot",
    description="RAG-powered enterprise knowledge assistant",
    version="1.0.0",
)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    answer_text = answer_question(request.question)
    return QuestionResponse(answer=answer_text)