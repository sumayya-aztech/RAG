from fastapi import FastAPI
from pydantic import BaseModel
from src.search import RAGSearch

app = FastAPI(title="AI Knowledge Assistant")
rag_search = RAGSearch()


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "AI Knowledge Assistant API is running"
    }


@app.post("/chat")
def chat(request: ChatRequest):
    result = rag_search.search_and_summarize(
        request.question,
        top_k=3
    )

    return result