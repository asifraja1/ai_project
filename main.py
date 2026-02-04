from fastapi import FastAPI
from pydantic import BaseModel

from rag_chain import rag_chain


# -------- Create API --------
app = FastAPI(
    title="Antigravity Cleaned RAG API"
)


# -------- Request Model --------
class QuestionRequest(BaseModel):
    question: str


# -------- Health Check --------
@app.get("/")
def home():
    return {"message": "API running "}


# -------- Ask Endpoint --------
@app.post("/ask")
def ask_question(req: QuestionRequest):

    response = rag_chain.invoke(req.question)

    return {
        "question": req.question,
        "answer": response.content
    }