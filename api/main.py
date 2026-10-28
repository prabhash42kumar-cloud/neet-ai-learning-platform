import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="NEET AI Tutor API", version="0.1.0")

# For production, replace "*" with your exact website domain.
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:1b")

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

@app.get("/")
def home():
    return {"service": "NEET AI Tutor API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/ask")
def ask(payload: AskRequest):
    prompt = (
        "You are a careful tutor for Indian students preparing for NEET. "
        "Answer in simple Hindi unless the student asks for another language. "
        "Explain science step by step, distinguish facts from uncertainty, "
        "and do not invent textbook citations.\n\n"
        f"Student question: {payload.question}"
    )
    try:
        response = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
        if not answer:
            raise HTTPException(status_code=502, detail="Model returned an empty answer.")
        return {"answer": answer, "model": MODEL_NAME}
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="AI took too long. Please try again.")
    except requests.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Ollama/model server is not reachable. Check that Ollama is running."
        )
