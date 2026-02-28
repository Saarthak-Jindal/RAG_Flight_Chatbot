"""
server.py - FastAPI REST server for the RAG Flight Chatbot.

Endpoints:
  POST   /chat                → Ask a flight question
  POST   /ingest              → Load flight data into ChromaDB
  GET    /health              → Health + DB status
  GET    /sessions/{id}       → View chat history
  DELETE /sessions/{id}       → Clear session memory
  GET    /sessions            → List active sessions

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import uuid
import json
from typing import Optional
from contextlib import asynccontextmanager

import cohere
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import cfg
from memory import memory_manager
from rag_query import rag_query
from ingest_flights import (
    get_chroma_client,
    get_or_create_collection,
    ingest_flights,
    generate_embeddings,
)
from flight_api import fetch_flights, normalize_flights, load_from_json


# ── Pydantic models ───────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Flight question from the user")
    session_id: Optional[str] = Field(None, description="Session ID for memory. Auto-generated if omitted.")
    top_k_retrieve: Optional[int] = Field(None, description="Override ChromaDB retrieval count")
    top_k_rerank: Optional[int] = Field(None, description="Override rerank top-N")


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    query: str


class IngestRequest(BaseModel):
    origin: str = Field("DEL", description="IATA departure code")
    destination: str = Field("BOM", description="IATA arrival code")
    date: str = Field("25032026", description="DDMMYYYY format")
    adults: int = Field(1, ge=1)
    seat_class: str = Field("ECONOMY")
    clear_existing: bool = Field(False, description="Wipe DB before ingesting")


class IngestResponse(BaseModel):
    status: str
    documents_added: int
    total_in_db: int
    message: str


class SessionResponse(BaseModel):
    session_id: str
    message_count: int
    messages: list[dict]


# ── App setup ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        cfg.validate()
        print("[server] ✅ API keys validated.")
    except EnvironmentError as e:
        print(f"[server] ⚠ {e}")
    yield


app = FastAPI(
    title="RAG Flight Chatbot API",
    description=(
        "Semantic flight search powered by:\n"
        "- Cohere embeddings (embed-english-v3.0)\n"
        "- ChromaDB vector store\n"
        "- Cohere Rerank (rerank-english-v3.0)\n"
        "- Groq LLM (llama3-70b-8192)\n"
        "- BudgetAir compare API (tripsaverz.in)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """Server + DB health check."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    return {
        "status": "healthy",
        "chroma_documents": collection.count(),
        "active_sessions": memory_manager.count(),
        "groq_model": cfg.GROQ_MODEL,
        "embed_model": cfg.COHERE_EMBED_MODEL,
        "rerank_model": cfg.COHERE_RERANK_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Send a flight query and get an AI-generated answer.

    Use the same session_id across requests to enable follow-up questions.

    Example body:
        {
          "query": "What is the cheapest non-stop flight from Delhi to Mumbai?",
          "session_id": "user123"
        }

    Follow-up (same session_id):
        {
          "query": "Is it refundable?",
          "session_id": "user123"
        }
    """
    session_id = req.session_id or str(uuid.uuid4())[:8]

    try:
        result = rag_query(
            query=req.query,
            session_id=session_id,
            top_k_retrieve=req.top_k_retrieve,
            top_k_rerank=req.top_k_rerank,
        )
        return ChatResponse(**result)

    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"[server] ❌ /chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """
    Fetch flights from BudgetAir API and ingest into ChromaDB.

    Tip: You can also ingest from a saved file via CLI:
        python ingest_flights.py --file flight_response.json
    """
    try:
        cfg.validate()
        co = cohere.Client(cfg.COHERE_API_KEY)
        client = get_chroma_client()
        collection = get_or_create_collection(client)

        raw = fetch_flights(
            origin=req.origin,
            destination=req.destination,
            doj=req.date,
            adults=req.adults,
            seating_class=req.seat_class,
        )

        flights = normalize_flights(raw, req.origin, req.destination)

        if not flights:
            return IngestResponse(
                status="warning",
                documents_added=0,
                total_in_db=collection.count(),
                message="No flights found in API response.",
            )

        added = ingest_flights(flights, collection, co, clear_first=req.clear_existing)
        return IngestResponse(
            status="success",
            documents_added=added,
            total_in_db=collection.count(),
            message=f"Successfully ingested {added} flight documents.",
        )

    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"[server] ❌ /ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """View full conversation history for a session."""
    session = memory_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    messages = [
        {
            "role": m.role,
            "content": m.content,
            "timestamp": m.timestamp,
            "context_flights": m.context_flight_ids,
        }
        for m in session.messages
    ]

    return SessionResponse(
        session_id=session_id,
        message_count=session.message_count,
        messages=messages,
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Clear a session's conversation memory."""
    deleted = memory_manager.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "success", "message": f"Session '{session_id}' deleted."}


@app.get("/sessions")
async def list_sessions():
    """List all active session IDs."""
    return {
        "sessions": memory_manager.all_session_ids(),
        "count": memory_manager.count(),
    }


# ── Entry point ───────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=cfg.SERVER_HOST,
        port=cfg.SERVER_PORT,
        reload=True,
    )
