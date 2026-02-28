"""
rag_query.py - Full RAG pipeline: Embed → Retrieve → Rerank → Generate

Steps:
  1. Embed query with Cohere (input_type='search_query')
  2. Retrieve top-K from ChromaDB (cosine similarity)
  3. Rerank with Cohere Rerank for better relevance
  4. Build prompt: system + flight context + chat history + query
  5. Generate answer with Groq (llama3-70b)
  6. Save to session memory

Usage (CLI):
    python rag_query.py
"""

import json
import cohere
import chromadb
from chromadb.config import Settings
from groq import Groq

from config import cfg
from memory import memory_manager, SessionMemory


# ── Clients ──────────────────────────────────────────────

def get_clients():
    co = cohere.Client(cfg.COHERE_API_KEY)
    groq_client = Groq(api_key=cfg.GROQ_API_KEY)
    chroma_client = chromadb.PersistentClient(
        path=cfg.CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        name=cfg.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return co, groq_client, collection


# ── Step 1: Embed query ───────────────────────────────────

def embed_query(query: str, co: cohere.Client) -> list[float]:
    """
    Embed the user query.
    IMPORTANT: input_type='search_query' (different from 'search_document' used in ingestion).
    This asymmetric embedding is key for retrieval quality.
    """
    response = co.embed(
        texts=[query],
        model=cfg.COHERE_EMBED_MODEL,
        input_type="search_query",
    )
    return response.embeddings[0]


# ── Step 2: Retrieve from ChromaDB ───────────────────────

def retrieve_candidates(
    query_embedding: list[float],
    collection: chromadb.Collection,
    top_k: int = None,
) -> list[dict]:
    """Semantic search in ChromaDB. Returns top_k flight candidates."""
    k = top_k or cfg.TOP_K_RETRIEVE

    if collection.count() == 0:
        print("[rag] ⚠ ChromaDB is empty — run: python ingest_flights.py --file flight_response.json")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        candidates.append({
            "document": doc,         # text_chunk (natural language)
            "metadata": meta,        # all flight fields
            "similarity": 1 - dist,  # cosine distance → similarity
        })

    print(f"[rag] 🔍 Retrieved {len(candidates)} candidates (top similarity: {candidates[0]['similarity']:.4f})")
    return candidates


# ── Step 3: Rerank with Cohere ────────────────────────────

def rerank_results(
    query: str,
    candidates: list[dict],
    co: cohere.Client,
    top_n: int = None,
) -> list[dict]:
    """
    Rerank ChromaDB results using Cohere's cross-encoder reranker.
    This captures nuanced relevance that pure embedding similarity misses.
    E.g., 'refundable flights' will score refundable flights higher even if
    semantic similarity alone didn't distinguish them.
    """
    if not candidates:
        return []

    n = top_n or cfg.TOP_K_RERANK
    docs = [c["document"] for c in candidates]

    print(f"[rag] 🔄 Reranking {len(docs)} candidates → keeping top {n}...")

    response = co.rerank(
        query=query,
        documents=docs,
        model=cfg.COHERE_RERANK_MODEL,
        top_n=n,
    )

    reranked = []
    for result in response.results:
        orig = candidates[result.index]
        reranked.append({
            **orig,
            "rerank_score": result.relevance_score,
        })

    print(f"[rag] ✅ Top rerank score: {reranked[0]['rerank_score']:.4f}")
    return reranked


# ── Step 4: Build prompt ──────────────────────────────────

SYSTEM_PROMPT = """You are FlightBot, a smart travel assistant for BudgetAir (tripsaverz.in).

You help users find, compare, and understand flights using real-time data provided to you.

Guidelines:
- Answer ONLY using the flight data in the context. Do not make up flights.
- Use ₹ symbol for Indian Rupee prices.
- Mention airline name, flight code, departure/arrival times, duration, stops, price, and refund policy when relevant.
- For price comparisons, highlight the cheapest provider (EASEMYTRIP, AERTRIP, etc.) and price differences.
- When a user asks follow-up questions ("Is it refundable?", "How long is that?"), refer to conversation history to understand which flight they mean.
- If asking about non-stop vs with-stops, check the 'stops' field (0 = non-stop).
- Be concise, clear, and helpful. Bullet points are fine for comparisons.
- If you don't have enough data to answer, say so honestly.
"""


def build_prompt(
    query: str,
    reranked: list[dict],
    session: SessionMemory,
) -> list[dict]:
    """Assemble the full message list for Groq."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject flight context
    if reranked:
        context_lines = ["FLIGHT DATA (use this to answer the user):\n"]
        for i, doc in enumerate(reranked, 1):
            meta = doc["metadata"]
            # Also parse provider fares if available
            provider_info = ""
            try:
                fares = json.loads(meta.get("provider_fares", "{}"))
                if fares:
                    fare_parts = [f"{p}: ₹{v['total']}" for p, v in fares.items()]
                    provider_info = f" | Prices: {', '.join(fare_parts)}"
            except Exception:
                pass

            context_lines.append(f"[Option {i}] {doc['document']}{provider_info}")
            context_lines.append("")

        # Add recent flight context summary for follow-ups
        summary = session.get_context_summary()
        if summary and session.message_count > 2:
            context_lines.append(summary)

        messages.append({
            "role": "system",
            "content": "\n".join(context_lines),
        })

    # Add conversation history (memory window)
    history = session.get_chat_history()
    messages.extend(history)

    # Append current query if not already last in history
    last = next((m for m in reversed(history) if m["role"] == "user"), None)
    if not last or last["content"] != query:
        messages.append({"role": "user", "content": query})

    return messages


# ── Step 5: Generate answer ───────────────────────────────

def generate_answer(messages: list[dict], groq_client: Groq) -> str:
    response = groq_client.chat.completions.create(
        model=cfg.GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── Main RAG function ─────────────────────────────────────

def rag_query(
    query: str,
    session_id: str = "default",
    top_k_retrieve: int = None,
    top_k_rerank: int = None,
) -> dict:
    """
    Full RAG pipeline for a user flight query.

    Returns:
        {
          "answer": str,
          "sources": list of flight source dicts,
          "session_id": str,
          "query": str
        }
    """
    print(f"\n[rag] 💬 '{query}' (session: {session_id})")

    cfg.validate()
    co, groq_client, collection = get_clients()
    session = memory_manager.get_or_create(session_id)

    # Add to memory
    session.add_user_message(query)

    # Step 1: Embed
    print("[rag] 🔢 Embedding query...")
    query_vec = embed_query(query, co)

    # Step 2: Retrieve
    candidates = retrieve_candidates(query_vec, collection, top_k=top_k_retrieve)

    # Step 3: Rerank
    reranked = rerank_results(query, candidates, co, top_n=top_k_rerank) if candidates else []

    # Update memory context
    if reranked:
        session.update_flight_context([r["metadata"] for r in reranked])

    # Step 4: Build prompt
    messages = build_prompt(query, reranked, session)

    # Step 5: Generate
    print("[rag] 🤖 Generating answer via Groq...")
    answer = generate_answer(messages, groq_client)

    # Save to memory
    flight_ids = [r["metadata"].get("flight_id", "") for r in reranked]
    session.add_assistant_message(answer, flight_ids=flight_ids)

    # Build sources for API response
    sources = [
        {
            "flight_id": r["metadata"].get("flight_id"),
            "airline": r["metadata"].get("airline"),
            "flight_code": r["metadata"].get("flight_code"),
            "dep_time": r["metadata"].get("dep_time"),
            "arr_time": r["metadata"].get("arr_time"),
            "cheapest_fare": r["metadata"].get("cheapest_fare"),
            "cheapest_provider": r["metadata"].get("cheapest_provider"),
            "stops": r["metadata"].get("stops"),
            "is_refundable": r["metadata"].get("is_refundable"),
            "rerank_score": round(r.get("rerank_score", 0), 4),
        }
        for r in reranked
    ]

    print(f"[rag] ✅ Done ({len(answer)} chars)")
    return {
        "answer": answer,
        "sources": sources,
        "session_id": session_id,
        "query": query,
    }


# ── CLI interactive mode ──────────────────────────────────

if __name__ == "__main__":
    import uuid

    print("\n" + "=" * 60)
    print("  ✈  RAG Flight Chatbot — Interactive CLI")
    print("=" * 60)
    print("Commands: 'quit' to exit | 'clear' to reset memory\n")

    session_id = str(uuid.uuid4())[:8]
    print(f"Session: {session_id}\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "clear":
            memory_manager.get_or_create(session_id).clear()
            print("[Memory cleared]\n")
            continue

        result = rag_query(query, session_id=session_id)
        print(f"\nFlightBot: {result['answer']}")
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                stops = "Non-stop" if s.get("stops") == 0 else f"{s.get('stops')} stop(s)"
                print(f"  → {s['airline']} {s['flight_code']} | ₹{s['cheapest_fare']} | {stops} | Score: {s['rerank_score']}")
        print()
