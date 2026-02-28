"""
ingest_flights.py - Embeds flight documents into ChromaDB using Cohere.

Flow:
  1. Load flight data (from JSON file or live API)
  2. Normalize into flat documents using real field names
  3. Generate Cohere embeddings (input_type='search_document')
  4. Store in ChromaDB for semantic retrieval

Usage:
    python ingest_flights.py --file flight_response.json         # From Postman
    python ingest_flights.py --file flight_response.json --clear # Re-ingest fresh
    python ingest_flights.py --from DEL --to BOM --date 25032026 # Live API
"""

import time
import argparse

import cohere
import chromadb
from chromadb.config import Settings

from config import cfg
from flight_api import fetch_flights, normalize_flights, load_from_json


def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=cfg.CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=cfg.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def generate_embeddings(texts: list[str], co: cohere.Client) -> list[list[float]]:
    """Embed documents using Cohere. Batches to respect rate limits."""
    all_embeddings = []
    batch_size = 96

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        print(f"[ingest] Embedding batch {i // batch_size + 1} ({len(batch)} texts)...")

        response = co.embed(
            texts=batch,
            model=cfg.COHERE_EMBED_MODEL,
            input_type="search_document",  # MUST be 'search_document' for ingestion
        )
        all_embeddings.extend(response.embeddings)

        if i + batch_size < len(texts):
            time.sleep(0.3)

    return all_embeddings


def ingest_flights(
    flights: list[dict],
    collection: chromadb.Collection,
    co: cohere.Client,
    clear_first: bool = False,
) -> int:
    """Embed and store flight documents in ChromaDB."""
    if not flights:
        print("[ingest] ⚠ No flights to ingest.")
        return 0

    if clear_first:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"[ingest] 🗑 Cleared {len(existing['ids'])} existing documents.")

    # Check which IDs already exist
    existing_ids = set(collection.get()["ids"])
    new_flights = [f for f in flights if f["flight_id"] not in existing_ids]

    if not new_flights:
        print("[ingest] ℹ All flights already in DB. Use --clear to re-ingest.")
        return 0

    texts = [f["text_chunk"] for f in new_flights]
    ids = [f["flight_id"] for f in new_flights]

    print(f"[ingest] 🔢 Generating embeddings for {len(texts)} flights via Cohere...")
    embeddings = generate_embeddings(texts, co)

    # Clean metadata — ChromaDB only accepts str/int/float/bool values
    metadatas = []
    for f in new_flights:
        meta = {}
        for k, v in f.items():
            if k == "text_chunk":
                continue
            if isinstance(v, bool):
                meta[k] = v
            elif isinstance(v, (int, float)):
                meta[k] = v
            else:
                meta[k] = str(v) if v is not None else ""
        metadatas.append(meta)

    print(f"[ingest] 💾 Storing in ChromaDB collection '{cfg.CHROMA_COLLECTION_NAME}'...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    total = collection.count()
    print(f"[ingest] ✅ Done! Added {len(new_flights)} docs. Total in DB: {total}")
    return len(new_flights)


def main():
    parser = argparse.ArgumentParser(description="Ingest flight data into ChromaDB")
    parser.add_argument("--file", help="Path to Postman JSON response file")
    parser.add_argument("--clear", action="store_true", help="Clear DB before ingesting")
    parser.add_argument("--from", dest="origin", default="DEL")
    parser.add_argument("--to", dest="destination", default="BOM")
    parser.add_argument("--date", default="25032026")
    args = parser.parse_args()

    cfg.validate()

    print("\n" + "=" * 50)
    print("  ✈  RAG Flight Chatbot — Data Ingestion")
    print("=" * 50 + "\n")

    co = cohere.Client(cfg.COHERE_API_KEY)
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    print(f"[ingest] Current DB size: {collection.count()} documents\n")

    # Load data
    if args.file:
        raw = load_from_json(args.file)
    else:
        raw = fetch_flights(origin=args.origin, destination=args.destination, doj=args.date)

    flights = normalize_flights(raw, args.origin, args.destination)

    if not flights:
        print("[ingest] ❌ No flights found. Exiting.")
        return

    print(f"\n[ingest] Sample flight text chunk:\n{flights[0]['text_chunk']}\n")

    added = ingest_flights(flights, collection, co, clear_first=args.clear)

    print(f"\n[ingest] 🎉 Ingestion complete! Added {added} new documents.")
    print(f"[ingest] Total in ChromaDB: {collection.count()}")


if __name__ == "__main__":
    main()
