import hashlib
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings


class VectorStore:
    """Lightweight ChromaDB wrapper for RAG."""

    def __init__(self, persist_dir: str | Path = "./chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def get_collection(self, name: str = "default"):
        return self.client.get_or_create_collection(name=name)

    def add_chunks(
        self,
        chunks: list[str],
        source: str,
        collection_name: str = "default",
    ) -> int:
        """Add text chunks to the vector store. Returns number of chunks added."""
        collection = self.get_collection(collection_name)
        ids = [hashlib.md5(f"{source}:{i}:{chunk[:64]}".encode()).hexdigest() for i, chunk in enumerate(chunks)]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        return len(chunks)

    def search(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int = 5,
    ) -> list[dict]:
        """Search the vector store and return top-k results."""
        collection = self.get_collection(collection_name)
        results = collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []
        return [
            {
                "document": doc,
                "metadata": meta,
                "distance": dist,
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def clear_collection(self, collection_name: str = "default"):
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
