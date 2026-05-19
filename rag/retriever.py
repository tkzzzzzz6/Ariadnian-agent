from pathlib import Path

from rag.document_loader import DocumentLoader
from rag.embedder import Embedder
from rag.vector_store import VectorStore


class RAGRetriever:
    """End-to-end RAG retriever: ingest documents and retrieve relevant chunks."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        collection_name: str = "default",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or Embedder()
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, file_path: str | Path) -> int:
        """Load, chunk, and store a document. Returns number of chunks ingested."""
        path = Path(file_path)
        chunks = DocumentLoader.load_and_chunk(
            path,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        if not chunks:
            return 0
        return self.vector_store.add_chunks(
            chunks=chunks,
            source=str(path.name),
            collection_name=self.collection_name,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve top-k relevant text chunks for a query."""
        results = self.vector_store.search(
            query=query,
            collection_name=self.collection_name,
            top_k=top_k,
        )
        return [r["document"] for r in results]

    def reset(self):
        """Clear all documents from the collection."""
        self.vector_store.clear_collection(self.collection_name)
