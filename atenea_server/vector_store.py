from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Optional
import os
from .chunker import Chunk

class VectorStore:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.default_collection = "atenea_code"

    def _ensure_collection(self, collection_name: str):
        try:
            self.client.get_collection(collection_name)
        except Exception:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # nomic-embed-text dimension
                    distance=models.Distance.COSINE
                )
            )

    def list_collections(self) -> List[str]:
        response = self.client.get_collections()
        return [c.name for c in response.collections]

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]], collection_name: Optional[str] = None):
        if not chunks:
            return

        collection_name = collection_name or self.default_collection

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Use stable ID (deterministic MD5 of path + lines)
            id_input = f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}"
            from hashlib import md5
            point_id = md5(id_input.encode()).hexdigest()

            points.append(models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content": chunk.content,
                    "language": chunk.language
                }
            ))

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
        except Exception:
            # Collection may have been deleted externally; recreate and retry
            self._ensure_collection(collection_name)
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

    def search(self, query_vector: List[float], limit: int = 20, collection_name: Optional[str] = None) -> List[dict]:
        collection_name = collection_name or self.default_collection
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )
        return [hit.payload for hit in results.points]

    def clear_collection(self, collection_name: Optional[str] = None):
        """Delete and recreate the collection."""
        target = collection_name or self.default_collection
        try:
            self.client.delete_collection(target)
        except Exception:
            pass
        self._ensure_collection(target)

    def has_data(self, collection_name: Optional[str] = None) -> bool:
        """Check if the collection has any points."""
        target = collection_name or self.default_collection
        try:
            res = self.client.count(collection_name=target, exact=True)
            return res.count > 0
        except Exception:
            return False
