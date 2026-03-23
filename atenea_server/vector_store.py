import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from typing import List, Optional, Dict
import logging
import os
from .chunker import Chunk

# Namespace UUID for generating deterministic point IDs
_ATENEA_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

logger = logging.getLogger(__name__)

# Default configuration - can be overridden via environment variables
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_EMBEDDING_DIMENSION = 768  # nomic-embed-text dimension


class VectorStore:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        embedding_dimension: Optional[int] = None
    ):
        self.host = host or os.environ.get("QDRANT_HOST", DEFAULT_QDRANT_HOST)
        self.port = port or int(os.environ.get("QDRANT_PORT", DEFAULT_QDRANT_PORT))
        self.embedding_dimension = embedding_dimension or int(
            os.environ.get("EMBEDDING_DIMENSION", DEFAULT_EMBEDDING_DIMENSION)
        )
        self.client = QdrantClient(host=self.host, port=self.port)
        self.default_collection = "atenea_code"

    def _ensure_collection(self, collection_name: str) -> None:
        try:
            self.client.get_collection(collection_name)
        except UnexpectedResponse:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dimension,
                    distance=models.Distance.COSINE
                )
            )

    def list_collections(self) -> List[str]:
        response = self.client.get_collections()
        return [c.name for c in response.collections]

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]], collection_name: Optional[str] = None, content_hash: Optional[str] = None) -> None:
        if not chunks:
            return

        collection_name = collection_name or self.default_collection

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Use stable deterministic UUID5 for point ID
            id_input = f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}"
            point_id = str(uuid.uuid5(_ATENEA_NS, id_input))

            # Retrieve content_hash from chunk if available
            c_hash = chunk.content_hash or content_hash

            # Build payload with enhanced metadata
            payload = {
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content,
                "language": chunk.language,
                "content_hash": c_hash,
                # Enhanced metadata for better retrieval
                "symbol_name": getattr(chunk, 'symbol_name', None),
                "symbol_type": getattr(chunk, 'symbol_type', None),
                "parent_context": getattr(chunk, 'parent_context', None),
                "parent_symbols": getattr(chunk, 'parent_symbols', []),
                "docstring": getattr(chunk, 'docstring', None),
            }

            # Remove None values to save space
            payload = {k: v for k, v in payload.items() if v is not None}

            points.append(models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
        except UnexpectedResponse:
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

    def clear_collection(self, collection_name: Optional[str] = None) -> None:
        """Delete and recreate the collection."""
        target = collection_name or self.default_collection
        try:
            self.client.delete_collection(target)
        except UnexpectedResponse:
            pass  # Collection didn't exist, that's fine
        self._ensure_collection(target)

    def has_data(self, collection_name: Optional[str] = None) -> bool:
        """Check if the collection has any points."""
        target = collection_name or self.default_collection
        try:
            res = self.client.count(collection_name=target, exact=True)
            return res.count > 0
        except UnexpectedResponse:
            return False

    def get_file_hashes(self, collection_name: Optional[str] = None) -> Dict[str, str]:
        """Fetch all unique file_path -> content_hash mappings from the collection."""
        target = collection_name or self.default_collection
        hashes = {}
        
        try:
            # Scroll through points to get metadata
            # For a massive codebase, we might want a more efficient way or indexing hashes specifically,
            # but for MVP scrolling is fine.
            offset = None
            while True:
                response = self.client.scroll(
                    collection_name=target,
                    limit=100,
                    with_payload=["file_path", "content_hash"],
                    offset=offset
                )
                for point in response[0]:
                    payload = point.payload
                    path = payload.get("file_path")
                    h = payload.get("content_hash")
                    if path and h:
                        hashes[path] = h
                
                offset = response[1]
                if offset is None:
                    break
        except Exception as e:
            logger.warning(f"Error fetching existing hashes: {e}")
            
        return hashes

    def delete_by_file_paths(self, file_paths: List[str], collection_name: Optional[str] = None) -> None:
        """Delete all points associated with the given file paths."""
        if not file_paths:
            return
            
        target = collection_name or self.default_collection
        try:
            self.client.delete(
                collection_name=target,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchAny(any=file_paths)
                        )
                    ]
                )
            )
        except Exception as e:
            logger.error(f"Error deleting old chunks: {e}")
