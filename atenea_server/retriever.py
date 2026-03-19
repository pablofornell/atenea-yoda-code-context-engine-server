from typing import List, Optional
from .embedder import Embedder
from .vector_store import VectorStore

class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embeder = embedder
        self.vector_store = vector_store

    async def retrieve(self, query: str, limit: int = 20, collection_name: Optional[str] = None) -> List[dict]:
        # 1. Embed query
        query_embeddings = await self.embeder.embed([query])
        if not query_embeddings:
            return []
            
        # 2. Search in vector store
        # We request more than 'limit' to allow for deduplication
        raw_results = self.vector_store.search(query_embeddings[0], limit=limit * 2, collection_name=collection_name)
        
        # 3. Deduplicate by file path (one chunk per file, highest score kept)
        seen_files = set()
        deduplicated = []
        for res in raw_results:
            file_path = res["file_path"]
            if file_path not in seen_files:
                deduplicated.append(res)
                seen_files.add(file_path)
            if len(deduplicated) >= limit:
                break
                
        return deduplicated
