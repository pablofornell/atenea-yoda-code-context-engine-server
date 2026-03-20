import asyncio
import logging
import os
from aiohttp import web
from typing import List

from .chunker import Chunker, Chunk
from .embedder import Embedder
from .vector_store import VectorStore
from .indexer import Indexer
from .retriever import Retriever
from .formatter import Formatter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("atenea.api")

class AteneaAPI:
    def __init__(self):
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.indexer = Indexer(self.chunker, self.embedder, self.vector_store)
        self.retriever = Retriever(self.embedder, self.vector_store)
        self.formatter = Formatter()

    async def handle_status(self, request):
        collections = self.vector_store.list_collections()
        return web.json_response({
            "status": "ok",
            "collections": collections,
            "engine": "Atenea Context Engine"
        })

    async def handle_list(self, request):
        try:
            collections = self.vector_store.list_collections()
            return web.json_response({
                "status": "ok",
                "collections": collections
            })
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_hashes(self, request):
        try:
            collection_name = request.query.get("collection")
            hashes = self.vector_store.get_file_hashes(collection_name=collection_name)
            return web.json_response({
                "status": "ok",
                "hashes": hashes
            })
        except Exception as e:
            logger.error(f"Error fetching hashes: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_index(self, request):
        try:
            data = await request.json()
            files = data.get("files", [])
            collection_name = data.get("collection")
            deleted_files = data.get("deleted_files", [])
            
            # Handle deletions first
            if deleted_files:
                logger.info(f"Deleting {len(deleted_files)} files from {collection_name or 'default'}")
                self.vector_store.delete_by_file_paths(deleted_files, collection_name=collection_name)

            if not files and not deleted_files:
                return web.json_response({"error": "No files or deleted_files provided"}, status=400)
            
            if not files:
                return web.json_response({
                    "status": "ok", 
                    "message": f"Successfully deleted {len(deleted_files)} files",
                    "chunks": 0
                })

            async def process_chunks(chunks_to_index: List[Chunk]):
                async with semaphore:
                    contents = [c.content for c in chunks_to_index]
                    embeddings = await self.embedder.embed(contents)
                    self.vector_store.upsert_chunks(chunks_to_index, embeddings, collection_name=collection_name)

            semaphore = asyncio.Semaphore(2)
            total_chunks = 0
            
            # Process files in small groups to avoid memory spikes
            file_batch_size = 5
            for i in range(0, len(files), file_batch_size):
                file_batch = files[i : i + file_batch_size]
                current_batch_chunks = []
                
                for f in file_batch:
                    path = f.get("path")
                    content = f.get("content", "")
                    content_hash = f.get("content_hash")
                    if not path or not content.strip():
                        continue
                    
                    file_chunks = self.chunker.chunk_file(path, content)
                    for chunk in file_chunks:
                        setattr(chunk, 'content_hash', content_hash)
                    current_batch_chunks.extend(file_chunks)
                
                if current_batch_chunks:
                    # Further batch chunks for embedding if they are too many
                    chunk_batch_size = 50
                    tasks = []
                    for j in range(0, len(current_batch_chunks), chunk_batch_size):
                        chunk_batch = current_batch_chunks[j : j + chunk_batch_size]
                        tasks.append(process_chunks(chunk_batch))
                    
                    await asyncio.gather(*tasks)
                    total_chunks += len(current_batch_chunks)
                    logger.info(f"Indexed batch of {len(file_batch)} files ({len(current_batch_chunks)} chunks) to {collection_name or 'default'}...")

            return web.json_response({
                "status": "ok",
                "message": f"Successfully indexed {len(files)} files",
                "chunks": total_chunks
            })
        except Exception as e:
            logger.error(f"Error indexing: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_query(self, request):
        try:
            data = await request.json()
            query = data.get("query")
            limit = data.get("limit", 20)
            collection_name = data.get("collection")

            if not query:
                return web.json_response({"error": "No query provided"}, status=400)

            chunks = await self.retriever.retrieve(query, limit=limit, collection_name=collection_name)
            formatted = self.formatter.format(chunks)

            return web.json_response({
                "status": "ok",
                "results": formatted,
                "count": len(chunks)
            })
        except Exception as e:
            error_msg = str(e)
            if "doesn't exist" in error_msg or "Not found" in error_msg:
                logger.warning(f"Query failed: Collection {collection_name} does not exist.")
                return web.json_response({
                    "error": f"Collection '{collection_name}' not found. Please index it first using 'atenea index'.",
                    "code": "collection_not_found"
                }, status=404)
            
            logger.error(f"Error querying: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_clean(self, request):
        try:
            data = await request.json() if request.has_body else {}
            collection_name = data.get("collection")
            self.vector_store.clear_collection(collection_name=collection_name)
            return web.json_response({"status": "ok", "message": f"Index {collection_name or 'default'} cleared"})
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return web.json_response({"error": str(e)}, status=500)

def main():
    api = AteneaAPI()
    app = web.Application(client_max_size=100 * 1024 * 1024)  # 100MB
    app.add_routes([
        web.get('/api/status', api.handle_status),
        web.get('/api/list', api.handle_list),
        web.get('/api/index/hashes', api.handle_hashes),
        web.post('/api/index', api.handle_index),
        web.post('/api/query', api.handle_query),
        web.delete('/api/index', api.handle_clean),
    ])
    
    port = int(os.environ.get("PORT", 8080))
    web.run_app(app, port=port)

if __name__ == "__main__":
    main()
