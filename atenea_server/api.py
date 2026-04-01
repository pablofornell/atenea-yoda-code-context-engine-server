import asyncio
import json as json_mod
import os
from pathlib import Path
from aiohttp import web
from typing import List

from .chunker import Chunker, Chunk
from .embedder import Embedder, EmbeddingError
from .vector_store import VectorStore
from .indexer import Indexer
from .retriever import Retriever
from .formatter import Formatter
from .logging_config import setup_logging, get_logger
from .crypto import get_secret, encrypt, decrypt, ENCRYPTED_HEADER

# Setup logging once at module import
setup_logging()
logger = get_logger(__name__)


@web.middleware
async def encryption_middleware(request, handler):
    """
    aiohttp middleware that handles AES-256-GCM encryption/decryption.

    - If ATENEA_SECRET is not set, passes requests through unchanged.
    - Incoming requests with X-Atenea-Encrypted header: decrypt the body.
    - Outgoing responses: encrypt the body and set X-Atenea-Encrypted header.
    """
    key = get_secret()
    if key is None:
        return await handler(request)

    # --- Decrypt incoming request body ---
    if request.headers.get(ENCRYPTED_HEADER) == "1" and request.can_read_body:
        raw_body = await request.read()
        try:
            decrypted = decrypt(raw_body, key)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return web.json_response({"error": "Decryption failed"}, status=400)

        # Replace the cached read bytes so handlers see plain JSON
        # (request.read() / request.json() check _read_bytes first)
        request._read_bytes = decrypted

    # --- Call the actual handler ---
    response = await handler(request)

    # --- Encrypt outgoing response body ---
    if isinstance(response, web.Response) and response.body:
        plain_body = response.body
        if isinstance(plain_body, str):
            plain_body = plain_body.encode("utf-8")
        encrypted_body = encrypt(plain_body, key)
        response = web.Response(
            body=encrypted_body,
            status=response.status,
            headers={ENCRYPTED_HEADER: "1", "Content-Type": "application/octet-stream"},
        )

    return response

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

            async def process_chunks(chunks_to_index: List[Chunk]) -> int:
                """Process a batch of chunks, returning count of successfully indexed."""
                async with semaphore:
                    contents = [c.content for c in chunks_to_index]
                    try:
                        embeddings, failed_indices = await self.embedder.embed_with_fallback(contents)

                        if failed_indices:
                            # Only index successful chunks
                            successful_chunks = [c for i, c in enumerate(chunks_to_index) if i not in failed_indices]
                            successful_embeddings = embeddings
                            if successful_chunks:
                                self.vector_store.upsert_chunks(successful_chunks, successful_embeddings, collection_name=collection_name)
                            return len(successful_chunks)
                        else:
                            self.vector_store.upsert_chunks(chunks_to_index, embeddings, collection_name=collection_name)
                            return len(chunks_to_index)
                    except EmbeddingError as e:
                        logger.error(f"Failed to embed batch: {e}")
                        return 0

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
                        chunk.content_hash = content_hash
                    current_batch_chunks.extend(file_chunks)
                
                if current_batch_chunks:
                    # Further batch chunks for embedding if they are too many
                    chunk_batch_size = 20
                    tasks = []
                    for j in range(0, len(current_batch_chunks), chunk_batch_size):
                        chunk_batch = current_batch_chunks[j : j + chunk_batch_size]
                        tasks.append(process_chunks(chunk_batch))
                    
                    results = await asyncio.gather(*tasks)
                    batch_indexed = sum(results)
                    total_chunks += batch_indexed
                    logger.info(f"Indexed batch of {len(file_batch)} files ({batch_indexed} chunks) to {collection_name or 'default'}...")

            # Invalidate FTS index so it rebuilds on next query
            self.retriever.invalidate_fts_index(collection_name=collection_name)

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
            # Also clear FTS index
            self.retriever.invalidate_fts_index(collection_name=collection_name)
            return web.json_response({"status": "ok", "message": f"Index {collection_name or 'default'} cleared"})
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return web.json_response({"error": str(e)}, status=500)

def _load_dotenv() -> None:
    """
    Load variables from a .env file into os.environ.
    Looks for .env in the current working directory, then in the directory
    one level above this package (i.e. next to the Makefile).
    Already-set variables are never overwritten, so shell overrides still work.
    """
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent / ".env",  # atenea-server/.env
    ]
    for env_path in candidates:
        if env_path.exists():
            logger.info(f"Loading config from {env_path}")
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            break  # stop at the first file found


def main():
    _load_dotenv()
    api = AteneaAPI()

    middlewares = []
    if get_secret() is not None:
        middlewares.append(encryption_middleware)
        logger.info("Encryption enabled (ATENEA_SECRET is set)")
    else:
        logger.warning("Encryption disabled (ATENEA_SECRET not set) — traffic is in plaintext")

    app = web.Application(client_max_size=100 * 1024 * 1024, middlewares=middlewares)  # 100MB
    app.add_routes([
        web.get('/api/status', api.handle_status),
        web.get('/api/list', api.handle_list),
        web.get('/api/index/hashes', api.handle_hashes),
        web.post('/api/index', api.handle_index),
        web.post('/api/query', api.handle_query),
        web.delete('/api/index', api.handle_clean),
    ])
    
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8080))
    web.run_app(app, host=host, port=port)

if __name__ == "__main__":
    main()
