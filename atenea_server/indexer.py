import os
import asyncio
from typing import List, Set, Tuple

from .chunker import Chunker, Chunk
from .constants import IGNORED_DIRS, BINARY_EXTS, IGNORED_FILES
from .embedder import Embedder, EmbeddingError
from .vector_store import VectorStore
from .logging_config import get_logger

logger = get_logger(__name__)

class Indexer:
    def __init__(self, chunker: Chunker, embedder: Embedder, vector_store: VectorStore):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.ignored_dirs = IGNORED_DIRS
        self.binary_exts = BINARY_EXTS
        self.ignored_files = IGNORED_FILES

    async def index_directory(self, root_path: str):
        logger.info(f"Indexing directory: {root_path}")
        
        all_chunks = []
        for root, dirs, files in os.walk(root_path):
            # Prune ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]
            
            for file in files:
                if file in self.ignored_files:
                    continue
                    
                ext = os.path.splitext(file)[1].lower()
                if ext in self.binary_exts:
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_path)
                
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if not content.strip():
                            continue
                            
                        file_chunks = self.chunker.chunk_file(rel_path, content)
                        all_chunks.extend(file_chunks)
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")

        if not all_chunks:
            logger.info("No chunks to index.")
            return

        logger.info(f"Found {len(all_chunks)} chunks. Generating embeddings...")

        # Batch processing with parallelism
        batch_size = 20
        semaphore = asyncio.Semaphore(2)  # Limit concurrency to avoid overloading Ollama
        failed_chunks: List[Tuple[int, str]] = []  # (batch_idx, error_message)

        async def process_batch(batch_idx: int, batch_chunks: List[Chunk]) -> int:
            """Process a batch and return the number of successfully indexed chunks."""
            async with semaphore:
                contents = [c.content for c in batch_chunks]
                try:
                    embeddings, failed_indices = await self.embedder.embed_with_fallback(contents)

                    if failed_indices:
                        # Some chunks failed - only index the successful ones
                        successful_embeddings = []
                        successful_batch_chunks = []
                        for i, (chunk, emb) in enumerate(zip(batch_chunks, embeddings)):
                            if i not in failed_indices:
                                successful_embeddings.append(emb)
                                successful_batch_chunks.append(chunk)

                        if successful_embeddings:
                            self.vector_store.upsert_chunks(successful_batch_chunks, successful_embeddings)

                        failed_chunks.append((batch_idx, f"{len(failed_indices)} chunks failed"))
                        logger.warning(f"Batch {batch_idx + 1}: {len(failed_indices)} chunks skipped due to embedding errors")
                        count = len(successful_batch_chunks)
                    else:
                        # All chunks succeeded
                        self.vector_store.upsert_chunks(batch_chunks, embeddings)
                        count = len(batch_chunks)

                    logger.info(f"Indexed batch {batch_idx + 1}/{len(batches)}...")
                    return count

                except EmbeddingError as e:
                    failed_chunks.append((batch_idx, str(e)))
                    logger.error(f"Batch {batch_idx + 1} failed completely: {e}")
                    return 0

        batches = [all_chunks[i:i+batch_size] for i in range(0, len(all_chunks), batch_size)]
        tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]

        results = await asyncio.gather(*tasks)
        successful_chunks = sum(results)

        # Report results
        if failed_chunks:
            logger.warning(f"Indexing completed with {len(failed_chunks)} failed batches. "
                          f"Successfully indexed {successful_chunks}/{len(all_chunks)} chunks.")
        else:
            logger.info(f"Indexing complete. Successfully indexed {successful_chunks} chunks.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m atenea.indexer <directory_path>")
        sys.exit(1)
        
    dir_to_index = sys.argv[1]
    chunker = Chunker()
    embedder = Embedder()
    vector_store = VectorStore()
    indexer = Indexer(chunker, embedder, vector_store)
    
    asyncio.run(indexer.index_directory(dir_to_index))
