import httpx
import os
from typing import List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Default configuration - can be overridden via environment variables
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class EmbeddingTaskType(Enum):
    """
    Task types for asymmetric retrieval.

    For models like nomic-embed-text, using different prefixes for documents
    vs queries improves retrieval quality because the model can optimize
    embeddings for each use case.
    """
    SEARCH_DOCUMENT = "search_document"  # For indexing documents
    SEARCH_QUERY = "search_query"        # For search queries
    CLUSTERING = "clustering"            # For clustering tasks
    CLASSIFICATION = "classification"    # For classification tasks


# Prefix templates for different models
# nomic-embed-text uses these specific prefixes
MODEL_PREFIXES = {
    "nomic-embed-text": {
        EmbeddingTaskType.SEARCH_DOCUMENT: "search_document: ",
        EmbeddingTaskType.SEARCH_QUERY: "search_query: ",
        EmbeddingTaskType.CLUSTERING: "clustering: ",
        EmbeddingTaskType.CLASSIFICATION: "classification: ",
    },
    # Add other models as needed
    "default": {
        EmbeddingTaskType.SEARCH_DOCUMENT: "",
        EmbeddingTaskType.SEARCH_QUERY: "",
        EmbeddingTaskType.CLUSTERING: "",
        EmbeddingTaskType.CLASSIFICATION: "",
    }
}


class Embedder:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.model = model or os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        ollama_url = base_url or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.base_url = f"{ollama_url}/api/embed"
        self._dimension: Optional[int] = None
        self._client = httpx.AsyncClient(timeout=120.0)

        # Get prefix configuration for this model
        self._prefixes = MODEL_PREFIXES.get(self.model, MODEL_PREFIXES["default"])

    def _apply_prefix(self, texts: List[str], task_type: EmbeddingTaskType) -> List[str]:
        """Apply task-specific prefix to texts for asymmetric retrieval."""
        prefix = self._prefixes.get(task_type, "")
        if not prefix:
            return texts
        return [f"{prefix}{text}" for text in texts]

    async def embed(
        self,
        texts: List[str],
        raise_on_error: bool = True,
        task_type: EmbeddingTaskType = EmbeddingTaskType.SEARCH_DOCUMENT
    ) -> List[List[float]]:
        """
        Generate embeddings for the given texts. Handles internal batching to avoid
        exceeding model context length.
        """
        if not texts:
            return []

        # nomic-embed-text has context window of 8192 tokens.
        # 1 token is roughly 4 chars. 25000 chars is ~6k tokens, providing a safe buffer.
        max_batch_chars = 25000
        
        # Check if we need to split this request into smaller sub-batches
        total_chars = sum(len(t) for t in texts)
        if total_chars > max_batch_chars and len(texts) > 1:
            logger.info(f"Batch too large ({total_chars} chars), splitting into sub-batches...")
            all_embeddings = []
            current_sub_batch = []
            current_chars = 0
            
            for text in texts:
                text_len = len(text)
                if current_chars + text_len > max_batch_chars and current_sub_batch:
                    # Send current sub-batch
                    embeddings = await self.embed(current_sub_batch, raise_on_error, task_type)
                    all_embeddings.extend(embeddings)
                    current_sub_batch = [text]
                    current_chars = text_len
                else:
                    current_sub_batch.append(text)
                    current_chars += text_len
            
            # Send last sub-batch
            if current_sub_batch:
                embeddings = await self.embed(current_sub_batch, raise_on_error, task_type)
                all_embeddings.extend(embeddings)
                
            return all_embeddings

        # Apply task-specific prefix for asymmetric retrieval
        prefixed_texts = self._apply_prefix(texts, task_type)

        try:
            response = await self._client.post(
                self.base_url,
                json={
                    "model": self.model,
                    "input": prefixed_texts
                }
            )
            if response.status_code != 200:
                error_msg = f"Ollama error {response.status_code}: {response.text}"
                logger.error(error_msg)
                
                # If we get a 400 and haven't split yet (only 1 text), we can't do much but fail
                if response.status_code == 400 and "context length" in response.text.lower() and len(texts) == 1:
                    logger.warning(f"Single text too large for Ollama context: {len(texts[0])} chars")
                    # Truncate if it's a single massive text that can't be chunked further?
                    # For now, just raise or return empty.
                
                if raise_on_error:
                    raise EmbeddingError(error_msg)
                return []

            data = response.json()
            embeddings = data.get("embeddings", [])

            # Cache dimension for validation
            if embeddings and self._dimension is None:
                self._dimension = len(embeddings[0])

            return embeddings

        except httpx.TimeoutException as e:
            error_msg = f"Ollama embedding timed out: {e}"
            logger.error(error_msg)
            if raise_on_error:
                raise EmbeddingError(error_msg) from e
            return []

        except httpx.ConnectError as e:
            error_msg = f"Could not connect to Ollama at {self.base_url}: {e}"
            logger.error(error_msg)
            if raise_on_error:
                raise EmbeddingError(error_msg) from e
            return []

        except Exception as e:
            error_msg = f"Ollama embedding failed ({type(e).__name__}): {e}"
            logger.error(error_msg)
            if raise_on_error:
                raise EmbeddingError(error_msg) from e
            return []

    async def embed_with_fallback(
        self,
        texts: List[str],
        max_retries: int = 2,
        task_type: EmbeddingTaskType = EmbeddingTaskType.SEARCH_DOCUMENT
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Embed texts with retry logic. Returns successful embeddings and indices of failed texts.

        Args:
            texts: List of text strings to embed
            max_retries: Number of retry attempts for failed batches
            task_type: The embedding task type for asymmetric retrieval.

        Returns:
            Tuple of (embeddings, failed_indices) where failed_indices contains
            the indices of texts that could not be embedded after retries
        """
        if not texts:
            return [], []

        embeddings = []
        failed_indices = []

        for attempt in range(max_retries + 1):
            try:
                result = await self.embed(texts, raise_on_error=True, task_type=task_type)
                return result, []
            except EmbeddingError as e:
                if attempt < max_retries:
                    logger.warning(f"Embedding attempt {attempt + 1} failed, retrying... Error: {e}")
                    continue
                else:
                    logger.error(f"Embedding failed after {max_retries + 1} attempts")
                    failed_indices = list(range(len(texts)))
                    return [], failed_indices

        return embeddings, failed_indices

    async def embed_query(self, query: str, raise_on_error: bool = False) -> Optional[List[float]]:
        """
        Convenience method to embed a single query for search.

        Args:
            query: The search query text
            raise_on_error: If True, raises EmbeddingError on failure

        Returns:
            Embedding vector or None on failure
        """
        result = await self.embed([query], raise_on_error=raise_on_error,
                                  task_type=EmbeddingTaskType.SEARCH_QUERY)
        return result[0] if result else None

    async def embed_documents(
        self,
        documents: List[str],
        raise_on_error: bool = True
    ) -> List[List[float]]:
        """
        Convenience method to embed documents for indexing.

        Args:
            documents: List of document texts to embed
            raise_on_error: If True, raises EmbeddingError on failure

        Returns:
            List of embedding vectors
        """
        return await self.embed(documents, raise_on_error=raise_on_error,
                               task_type=EmbeddingTaskType.SEARCH_DOCUMENT)
