"""Tests for the Embedder module."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from atenea_server.embedder import (
    Embedder, EmbeddingError, EmbeddingTaskType, MODEL_PREFIXES
)


class TestEmbedder:
    """Test suite for the Embedder class."""

    def setup_method(self):
        self.embedder = Embedder(model="nomic-embed-text", base_url="http://localhost:11434")

    def test_init_defaults(self):
        e = Embedder()
        assert e.model == "nomic-embed-text"
        assert "api/embed" in e.base_url

    def test_apply_prefix_search_document(self):
        texts = ["hello world"]
        result = self.embedder._apply_prefix(texts, EmbeddingTaskType.SEARCH_DOCUMENT)
        assert result[0].startswith("search_document: ")

    def test_apply_prefix_search_query(self):
        texts = ["find me"]
        result = self.embedder._apply_prefix(texts, EmbeddingTaskType.SEARCH_QUERY)
        assert result[0].startswith("search_query: ")

    def test_apply_prefix_unknown_model(self):
        e = Embedder(model="unknown-model")
        texts = ["test"]
        result = e._apply_prefix(texts, EmbeddingTaskType.SEARCH_DOCUMENT)
        # Default prefix is empty string
        assert result[0] == "test"

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        result = await self.embedder.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        }

        self.embedder._client = AsyncMock()
        self.embedder._client.post = AsyncMock(return_value=mock_response)

        result = await self.embedder.embed(["text1", "text2"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_caches_dimension(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }

        self.embedder._client = AsyncMock()
        self.embedder._client.post = AsyncMock(return_value=mock_response)

        assert self.embedder._dimension is None
        await self.embedder.embed(["text1"])
        assert self.embedder._dimension == 3

    @pytest.mark.asyncio
    async def test_embed_error_raise(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        self.embedder._client = AsyncMock()
        self.embedder._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(EmbeddingError, match="Ollama error 500"):
            await self.embedder.embed(["text"], raise_on_error=True)

    @pytest.mark.asyncio
    async def test_embed_error_no_raise(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "error"

        self.embedder._client = AsyncMock()
        self.embedder._client.post = AsyncMock(return_value=mock_response)

        result = await self.embedder.embed(["text"], raise_on_error=False)
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_timeout(self):
        self.embedder._client = AsyncMock()
        self.embedder._client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with pytest.raises(EmbeddingError, match="timed out"):
            await self.embedder.embed(["text"], raise_on_error=True)

    @pytest.mark.asyncio
    async def test_embed_connect_error(self):
        self.embedder._client = AsyncMock()
        self.embedder._client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(EmbeddingError, match="Could not connect"):
            await self.embedder.embed(["text"], raise_on_error=True)

    @pytest.mark.asyncio
    async def test_embed_query_convenience(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2]]}

        self.embedder._client = AsyncMock()
        self.embedder._client.post = AsyncMock(return_value=mock_response)

        result = await self.embedder.embed_query("test query")
        assert result == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_embed_query_returns_none_on_failure(self):
        self.embedder._client = AsyncMock()
        self.embedder._client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        result = await self.embedder.embed_query("test", raise_on_error=False)
        assert result is None

