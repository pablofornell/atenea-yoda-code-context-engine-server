import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from atenea_server.chunker import Chunker
from atenea_server.embedder import Embedder

class TestFixOverflow:
    @pytest.mark.asyncio
    async def test_chunker_respects_char_limit(self):
        chunker = Chunker()
        # Set a very low char limit for testing
        chunker.max_chunk_chars = 100
        
        # 10 lines of 20 characters each = 200 chars
        content = "\n".join(["x" * 19 for _ in range(10)])
        
        # This should trigger splitting by char count
        chunks = chunker.chunk_file("test.py", content)
        
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.content) <= 150 # allow some buffer for semantic boundary logic

    @pytest.mark.asyncio
    async def test_embedder_sub_batching(self):
        embedder = Embedder()
        # total chars = 30,000 (exceeds default 8,000)
        
        def side_effect(url, json=None, **kwargs):
            inputs = json.get("input", [])
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"embeddings": [[0.1] * 768 for _ in inputs]}
            return resp
        
        embedder._client = AsyncMock()
        embedder._client.post = AsyncMock(side_effect=side_effect)
        
        texts = ["a" * 5000, "b" * 5000, "c" * 5000]
        
        embeddings = await embedder.embed(texts)
        
        assert len(embeddings) == 3
        # Should have been called twice (one batch of 1, one batch of 1, one batch of 1)
        # Because 5000+5000 > 8000
        assert embedder._client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_embedder_single_text_truncation(self):
        embedder = Embedder()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
        
        embedder._client = AsyncMock()
        embedder._client.post = AsyncMock(return_value=mock_response)
        
        # Single text = 10,000 chars (exceeds 8,000)
        texts = ["x" * 10000]
        
        embeddings = await embedder.embed(texts)
        
        assert len(embeddings) == 1
        # Verify it was truncated before being sent
        call_args = embedder._client.post.call_args
        sent_input = call_args[1]["json"]["input"][0]
        # "search_document: " is 18 chars
        assert len(sent_input) <= 8000 + 18
        assert "search_document: " + ("x" * 8000) == sent_input
