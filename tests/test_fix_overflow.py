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
        # total chars = 30,000 (exceeds default 20,000)
        
        def side_effect(url, json=None, **kwargs):
            inputs = json.get("input", [])
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"embeddings": [[0.1] * 768 for _ in inputs]}
            return resp
        
        embedder._client = AsyncMock()
        embedder._client.post = AsyncMock(side_effect=side_effect)
        
        texts = ["a" * 10000, "b" * 10000, "c" * 10000]
        
        embeddings = await embedder.embed(texts)
        
        assert len(embeddings) == 3
        # 10000+10000 > 20000, so it splits. Should be at least 2 calls.
        assert embedder._client.post.call_count >= 2
        # Verify num_ctx is passed in the request
        call_args = embedder._client.post.call_args
        assert call_args[1]["json"]["options"]["num_ctx"] == 8192

    @pytest.mark.asyncio
    async def test_embedder_single_text_truncation(self):
        embedder = Embedder()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
        
        embedder._client = AsyncMock()
        embedder._client.post = AsyncMock(return_value=mock_response)
        
        # Single text = 25,000 chars (exceeds 20,000)
        texts = ["x" * 25000]
        
        embeddings = await embedder.embed(texts)
        
        assert len(embeddings) == 1
        # Verify it was truncated before being sent
        call_args = embedder._client.post.call_args
        sent_input = call_args[1]["json"]["input"][0]
        # "search_document: " is 18 chars, text truncated to 20000
        assert len(sent_input) <= 20000 + 18
        # Verify num_ctx is passed
        assert call_args[1]["json"]["options"]["num_ctx"] == 8192
