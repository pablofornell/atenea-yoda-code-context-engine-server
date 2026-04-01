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
        # Set a very low batch char limit for testing
        # We need to monkeypatch the local variable in the method or just test with 25000 chars
        
        def side_effect(url, json=None, **kwargs):
            inputs = json.get("input", [])
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"embeddings": [[0.1] * 768 for _ in inputs]}
            return resp
        
        embedder._client = AsyncMock()
        embedder._client.post = AsyncMock(side_effect=side_effect)
        
        # total chars = 30,000 (exceeds default 25,000)
        texts = ["a" * 10000, "b" * 10000, "c" * 10000]
        
        embeddings = await embedder.embed(texts)
        
        assert len(embeddings) == 3
        # Should have been called twice (one batch of 2, one batch of 1 or similar)
        assert embedder._client.post.call_count >= 2
