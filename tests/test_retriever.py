"""Tests for the Retriever module."""

import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from atenea_server.retriever import Retriever
from atenea_server.embedder import Embedder


class TestRetriever:
    """Test suite for the Retriever class."""

    def setup_method(self):
        self.embedder = MagicMock(spec=Embedder)
        self.vector_store = MagicMock()
        self.vector_store.default_collection = "test_collection"
        # Use temp file for FTS DB in tests
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            fts_db_path=self.tmp_db.name,
            enable_query_expansion=False,
        )

    def teardown_method(self):
        if os.path.exists(self.tmp_db.name):
            os.unlink(self.tmp_db.name)

    def test_init_weights(self):
        assert self.retriever.vector_weight == 0.7
        assert self.retriever.bm25_weight == 0.3

    def test_get_doc_key(self):
        payload = {"file_path": "test.py", "start_line": 1, "end_line": 10}
        key = self.retriever._get_doc_key(payload)
        assert key == "test.py:1:10"

    def test_deduplicate_results_removes_duplicates(self):
        results = [
            {"file_path": "a.py", "start_line": 1, "end_line": 5, "content": "x"},
            {"file_path": "a.py", "start_line": 1, "end_line": 5, "content": "x"},
            {"file_path": "b.py", "start_line": 1, "end_line": 5, "content": "y"},
        ]
        deduped = self.retriever._deduplicate_results(results, limit=10)
        assert len(deduped) == 2

    def test_deduplicate_results_enforces_per_file_limit(self):
        results = [
            {"file_path": "a.py", "start_line": i, "end_line": i + 5, "content": f"c{i}"}
            for i in range(10)
        ]
        deduped = self.retriever._deduplicate_results(results, limit=20, max_per_file=3)
        assert len(deduped) == 3

    def test_deduplicate_results_enforces_total_limit(self):
        results = [
            {"file_path": f"file{i}.py", "start_line": 1, "end_line": 5, "content": f"c{i}"}
            for i in range(20)
        ]
        deduped = self.retriever._deduplicate_results(results, limit=5)
        assert len(deduped) == 5

    def test_reciprocal_rank_fusion(self):
        vec_results = [
            ("a.py:1:5", 1.0, {"file_path": "a.py", "start_line": 1, "end_line": 5}),
            ("b.py:1:5", 0.5, {"file_path": "b.py", "start_line": 1, "end_line": 5}),
        ]
        fts_results = [
            ("b.py:1:5", 1.0, {"file_path": "b.py", "start_line": 1, "end_line": 5}),
            ("c.py:1:5", 0.5, {"file_path": "c.py", "start_line": 1, "end_line": 5}),
        ]
        fused = self.retriever._reciprocal_rank_fusion(vec_results, fts_results)
        # b.py appears in both, should be ranked high
        file_paths = [r["file_path"] for r in fused]
        assert "b.py" in file_paths
        assert len(fused) == 3  # a, b, c

    def test_is_potentially_truncated_function(self):
        content = "def foo():\n    x = 1\n    {"
        assert self.retriever._is_potentially_truncated(content, "function") is True

    def test_is_potentially_truncated_not_symbol(self):
        content = "some random text {"
        assert self.retriever._is_potentially_truncated(content, "comment") is False

    def test_is_potentially_truncated_balanced(self):
        content = "def foo():\n    return 1"
        assert self.retriever._is_potentially_truncated(content, "function") is False

    def test_invalidate_fts_index(self):
        self.retriever._fts_initialized["test_collection"] = True
        self.retriever.invalidate_fts_index("test_collection")
        assert self.retriever._fts_initialized["test_collection"] is False

    @pytest.mark.asyncio
    async def test_retrieve_vector_only(self):
        self.embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        self.vector_store.search.return_value = [
            {"file_path": "a.py", "start_line": 1, "end_line": 5, "content": "hello"}
        ]

        results = await self.retriever.retrieve("test query", use_hybrid=False)
        assert len(results) == 1
        assert results[0]["file_path"] == "a.py"

    @pytest.mark.asyncio
    async def test_retrieve_empty_embeddings(self):
        self.embedder.embed = AsyncMock(return_value=[])

        results = await self.retriever.retrieve("test query")
        assert results == []

    def test_apply_filters_language(self):
        results = [
            {"file_path": "a.py", "language": "python"},
            {"file_path": "b.js", "language": "javascript"},
        ]
        filtered = self.retriever._apply_filters(results, "python", None, None)
        assert len(filtered) == 1
        assert filtered[0]["language"] == "python"

    def test_apply_filters_symbol_type(self):
        results = [
            {"file_path": "a.py", "symbol_type": "function"},
            {"file_path": "b.py", "symbol_type": "class"},
        ]
        filtered = self.retriever._apply_filters(results, None, "function", None)
        assert len(filtered) == 1
        assert filtered[0]["symbol_type"] == "function"

