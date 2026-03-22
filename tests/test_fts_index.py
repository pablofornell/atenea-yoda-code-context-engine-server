"""Tests for the FTSIndex module."""

import pytest
import os
import tempfile
from atenea_server.fts_index import FTSIndex


class TestFTSIndex:
    """Test suite for the FTSIndex class."""

    def setup_method(self):
        """Set up test fixtures with a temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_fts.db")
        self.index = FTSIndex(db_path=self.db_path)

    def teardown_method(self):
        """Clean up temporary database."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_add_document(self):
        """Test adding a document to the index."""
        self.index.add_document(
            doc_id="doc1",
            file_path="test.py",
            content="def hello_world(): print('Hello')",
            start_line=1,
            end_line=1,
            language="python",
            collection="test"
        )
        
        stats = self.index.get_stats(collection="test")
        assert stats["total_chunks"] == 1

    def test_search_returns_matching_documents(self):
        """Test that search returns documents matching the query."""
        self.index.add_document(
            doc_id="doc1",
            file_path="auth.py",
            content="def authenticate_user(username, password): pass",
            start_line=1,
            end_line=1,
            language="python",
            collection="test",
            symbol_name="authenticate_user",
            symbol_type="function"
        )
        self.index.add_document(
            doc_id="doc2",
            file_path="utils.py",
            content="def format_date(date): return str(date)",
            start_line=1,
            end_line=1,
            language="python",
            collection="test",
            symbol_name="format_date",
            symbol_type="function"
        )
        
        results = self.index.search("authenticate user", limit=10, collection="test")
        
        assert len(results) >= 1
        # The auth document should be found
        doc_ids = [r[0] for r in results]
        assert "doc1" in doc_ids

    def test_search_empty_query(self):
        """Test that empty query returns no results."""
        self.index.add_document(
            doc_id="doc1",
            file_path="test.py",
            content="some content",
            start_line=1,
            end_line=1,
            language="python",
            collection="test"
        )
        
        results = self.index.search("", limit=10)
        assert len(results) == 0

    def test_language_filter(self):
        """Test filtering by language."""
        self.index.add_document(
            doc_id="doc1", file_path="test.py", content="python code",
            start_line=1, end_line=1, language="python", collection="test"
        )
        self.index.add_document(
            doc_id="doc2", file_path="test.js", content="javascript code",
            start_line=1, end_line=1, language="javascript", collection="test"
        )
        
        results = self.index.search("code", language_filter="python", collection="test")
        assert len(results) == 1
        assert results[0][2]["language"] == "python"

    def test_symbol_type_filter(self):
        """Test filtering by symbol type."""
        self.index.add_document(
            doc_id="doc1", file_path="test.py", content="class MyClass: pass",
            start_line=1, end_line=1, language="python", collection="test",
            symbol_type="class", symbol_name="MyClass"
        )
        self.index.add_document(
            doc_id="doc2", file_path="test.py", content="def my_function(): pass",
            start_line=2, end_line=2, language="python", collection="test",
            symbol_type="function", symbol_name="my_function"
        )
        
        results = self.index.search("my", symbol_type_filter="class", collection="test")
        assert len(results) == 1
        assert results[0][2]["symbol_type"] == "class"

    def test_clear_collection(self):
        """Test clearing a specific collection."""
        self.index.add_document(
            doc_id="doc1", file_path="test.py", content="content",
            start_line=1, end_line=1, language="python", collection="test1"
        )
        self.index.add_document(
            doc_id="doc2", file_path="test.py", content="content",
            start_line=1, end_line=1, language="python", collection="test2"
        )
        
        self.index.clear(collection="test1")
        
        assert self.index.get_stats(collection="test1")["total_chunks"] == 0
        assert self.index.get_stats(collection="test2")["total_chunks"] == 1

    def test_camel_case_tokenization(self):
        """Test that camelCase tokens are properly split."""
        self.index.add_document(
            doc_id="doc1", file_path="test.py",
            content="getUserById findAllUsers",
            start_line=1, end_line=1, language="python", collection="test"
        )
        
        results = self.index.search("user", collection="test")
        assert len(results) >= 1

    def test_delete_by_file_paths(self):
        """Test deleting documents by file path."""
        self.index.add_document(
            doc_id="doc1", file_path="file1.py", content="content1",
            start_line=1, end_line=1, language="python", collection="test"
        )
        self.index.add_document(
            doc_id="doc2", file_path="file2.py", content="content2",
            start_line=1, end_line=1, language="python", collection="test"
        )
        
        deleted = self.index.delete_by_file_paths(["file1.py"], collection="test")
        
        assert deleted == 1
        assert self.index.get_stats(collection="test")["total_chunks"] == 1

