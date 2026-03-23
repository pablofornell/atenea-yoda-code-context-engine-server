"""
SQLite FTS5-based full-text search index for hybrid retrieval.

This module provides a persistent BM25-style search using SQLite's FTS5 extension,
which offers better performance and persistence compared to in-memory solutions.
"""

import sqlite3
import os
import re
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Default database location - persist in user's home directory
_DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".atenea")
DEFAULT_DB_PATH = os.environ.get("ATENEA_FTS_DB", os.path.join(_DEFAULT_DIR, "fts.db"))


class FTSIndex:
    """
    SQLite FTS5-based full-text search index.
    
    Uses FTS5's built-in BM25 ranking for efficient keyword search with persistence.
    Supports code-aware tokenization by pre-processing content before indexing.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the FTS index.
        
        Args:
            db_path: Path to SQLite database. If None, uses DEFAULT_DB_PATH.
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self._ensure_directory()
        self._init_db()
        
        # Code-aware tokenization patterns
        self._camel_case_pattern = re.compile(r'(?<!^)(?=[A-Z])')
        self._snake_case_pattern = re.compile(r'_+')
        self._non_alphanum_pattern = re.compile(r'[^a-zA-Z0-9_]')

    def _ensure_directory(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create FTS5 virtual table for full-text search
            # Using porter tokenizer for stemming
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    doc_id,
                    file_path,
                    content,
                    symbol_name,
                    symbol_type,
                    parent_context,
                    language,
                    start_line UNINDEXED,
                    end_line UNINDEXED,
                    collection UNINDEXED,
                    tokenize='porter unicode61'
                )
            """)
            
            # Create regular table for metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunk_metadata (
                    doc_id TEXT PRIMARY KEY,
                    file_path TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    language TEXT,
                    symbol_name TEXT,
                    symbol_type TEXT,
                    parent_context TEXT,
                    collection TEXT,
                    content TEXT
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_file_path 
                ON chunk_metadata(file_path)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_collection 
                ON chunk_metadata(collection)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_symbol_type 
                ON chunk_metadata(symbol_type)
            """)
            
            conn.commit()

    def _tokenize_for_index(self, text: str) -> str:
        """
        Pre-process text for indexing with code-aware tokenization.
        
        Splits camelCase and snake_case while preserving original tokens.
        """
        tokens = []
        words = self._non_alphanum_pattern.sub(' ', text).split()
        
        for word in words:
            if len(word) < 2:
                continue
            
            word_lower = word.lower()
            tokens.append(word_lower)
            
            # Split camelCase
            camel_parts = self._camel_case_pattern.split(word)
            if len(camel_parts) > 1:
                for part in camel_parts:
                    if len(part) >= 2:
                        tokens.append(part.lower())
            
            # Split snake_case
            snake_parts = self._snake_case_pattern.split(word)
            if len(snake_parts) > 1:
                for part in snake_parts:
                    if len(part) >= 2:
                        tokens.append(part.lower())
        
        return ' '.join(tokens)

    def add_document(
        self,
        doc_id: str,
        file_path: str,
        content: str,
        start_line: int,
        end_line: int,
        language: str,
        collection: str = "default",
        symbol_name: Optional[str] = None,
        symbol_type: Optional[str] = None,
        parent_context: Optional[str] = None
    ) -> None:
        """Add or update a document in the index."""
        # Pre-process content for better search
        processed_content = self._tokenize_for_index(content)
        processed_symbol = self._tokenize_for_index(symbol_name or "")
        processed_parent = self._tokenize_for_index(parent_context or "")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Remove existing document if present
            cursor.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM chunk_metadata WHERE doc_id = ?", (doc_id,))

            # Insert into FTS table
            cursor.execute("""
                INSERT INTO chunks_fts
                (doc_id, file_path, content, symbol_name, symbol_type,
                 parent_context, language, start_line, end_line, collection)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id, file_path, processed_content, processed_symbol,
                symbol_type or "", processed_parent, language,
                start_line, end_line, collection
            ))

            # Insert into metadata table (with original content)
            cursor.execute("""
                INSERT INTO chunk_metadata
                (doc_id, file_path, start_line, end_line, language,
                 symbol_name, symbol_type, parent_context, collection, content)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id, file_path, start_line, end_line, language,
                symbol_name, symbol_type, parent_context, collection, content
            ))

            conn.commit()

    def search(
        self,
        query: str,
        limit: int = 20,
        collection: Optional[str] = None,
        language_filter: Optional[str] = None,
        symbol_type_filter: Optional[str] = None,
        file_path_pattern: Optional[str] = None
    ) -> List[Tuple[str, float, dict]]:
        """
        Search the index with a query.

        Args:
            query: Search query string
            limit: Maximum number of results
            collection: Optional collection filter
            language_filter: Optional language filter
            symbol_type_filter: Optional symbol type filter (e.g., 'function', 'class')
            file_path_pattern: Optional file path pattern (SQL LIKE pattern)

        Returns:
            List of (doc_id, score, payload) tuples sorted by relevance
        """
        # Pre-process query for matching
        processed_query = self._tokenize_for_index(query)

        if not processed_query.strip():
            return []

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build query with filters
            where_clauses = []
            params = []

            if collection:
                where_clauses.append("m.collection = ?")
                params.append(collection)

            if language_filter:
                where_clauses.append("m.language = ?")
                params.append(language_filter)

            if symbol_type_filter:
                where_clauses.append("m.symbol_type = ?")
                params.append(symbol_type_filter)

            if file_path_pattern:
                where_clauses.append("m.file_path LIKE ?")
                params.append(file_path_pattern)

            where_sql = ""
            if where_clauses:
                where_sql = "AND " + " AND ".join(where_clauses)

            # Use FTS5's BM25 ranking
            sql = f"""
                SELECT
                    m.doc_id,
                    bm25(chunks_fts) as score,
                    m.file_path,
                    m.content,
                    m.start_line,
                    m.end_line,
                    m.language,
                    m.symbol_name,
                    m.symbol_type,
                    m.parent_context
                FROM chunks_fts f
                JOIN chunk_metadata m ON f.doc_id = m.doc_id
                WHERE chunks_fts MATCH ?
                {where_sql}
                ORDER BY bm25(chunks_fts)
                LIMIT ?
            """

            cursor.execute(sql, [processed_query] + params + [limit])

            results = []
            for row in cursor.fetchall():
                payload = {
                    "file_path": row["file_path"],
                    "content": row["content"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "language": row["language"],
                    "symbol_name": row["symbol_name"],
                    "symbol_type": row["symbol_type"],
                    "parent_context": row["parent_context"],
                }
                # BM25 scores are negative (lower is better), convert to positive
                score = -row["score"] if row["score"] else 0.0
                results.append((row["doc_id"], score, payload))

            return results

    def delete_by_file_paths(self, file_paths: List[str], collection: Optional[str] = None) -> int:
        """Delete all documents with the given file paths."""
        if not file_paths:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()

            placeholders = ",".join("?" * len(file_paths))

            # Get doc_ids first
            if collection:
                cursor.execute(f"""
                    SELECT doc_id FROM chunk_metadata
                    WHERE file_path IN ({placeholders}) AND collection = ?
                """, file_paths + [collection])
            else:
                cursor.execute(f"""
                    SELECT doc_id FROM chunk_metadata
                    WHERE file_path IN ({placeholders})
                """, file_paths)

            doc_ids = [row["doc_id"] for row in cursor.fetchall()]

            if not doc_ids:
                return 0

            id_placeholders = ",".join("?" * len(doc_ids))
            cursor.execute(f"DELETE FROM chunks_fts WHERE doc_id IN ({id_placeholders})", doc_ids)
            cursor.execute(f"DELETE FROM chunk_metadata WHERE doc_id IN ({id_placeholders})", doc_ids)

            conn.commit()
            return len(doc_ids)

    def clear(self, collection: Optional[str] = None) -> None:
        """Clear all documents from the index, optionally filtered by collection."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if collection:
                cursor.execute("SELECT doc_id FROM chunk_metadata WHERE collection = ?", (collection,))
                doc_ids = [row["doc_id"] for row in cursor.fetchall()]

                if doc_ids:
                    placeholders = ",".join("?" * len(doc_ids))
                    cursor.execute(f"DELETE FROM chunks_fts WHERE doc_id IN ({placeholders})", doc_ids)
                    cursor.execute(f"DELETE FROM chunk_metadata WHERE doc_id IN ({placeholders})", doc_ids)
            else:
                cursor.execute("DELETE FROM chunks_fts")
                cursor.execute("DELETE FROM chunk_metadata")

            conn.commit()

    def get_stats(self, collection: Optional[str] = None) -> dict:
        """Get statistics about the index."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if collection:
                cursor.execute("""
                    SELECT COUNT(*) as count FROM chunk_metadata WHERE collection = ?
                """, (collection,))
            else:
                cursor.execute("SELECT COUNT(*) as count FROM chunk_metadata")

            count = cursor.fetchone()["count"]

            # Get unique file count
            if collection:
                cursor.execute("""
                    SELECT COUNT(DISTINCT file_path) as files FROM chunk_metadata
                    WHERE collection = ?
                """, (collection,))
            else:
                cursor.execute("SELECT COUNT(DISTINCT file_path) as files FROM chunk_metadata")

            files = cursor.fetchone()["files"]

            return {
                "total_chunks": count,
                "unique_files": files,
                "db_path": self.db_path
            }

    def build_from_vector_store(self, vector_store, collection_name: Optional[str] = None) -> None:
        """
        Build FTS index from existing vector store data.

        Args:
            vector_store: VectorStore instance to read from
            collection_name: Optional collection name filter
        """
        from hashlib import md5

        target = collection_name or vector_store.default_collection

        try:
            offset = None
            indexed_count = 0

            with self._get_connection() as conn:
                cursor = conn.cursor()

                while True:
                    response = vector_store.client.scroll(
                        collection_name=target,
                        limit=100,
                        with_payload=True,
                        offset=offset
                    )

                    for point in response[0]:
                        payload = point.payload
                        file_path = payload.get("file_path", "")
                        content = payload.get("content", "")
                        start_line = payload.get("start_line", 1)
                        end_line = payload.get("end_line", 1)
                        language = payload.get("language", "text")
                        symbol_name = payload.get("symbol_name")
                        symbol_type = payload.get("symbol_type")
                        parent_context = payload.get("parent_context")

                        # Generate same ID as vector store
                        id_input = f"{file_path}:{start_line}:{end_line}"
                        doc_id = md5(id_input.encode()).hexdigest()

                        # Pre-process for FTS
                        processed_content = self._tokenize_for_index(content)
                        processed_symbol = self._tokenize_for_index(symbol_name or "")
                        processed_parent = self._tokenize_for_index(parent_context or "")

                        # Insert into FTS
                        cursor.execute("""
                            INSERT OR REPLACE INTO chunks_fts
                            (doc_id, file_path, content, symbol_name, symbol_type,
                             parent_context, language, start_line, end_line, collection)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            doc_id, file_path, processed_content, processed_symbol,
                            symbol_type or "", processed_parent, language,
                            start_line, end_line, target
                        ))

                        # Insert into metadata
                        cursor.execute("""
                            INSERT OR REPLACE INTO chunk_metadata
                            (doc_id, file_path, start_line, end_line, language,
                             symbol_name, symbol_type, parent_context, collection, content)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            doc_id, file_path, start_line, end_line, language,
                            symbol_name, symbol_type, parent_context, target, content
                        ))

                        indexed_count += 1

                    conn.commit()

                    offset = response[1]
                    if offset is None:
                        break

            logger.info(f"Built FTS index with {indexed_count} documents from collection '{target}'")

        except Exception as e:
            logger.error(f"Error building FTS index: {e}")

