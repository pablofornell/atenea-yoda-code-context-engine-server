from typing import List, Optional, Dict, Any
import logging
import os
from .embedder import Embedder, EmbeddingTaskType
from .vector_store import VectorStore
from .fts_index import FTSIndex
from .query_expander import QueryExpander

logger = logging.getLogger(__name__)


class Retriever:
    """
    Hybrid retriever combining semantic vector search with FTS5 keyword search.

    The hybrid approach improves retrieval quality by:
    - Using vector search for semantic similarity (captures meaning)
    - Using FTS5/BM25 for exact keyword matches (captures specific symbols, names)
    - Query expansion for improved recall on code-specific terms
    - Combining scores with Reciprocal Rank Fusion
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore,
                 vector_weight: float = 0.7, bm25_weight: float = 0.3,
                 fts_db_path: Optional[str] = None,
                 enable_query_expansion: bool = True):
        """
        Initialize the hybrid retriever.

        Args:
            embedder: Embedder instance for generating query embeddings
            vector_store: VectorStore instance for semantic search
            vector_weight: Weight for vector search scores (default: 0.7)
            bm25_weight: Weight for BM25/FTS search scores (default: 0.3)
            fts_db_path: Optional path to SQLite FTS database
            enable_query_expansion: Whether to expand queries with related terms
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.enable_query_expansion = enable_query_expansion

        # Persistent FTS index
        self._fts_index = FTSIndex(db_path=fts_db_path)
        self._fts_initialized: Dict[str, bool] = {}

        # Query expander for better recall
        self._query_expander = QueryExpander(max_expansions=3)

    def _ensure_fts_index(self, collection_name: Optional[str] = None) -> FTSIndex:
        """Ensure FTS index is built for the collection."""
        target = collection_name or self.vector_store.default_collection

        # Check if we need to initialize from vector store
        if not self._fts_initialized.get(target, False):
            stats = self._fts_index.get_stats(collection=target)
            if stats["total_chunks"] == 0:
                logger.info(f"Building FTS index for collection '{target}'...")
                self._fts_index.build_from_vector_store(
                    self.vector_store, collection_name=target
                )
            self._fts_initialized[target] = True

        return self._fts_index

    def invalidate_fts_index(self, collection_name: Optional[str] = None):
        """
        Invalidate the FTS index for a collection, forcing rebuild on next search.

        Call this after indexing new documents.
        """
        target = collection_name or self.vector_store.default_collection
        self._fts_initialized[target] = False
        self._fts_index.clear(collection=target)



    async def retrieve(
        self,
        query: str,
        limit: int = 20,
        collection_name: Optional[str] = None,
        use_hybrid: bool = True,
        language_filter: Optional[str] = None,
        symbol_type_filter: Optional[str] = None,
        file_path_pattern: Optional[str] = None,
        max_results_per_file: int = 3
    ) -> List[dict]:
        """
        Retrieve relevant code chunks using hybrid search.

        Args:
            query: Search query
            limit: Maximum number of results to return
            collection_name: Optional collection to search in
            use_hybrid: If True, use hybrid search. If False, use vector-only search.
            language_filter: Optional filter by programming language
            symbol_type_filter: Optional filter by symbol type ('function', 'class', etc.)
            file_path_pattern: Optional SQL LIKE pattern for file paths
            max_results_per_file: Maximum results from the same file (for diversity)

        Returns:
            List of matching chunks with their payloads
        """
        # 0. Expand query with related terms for better recall
        search_query = query
        if self.enable_query_expansion:
            search_query = self._query_expander.expand(query)
            if search_query != query:
                logger.debug(f"Expanded query: '{query}' -> '{search_query}'")

        # 1. Vector search (use SEARCH_QUERY task type for asymmetric retrieval)
        # Note: We use the original query for vector search (semantic meaning)
        # and the expanded query for FTS (keyword matching)
        query_embeddings = await self.embedder.embed(
            [query],  # Original query for semantic search
            raise_on_error=False,
            task_type=EmbeddingTaskType.SEARCH_QUERY
        )

        vector_results = []
        if query_embeddings:
            # Request more results to allow for fusion and deduplication
            raw_vector_results = self.vector_store.search(
                query_embeddings[0],
                limit=limit * 3,
                collection_name=collection_name
            )
            # Apply filters to vector results if specified
            if language_filter or symbol_type_filter or file_path_pattern:
                raw_vector_results = self._apply_filters(
                    raw_vector_results, language_filter,
                    symbol_type_filter, file_path_pattern
                )

            # Convert to (key, score, payload) format
            for i, res in enumerate(raw_vector_results):
                score = 1.0 / (i + 1)  # Reciprocal rank
                vector_results.append((self._get_doc_key(res), score, res))

        if not use_hybrid or not vector_results:
            # Fallback to vector-only results
            return self._deduplicate_results(
                [r[2] for r in vector_results], limit, max_results_per_file
            )

        # 2. FTS/BM25 search with filters (using expanded query)
        fts_index = self._ensure_fts_index(collection_name)
        fts_results = fts_index.search(
            search_query,  # Expanded query for keyword search
            limit=limit * 3,
            collection=collection_name,
            language_filter=language_filter,
            symbol_type_filter=symbol_type_filter,
            file_path_pattern=file_path_pattern
        )

        # Convert to (key, score, payload) format with reciprocal rank
        normalized_fts = []
        for i, (doc_id, score, payload) in enumerate(fts_results):
            normalized_fts.append((self._get_doc_key(payload), 1.0 / (i + 1), payload))

        # 3. Reciprocal Rank Fusion (RRF)
        fused_results = self._reciprocal_rank_fusion(
            vector_results, normalized_fts, k=60
        )

        # 4. Deduplicate with diversity and limit
        results = self._deduplicate_results(fused_results, limit, max_results_per_file)

        # 5. Expand context for truncated chunks if needed
        return self._expand_context(results)

    def _apply_filters(
        self,
        results: List[dict],
        language_filter: Optional[str],
        symbol_type_filter: Optional[str],
        file_path_pattern: Optional[str]
    ) -> List[dict]:
        """Apply metadata filters to results."""
        filtered = []
        for res in results:
            if language_filter and res.get("language") != language_filter:
                continue
            if symbol_type_filter and res.get("symbol_type") != symbol_type_filter:
                continue
            if file_path_pattern:
                import fnmatch
                if not fnmatch.fnmatch(res.get("file_path", ""), file_path_pattern.replace("%", "*")):
                    continue
            filtered.append(res)
        return filtered

    def _get_doc_key(self, payload: dict) -> str:
        """Generate a unique key for a document/chunk."""
        return f"{payload.get('file_path', '')}:{payload.get('start_line', 0)}:{payload.get('end_line', 0)}"

    def _reciprocal_rank_fusion(self, vector_results: List[tuple],
                                  bm25_results: List[tuple],
                                  k: int = 60) -> List[dict]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF is a robust method for combining ranked lists that doesn't require
        score normalization across different retrieval methods.

        Args:
            vector_results: List of (key, score, payload) from vector search
            bm25_results: List of (key, score, payload) from BM25 search
            k: Constant to prevent high ranks from dominating (default: 60)

        Returns:
            List of payloads sorted by fused score
        """
        scores: Dict[str, float] = {}
        payloads: Dict[str, dict] = {}

        # Process vector results
        for rank, (key, _, payload) in enumerate(vector_results, start=1):
            scores[key] = scores.get(key, 0) + self.vector_weight / (k + rank)
            payloads[key] = payload

        # Process BM25 results
        for rank, (key, _, payload) in enumerate(bm25_results, start=1):
            scores[key] = scores.get(key, 0) + self.bm25_weight / (k + rank)
            if key not in payloads:
                payloads[key] = payload

        # Sort by fused score
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        return [payloads[key] for key in sorted_keys]

    def _deduplicate_results(
        self,
        results: List[dict],
        limit: int,
        max_per_file: int = 3
    ) -> List[dict]:
        """
        Deduplicate and diversify results.

        Args:
            results: List of result payloads
            limit: Maximum total results
            max_per_file: Maximum results from the same file (for diversity)

        Returns:
            Deduplicated and diversified results
        """
        file_counts: Dict[str, int] = {}
        seen_keys = set()
        deduplicated = []

        for res in results:
            file_path = res.get("file_path", "")
            key = self._get_doc_key(res)

            # Skip exact duplicates
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Enforce per-file limit for diversity
            current_count = file_counts.get(file_path, 0)
            if current_count >= max_per_file:
                continue

            deduplicated.append(res)
            file_counts[file_path] = current_count + 1

            if len(deduplicated) >= limit:
                break

        return deduplicated

    def _expand_context(
        self,
        results: List[dict],
        max_expansion_lines: int = 50
    ) -> List[dict]:
        """
        Expand truncated chunks to include more context when available.

        This method checks if a chunk appears to be truncated (ends mid-function)
        and attempts to expand it by reading from the original file.

        Args:
            results: List of result payloads
            max_expansion_lines: Maximum lines to expand beyond the chunk

        Returns:
            Results with expanded context where applicable
        """
        expanded_results = []

        for res in results:
            file_path = res.get("file_path", "")
            content = res.get("content", "")
            start_line = res.get("start_line", 1)
            end_line = res.get("end_line", 1)
            symbol_type = res.get("symbol_type")

            # Check if this looks like a truncated chunk
            # (ends with incomplete syntax indicators)
            if not self._is_potentially_truncated(content, symbol_type):
                expanded_results.append(res)
                continue

            # Try to expand the context by reading more from the file
            expanded = self._try_expand_chunk(
                file_path, content, start_line, end_line, max_expansion_lines
            )

            if expanded:
                # Create a new result with expanded content
                expanded_res = res.copy()
                expanded_res["content"] = expanded["content"]
                expanded_res["end_line"] = expanded["end_line"]
                expanded_res["_expanded"] = True  # Mark as expanded
                expanded_results.append(expanded_res)
            else:
                expanded_results.append(res)

        return expanded_results

    def _is_potentially_truncated(self, content: str, symbol_type: Optional[str]) -> bool:
        """
        Check if a chunk appears to be truncated.

        Looks for indicators that the chunk was cut off mid-definition.
        """
        if not content:
            return False

        # Only expand function/method/class definitions
        if symbol_type not in ("function", "method", "class", "function_definition",
                                "method_definition", "class_definition"):
            return False

        lines = content.strip().splitlines()
        if not lines:
            return False

        last_line = lines[-1].strip()

        # Check for truncation indicators
        truncation_indicators = [
            # Incomplete blocks
            last_line.endswith("{") and content.count("{") > content.count("}"),
            last_line.endswith(":") and not last_line.startswith("return"),
            # Trailing ellipsis or continuation
            last_line.endswith("..."),
            last_line.endswith("\\"),
            # Unbalanced parentheses/brackets
            content.count("(") > content.count(")"),
            content.count("[") > content.count("]"),
        ]

        return any(truncation_indicators)

    def _try_expand_chunk(
        self,
        file_path: str,
        original_content: str,
        start_line: int,
        end_line: int,
        max_expansion: int
    ) -> Optional[dict]:
        """
        Try to expand a chunk by reading more context from the original file.

        Returns:
            Dict with expanded content and new end_line, or None if expansion fails
        """
        try:
            # Read the original file
            if not os.path.exists(file_path):
                return None

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_lines = f.readlines()

            total_lines = len(file_lines)

            # Calculate new end line
            new_end = min(end_line + max_expansion, total_lines)

            # Extract expanded content
            # Note: line numbers are 1-based, list indices are 0-based
            expanded_lines = file_lines[start_line - 1:new_end]
            expanded_content = "".join(expanded_lines)

            # Check if we've reached a natural boundary
            # (matching braces, dedented line, etc.)
            natural_end = self._find_natural_boundary(
                expanded_lines, end_line - start_line
            )

            if natural_end:
                final_end = start_line + natural_end
                expanded_content = "".join(expanded_lines[:natural_end + 1])
            else:
                final_end = new_end

            return {
                "content": expanded_content,
                "end_line": final_end
            }

        except Exception as e:
            logger.debug(f"Failed to expand chunk from {file_path}: {e}")
            return None

    def _find_natural_boundary(
        self,
        lines: List[str],
        start_from: int
    ) -> Optional[int]:
        """
        Find a natural code boundary in the lines (end of function, class, etc.).

        Returns:
            Line index of the natural boundary, or None if not found
        """
        if not lines or start_from >= len(lines):
            return None

        # Track brace/bracket balance
        brace_count = 0
        paren_count = 0
        bracket_count = 0

        # Count existing balance from start
        for i in range(start_from + 1):
            line = lines[i]
            brace_count += line.count("{") - line.count("}")
            paren_count += line.count("(") - line.count(")")
            bracket_count += line.count("[") - line.count("]")

        # Look for balance point or significant dedent
        initial_indent = len(lines[0]) - len(lines[0].lstrip()) if lines[0].strip() else 0

        for i in range(start_from + 1, len(lines)):
            line = lines[i]
            brace_count += line.count("{") - line.count("}")
            paren_count += line.count("(") - line.count(")")
            bracket_count += line.count("[") - line.count("]")

            # Check if we've reached balance
            if brace_count <= 0 and paren_count <= 0 and bracket_count <= 0:
                return i

            # Check for significant dedent (Python-style)
            if line.strip() and not line.strip().startswith("#"):
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= initial_indent and i > start_from + 2:
                    # Found a line at the same or lower indent level
                    return i - 1

        return None
