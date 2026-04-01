import os
from dataclasses import dataclass, field
from typing import List, Optional, Set
import tree_sitter_languages
from tree_sitter import Node


@dataclass
class Chunk:
    """
    Represents a semantic chunk of code with metadata for improved retrieval.
    """
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str
    # Enhanced metadata for better retrieval
    symbol_name: Optional[str] = None          # Function/class name if applicable
    symbol_type: Optional[str] = None          # 'class', 'function', 'method', etc.
    parent_context: Optional[str] = None       # Parent class/module name
    parent_symbols: List[str] = field(default_factory=list)  # Hierarchy of parent symbols
    docstring: Optional[str] = None            # Extracted docstring if any
    imports_context: Optional[str] = None      # Related imports for context
    content_hash: Optional[str] = None         # Hash of the source file content

class Chunker:
    def __init__(self):
        # We'll support a few common languages initially
        self.supported_langs = {
            "kt": "kotlin",
            "java": "java",
            "xml": "xml",
            "py": "python",
            "gradle": "kotlin",  # Build scripts are often Kotlin
            "kts": "kotlin",
            "js": "javascript",
            "jsx": "javascript",
            "ts": "typescript",
            "tsx": "tsx",
            "go": "go",
            "rs": "rust",
            "rb": "ruby",
            "c": "c",
            "cpp": "cpp",
            "h": "c",
            "hpp": "cpp",
            "cs": "c_sharp",
            "swift": "swift",
            "php": "php",
        }

        # Node types that represent significant code structures (language-agnostic)
        self.significant_node_types = {
            # Classes and similar structures
            "class_definition", "class_declaration", "class_specifier",
            "interface_declaration", "struct_specifier", "enum_declaration",
            "trait_declaration", "impl_item", "module_definition",
            # Functions and methods
            "function_definition", "function_declaration", "method_definition",
            "method_declaration", "function_item", "arrow_function",
            "lambda", "lambda_expression",
            # Decorators (Python) - we want to include these with their target
            "decorated_definition",
            # Other significant blocks
            "export_statement", "const_declaration", "let_declaration",
            "variable_declaration",  # For top-level consts/vars
            "type_alias_declaration", "interface_declaration",
            # XML/HTML
            "tag", "element",
        }

        # Map node types to semantic symbol types
        self._symbol_type_map = {
            "class_definition": "class",
            "class_declaration": "class",
            "class_specifier": "class",
            "interface_declaration": "interface",
            "struct_specifier": "struct",
            "enum_declaration": "enum",
            "trait_declaration": "trait",
            "impl_item": "impl",
            "module_definition": "module",
            "function_definition": "function",
            "function_declaration": "function",
            "method_definition": "method",
            "method_declaration": "method",
            "function_item": "function",
            "arrow_function": "function",
            "lambda": "lambda",
            "lambda_expression": "lambda",
            "decorated_definition": "decorated",
            "const_declaration": "const",
            "let_declaration": "variable",
            "variable_declaration": "variable",
            "type_alias_declaration": "type",
        }

        # Node types that are class-like containers
        self._class_like_types = {
            "class_definition", "class_declaration", "class_specifier",
            "interface_declaration", "struct_specifier", "trait_declaration",
            "impl_item", "module_definition", "enum_declaration"
        }

        # Minimum lines for a chunk to be worthwhile on its own
        self.min_chunk_lines = 3
        # Maximum lines before we try to split further
        self.max_chunk_lines = 150
        # Maximum characters before we try to split further (~1.5k-2k tokens)
        self.max_chunk_chars = 6000

    def chunk_file(self, file_path: str, content: str) -> List[Chunk]:
        ext = file_path.split(".")[-1].lower()
        if ext not in self.supported_langs:
            return self._generic_chunk(file_path, content, "text")

        lang_name = self.supported_langs[ext]
        try:
            parser = tree_sitter_languages.get_parser(lang_name)
            tree = parser.parse(bytes(content, "utf8"))
            return self._ast_chunk(file_path, content, tree, lang_name)
        except Exception:
            # Fallback to generic chunking if AST parsing fails
            return self._generic_chunk(file_path, content, lang_name)

    def _extract_symbol_name(self, node: Node) -> Optional[str]:
        """Extract the name of a symbol (function, class, etc.) from a node."""
        # Common child types that contain the name
        name_child_types = {"name", "identifier", "property_identifier",
                           "type_identifier", "field_identifier"}

        for child in node.children:
            if child.type in name_child_types:
                return child.text.decode("utf-8") if child.text else None
            # Handle decorated definitions (Python)
            if child.type == "function_definition" or child.type == "class_definition":
                return self._extract_symbol_name(child)

        return None

    def _extract_docstring(self, node: Node, lines: List[str]) -> Optional[str]:
        """Extract docstring from a function or class node."""
        # Look for string literal as first statement in body
        for child in node.children:
            if child.type in ("block", "body", "class_body", "function_body"):
                for stmt in child.children:
                    if stmt.type in ("expression_statement", "string"):
                        text = stmt.text.decode("utf-8") if stmt.text else ""
                        # Check if it looks like a docstring
                        if text.startswith('"""') or text.startswith("'''"):
                            return text[:200]  # Truncate long docstrings
                        if text.startswith('"') or text.startswith("'"):
                            return text[:200]
                    # Only check first statement
                    break
        return None

    def _find_parent_context(self, node: Node, tree_root: Node) -> List[str]:
        """Find parent class/module names for a node."""
        parents = []
        current = node.parent

        while current and current != tree_root:
            if current.type in self._class_like_types:
                name = self._extract_symbol_name(current)
                if name:
                    parents.append(name)
            current = current.parent

        # Reverse to get outermost first
        parents.reverse()
        return parents

    def _ast_chunk(self, file_path: str, content: str, tree, language: str) -> List[Chunk]:
        lines = content.splitlines()

        # For very small files, return as single chunk
        if len(lines) <= 30:
            return [Chunk(file_path, 1, len(lines), content, language)]

        # Extract file-level imports for context
        imports_context = self._extract_imports(tree.root_node, lines)

        # Recursively find all significant nodes with parent tracking
        significant_nodes = self._find_significant_nodes(tree.root_node, lines)

        if not significant_nodes:
            return self._generic_chunk(file_path, content, language)

        chunks = []
        covered_lines = set()

        for node in significant_nodes:
            start_row = node.start_point[0]
            end_row = node.end_point[0]

            # Skip if this range is already covered by a parent chunk
            node_range = set(range(start_row, end_row + 1))
            if node_range.issubset(covered_lines):
                continue

            # Extract content for this node
            node_lines = lines[start_row:end_row + 1]
            node_content = "\n".join(node_lines)

            # Extract metadata
            symbol_name = self._extract_symbol_name(node)
            symbol_type = self._symbol_type_map.get(node.type)
            parent_symbols = self._find_parent_context(node, tree.root_node)
            parent_context = ".".join(parent_symbols) if parent_symbols else None
            docstring = self._extract_docstring(node, lines)

            # If chunk is too large (lines or characters), try to split it into smaller pieces
            if len(node_lines) > self.max_chunk_lines or len(node_content) > self.max_chunk_chars:
                sub_chunks = self._split_large_node(file_path, node, lines, language,
                                                    parent_symbols, imports_context)
                chunks.extend(sub_chunks)
                for sc in sub_chunks:
                    covered_lines.update(range(sc.start_line - 1, sc.end_line))
            else:
                chunks.append(Chunk(
                    file_path=file_path,
                    start_line=start_row + 1,
                    end_line=end_row + 1,
                    content=node_content,
                    language=language,
                    symbol_name=symbol_name,
                    symbol_type=symbol_type,
                    parent_context=parent_context,
                    parent_symbols=parent_symbols,
                    docstring=docstring,
                    imports_context=imports_context
                ))
                covered_lines.update(node_range)

        # Add any uncovered significant sections (imports, module-level code)
        chunks.extend(self._capture_uncovered_sections(file_path, lines, covered_lines, language))

        # Sort chunks by start line
        chunks.sort(key=lambda c: c.start_line)

        return chunks if chunks else self._generic_chunk(file_path, content, language)

    def _extract_imports(self, root_node: Node, lines: List[str]) -> Optional[str]:
        """Extract import statements from the file for context."""
        import_types = {"import_statement", "import_from_statement", "import_declaration",
                       "use_declaration", "require_statement", "include_statement"}
        imports = []

        for child in root_node.children:
            if child.type in import_types:
                start = child.start_point[0]
                end = child.end_point[0]
                import_text = "\n".join(lines[start:end + 1])
                imports.append(import_text)
            # Limit to first 10 imports to avoid huge context
            if len(imports) >= 10:
                break

        return "\n".join(imports) if imports else None

    def _find_significant_nodes(self, node: Node, lines: List[str], depth: int = 0) -> List[Node]:
        """Recursively find all significant code structure nodes."""
        significant = []

        # Check if current node is significant
        if node.type in self.significant_node_types:
            node_lines = node.end_point[0] - node.start_point[0] + 1
            # Only include if it's meaningful (not too small)
            if node_lines >= self.min_chunk_lines:
                significant.append(node)
                # Don't recurse into children of small-medium nodes
                # But do recurse if characters exceed limit
                node_content_len = node.end_byte - node.start_byte
                if node_lines <= self.max_chunk_lines and node_content_len <= self.max_chunk_chars:
                    return significant

        # Recurse into children
        for child in node.children:
            significant.extend(self._find_significant_nodes(child, lines, depth + 1))

        return significant

    def _split_large_node(self, file_path: str, node: Node, lines: List[str],
                          language: str, parent_symbols: Optional[List[str]] = None,
                          imports_context: Optional[str] = None) -> List[Chunk]:
        """Split a large node into smaller chunks based on its children."""
        chunks = []
        parent_symbols = parent_symbols or []

        # The large node itself might be a class - add it to parent context
        node_name = self._extract_symbol_name(node)
        if node.type in self._class_like_types and node_name:
            current_parent_symbols = parent_symbols + [node_name]
        else:
            current_parent_symbols = parent_symbols

        # Find significant children within this node
        child_nodes = []
        for child in node.children:
            if child.type in self.significant_node_types:
                child_lines = child.end_point[0] - child.start_point[0] + 1
                if child_lines >= self.min_chunk_lines:
                    child_nodes.append(child)

        if child_nodes:
            # Create chunks for each significant child
            for child in child_nodes:
                start_row = child.start_point[0]
                end_row = child.end_point[0]
                node_lines = lines[start_row:end_row + 1]
                node_content = "\n".join(node_lines)

                # Extract metadata for child
                symbol_name = self._extract_symbol_name(child)
                symbol_type = self._symbol_type_map.get(child.type)
                parent_context = ".".join(current_parent_symbols) if current_parent_symbols else None
                docstring = self._extract_docstring(child, lines)

                chunks.append(Chunk(
                    file_path=file_path,
                    start_line=start_row + 1,
                    end_line=end_row + 1,
                    content=node_content,
                    language=language,
                    symbol_name=symbol_name,
                    symbol_type=symbol_type,
                    parent_context=parent_context,
                    parent_symbols=current_parent_symbols,
                    docstring=docstring,
                    imports_context=imports_context
                ))
        else:
            # No significant children, use generic chunking for this range
            start_row = node.start_point[0]
            end_row = node.end_point[0]
            node_content = "\n".join(lines[start_row:end_row + 1])
            parent_context = ".".join(current_parent_symbols) if current_parent_symbols else None
            chunks.extend(self._generic_chunk_content(
                file_path, node_content, language,
                start_offset=start_row,
                parent_context=parent_context,
                parent_symbols=current_parent_symbols,
                imports_context=imports_context
            ))

        return chunks

    def _capture_uncovered_sections(self, file_path: str, lines: List[str],
                                     covered_lines: set, language: str) -> List[Chunk]:
        """Capture important uncovered sections like imports and module-level code."""
        chunks = []
        uncovered_start = None

        for i, line in enumerate(lines):
            if i not in covered_lines:
                if uncovered_start is None:
                    uncovered_start = i
            else:
                if uncovered_start is not None:
                    # We have an uncovered section
                    section_lines = lines[uncovered_start:i]
                    # Only create chunk if section has meaningful content
                    if self._has_meaningful_content(section_lines):
                        chunks.append(Chunk(
                            file_path=file_path,
                            start_line=uncovered_start + 1,
                            end_line=i,
                            content="\n".join(section_lines),
                            language=language
                        ))
                    uncovered_start = None

        # Handle trailing uncovered section
        if uncovered_start is not None:
            section_lines = lines[uncovered_start:]
            if self._has_meaningful_content(section_lines):
                chunks.append(Chunk(
                    file_path=file_path,
                    start_line=uncovered_start + 1,
                    end_line=len(lines),
                    content="\n".join(section_lines),
                    language=language
                ))

        return chunks

    def _has_meaningful_content(self, lines: List[str]) -> bool:
        """Check if lines contain meaningful code (not just whitespace/comments)."""
        non_empty = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]
        return len(non_empty) >= 2

    def _generic_chunk_content(self, file_path: str, content: str, language: str,
                                start_offset: int = 0,
                                parent_context: Optional[str] = None,
                                parent_symbols: Optional[List[str]] = None,
                                imports_context: Optional[str] = None) -> List[Chunk]:
        """Generic chunking for a specific content section with semantic boundaries."""
        lines = content.splitlines()
        chunk_size = self.max_chunk_lines
        overlap = 5
        chunks = []
        parent_symbols = parent_symbols or []
        content_len = len(content)

        if len(lines) <= chunk_size and content_len <= self.max_chunk_chars:
            return [Chunk(
                file_path=file_path,
                start_line=start_offset + 1,
                end_line=start_offset + len(lines),
                content=content,
                language=language,
                parent_context=parent_context,
                parent_symbols=parent_symbols,
                imports_context=imports_context
            )]

        i = 0
        while i < len(lines):
            # Find end of chunk, respecting semantic boundaries
            target_end = min(i + chunk_size, len(lines))
            
            # Also respect character limit
            actual_end_limit = target_end
            while actual_end_limit > i + 1:
                chunk_len = sum(len(lines[j]) + 1 for j in range(i, actual_end_limit))
                if chunk_len <= self.max_chunk_chars:
                    break
                actual_end_limit -= 1
            
            actual_end = self._find_semantic_boundary(lines, i, actual_end_limit, chunk_size)

            chunk_content = "\n".join(lines[i:actual_end])
            chunks.append(Chunk(
                file_path=file_path,
                start_line=start_offset + i + 1,
                end_line=start_offset + actual_end,
                content=chunk_content,
                language=language,
                parent_context=parent_context,
                parent_symbols=parent_symbols,
                imports_context=imports_context
            ))

            if actual_end >= len(lines):
                break

            # Start next chunk with overlap, but respect boundaries
            i = max(actual_end - overlap, i + 1)

        return chunks

    def _find_semantic_boundary(
        self,
        lines: List[str],
        start: int,
        target_end: int,
        max_chunk_size: int
    ) -> int:
        """
        Find a semantic boundary near the target end position.

        Tries to avoid splitting:
        - Docstrings (triple-quoted strings)
        - Multi-line expressions
        - Blocks with unbalanced braces

        Returns:
            Adjusted end position
        """
        if target_end >= len(lines):
            return len(lines)

        # Check if we're inside a docstring
        in_docstring = self._check_in_docstring(lines, start, target_end)
        if in_docstring:
            # Find end of docstring
            docstring_end = self._find_docstring_end(lines, target_end)
            if docstring_end and docstring_end < start + max_chunk_size + 20:
                return docstring_end + 1

        # Look for good break points within a small range
        search_range = min(10, target_end - start)
        best_boundary = target_end

        for offset in range(search_range):
            check_pos = target_end - offset
            if check_pos <= start:
                break

            line = lines[check_pos - 1] if check_pos > 0 else ""

            # Good boundary: empty line
            if not line.strip():
                best_boundary = check_pos
                break

            # Good boundary: line at same or lower indent as function start
            if self._is_block_boundary(lines, start, check_pos):
                best_boundary = check_pos
                break

        return best_boundary

    def _check_in_docstring(self, lines: List[str], start: int, end: int) -> bool:
        """Check if the end position is inside a docstring."""
        # Count triple quotes from start to end
        triple_double = 0
        triple_single = 0

        for i in range(start, min(end, len(lines))):
            line = lines[i]
            triple_double += line.count('"""')
            triple_single += line.count("'''")

        # Odd count means we're inside a docstring
        return (triple_double % 2 == 1) or (triple_single % 2 == 1)

    def _find_docstring_end(self, lines: List[str], from_pos: int) -> Optional[int]:
        """Find the end of a docstring starting from position."""
        for i in range(from_pos, min(from_pos + 30, len(lines))):
            line = lines[i]
            if '"""' in line or "'''" in line:
                return i
        return None

    def _is_block_boundary(self, lines: List[str], start: int, pos: int) -> bool:
        """Check if position is at a good block boundary (dedented line, etc.)."""
        if pos <= 0 or pos >= len(lines):
            return False

        # Get indent of first non-empty line after start
        start_indent = None
        for i in range(start, min(start + 5, len(lines))):
            if lines[i].strip():
                start_indent = len(lines[i]) - len(lines[i].lstrip())
                break

        if start_indent is None:
            return True

        current_line = lines[pos]
        if not current_line.strip():
            return True

        current_indent = len(current_line) - len(current_line.lstrip())

        # Boundary if we've dedented back to or past the starting indent
        return start_indent is not None and current_indent <= start_indent

    def _generic_chunk(self, file_path: str, content: str, language: str) -> List[Chunk]:
        """Line-based chunking with semantic boundary awareness."""
        lines = content.splitlines()
        chunk_size = self.max_chunk_lines
        overlap = 5
        chunks = []

        if len(lines) <= chunk_size:
            # Still check character limit for safety
            if len(content) <= self.max_chunk_chars:
                chunks.append(Chunk(file_path, 1, len(lines), content, language))
                return chunks
            # Else fall through to chunking loop

        i = 0
        while i < len(lines):
            target_end = min(i + chunk_size, len(lines))
            
            # Also respect character limit
            actual_end_limit = target_end
            while actual_end_limit > i + 1:
                chunk_len = sum(len(lines[j]) + 1 for j in range(i, actual_end_limit))
                if chunk_len <= self.max_chunk_chars:
                    break
                actual_end_limit -= 1
            
            actual_end = self._find_semantic_boundary(lines, i, actual_end_limit, chunk_size)

            chunk_content = "\n".join(lines[i:actual_end])
            chunks.append(Chunk(
                file_path=file_path,
                start_line=i + 1,
                end_line=actual_end,
                content=chunk_content,
                language=language
            ))

            if actual_end >= len(lines):
                break

            i = max(actual_end - overlap, i + 1)

        return chunks
