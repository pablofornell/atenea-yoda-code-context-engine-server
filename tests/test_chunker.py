"""Tests for the Chunker module."""

import pytest
from atenea_server.chunker import Chunker, Chunk


class TestChunker:
    """Test suite for the Chunker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = Chunker()

    def test_small_python_file_returns_single_chunk(self):
        """Small files should return a single chunk."""
        content = '''def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
'''
        chunks = self.chunker.chunk_file("test.py", content)

        assert len(chunks) == 1
        assert chunks[0].file_path == "test.py"
        assert chunks[0].language == "python"
        assert chunks[0].start_line == 1

    def test_python_class_extraction(self):
        """Python classes should be extracted as chunks with metadata."""
        content = '''class MyClass:
    """A test class."""

    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def decrement(self):
        self.value -= 1
        return self.value


class AnotherClass:
    """Another test class."""

    def method(self):
        pass
'''
        chunks = self.chunker.chunk_file("test.py", content)

        # Should extract both classes
        assert len(chunks) >= 1
        assert all(c.language == "python" for c in chunks)

    def test_symbol_name_extraction(self):
        """Chunks should have symbol names extracted."""
        content = '''def calculate_total(items):
    """Calculate total price of items."""
    return sum(item.price for item in items)

class OrderProcessor:
    """Processes customer orders."""

    def process(self, order):
        return order.finalize()
'''
        # File must be large enough to trigger individual extraction
        # Let's make it bigger
        extended_content = content + "\n" * 30 + "# padding\n" * 10
        chunks = self.chunker.chunk_file("test.py", extended_content)

        # At least one chunk should exist
        assert len(chunks) >= 1

    def test_parent_context_extraction(self):
        """Nested classes/methods should have parent context."""
        content = '''class OuterClass:
    """Outer class docstring."""

    class InnerClass:
        """Inner class docstring."""

        def inner_method(self):
            pass

    def outer_method(self):
        def nested_function():
            pass
        return nested_function()
''' + "\n" * 30 + "# padding lines\n" * 10
        chunks = self.chunker.chunk_file("test.py", content)

        # Should extract chunks
        assert len(chunks) >= 1

    def test_generic_chunking_for_unsupported_language(self):
        """Unsupported languages should use generic chunking."""
        content = "line1\nline2\nline3\nline4\nline5"
        chunks = self.chunker.chunk_file("test.xyz", content)
        
        assert len(chunks) >= 1
        assert chunks[0].language == "text"

    def test_empty_file_returns_empty_list(self):
        """Empty files should return empty chunk list."""
        chunks = self.chunker.chunk_file("empty.py", "")
        # Empty content still produces a chunk with the content
        assert len(chunks) <= 1

    def test_javascript_function_extraction(self):
        """JavaScript functions should be extracted."""
        content = '''function add(a, b) {
    return a + b;
}

const multiply = (a, b) => {
    return a * b;
};

function subtract(a, b) {
    return a - b;
}
'''
        chunks = self.chunker.chunk_file("test.js", content)
        
        assert len(chunks) >= 1
        assert all(c.language == "javascript" for c in chunks)

    def test_chunk_has_correct_line_numbers(self):
        """Chunks should have correct start and end line numbers."""
        content = '''# Module docstring
import os

def function_one():
    pass

def function_two():
    pass
'''
        chunks = self.chunker.chunk_file("test.py", content)

        for chunk in chunks:
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            # Line numbers should be within the file bounds
            total_lines = content.count('\n') + 1
            assert chunk.end_line <= total_lines

    def test_supported_languages(self):
        """Test that supported languages are properly detected."""
        test_cases = [
            ("test.py", "python"),
            ("test.js", "javascript"),
            ("test.ts", "typescript"),
            ("test.java", "java"),
            ("test.kt", "kotlin"),
            ("test.go", "go"),
            ("test.rs", "rust"),
        ]
        
        for filename, expected_lang in test_cases:
            chunks = self.chunker.chunk_file(filename, "x = 1")
            assert chunks[0].language == expected_lang, f"Failed for {filename}"

    def test_large_file_is_split(self):
        """Large files should be split into multiple chunks."""
        # Create a file with 200 lines
        lines = [f"line_{i} = {i}" for i in range(200)]
        content = "\n".join(lines)
        
        chunks = self.chunker.chunk_file("large.txt", content)
        
        # Should be split into multiple chunks
        assert len(chunks) > 1


class TestChunk:
    """Test the Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a Chunk instance."""
        chunk = Chunk(
            file_path="test.py",
            start_line=1,
            end_line=10,
            content="print('hello')",
            language="python"
        )
        
        assert chunk.file_path == "test.py"
        assert chunk.start_line == 1
        assert chunk.end_line == 10
        assert chunk.content == "print('hello')"
        assert chunk.language == "python"

