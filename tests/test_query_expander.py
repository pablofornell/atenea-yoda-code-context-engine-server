"""Tests for the QueryExpander module."""

import pytest
from atenea_server.query_expander import QueryExpander, CODE_EXPANSIONS


class TestQueryExpander:
    """Test suite for the QueryExpander class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.expander = QueryExpander(max_expansions=5)

    def test_expand_auth_query(self):
        """Test expanding authentication-related terms."""
        expanded = self.expander.expand("auth function")
        
        # Should include authentication-related terms
        assert "auth" in expanded
        # Should have added related terms
        assert len(expanded) > len("auth function")
        # Check for expected expansions
        assert any(term in expanded.lower() for term in ["authentication", "authorization", "login"])

    def test_expand_crud_operations(self):
        """Test expanding CRUD operation terms."""
        expanded = self.expander.expand("create user")
        
        # Should include create-related terms
        assert any(term in expanded.lower() for term in ["add", "insert", "new"])

    def test_expand_database_terms(self):
        """Test expanding database-related terms."""
        expanded = self.expander.expand("db query")
        
        assert "db" in expanded
        # Should include database-related expansions
        assert any(term in expanded.lower() for term in ["database", "sql", "select"])

    def test_no_expansion_for_unknown_terms(self):
        """Test that unknown terms are not expanded."""
        query = "xyzunknown abcterm"
        expanded = self.expander.expand(query)
        
        # Should return original query unchanged
        assert expanded == query

    def test_get_related_terms(self):
        """Test getting related terms for a single word."""
        related = self.expander.get_related_terms("config")
        
        assert len(related) > 0
        assert "configuration" in related or "settings" in related

    def test_abbreviation_expansion(self):
        """Test expanding common abbreviations."""
        test_cases = [
            ("msg", "message"),
            ("ctx", "context"),
            ("env", "environment"),
            ("repo", "repository"),
            ("util", "utility"),
            ("impl", "implementation"),
            ("param", "parameter"),
            ("func", "function"),
        ]
        
        for abbrev, expected in test_cases:
            related = self.expander.get_related_terms(abbrev)
            assert expected in related or any(expected in r for r in related), \
                f"Expected '{expected}' in related terms for '{abbrev}'"

    def test_max_expansions_limit(self):
        """Test that max_expansions parameter is respected."""
        limited_expander = QueryExpander(max_expansions=2)
        
        # A term with many expansions
        expanded = limited_expander.expand("auth")
        
        # Count new terms added (original + limited expansions)
        terms = expanded.split()
        # Should not have too many terms
        assert len(terms) <= 5  # Original term + max 2 expansions * 2 sides

    def test_reverse_expansion(self):
        """Test that reverse expansions work (e.g., 'authentication' -> 'auth')."""
        related = self.expander.get_related_terms("authentication")
        
        # Should find the abbreviation
        assert "auth" in related or any("auth" in r for r in related)

    def test_case_insensitivity(self):
        """Test that expansion is case-insensitive."""
        expanded_lower = self.expander.expand("Auth")
        
        # Should still expand
        assert len(expanded_lower) > len("Auth")

    def test_empty_query(self):
        """Test handling of empty query."""
        expanded = self.expander.expand("")
        assert expanded == ""

    def test_code_expansions_dict_populated(self):
        """Test that CODE_EXPANSIONS dictionary has entries."""
        assert len(CODE_EXPANSIONS) > 0
        assert "auth" in CODE_EXPANSIONS
        assert "create" in CODE_EXPANSIONS
        assert "db" in CODE_EXPANSIONS

