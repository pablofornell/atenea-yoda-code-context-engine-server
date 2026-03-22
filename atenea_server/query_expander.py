"""
Query expansion for improved code search.

This module provides query expansion capabilities to improve retrieval
by adding synonyms, related terms, and common variations for code-related queries.
"""

from typing import List, Set
import re


# Common programming term expansions
CODE_EXPANSIONS = {
    # Authentication & Authorization
    "auth": ["authentication", "authorization", "login", "authenticate", "authorize"],
    "login": ["signin", "sign_in", "authenticate", "logon"],
    "logout": ["signout", "sign_out", "logoff"],
    "password": ["passwd", "pwd", "secret", "credential"],
    "token": ["jwt", "bearer", "access_token", "refresh_token"],
    
    # CRUD operations
    "create": ["add", "insert", "new", "make", "generate"],
    "read": ["get", "fetch", "retrieve", "find", "load", "query"],
    "update": ["edit", "modify", "change", "patch", "put"],
    "delete": ["remove", "destroy", "drop", "erase"],
    
    # Data types
    "string": ["str", "text", "varchar", "char"],
    "integer": ["int", "number", "num", "long"],
    "boolean": ["bool", "flag", "is_"],
    "array": ["list", "collection", "vector", "slice"],
    "dict": ["dictionary", "map", "hashmap", "object", "hash"],
    
    # Common patterns
    "config": ["configuration", "settings", "options", "preferences", "conf"],
    "init": ["initialize", "initialization", "setup", "bootstrap"],
    "parse": ["parsing", "parser", "decode", "deserialize"],
    "serialize": ["marshal", "encode", "dump", "stringify"],
    "validate": ["validation", "validator", "check", "verify"],
    
    # Error handling
    "error": ["err", "exception", "failure", "fault"],
    "handle": ["handler", "handling", "catch", "process"],
    "retry": ["retries", "backoff", "reattempt"],
    
    # Async/Concurrency
    "async": ["asynchronous", "await", "promise", "future"],
    "sync": ["synchronous", "blocking", "sequential"],
    "thread": ["threading", "concurrent", "parallel"],
    "lock": ["mutex", "semaphore", "synchronize"],
    
    # Testing
    "test": ["testing", "unittest", "spec", "assert"],
    "mock": ["stub", "fake", "spy", "double"],
    
    # Database
    "db": ["database", "datastore", "storage"],
    "query": ["sql", "select", "find", "search"],
    "connect": ["connection", "conn", "client"],
    
    # Web/API
    "api": ["endpoint", "route", "handler", "controller"],
    "request": ["req", "http_request"],
    "response": ["res", "resp", "http_response"],
    "middleware": ["interceptor", "filter", "hook"],
    
    # Common abbreviations
    "msg": ["message"],
    "ctx": ["context"],
    "env": ["environment"],
    "repo": ["repository"],
    "util": ["utility", "utils", "helper", "helpers"],
    "impl": ["implementation", "implement"],
    "param": ["parameter", "params", "argument", "args"],
    "func": ["function", "fn", "method"],
    "var": ["variable"],
    "const": ["constant"],
    "attr": ["attribute", "property", "prop"],
    "idx": ["index"],
    "len": ["length", "size", "count"],
    "max": ["maximum"],
    "min": ["minimum"],
    "avg": ["average", "mean"],
    "tmp": ["temp", "temporary"],
    "src": ["source"],
    "dst": ["dest", "destination"],
    "buf": ["buffer"],
    "ptr": ["pointer"],
    "ref": ["reference"],
}

# Build reverse mapping
_REVERSE_EXPANSIONS = {}
for key, values in CODE_EXPANSIONS.items():
    for value in values:
        if value not in _REVERSE_EXPANSIONS:
            _REVERSE_EXPANSIONS[value] = set()
        _REVERSE_EXPANSIONS[value].add(key)
        _REVERSE_EXPANSIONS[value].update(v for v in values if v != value)


class QueryExpander:
    """
    Expands search queries with related terms for improved recall.
    """
    
    def __init__(self, max_expansions: int = 5):
        """
        Initialize the query expander.
        
        Args:
            max_expansions: Maximum number of expanded terms to add per query term
        """
        self.max_expansions = max_expansions
        self._word_pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    
    def expand(self, query: str) -> str:
        """
        Expand a query with related terms.
        
        Args:
            query: Original search query
            
        Returns:
            Expanded query with additional terms
        """
        words = self._word_pattern.findall(query.lower())
        expanded_terms: Set[str] = set(words)
        
        for word in words:
            # Check direct expansions
            if word in CODE_EXPANSIONS:
                expansions = CODE_EXPANSIONS[word][:self.max_expansions]
                expanded_terms.update(expansions)
            
            # Check reverse expansions
            if word in _REVERSE_EXPANSIONS:
                expansions = list(_REVERSE_EXPANSIONS[word])[:self.max_expansions]
                expanded_terms.update(expansions)
        
        # Return original query plus expansions
        new_terms = expanded_terms - set(words)
        if new_terms:
            return f"{query} {' '.join(new_terms)}"
        return query
    
    def get_related_terms(self, term: str) -> List[str]:
        """
        Get related terms for a single word.
        
        Args:
            term: Single term to expand
            
        Returns:
            List of related terms
        """
        term_lower = term.lower()
        related = set()
        
        if term_lower in CODE_EXPANSIONS:
            related.update(CODE_EXPANSIONS[term_lower])
        
        if term_lower in _REVERSE_EXPANSIONS:
            related.update(_REVERSE_EXPANSIONS[term_lower])
        
        return list(related)

