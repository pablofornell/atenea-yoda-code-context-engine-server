# Atenea Context Engine - Server

This is the backend server for the Atenea Context Engine. It provides a REST API for indexing and querying codebases. It is designed to be self-hosted and acts as a "black box" for the CLI client.

## Components
- **HTTP API**: Handles communication with the CLI.
- **Chunker**: Splits source files into semantic chunks using Tree-sitter.
- **Embedder**: Generates vector embeddings using Ollama (`nomic-embed-text`).
- **Vector Store**: Manages storage and retrieval using Qdrant.

## Setup

1. **Prerequisites**:
   - Docker & Docker Compose
   - Ollama installed on the host or accessible via network.

2. **Installation**:
   ```bash
   make setup
   ```

3. **Running**:
   ```bash
   make run
   ```
   Or if you want to run it on a different host:
   ```bash
   HOST=0.0.0.0 make run
   ```

The server will be available at `http://localhost:8080`.
