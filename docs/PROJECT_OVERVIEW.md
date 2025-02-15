  
# Project Overview: NYTBS Pipeline (Phase 1 - Walking Skeleton)

## Core Vision

The NYTBS Pipeline creates a frictionless writing environment that enhances creative flow. Phase 1 establishes a minimal “walking skeleton” to capture, embed, store, and search text notes—technology remains invisible to the user.

## Key Components (Phase 1)

- **Qdrant:** Vector database for storing and searching 1024-dimensional text embeddings.
- **ModernBERT:** Model for generating text embeddings.
- **FastAPI:** Backend API for search (e.g., `/api/search`).
- **React:** Minimal UI for query input and results display.
- **TextChunker:** Processes raw text into semantically meaningful chunks.

## Data Flow (Phase 1)

```Markdown
User Input → React Frontend → FastAPI → ModernBERT (Embeddings) → Qdrant (Store: test_embeddings)

Search Query → React Frontend → FastAPI → ModernBERT → Qdrant (Search) → Results → FastAPI → React Frontend
```
      
## Technology Stack (Phase 1)

- **Vector Database:** Qdrant
- **Embedding Model:** ModernBERT
- **Backend Framework:** FastAPI (Python)
- **Frontend Framework:** React (JavaScript)
- **Text Processing:** TextChunker (Python)
- **Environment:** Docker, Conda

## Hardware & Environment

- **GPU:** RTX 4090
- **RAM:** 128GB
- **CPU:** 16c/32t
- **Docker Network:** `krs_network`
- **CUDA:** 12.1

## Roadmap

- **Phase 1:** Walking Skeleton – Capture, embed, store, search text notes.
- **Phase 2:** Core Writing Workflow – Project/chapter structure, PostgreSQL.
- **Phase 3:** Enhanced Functionality – Elasticsearch, Neo4j, basic ControlFlow.
- **Phase 4:** Intelligence & Automation – Advanced ControlFlow, LLM integration.

## Key Principles

- **Frictionless Operation:** Background processing with simple interfaces.
- **Invisible Complexity:** Modular components that are easy to use.
- **Start Simple:** Focus on core functionality before scaling.