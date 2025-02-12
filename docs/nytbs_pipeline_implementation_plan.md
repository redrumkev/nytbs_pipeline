# NYT Best-Seller Pipeline Implementation Plan

## Core Components
1. Vector Search & Context
   - Qdrant for vector storage
   - ModernBERT (PyTorch) for embeddings
   - Elasticsearch for keyword search
   - Hybrid search integration

2. Agent Orchestration
   - ControlFlow for workflow management
   - Claude Sonnet 3.5 for orchestration
   - Claude Haiku 3.5 for reviews
   - ModernBERT for local ranking/embedding

3. Storage & Database
   - PostgreSQL (nytbs_db) for structured data
   - Neo4j for relationship tracking
   - Qdrant for vector storage
   - Elasticsearch for text search

4. Frontend & API
   - React dashboard for workflow management
   - FastAPI backend for service orchestration
   - WebSocket for real-time updates
   - Local browser workspace setup

## Implementation Strategy
1. Foundation setup with vector search
2. Hybrid search implementation
3. ControlFlow integration
4. Database and storage layer
5. Frontend and API development
6. Integration and testing

## Core Workflows
- Chapter generation
- Multi-agent review
- Market analysis
- Content refinement

## Development Stack
- VS Code + PowerShell
- Docker + krs_network
- Conda environment (nytbs_pipeline)
- Local ModernBERT deployment via PyTorch
