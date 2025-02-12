# Writing Pipeline Architecture

## Core Components
1. Vector Search (Qdrant + ModernBERT)
   - Embedding generation
   - Semantic search
   - Re-ranking

2. Text Search (Elasticsearch)
   - Keyword-based retrieval
   - Publishing metadata

3. Agent Orchestration (ControlFlow)
   - Writer agents
   - Review agents
   - Publishing analysis

4. Storage Layer
   - PostgreSQL: Drafts, versions, prompts
   - Neo4j: Theme relationships
   - Qdrant: Semantic vectors
   - Elasticsearch: Text search

5. Frontend
   - React dashboard
   - FastAPI backend
   - WebSocket updates

## Standard Workflows
1. Chapter Generation
2. Multi-Agent Review
3. Market Analysis
4. Content Refinement