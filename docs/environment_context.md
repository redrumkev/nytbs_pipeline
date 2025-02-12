# Development Environment Context

## Core Environment
- VS Code with PowerShell Terminal
- Conda Environment: nytbs_pipeline
- CUDA Support: 12.1
- Docker Network: krs_network

## Model Configuration
### Primary Models
- Claude Sonnet 3.5 (Orchestrator)
  - Role: Main workflow orchestration
  - API Integration: OpenAI-compatible endpoint
  
- Claude Haiku 3.5 (Reviews)
  - Role: Specialized reviews and critiques
  - API Integration: OpenAI-compatible endpoint

### Local Models
- ModernBERT-large (PyTorch)
  - Purpose: Embedding generation
  - Use: Vector search and re-ranking
  - CUDA Configuration: Optimized for RTX 4090
  - Model Path: src/config/modernbert/

## Infrastructure Configuration
### Docker Containers
1. Qdrant
   - Port: 6333, 6334
   - Network: krs_network
   - Volume: D:\qdrant
   - Auth: [specify settings]

2. PostgreSQL 17
   - Port: 5432
   - Database: nytbs_db
   - Network: krs_network
   - Volume: P:/PostgreSQL_new_data:/var/lib/postgresql/data

3. Neo4j
   - Ports: 7474, 7687
   - Network: krs_network
   - Volume: F:/Neo4j_data:/data

4. Elasticsearch
   - Port: 9200
   - Network: krs_network
   - Volume: F:/Elasticsearch_data:/data

## Storage Configuration
- Document Storage: G:/Data/nytbs_documents
- Vector Store: D:/Docker/qdrant_storage
- Database Volumes: As specified in container configs

## Hardware Utilization
- GPU: RTX 4090 (CUDA 12.1)
- RAM Allocation: 
  - PostgreSQL: 16GB
  - Elasticsearch: 4GB
  - Neo4j: 8GB
  - Qdrant: 8GB
