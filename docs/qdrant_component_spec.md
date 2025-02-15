# Component Specification: Qdrant

## Purpose

Provides vector storage and semantic search for text embeddings.

## Configuration

- **Docker:**  
  - Image: `qdrant/qdrant`  
  - Ports: 6333 (HTTP)  
  - Network: `krs_network`  
  - Volume: e.g., `/path/to/qdrant_storage:/qdrant/storage`
- **Collection:** `test_embeddings`
- **Vector Size:** 1024  
- **Distance:** Cosine  
- **Storage:** On-disk

## API (via `vector_store/manager.py`)

- `create_collection(collection_name, vector_size)`: Creates/recreates the collection.
- `store_embeddings(embeddings, metadata)`: Stores embeddings with minimal metadata.
- `search_similar(query_embedding, limit, score_threshold)`: Performs semantic search.
- `get_collection_stats()`: Retrieves collection statistics.

## Data Schema (for `test_embeddings`)

- **vector:** 1024-dimensional float array.
- **payload:** Minimal metadata:
  - `file_name`: Filename (string)
  - `text`: A text snippet (string)
  - `chunk_index`: Chunk index (integer)
  - `total_chunks`: Total chunks (integer)
  - `is_complete_note`: Boolean flag

## Integration

- Receives embeddings from ModernBERT (via FastAPI).
- Returns search results with scores and metadata to FastAPI.

## Testing

- Connection and collection creation tests.
- Embedding storage and retrieval tests.
- Semantic search functionality.