# Component Specification: FastAPI

## Purpose

Provides the backend API for Phase 1, including search functionality.

## Configuration

- **Host:** 0.0.0.0  
- **Port:** 8000  
- **Debug:** false

## API Endpoints (Phase 1)

- **POST /api/search**:  
  - **Request:** `{ "query": "text query" }`  
  - **Response:**  
    ```json
    {
      "results": [
        {
          "payload": { ... },
          "score": 0.95
        },
        ...
      ]
    }
    ```

## Integration

- Receives requests from React.
- Calls ModernBERT for embeddings and Qdrant for semantic search.
- Returns search results to the frontend.

## Testing

- Test endpoint functionality and error handling.