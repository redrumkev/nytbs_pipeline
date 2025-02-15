      
# System Patterns: NYTBS Pipeline (Phase 1)

## System Architecture

The Phase 1 system follows a simple, layered architecture:

1.  **Presentation Layer (React):** Minimal UI for user interaction.
2.  **API Layer (FastAPI):**  Provides a single `/api/search` endpoint.
3.  **Embedding Layer (ModernBERT):**  Generates vector embeddings.
4.  **Storage Layer (Qdrant):**  Stores and searches embeddings.
5.  **Text Processing Layer (TextChunker):** Processes text into appropriately sized chunks.

## Key Technical Decisions

- **Vector Database (Qdrant):** Chosen for its speed, efficiency, and support for semantic search.
- **Embedding Model (ModernBERT):**  Chosen for its performance and ability to run locally on the RTX 4090.
- **API Framework (FastAPI):** Chosen for its speed, ease of use, and automatic data validation.
- **Component-Based Design:**  Each component (Qdrant, ModernBERT, FastAPI, React) is designed to be independent and reusable.

## Component Relationships

React Frontend <--> FastAPI <--> ModernBERT <--> Qdrant
^
|
TextChunker
      
## Design Patterns

- **Single Responsibility Principle:** Each component has a single, well-defined responsibility.
- **Dependency Inversion:**  High-level modules (e.g., FastAPI) do not depend on low-level modules (e.g., Qdrant).  Both depend on abstractions (e.g., the `QdrantManager` class).