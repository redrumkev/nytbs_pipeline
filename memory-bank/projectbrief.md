# Project Brief: NYTBS Pipeline (Phase 1)

## Core Goal

Create a frictionless, local writing environment for producing NYT bestselling non-fiction.  Phase 1 focuses on a "walking skeleton" for capturing, embedding, storing, and semantically searching text notes.

## Key Requirements

- **Frictionless Operation:**  The system must be invisible during the writing process.
- **Semantic Search:**  Notes should be retrievable based on meaning, not just keywords.
- **Local Deployment:**  The system must run entirely locally on the specified hardware.
- **Phase 1 Focus:**  Implement only the core functionality: text input, embedding, storage, and search.
- **Future-Proofing:** The design should anticipate future expansion (agent workflows, relationship mapping, etc.) without requiring major refactoring.

## Technology Constraints

- **Primary Language:** Python
- **Vector Database:** Qdrant
- **Embedding Model:** ModernBERT (local, optimized for RTX 4090)
- **Backend:** FastAPI
- **Frontend:** React (minimal for Phase 1)

## Hardware

- RTX 4090 GPU
- 128GB RAM
- 16c/32t CPU
- Fast NVMe SSDs