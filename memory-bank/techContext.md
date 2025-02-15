# Technical Context: NYTBS Pipeline (Phase 1)

## Technologies Used

- **Programming Language:** Python
- **Vector Database:** Qdrant
- **Embedding Model:** ModernBERT (PyTorch)
- **API Framework:** FastAPI
- **Frontend Framework:** React
- **Text Processing:** TextChunker
- **Containerization:** Docker
- **Environment Management:** Conda

## Development Setup

- **IDE:** VS Code
- **Terminal:** PowerShell
- **Conda Environment:** `nytbs_pipeline`
- **CUDA Version:** 12.1
- **Docker Network:** `krs_network`

## Technical Constraints

- **Local Deployment:**  The system must run entirely locally.
- **GPU Utilization:**  ModernBERT must utilize the RTX 4090 GPU.
- **Memory Management:**  Must be efficient to avoid exceeding system memory (128GB).

## Dependencies

- **qdrant-client:** Python client for Qdrant.
- **transformers:**  Hugging Face Transformers library (for ModernBERT).
- **torch:**  PyTorch library.
- **fastapi:**  FastAPI framework.
- **uvicorn:**  ASGI server for FastAPI.
- **react:**  JavaScript library for building user interfaces.