# Component Specification: ModernBERT

## Purpose

Generates 1024-dimensional vector embeddings from text.

## Configuration

- **Model Directory:** `src/config/modernbert/`
- **FP16:** true
- **Device:** "cuda"
- **Batch Size:** 32  
- **Max Length:** 8192 tokens

## API (via `modernbert_processor.py`)

- `process_batch(texts)`: Generates embeddings for a batch of texts.
- `process_realtime(text)`: Generates an embedding for a single text.

## Integration

- Receives text from FastAPI.
- Provides embeddings to FastAPI for storage in Qdrant.

## Testing

- Verify CUDA setup and performance.
- Test embedding generation for varying text lengths.
- Validate caching and memory efficiency.