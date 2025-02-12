# Integration Guides

## ModernBERT Setup

### Prerequisites
- CUDA 12.1 or higher
- PyTorch with CUDA support
- 24GB+ GPU VRAM
- Python 3.9+

### Installation Steps
1. Clone the model files:
```bash
# From project root
mkdir -p src/config/modernbert
cp /path/to/modernbert/* src/config/modernbert/
```

2. Install dependencies:
```bash
conda env create -f environment_setup/nytbs_pipeline.yml
conda activate nytbs_pipeline
```

3. Verify CUDA setup:
```bash
python scripts/test_cuda.py
```

### Integration Example
```python
from src.core.embeddings.modernbert_processor import ModernBERTProcessor

# Initialize processor
model_dir = "src/config/modernbert"
processor = ModernBERTProcessor(model_dir)

# Process text
async def process_content(texts):
    # Generate embeddings
    embeddings = await processor.process_batch(
        texts,
        batch_size=32,
        show_progress=True
    )
    
    # Real-time processing
    single_embedding = await processor.process_realtime(
        texts[0],
        priority=0  # Higher priority for real-time
    )
    
    return embeddings, single_embedding
```

## Qdrant Vector Store

### Docker Setup
1. Create network:
```bash
docker network create krs_network
```

2. Create storage directory:
```bash
mkdir -p D:/Docker/qdrant_storage
```

3. Launch container:
```bash
docker run -d \
  --name qdrant \
  --network krs_network \
  -p 6333:6333 \
  -p 6334:6334 \
  -v D:/Docker/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### Collection Setup
```python
from src.core.vector_store.manager import QdrantManager

# Initialize manager
manager = QdrantManager()

# Initialize collections from config
manager.initialize_collections()

# Verify setup
stats = manager.get_cluster_stats()
print(f"Cluster stats: {stats}")
```

### Integration Example
```python
async def store_and_search():
    # Store embeddings
    success = manager.store_embeddings(
        collection_name='writing_vectors',
        embeddings=embeddings,  # torch.Tensor from ModernBERT
        metadata=[{
            'content_type': 'chapter',
            'project_id': 'book_123',
            'chapter_id': 'ch_1',
            'flow_state': 0.85
        }]
    )
    
    # Search similar content
    results = manager.search_similar(
        collection_name='writing_vectors',
        query_embedding=query_vector,
        limit=10,
        score_threshold=0.7
    )
    
    return results
```

## FastAPI Server Integration

### Server Setup
1. Create server instance:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(**app_config)

# Add CORS middleware
app.add_middleware(CORSMiddleware, **cors_config)
```

2. Add routes:
```python
from src.core.vector_store.manager import QdrantManager
from src.core.embeddings.modernbert_processor import ModernBERTProcessor

# Initialize components
qdrant = QdrantManager()
processor = ModernBERTProcessor("src/config/modernbert")

@app.post("/api/search")
async def semantic_search(query: str):
    # Generate query embedding
    embedding = await processor.process_realtime(query)
    
    # Search vector store
    results = qdrant.search_similar(
        collection_name='writing_vectors',
        query_embedding=embedding
    )
    
    return results
```

### WebSocket Integration
```python
from fastapi import WebSocket

@app.websocket("/ws/workflow/{workflow_id}")
async def workflow_socket(websocket: WebSocket):
    await websocket.accept()
    
    # Authenticate
    auth = await websocket.receive_json()
    if not verify_token(auth['token']):
        await websocket.close(code=4001)
        return
    
    try:
        while True:
            # Send updates
            await websocket.send_json({
                "type": "workflow.progress",
                "percentage": progress
            })
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected")
```

## Error Handling

### Exception Hierarchy
```python
class PipelineException(Exception):
    """Base exception for pipeline errors"""
    pass

class VectorStoreError(PipelineException):
    """Vector store related errors"""
    pass

class ProcessingError(PipelineException):
    """Embedding processing errors"""
    pass
```

### Error Handling Example
```python
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def safe_process(text: str):
    try:
        embedding = await processor.process_realtime(text)
        return embedding
    except ProcessingError as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise PipelineException(f"Pipeline error: {str(e)}")
```

## Monitoring Integration

### Logging Setup
```python
import logging.config
from src.utils.logger import logging_config

# Configure logging
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)
```

### Performance Monitoring
```python
from src.utils.monitoring import track_performance

@track_performance
async def monitored_operation():
    # Operation code here
    pass
```

## Testing Integration

### Test Configuration
```python
# tests/conftest.py
import pytest
from src.core.vector_store.manager import QdrantManager
from src.core.embeddings.modernbert_processor import ModernBERTProcessor

@pytest.fixture
async def qdrant():
    manager = QdrantManager()
    yield manager
    await manager.cleanup()

@pytest.fixture
def processor():
    return ModernBERTProcessor("src/config/modernbert")
```

### Test Examples
```python
# tests/test_pipeline.py
import pytest

async def test_embedding_generation(processor):
    text = "Test content"
    embedding = await processor.process_realtime(text)
    assert embedding.shape == (1, 1024)
    assert torch.isfinite(embedding).all()

async def test_vector_search(qdrant, processor):
    # Test search functionality
    query = "Test query"
    embedding = await processor.process_realtime(query)
    results = qdrant.search_similar(
        collection_name='writing_vectors',
        query_embedding=embedding
    )
    assert len(results) >= 0
