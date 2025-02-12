# Configuration Reference

## Vector Store (Qdrant)

### Collection Configuration
```yaml
collections:
  writing_vectors:
    vectors:
      size: 1024  # Matches ModernBERT output dimension
      distance: Cosine  # Using normalized vectors
      on_disk: true  # Persistence for large datasets
    
    optimizers_config:
      default_segment_number: 2  # Balance between update speed and search performance
      indexing_threshold: 50000  # Large enough for batch operations
      memmap_threshold: 20000    # Efficient memory mapping
      max_segment_size: 100000   # Prevent oversized segments
    
    quantization_config:
      scalar_config:
        type: "int8"
        quantile: 0.99
        always_ram: true
    
    hnsw_config:
      m: 16  # Better for high-dimensional vectors
      ef_construct: 100
      full_scan_threshold: 10000
    
    payload_schema:
      timestamp:
        type: datetime
      content_type:
        type: keyword
      project_id:
        type: keyword
      chapter_id:
        type: keyword
      section_id:
        type: keyword
      flow_state:
        type: float
      priority:
        type: integer
```

### Environment Variables
```bash
# Qdrant Connection
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334

# Storage Configuration
QDRANT_STORAGE_PATH=D:/Docker/qdrant_storage
```

## ModernBERT Configuration

### Processing Settings
```python
class ProcessingConfig:
    batch_size: int = 32
    max_length: int = 8192
    use_fp16: bool = True  # Optimized for RTX 4090
    device: str = "cuda"
    cache_size: int = 1000
    priority_levels: int = 3
```

### Model Configuration
```json
{
  "model_path": "src/config/modernbert",
  "tokenizer_config": {
    "model_max_length": 8192,
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "cls_token": "[CLS]"
  }
}
```

## FastAPI Server

### Base Configuration
```python
app_config = {
    "title": "NYTBS Pipeline API",
    "description": "Writing Pipeline Management API",
    "version": "1.0.0",
    "docs_url": "/api/docs",
    "redoc_url": "/api/redoc",
    "openapi_url": "/api/openapi.json"
}
```

### CORS Settings
```python
cors_config = {
    "allow_origins": ["http://localhost:3000"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"]
}
```

### Environment Variables
```bash
# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG_MODE=false

# Security
JWT_SECRET_KEY=your_secure_secret_key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

## Hardware Optimization

### CUDA Settings
```python
# GPU Memory Management
torch.cuda.set_per_process_memory_fraction(0.8)  # Reserve 20% for system
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

### System Requirements
- NVIDIA RTX 4090 or equivalent
- CUDA 12.1+
- 128GB RAM minimum
- 16 core / 32 thread CPU
- NVMe SSD for storage

## Logging Configuration
```python
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "logs/nytbs_pipeline.log",
            "formatter": "default",
            "level": "DEBUG"
        }
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO"
    }
}
```

## Docker Network Configuration
```yaml
networks:
  krs_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## Volume Mappings
```yaml
volumes:
  qdrant_storage:
    driver: local
    driver_opts:
      type: none
      device: D:/Docker/qdrant_storage
      o: bind
