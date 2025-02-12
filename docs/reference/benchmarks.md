# Performance Benchmarks

## Hardware Configuration
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- CPU: 16 cores / 32 threads
- RAM: 128GB DDR5
- Storage: NVMe SSD

## Vector Search Performance

### Qdrant Query Latency
| Vector Count | Batch Size | Average Latency | p95 Latency | p99 Latency |
|--------------|------------|-----------------|-------------|-------------|
| 10,000       | 1          | 5ms            | 7ms         | 9ms         |
| 100,000      | 1          | 8ms            | 12ms        | 15ms        |
| 1,000,000    | 1          | 15ms           | 22ms        | 28ms        |
| 10,000       | 32         | 42ms           | 55ms        | 65ms        |
| 100,000      | 32         | 68ms           | 85ms        | 98ms        |
| 1,000,000    | 32         | 125ms          | 155ms       | 180ms       |

### Search Configuration
```yaml
hnsw_config:
  m: 16
  ef_construct: 100
  full_scan_threshold: 10000

quantization:
  type: int8
  always_ram: true
```

## ModernBERT Processing

### Embedding Generation
| Batch Size | Sequence Length | Memory Usage | Throughput (texts/sec) | GPU Utilization |
|------------|----------------|--------------|----------------------|-----------------|
| 1          | 512           | 0.74 GB      | 45                   | 15%             |
| 8          | 512           | 1.2 GB       | 280                  | 45%             |
| 32         | 512           | 2.8 GB       | 850                  | 85%             |
| 1          | 2048          | 1.1 GB       | 12                   | 20%             |
| 8          | 2048          | 2.4 GB       | 75                   | 60%             |
| 32         | 2048          | 5.6 GB       | 220                  | 90%             |

### FP16 vs FP32 Comparison
| Precision | Memory Usage | Relative Speed | Quality Impact |
|-----------|--------------|----------------|----------------|
| FP32      | 1.48 GB     | 1.0x           | Baseline       |
| FP16      | 0.74 GB     | 1.8x           | Negligible     |

## Memory Usage Patterns

### Peak Memory Usage
| Component           | GPU Memory | System RAM |
|--------------------|------------|------------|
| ModernBERT (idle)  | 0.74 GB    | 1.2 GB     |
| Vector Search      | 1.5 GB     | 4.5 GB     |
| Batch Processing   | 5.6 GB     | 8.2 GB     |
| Full Pipeline      | 8.2 GB     | 15.4 GB    |

### Memory Optimization Settings
```python
# GPU Memory Management
torch.cuda.set_per_process_memory_fraction(0.8)  # Reserve 20% for system
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Qdrant Memory Settings
memmap_threshold: 20000
indexing_threshold: 50000
```

## Disk I/O Performance

### Vector Storage
| Operation    | Data Size | Average Speed | IOPS    |
|--------------|-----------|---------------|---------|
| Vector Write | 1GB       | 850 MB/s      | 12,000  |
| Vector Read  | 1GB       | 2.2 GB/s      | 25,000  |
| Index Update | 1GB       | 450 MB/s      | 8,000   |

### Configuration
```yaml
storage:
  type: nvme
  path: D:/Docker/qdrant_storage
  on_disk: true
```

## Network Performance

### API Latency
| Endpoint          | Average | p95    | p99    |
|-------------------|---------|--------|---------|
| /api/projects     | 12ms    | 18ms   | 25ms    |
| /api/search       | 25ms    | 35ms   | 45ms    |
| WebSocket Update  | 5ms     | 8ms    | 12ms    |

### WebSocket Performance
- Maximum concurrent connections: 10,000
- Message throughput: 50,000 msg/sec
- Average message size: 2KB

## Scaling Characteristics

### Linear Scaling (up to)
- Vector count: 10M
- Concurrent users: 1,000
- Batch size: 32
- Sequence length: 8,192

### Resource Utilization at Scale
| Scale Level | GPU Memory | System RAM | CPU Usage |
|-------------|------------|------------|-----------|
| Light       | 2 GB       | 16 GB      | 20%       |
| Medium      | 8 GB       | 32 GB      | 45%       |
| Heavy       | 16 GB      | 64 GB      | 80%       |

## Optimization Recommendations

### For Best Performance
1. Use FP16 precision for ModernBERT
2. Maintain batch sizes between 16-32
3. Keep vector count per collection under 5M
4. Enable quantization for large collections
5. Use SSD/NVMe storage for vector data
6. Maintain GPU memory usage under 80%
7. Enable CUDA graph optimization for repeated operations

### Memory/Speed Trade-offs
1. Reduce batch size if memory constrained
2. Adjust HNSW parameters for search speed vs accuracy
3. Use quantization for large datasets
4. Cache frequently accessed vectors in RAM
5. Balance between on-disk and in-memory storage
