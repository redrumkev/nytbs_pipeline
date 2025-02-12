# Configuration Templates

## Qdrant Collection Configuration
```yaml
name: writing_vectors
vectors:
  size: 1024
  distance: Cosine
  on_disk: true
optimizers_config:
  default_segment_number: 2
  memmap_threshold: 20000
payload_index:
  text_index:
    type: text
    tokenizer: word
    min_token_len: 2
    max_token_len: 15
    lowercase: true
```

## Elasticsearch Index Template
```json
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "analysis": {
      "analyzer": {
        "writing_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": [
            "lowercase",
            "stop",
            "porter_stem"
          ]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "content": {
        "type": "text",
        "analyzer": "writing_analyzer"
      },
      "metadata": {
        "type": "object"
      }
    }
  }
}
```

## Neo4j Initial Schema
```cypher
// Book Structure
CREATE CONSTRAINT book_id IF NOT EXISTS ON (b:Book) ASSERT b.id IS UNIQUE;
CREATE CONSTRAINT chapter_id IF NOT EXISTS ON (c:Chapter) ASSERT c.id IS UNIQUE;

// Theme Relationships
CREATE INDEX theme_name IF NOT EXISTS FOR (t:Theme) ON (t.name);
```

## FastAPI Base Configuration
```python
app_config = {
    "title": "NYTBS Pipeline API",
    "description": "Writing Pipeline Management API",
    "version": "1.0.0",
    "docs_url": "/api/docs",
    "redoc_url": "/api/redoc",
    "openapi_url": "/api/openapi.json"
}

cors_config = {
    "allow_origins": ["http://localhost:3000"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"]
}
```