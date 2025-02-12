# src/core/vector_store/manager.py

import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        config_path: Optional[str] = None
    ):
        """Initialize Qdrant manager with configuration"""
        self.client = QdrantClient(host=host, port=port)
        self.config_path = config_path or Path("src/config/qdrant/collections.yaml")
        self._load_config()

    def _load_config(self):
        """Load collection configuration from YAML"""
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    def _create_collection(self, name: str, settings: Dict[str, Any]) -> None:
        """Create a collection with specified settings"""
        # Basic vector configuration
        vectors_config = models.VectorParams(
            size=settings['vectors']['size'],
            distance=models.Distance.COSINE
        )
        
        # Optimizer configuration
        optimizers_config = models.OptimizersConfigDiff(
            **settings['optimizers_config']
        )
        
        # Collection creation parameters
        create_params = {
            "collection_name": name,
            "vectors_config": vectors_config,
            "optimizers_config": optimizers_config
        }
        
        # Add HNSW config if present
        if 'hnsw_config' in settings:
            create_params["hnsw_config"] = models.HnswConfigDiff(
                **settings['hnsw_config']
            )
        
        # Add quantization config if present
        if 'quantization_config' in settings:
            create_params["quantization_config"] = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=settings['quantization_config']['scalar_config']['quantile'],
                    always_ram=settings['quantization_config']['scalar_config']['always_ram']
                )
            )
        
        # Create the collection
        self.client.create_collection(**create_params)
        
        # Create payload indexes
        for field_name, schema in settings['payload_schema'].items():
            self.client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=schema['type']
            )

    def initialize_collections(self) -> None:
        """Initialize collections from config"""
        try:
            collections = self.client.get_collections()
            existing = {c.name for c in collections.collections}

            for name, settings in self.config['collections'].items():
                if name not in existing:
                    logger.info(f"Creating collection: {name}")
                    self._create_collection(name, settings)
                else:
                    logger.info(f"Collection {name} already exists")

        except Exception as e:
            logger.error(f"Failed to initialize collections: {str(e)}")
            raise

    def store_embeddings(
        self,
        collection_name: str,
        embeddings: torch.Tensor,
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """Store embeddings with metadata"""
        try:
            # Convert embeddings to numpy and ensure correct shape
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            points = []
            for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
                # Add timestamp to metadata
                meta['last_modified'] = datetime.utcnow().isoformat()
                
                # Generate a UUID for the point
                point_id = str(uuid.uuid4())
                
                # Ensure embedding is 1D
                if len(embedding.shape) > 1:
                    embedding = embedding.flatten()
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=meta
                ))

            # Store in batches of 100
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )

            return True

        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
            return False

    def search_similar(
        self,
        collection_name: str,
        query_embedding: torch.Tensor,
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors with filtering"""
        try:
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.cpu().numpy()

            # Build search parameters
            search_params = models.SearchParams(
                hnsw_ef=128,
                exact=False
            )

            # Convert filter conditions if provided
            filter_model = None
            if filter_conditions:
                conditions = []
                for field, value in filter_conditions.items():
                    conditions.append(
                        models.FieldCondition(
                            key=field,
                            match=models.MatchValue(value=value)
                        )
                    )
                filter_model = models.Filter(
                    must=conditions
                )

            # Ensure vector is properly formatted
            if len(query_embedding.shape) == 2:
                query_vector = query_embedding[0].flatten().tolist()  # Take first vector if batched
            else:
                query_vector = query_embedding.flatten().tolist()

            # Perform search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                search_params=search_params,
                query_filter=filter_model,
                with_payload=True,
                with_vectors=True  # Request vectors in response
            )

            # Format results
            return [{
                'id': hit.id,
                'score': hit.score,
                'payload': hit.payload,
                'vector': np.array(hit.vector) if hasattr(hit, 'vector') and hit.vector else None
            } for hit in results]

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def delete_by_metadata(
        self,
        collection_name: str,
        metadata_filter: Dict[str, Any]
    ) -> bool:
        """Delete points matching metadata filter"""
        try:
            # Convert metadata filter to Qdrant filter conditions
            conditions = []
            for field, value in metadata_filter.items():
                conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=value)
                    )
                )
            
            filter_model = models.Filter(
                must=conditions
            )
            
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=filter_model
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete points: {str(e)}")
            return False

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics"""
        try:
            collections = self.client.get_collections()
            stats = {
                'collections': {}
            }
            
            for collection in collections.collections:
                collection_info = self.client.get_collection(collection.name)
                stats['collections'][collection.name] = {
                    'name': collection.name,
                    'vectors_count': collection_info.vectors_count,
                    'points_count': collection_info.points_count,
                    'segments_count': collection_info.segments_count,
                    'status': collection_info.status
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get cluster stats: {str(e)}")
            return {'error': str(e)}

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            self.client.close()
        except Exception as e:
            logger.error(f"Failed to cleanup: {str(e)}")
