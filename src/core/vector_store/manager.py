import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333
    ):
        """Initialize Qdrant manager"""
        self.client = QdrantClient(host=host, port=port)
        
    async def list_collections(self) -> List[str]:
        """Get list of collection names"""
        collections = self.client.get_collections()
        return [c.name for c in collections.collections]

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1024
    ) -> bool:
        """Create a new collection with vector support"""
        try:
            self.client.recreate_collection(  # Force recreation of collection
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(  # Ensure Qdrant optimizes the index
                    indexing_threshold=20000
                ),
                shard_number=1  # Ensure sharding is set
            )
            logger.info(f"Collection '{collection_name}' created with vector support.")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            return False
    
    async def delete_collection_points(self, collection_name: str) -> bool:
        """Delete all points in a collection"""
        try:
            self.client.delete_collection(collection_name)
            await self.create_collection(collection_name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete points: {str(e)}")
            return False

    async def store_embeddings(
        self,
        collection_name: str,
        embeddings: torch.Tensor,
        metadata: List[Dict[str, Any]],
        ids: Optional[List[int]] = None
    ) -> bool:
        """Store embeddings with metadata"""
        try:
            # Convert embeddings to numpy
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            # Debug: Print embeddings before storing
            logger.info(f"DEBUG: Embeddings shape: {embeddings.shape}")
            logger.info(f"DEBUG: First embedding sample (first 5 dims): {embeddings[0][:5]}")

            # Create points
            points = []
            for idx, (embedding, meta) in enumerate(zip(embeddings, metadata)):
                if embedding is None:
                    logger.error(f"ERROR: Embedding at index {idx} is None!")
                    continue
                
                point_id = ids[idx] if ids is not None else idx
                points.append(models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=meta
                ))

            # Ensure points are not empty
            if not points:
                logger.error("ERROR: No valid points to store! Check embeddings input.")
                return False

            # Store in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                logger.info(f"DEBUG: Storing batch of {len(batch)} points in Qdrant...")
                logger.info(f"DEBUG: First vector being sent to Qdrant (first 5 dims): {points[0].vector[:5]}")
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )

            return True

        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
            return False

    async def optimize_collection(self, collection_name: str) -> bool:
        """Trigger optimization by updating collection settings"""
        try:
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    max_optimization_threads=2  # Set number of threads for optimization
                )
            )
            logger.info(f"Collection '{collection_name}' optimization triggered.")
            return True
        except Exception as e:
            logger.error(f"Failed to optimize collection: {str(e)}")
            return False

    async def search_similar(
        self,
        collection_name: str,
        query_embedding: torch.Tensor,
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.cpu().numpy().squeeze()
            query_vector = query_embedding.tolist()
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                score_threshold=score_threshold
            )

            return [{
                'id': hit.id,
                'score': hit.score,
                'payload': hit.payload
            } for hit in results]

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            collection_info = self.client.get_collection(collection_name)
            return {
                'vectors_count': collection_info.vectors_count,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {}

    async def retrieve_point(
        self,
        collection_name: str,
        point_id: int
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific point with its vector"""
        try:
            point = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_vectors=True  # Always request vectors
            )
            
            if not point or len(point) == 0:
                logger.error(f"Point with ID {point_id} not found in collection {collection_name}")
                return None
                
            point = point[0]  # Get first point since we only requested one
            
            # Debug logging
            logger.info(f"Retrieved point {point_id}")
            logger.info(f"Vector exists: {point.vector is not None}")
            if point.vector:
                logger.info(f"Vector sample (first 5 dims): {point.vector[:5]}")
            
            return {
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload
            }

        except Exception as e:
            logger.error(f"Failed to retrieve point {point_id}: {str(e)}")
            return None

    async def retrieve_all_points(self, collection_name: str) -> List[Dict[str, Any]]:
        """Retrieve all stored embeddings"""
        try:
            points, next_page = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                limit=100,
                with_vectors=True  # Explicitly request vectors
            )

            if points is None:
                logger.error("No points returned from Qdrant. Check if data was inserted correctly.")
                return []

            return [{
                "id": point.id,
                "vector": point.vector[:5] if point.vector else None,  # Handle potential None
                "payload": point.payload
            } for point in points]

        except Exception as e:
            logger.error(f"Failed to retrieve points: {str(e)}")
            return []
