import sys
from pathlib import Path
import logging
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.vector_store.manager import QdrantManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_collections() -> bool:
    """Setup and verify Qdrant collections"""
    try:
        # Initialize manager
        manager = QdrantManager()
        
        # Initialize collections
        logger.info("Initializing collections...")
        manager.initialize_collections()
        
        # Get cluster stats
        stats = manager.get_cluster_stats()
        
        # Print collection information
        logger.info("\nCollection Status:")
        for name, info in stats['collections'].items():
            logger.info(f"\nCollection: {name}")
            logger.info(f"Status: {info['status']}")
            logger.info(f"Vectors: {info['vectors_count']}")
            logger.info(f"Points: {info['points_count']}")
            logger.info(f"Segments: {info['segments_count']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        return False

async def verify_collections() -> bool:
    """Verify collections are properly configured"""
    try:
        manager = QdrantManager()
        
        # Test metadata handling
        test_metadata = {
            "file_path": "test/sample.md",
            "last_modified": "2025-02-08T22:00:00",
            "chunk_index": 0,
            "total_chunks": 1,
            "is_complete": True
        }
        
        # Try to delete any existing test data
        logger.info("\nCleaning up test data...")
        manager.delete_by_metadata(
            collection_name="test_embeddings",
            metadata_filter={"file_path": "test/sample.md"}
        )
        
        # Create a test embedding
        import torch
        test_embedding = torch.randn(1, 1024)  # Match vector size
        
        # Store test data
        logger.info("Testing storage...")
        success = manager.store_embeddings(
            collection_name="test_embeddings",
            embeddings=test_embedding,
            metadata=[test_metadata]
        )
        
        if not success:
            raise Exception("Failed to store test embedding")
        
        # Test retrieval
        logger.info("Testing retrieval...")
        results = manager.search_similar(
            collection_name="test_embeddings",
            query_embedding=test_embedding,
            limit=1
        )
        
        if not results:
            raise Exception("Failed to retrieve test embedding")
            
        # Verify metadata
        stored_metadata = results[0]['payload']
        for key in test_metadata:
            if key not in stored_metadata:
                raise Exception(f"Missing metadata field: {key}")
        
        logger.info("\nCollection verification successful!")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False

async def main():
    """Setup and verify collections"""
    try:
        # Setup collections
        logger.info("Setting up collections...")
        setup_success = await setup_collections()
        if not setup_success:
            raise Exception("Collection setup failed")
            
        # Verify collections
        logger.info("\nVerifying collections...")
        verify_success = await verify_collections()
        if not verify_success:
            raise Exception("Collection verification failed")
            
        logger.info("\nCollection setup and verification completed successfully!")
        
    except Exception as e:
        logger.error(f"Setup and verification failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
