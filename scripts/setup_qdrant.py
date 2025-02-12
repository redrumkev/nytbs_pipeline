# scripts/setup_qdrant.py

import sys
from pathlib import Path
import logging
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.vector_store.manager import QdrantManager
from src.core.embeddings.modernbert_processor import ModernBERTProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_vector_store(manager: QdrantManager, processor: ModernBERTProcessor) -> bool:
    """Test vector store functionality"""
    try:
        # Test texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs."
        ]
        
        # Generate embeddings
        embeddings = await processor.process_batch(texts, show_progress=True)
        
        # Create metadata
        metadata = [
            {'content_type': 'test', 'project_id': 'test_project', 'section': i}
            for i in range(len(texts))
        ]
        
        # Store embeddings
        success = manager.store_embeddings(
            collection_name='writing_vectors',
            embeddings=embeddings,
            metadata=metadata
        )
        if not success:
            return False
            
        # Test search
        query_embedding = await processor.process_realtime(texts[0])
        results = manager.search_similar(
            collection_name='writing_vectors',
            query_embedding=query_embedding,
            limit=2
        )
        
        # Verify results
        return len(results) > 0

    except Exception as e:
        logger.error(f"Vector store test failed: {str(e)}")
        return False

async def setup_qdrant() -> bool:
    """Setup and test Qdrant integration"""
    try:
        # Initialize Qdrant manager
        manager = QdrantManager()
        
        # Initialize collections
        manager.initialize_collections()
        
        # Get cluster stats
        stats = manager.get_cluster_stats()
        logger.info(f"Cluster stats: {stats}")
        
        # Initialize ModernBERT processor
        model_dir = project_root / "src" / "config" / "modernbert"
        processor = ModernBERTProcessor(model_dir)
        
        # Test vector store functionality
        test_success = await test_vector_store(manager, processor)
        if not test_success:
            return False
            
        logger.info("Qdrant setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(setup_qdrant())
    sys.exit(0 if success else 1)
