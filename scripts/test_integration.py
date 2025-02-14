import sys
from pathlib import Path
import asyncio
import logging
from typing import List, Dict, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.embeddings.modernbert_processor import ModernBERTProcessor
from src.core.vector_store.manager import QdrantManager
from src.core.embeddings.text_chunker import TextChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def ensure_collection_exists(manager: QdrantManager, collection_name: str) -> bool:
    """Ensure collection exists and is ready"""
    try:
        # Check if collection exists
        collections = await manager.list_collections()
        
        if collection_name not in collections:
            logger.info(f"Creating collection: {collection_name}")
            created = await manager.create_collection(
                collection_name=collection_name,
                vector_size=1024
            )
            if not created:
                raise RuntimeError(f"Failed to create collection {collection_name}")
        else:
            logger.info(f"Collection {collection_name} already exists")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to ensure collection: {str(e)}")
        return False

async def run_integration_test():
    """Test complete pipeline from text to vector search"""
    try:
        # Initialize components
        model_dir = project_root / "src" / "config" / "modernbert"
        processor = ModernBERTProcessor(model_dir)
        chunker = TextChunker(model_dir)
        manager = QdrantManager()

        # Test data
        test_texts = [
            "This is a test document about machine learning and AI. " * 10,
            "Another document discussing vector databases and search. " * 10,
            "A third document about natural language processing. " * 10
        ]

        # Ensure collection exists
        collection_name = "test_embeddings"
        if not await ensure_collection_exists(manager, collection_name):
            raise RuntimeError("Failed to ensure collection exists")

        logger.info("1. Processing text into chunks...")
        all_chunks = []
        for i, text in enumerate(test_texts):
            chunks = chunker.create_chunks(text, f"test_doc_{i}.txt")
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks")

        # Generate embeddings
        logger.info("2. Generating embeddings...")
        texts = [chunk.text for chunk in all_chunks]
        embeddings = await processor.process_batch(texts, show_progress=True)
        
        # Create metadata (fixed to match ChunkMetadata fields)
        metadata = [{
            'text': chunk.text,
            'doc_id': f"test_doc_{i}",
            'chunk_index': chunk.metadata.chunk_index,
            'total_chunks': chunk.metadata.total_chunks,
            'token_count': chunk.metadata.token_count,
            'is_complete_note': chunk.metadata.is_complete_note,
            'original_file': chunk.metadata.original_file
        } for i, chunk in enumerate(all_chunks)]

        # Store in Qdrant
        logger.info("3. Storing in Qdrant...")
        
        # Clear existing points
        logger.info("Clearing existing points...")
        await manager.delete_collection_points(collection_name)
        
        # Store embeddings
        logger.info("Storing new embeddings...")
        ids = list(range(len(embeddings)))  # Assign IDs 0,1,2...
        success = await manager.store_embeddings(
            collection_name=collection_name,
            embeddings=embeddings,
            metadata=metadata,
            ids=ids  # Ensure IDs match retrieval expectations
        )
        
        print(json.dumps({
            "collection_name": collection_name,
            "embeddings": [emb.tolist() for emb in embeddings[:3]],  # Convert tensor to list
            "metadata": metadata[:3]
        }, indent=2))


        if not success:
            raise RuntimeError("Failed to store embeddings")

        # Manually retrieve a stored point to check if vector exists
        point = manager.client.retrieve(
            collection_name="test_embeddings",
            ids=[0]  # Retrieve the point with ID 0
        )
        logger.info(f"\nRetrieved Point from Qdrant: {point}")


        # Force optimization after inserting
        logger.info("Optimizing collection to ensure vectors are indexed...")
        await manager.optimize_collection(collection_name)

        # Test retrieval
        logger.info("4. Testing retrieval...")
        query_text = "Tell me about machine learning"
        query_embedding = await processor.process_realtime(query_text)
        
        results = await manager.search_similar(
            collection_name=collection_name,
            query_embedding=query_embedding,
            limit=2
        )

        logger.info("\nSearch Results:")
        for i, result in enumerate(results):
            logger.info(f"\nResult {i+1}:")
            logger.info(f"Score: {result['score']:.3f}")
            logger.info(f"Text: {result['payload']['text'][:100]}...")

        logger.info("\nRetrieving stored embeddings...")
        stored_points = await manager.retrieve_all_points(collection_name)
    
        for i, point in enumerate(stored_points):
            logger.info(f"\nPoint {i+1}:")
            logger.info(f"ID: {point['id']}")
            logger.info(f"Vector (first 5 dims): {point['vector']}")
            logger.info(f"Metadata: {point['payload']}")

        # Get collection stats
        stats = await manager.get_collection_stats(collection_name)
        logger.info(f"\nCollection Stats:")
        logger.info(f"Total vectors: {stats.get('vectors_count', 0)}")
        logger.info(f"Total points: {stats.get('points_count', 0)}")
        logger.info(f"Segments: {stats.get('segments_count', 0)}")

        return True

    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = asyncio.run(run_integration_test())
    sys.exit(0 if success else 1)