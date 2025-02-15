import sys 
from pathlib import Path
import logging
import asyncio
from typing import List, Dict, Any, Optional
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.embeddings.modernbert_processor import ModernBERTProcessor
from src.core.embeddings.text_chunker import TextChunker
from src.core.vector_store.manager import QdrantManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MixtureValidator:
    def __init__(self):
        # List of directories to process
        self.test_dirs = [
            project_root / "test_data" / "generated_content",
            project_root / "test_data" / "writing_samples"
        ]
        self.model_dir = project_root / "src" / "config" / "modernbert"
        
        # Initialize components
        self.processor = ModernBERTProcessor(self.model_dir)
        self.chunker = TextChunker(self.model_dir)
        # Set chunking parameters after initialization
        self.chunker.chunk_size = 512
        self.chunker.overlap_size = 128
        
        # Initialize Qdrant with default settings
        self.qdrant = QdrantManager()
        
        # Use the "test_embeddings" collection
        self.collection_name = "test_embeddings"

    async def process_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single file with robust error handling"""
        try:
            logger.info(f"Processing {file_path.name}")
            
            # Process chunks
            chunks = self.chunker.process_file(file_path)
            if not chunks:
                logger.warning(f"No valid chunks generated for {file_path.name}")
                return None
            
            # Get embeddings
            texts = [chunk.text for chunk in chunks]
            try:
                embeddings = await self.processor.process_batch(texts)
            except Exception as e:
                logger.error(f"Embedding generation failed for {file_path.name}: {str(e)}")
                return None
            
            # Create metadata
            metadata = [
                {
                    "text": chunk.text,
                    "file_name": file_path.name,
                    "is_complete_note": chunk.metadata.is_complete_note,
                    "chunk_index": chunk.metadata.chunk_index,
                    "total_chunks": chunk.metadata.total_chunks
                }
                for chunk in chunks
            ]
            
            # Store in Qdrant
            success = await self.qdrant.store_embeddings(
                collection_name=self.collection_name,
                embeddings=embeddings,
                metadata=metadata
            )
            
            return {
                "file_name": file_path.name,
                "num_chunks": len(chunks),
                "success": success
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None

    async def test_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Test semantic search with improved filtering"""
        try:
            # Get query embedding
            query_embedding = await self.processor.process_realtime(query)
            
            # Search with adjusted parameters
            results = await self.qdrant.search_similar(
                collection_name=self.collection_name,
                query_embedding=query_embedding,
                limit=limit * 3,  # Get more results for better filtering
                score_threshold=0.80,  # Stricter threshold
                with_payload=True
            )
            
            # Advanced result filtering
            filtered_results = []
            seen_content = set()
            seen_sections = set()
            
            for result in results:
                text = result['payload']['text']
                file_name = result['payload']['file_name']
                chunk_index = result['payload'].get('chunk_index', 0)
                
                # Generate content signature
                content_sig = f"{file_name}:{chunk_index}"
                section_sig = text[:50]  # First 50 chars as section identifier
                
                # Skip if we've seen similar content
                if content_sig in seen_content or section_sig in seen_sections:
                    continue
                    
                # Skip fragments and incomplete content
                if len(text.strip()) < 50 or text.endswith('...'):
                    continue
                    
                seen_content.add(content_sig)
                seen_sections.add(section_sig)
                filtered_results.append(result)
                
                if len(filtered_results) >= limit:
                    break
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get detailed collection information"""
        try:
            # Get collection stats
            stats = await self.qdrant.get_collection_stats(self.collection_name)
            
            # Get sample points for validation
            points = await self.qdrant.retrieve_all_points(self.collection_name)
            
            return {
                "vectors_count": stats.get('vectors_count', 0),
                "points_count": stats.get('points_count', 0),
                "segments_count": stats.get('segments_count', 0),
                "sample_count": len(points),
                "has_vectors": any(p.get('vector') is not None for p in points[:5])
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {}

    async def run_validation(self):
        """Run validation on mixture test data"""
        try:
            # Create new collection
            logger.info(f"Creating collection: {self.collection_name}")
            await self.qdrant.create_collection(
                collection_name=self.collection_name,
                vector_size=1024  # ModernBERT embedding dimension
            )
            
            # Process all test files from all directories
            test_files = []
            for test_dir in self.test_dirs:
                test_files.extend(list(test_dir.glob("*.md")))
            
            successful_files = 0
            
            logger.info(f"\nProcessing {len(test_files)} test files...")
            
            for file_path in test_files:
                result = await self.process_file(file_path)
                if result and result.get('success'):
                    successful_files += 1
                    logger.info(f"Successfully processed {file_path.name} - {result.get('num_chunks', 0)} chunks")
                else:
                    logger.warning(f"Failed to process {file_path.name}")
            
            # Optimize collection after loading
            if successful_files > 0:
                logger.info("\nOptimizing collection...")
                await self.qdrant.optimize_collection(self.collection_name)
            
            logger.info(f"\nSuccessfully processed {successful_files}/{len(test_files)} files")
            
            # Only test search if we have processed files
            if successful_files > 0:
                logger.info("\nTesting search functionality...")
                
                test_queries = [
                    ("quantum_flow", "How do quantum states relate to creative flow?"),
                    ("vector_search", "What are the key components of the vector search implementation?"),
                    ("chunking", "Explain the chunking process for text documents"),
                    ("setup", "Development environment setup and configuration"),
                    ("structure", "Project structure and organization")
                ]
                
                successful_searches = 0
                for query_id, query in test_queries:
                    logger.info(f"\nSearch Test: {query_id}")
                    logger.info(f"Query: {query}")
                    results = await self.test_search(query)
                    
                    if results:
                        successful_searches += 1
                        logger.info(f"Found {len(results)} unique results")
                        for i, result in enumerate(results[:3], 1):
                            logger.info(f"\nResult {i}:")
                            logger.info(f"File: {result['payload']['file_name']}")
                            logger.info(f"Score: {result['score']:.3f}")
                            logger.info(f"Text: {result['payload']['text'][:200]}...")
                            logger.info(f"Chunk Info: {result['payload'].get('chunk_index', 'N/A')}/{result['payload'].get('total_chunks', 'N/A')}")
                    else:
                        logger.warning(f"No results found for query: {query}")
                
                logger.info(f"\nCompleted {successful_searches}/{len(test_queries)} successful searches")
                
                # Get detailed collection info
                collection_info = await self.get_collection_info()
                logger.info("\nCollection Information:")
                logger.info(f"Total vectors: {collection_info.get('vectors_count', 'N/A')}")
                logger.info(f"Total points: {collection_info.get('points_count', 'N/A')}")
                logger.info(f"Total segments: {collection_info.get('segments_count', 'N/A')}")
                logger.info(f"Sample points available: {collection_info.get('sample_count', 'N/A')}")
                logger.info(f"Vectors validated: {collection_info.get('has_vectors', False)}")
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise

async def main():
    """Run validation"""
    try:
        logger.info("Starting validation process...")
        validator = MixtureValidator()
        await validator.run_validation()
        logger.info("Validation completed successfully")
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise  # Re-raise to ensure we see the full traceback
    finally:
        # Cleanup
        await asyncio.sleep(0.1)  # Ensure any pending logs are flushed

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
