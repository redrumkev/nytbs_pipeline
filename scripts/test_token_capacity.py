import sys
from pathlib import Path
import logging
import asyncio
import time
from typing import List, Dict, Any
import json
from datetime import datetime
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.embeddings.modernbert_processor import ModernBERTProcessor
from src.core.embeddings.text_chunker import TextChunker, ProcessedChunk
from src.core.vector_store.manager import QdrantManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TokenCapacityTestManager:
    def __init__(self):
        self.model_dir = project_root / "src" / "config" / "modernbert"
        self.test_data_dir = project_root / "test_data"
        self.results_dir = project_root / "tests" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.processor = ModernBERTProcessor(self.model_dir)
        self.chunker = TextChunker(self.model_dir)
        self.qdrant = QdrantManager()
        
        # Test configurations
        self.target_token_counts = [
            4096,  # Half capacity
            6144,  # 75% capacity
            8192   # Full capacity
        ]
        self.collection = "test_embeddings"
        
    def analyze_token_distribution(self, chunks: List[ProcessedChunk]) -> Dict[str, Any]:
        """Analyze token distribution across chunks"""
        token_counts = [chunk.metadata.token_count for chunk in chunks]
        complete_notes = [chunk for chunk in chunks if chunk.metadata.is_complete_note]
        chunked_notes = [chunk for chunk in chunks if not chunk.metadata.is_complete_note]
        
        return {
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(chunks) if chunks else 0,
            "complete_notes": len(complete_notes),
            "chunked_notes": len(chunked_notes),
            "complete_note_tokens": sum(c.metadata.token_count for c in complete_notes),
            "chunked_note_tokens": sum(c.metadata.token_count for c in chunked_notes)
        }
        
    def validate_context_window(self, chunks: List[ProcessedChunk]) -> Dict[str, Any]:
        """Validate chunks fit within context window"""
        token_analysis = self.analyze_token_distribution(chunks)
        
        # Calculate optimal chunk combinations
        total_tokens = token_analysis["total_tokens"]
        complete_tokens = token_analysis["complete_note_tokens"]
        chunked_tokens = token_analysis["chunked_note_tokens"]
        
        return {
            "token_analysis": token_analysis,
            "fits_context_window": total_tokens <= 8192,
            "window_utilization": total_tokens / 8192,
            "complete_note_ratio": complete_tokens / total_tokens if total_tokens > 0 else 0,
            "chunked_note_ratio": chunked_tokens / total_tokens if total_tokens > 0 else 0
        }
        
    async def test_capacity(self, target_tokens: int) -> Dict[str, Any]:
        """Test token capacity with smart chunk handling"""
        try:
            start_time = time.time()
            
            # Process test files
            content_dir = self.test_data_dir / "generated_content"
            test_file = next(content_dir.glob("*.md"))
            
            # Process content into chunks
            chunks = self.chunker.process_file(test_file)
            
            # Generate embeddings for chunks
            texts = [chunk.text for chunk in chunks]
            embeddings = await self.processor.process_batch(texts, show_progress=True)
            
            # Store in Qdrant
            metadata = [
                {
                    "text": chunk.text,
                    "is_complete_note": chunk.metadata.is_complete_note,
                    "token_count": chunk.metadata.token_count,
                    "chunk_index": chunk.metadata.chunk_index,
                    "total_chunks": chunk.metadata.total_chunks
                }
                for chunk in chunks
            ]
            
            success = self.qdrant.store_embeddings(
                collection_name=self.collection,
                embeddings=embeddings,
                metadata=metadata
            )
            
            # Validate context window utilization
            validation_results = self.validate_context_window(chunks[:target_tokens // 768 + 1])
            
            # Calculate processing metrics
            processing_time = time.time() - start_time
            
            return {
                "target_tokens": target_tokens,
                "processing_time": processing_time,
                "num_chunks": len(chunks),
                "validation": validation_results,
                "chunk_stats": {
                    "complete_notes": validation_results["token_analysis"]["complete_notes"],
                    "chunked_notes": validation_results["token_analysis"]["chunked_notes"],
                    "avg_tokens_per_chunk": validation_results["token_analysis"]["avg_tokens_per_chunk"]
                },
                "success": success
            }
            
        except Exception as e:
            logger.error(f"Error in capacity test: {str(e)}")
            return {
                "error": str(e)
            }
            
    async def run_tests(self):
        """Run token capacity tests"""
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "configurations": {
                    "target_token_counts": self.target_token_counts,
                    "collection": self.collection
                },
                "results": {}
            }
            
            # Run tests for each target token count
            for target_tokens in self.target_token_counts:
                logger.info(f"\nTesting with target_tokens={target_tokens}")
                result = await self.test_capacity(target_tokens)
                results["results"][f"tokens_{target_tokens}"] = result
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.results_dir / f"token_capacity_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"\nResults saved to: {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            raise

async def main():
    """Run token capacity tests"""
    try:
        manager = TokenCapacityTestManager()
        results = await manager.run_tests()
        
        # Print summary
        logger.info("\nTest Summary:")
        for token_config, result in results["results"].items():
            if "error" not in result:
                validation = result["validation"]
                
                logger.info(f"\n{token_config}:")
                logger.info(f"Total tokens: {validation['token_analysis']['total_tokens']}")
                logger.info(f"Window utilization: {validation['window_utilization']:.2%}")
                logger.info(f"Fits context window: {validation['fits_context_window']}")
                logger.info(f"Complete notes: {validation['token_analysis']['complete_notes']}")
                logger.info(f"Chunked notes: {validation['token_analysis']['chunked_notes']}")
                logger.info(f"Processing time: {result['processing_time']:.3f} seconds")
            else:
                logger.error(f"\n{token_config}: Failed - {result['error']}")
                
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
