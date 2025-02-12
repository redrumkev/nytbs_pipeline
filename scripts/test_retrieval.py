import sys
from pathlib import Path
import logging
import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
import torch

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

class RetrievalTestManager:
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
        self.top_k_values = [5, 10, 20, 50]
        self.collection = "test_embeddings"
        
    def calculate_semantic_score(self, query_embedding: torch.Tensor, result_embedding: torch.Tensor) -> float:
        """Calculate semantic similarity score"""
        query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
        result_norm = torch.nn.functional.normalize(result_embedding, p=2, dim=0)
        return float(torch.dot(query_norm, result_norm))
        
    def evaluate_coherence(self, chunks: List[str], is_complete: List[bool]) -> float:
        """Evaluate coherence between retrieved chunks"""
        if len(chunks) < 2:
            return 1.0
            
        if all(is_complete):  # All complete notes
            return 1.0  # Perfect coherence for complete notes
            
        # For mixed or all chunked content
        chunk_embeddings = torch.stack([
            self.processor.get_embedding(chunk)
            for chunk, complete in zip(chunks, is_complete)
            if not complete  # Only evaluate coherence for chunked content
        ])
        
        # Calculate pairwise similarities
        chunk_embeddings = torch.nn.functional.normalize(chunk_embeddings, p=2, dim=1)
        similarities = torch.mm(chunk_embeddings, chunk_embeddings.t())
        
        # Get average of adjacent chunk similarities
        adjacent_similarities = [
            float(similarities[i, i+1])
            for i in range(len(chunks)-1)
        ]
        
        return sum(adjacent_similarities) / len(adjacent_similarities)
        
    async def test_retrieval(self, collection: str, query: str, top_k: int) -> Dict[str, Any]:
        """Test retrieval for a single query"""
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = await self.processor.process_realtime(query)
            
            # Perform search
            results = self.qdrant.search_similar(
                collection_name=collection,
                query_embedding=query_embedding,
                limit=top_k
            )
            
            retrieval_time = time.time() - start_time
            
            # Calculate metrics
            semantic_scores = [
                self.calculate_semantic_score(query_embedding, torch.tensor(r.vector))
                for r in results
            ]
            
            # Get chunks and their metadata
            chunks = [r.payload.get("text", "") for r in results]
            is_complete = [r.payload.get("is_complete_note", False) for r in results]
            coherence_score = self.evaluate_coherence(chunks, is_complete)
            
            # Calculate result quality metrics
            complete_notes = sum(1 for r in results if r.payload.get("is_complete_note", False))
            chunked_notes = len(results) - complete_notes
            
            return {
                "retrieval_time": retrieval_time,
                "num_results": len(results),
                "semantic_scores": {
                    "min": min(semantic_scores),
                    "max": max(semantic_scores),
                    "avg": sum(semantic_scores) / len(semantic_scores)
                },
                "coherence_score": coherence_score,
                "complete_notes": complete_notes,
                "chunked_notes": chunked_notes
            }
            
        except Exception as e:
            logger.error(f"Error in retrieval test: {str(e)}")
            return {
                "error": str(e)
            }
            
    async def run_tests(self):
        """Run retrieval tests with different configurations"""
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "configurations": {
                    "top_k_values": self.top_k_values,
                    "collection": self.collection
                },
                "results": {}
            }
            
            # Generate test queries from writing samples
            samples_dir = self.test_data_dir / "writing_samples"
            test_files = list(samples_dir.glob("*.md"))[:5]  # Use first 5 files for queries
            test_queries = []
            
            for file_path in test_files:
                content = file_path.read_text(encoding='utf-8')
                # Take first paragraph as query
                paragraphs = content.split('\n\n')
                if paragraphs:
                    test_queries.append(paragraphs[0])
            
            # Run tests for each configuration
            # Run tests for each top_k value
            for top_k in self.top_k_values:
                logger.info(f"\nTesting with top_k={top_k}")
                query_results = []
                
                for i, query in enumerate(test_queries):
                    logger.info(f"Processing query {i+1}/{len(test_queries)}")
                    result = await self.test_retrieval(self.collection, query, top_k)
                    query_results.append({
                        "query_index": i,
                        "query_text": query[:100] + "...",  # Truncate for logging
                        "metrics": result
                    })
                    
                results["results"][f"top_k_{top_k}"] = query_results
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.results_dir / f"retrieval_test_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"\nResults saved to: {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            raise

async def main():
    """Run retrieval tests"""
    try:
        manager = RetrievalTestManager()
        results = await manager.run_tests()
        
        # Print summary
        logger.info("\nTest Summary:")
        for top_k_config, query_results in results["results"].items():
                successful = [r for r in query_results if "error" not in r["metrics"]]
                failed = [r for r in query_results if "error" in r["metrics"]]
                
                avg_time = sum(r["metrics"]["retrieval_time"] for r in successful) / len(successful) if successful else 0
                avg_coherence = sum(r["metrics"]["coherence_score"] for r in successful) / len(successful) if successful else 0
                
                avg_complete = sum(r["metrics"]["complete_notes"] for r in successful) / len(successful) if successful else 0
                avg_chunked = sum(r["metrics"]["chunked_notes"] for r in successful) / len(successful) if successful else 0
                
                logger.info(f"\n{top_k_config}:")
                logger.info(f"Successfully processed: {len(successful)} queries")
                logger.info(f"Failed: {len(failed)} queries")
                logger.info(f"Average retrieval time: {avg_time:.3f} seconds")
                logger.info(f"Average coherence score: {avg_coherence:.3f}")
                logger.info(f"Average complete notes retrieved: {avg_complete:.1f}")
                logger.info(f"Average chunked notes retrieved: {avg_chunked:.1f}")
                
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
