import sys
from pathlib import Path
import logging
import asyncio
import time
import psutil
import torch
from typing import List, Dict, Any
import json
from datetime import datetime

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

class ChunkingTestManager:
    def __init__(self):
        self.model_dir = project_root / "src" / "config" / "modernbert"
        self.test_data_dir = project_root / "test_data"
        self.results_dir = project_root / "tests" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.processor = ModernBERTProcessor(self.model_dir)
        self.chunker = TextChunker(self.model_dir)
        self.qdrant = QdrantManager()
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss / (1024 * 1024 * 1024),  # GB
            "vms": memory_info.vms / (1024 * 1024 * 1024),  # GB
            "gpu_allocated": torch.cuda.memory_allocated() / (1024 * 1024 * 1024) if torch.cuda.is_available() else 0,
            "gpu_reserved": torch.cuda.memory_reserved() / (1024 * 1024 * 1024) if torch.cuda.is_available() else 0
        }
        
    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file and return metrics"""
        try:
            start_time = time.time()
            start_memory = self.get_memory_info()
            
            # Read file content
            content = file_path.read_text(encoding='utf-8')
            
            # Process text into chunks
            chunks = self.chunker.process_file(file_path)
            
            # Process chunks into embeddings
            texts = [chunk.text for chunk in chunks]
            embeddings = await self.processor.process_batch(texts, show_progress=True)
            
            end_time = time.time()
            end_memory = self.get_memory_info()
            
            # Store in Qdrant
            collection_name = "test_embeddings"
            # Create metadata for each chunk
            metadata = [
                {
                    "file_name": file_path.name,
                    "is_complete_note": chunk.metadata.is_complete_note,
                    "token_count": chunk.metadata.token_count,
                    "chunk_index": chunk.metadata.chunk_index,
                    "total_chunks": chunk.metadata.total_chunks
                }
                for chunk in chunks
            ]
            
            success = self.qdrant.store_embeddings(
                collection_name=collection_name,
                embeddings=embeddings,
                metadata=metadata
            )
            
            return {
                "file_name": file_path.name,
                "processing_time": end_time - start_time,
                "num_chunks": len(chunks),
                "complete_notes": sum(1 for c in chunks if c.metadata.is_complete_note),
                "chunked_notes": sum(1 for c in chunks if not c.metadata.is_complete_note),
                "memory_delta": {
                    k: end_memory[k] - start_memory[k]
                    for k in start_memory
                },
                "success": success
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return {
                "file_name": file_path.name,
                "error": str(e)
            }
            
    async def run_tests(self):
        """Run chunking tests on all test data"""
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "results": []
            }
            
            # Test writing samples
            samples_dir = self.test_data_dir / "writing_samples"
            samples_files = list(samples_dir.glob("*.md"))
            
            # Test generated content
            content_dir = self.test_data_dir / "generated_content"
            content_files = list(content_dir.glob("*.md"))
            
            all_files = samples_files + content_files
            
            for file_path in all_files:
                logger.info(f"Processing {file_path.name}")
                result = await self.process_file(file_path)
                results["results"].append(result)
                
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.results_dir / f"chunking_test_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"\nResults saved to: {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            raise

async def main():
    """Run chunking tests"""
    try:
        manager = ChunkingTestManager()
        results = await manager.run_tests()
        
        # Print summary
        logger.info("\nTest Summary:")
        successful = [r for r in results["results"] if r.get("success", False)]
        failed = [r for r in results["results"] if not r.get("success", False)]
        
        total_time = sum(r.get("processing_time", 0) for r in successful)
        avg_chunks = sum(r.get("num_chunks", 0) for r in successful) / len(successful) if successful else 0
        complete_notes = sum(r.get("complete_notes", 0) for r in successful)
        chunked_notes = sum(r.get("chunked_notes", 0) for r in successful)
        
        logger.info("\nProcessing Summary:")
        logger.info(f"Successfully processed: {len(successful)} files")
        logger.info(f"Failed: {len(failed)} files")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Average chunks per file: {avg_chunks:.2f}")
        logger.info(f"Complete notes (no chunking): {complete_notes}")
        logger.info(f"Notes requiring chunking: {chunked_notes}")
            
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
