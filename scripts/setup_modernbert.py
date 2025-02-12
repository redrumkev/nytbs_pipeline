import sys
from pathlib import Path
import logging
import asyncio
import torch
from typing import List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.embeddings.modernbert_processor import ModernBERTProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_processor(processor: ModernBERTProcessor) -> Tuple[bool, Optional[str]]:
    """Test both batch and realtime processing"""
    try:
        # Test texts
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump!",
            "The five boxing wizards jump quickly."
        ]
        single_text = "Testing real-time processing capabilities."
        
        logger.info("Testing batch processing...")
        batch_embeddings = await processor.process_batch(test_texts, show_progress=True)
        logger.info(f"Batch embeddings shape: {batch_embeddings.shape}")
        
        logger.info("\nTesting real-time processing...")
        realtime_embedding = await processor.process_realtime(single_text)
        logger.info(f"Real-time embedding shape: {realtime_embedding.shape}")
        
        # Verify embeddings
        assert batch_embeddings.shape[0] == len(test_texts), "Incorrect batch size"
        assert realtime_embedding.shape[0] == 1, "Incorrect real-time batch size"
        assert torch.isfinite(batch_embeddings).all(), "Found non-finite values in batch embeddings"
        assert torch.isfinite(realtime_embedding).all(), "Found non-finite values in real-time embedding"
        
        # Test cache functionality
        logger.info("\nTesting cache functionality...")
        cached_embedding = await processor.process_realtime(single_text)
        assert torch.allclose(realtime_embedding, cached_embedding), "Cache retrieval failed"
        
        logger.info("All processor tests passed successfully!")
        return True, None
        
    except Exception as e:
        return False, f"Processor test failed: {str(e)}"

def verify_model_files(model_dir: Path) -> Tuple[bool, Optional[str]]:
    """Verify all required model files exist and have correct sizes"""
    required_files = {
        "config.json": 1024,      # Minimum size in bytes
        "tokenizer.json": 1024,
        "pytorch_model.bin": 1024 * 1024  # At least 1MB
    }
    
    for filename, min_size in required_files.items():
        file_path = model_dir / filename
        if not file_path.exists():
            return False, f"Missing required file: {filename}"
        if file_path.stat().st_size < min_size:
            return False, f"File too small (possibly corrupted): {filename}"
    
    return True, None

async def setup_modernbert() -> bool:
    """Setup ModernBERT with optimized configuration"""
    try:
        # Get model directory
        model_dir = project_root / "src" / "config" / "modernbert"
        
        # Step 1: Test CUDA setup
        logger.info("Testing CUDA setup...")
        cuda_ok, cuda_error = ModernBERTProcessor.test_cuda()
        if not cuda_ok:
            logger.error(f"CUDA setup verification failed: {cuda_error}")
            return False
        
        # Step 2: Verify model files
        logger.info("Verifying model files...")
        files_ok, files_error = verify_model_files(model_dir)
        if not files_ok:
            logger.error(f"Model files verification failed: {files_error}")
            return False
        
        # Step 3: Initialize and test processor
        logger.info("Initializing ModernBERT processor...")
        processor = ModernBERTProcessor(model_dir)
        
        logger.info("Testing processor functionality...")
        test_ok, test_error = await test_processor(processor)
        if not test_ok:
            logger.error(f"Processor testing failed: {test_error}")
            return False
        
        # Print final configuration
        memory_info = processor.get_memory_info()
        logger.info("\nModernBERT Configuration:")
        logger.info("------------------------")
        logger.info(f"Model Directory: {model_dir}")
        logger.info(f"Context Length: {processor.config.max_length}")
        logger.info(f"FP16 Enabled: {processor.config.use_fp16}")
        logger.info(f"Batch Size: {processor.config.batch_size}")
        logger.info(f"Cache Size: {processor.config.cache_size} embeddings")
        logger.info(f"GPU Memory:")
        logger.info(f"  - Allocated: {memory_info['allocated']:.2f} GB")
        logger.info(f"  - Reserved: {memory_info['reserved']:.2f} GB")
        logger.info(f"  - Total: {memory_info['max']:.2f} GB")
        
        logger.info("\nModernBERT setup completed successfully!")
        return True
            
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(setup_modernbert())
    sys.exit(0 if success else 1)
