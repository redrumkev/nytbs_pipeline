import sys
from pathlib import Path
import logging
import asyncio
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.embeddings.modernbert_processor import ModernBERTProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def verify_environment():
    """Verify all components are working"""
    try:
        logger.info("Verifying environment setup...")
        
        # Check CUDA
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Check ModernBERT
        model_dir = project_root / "src" / "config" / "modernbert"
        logger.info(f"Checking ModernBERT at: {model_dir}")
        processor = ModernBERTProcessor(model_dir)
        
        # Test embedding generation
        test_text = "This is a test of the embedding system."
        logger.info("Testing embedding generation...")
        embedding = await processor.process_realtime(test_text)
        logger.info(f"Generated embedding shape: {embedding.shape}")
        
        logger.info("Environment verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Environment verification failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(verify_environment())
    except KeyboardInterrupt:
        logger.info("Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)