import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_collections():
    """Delete existing collections to start fresh"""
    try:
        client = QdrantClient(host="localhost", port=6333)
        
        # Get existing collections
        collections = client.get_collections()
        
        # Delete each collection
        for collection in collections.collections:
            logger.info(f"Deleting collection: {collection.name}")
            client.delete_collection(collection_name=collection.name)
            
        logger.info("All collections deleted successfully")
        
    except Exception as e:
        logger.error(f"Failed to reset collections: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    reset_collections()
