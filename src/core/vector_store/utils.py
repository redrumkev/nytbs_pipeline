import logging
from qdrant_client import QdrantClient
from typing import Dict, Any

# Configure logging to see debug messages
# logging.basicConfig(level=logging.DEBUG)

def test_qdrant_connection() -> Dict[str, Any]:
    """Test connection to Qdrant server using client.info() and get_collections()"""
    try:
        # Initialize the Qdrant client (adjust host/port if needed)
        client = QdrantClient(host="localhost", port=6333)
        
        # Retrieve server information using the built-in info() method.
        cluster_info = client.info()
        
        # Retrieve information about collections in the server.
        collections_info = client.get_collections()
        
        return {
            "status": "connected",
            "cluster_info": cluster_info,  # Contains keys like title, version, commit.
            "collections": [c.name for c in collections_info.collections]
        }
    except Exception as e:
        logging.error(f"Qdrant connection failed: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    result = test_qdrant_connection()
    print(f"Connection test result: {result}")