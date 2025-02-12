# API Usage Examples

## Project Management

### List All Projects
```python
import requests

def list_projects(api_token):
    headers = {"Authorization": f"Bearer {api_token}"}
    response = requests.get("http://localhost:8000/api/projects", headers=headers)
    return response.json()
```

```javascript
async function listProjects(apiToken) {
    const response = await fetch('http://localhost:8000/api/projects', {
        headers: {
            'Authorization': `Bearer ${apiToken}`
        }
    });
    return await response.json();
}
```

```bash
curl -X GET http://localhost:8000/api/projects \
  -H "Authorization: Bearer your_api_token"
```

### Create New Project
```python
def create_project(api_token, project_data):
    headers = {"Authorization": f"Bearer {api_token}"}
    response = requests.post(
        "http://localhost:8000/api/projects",
        headers=headers,
        json=project_data
    )
    return response.json()

# Example usage
project_data = {
    "title": "My Bestseller",
    "description": "A groundbreaking book on creative flow states",
    "target_completion": "2025-12-31"
}
```

## Workflow Management

### Real-time Workflow Updates (WebSocket)
```python
import websockets
import asyncio
import json

async def monitor_workflow(workflow_id, api_token):
    uri = f"ws://localhost:8000/ws/workflow/{workflow_id}"
    
    async with websockets.connect(uri) as websocket:
        # Send authentication
        await websocket.send(json.dumps({
            "type": "auth",
            "token": api_token
        }))
        
        while True:
            message = await websocket.recv()
            data = json.dumps(message)
            
            if data["type"] == "workflow.status":
                print(f"Workflow Status: {data['status']}")
            elif data["type"] == "workflow.progress":
                print(f"Progress: {data['percentage']}%")
            elif data["type"] == "agent.completion":
                print(f"Agent completed: {data['result']}")
```

```javascript
function connectToWorkflow(workflowId, apiToken) {
    const ws = new WebSocket(`ws://localhost:8000/ws/workflow/${workflowId}`);
    
    ws.onopen = () => {
        ws.send(JSON.stringify({
            type: 'auth',
            token: apiToken
        }));
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch(data.type) {
            case 'workflow.status':
                console.log(`Workflow Status: ${data.status}`);
                break;
            case 'workflow.progress':
                console.log(`Progress: ${data.percentage}%`);
                break;
            case 'agent.completion':
                console.log(`Agent completed: ${data.result}`);
                break;
        }
    };
}
```

## Vector Search Integration

### Store Embeddings
```python
from src.core.vector_store.manager import QdrantManager
import torch

async def store_text_embeddings(texts, embeddings, metadata):
    manager = QdrantManager()
    
    success = manager.store_embeddings(
        collection_name='writing_vectors',
        embeddings=embeddings,  # torch.Tensor of shape [N, 1024]
        metadata=metadata      # List of dicts with metadata
    )
    
    return success
```

### Semantic Search
```python
async def search_similar_content(query_embedding, limit=10):
    manager = QdrantManager()
    
    results = manager.search_similar(
        collection_name='writing_vectors',
        query_embedding=query_embedding,  # torch.Tensor of shape [1, 1024]
        limit=limit,
        score_threshold=0.7
    )
    
    return results
```

## Authentication

### Generate API Token
```python
def get_api_token(username, password):
    response = requests.post(
        "http://localhost:8000/api/auth",
        json={
            "username": username,
            "password": password
        }
    )
    return response.json()["token"]
```

### Verify Token
```python
def verify_token(api_token):
    response = requests.post(
        "http://localhost:8000/api/auth/verify",
        headers={"Authorization": f"Bearer {api_token}"}
    )
    return response.status_code == 200
```

## Error Handling

All API endpoints follow a consistent error response format:

```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable error message",
        "details": {
            "additional": "error specific details"
        }
    }
}
```

Example error handling:
```python
def handle_api_request(response):
    if response.status_code >= 400:
        error = response.json().get("error", {})
        raise Exception(
            f"API Error: {error.get('code')} - {error.get('message')}"
        )
    return response.json()
