# API Endpoints Specification

## Core Endpoints

### Project Management
```yaml
/api/projects:
  GET: List all writing projects
  POST: Create new writing project
  
/api/projects/{project_id}:
  GET: Get project details
  PUT: Update project details
  DELETE: Archive project

/api/projects/{project_id}/chapters:
  GET: List project chapters
  POST: Create new chapter
```

### Workflow Management
```yaml
/api/workflows:
  GET: List available workflows
  POST: Create custom workflow

/api/workflows/{workflow_id}/execute:
  POST: Execute workflow
  GET: Get workflow status

/api/workflows/{workflow_id}/results:
  GET: Get workflow results
```

### Agent Management
```yaml
/api/agents:
  GET: List available agents
  POST: Configure new agent

/api/agents/{agent_id}/status:
  GET: Get agent status
```

## WebSocket Events
```yaml
/ws/workflow/{workflow_id}:
  events:
    workflow.status: Workflow status updates
    workflow.progress: Progress updates
    agent.message: Agent messages
    agent.completion: Task completion
```

## Authentication
```yaml
/api/auth:
  POST: Generate API token
  
/api/auth/verify:
  POST: Verify token validity
```