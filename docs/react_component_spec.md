# Component Specification: React

## Purpose

Provides the minimal user interface for Phase 1.

## Configuration

- **Port:** 3000  
- **API URL:** `http://localhost:8000/api`

## UI (Phase 1)

- Text input field for search queries.
- Submit button.
- Results display area (showing file names, text snippets, and scores).

## API Interaction

- Sends POST requests to `/api/search`.
- Displays results returned from FastAPI.

## Testing

- Verify UI rendering and responsiveness.
- Test API integration with mocked endpoints.
- Validate user interactions.