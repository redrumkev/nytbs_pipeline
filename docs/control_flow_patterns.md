# ControlFlow Patterns

## Standard Agent Roles
```python
# Writer Agent Configuration
writer_agent = {
    "name": "Writer",
    "model": "claude-3-sonnet-20240229",
    "temperature": 0.7,
    "max_tokens": 4096
}

# Editor Agent Configuration
editor_agent = {
    "name": "Editor",
    "model": "claude-3-haiku-20240307",
    "temperature": 0.3,
    "max_tokens": 2048
}

# Publisher Agent Configuration
publisher_agent = {
    "name": "Publisher",
    "model": "claude-3-haiku-20240307",
    "temperature": 0.4,
    "max_tokens": 2048
}
```

## Standard Workflows
```python
# Chapter Generation Workflow
chapter_workflow = {
    "name": "Chapter Generation",
    "steps": [
        "outline_review",
        "content_assembly",
        "initial_draft",
        "technical_review",
        "style_review",
        "final_revision"
    ],
    "turn_strategy": "RoundRobin"
}

# Review Workflow
review_workflow = {
    "name": "Multi-Agent Review",
    "steps": [
        "publisher_review",
        "demographic_analysis",
        "market_positioning"
    ],
    "turn_strategy": "Moderated"
}
```

## Task Templates
```python
def create_chapter_task(chapter_number: int, outline_path: str):
    """Template for chapter generation tasks"""
    return {
        "objective": f"Generate Chapter {chapter_number}",
        "context_files": [outline_path],
        "stages": [
            "outline_review",
            "draft_generation",
            "review",
            "revision"
        ]
    }
```