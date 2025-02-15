# Context & Custom Instructions

## Overview

This document explains how our project uses a memory bank and custom instructions (.clinerules) to maintain iterative context and streamline our development workflow. This ensures that our active context remains focused and efficient, supporting the dual mandate of 100% quality and 100% efficiency.

## How It Works

- **Memory Bank:**  
  Our memory bank (located in the `memory-bank/` folder) contains dynamic context files such as:
  - **activeContext.md:** Current focus and priorities.
  - **productContext.md:** The product vision and Phase 1 focus.
  - **progress.md:** Progress updates and remaining tasks.
  - **projectbrief.md:** A concise project brief.
  - **systemPatterns.md:** The system’s architecture and design patterns.
  - **techContext.md:** Technical context and development setup.
  
  Updates to these files help maintain a focused context window for CLINE.

- **.clinerules:**  
  Contains recurring instructions for interacting with CLINE. Update this file as new patterns emerge.

## Usage Instructions

1. **Initialize Memory Bank:**  
   When starting work, instruct CLINE:  
   > "Initialize memory bank."  
   This loads the contents of the memory bank files into the active context.

2. **Start a Task:**  
   Begin tasks by saying:  
   > "Follow your custom instructions."  
   This ensures that CLINE uses both the memory bank and .clinerules.

3. **Plan & Act Modes:**  
   - **Plan Mode:**  
     Use prompts like:  
     > "Plan: [Task description]. What’s the likely cause and recommended approach?"  
   - **Act Mode:**  
     Use prompts like:  
     > "Act: Implement [task] based on the plan."

4. **Update Memory Bank:**  
   When context shifts, say:  
   > "Update memory bank."  
   This forces CLINE to re-read the memory bank files.

5. **Review .clinerules:**  
   If repetitive instructions are needed, update the .clinerules file.

## Integration with Formal Docs

For detailed component specifications, refer to the files in the `/docs` folder (e.g., modernbert_component_spec.md, qdrant_component_spec.md). These formal documents provide static, in-depth information, while this document and the memory bank capture the evolving context.

Keep this document succinct. Every token counts!