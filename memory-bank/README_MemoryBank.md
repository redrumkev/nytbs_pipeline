# Memory Bank Folder – Overview

This folder contains dynamic context files used to support iterative prompting and custom instructions for the NYTBS Pipeline. These files capture evolving information and are referenced by our automated coder (CLINE) to maintain focus and efficiency.

## Files Included

- **activeContext.md:**  
  Current active focus. (Update this file when shifting priorities; it now should reflect our focus on building tests and establishing baseline configurations.)

- **productContext.md:**  
  Provides the product context and Phase 1 focus. Future updates will incorporate dynamic context for creative flow.

- **progress.md:**  
  Tracks progress, noting what has been validated and what remains to be built (e.g., minimal test suite, integration of memory-bank workflows).

- **projectbrief.md:**  
  Summarizes the core goal and key requirements of the project.

- **systemPatterns.md:**  
  Describes the overall system architecture and design patterns.

- **techContext.md:**  
  Details the technology stack, development environment, and technical constraints.

- **.clinerules:**  
  Contains recurring instructions and best practices for interacting with CLINE.

## Purpose and Usage

- **Iterative Context:**  
  These files are used by CLINE to “lock in” essential context between iterations.  
- **Updates:**  
  When significant changes occur, update these files and run the "update memory bank" command.
- **Cross-Reference:**  
  Refer to these files from formal documentation (in the `/docs` folder) to avoid duplication while maintaining up-to-date context.

Keep this folder current to ensure that our automated processes always have the latest project context.