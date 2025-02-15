# Component Specification: TextChunker

## Purpose
The TextChunker processes raw text from files or upstream processes into discrete, semantically meaningful chunks that are suitable for generating vector embeddings. It preserves important structures such as headers and code blocks while removing extraneous content (e.g., system paths).

## Configuration
- Model Directory: src/config/modernbert/
- Chunk Size: 512 tokens (maximum tokens per chunk)
- Overlap Size: 128 tokens (set for future enhancements, not currently used in the simplified version)
- Minimum Chunk Size: 50 tokens (to filter out overly short or irrelevant segments)

## API (via text_chunker.py)
- process_file(file_path: Path) -> List[ProcessedChunk]:
Reads a file and processes its content into a list of text chunks, each accompanied by metadata (e.g., file name, token count, chunk index, and total chunks).

- create_chunks(text: str, filename: Optional[str] = None) -> List[ProcessedChunk]:
Cleans the input text and splits it into chunks based on natural paragraph boundaries. Preserves headers and code blocks.

## Integration
- Input:
Receives raw text data from file ingestion or other upstream processes.

- Output:
Produces a series of text chunks along with metadata. These chunks are then used by the ModernBERT processor to generate embeddings, which are stored in Qdrant.

- Context:
Designed for a frictionless, one-person local setup. It is a key component in the overall pipeline that supports automated semantic search and serves as part of your “second brain.”

- Testing
Unit Testing:
Verify file processing for various cases (empty files, short content, typical markdown files) to ensure correct chunking and preservation of natural breakpoints.

- Integration Testing:
Ensure that the text chunks integrate seamlessly with the ModernBERT embedding generation and Qdrant storage, supporting queries that return relevant semantic matches.

- Performance:
Validate that the chunker works efficiently with both individual files and batch processing without adding overhead to the pipeline.