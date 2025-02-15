from dataclasses import dataclass
from typing import List, Optional
import re
from transformers import AutoTokenizer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    is_complete_note: bool
    token_count: int
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    original_file: Optional[str] = None

@dataclass
class ProcessedChunk:
    """A processed text chunk with its metadata."""
    text: str
    metadata: ChunkMetadata

class TextChunker:
    def __init__(self, model_dir: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.chunk_size = 512        # Maximum tokens per chunk
        self.overlap_size = 128        # Overlap (not used in this simplified version)
        self.min_chunk_size = 50       # Minimum tokens per chunk

    def clean_text(self, text: str) -> str:
        """
        Perform minimal cleaning on the text.
        Removes common system path patterns while preserving overall structure,
        including code blocks and headers.
        """
        # Remove Windows-style paths (e.g., "C:\Projects\...") and Unix paths from /etc, /var, or /usr
        text = re.sub(r'(?:[A-Z]:\\[^\s]+)', '', text)
        text = re.sub(r'(?:\/(?:etc|var|usr)\/[^\s]+)', '', text)
        return text.strip()

    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Splits the cleaned text into paragraphs based on double newlines.
        This method preserves code blocks and header structures.
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs

    def combine_paragraphs_into_chunks(self, paragraphs: List[str]) -> List[str]:
        """
        Combine paragraphs into chunks that do not exceed the target token length.
        Paragraphs are concatenated with double newlines.
        """
        chunks = []
        current_chunk = ""
        for paragraph in paragraphs:
            if current_chunk:
                potential_chunk = current_chunk + "\n\n" + paragraph
            else:
                potential_chunk = paragraph
            tokens = self.tokenizer.encode(potential_chunk)
            if len(tokens) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # If current chunk exists and is long enough, add it as a chunk
                if current_chunk and len(self.tokenizer.encode(current_chunk)) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                # Start a new chunk with the current paragraph
                current_chunk = paragraph
        if current_chunk and len(self.tokenizer.encode(current_chunk)) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        return chunks

    def create_chunks(self, text: str, filename: Optional[str] = None) -> List[ProcessedChunk]:
        """
        Clean the text, split it into paragraphs, combine paragraphs into chunks,
        and then wrap each chunk with metadata.
        """
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []
        paragraphs = self.split_into_paragraphs(cleaned_text)
        chunk_texts = self.combine_paragraphs_into_chunks(paragraphs)
        total_chunks = len(chunk_texts)
        chunks = []
        for idx, chunk in enumerate(chunk_texts):
            tokens = self.tokenizer.encode(chunk)
            if len(tokens) < self.min_chunk_size:
                continue
            chunks.append(ProcessedChunk(
                text=chunk,
                metadata=ChunkMetadata(
                    is_complete_note=(total_chunks == 1),
                    token_count=len(tokens),
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    original_file=filename
                )
            ))
        return chunks

    def process_file(self, file_path: Path) -> List[ProcessedChunk]:
        """
        Reads the file and processes its content into chunks.
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            if not content.strip():
                logger.warning(f"Empty or invalid content in {file_path}")
                return []
            return self.create_chunks(content, filename=file_path.name)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []