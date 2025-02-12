from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re
from transformers import AutoTokenizer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for a text chunk"""
    is_complete_note: bool
    token_count: int
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    original_file: Optional[str] = None

@dataclass
class ProcessedChunk:
    """A processed text chunk with its metadata"""
    text: str
    metadata: ChunkMetadata

class TextChunker:
    def __init__(self, model_dir: Path):
        """Initialize with model tokenizer for accurate token counting"""
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.chunk_size = 768
        self.overlap_size = 192
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using model's tokenizer"""
        return len(self.tokenizer.encode(text))
        
    def find_break_point(self, text: str, target_idx: int, window: int = 100) -> int:
        """Find the best break point near the target index"""
        # Define break points in order of preference
        break_patterns = [
            r'\n\n',     # Double newline (paragraph)
            r'\.\s',     # End of sentence
            r'\,\s',     # Comma
            r'\s'        # Any whitespace
        ]
        
        # Search within window before and after target
        start = max(0, target_idx - window)
        end = min(len(text), target_idx + window)
        search_text = text[start:end]
        
        # Try each pattern in order of preference
        for pattern in break_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Find closest match to target
                closest = min(matches, key=lambda x: abs(x.start() + start - target_idx))
                return closest.start() + start
                
        # If no good break point found, use target_idx
        return target_idx
        
    def create_chunks(self, text: str, filename: Optional[str] = None) -> List[ProcessedChunk]:
        """Process text into chunks with smart handling for short notes"""
        token_count = self.count_tokens(text)
        
        # Handle short notes
        if token_count < self.chunk_size:
            return [ProcessedChunk(
                text=text,
                metadata=ChunkMetadata(
                    is_complete_note=True,
                    token_count=token_count,
                    chunk_index=0,
                    total_chunks=1,
                    original_file=filename
                )
            )]
            
        # Process longer texts into chunks
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(text):
            # Calculate end position for current chunk
            if current_pos == 0:
                # First chunk: no leading overlap
                target_end = self._find_chunk_end(text, current_pos, self.chunk_size)
            else:
                # Include overlap
                target_end = self._find_chunk_end(text, current_pos, self.chunk_size - self.overlap_size)
            
            # Find natural break point
            end_pos = self.find_break_point(text, target_end)
            
            # Extract chunk
            chunk_text = text[current_pos:end_pos].strip()
            if chunk_text:  # Only add non-empty chunks
                chunks.append(ProcessedChunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        is_complete_note=False,
                        token_count=self.count_tokens(chunk_text),
                        chunk_index=chunk_index,
                        total_chunks=None,  # Will set after all chunks created
                        original_file=filename
                    )
                ))
                chunk_index += 1
            
            # Move position for next chunk
            current_pos = max(current_pos + 1, end_pos - self.overlap_size)
            
        # Update total_chunks in metadata
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.metadata.total_chunks = total_chunks
            
        return chunks
        
    def _find_chunk_end(self, text: str, start: int, length: int) -> int:
        """Find position that approximately contains desired token length"""
        # Estimate characters per token (rough approximation)
        chars_per_token = 4
        target_chars = length * chars_per_token
        
        # Don't exceed text length
        return min(start + target_chars, len(text))
        
    def process_file(self, file_path: Path) -> List[ProcessedChunk]:
        """Process a file into chunks"""
        try:
            text = file_path.read_text(encoding='utf-8')
            return self.create_chunks(text, filename=file_path.name)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
            
    def process_batch(self, texts: List[str], filenames: Optional[List[str]] = None) -> List[ProcessedChunk]:
        """Process multiple texts into chunks"""
        if filenames is None:
            filenames = [None] * len(texts)
            
        all_chunks = []
        for text, filename in zip(texts, filenames):
            chunks = self.create_chunks(text, filename)
            all_chunks.extend(chunks)
            
        return all_chunks
