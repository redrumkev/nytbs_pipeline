import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Union, Dict, Optional, Tuple
from dataclasses import dataclass
import asyncio
import logging
from pathlib import Path
import gc
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    batch_size: int = 32
    max_length: int = 8192
    use_fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_size: int = 1000

class ModernBERTProcessor:
    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        self.config = ProcessingConfig()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available")
        logger.info(f"Initializing ModernBERT on {self.config.device}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load model and tokenizer
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        
        # Initialize embedding cache (LRU)
        self.embedding_cache = {}
        self.cache_priority = []
        
        torch.cuda.empty_cache()
        gc.collect()

    def _load_model(self) -> AutoModel:
        try:
            model = AutoModel.from_pretrained(str(self.model_dir))
            if self.config.use_fp16:
                model = model.half()  # Convert to FP16 for faster inference
            model = model.to(self.config.device)
            model.eval()  # Set to evaluation mode
            model.gradient_checkpointing_enable()  # Improve memory efficiency
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _load_tokenizer(self) -> AutoTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            tokenizer.model_max_length = self.config.max_length
            if tokenizer.pad_token is None:
                if tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.model.resize_token_embeddings(len(tokenizer))
            if tokenizer.unk_token is None:
                tokenizer.unk_token = '[UNK]'
            if tokenizer.sep_token is None:
                tokenizer.sep_token = '[SEP]'
            if tokenizer.cls_token is None:
                tokenizer.cls_token = '[CLS]'
            return tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {str(e)}")

    def _update_cache(self, text: str, embedding: torch.Tensor):
        if len(self.embedding_cache) >= self.config.cache_size:
            lru_text = self.cache_priority.pop(0)
            del self.embedding_cache[lru_text]
        self.embedding_cache[text] = embedding
        self.cache_priority.append(text)

    def _get_cached_embedding(self, text: str) -> Optional[torch.Tensor]:
        if text in self.embedding_cache:
            self.cache_priority.remove(text)
            self.cache_priority.append(text)
            return self.embedding_cache[text]
        return None

    async def process_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> torch.Tensor:
        batch_size = batch_size or self.config.batch_size
        if not texts:
            return torch.empty(0)
        try:
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            all_embeddings = []
            pbar = tqdm(total=len(texts), disable=not show_progress)
            for batch in batches:
                cached_embeddings = []
                texts_to_process = []
                for text in batch:
                    text = text.strip()
                    if not text:
                        continue
                    cached_emb = self._get_cached_embedding(text)
                    if cached_emb is not None:
                        cached_embeddings.append(cached_emb)
                    else:
                        texts_to_process.append(text)
                if texts_to_process:
                    inputs = self.tokenizer(
                        texts_to_process,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                        add_special_tokens=True,
                        return_tensors="pt"
                    )
                    if 'token_type_ids' in inputs:
                        del inputs['token_type_ids']
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        if self.config.use_fp16:
                            with torch.amp.autocast('cuda'):
                                outputs = self.model(**inputs)
                        else:
                            outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    for text, emb in zip(texts_to_process, embeddings):
                        self._update_cache(text, emb.cpu())
                    all_embeddings.extend([emb.cpu() for emb in embeddings])
                all_embeddings.extend(cached_embeddings)
                pbar.update(len(batch))
                if texts_to_process:
                    del inputs, outputs, embeddings
                    torch.cuda.empty_cache()
            pbar.close()
            final_embeddings = torch.stack(all_embeddings) if all_embeddings else torch.empty(0)
            return final_embeddings
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            raise

    async def process_realtime(
        self,
        text: str,
        priority: int = 0
    ) -> torch.Tensor:
        try:
            text = text.strip()
            if not text:
                raise ValueError("Empty text input provided for realtime processing")
            cached_emb = self._get_cached_embedding(text)
            if cached_emb is not None:
                return cached_emb
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                add_special_tokens=True,
                return_tensors="pt"
            )
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            with torch.no_grad():
                if self.config.use_fp16:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = F.normalize(embedding, p=2, dim=1)
            self._update_cache(text, embedding.cpu())
            del inputs, outputs
            torch.cuda.empty_cache()
            return embedding.cpu()
        except Exception as e:
            logger.error(f"Realtime processing error: {str(e)}")
            raise

    def get_memory_info(self) -> Dict[str, float]:
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "max": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }

    @staticmethod
    def test_cuda() -> Tuple[bool, Optional[str]]:
        try:
            if not torch.cuda.is_available():
                return False, "CUDA is not available"
            device_name = torch.cuda.get_device_name(0)
            compute_capability = torch.cuda.get_device_capability()
            logger.info(f"Found GPU: {device_name}")
            logger.info(f"Compute Capability: {compute_capability}")
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            z = torch.matmul(x, y)
            del x, y, z
            x = torch.randn(1000, 1000, device="cuda", dtype=torch.float16)
            y = torch.randn(1000, 1000, device="cuda", dtype=torch.float16)
            z = torch.matmul(x, y)
            del x, y, z
            torch.cuda.empty_cache()
            return True, None
        except Exception as e:
            return False, f"CUDA test failed: {str(e)}"