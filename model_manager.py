import os
import logging
import torch
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class BaseModelManager:
    """Manager for base model - loaded once at startup"""

    def __init__(self, model_name: str = "EleutherAI/pythia-2.8b", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

        logger.info(f"Initializing BaseModelManager for {model_name}")
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading base model: {self.model_name}")
            logger.info(f"Target device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )

            self.model.to(self.device)
            self.model.eval()

            param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
            logger.info(f"Base model loaded: {param_count:.2f}B parameters")

            if self.device == "cuda":
                vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.info(f"VRAM allocated: {vram_allocated:.2f} GB")

        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, top_p: float = 0.9, repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.eos_token_id
            }

            if repetition_penalty > 1.0:
                gen_kwargs["repetition_penalty"] = repetition_penalty
            if no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return output_text.strip()

        except Exception as e:
            logger.error(f"Base model generation failed: {e}")
            raise

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None


class FineTunedModelManager:
    """Manager for fine-tuned models - loaded on-demand"""

    def __init__(self, base_tokenizer: AutoTokenizer, device: Optional[str] = None):
        self.tokenizer = base_tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_model_id = None

        logger.info("FineTunedModelManager initialized")

    def load_model(self, model_path: Path, model_id: str):
        try:
            logger.info(f"Loading fine-tuned model: {model_id}")

            if self.model is not None:
                self.unload()

            logger.info(f"Loading checkpoint from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract model weights from state key if present
            if isinstance(checkpoint, dict) and "state" in checkpoint:
                checkpoint = checkpoint["state"]

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model = AutoModelForCausalLM.from_pretrained(
                    "EleutherAI/pythia-2.8b",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, torch.nn.Module):
                self.model = checkpoint
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    "EleutherAI/pythia-2.8b",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            self.current_model_id = model_id

            logger.info(f"Fine-tuned model loaded: {model_id}")

            if self.device == "cuda":
                vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.info(f"VRAM allocated: {vram_allocated:.2f} GB")

        except Exception as e:
            logger.error(f"Failed to load fine-tuned model {model_id}: {e}")
            self.model = None
            self.current_model_id = None
            raise

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, top_p: float = 0.9, repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0) -> str:
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.eos_token_id
            }

            if repetition_penalty > 1.0:
                gen_kwargs["repetition_penalty"] = repetition_penalty
            if no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return output_text.strip()

        except Exception as e:
            logger.error(f"Fine-tuned model generation failed: {e}")
            raise

    def unload(self):
        if self.model is not None:
            logger.info(f"Unloading fine-tuned model: {self.current_model_id}")
            del self.model
            self.model = None
            self.current_model_id = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.info(f"VRAM after unload: {vram_allocated:.2f} GB")

    def is_loaded(self) -> bool:
        return self.model is not None
