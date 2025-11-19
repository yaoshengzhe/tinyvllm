"""
tinyvllm - A tiny version of vllm inspired by nano-vllm.
"""

import os
from dataclasses import dataclass
import torch
import numpy as np
from transformers import AutoConfig

@dataclass
class Config:
    """Configuration for the TinyVLLM engine."""
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        # Validate model path - must be a local directory for this implementation
        if not os.path.isdir(self.model):
            raise ValueError(f"Model path '{self.model}' is not a valid directory.")

        # Validate block size
        if self.kvcache_block_size % 256 != 0:
            raise ValueError(f"kvcache_block_size ({self.kvcache_block_size}) must be a multiple of 256.")

        # Validate tensor parallel size
        if not (1 <= self.tensor_parallel_size <= 8):
            raise ValueError(f"tensor_parallel_size ({self.tensor_parallel_size}) must be between 1 and 8.")

        # Load HuggingFace config
        try:
            self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        except Exception as e:
            raise ValueError(f"Failed to load config from {self.model}: {e}")

        # Adjust max_model_len based on model's capabilities
        if hasattr(self.hf_config, "max_position_embeddings"):
            self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        
        # Ensure batched tokens capacity is sufficient
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be >= "
                f"max_model_len ({self.max_model_len})"
            )

class SamplingParams:
    def __init__(self, temperature=1.0, top_p=1.0):
        self.temperature = temperature
        self.top_p = top_p

class LLM:
    def __init__(self, model):
        self.model = model

    def generate(self, prompts, sampling_params=None):
        # TODO: Implement generation logic
        return []

def main():
    print("Hello from tinyvllm!")

if __name__ == "__main__":
    main()
