"""
tinyvllm - A tiny version of vllm inspired by nano-vllm.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import AsyncGenerator
from collections import deque

import asyncio
import os
import uuid
from dataclasses import dataclass
import torch
import numpy as np
import xxhash
from transformers import AutoConfig

DEFAULT_BLOCK_SIZE = 256


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
            raise ValueError(
                f"kvcache_block_size ({self.kvcache_block_size}) must be a multiple of 256."
            )

        # Validate tensor parallel size
        if not (1 <= self.tensor_parallel_size <= 8):
            raise ValueError(
                f"tensor_parallel_size ({self.tensor_parallel_size}) must be between 1 and 8."
            )

        # Load HuggingFace config
        try:
            self.hf_config = AutoConfig.from_pretrained(
                self.model, trust_remote_code=True
            )
        except Exception as e:
            raise ValueError(f"Failed to load config from {self.model}: {e}")

        # Adjust max_model_len based on model's capabilities
        if hasattr(self.hf_config, "max_position_embeddings"):
            self.max_model_len = min(
                self.max_model_len, self.hf_config.max_position_embeddings
            )

        # Ensure batched tokens capacity is sufficient
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be >= "
                f"max_model_len ({self.max_model_len})"
            )


class SamplingParams:
    """Sampling parameters for generation."""

    def __init__(self, temperature=1.0, top_p=1.0):
        self.temperature = temperature
        self.top_p = top_p


class Sequence:
    """Represents a single input/output sequence."""

    def __init__(
        self,
        token_ids: list[int],
        params: SamplingParams = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ):
        self.id = uuid.uuid4().hex
        self.token_ids = token_ids
        self.params = params
        self.block_size = block_size
        self.block_table: list[int] = []

    def __len__(self):
        return len(self.token_ids)

    def __str__(self):
        return (
            f"Sequence(id={self.id}, token_ids={self.token_ids}, params={self.params})"
        )

    def num_blocks(self) -> int:
        """Return the number of blocks needed for this sequence."""
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    def block(self, idx: int) -> list[int]:
        """Return the tokens in the specified block."""
        return self.token_ids[idx * self.block_size : (idx + 1) * self.block_size]


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    hash: int = -1
    token_ids: list[int] = None

    def __len__(self):
        return len(self.token_ids)

    def __str__(self):
        return f"Block(block_id={self.block_id}, ref_count={self.ref_count}, hash={self.hash}, token_ids={self.token_ids})"

    def __eq__(self, other):
        if not isinstance(other, Block):
            return False
        return self.block_id == other.block_id

    def reset_states(self):
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.stats = {
            "cache_hit": 0,
            "cache_miss": 0,
        }

    @classmethod
    def _hash_tokens(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_free_block(self, hash: int, token_ids: list[int]):
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        # reset block
        block.hash = hash
        block.token_ids = token_ids
        block.ref_count = 1
        self.used_block_ids.add(block_id)
        # only add to hash map if it's the current block for that hash
        if hash != -1:
            self.hash_to_block_id[hash] = block_id
        return block

    def _deallocate_block(self, block_id: int) -> Block:
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        block = self.blocks[block_id]
        if block.hash != -1:
            # Only delete from hash map if it's the current block for that hash
            if self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]
        block.reset_states()
        return block

    def allocate(self, seq: Sequence):
        h = -1
        for i in range(seq.num_blocks()):
            token_ids = seq.block(i)
            h = self._hash_tokens(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            # If the block is already in the cache, reuse it
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                self.stats["cache_miss"] += 1
                block = self._allocate_free_block(h, token_ids)
            else:
                self.stats["cache_hit"] += 1
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # block is in free list
                    block = self._allocate_free_block(h, token_ids)

            seq.block_table.append(block.block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.block_table.clear()


@dataclass
class CompletionOutput:
    """Output of a single completion."""

    index: int
    text: str
    token_ids: list[int]
    finish_reason: str | None = None
    stop_reason: int | str | None = None

    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, text={self.text!r}, "
            f"token_ids={self.token_ids}, finish_reason={self.finish_reason}, "
            f"stop_reason={self.stop_reason})"
        )


class RequestOutput:
    """Output of a generation request, containing multiple completions."""

    def __init__(self, request_id: str, prompt: str, outputs: list[CompletionOutput]):
        self.request_id = request_id
        self.prompt = prompt
        self.outputs = outputs

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id!r}, "
            f"prompt={self.prompt!r}, outputs={self.outputs!r})"
        )


class BaseLLMEngine(ABC):
    """Base class for LLM engines."""

    @abstractmethod
    def generate(
        self, prompt: list[str], sampling_params: SamplingParams
    ) -> list[RequestOutput]:
        """Generate outputs for the given prompts."""
        raise NotImplementedError(f"{type(self)} is not implemented")


class AsyncLLMEngine(BaseLLMEngine):
    def __init__(self, model: str, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)

    def generate(
        self, prompts: list[str], sampling_params: SamplingParams
    ) -> list[RequestOutput]:
        async def _collect_results():
            results = []
            async for output in self.generate_async(prompts, sampling_params):
                results.extend(output)
            return results

        return asyncio.run(_collect_results())

    async def generate_async(
        self, prompts: list[str], sampling_params: SamplingParams
    ) -> AsyncGenerator[list[RequestOutput], None]:
        yield [
            RequestOutput(
                uuid.uuid4().hex, prompt, [CompletionOutput(0, "Hello World", [])]
            )
            for prompt in prompts
        ]


class LLM(AsyncLLMEngine):
    pass
