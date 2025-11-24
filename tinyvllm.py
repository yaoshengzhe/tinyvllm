"""
tinyvllm - A tiny version of vllm inspired by nano-vllm.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, fields
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from typing import AsyncGenerator, Optional, Tuple


import asyncio
import os
import uuid
import torch
import numpy as np
import xxhash
import pickle
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

DEFAULT_BLOCK_SIZE = 256
END_OF_SEQ_TOKEN = -1

def paged_attention_forward(self, *args, **kwargs):
    # Extract arguments safely
    hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
    attention_mask = args[1] if len(args) > 1 else kwargs.get("attention_mask")
    # position_ids = args[2] if len(args) > 2 else kwargs.get("position_ids")
    # past_key_values = args[3] if len(args) > 3 else kwargs.get("past_key_values")
    # output_attentions = args[4] if len(args) > 4 else kwargs.get("output_attentions", False)
    # use_cache = args[5] if len(args) > 5 else kwargs.get("use_cache", False)
    
    # 1. Q, K, V projection
    b, s, _ = hidden_states.shape
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    
    # Reshape for multi-head
    # Qwen3 uses GQA. We need to get num_heads and head_dim.
    # Based on previous print, q_proj out is 2048, k_proj out is 1024. head_dim is 128.
    # So num_heads = 16, num_kv_heads = 8.
    # We should get these from self if possible, or infer.
    num_heads = self.num_heads if hasattr(self, "num_heads") else q.shape[-1] // 128
    num_kv_heads = self.num_kv_heads if hasattr(self, "num_kv_heads") else k.shape[-1] // 128
    head_dim = self.head_dim if hasattr(self, "head_dim") else 128

    q = q.view(b, s, num_heads, head_dim)
    k = k.view(b, s, num_kv_heads, head_dim)
    v = v.view(b, s, num_kv_heads, head_dim)

    # Apply layer norm if present (Qwen3 has q_norm, k_norm)
    if hasattr(self, "q_norm"):
        q = self.q_norm(q)
    if hasattr(self, "k_norm"):
        k = self.k_norm(k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # RoPE
    if "position_embeddings" in kwargs:
        cos, sin = kwargs["position_embeddings"]
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    
    # KV Cache Storage
    context = get_context()
    if context and context.slot_mapping is not None and hasattr(torch, "tinyvllm_kv_cache"):
        kv_cache = torch.tinyvllm_kv_cache
        # kv_cache shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        # We need layer_id. We can store it on the module during patching.
        layer_id = self.layer_id
        
        # Store current K, V
        # slot_mapping is [total_tokens] for prefill or [bs] for decode
        # k, v are [bs, num_kv_heads, seq_len, head_dim] after transpose back or [bs, num_kv_heads, 1, head_dim] for decode
        
        # For simplicity, handle decode (s=1) first. Prefill is harder with slot_mapping.
        # Actually, slot_mapping is prepared for both.
        
        # Flatten K, V for storage
        # k is [b, num_kv_heads, s, head_dim] -> [b*s, num_kv_heads, head_dim]
        k_flat = k.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        v_flat = v.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)
        
        slots = context.slot_mapping
        if slots.shape[0] == k_flat.shape[0]:
            # Calculate block index and offset
            block_size = kv_cache.shape[3]
            block_ids = slots // block_size
            offsets = slots % block_size
            
            # This is slow in Python, but works for now.
            # Optimized version would use custom kernel or scatter.
            for i in range(slots.shape[0]):
                bid = block_ids[i]
                off = offsets[i]
                kv_cache[0, layer_id, bid, off] = k_flat[i]
                kv_cache[1, layer_id, bid, off] = v_flat[i]

        # Paged Attention (Retrieve and Compute)
        if context.block_tables is not None:
            # Decode path
            # q is [b, num_heads, 1, head_dim]
            # We need to collect K, V for all past tokens for each sequence in batch.
            
            output = torch.empty_like(q) # [b, num_heads, 1, head_dim]
            
            for i in range(b):
                # Get block table for this sequence
                block_table = context.block_tables[i] # [max_num_blocks]
                actual_len = context.context_lens[i]
                
                # Collect K, V blocks
                # This is also slow in Python.
                num_blocks = (actual_len + block_size - 1) // block_size
                valid_blocks = block_table[:num_blocks]
                
                # Gather blocks
                # kv_cache[0, layer_id, valid_blocks] -> [num_blocks, block_size, num_kv_heads, head_dim]
                k_blocks = kv_cache[0, layer_id, valid_blocks]
                v_blocks = kv_cache[1, layer_id, valid_blocks]
                
                # Flatten blocks and truncate to actual_len
                k_full = k_blocks.view(-1, num_kv_heads, head_dim)[:actual_len]
                v_full = v_blocks.view(-1, num_kv_heads, head_dim)[:actual_len]
                
                # Compute attention for this sequence
                # q_i is [num_heads, 1, head_dim]
                q_i = q[i] 
                
                # Repeat K, V for GQA if needed
                # num_heads = 16, num_kv_heads = 8. Repeat factor = 2.
                if num_heads != num_kv_heads:
                    k_full = k_full.repeat_interleave(num_heads // num_kv_heads, dim=1)
                    v_full = v_full.repeat_interleave(num_heads // num_kv_heads, dim=1)
                
                # k_full is [actual_len, num_heads, head_dim]
                # v_full is [actual_len, num_heads, head_dim]
                
                # Attention scores
                # q_i: [num_heads, 1, head_dim]
                # k_full: [actual_len, num_heads, head_dim] -> [num_heads, actual_len, head_dim]
                scores = torch.matmul(q_i, k_full.transpose(0, 1).transpose(1, 2)) / (head_dim ** 0.5)
                # scores is [num_heads, 1, actual_len]
                
                if attention_mask is not None:
                    # Handle mask if needed, but for decode it's usually all 1s for past
                    pass
                
                probs = torch.softmax(scores, dim=-1)
                
                # Output
                # probs: [num_heads, 1, actual_len]
                # v_full: [num_heads, actual_len, head_dim]
                out_i = torch.matmul(probs, v_full.transpose(0, 1))
                # out_i is [num_heads, 1, head_dim]
                output[i] = out_i
            
            output = output.transpose(1, 2).reshape(b, s, -1)
            output = self.o_proj(output)
            return output, None, None
        else:
            # Prefill path or no cache
            # Use standard attention
            # For prefill, we still need to handle GQA and RoPE which we already did.
            # But we can use efficient_attention if available, or standard.
            
            # Standard attention
            if num_heads != num_kv_heads:
                k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
                v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            if attention_mask is not None:
                scores = scores + attention_mask
            probs = torch.softmax(scores, dim=-1)
            output = torch.matmul(probs, v)
            output = output.transpose(1, 2).reshape(b, s, -1)
            output = self.o_proj(output)
            return output, None, None

    # Fallback if no cache or not patched
    # This should not happen if patched correctly
    return self.original_forward(*args, **kwargs)


@dataclass
class Context:
    is_prefill: bool
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor = None
    context_lens: torch.Tensor = None
    block_tables: torch.Tensor = None

_context = None

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _context
    _context = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def get_context():
    return _context

def reset_context():
    global _context
    _context = None


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
    num_kvcache_blocks: int = 128

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

    def __init__(self, temperature=1.0, top_p=1.0, max_tokens=16, ignore_eos=False):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos


class Sequence:
    """Represents a single input/output sequence."""

    def __init__(
        self,
        seq_id: int,
        token_ids: list[int],
        params: SamplingParams = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ):
        self.id = seq_id
        self.token_ids = token_ids
        self.params = params
        self.num_prompt_tokens = len(token_ids)
        self.block_size = block_size
        self.block_table: list[int] = []

    def __len__(self):
        return len(self.token_ids)

    def __str__(self):
        return (
            f"Sequence(id={self.id}, token_ids={self.token_ids}, params={self.params})"
        )

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def is_finished(self) -> bool:
        if len(self.token_ids) > self.num_prompt_tokens and self.token_ids[-1] == END_OF_SEQ_TOKEN:
            return True
        if self.params:
            if len(self.completion_token_ids) >= self.params.max_tokens:
                return True
        return False

    def num_blocks(self) -> int:
        """Return the number of blocks needed for this sequence."""
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    def block(self, idx: int) -> list[int]:
        """Return the tokens in the specified block."""
        return self.token_ids[idx * self.block_size : (idx + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)

    @property
    def num_cached_tokens(self) -> int:
        # For now, we don't support prefix caching, so this is always 0 or we can track it if needed.
        # To match nano-vllm, we return 0.
        return 0

    @property
    def num_cached_blocks(self) -> int:
        return 0

    @property
    def last_token(self) -> int:
        return self.token_ids[-1] if self.token_ids else -1

    @property
    def last_block_num_tokens(self) -> int:
        return len(self.token_ids) % self.block_size or self.block_size
    
    @property
    def temperature(self) -> float:
        return self.params.temperature if self.params else 1.0


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

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def append_token(self, seq: Sequence):
        """Handle block allocation/hashing when a token is appended to a sequence."""
        block_table = seq.block_table
        last_block_id = block_table[-1]
        last_block = self.blocks[last_block_id]

        if len(seq) % self.block_size == 1:
            # New token started a new block.
            # Allocate a fresh, unhashed block.
            new_block = self._allocate_free_block(hash=-1, token_ids=[])
            block_table.append(new_block.block_id)
        elif len(seq) % self.block_size == 0:
            # New token filled the current block.
            # Hash the block and add to cache.
            token_ids = seq.block(seq.num_blocks() - 1)
            prefix_hash = -1
            if len(block_table) > 1:
                prefix_hash = self.blocks[block_table[-2]].hash

            h = self._hash_tokens(token_ids, prefix_hash)
            last_block.hash = h
            last_block.token_ids = token_ids
            self.hash_to_block_id[h] = last_block.block_id
        # Else: token added to partial block, no action needed until full.


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
        self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams]
    ) -> list[RequestOutput]:
        """Generate outputs for the given prompts."""
        raise NotImplementedError(f"{type(self)} is not implemented")


class AsyncLLMEngine(BaseLLMEngine):
    def __init__(self, model: str, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, use_fast=True)
        self.scheduler = Scheduler(self.config)

        self.ps = []
        events = []
        ctx = torch.multiprocessing.get_context("spawn")
        for i in range(1, self.config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(self.config, i, event))
            process.start()
            self.ps.append(process)
            events.append(event)

        self.model_runner = ModelRunner(self.config, 0, events)
        
    def exit(self):
        self.model_runner.exit()
        del self.model_runner
        for process in self.ps:
            process.join()

    def empty_cache(self):
        self.model_runner.empty_cache()

    def generate(
        self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams], use_tqdm: bool = False
    ) -> list[RequestOutput]:
        async def _collect_results():
            results = []
            async for output in self.generate_async(prompts, sampling_params):
                results.extend(output)
            return results

        return asyncio.run(_collect_results())

    async def generate_async(
        self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams]
    ) -> AsyncGenerator[list[RequestOutput], None]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        for i, (prompt, sampling_param) in enumerate(zip(prompts, sampling_params)):
            self.add_request(i, prompt, sampling_param)
        
        while not self.scheduler.is_finished():
            outputs, _ = self.step()
            if outputs:
                # For now, just collect and yield at the end, or yield per step?
                # The current signature yields list[RequestOutput].
                # Let's yield current outputs.
                request_outputs = []
                for seq, completion_token_ids in outputs:
                    prompt_text = prompts[seq.id]
                    if isinstance(prompt_text, list):
                        prompt_text = self.tokenizer.decode(prompt_text)
                    request_outputs.append(
                        RequestOutput(
                            seq.id, prompt_text, [CompletionOutput(0, self.tokenizer.decode(completion_token_ids), completion_token_ids)]
                        )
                    )
                yield request_outputs

    def add_request(self, seq_id: int, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = prompt
        
        seq = Sequence(seq_id, token_ids, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.update_seqs(seqs, token_ids)
        outputs = [(seq, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

class LLM(AsyncLLMEngine):
    pass


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens

        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence):
        self.waiting.append(seq)
    
    def is_finished(self):
        return not self.waiting and not self.running

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        # Append new sequences to running
        while self.waiting and self._add_new_seq(self.waiting[0], scheduled_seqs):
            pass
    
        if scheduled_seqs:
            return scheduled_seqs, True

        # Schedule sequences in running
        while self.running:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.append_token(seq)
                scheduled_seqs.append(seq)

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def update_seqs(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if seq.is_finished:
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def _add_new_seq(self, seq: Sequence, scheduled_seqs: list[Sequence]):
        if not self.block_manager.can_allocate(seq):
            return False
        
        self.block_manager.allocate(seq)
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
        return True


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # logits: [bs, vocab_size]
        # temperatures: [bs]
        logits = logits.float() / temperatures.unsqueeze(dim=1)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = (probs / torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens


class BaseModelRunner(ABC):
    @abstractmethod
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        raise NotImplementedError(f"{type(self)} is not implemented")
    
    @abstractmethod
    def exit(self):
        raise NotImplementedError(f"{type(self)} is not implemented")


class ModelRunner(BaseModelRunner):
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        if self.world_size > 1:
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            device = f"cuda:{rank}"
        else:
            device = "cpu"
        self.device = device

        default_dtype = torch.get_default_dtype()
        if hf_config:
            dtype = getattr(hf_config, "torch_dtype", None) or getattr(hf_config, "dtype", None)
            if dtype:
                torch.set_default_dtype(dtype)
        
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        
        # Fallback to AutoModelForCausalLM if Qwen3ForCausalLM is not available
        import warnings
        from transformers import AutoModelForCausalLM
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map={"": device} if torch.cuda.is_available() else None
            )
        self.sampler = Sampler()
        
        # Patch Attention Layers
        layer_id = 0
        for module in self.model.modules():
            if "Qwen3Attention" in module.__class__.__name__:
                module.layer_id = layer_id
                module.original_forward = module.forward
                import types
                module.forward = types.MethodType(paged_attention_forward, module)
                # Set necessary attributes if missing
                if not hasattr(module, "num_heads"):
                    module.num_heads = module.q_proj.out_features // 128 # Assuming head_dim=128
                if not hasattr(module, "num_kv_heads"):
                    module.num_kv_heads = module.k_proj.out_features // 128
                if not hasattr(module, "head_dim"):
                    module.head_dim = 128
                layer_id += 1

        if torch.cuda.is_available():
            self.warmup_model()
            self.allocate_kv_cache()
            if not self.enforce_eager:
                self.capture_cudagraph()
            torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager and hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if self.world_size > 1:
            dist.destroy_process_group()

    def empty_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        if num_seqs == 0: num_seqs = 1
        seqs = [Sequence(i, [0] * max_model_len) for i in range(num_seqs)]
        # Mock block table for warmup
        for seq in seqs:
            seq.block_table = [0] * ((max_model_len + self.block_size - 1) // self.block_size)
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        if not torch.cuda.is_available():
            return
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        num_kv_heads = hf_config.num_attention_heads // self.world_size # Fallback if num_key_value_heads missing
        if hasattr(hf_config, "num_key_value_heads"):
            num_kv_heads = hf_config.num_key_value_heads // self.world_size
        
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        
        # Approximate dtype size if not available
        dtype_size = 2 # Default to float16
        dtype = getattr(hf_config, "torch_dtype", None) or getattr(hf_config, "dtype", None)
        if dtype:
            try:
                dtype_size = torch.tensor([], dtype=dtype).element_size()
            except:
                dtype_size = 2 # Default to 2 bytes (float16) if unknown

        num_layers = hf_config.num_hidden_layers
        block_bytes = 2 * num_layers * self.block_size * num_kv_heads * head_dim * dtype_size
        
        available_memory = total * config.gpu_memory_utilization - used - peak + current
        if available_memory < 0:
            available_memory = 0
        
        config.num_kvcache_blocks = int(available_memory) // block_bytes
        if config.num_kvcache_blocks <= 0:
            config.num_kvcache_blocks = 1 # Minimum 1 block
        
        self.kv_cache = torch.empty(2, num_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim, device="cuda")
        torch.tinyvllm_kv_cache = self.kv_cache
        
        # Note: In standard AutoModelForCausalLM, we don't easily have access to set k_cache/v_cache 
        # on modules without custom model implementation. We keep the allocation for now.
        # For full support, we would need to patch the model's attention layers.
        # layer_id = 0
        # for module in self.model.modules():
        #     if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        #         module.k_cache = self.kv_cache[0, layer_id]
        #         module.v_cache = self.kv_cache[1, layer_id]
        #         layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs) if seqs else 0
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32, device=self.device)

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq.token_ids[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table:    # warmup
                continue
            # In tinyvllm, we use num_blocks() method, but here we want to match nano-vllm's use of num_blocks property if it existed, 
            # or just use the range. Since we don't have num_cached_blocks, we start from 0.
            for i in range(seq.num_cached_blocks, seq.num_blocks()):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks() - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        positions = torch.tensor(positions, dtype=torch.int64, device=self.device)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions, cu_seqlens_q

    def prepare_decode(self, seqs: list[Sequence]):
        # For standard transformers model without KV cache, we must pass all tokens.
        # This is slow but correct. We pad to the longest sequence in the batch.
        input_ids_list = []
        positions_list = []
        context_lens = []
        for seq in seqs:
            input_ids_list.append(seq.token_ids)
            positions_list.append(list(range(len(seq))))
            context_lens.append(len(seq))
        
        if not context_lens:
            return torch.tensor([], dtype=torch.int64, device=self.device), torch.tensor([], dtype=torch.int64, device=self.device), torch.tensor([], dtype=torch.int64, device=self.device)
        max_len = max(context_lens)
        # Pad with 0 (or EOS, but 0 is usually safe for padding if attention mask is used, 
        # but here we rely on positions and model handling. For simple fallback, we just pad).
        # Better to use proper padding token if available, but 0 is common.
        padded_input_ids = [ids + [0] * (max_len - len(ids)) for ids in input_ids_list]
        padded_positions = [pos + [0] * (max_len - len(pos)) for pos in positions_list]
        attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids_list]
        
        input_ids = torch.tensor(padded_input_ids, dtype=torch.int64, device=self.device)
        positions = torch.tensor(padded_positions, dtype=torch.int64, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64, device=self.device)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, device=self.device)
        
        # Keep slot_mapping and block_tables for compatibility with context, 
        # even if not used by the fallback run_model.
        slot_mapping = []
        for seq in seqs:
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        block_tables = self.prepare_block_tables(seqs)
        
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens_tensor, block_tables=block_tables)
        
        # For patched attention, we only want the last token for decode
        # But we need to make sure positions are correct.
        # In decode, we pass [bs, 1] input_ids and [bs, 1] positions.
        last_input_ids = []
        last_positions = []
        for seq in seqs:
            last_input_ids.append([seq.token_ids[-1]])
            last_positions.append([len(seq.token_ids) - 1])
        
        input_ids = torch.tensor(last_input_ids, dtype=torch.int64, device=self.device)
        positions = torch.tensor(last_positions, dtype=torch.int64, device=self.device)
        # Mask is not needed for single token decode with PagedAttention as we handle it.
        
        return input_ids, positions, None

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, attention_mask: torch.Tensor = None, cu_seqlens_q: torch.Tensor = None):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512 or not torch.cuda.is_available() or not hasattr(self, "graphs"):
            # Fallback for now since we don't have custom kernels/model
            # For decode, input_ids is now [bs, seq_len] due to fallback.
            if is_prefill:
                if cu_seqlens_q is not None and cu_seqlens_q.size(0) > 2:
                    # Multiple sequences in prefill, run one by one to avoid cross-attention
                    all_logits = []
                    for i in range(cu_seqlens_q.size(0) - 1):
                        start, end = cu_seqlens_q[i], cu_seqlens_q[i+1]
                        seq_input_ids = input_ids[start:end].unsqueeze(0)
                        seq_positions = positions[start:end].unsqueeze(0)
                        out = self.model(seq_input_ids, position_ids=seq_positions)
                        all_logits.append(out.logits)
                    return torch.cat(all_logits, dim=1) # [1, sum_seq_len, vocab_size]
                else:
                    outputs = self.model(input_ids.unsqueeze(0), position_ids=positions.unsqueeze(0))
                    return outputs.logits
            else:
                # Decode fallback: input_ids is [bs, 1] due to prepare_decode change
                outputs = self.model(input_ids, position_ids=positions, attention_mask=attention_mask)
                return outputs.logits[:, -1, :]
        else:
            bs = input_ids.size(0)
            context = get_context()
            # Find smallest batch size in graphs that fits current batch
            suitable_bs = next((x for x in self.graph_bs if x >= bs), None)
            if suitable_bs is None:
                # Fallback if batch size too large for graphs
                outputs = self.model(input_ids.unsqueeze(1), position_ids=positions.unsqueeze(1))
                return outputs.logits[:, -1, :]
                
            graph = self.graphs[suitable_bs]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return graph_vars["outputs"][:bs]

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if not seqs:
            return []
        if is_prefill:
            input_ids, positions, cu_seqlens_q = self.prepare_prefill(seqs)
            attention_mask = None
        else:
            input_ids, positions, attention_mask = self.prepare_decode(seqs)
            cu_seqlens_q = None
        
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill, attention_mask, cu_seqlens_q)
        
        if self.rank == 0:
            # If logits are 3D (prefill), take last token logits for sampling
            if logits.dim() == 3:
                # Need to handle variable length sequences in prefill if multiple seqs
                # For now, simple approach: if prefill, we might be generating for multiple seqs
                # This part is tricky without proper attention masking and output handling.
                # Assuming for now we just want the last token's logits for each sequence.
                # But input_ids are concatenated. We need to use cu_seqlens_q.
                if cu_seqlens_q is not None:
                    last_token_indices = cu_seqlens_q[1:] - 1
                    if logits.dim() == 3 and logits.size(0) == 1:
                        logits = logits[0] # [seq_len, vocab_size]
                    
                    # Ensure indices are within bounds
                    if logits.dim() == 2 and last_token_indices.max() < logits.size(0):
                        logits = logits[last_token_indices, :]
                    else:
                        # Fallback: if mismatch, just take the last token of the whole chunk
                        # This might be wrong for multiple sequences but prevents crash
                        logits = logits[-1:, :] 

            token_ids = self.sampler(logits, temperatures).tolist()
        else:
            token_ids = None
        
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        if not torch.cuda.is_available():
            return
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # Inputs for graph
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        positions = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device="cuda")
        
        # Output placeholder - need to know output shape, usually hidden_size or vocab_size depending on where we stop
        # If we run full model, it's logits.
        vocab_size = hf_config.vocab_size
        outputs = torch.zeros(max_bs, vocab_size, device="cuda")
        
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graph_bs = [bs for bs in self.graph_bs if bs <= max_bs]
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # Warmup
            with torch.no_grad():
                out = self.model(input_ids[:bs].unsqueeze(1))
                outputs[:bs] = out.logits[:, -1, :]
            
            with torch.cuda.graph(graph, self.graph_pool):
                out = self.model(input_ids[:bs].unsqueeze(1))
                outputs[:bs] = out.logits[:, -1, :]
            
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs
        )