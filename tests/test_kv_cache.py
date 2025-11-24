import torch
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the current directory to sys.path to import tinyvllm
sys.path.append(os.getcwd())

from tinyvllm import paged_attention_forward, Context, set_context, reset_context

class MockAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.num_kv_heads = 2
        self.head_dim = 64
        self.q_proj = torch.nn.Linear(256, 256)
        self.k_proj = torch.nn.Linear(256, 128)
        self.v_proj = torch.nn.Linear(256, 128)
        self.o_proj = torch.nn.Linear(256, 256)
        self.layer_id = 0
        self.original_forward = MagicMock()

class TestKVCache(unittest.TestCase):
    def setUp(self):
        self.module = MockAttention()
        self.module.forward = paged_attention_forward.__get__(self.module, MockAttention)
        
        # Mock KV cache
        self.kv_cache = torch.zeros(2, 1, 10, 16, 2, 64) # [2, layers, blocks, block_size, heads, dim]
        torch.tinyvllm_kv_cache = self.kv_cache
        
    def tearDown(self):
        reset_context()
        if hasattr(torch, "tinyvllm_kv_cache"):
            del torch.tinyvllm_kv_cache

    def test_paged_attention_forward_prefill(self):
        # Prefill: slot_mapping is provided, block_tables is None
        b, s = 1, 5
        hidden_states = torch.randn(b, s, 256)
        slot_mapping = torch.arange(s, dtype=torch.int32)
        set_context(is_prefill=True, slot_mapping=slot_mapping, context_lens=torch.tensor([s]))
        
        # Mock position_embeddings in kwargs
        cos = torch.randn(s, 64)
        sin = torch.randn(s, 64)
        
        with patch('transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb', return_value=(torch.randn(b, 4, s, 64), torch.randn(b, 2, s, 64))):
            out, _ = self.module.forward(hidden_states, position_embeddings=(cos, sin))
            
        self.assertEqual(out.shape, (b, s, 256))
        # Check if KV cache was updated (simplified check)
        self.assertFalse(torch.all(self.kv_cache == 0))

    def test_paged_attention_forward_decode(self):
        # Decode: slot_mapping and block_tables provided
        b, s = 1, 1
        hidden_states = torch.randn(b, s, 256)
        slot_mapping = torch.tensor([5], dtype=torch.int32) # 6th token
        context_lens = torch.tensor([6], dtype=torch.int32)
        block_tables = torch.zeros(b, 10, dtype=torch.int32) # block 0
        
        # Pre-fill some data in KV cache for block 0
        self.kv_cache[0, 0, 0, :5, :, :] = torch.randn(5, 2, 64) # past 5 tokens
        self.kv_cache[1, 0, 0, :5, :, :] = torch.randn(5, 2, 64)
        
        set_context(is_prefill=False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        
        # Mock position_embeddings
        cos = torch.randn(s, 64)
        sin = torch.randn(s, 64)
        
        with patch('transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb', return_value=(torch.randn(b, 4, s, 64), torch.randn(b, 2, s, 64))):
            out, _ = self.module.forward(hidden_states, position_embeddings=(cos, sin))
            
        self.assertEqual(out.shape, (b, s, 256))
        # Check if 6th token was stored
        self.assertFalse(torch.all(self.kv_cache[0, 0, 0, 5, :, :] == 0))

    def test_paged_attention_forward_decode_multi_seq(self):
        # Decode: Multiple sequences in batch
        b, s = 2, 1
        hidden_states = torch.randn(b, s, 256)
        # Seq 1: length 6 (index 5), block 0
        # Seq 2: length 3 (index 2), block 1
        slot_mapping = torch.tensor([5, 16 + 2], dtype=torch.int32) # block_size=16
        context_lens = torch.tensor([6, 3], dtype=torch.int32)
        block_tables = torch.zeros(b, 10, dtype=torch.int32)
        block_tables[0, 0] = 0
        block_tables[1, 0] = 1
        
        # Pre-fill KV cache
        self.kv_cache[0, 0, 0, :5, :, :] = torch.randn(5, 2, 64) # Seq 1 past
        self.kv_cache[1, 0, 0, :5, :, :] = torch.randn(5, 2, 64)
        self.kv_cache[0, 0, 1, :2, :, :] = torch.randn(2, 2, 64) # Seq 2 past
        self.kv_cache[1, 0, 1, :2, :, :] = torch.randn(2, 2, 64)
        
        set_context(is_prefill=False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        
        # Mock position_embeddings
        cos = torch.randn(s, 64)
        sin = torch.randn(s, 64)
        
        with patch('transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb', return_value=(torch.randn(b, 4, s, 64), torch.randn(b, 2, s, 64))):
            out, _ = self.module.forward(hidden_states, position_embeddings=(cos, sin))
            
        self.assertEqual(out.shape, (b, s, 256))
        # Check if tokens were stored
        self.assertFalse(torch.all(self.kv_cache[0, 0, 0, 5, :, :] == 0))
        self.assertFalse(torch.all(self.kv_cache[0, 0, 1, 2, :, :] == 0))

    def test_paged_attention_forward_decode_multi_block(self):
        # Decode: Sequence spanning multiple blocks
        b, s = 1, 1
        hidden_states = torch.randn(b, s, 256)
        # Seq 1: length 17 (index 16), block 0 (full) and block 1 (1 token)
        # Block size is 16. So index 16 is block 1, offset 0.
        slot_mapping = torch.tensor([16], dtype=torch.int32) 
        context_lens = torch.tensor([17], dtype=torch.int32)
        block_tables = torch.zeros(b, 10, dtype=torch.int32)
        block_tables[0, 0] = 0
        block_tables[0, 1] = 1
        
        # Pre-fill KV cache for block 0 (full)
        self.kv_cache[0, 0, 0, :, :, :] = torch.randn(16, 2, 64)
        self.kv_cache[1, 0, 0, :, :, :] = torch.randn(16, 2, 64)
        
        set_context(is_prefill=False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        
        # Mock position_embeddings
        cos = torch.randn(s, 64)
        sin = torch.randn(s, 64)
        
        with patch('transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb', return_value=(torch.randn(b, 4, s, 64), torch.randn(b, 2, s, 64))):
            out, _ = self.module.forward(hidden_states, position_embeddings=(cos, sin))
            
        self.assertEqual(out.shape, (b, s, 256))
        # Check if 17th token was stored in block 1, offset 0
        self.assertFalse(torch.all(self.kv_cache[0, 0, 1, 0, :, :] == 0))

if __name__ == "__main__":
    unittest.main()
