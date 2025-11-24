import unittest
from unittest.mock import MagicMock, patch
import torch
from tinyvllm import ModelRunner, Config, Sequence, SamplingParams

class TestModelRunner(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.model = "mock_model"
        self.config.kvcache_block_size = 4
        self.config.enforce_eager = True
        self.config.tensor_parallel_size = 1
        self.config.max_num_seqs = 5
        self.config.max_num_batched_tokens = 100
        self.config.max_model_len = 10
        self.config.gpu_memory_utilization = 0.9
        
        self.hf_config = MagicMock()
        self.hf_config.num_attention_heads = 4
        self.hf_config.hidden_size = 16
        self.hf_config.num_hidden_layers = 2
        self.hf_config.torch_dtype = torch.float32
        self.hf_config.vocab_size = 100
        self.config.hf_config = self.hf_config

        # Mock AutoModelForCausalLM
        self.patcher_model = patch('transformers.AutoModelForCausalLM.from_pretrained')
        self.mock_from_pretrained = self.patcher_model.start()
        self.mock_model_instance = MagicMock()
        self.mock_from_pretrained.return_value = self.mock_model_instance
        
        # Mock torch.cuda.is_available
        self.patcher_cuda = patch('torch.cuda.is_available', return_value=False)
        self.patcher_cuda.start()

        self.runner = ModelRunner(self.config, 0, [])

    def tearDown(self):
        self.patcher_model.stop()
        self.patcher_cuda.stop()

    def test_init(self):
        self.assertEqual(self.runner.rank, 0)
        self.assertEqual(self.runner.world_size, 1)
        self.assertFalse(torch.cuda.is_available())

    def test_prepare_prefill(self):
        seq1 = Sequence(1, [1, 2, 3], block_size=4)
        seq1.block_table = [0]
        seq2 = Sequence(2, [4, 5], block_size=4)
        seq2.block_table = [1]
        
        input_ids, positions, _ = self.runner.prepare_prefill([seq1, seq2])
        
        self.assertEqual(input_ids.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(positions.tolist(), [0, 1, 2, 0, 1])
        
        from tinyvllm import get_context
        context = get_context()
        self.assertTrue(context.is_prefill)
        self.assertEqual(context.cu_seqlens_q.tolist(), [0, 3, 5])
        self.assertEqual(context.slot_mapping.tolist(), [0, 1, 2, 4, 5]) # Block 0: slots 0-3, Block 1: slots 4-7

    def test_prepare_decode(self):
        seq1 = Sequence(1, [1, 2, 3, 4], block_size=4)
        seq1.block_table = [0]
        seq2 = Sequence(2, [5, 6, 7], block_size=4)
        seq2.block_table = [1]
        
        input_ids, positions, attention_mask = self.runner.prepare_decode([seq1, seq2])
        
        self.assertEqual(input_ids.tolist(), [[4], [7]])
        self.assertEqual(positions.tolist(), [[3], [2]])
        
        from tinyvllm import get_context
        context = get_context()
        self.assertFalse(context.is_prefill)
        self.assertEqual(context.slot_mapping.tolist(), [3, 6]) # Seq1: block 0, pos 3 -> slot 3. Seq2: block 1, pos 2 -> slot 4+2=6.

    def test_prepare_prefill_complex(self):
        # Seq1: 6 tokens, block_size=4. Blocks 10 (full), 11 (2 tokens).
        seq1 = Sequence(1, [1, 2, 3, 4, 5, 6], block_size=4)
        seq1.block_table = [10, 11] # Using non-zero block IDs to verify mapping
        
        # Seq2: 4 tokens, block_size=4. Block 12 (full).
        seq2 = Sequence(2, [7, 8, 9, 10], block_size=4)
        seq2.block_table = [12]
        
        input_ids, positions, _ = self.runner.prepare_prefill([seq1, seq2])
        
        self.assertEqual(input_ids.tolist(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(positions.tolist(), [0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
        
        from tinyvllm import get_context
        context = get_context()
        self.assertEqual(context.cu_seqlens_q.tolist(), [0, 6, 10])
        # Seq1 slots: Block 10 (0,1,2,3), Block 11 (0,1) -> 10*4+0..3, 11*4+0..1
        # Seq2 slots: Block 12 (0,1,2,3) -> 12*4+0..3
        expected_slots = [
            10*4+0, 10*4+1, 10*4+2, 10*4+3, 11*4+0, 11*4+1,
            12*4+0, 12*4+1, 12*4+2, 12*4+3
        ]
        self.assertEqual(context.slot_mapping.tolist(), expected_slots)

    def test_prepare_decode_boundary(self):
        # Seq1: 4 tokens, block_size=4. Just filled block 10.
        seq1 = Sequence(1, [1, 2, 3, 4], block_size=4)
        seq1.block_table = [10]
        
        # Seq2: 5 tokens, block_size=4. Just started block 12.
        seq2 = Sequence(2, [1, 2, 3, 4, 5], block_size=4)
        seq2.block_table = [11, 12]
        
        input_ids, positions, attention_mask = self.runner.prepare_decode([seq1, seq2])
        
        self.assertEqual(input_ids.tolist(), [[4], [5]])
        self.assertEqual(positions.tolist(), [[3], [4]])
        
        from tinyvllm import get_context
        context = get_context()
        # Seq1: Last token is 4th token (index 3). Block 10, slot 10*4 + 3.
        # Seq2: Last token is 5th token (index 4). Block 12, slot 12*4 + 0.
        expected_slots = [10*4+3, 12*4+0]
        self.assertEqual(context.slot_mapping.tolist(), expected_slots)

    def test_prepare_empty(self):
        input_ids, positions, _ = self.runner.prepare_prefill([])
        self.assertEqual(input_ids.tolist(), [])
        self.assertEqual(positions.tolist(), [])
        
        input_ids, positions, attention_mask = self.runner.prepare_decode([])
        self.assertEqual(input_ids.tolist(), [])
        self.assertEqual(positions.tolist(), [])

    def test_prepare_sample(self):
        seq1 = Sequence(1, [1], params=SamplingParams(temperature=0.5))
        seq2 = Sequence(2, [1], params=SamplingParams(temperature=1.5))
        seq3 = Sequence(3, [1]) # Default temperature 1.0
        
        temperatures = self.runner.prepare_sample([seq1, seq2, seq3])
        self.assertEqual(temperatures.tolist(), [0.5, 1.5, 1.0])

    def test_run_model_fallback(self):
        # Mock model output
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 5, 100) # Prefill output
        self.mock_model_instance.return_value = mock_output
        
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        positions = torch.tensor([0, 1, 2, 0, 1])
        
        logits = self.runner.run_model(input_ids, positions, is_prefill=True)
        self.assertEqual(logits.shape, (1, 5, 100))

if __name__ == '__main__':
    unittest.main()
