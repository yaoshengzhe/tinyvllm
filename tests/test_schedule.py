import unittest
from unittest.mock import MagicMock, patch
from tinyvllm import Scheduler, Config, Sequence, BlockManager

class TestScheduler(unittest.TestCase):
    def setUp(self):
        # Mock Config to avoid directory checks and other post_init logic
        self.config = MagicMock()
        self.config.max_num_seqs = 5
        self.config.max_num_batched_tokens = 100
        self.config.num_kvcache_blocks = 10
        self.config.kvcache_block_size = 4
        self.scheduler = Scheduler(self.config)

    def test_add_sequence(self):
        seq = Sequence(token_ids=[1, 2, 3], block_size=4)
        self.scheduler.add(seq)
        self.assertEqual(len(self.scheduler.waiting), 1)
        self.assertEqual(len(self.scheduler.running), 0)

    def test_schedule_new_sequence_success(self):
        seq = Sequence(token_ids=[1, 2, 3], block_size=4)
        self.scheduler.add(seq)
        scheduled, is_prefill = self.scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(len(self.scheduler.waiting), 0)
        self.assertEqual(len(self.scheduler.running), 1)
        self.assertEqual(len(seq.block_table), 1)

    def test_schedule_new_sequence_insufficient_memory(self):
        # Fill up memory
        self.scheduler.block_manager.free_block_ids.clear() # Force OOM
        seq = Sequence(token_ids=[1, 2, 3], block_size=4)
        self.scheduler.add(seq)
        scheduled, is_prefill = self.scheduler.schedule()
        self.assertFalse(is_prefill) # Returns False because scheduled_seqs is empty and falls through to generation
        self.assertEqual(scheduled, [])
        self.assertEqual(len(self.scheduler.waiting), 1)
        self.assertEqual(len(self.scheduler.running), 0)

    def test_schedule_generation_success(self):
        seq = Sequence(token_ids=[1, 2, 3], block_size=4)
        self.scheduler.add(seq)
        self.scheduler.schedule() # Prefill
        
        # Add token to seq to simulate generation
        seq.token_ids.append(4) 
        # Now it has 4 tokens, still 1 block. Next append will require new block if we add 5th token.
        
        scheduled, is_prefill = self.scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(len(seq.block_table), 1) # Still 1 block, but now full

        seq.token_ids.append(5) # Needs new block
        scheduled, is_prefill = self.scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(len(seq.block_table), 2)

    def test_schedule_preemption_self(self):
        # Setup: 1 block available. 
        # Seq1 takes 1 block.
        self.scheduler.block_manager = BlockManager(num_blocks=1, block_size=4)
        seq1 = Sequence(token_ids=[1, 2, 3, 4], block_size=4)
        self.scheduler.add(seq1)
        self.scheduler.schedule() # Prefill, takes the only block
        
        # Now seq1 needs a block to append, but none available.
        seq1.token_ids.append(5) 
        # Since seq1 is the only one running, it should preempt itself.
        scheduled, is_prefill = self.scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [])
        self.assertEqual(len(self.scheduler.running), 0)
        self.assertEqual(len(self.scheduler.waiting), 1)
        self.assertEqual(self.scheduler.waiting[0], seq1)
        self.assertEqual(len(seq1.block_table), 0) # Deallocated

    def test_schedule_preemption_other(self):
        # Setup: 2 blocks available.
        self.scheduler.block_manager = BlockManager(num_blocks=2, block_size=4)
        seq1 = Sequence(token_ids=[1, 2, 3, 4], block_size=4)
        seq2 = Sequence(token_ids=[5, 6, 7, 8], block_size=4)
        self.scheduler.add(seq1)
        self.scheduler.add(seq2)
        self.scheduler.schedule() # Prefill both, uses 2 blocks.
        
        self.assertEqual(len(self.scheduler.running), 2)
        
        # seq1 needs a block. seq2 should be preempted.
        # Scheduler order: popleft gives seq1, then seq2.
        # In schedule():
        # popleft gives seq1. can_append(seq1) is False.
        # running has [seq2].
        # preempt(running.pop()) -> preempt(seq2).
        # now can_append(seq1) is True (seq2 freed its block).
        # append_token(seq1).
        
        seq1.token_ids.append(5)
        scheduled, is_prefill = self.scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [seq1])
        self.assertEqual(len(self.scheduler.running), 1)
        self.assertEqual(self.scheduler.running[0], seq1)
        self.assertEqual(len(self.scheduler.waiting), 1)
        self.assertEqual(self.scheduler.waiting[0], seq2)

if __name__ == '__main__':
    unittest.main()
