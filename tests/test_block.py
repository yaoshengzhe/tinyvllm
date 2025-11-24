import unittest
from collections import deque
from tinyvllm import Block, BlockManager, Sequence, SamplingParams, DEFAULT_BLOCK_SIZE

class TestBlock(unittest.TestCase):
    def test_block_init(self):
        block = Block(block_id=1)
        self.assertEqual(block.block_id, 1)
        self.assertEqual(block.ref_count, 0)
        self.assertEqual(block.hash, -1)
        self.assertIsNone(block.token_ids)

    def test_block_len(self):
        block = Block(block_id=1, token_ids=[1, 2, 3])
        self.assertEqual(len(block), 3)

    def test_block_str(self):
        block = Block(block_id=1, token_ids=[1, 2])
        self.assertIn("Block(block_id=1", str(block))
        self.assertIn("token_ids=[1, 2]", str(block))

    def test_block_reset(self):
        block = Block(block_id=1, ref_count=5, hash=123, token_ids=[1, 2])
        block.reset_states()
        self.assertEqual(block.ref_count, 0)
        self.assertEqual(block.hash, -1)
        self.assertEqual(block.token_ids, [])

class TestBlockManager(unittest.TestCase):
    def setUp(self):
        self.block_size = 4
        self.num_blocks = 10
        self.bm = BlockManager(num_blocks=self.num_blocks, block_size=self.block_size)

    def test_init(self):
        self.assertEqual(self.bm.num_blocks, self.num_blocks)
        self.assertEqual(self.bm.block_size, self.block_size)
        self.assertEqual(len(self.bm.blocks), self.num_blocks)
        self.assertEqual(len(self.bm.free_block_ids), self.num_blocks)
        self.assertEqual(len(self.bm.used_block_ids), 0)

    def test_hash_tokens(self):
        tokens = [1, 2, 3, 4]
        h1 = BlockManager._hash_tokens(tokens)
        h2 = BlockManager._hash_tokens(tokens)
        self.assertEqual(h1, h2)
        
        h3 = BlockManager._hash_tokens(tokens, prefix=h1)
        self.assertNotEqual(h1, h3)

    def test_allocate_new_blocks(self):
        seq = Sequence(1, token_ids=[1, 2, 3, 4, 5], block_size=self.block_size)
        self.bm.allocate(seq)
        
        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(self.bm.stats["cache_miss"], 2)
        self.assertEqual(self.bm.stats["cache_hit"], 0)
        self.assertEqual(len(self.bm.used_block_ids), 2)
        
        block0 = self.bm.blocks[seq.block_table[0]]
        block1 = self.bm.blocks[seq.block_table[1]]
        self.assertEqual(block0.token_ids, [1, 2, 3, 4])
        self.assertEqual(block1.token_ids, [5])
        self.assertEqual(block0.ref_count, 1)
        self.assertEqual(block1.ref_count, 1)

    def test_allocate_cached_blocks(self):
        seq1 = Sequence(1, token_ids=[1, 2, 3, 4], block_size=self.block_size)
        self.bm.allocate(seq1)
        
        seq2 = Sequence(2, token_ids=[1, 2, 3, 4], block_size=self.block_size)
        self.bm.allocate(seq2)
        
        self.assertEqual(self.bm.stats["cache_miss"], 1)
        self.assertEqual(self.bm.stats["cache_hit"], 1)
        self.assertEqual(seq1.block_table[0], seq2.block_table[0])
        self.assertEqual(self.bm.blocks[seq1.block_table[0]].ref_count, 2)

    def test_deallocate(self):
        seq = Sequence(1, token_ids=[1, 2, 3, 4], block_size=self.block_size)
        self.bm.allocate(seq)
        block_id = seq.block_table[0]
        
        self.bm.deallocate(seq)
        self.assertEqual(len(seq.block_table), 0)
        self.assertEqual(self.bm.blocks[block_id].ref_count, 0)
        self.assertIn(block_id, self.bm.free_block_ids)
        self.assertNotIn(block_id, self.bm.used_block_ids)

    def test_deallocate_shared_block(self):
        seq1 = Sequence(1, token_ids=[1, 2, 3, 4], block_size=self.block_size)
        self.bm.allocate(seq1)
        seq2 = Sequence(2, token_ids=[1, 2, 3, 4], block_size=self.block_size)
        self.bm.allocate(seq2)
        
        block_id = seq1.block_table[0]
        self.bm.deallocate(seq1)
        
        self.assertEqual(self.bm.blocks[block_id].ref_count, 1)
        self.assertIn(block_id, self.bm.used_block_ids)
        self.assertNotIn(block_id, self.bm.free_block_ids)

    def test_out_of_memory(self):
        # Allocate all blocks
        seq1 = Sequence(1, token_ids=[1] * (self.num_blocks * self.block_size), block_size=self.block_size)
        self.bm.allocate(seq1)
        self.assertEqual(len(self.bm.used_block_ids), self.num_blocks)
        self.assertEqual(len(self.bm.free_block_ids), 0)
        
        # Try to allocate one more block
        seq2 = Sequence(2, token_ids=[1], block_size=self.block_size)
        with self.assertRaises(IndexError):
            self.bm.allocate(seq2)

    def test_partial_blocks_not_shared_if_different(self):
        # Partial blocks (length < block_size) should not be shared if they differ
        seq1 = Sequence(1, token_ids=[1, 2], block_size=self.block_size) # partial
        self.bm.allocate(seq1)
        
        seq2 = Sequence(2, token_ids=[1, 3], block_size=self.block_size) # partial, different
        self.bm.allocate(seq2)
        
        self.assertNotEqual(seq1.block_table[0], seq2.block_table[0])
        self.assertEqual(self.bm.blocks[seq1.block_table[0]].token_ids, [1, 2])
        self.assertEqual(self.bm.blocks[seq2.block_table[0]].token_ids, [1, 3])

    def test_mixed_sharing(self):
        # seq1: [1, 2, 3, 4], [5, 6]
        # seq2: [1, 2, 3, 4], [7, 8]
        seq1 = Sequence(1, token_ids=[1, 2, 3, 4, 5, 6], block_size=self.block_size)
        self.bm.allocate(seq1)
        
        seq2 = Sequence(2, token_ids=[1, 2, 3, 4, 7, 8], block_size=self.block_size)
        self.bm.allocate(seq2)
        
        # First block should be shared
        self.assertEqual(seq1.block_table[0], seq2.block_table[0])
        self.assertEqual(self.bm.blocks[seq1.block_table[0]].ref_count, 2)
        
        # Second block should not be shared (different partial blocks)
        self.assertNotEqual(seq1.block_table[1], seq2.block_table[1])
        self.assertEqual(self.bm.blocks[seq1.block_table[1]].ref_count, 1)
        self.assertEqual(self.bm.blocks[seq2.block_table[1]].ref_count, 1)

    def test_reallocation_after_deallocation(self):
        seq1 = Sequence(1, token_ids=[1, 2, 3, 4], block_size=self.block_size)
        self.bm.allocate(seq1)
        block_id = seq1.block_table[0]
        self.bm.deallocate(seq1)
        
        # Allocate same tokens again, should reuse from cache if still there, 
        # but here it was deallocated and moved to free list.
        # Since it's in free list, it might be reused but needs re-initialization or check.
        # In our current impl, deallocate calls reset_states, so it loses hash and tokens.
        
        seq2 = Sequence(2, token_ids=[1, 2, 3, 4], block_size=self.block_size)
        self.bm.allocate(seq2)
        
        # It should get a block, possibly the same ID if it's LIFO or FIFO, 
        # but it's a cache miss because we wiped it.
        self.assertEqual(self.bm.stats["cache_miss"], 2) # 1 for seq1, 1 for seq2
        # In our implementation, free_block_ids is a deque and we append to the right, pop from left.
        # So it's FIFO. Wait, if we pop from left and append to right, it's a queue.
        # Block 0 was used, then returned to right. If there were 10 blocks (0-9).
        # Initially free: 0, 1, 2, ... 9
        # Pop 0. Used: 0. Free: 1, 2, ... 9
        # Return 0. Free: 1, 2, ... 9, 0
        # Next pop will be 1.
        # So it should be 1, not 0.
        self.assertEqual(seq2.block_table[0], 1) 

    def test_hash_collision_handling(self):
        # Mock hash to cause collision but different tokens
        original_hash_tokens = BlockManager._hash_tokens
        try:
            BlockManager._hash_tokens = lambda cls, tokens, prefix=-1: 12345 # Constant hash
            
            seq1 = Sequence(1, token_ids=[1, 2, 3, 4], block_size=self.block_size)
            self.bm.allocate(seq1)
            
            seq2 = Sequence(2, token_ids=[5, 6, 7, 8], block_size=self.block_size)
            self.bm.allocate(seq2)
            
            # Should not share even if hash is same, because tokens differ
            self.assertNotEqual(seq1.block_table[0], seq2.block_table[0])
            self.assertEqual(self.bm.stats["cache_miss"], 2)
        finally:
            BlockManager._hash_tokens = original_hash_tokens

    def test_complex_sharing_and_deallocation(self):
        # Seq1: [B1, B2, B3]
        # Seq2: [B1, B2, B4]
        # Seq3: [B1, B5, B6]
        self.bm.allocate(Sequence(1, token_ids=[1]*4 + [2]*4 + [3]*4, block_size=4)) # Seq1
        self.bm.allocate(Sequence(2, token_ids=[1]*4 + [2]*4 + [4]*4, block_size=4)) # Seq2
        self.bm.allocate(Sequence(3, token_ids=[1]*4 + [5]*4 + [6]*4, block_size=4)) # Seq3
        
        # B1 (tokens [1]*4) should have ref_count 3
        # B2 (tokens [2]*4) should have ref_count 2
        # B3, B4, B5, B6 should have ref_count 1
        
        b1_id = self.bm.hash_to_block_id[BlockManager._hash_tokens([1]*4, -1)]
        b2_id = self.bm.hash_to_block_id[BlockManager._hash_tokens([2]*4, self.bm.blocks[b1_id].hash)]
        
        self.assertEqual(self.bm.blocks[b1_id].ref_count, 3)
        self.assertEqual(self.bm.blocks[b2_id].ref_count, 2)
        
        # Deallocate Seq1
        # Need to keep track of seqs to deallocate them. Let's re-do with variables.
        self.bm = BlockManager(num_blocks=10, block_size=4) # Reset
        seq1 = Sequence(1, token_ids=[1]*4 + [2]*4 + [3]*4, block_size=4)
        seq2 = Sequence(2, token_ids=[1]*4 + [2]*4 + [4]*4, block_size=4)
        seq3 = Sequence(3, token_ids=[1]*4 + [5]*4 + [6]*4, block_size=4)
        self.bm.allocate(seq1)
        self.bm.allocate(seq2)
        self.bm.allocate(seq3)
        
        b1_id = seq1.block_table[0]
        b2_id = seq1.block_table[1]
        b3_id = seq1.block_table[2]
        
        self.bm.deallocate(seq1)
        self.assertEqual(self.bm.blocks[b1_id].ref_count, 2)
        self.assertEqual(self.bm.blocks[b2_id].ref_count, 1)
        self.assertEqual(self.bm.blocks[b3_id].ref_count, 0)
        self.assertIn(b3_id, self.bm.free_block_ids)

    def test_memory_pressure_and_recovery(self):
        # Fill up memory
        seqs = []
        for i in range(self.num_blocks):
            seq = Sequence(i, token_ids=[i]*self.block_size, block_size=self.block_size)
            self.bm.allocate(seq)
            seqs.append(seq)
        
        self.assertEqual(len(self.bm.used_block_ids), self.num_blocks)
        
        # Deallocate alternate blocks
        for i in range(0, self.num_blocks, 2):
            self.bm.deallocate(seqs[i])
        
        self.assertEqual(len(self.bm.used_block_ids), self.num_blocks // 2)
        self.assertEqual(len(self.bm.free_block_ids), self.num_blocks // 2)
        
        # Allocate new blocks, should succeed
        for i in range(self.num_blocks // 2):
            seq = Sequence(100+i, token_ids=[100+i]*self.block_size, block_size=self.block_size)
            self.bm.allocate(seq)
            self.assertEqual(len(seq.block_table), 1)

    def test_partial_block_not_cached_even_if_same_content(self):
        # Partial blocks should not be cached/shared even if content is same, 
        # because they are not hashed.
        seq1 = Sequence(1, token_ids=[1, 2], block_size=4)
        self.bm.allocate(seq1)
        
        seq2 = Sequence(2, token_ids=[1, 2], block_size=4)
        self.bm.allocate(seq2)
        
        self.assertNotEqual(seq1.block_table[0], seq2.block_table[0])
        self.assertEqual(self.bm.stats["cache_miss"], 2)

if __name__ == '__main__':
    unittest.main()
