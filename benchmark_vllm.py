import os
import time
import random
from random import randint, seed
from vllm import LLM, SamplingParams
import numpy as np

def main():
    # Configuration
    seed(0)
    random.seed(0)
    np.random.seed(0)
    
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024
    
    # Model path handling
    model_id = "Qwen/Qwen3-0.6B"
    local_path = "models/Qwen3-0.6B"
    if os.path.isdir(local_path):
        print(f"Using local model: {local_path}")
        model_id = local_path
    else:
        print(f"Using HuggingFace model: {model_id}")

    # Clean up memory before starting
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize LLM
    print("\n" + "="*50)
    print("Initializing vLLM Engine...")
    print("="*50)
    # Reduce GPU memory utilization to avoid OOM if other processes are running
    # Default is 0.9, setting to 0.6 to be safer
    llm = LLM(
        model=model_id,
        enforce_eager=False,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.6
    )

    # Prepare Data
    print(f"\nPreparing {num_seqs} sequences...")
    print(f"Input Length:  up to {max_input_len}")
    print(f"Output Length: up to {max_output_len}")
    
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] 
        for _ in range(num_seqs)
    ]
    
    # Create sampling params for each request
    sampling_params = [
        SamplingParams(
            temperature=0.6, 
            ignore_eos=True, 
            max_tokens=randint(100, max_output_len)
        ) 
        for _ in range(num_seqs)
    ]

    # Warmup
    print("\n" + "-"*50)
    print("Warming up...")
    print("-"*50)
    # Match reference warmup
    llm.generate(["Benchmark: "], SamplingParams())

    # Benchmark
    print("\n" + "="*50)
    print("Starting Benchmark...")
    print("="*50)
    
    # Format inputs for vLLM (requires list of dicts if passing token IDs directly)
    inputs = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    start_time = time.time()
    # vLLM generate expects inputs as first argument (prompts)
    outputs = llm.generate(
        prompts=inputs,
        sampling_params=sampling_params, 
        use_tqdm=True  # Keep True for visual progress as requested
    )
    end_time = time.time()
    
    duration = end_time - start_time
    
    # Calculate Metrics
    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    total_gen_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    total_tokens = total_gen_tokens # Throughput usually refers to generated tokens
    
    throughput_gen = total_gen_tokens / duration
    throughput_total = (total_input_tokens + total_gen_tokens) / duration
    
    # Visual Results
    print("\n" + "#"*50)
    print("BENCHMARK RESULTS")
    print("#"*50)
    print(f"{'Metric':<25} | {'Value':<15}")
    print("-" * 43)
    print(f"{'Total Sequences':<25} | {num_seqs:<15}")
    print(f"{'Total Input Tokens':<25} | {total_input_tokens:<15}")
    print(f"{'Total Gen Tokens':<25} | {total_gen_tokens:<15}")
    print(f"{'Total Duration':<25} | {duration:.2f} s")
    print("-" * 43)
    print(f"{'Gen Throughput':<25} | {throughput_gen:.2f} tok/s")
    print(f"{'Total Throughput':<25} | {throughput_total:.2f} tok/s")
    print("#"*50 + "\n")

if __name__ == "__main__":
    main()
