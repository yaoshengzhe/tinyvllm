import os
import time
import random
import argparse
import importlib
from random import randint, seed
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Benchmark LLM engines")
    parser.add_argument("--backend", type=str, choices=["vllm", "nanovllm", "tinyvllm"], default="nanovllm", help="Backend to benchmark (default: vllm)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model ID or path")
    parser.add_argument("--num-seqs", type=int, default=256, help="Number of sequences")
    parser.add_argument("--max-input-len", type=int, default=1024, help="Maximum input length")
    parser.add_argument("--max-output-len", type=int, default=1024, help="Maximum output length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6, help="GPU memory utilization")
    args, _ = parser.parse_known_args()
    return args

def main():
    args = get_args()
    
    # Configuration
    seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # Model path handling
    model_id = args.model
    local_path = f"models/{model_id.split('/')[-1]}"
    if os.path.isdir(local_path):
        print(f"Using local model: {local_path}")
        model_id = local_path
    elif os.path.isdir(args.model):
        print(f"Using local model: {args.model}")
        model_id = args.model
    else:
        print(f"Using HuggingFace model: {model_id}")
        
        # nanovllm requires a local directory
        if args.backend == "nanovllm":
            print(f"nanovllm requires a local model directory. Downloading {model_id}...")
            try:
                from huggingface_hub import snapshot_download
                local_path = f"models/{model_id.split('/')[-1]}"
                snapshot_download(repo_id=model_id, local_dir=local_path)
                print(f"Model downloaded to: {local_path}")
                model_id = local_path
            except ImportError:
                print("Error: huggingface_hub not installed. Please run: pip install huggingface_hub")
                return
            except Exception as e:
                print(f"Error downloading model: {e}")
                return

    # Clean up memory before starting
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA available: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Debug flash_attn
    try:
        import flash_attn
        print(f"flash_attn version: {flash_attn.__version__}")
        print(f"flash_attn path: {flash_attn.__file__}")
    except ImportError:
        print("flash_attn not installed")
    except Exception as e:
        print(f"Error importing flash_attn directly: {e}")

    # Import backend
    print("\n" + "="*50)
    print(f"Initializing {args.backend} Engine...")
    print("="*50)
    
    LLM = None
    SamplingParams = None
    try:
        if args.backend == "vllm":
            from vllm import LLM, SamplingParams
        elif args.backend == "nanovllm":
            from nanovllm import LLM, SamplingParams
        elif args.backend == "tinyvllm":
            from tinyvllm import LLM, SamplingParams
    except ImportError as e:
        print(f"\n{'!'*50}")
        print(f"Error: Could not import backend '{args.backend}'")
        print(f"Details: {e}")
        print(f"{'!'*50}\n")
        
        print("To fix this, please install the required library:")
        if args.backend == "vllm":
            print("  pip install vllm")
        elif args.backend == "nanovllm":
            print("  pip install git+https://github.com/GeeeekExplorer/nano-vllm.git")
            print("  Note: If you see 'undefined symbol' errors with flash_attn, try reinstalling it:")
            print("  pip uninstall -y flash_attn && pip install flash_attn --no-build-isolation")
        elif args.backend == "tinyvllm":
            print("  Ensure 'tinyvllm.py' is in the current directory or PYTHONPATH.")
            
        return

    if LLM is None:
        print(f"Error: LLM class was not imported for backend '{args.backend}'.")
        return
    
    # Initialize LLM
    # Common kwargs
    kwargs = {
        "model": model_id,
        "enforce_eager": False,
        "max_model_len": 4096,
    }
    
    # Backend specific kwargs
    if args.backend == "vllm":
        kwargs["trust_remote_code"] = True
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    elif args.backend == "nanovllm":
        # nanovllm might not support trust_remote_code or gpu_memory_utilization in the same way
        # Based on reference, it supports enforce_eager and max_model_len
        pass
    elif args.backend == "tinyvllm":
        # tinyvllm Config might support these
        pass

    try:
        llm = LLM(**kwargs)
    except TypeError as e:
        print(f"Warning: {e}. Retrying with minimal arguments...")
        # Fallback for backends with different signatures
        llm = LLM(model=model_id)

    # Prepare Data
    print(f"\nPreparing {args.num_seqs} sequences...")
    print(f"Input Length:  up to {args.max_input_len}")
    print(f"Output Length: up to {args.max_output_len}")
    
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(1, args.max_input_len))] 
        for _ in range(args.num_seqs)
    ]
    
    # Create sampling params for each request
    sampling_params = [
        SamplingParams(
            temperature=0.6, 
            ignore_eos=True, 
            max_tokens=randint(1, args.max_output_len)
        ) 
        for _ in range(args.num_seqs)
    ]

    # Warmup
    print("\n" + "-"*50)
    print("Warming up...")
    print("-"*50)
    llm.generate(["Benchmark: "], SamplingParams())

    # Benchmark
    print("\n" + "="*50)
    print("Starting Benchmark...")
    print("="*50)
    
    start_time = time.time()
    
    if args.backend == "vllm":
        # vLLM specific input formatting
        inputs = [dict(prompt_token_ids=p) for p in prompt_token_ids]
        outputs = llm.generate(
            prompts=inputs,
            sampling_params=sampling_params, 
            use_tqdm=True
        )
    else:
        # nanovllm and tinyvllm (assuming similar API to nanovllm/reference)
        # Reference passed prompt_token_ids directly
        outputs = llm.generate(
            prompt_token_ids, 
            sampling_params, 
            use_tqdm=True
        )
        
    end_time = time.time()
    duration = end_time - start_time
    
    # Calculate Metrics
    if outputs and hasattr(outputs[0], 'outputs'):
        total_gen_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    else:
        # Fallback if outputs structure is different or not returned
        total_gen_tokens = sum(sp.max_tokens for sp in sampling_params)

    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    total_tokens = total_gen_tokens
    
    throughput_gen = total_gen_tokens / duration
    throughput_total = (total_input_tokens + total_gen_tokens) / duration
    
    # Visual Results
    print("\n" + "#"*50)
    print(f"BENCHMARK RESULTS ({args.backend})")
    print("#"*50)
    print(f"{'Metric':<25} | {'Value':<15}")
    print("-" * 43)
    print(f"{'Total Sequences':<25} | {args.num_seqs:<15}")
    print(f"{'Total Input Tokens':<25} | {total_input_tokens:<15}")
    print(f"{'Total Gen Tokens':<25} | {total_gen_tokens:<15}")
    print(f"{'Total Duration':<25} | {duration:.2f} s")
    print("-" * 43)
    print(f"{'Gen Throughput':<25} | {throughput_gen:.2f} tok/s")
    print(f"{'Total Throughput':<25} | {throughput_total:.2f} tok/s")
    print("#"*50 + "\n")

if __name__ == "__main__":
    main()
