# tinyvllm

A tiny version of vllm inspired by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## Description
This project aims to build a lightweight inference engine for LLMs, keeping all code in a single file `tinyvllm.py` for simplicity and educational purposes.

## Installation

```bash
pip install -r requirements.txt
```

## Model Download

Before running the example, download the model from Hugging Face:

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B --local-dir-use-symlinks False
```

Or simply let `transformers` handle it automatically (cached in `~/.cache/huggingface`), but for explicit management:

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-0.6B
```

## Usage

```bash
python example.py
```

## Benchmark

To benchmark vLLM performance (requires `vllm` installed):

```bash
python benchmark_vllm.py
```

### vLLM Benchmark Results

Hardware: **a2-ultragpu-1g** (NVIDIA A100 80GB x 1) on Google Compute Engine (GCE).

> **Note:** `tinyvllm` benchmark results will be added later using the same setup.

```text
==================================================
Starting Benchmark...
==================================================
Adding requests: 100%
 256/256 [00:00<00:00, 2912.32it/s]
Processed prompts: 100%
 256/256 [00:16<00:00, 48.64it/s, est. speed input: 8721.47 toks/s, output: 8180.38 toks/s]

##################################################
BENCHMARK RESULTS
##################################################
Metric                    | Value          
-------------------------------------------
Total Sequences           | 256            
Total Input Tokens        | 142827         
Total Gen Tokens          | 133966         
Total Duration            | 16.47 s
-------------------------------------------
Gen Throughput            | 8135.63 tok/s
Total Throughput          | 16809.38 tok/s
##################################################
```

## Testing

```bash
python -m unittest discover tests
```
