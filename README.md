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

## Testing

```bash
python -m unittest discover tests
```
