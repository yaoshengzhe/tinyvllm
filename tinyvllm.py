"""
tinyvllm - A tiny version of vllm inspired by nano-vllm.
"""

import torch
import numpy as np

class SamplingParams:
    def __init__(self, temperature=1.0, top_p=1.0):
        self.temperature = temperature
        self.top_p = top_p

class LLM:
    def __init__(self, model):
        self.model = model

    def generate(self, prompts, sampling_params=None):
        # TODO: Implement generation logic
        return []

def main():
    print("Hello from tinyvllm!")

if __name__ == "__main__":
    main()
