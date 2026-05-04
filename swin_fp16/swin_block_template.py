"""
Template for a custom nn.Module to optimize with K-Search.

Requirements:
1. Define a `Model` class that inherits from nn.Module
2. Define `get_inputs()` that returns a list of input tensors
3. Define `get_init_inputs()` that returns a list of constructor arguments

K-Search will generate a `ModelNew` class with optimized Triton/CUDA kernels
that matches the same forward() signature and output.

Usage:
    python -u generate_kernels_and_eval.py \
      --task-source kernelbench \
      --task-path swin_block_template.py \
      --model-name gpt-5.4 \
      --api-key "$GNAI_TOKEN" \
      --base-url "https://gnai.intel.com/api/providers/openai/v1" \
      --language triton \
      --target-gpu RTX5090 \
      --max-opt-rounds 10 \
      --world-model \
      --save-solutions \
      --artifacts-dir .ksearch-swin \
      --kernelbench-eval-mode local
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Replace this with your Swin Transformer block implementation.
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        # TODO: Your Swin Transformer block layers here
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, seq_len, dim)
        Returns:
            Output tensor of shape (B, seq_len, dim)
        """
        # TODO: Your forward pass here
        shortcut = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = shortcut + x
        x = x + self.mlp(self.norm(x))
        return x


# --- These functions are required by KernelBench evaluator ---

# Example dimensions - adjust to match your actual use case
batch_size = 4
seq_len = 3136  # e.g. 56x56 spatial tokens
dim = 96
num_heads = 4


def get_inputs():
    """Return a list of input tensors (on CPU; evaluator moves them to GPU)."""
    return [torch.randn(batch_size, seq_len, dim)]


def get_init_inputs():
    """Return a list of constructor arguments for Model(...)."""
    return [dim, num_heads]
