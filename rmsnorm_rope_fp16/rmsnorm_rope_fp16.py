import torch
import math
from torch import nn, Tensor


def get_device():
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def precompute_freqs(head_dim: int, seq_len: int, theta: float = 10000.0):
    """Precompute cos/sin frequencies for RoPE (rotate_half variant)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


class Model(nn.Module):
    """RMSNorm followed by Rotary Position Embedding.
    
    Forward:
        x (B, SeqLen, Dim) → RMSNorm → reshape to heads → RoPE → output (B, SeqLen, Dim)
    """

    def __init__(self, dim=4096, num_heads=32, seq_len=2048, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        # RMSNorm learnable scale
        self.weight = nn.Parameter(torch.ones(dim))

        # Precompute RoPE cos/sin — (seq_len, head_dim // 2)
        cos, sin = precompute_freqs(self.head_dim, seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # Run in fp16
        self.half()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, SeqLen, Dim) float16 hidden states.

        Returns:
            (B, SeqLen, Dim) float16 — normalized and position-encoded.
        """
        B, S, D = x.shape

        # --- RMSNorm ---
        # Accumulate in fp32 for numerical stability (matches HF LlamaRMSNorm)
        variance = x.float().pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        normed = (normed * self.weight).to(x.dtype)

        # --- Reshape to multi-head: (B, S, NumHeads, HeadDim) ---
        normed = normed.view(B, S, self.num_heads, self.head_dim)

        # --- RoPE (rotate_half variant) ---
        half = self.head_dim // 2
        x1 = normed[..., :half]
        x2 = normed[..., half:]

        # Broadcast cos/sin: (S, half) → (1, S, 1, half)
        cos = self.rope_cos[:S].unsqueeze(0).unsqueeze(2)
        sin = self.rope_sin[:S].unsqueeze(0).unsqueeze(2)

        # Rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)

        # Flatten heads back: (B, S, Dim)
        return rotated.reshape(B, S, D)


# ---------------------------------------------------------------------------
# KernelBench conventions
# ---------------------------------------------------------------------------

def get_inputs():
    """Return input tensors for benchmarking (fp16, on accelerator)."""
    device = get_device()
    x = torch.randn(8, 2048, 4096, dtype=torch.float16, device=device)
    return [x]


def get_init_inputs():
    """Return constructor arguments for Model."""
    return []


if __name__ == "__main__":
    device = get_device()
    model = Model().to(device)
    inputs = get_inputs()
    out = model(*inputs)
    print(f"Input shape:  {inputs[0].shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    assert out.shape == (8, 2048, 4096), f"Expected (8, 2048, 4096), got {out.shape}"
    assert out.dtype == torch.float16
    # Run twice to check determinism
    out2 = model(*inputs)
    assert torch.allclose(out, out2, atol=0.0), "Non-deterministic!"
    print("OK")
