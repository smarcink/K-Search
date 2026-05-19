import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
    x_ptr,          # [N, D] fp16
    gate_ptr,       # [E, D] fp16  (nn.Linear weight: [E, D])
    w1_ptr,         # [E, I, D] fp16
    w2_ptr,         # [E, D, I] fp16
    w3_ptr,         # [E, I, D] fp16
    out_ptr,        # [N, D] fp16
    N, D: tl.constexpr, I: tl.constexpr, E: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    d_range = tl.arange(0, BLOCK_D)
    i_range = tl.arange(0, BLOCK_I)
    e_range = tl.arange(0, BLOCK_E)

    d_mask = d_range < D
    i_mask = i_range < I
    e_mask = e_range < E

    # Load token x : [D]
    x = tl.load(x_ptr + pid * D + d_range, mask=d_mask, other=0.0).to(tl.float32)

    # --- Gate: scores[e] = sum_d gate[e,d] * x[d] ---
    # gate_ptr layout: [E, D]
    gate = tl.load(
        gate_ptr + e_range[:, None] * D + d_range[None, :],
        mask=e_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(gate * x[None, :], axis=1)  # [E]
    scores = tl.where(e_mask, scores, -float("inf"))

    # Softmax
    smax = tl.max(scores, axis=0)
    exps = tl.exp(scores - smax)
    exps = tl.where(e_mask, exps, 0.0)
    ssum = tl.sum(exps, axis=0)
    probs = exps / ssum

    # Top-1
    top_val = tl.max(probs, axis=0)
    # argmax
    is_max = probs == top_val
    # break ties: pick smallest index
    idx_vals = tl.where(is_max, e_range, E)
    expert_id = tl.min(idx_vals, axis=0)
    weight = top_val  # fp32

    # --- W1: gate_proj  [I, D] @ x[D] -> [I] ---
    w1_base = w1_ptr + expert_id * I * D
    w1 = tl.load(
        w1_base + i_range[:, None] * D + d_range[None, :],
        mask=i_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    gate_out = tl.sum(w1 * x[None, :], axis=1)  # [I]
    # SiLU
    gate_out = gate_out * (1.0 / (1.0 + tl.exp(-gate_out)))

    # --- W3: up_proj [I, D] @ x[D] -> [I] ---
    w3_base = w3_ptr + expert_id * I * D
    w3 = tl.load(
        w3_base + i_range[:, None] * D + d_range[None, :],
        mask=i_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    up_out = tl.sum(w3 * x[None, :], axis=1)  # [I]

    intermediate = gate_out * up_out  # [I]
    intermediate = tl.where(i_mask, intermediate, 0.0)

    # --- W2: down_proj [D, I] @ intermediate[I] -> [D] ---
    w2_base = w2_ptr + expert_id * D * I
    w2 = tl.load(
        w2_base + d_range[:, None] * I + i_range[None, :],
        mask=d_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    out = tl.sum(w2 * intermediate[None, :], axis=1)  # [D]

    out = out * weight
    tl.store(out_ptr + pid * D + d_range, out.to(tl.float16), mask=d_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return max(p, 1)


class ModelNew(nn.Module):
    def __init__(
        self,
        dim: int = 32,
        intermediate_size: int = 32,
        num_experts: int = 8,
        top_k: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, intermediate_size))
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, dim))

        nn.init.kaiming_uniform_(self.w1, a=2.236)
        nn.init.kaiming_uniform_(self.w2, a=2.236)
        nn.init.kaiming_uniform_(self.w3, a=2.236)

        self.half()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, D = hidden_states.shape
        N = B * S
        flat = hidden_states.reshape(N, D).contiguous()
        out = torch.empty((N, D), device=flat.device, dtype=flat.dtype)

        BLOCK_D = _next_pow2(D)
        BLOCK_I = _next_pow2(self.intermediate_size)
        BLOCK_E = _next_pow2(self.num_experts)

        grid = (N,)
        fused_moe_kernel[grid](
            flat,
            self.gate.weight,
            self.w1,
            self.w2,
            self.w3,
            out,
            N, D, self.intermediate_size, self.num_experts,
            BLOCK_D, BLOCK_I, BLOCK_E,
            num_warps=1,
            num_stages=2,
        )
        return out.view(B, S, D)