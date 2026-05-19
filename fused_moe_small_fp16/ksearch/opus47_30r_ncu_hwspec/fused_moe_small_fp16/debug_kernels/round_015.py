import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
    x_ptr,
    gate_ptr,
    w1_ptr,
    w2_ptr,
    w3_ptr,
    out_ptr,
    N, D: tl.constexpr, I: tl.constexpr, E: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    k_range = tl.arange(0, BLOCK_K)
    token_ids = pid * BLOCK_K + k_range
    token_mask = token_ids < N

    d_range = tl.arange(0, BLOCK_D)
    i_range = tl.arange(0, BLOCK_I)
    e_range = tl.arange(0, BLOCK_E)

    d_mask = d_range < D
    i_mask = i_range < I
    e_mask = e_range < E

    # Load X: [BLOCK_K, BLOCK_D]
    x = tl.load(
        x_ptr + token_ids[:, None] * D + d_range[None, :],
        mask=token_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Gate weight: [BLOCK_E, BLOCK_D]
    gate = tl.load(
        gate_ptr + e_range[:, None] * D + d_range[None, :],
        mask=e_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # scores: [BLOCK_K, BLOCK_E] = x @ gate.T
    scores = tl.sum(x[:, None, :] * gate[None, :, :], axis=2)
    scores = tl.where(e_mask[None, :], scores, -float("inf"))

    smax = tl.max(scores, axis=1)
    exps = tl.exp(scores - smax[:, None])
    exps = tl.where(e_mask[None, :], exps, 0.0)
    ssum = tl.sum(exps, axis=1)
    probs = exps / ssum[:, None]

    top_val = tl.max(probs, axis=1)  # [BLOCK_K]
    is_max = probs == top_val[:, None]
    idx_vals = tl.where(is_max, e_range[None, :], E)
    expert_id = tl.min(idx_vals, axis=1)  # [BLOCK_K]
    weight = top_val  # [BLOCK_K]

    # Per-token expert MLP. Loop over K tokens (small).
    for k in tl.static_range(0, BLOCK_K):
        active = (pid * BLOCK_K + k) < N
        if active:
            xk = x[k, :]  # [BLOCK_D]
            eid = tl.sum(tl.where(k_range == k, expert_id, 0))
            wk = tl.sum(tl.where(k_range == k, weight, 0.0))

            w1_base = w1_ptr + eid * I * D
            w1 = tl.load(
                w1_base + i_range[:, None] * D + d_range[None, :],
                mask=i_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gate_out = tl.sum(w1 * xk[None, :], axis=1)
            gate_out = gate_out * (1.0 / (1.0 + tl.exp(-gate_out)))

            w3_base = w3_ptr + eid * I * D
            w3 = tl.load(
                w3_base + i_range[:, None] * D + d_range[None, :],
                mask=i_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up_out = tl.sum(w3 * xk[None, :], axis=1)

            intermediate = gate_out * up_out
            intermediate = tl.where(i_mask, intermediate, 0.0)

            w2_base = w2_ptr + eid * D * I
            w2 = tl.load(
                w2_base + d_range[:, None] * I + i_range[None, :],
                mask=d_mask[:, None] & i_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            out = tl.sum(w2 * intermediate[None, :], axis=1)
            out = out * wk

            tok = pid * BLOCK_K + k
            tl.store(out_ptr + tok * D + d_range, out.to(tl.float16), mask=d_mask)


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

        self._BLOCK_D = _next_pow2(dim)
        self._BLOCK_I = _next_pow2(intermediate_size)
        self._BLOCK_E = _next_pow2(num_experts)
        self._BLOCK_K = 8

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, D = hidden_states.shape
        N = B * S
        flat = hidden_states.reshape(N, D)
        out = torch.empty((N, D), device=flat.device, dtype=flat.dtype)

        grid = ((N + self._BLOCK_K - 1) // self._BLOCK_K,)
        fused_moe_kernel[grid](
            flat,
            self.gate.weight,
            self.w1,
            self.w2,
            self.w3,
            out,
            N, D, self.intermediate_size, self.num_experts,
            self._BLOCK_D, self._BLOCK_I, self._BLOCK_E,
            self._BLOCK_K,
            num_warps=2,
            num_stages=2,
        )
        return out.view(B, S, D)