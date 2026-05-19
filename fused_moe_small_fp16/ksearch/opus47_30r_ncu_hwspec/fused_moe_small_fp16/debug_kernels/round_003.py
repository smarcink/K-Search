import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel_all(
    x_ptr,
    gate_ptr,
    w1_ptr,
    w2_ptr,
    w3_ptr,
    out_ptr,
    N: tl.constexpr, D: tl.constexpr, I: tl.constexpr, E: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_E: tl.constexpr,
):
    n_range = tl.arange(0, BLOCK_N)
    d_range = tl.arange(0, BLOCK_D)
    i_range = tl.arange(0, BLOCK_I)
    e_range = tl.arange(0, BLOCK_E)

    n_mask = n_range < N
    d_mask = d_range < D
    i_mask = i_range < I
    e_mask = e_range < E

    # Load all tokens [BLOCK_N, BLOCK_D] in fp16
    x = tl.load(
        x_ptr + n_range[:, None] * D + d_range[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    # Gate weight [E, D] -> we want [D, E] for x @ gate.T
    gate_w = tl.load(
        gate_ptr + e_range[None, :] * D + d_range[:, None],
        mask=e_mask[None, :] & d_mask[:, None],
        other=0.0,
    )

    # scores = x @ gate_w  -> [BLOCK_N, BLOCK_E]
    scores = tl.dot(x, gate_w, out_dtype=tl.float32)
    scores = tl.where(e_mask[None, :], scores, -float("inf"))

    smax = tl.max(scores, axis=1, keep_dims=True)
    exps = tl.exp(scores - smax)
    exps = tl.where(e_mask[None, :], exps, 0.0)
    ssum = tl.sum(exps, axis=1, keep_dims=True)
    probs = exps / ssum

    top_val = tl.max(probs, axis=1)  # [BLOCK_N]
    is_max = probs == top_val[:, None]
    idx_vals = tl.where(is_max, e_range[None, :], E)
    expert_id = tl.min(idx_vals, axis=1)  # [BLOCK_N] int32

    x_f32 = x.to(tl.float32)

    for n in tl.static_range(0, BLOCK_N):
        if n < N:
            sel = (n_range == n).to(tl.int32)
            eid = tl.sum(expert_id * sel, axis=0)
            w_n = tl.sum(top_val * sel.to(tl.float32), axis=0)

            # x_n [D] fp32
            x_n = tl.sum(x_f32 * sel[:, None].to(tl.float32), axis=0)

            # W1 [I, D] fp32
            w1 = tl.load(
                w1_ptr + eid * I * D + i_range[:, None] * D + d_range[None, :],
                mask=i_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gate_out = tl.sum(w1 * x_n[None, :], axis=1)
            gate_out = gate_out * (1.0 / (1.0 + tl.exp(-gate_out)))

            w3 = tl.load(
                w3_ptr + eid * I * D + i_range[:, None] * D + d_range[None, :],
                mask=i_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up_out = tl.sum(w3 * x_n[None, :], axis=1)
            intermediate = gate_out * up_out
            intermediate = tl.where(i_mask, intermediate, 0.0)

            w2 = tl.load(
                w2_ptr + eid * D * I + d_range[:, None] * I + i_range[None, :],
                mask=d_mask[:, None] & i_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            out_n = tl.sum(w2 * intermediate[None, :], axis=1) * w_n

            tl.store(out_ptr + n * D + d_range, out_n.to(tl.float16), mask=d_mask)


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

    x = tl.load(x_ptr + pid * D + d_range, mask=d_mask, other=0.0).to(tl.float32)

    gate = tl.load(
        gate_ptr + e_range[:, None] * D + d_range[None, :],
        mask=e_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(gate * x[None, :], axis=1)
    scores = tl.where(e_mask, scores, -float("inf"))

    smax = tl.max(scores, axis=0)
    exps = tl.exp(scores - smax)
    exps = tl.where(e_mask, exps, 0.0)
    ssum = tl.sum(exps, axis=0)
    probs = exps / ssum

    top_val = tl.max(probs, axis=0)
    is_max = probs == top_val
    idx_vals = tl.where(is_max, e_range, E)
    expert_id = tl.min(idx_vals, axis=0)
    weight = top_val

    w1_base = w1_ptr + expert_id * I * D
    w1 = tl.load(
        w1_base + i_range[:, None] * D + d_range[None, :],
        mask=i_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    gate_out = tl.sum(w1 * x[None, :], axis=1)
    gate_out = gate_out * (1.0 / (1.0 + tl.exp(-gate_out)))

    w3_base = w3_ptr + expert_id * I * D
    w3 = tl.load(
        w3_base + i_range[:, None] * D + d_range[None, :],
        mask=i_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    up_out = tl.sum(w3 * x[None, :], axis=1)

    intermediate = gate_out * up_out
    intermediate = tl.where(i_mask, intermediate, 0.0)

    w2_base = w2_ptr + expert_id * D * I
    w2 = tl.load(
        w2_base + d_range[:, None] * I + i_range[None, :],
        mask=d_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    out = tl.sum(w2 * intermediate[None, :], axis=1)

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
        BLOCK_E = max(_next_pow2(self.num_experts), 16)
        BLOCK_N = max(_next_pow2(N), 16)

        # Use single-CTA fused kernel when shapes allow tl.dot (need BLOCK_N>=16, BLOCK_D>=16, BLOCK_E>=16)
        if BLOCK_N >= 16 and BLOCK_D >= 16 and BLOCK_E >= 16 and N <= BLOCK_N:
            fused_moe_kernel_all[(1,)](
                flat,
                self.gate.weight,
                self.w1,
                self.w2,
                self.w3,
                out,
                N, D, self.intermediate_size, self.num_experts,
                BLOCK_N, BLOCK_D, BLOCK_I, BLOCK_E,
                num_warps=2,
                num_stages=2,
            )
        else:
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