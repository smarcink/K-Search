import torch
import torch.nn.functional as F
from torch import nn, Tensor
import triton
import triton.language as tl


@triton.jit
def fused_gate_up_silu_kernel(
    X_ptr,           # [N, D] fp16
    W1_ptr,          # [E, I, D] fp16  (gate_proj)
    W3_ptr,          # [E, I, D] fp16  (up_proj)
    SortedTok_ptr,   # [M_total] int32 token ids in expert-sorted order
    ExpertOff_ptr,   # [E+1] int32 cumulative offsets
    Out_ptr,         # [M_total, I] fp16
    M_total, D, I,
    stride_xn, stride_xd,
    stride_w1e, stride_w1i, stride_w1d,
    stride_w3e, stride_w3i, stride_w3d,
    stride_on, stride_oi,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = tl.program_id(2)

    start = tl.load(ExpertOff_ptr + expert_id)
    end = tl.load(ExpertOff_ptr + expert_id + 1)
    num_tokens = end - start

    m_start = pid * BLOCK_M
    if m_start >= num_tokens:
        return

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < num_tokens
    # gather token ids
    tok_ids = tl.load(SortedTok_ptr + start + offs_m, mask=m_mask, other=0)

    # pointers
    # X rows
    x_row_ptrs = X_ptr + tok_ids[:, None] * stride_xn  # [BM]
    w1_base = W1_ptr + expert_id * stride_w1e
    w3_base = W3_ptr + expert_id * stride_w3e

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_mask = offs_n < I

    for k in range(0, D, BLOCK_K):
        k_offs = k + offs_k
        k_mask = k_offs < D
        # Load X tile: [BM, BK]
        x_ptrs = x_row_ptrs + k_offs[None, :] * stride_xd
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # Load W1 tile: [BN, BK] -> need [BK, BN] for dot
        w1_ptrs = w1_base + offs_n[:, None] * stride_w1i + k_offs[None, :] * stride_w1d
        w1 = tl.load(w1_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        w3_ptrs = w3_base + offs_n[:, None] * stride_w3i + k_offs[None, :] * stride_w3d
        w3 = tl.load(w3_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        acc1 += tl.dot(x, tl.trans(w1))
        acc3 += tl.dot(x, tl.trans(w3))

    # SiLU(acc1) * acc3
    sig = 1.0 / (1.0 + tl.exp(-acc1))
    silu = acc1 * sig
    out = silu * acc3
    out_fp16 = out.to(tl.float16)

    out_ptrs = Out_ptr + (start + offs_m)[:, None] * stride_on + offs_n[None, :] * stride_oi
    tl.store(out_ptrs, out_fp16, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        intermediate_size: int = 2816,
        num_experts: int = 8,
        top_k: int = 2,
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

    def forward(self, hidden_states: Tensor) -> Tensor:
        B, S, D = hidden_states.shape
        N = B * S
        flat = hidden_states.view(N, D).contiguous()

        # --- Gating ---
        scores = self.gate(flat).float()
        probs = F.softmax(scores, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states.dtype)

        flat_indices = topk_indices.view(-1)
        flat_weights = topk_weights.view(-1)
        token_ids = torch.arange(N, device=flat.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)

        sorted_order = flat_indices.argsort(stable=True)
        sorted_token_ids = token_ids[sorted_order].to(torch.int32)
        sorted_expert_ids = flat_indices[sorted_order]
        sorted_weights = flat_weights[sorted_order]

        M_total = sorted_token_ids.shape[0]
        I = self.intermediate_size
        E = self.num_experts

        # Expert offsets via bincount + cumsum
        counts = torch.bincount(sorted_expert_ids, minlength=E)
        expert_offsets = torch.zeros(E + 1, device=flat.device, dtype=torch.int32)
        expert_offsets[1:] = counts.cumsum(0).to(torch.int32)

        # Allocate intermediate output [M_total, I]
        intermediate = torch.empty((M_total, I), device=flat.device, dtype=torch.float16)

        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 32
        # Max tokens per expert (bound) — use ceiling of M_total for grid dim
        max_m_per_expert = M_total  # safe upper bound; kernel early-exits when out of range
        grid_m = triton.cdiv(max_m_per_expert, BLOCK_M)
        grid_n = triton.cdiv(I, BLOCK_N)

        # Tighter grid_m: use max count
        max_count = int(counts.max().item())
        grid_m = max(1, triton.cdiv(max_count, BLOCK_M))

        fused_gate_up_silu_kernel[(grid_m, grid_n, E)](
            flat, self.w1, self.w3,
            sorted_token_ids, expert_offsets,
            intermediate,
            M_total, D, I,
            flat.stride(0), flat.stride(1),
            self.w1.stride(0), self.w1.stride(1), self.w1.stride(2),
            self.w3.stride(0), self.w3.stride(1), self.w3.stride(2),
            intermediate.stride(0), intermediate.stride(1),
            E=E,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # --- down_proj per expert ---
        expert_outputs = torch.empty((M_total, D), device=flat.device, dtype=torch.float16)
        for expert_id in range(E):
            start = int(expert_offsets[expert_id].item())
            end = int(expert_offsets[expert_id + 1].item())
            if end == start:
                continue
            inter_e = intermediate[start:end]  # [m, I]
            out_e = inter_e @ self.w2[expert_id].t()  # [m, D]
            expert_outputs[start:end] = out_e

        weighted_outputs = expert_outputs * sorted_weights.unsqueeze(-1)

        output = torch.zeros(N, D, device=flat.device, dtype=flat.dtype)
        output.scatter_add_(0, sorted_token_ids.long().unsqueeze(-1).expand(-1, D), weighted_outputs)

        return output.view(B, S, D)