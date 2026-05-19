import torch
import torch.nn.functional as F
from torch import nn, Tensor
import triton
import triton.language as tl


@triton.jit
def fused_gate_up_silu_kernel(
    X_ptr,
    W1T_ptr,  # pre-transposed: (E, D, I)
    W3T_ptr,  # pre-transposed: (E, D, I)
    SortedTok_ptr,
    ExpertOff_ptr,
    Out_ptr,
    M_total, D, I,
    stride_xn, stride_xd,
    stride_w1e, stride_w1d, stride_w1i,
    stride_w3e, stride_w3d, stride_w3i,
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
    tok_ids = tl.load(SortedTok_ptr + start + offs_m, mask=m_mask, other=0)

    x_row_ptrs = X_ptr + tok_ids[:, None] * stride_xn
    w1_base = W1T_ptr + expert_id * stride_w1e
    w3_base = W3T_ptr + expert_id * stride_w3e

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_mask = offs_n < I

    for k in range(0, D, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_row_ptrs + k_offs[None, :] * stride_xd
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w1_ptrs = w1_base + k_offs[:, None] * stride_w1d + offs_n[None, :] * stride_w1i
        w1 = tl.load(w1_ptrs, mask=n_mask[None, :], other=0.0)
        w3_ptrs = w3_base + k_offs[:, None] * stride_w3d + offs_n[None, :] * stride_w3i
        w3 = tl.load(w3_ptrs, mask=n_mask[None, :], other=0.0)

        acc1 += tl.dot(x, w1)
        acc3 += tl.dot(x, w3)

    sig = 1.0 / (1.0 + tl.exp(-acc1))
    silu = acc1 * sig
    out = silu * acc3
    out_fp16 = out.to(tl.float16)

    out_ptrs = Out_ptr + (start + offs_m)[:, None] * stride_on + offs_n[None, :] * stride_oi
    tl.store(out_ptrs, out_fp16, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def fused_down_scatter_kernel(
    Inter_ptr,
    W2T_ptr,  # pre-transposed: (E, I, D)
    SortedTok_ptr,
    ExpertOff_ptr,
    Weights_ptr,
    Out_ptr,
    M_total, D, I,
    stride_in, stride_ii,
    stride_w2e, stride_w2i, stride_w2d,
    stride_on, stride_od,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = tl.program_id(2)

    start = tl.load(ExpertOff_ptr + expert_id)
    end = tl.load(ExpertOff_ptr + expert_id + 1)
    num_tokens = end - start

    m_start = pid_m * BLOCK_M
    if m_start >= num_tokens:
        return

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < num_tokens
    n_mask = offs_n < D

    tok_ids = tl.load(SortedTok_ptr + start + offs_m, mask=m_mask, other=0)
    weights = tl.load(Weights_ptr + start + offs_m, mask=m_mask, other=0.0)

    inter_row_ptrs = Inter_ptr + (start + offs_m)[:, None] * stride_in
    w2_base = W2T_ptr + expert_id * stride_w2e

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, I, BLOCK_K):
        k_offs = k + offs_k
        k_mask = k_offs < I
        x_ptrs = inter_row_ptrs + k_offs[None, :] * stride_ii
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_ptrs = w2_base + k_offs[:, None] * stride_w2i + offs_n[None, :] * stride_w2d
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.dot(x, w)

    acc = acc * weights[:, None].to(tl.float32)

    out_ptrs = Out_ptr + tok_ids[:, None] * stride_on + offs_n[None, :] * stride_od
    tl.atomic_add(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


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

        # Pre-transpose weights for efficient kernel access (no tl.trans needed)
        # w1: (E, I, D) -> w1_t: (E, D, I)
        # w3: (E, I, D) -> w3_t: (E, D, I)
        # w2: (E, D, I) -> w2_t: (E, I, D)
        self.register_buffer('w1_t', self.w1.data.transpose(1, 2).contiguous(), persistent=False)
        self.register_buffer('w3_t', self.w3.data.transpose(1, 2).contiguous(), persistent=False)
        self.register_buffer('w2_t', self.w2.data.transpose(1, 2).contiguous(), persistent=False)

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
        sorted_weights = flat_weights[sorted_order].contiguous()

        M_total = sorted_token_ids.shape[0]
        I = self.intermediate_size
        E = self.num_experts

        counts = torch.bincount(sorted_expert_ids, minlength=E)
        expert_offsets = torch.zeros(E + 1, device=flat.device, dtype=torch.int32)
        expert_offsets[1:] = counts.cumsum(0).to(torch.int32)

        intermediate = torch.empty((M_total, I), device=flat.device, dtype=torch.float16)

        BLOCK_M = 32
        BLOCK_N = 128
        BLOCK_K = 64
        max_count = int(counts.max().item())
        grid_m = max(1, triton.cdiv(max_count, BLOCK_M))
        grid_n = triton.cdiv(I, BLOCK_N)

        fused_gate_up_silu_kernel[(grid_m, grid_n, E)](
            flat, self.w1_t, self.w3_t,
            sorted_token_ids, expert_offsets,
            intermediate,
            M_total, D, I,
            flat.stride(0), flat.stride(1),
            self.w1_t.stride(0), self.w1_t.stride(1), self.w1_t.stride(2),
            self.w3_t.stride(0), self.w3_t.stride(1), self.w3_t.stride(2),
            intermediate.stride(0), intermediate.stride(1),
            E=E,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=4,
        )

        output_fp32 = torch.zeros((N, D), device=flat.device, dtype=torch.float32)

        BLOCK_M2 = 32
        BLOCK_N2 = 128
        BLOCK_K2 = 64
        grid_m2 = max(1, triton.cdiv(max_count, BLOCK_M2))
        grid_n2 = triton.cdiv(D, BLOCK_N2)

        fused_down_scatter_kernel[(grid_m2, grid_n2, E)](
            intermediate, self.w2_t,
            sorted_token_ids, expert_offsets,
            sorted_weights,
            output_fp32,
            M_total, D, I,
            intermediate.stride(0), intermediate.stride(1),
            self.w2_t.stride(0), self.w2_t.stride(1), self.w2_t.stride(2),
            output_fp32.stride(0), output_fp32.stride(1),
            E=E,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=4,
        )

        output = output_fp32.to(torch.float16)
        return output.view(B, S, D)