
import torch
import torch.nn.functional as F
from torch import nn, Tensor


def get_device():
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Model(nn.Module):
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

        # Gating linear
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # Expert FFN weights (stacked)
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, dim))   # gate_proj
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, intermediate_size))   # down_proj
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, dim))   # up_proj

        nn.init.kaiming_uniform_(self.w1, a=2.236)
        nn.init.kaiming_uniform_(self.w2, a=2.236)
        nn.init.kaiming_uniform_(self.w3, a=2.236)

        self.half()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (B, SeqLen, Dim) float16

        Returns:
            output: (B, SeqLen, Dim) float16
        """
        B, S, D = hidden_states.shape
        N = B * S

        flat = hidden_states.view(N, D)

        # --- Gating ---
        scores = self.gate(flat).float()
        probs = F.softmax(scores, dim=-1)

        # Top-1 selection (single active expert per token)
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        topk_weights = topk_weights.to(hidden_states.dtype)

        # Flatten (with top_k=1, these are just (N,))
        flat_indices = topk_indices.view(-1)
        flat_weights = topk_weights.view(-1)

        token_ids = torch.arange(N, device=flat.device)

        # Sort by expert for grouped execution
        sorted_order = flat_indices.argsort(stable=True)
        sorted_token_ids = token_ids[sorted_order]
        sorted_expert_ids = flat_indices[sorted_order]
        sorted_weights = flat_weights[sorted_order]

        # Gather tokens in expert-grouped order
        permuted_tokens = flat[sorted_token_ids]

        # --- Expert FFNs (SwiGLU) ---
        expert_outputs = torch.zeros_like(permuted_tokens)

        for expert_id in range(self.num_experts):
            mask = sorted_expert_ids == expert_id
            if not mask.any():
                continue
            expert_input = permuted_tokens[mask]

            # SwiGLU: SiLU(x @ W1.T) * (x @ W3.T), then @ W2.T
            gate_out = F.silu(expert_input @ self.w1[expert_id].t())
            up_out = expert_input @ self.w3[expert_id].t()
            intermediate = gate_out * up_out
            expert_out = intermediate @ self.w2[expert_id].t()

            expert_outputs[mask] = expert_out

        # --- Weighted scatter back ---
        weighted_outputs = expert_outputs * sorted_weights.unsqueeze(-1)

        output = torch.zeros(N, D, device=flat.device, dtype=flat.dtype)
        output.scatter_add_(0, sorted_token_ids.unsqueeze(-1).expand(-1, D), weighted_outputs)

        return output.view(B, S, D)


def get_inputs():
    """Return input tensors for benchmarking (fp16, on accelerator)."""
    device = get_device()
    B, S, D = 1, 16, 32
    hidden_states = torch.randn(B, S, D, dtype=torch.float16, device=device)
    return [hidden_states]


def get_init_inputs():
    """Return constructor arguments for Model."""
    return []  # defaults: dim=256, intermediate=512, num_experts=8, top_k=1


if __name__ == "__main__":
    device = get_device()
    model = Model().to(device)
    inputs = get_inputs()
    output = model(*inputs)
    B, S, D = 1, 16, 32
    print(f"Input shape: ({B}, {S}, {D})")
    print(f"Output shape: {output.shape}  (expected: ({B}, {S}, {D}))")
    print(f"Output dtype: {output.dtype}")
    assert not output.isnan().any(), "NaN in output!"
    assert not output.isinf().any(), "Inf in output!"
    print("OK")
