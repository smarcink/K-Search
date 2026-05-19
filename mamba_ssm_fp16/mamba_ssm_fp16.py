
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
    def __init__(self, d_model: int = 1024, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv

        # Input projection: x -> (x_proj for SSM params)
        # Projects to dt, B, C
        self.dt_rank = d_model // 16  # 64
        dt_proj_size = self.dt_rank + 2 * d_state  # dt_rank + B + C

        # Causal conv1d (depthwise)
        self.conv1d = nn.Conv1d(
            d_model, d_model, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_model
        )

        # SSM parameter projections
        self.x_proj = nn.Linear(d_model, dt_proj_size, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # A parameter (log-space, shared across input, not selective)
        # Initialized to -log(1, 2, ..., d_state) per channel
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))  # (D, N)

        # D skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        self.half()

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        N = self.d_state

        # --- 1. Causal Conv1D ---
        # (B, L, D) -> (B, D, L) for conv
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # causal: trim future
        x_conv = F.silu(x_conv)
        x_conv = x_conv.transpose(1, 2)  # back to (B, L, D)

        # --- 2. SSM parameter projection (input-dependent / selective) ---
        x_proj = self.x_proj(x_conv)  # (B, L, dt_rank + 2*N)

        # Split into dt, B, C
        dt_x = x_proj[..., :self.dt_rank]             # (B, L, dt_rank)
        B_param = x_proj[..., self.dt_rank:self.dt_rank + N]  # (B, L, N)
        C_param = x_proj[..., self.dt_rank + N:]      # (B, L, N)

        # --- 3. Discretization ---
        # dt: project from low-rank and apply softplus
        dt = self.dt_proj(dt_x)  # (B, L, D)
        dt = F.softplus(dt)      # ensure positive

        # A: from log-space, shape (D, N)
        A = -torch.exp(self.A_log.float())  # negative for stability

        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        # dt: (B, L, D), A: (D, N) -> A_bar: (B, L, D, N)
        dt_float = dt.float()
        A_bar = torch.exp(dt_float.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
        B_bar = dt_float.unsqueeze(-1) * B_param.float().unsqueeze(2)  # (B, L, D, N)

        # --- 4. Selective Scan (sequential reference implementation) ---
        # This is the recurrence:
        #   h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        #   y_t = C_t * h_t
        #
        # Optimization target: replace with parallel associative scan

        x_float = x_conv.float()  # (B, L, D)
        h = torch.zeros(B, D, N, device=x.device, dtype=torch.float32)  # state
        ys = []

        for t in range(L):
            # h = A_bar * h + B_bar * x
            h = A_bar[:, t] * h + B_bar[:, t] * x_float[:, t].unsqueeze(-1)
            # y = (C * h).sum over state dim
            y_t = (C_param[:, t].float().unsqueeze(1) * h).sum(dim=-1)  # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, D)

        # --- 5. Skip connection with D ---
        y = y + self.D.float() * x_float
        y = y.to(x.dtype)

        return y


def get_inputs():
    """Return input tensors for benchmarking (fp16, on accelerator)."""
    device = get_device()
    B, S, D = 4, 2048, 1024
    x = torch.randn(B, S, D, dtype=torch.float16, device=device)
    return [x]


def get_init_inputs():
    """Return constructor arguments for Model."""
    return []  # defaults: d_model=1024, d_state=16, d_conv=4


if __name__ == "__main__":
    device = get_device()
    model = Model().to(device)
    inputs = get_inputs()

    # Warmup
    with torch.no_grad():
        output = model(*inputs)

    B, S, D = 4, 2048, 1024
    print(f"Input shape:  ({B}, {S}, {D})")
    print(f"Output shape: {output.shape}  (expected: ({B}, {S}, {D}))")
    print(f"Output dtype: {output.dtype}")
    assert output.shape == (B, S, D), f"Shape mismatch: {output.shape}"
    assert not output.isnan().any(), "NaN in output!"
    assert not output.isinf().any(), "Inf in output!"
    print("OK")
