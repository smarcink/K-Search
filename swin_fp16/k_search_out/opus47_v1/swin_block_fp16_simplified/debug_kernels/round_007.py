import triton
import triton.language as tl
@triton.jit
def fused_mlp_kernel(
    x_ptr, out_ptr,
    ln_w_ptr, ln_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    fc2_w_ptr, fc2_b_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,   # 32
    BLOCK_H: tl.constexpr,   # 128
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    mask_r = rows < n_rows
    c = tl.arange(0, BLOCK_C)
    h = tl.arange(0, BLOCK_H)

    # Load (BLOCK_M, BLOCK_C)
    x = tl.load(x_ptr + rows[:, None] * BLOCK_C + c[None, :], mask=mask_r[:, None], other=0.0).to(tl.float32)

    # LN
    mean = tl.sum(x, axis=1) / BLOCK_C
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / BLOCK_C
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    xn = xc * rstd[:, None]
    ln_w = tl.load(ln_w_ptr + c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + c).to(tl.float32)
    xn = xn * ln_w[None, :] + ln_b[None, :]

    # FC1: xn (M,C) @ W1.T (C,H) where W1 is (H,C)
    w1 = tl.load(fc1_w_ptr + h[:, None] * BLOCK_C + c[None, :]).to(tl.float16)  # (H, C)
    b1 = tl.load(fc1_b_ptr + h).to(tl.float32)
    xn_f16 = xn.to(tl.float16)
    h_act = tl.dot(xn_f16, tl.trans(w1), out_dtype=tl.float32) + b1[None, :]   # (M, H)

    # GELU exact
    g = 0.5 * h_act * (1.0 + tl.erf(h_act * 0.7071067811865476))

    # FC2: g (M,H) @ W2.T (H,C) where W2 is (C,H)
    w2 = tl.load(fc2_w_ptr + c[:, None] * BLOCK_H + h[None, :]).to(tl.float16)  # (C, H)
    b2 = tl.load(fc2_b_ptr + c).to(tl.float32)
    g_f16 = g.to(tl.float16)
    y = tl.dot(g_f16, tl.trans(w2), out_dtype=tl.float32) + b2[None, :]  # (M, C)

    # Residual
    res = (y + x).to(tl.float16)
    tl.store(out_ptr + rows[:, None] * BLOCK_C + c[None, :], res, mask=mask_r[:, None])