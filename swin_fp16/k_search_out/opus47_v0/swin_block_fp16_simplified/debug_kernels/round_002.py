@triton.jit
def fused_attn_kernel(
    x_ptr,            # input after norm1's residual base, (B*nH*nW*N, C) fp16, but actually input is (B,H,W,C)
    # We need the original x (residual) plus the LN'd x
    # Actually let's load original, do LN, do attn, add residual
    g1_ptr, b1_ptr,   # norm1 gamma/beta (C,)
    wqkv_ptr, bqkv_ptr,  # (3C, C), (3C,)
    wproj_ptr, bproj_ptr,  # (C, C), (C,)
    rpe_ptr,          # (1, 1, N, N)
    out_ptr,          # output
    NW_total,         # number of windows
    C: tl.constexpr,  # 32
    N: tl.constexpr,  # 16
    eps,
    BLOCK_N: tl.constexpr,  # = N = 16
):
    pid = tl.program_id(0)
    
    # Load window: (N, C) = (16, 32)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, C)
    
    # Each window starts at pid * N * C
    x_window = x_ptr + pid * N * C
    
    x_in = tl.load(x_window + rows[:, None] * C + cols[None, :])  # (16, 32)
    x_f32 = x_in.to(tl.float32)
    
    # LayerNorm row-wise
    mean = tl.sum(x_f32, axis=1) / C
    xc = x_f32 - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    
    g = tl.load(g1_ptr + cols).to(tl.float32)
    b = tl.load(b1_ptr + cols).to(tl.float32)
    xn = xc * rstd[:, None] * g[None, :] + b[None, :]
    xn_fp16 = xn.to(tl.float16)
    
    # Load Wq, Wk, Wv each (C, C) = (32, 32)
    # Wqkv stored as (3C, C). Q rows: 0..C, K: C..2C, V: 2C..3C
    wq_off = 0
    wk_off = C * C
    wv_off = 2 * C * C
    
    w_rows = tl.arange(0, C)
    w_cols = tl.arange(0, C)
    wq = tl.load(wqkv_ptr + wq_off + w_rows[:, None] * C + w_cols[None, :])  # (C, C) -> Wq with shape (out, in)
    wk = tl.load(wqkv_ptr + wk_off + w_rows[:, None] * C + w_cols[None, :])
    wv = tl.load(wqkv_ptr + wv_off + w_rows[:, None] * C + w_cols[None, :])
    
    bq = tl.load(bqkv_ptr + tl.arange(0, C)).to(tl.float32)
    bk = tl.load(bqkv_ptr + C + tl.arange(0, C)).to(tl.float32)
    bv = tl.load(bqkv_ptr + 2*C + tl.arange(0, C)).to(tl.float32)
    
    # Q = xn @ Wq^T  : (N,C) @ (C,C) -> need to transpose Wq from (out,in) to (in,out)
    wq_t = tl.trans(wq)
    wk_t = tl.trans(wk)
    wv_t = tl.trans(wv)
    
    Q = tl.dot(xn_fp16, wq_t, out_dtype=tl.float32) + bq[None, :]  # (N, C)
    K = tl.dot(xn_fp16, wk_t, out_dtype=tl.float32) + bk[None, :]
    V = tl.dot(xn_fp16, wv_t, out_dtype=tl.float32) + bv[None, :]
    
    scale = 1.0 / tl.sqrt(float(C))
    Q_scaled = (Q * scale).to(tl.float16)
    K_fp16 = K.to(tl.float16)
    V_fp16 = V.to(tl.float16)
    
    # attn = Q @ K^T : (N,C) @ (C,N) -> (N,N)
    K_t = tl.trans(K_fp16)  # (C, N)
    attn = tl.dot(Q_scaled, K_t, out_dtype=tl.float32)  # (N, N)
    
    # add rpe_bias
    rpe = tl.load(rpe_ptr + rows[:, None] * N + tl.arange(0, BLOCK_N)[None, :]).to(tl.float32)
    attn = attn + rpe
    
    # softmax
    attn_max = tl.max(attn, axis=1)
    attn = attn - attn_max[:, None]
    attn_exp = tl.exp(attn)
    attn_sum = tl.sum(attn_exp, axis=1)
    attn_sm = attn_exp / attn_sum[:, None]
    attn_sm_fp16 = attn_sm.to(tl.float16)
    
    # out = attn @ V : (N,N) @ (N,C) -> (N,C)
    O = tl.dot(attn_sm_fp16, V_fp16, out_dtype=tl.float32)
    
    # proj
    wp = tl.load(wproj_ptr + w_rows[:, None] * C + w_cols[None, :])  # (C, C) out=C, in=C
    wp_t = tl.trans(wp)
    bp = tl.load(bproj_ptr + tl.arange(0, C)).to(tl.float32)
    
    O_fp16 = O.to(tl.float16)
    P = tl.dot(O_fp16, wp_t, out_dtype=tl.float32) + bp[None, :]
    
    # add residual (original x_in)
    P = P + x_in.to(tl.float32)
    
    # store
    tl.store(out_ptr + pid * N * C + rows[:, None] * C + cols[None, :], P.to(tl.float16))