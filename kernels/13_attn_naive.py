"""
13  Attention (naive)      out = softmax(Q Kᵀ / √d) V

Q, K, V : (2048, 64) float16
out     : (2048, 64) float32
契约: run(q, k, v, out)

transformer 里那个 self-attention 的前向。不拆多头、不加 causal mask、不加 dropout,
只留最核心的一步。它是第 14 题 (FlashAttention) 的分母。

= 这道题其实是三道旧题串起来 =

    attn_scores_kernel    S = Q Kᵀ * scale          <- 第 11 题的 GEMM
    row_softmax_kernel    P = softmax(S, dim=-1)    <- 第 06 题的 softmax
    attn_pv_kernel        O = P V                   <- 又是第 11 题的 GEMM

所以它不难写, 十一题下来三块积木都造好了。它的价值不在"实现"而在**暴露问题**:
中间那个 (S, S) 矩阵会真的写进显存, 再读出来。

= 为什么这样慢 =

S = 2048 时, S x S 是 4 M 元素 (fp16 8 MB)。三步走下来它在 HBM 上被搬四次:

    第 1 步 写 S     8 MB
    第 2 步 读 S     8 MB
    第 2 步 写 P     8 MB
    第 3 步 读 P     8 MB
    ----------------------
    合计            32 MB

32 MB / 180 GB/s = 0.178 ms;  而真正的计算只有 4*S*S*d = 1.07 GFLOP,
按 tensor core 12 TFLOPS 是 0.089 ms。
**访存是计算的两倍** —— 这就是 FlashAttention 论文说的 IO 问题:
attention 完全可以放在 SRAM 里算完, 不该把中间结果吐到 HBM 再捡回来。

第 14 题要做的就是把三步合成一个 kernel, (S,S) 一次都不落盘。

= 三个 kernel 各自的坑 =

1. attn_scores_kernel
   Q 和 K 的行距都是 d, 但输出的行距是 S —— 三个 stride 别写混。
   K 要转置才能和 Q 相乘: 用 tl.trans(k), 不要去改地址 (第 08 题的教训,
   tl.trans 只换寄存器里 tile 的维度, 不换地址)。
   scale 乘在 dot **之后**: 乘在 q 上会先在 fp16 里舍入一次。

2. row_softmax_kernel
   一行 S 个元素, BS = next_pow2(S), 尾部补 -inf —— 不能补 0。
   补 0 会让 max 变成 0, 整行 logit 都是负数时直接错 (第 04 题的教训)。
   注意这一行是 2048 个元素, 一个 program 干, num_warps=8 才吃得下。
=
3. attn_pv_kernel
   P 是 fp16、V 是 fp16, tl.dot 直接用, 累加器开 fp32。
   千万别 .to(tl.float32) —— 那会静默降级成 TF32, K 从 16 掉到 8 (第 11 题的教训)。

= 精度 =

p 存 fp16 是有损的: softmax 出来的概率量化到 fp16, 再和 V 做加权和。
误差量级 ~ sqrt(S) * 2^-11 * |v| ≈ 0.02, 所以 tol 放到 (1e-2, 2e-2)。
FlashAttention 原文也是把 P 转成 fp16 喂 tensor core 的, 这是行业标准做法。

= 实测：证据就在这两个数里 =

    13_attn_naive   0.2013 ms   5.33 TFLOPS   29.3% 峰值   启动 3 次

把中间矩阵的流量除一下:

    32 MB / 0.2013 ms = 159 GB/s    ← HBM 带宽用掉 88%
    1.07 GFLOP / 0.2013 ms = 5.33 TFLOPS    ← 算力只用掉 29%

**带宽 88%、算力 29%** —— 这不是"写得不够快", 是它在拿 HBM 当工作区。
tensor core 有 71% 的时间在等数据从显存爬回来。

同一个尺寸下 torch 的 SDPA (内部就是 FlashAttention) 是 0.1270 ms / 8.46 TFLOPS,
比你快 37%。它快的地方不在 mma 指令, 而在**根本没有那个 (S,S) 矩阵**。

验收标准:
    通过  -> 正确性 + 启动 3 次 (三个 kernel, 理想值就是 3)
    目标  -> >= 5% 峰值。这题是**故意的下限**, 数字难看是对的。
             第 13 题实际的 29.3% 会给 14 题一个很低的起点错觉 —— 见下面。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def attn_scores_kernel(
    q_ptr, k_ptr, s_ptr,
    S, d, scale,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
):
    """S[i, j] = (Q[i, :] · K[j, :]) * scale      形状 (S, S)"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, BD)

    q = tl.load(q_ptr + offs_m[:, None] * d + offs_d[None, :],
                mask=(offs_m[:, None] < S) & (offs_d[None, :] < d), other=0.0)
    k = tl.load(k_ptr + offs_n[:, None] * d + offs_d[None, :],
                mask=(offs_n[:, None] < S) & (offs_d[None, :] < d), other=0.0)

    s = tl.dot(q, tl.trans(k)) * scale          # (BM, BD) x (BD, BN) -> (BM, BN)

    tl.store(s_ptr + offs_m[:, None] * S + offs_n[None, :], s.to(tl.float16),
             mask=(offs_m[:, None] < S) & (offs_n[None, :] < S))


@triton.jit
def row_softmax_kernel(s_ptr, p_ptr, S, BS: tl.constexpr):
    """P[i, :] = softmax(S[i, :])      一 program 一行"""
    row = tl.program_id(0)
    offs = tl.arange(0, BS)
    m = offs < S

    x = tl.load(s_ptr + row * S + offs, mask=m, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, 0)
    e = tl.exp(x)
    p = e / tl.sum(e, 0)

    tl.store(p_ptr + row * S + offs, p.to(tl.float16), mask=m)


@triton.jit
def attn_pv_kernel(
    p_ptr, v_ptr, o_ptr,
    S, d,
    BM: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr,
):
    """O = P V      (BM, BK) 沿 K(=S) 方向分块累加"""
    pid_m = tl.program_id(0)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)

    acc = tl.zeros((BM, BD), tl.float32)
    for k0 in range(0, tl.cdiv(S, BK)):
        offs_k = k0 * BK + tl.arange(0, BK)
        p = tl.load(p_ptr + offs_m[:, None] * S + offs_k[None, :],
                    mask=(offs_m[:, None] < S) & (offs_k[None, :] < S), other=0.0)
        v = tl.load(v_ptr + offs_k[:, None] * d + offs_d[None, :],
                    mask=(offs_k[:, None] < S) & (offs_d[None, :] < d), other=0.0)
        acc = tl.dot(p, v, acc)

    tl.store(o_ptr + offs_m[:, None] * d + offs_d[None, :], acc,
             mask=(offs_m[:, None] < S) & (offs_d[None, :] < d))


def run(q, k, v, out):
    S, d = q.shape
    scale = 1.0 / (d ** 0.5)
    BD = triton.next_power_of_2(d)

    # 1) S = Q Kᵀ * scale
    s = torch.empty((S, S), device=q.device, dtype=torch.float16)
    BM, BN = 64, 64
    attn_scores_kernel[(triton.cdiv(S, BM), triton.cdiv(S, BN))](
        q, k, s, S, d, scale, BM, BN, BD, num_warps=4)

    # 2) P = softmax(S, dim=-1)
    p = torch.empty_like(s)
    BS = triton.next_power_of_2(S)
    row_softmax_kernel[(S,)](s, p, S, BS, num_warps=8)

    # 3) O = P V
    BM2, BK = 64, 64
    attn_pv_kernel[(triton.cdiv(S, BM2),)](
        p, v, out, S, d, BM2, BK, BD, num_warps=4)
