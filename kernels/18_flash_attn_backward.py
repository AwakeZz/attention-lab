"""
18  FlashAttention Backward    给定 dO, 求 dQ / dK / dV

输入  Q, K, V : (S, d)  float16          前向的三件套
      O       : (S, d)  float32          前向输出, 第 14 题的产物
      LSE     : (S,)    float32          = log Σ exp(S_ij), 前向顺手记下的行对数分母
      dO      : (S, d)  float16          上游梯度 (loss 对 O 的梯度)
输出  dQ,dK,dV : (S, d)  float32
契约  run(q, k, v, o, lse, do, dq, dk, dv), 内部启动 2 个 kernel

====================================================================
第一节  反向传播的数学 (从 softmax 的 Jacobian 推起)
====================================================================

前向两步:
    S   = Q Kᵀ / √d            (S,S)  分数矩阵
    P   = softmax(S)            (S,S)  按行归一化
    O   = P V                   (S,d)  输出

反向已知 dO = ∂L/∂O, 要的是 ∂L/∂Q, ∂L/∂K, ∂L/∂V。按链式法则从后往前拆:

--- 1.1 dV = Pᵀ dO ---

O = P V, 对 V 求导就是把 P 转置过来乘 dO:
    dV_j = Σ_i P_ij · dO_i      即 dV = Pᵀ dO
形状 (S,S)ᵀ (S,d) = (S,d)。

--- 1.2 dP = dO Vᵀ ---

同理对 P 求导, V 是常量:
    dP_ij = dO_i · V_j         即 dP = dO Vᵀ

--- 1.3 dS = P ⊙ (dP - D), 其中 D_i = Σ_j P_ij dP_ij ---

这一步是 softmax 的反向。softmax 的 Jacobian 不是逐元素的, 第 i 行的输出
P_i 依赖整行的 S_i, 所以 J 是一个 (S,S) 矩阵。但乘法 J·dP 可以化简:

    dS_ij = P_ij (dP_ij - Σ_k P_ik dP_ik) = P_ij (dP_ij - D_i)

D_i 是"第 i 行 dP 的加权和, 权重就是 P_i 自己"。逐行一个标量。

--- 1.4 D_i 的的神来之笔: D_i = dO_i · O_i ---
直接算 D_i 要先把 dP 整行算出来 (一次 dO Vᵀ 的 dot), 代价 2·S²·d。
但代入 dP_ij = dO_i · V_j 展开:

    D_i = Σ_j P_ij (dO_i · V_j)
        = dO_i · (Σ_j P_ij V_j)         把与 j 无关的 dO_i 提到求和号外
        = dO_i · O_i                    括号里正是前向输出 O 的第 i 行!

所以 **D_i 只要拿前向的 O 和上游 dO 点积就行**, 不用碰 dP。
O 本来就在 HBM 里 (前向写过的), 白捡的。这是 FlashAttention backward
能不落 (S,S) 的第二个关键 (第一个是前向的 online softmax)。

--- 1.5 dQ 和 dK ---

S = Q Kᵀ / √d, 分别求导:
    dQ = dS K / √d             (S,S)(S,d)
    dK = dSᵀ Q / √d            (S,S)ᵀ(S,d)

注意 dQ/dK 都要带 1/√d, dV 不带 (因为 P 已经是归一化后的, d 藏在 S 里)。

====================================================================
第二节  为什么必须拆成两个 kernel
====================================================================

看四个梯度的"归约方向":

    dQ_i = Σ_j (...)           对 key j 求和, query i 是输出行  -> 按 query 块并行
    dV_j = Σ_i (...)           对 query i 求和, key j 是输出行  -> 按 key 块并行
    dK_j = Σ_i (...)           同上

dQ 和 dV/dK 的输出行方向相反:
- 若一个 program 管 BM 行 query, 它能顺手累加 dQ 的这 BM 行 (遍历所有 KV 块),
  但算 dV 时它只贡献了"部分和", 要写回 dV 就得原子加 —— 32 个 program 抢同一行。
- 若一个 program 管 BN 行 key, dV/dK 是完整的, 但 dQ 要原子加。

两边都要原子加就两败俱伤。干脆拆开:
    kernel A: grid=(cdiv(S,BM),)  program 拥有 BM 行 query, 遍历所有 KV 块 -> dQ
    kernel B: grid=(cdiv(S,BN),)  program 拥有 BN 行 key,   遍历所有 Q 块  -> dK, dV
各自输出行归自己独占, 循环里只往寄存器累加, 一个原子加都不用。

代价: Q/K/V/O/dO/LSE 被读了两遍 (A 一遍 B 一遍)。对 S=2048 d=64,
这些张量加起来 ~4 MB, 读两遍 8 MB; 而省下的是 (S,S) 的 16 MB 落盘 + 原子加。
这笔账划算, 也是 FA2 backward 的标准结构。

====================================================================
第三节  变量词典
====================================================================

    中文名         符号      形状        定义式 / 含义
    ------------------------------------------------------------------
    分数块         qk        (BM,BN)    = dot(q, kᵀ) * scale, 重算出来的
    概率块         p         (BM,BN)    = exp(qk - lse_i), 就是 softmax 的 P
    行折扣标量     D_i       (BM,)      = dO_i · O_i, softmax 反向的修正项
    dP 块          dp        (BM,BN)    = dot(dO, vᵀ)
    dS 块          ds        (BM,BN)    = p * (dp - D_i[:, None])
    dQ 累加器      dq        (BM,BD)    遍历 KV 块累加, 循环外统一乘 scale
    dK 累加器      dk        (BN,BD)    遍历 Q 块累加, 循环外统一乘 scale
    dV 累加器      dv        (BN,BD)    同上, 但不乘 scale

====================================================================
第四节  kernel A: 按 query 块算 dQ
====================================================================

    pid_m = program_id(0);  offs_m = pid_m*BM + arange(BM)
    q  = load Q[offs_m]                       # (BM,BD)
    o  = load O[offs_m]                       # (BM,BD) fp32
    do = load dO[offs_m]                      # (BM,BD) fp16
    lse = load LSE[offs_m]                    # (BM,)
    D  = sum(do.to(fp32) * o, axis=1)         # (BM,)   <- 1.4 的技巧

    dq = zeros((BM,BD))
    for n0 in range(cdiv(S, BN)):
        k, v = load K[offs_n], V[offs_n]      # (BN,BD), 每轮只搬一块
        qk = dot(q, trans(k)) * scale         # (BM,BN)
        qk = where(offs_n < S, qk, -inf)      # 越界列填 -inf
        p  = exp(qk - lse[:, None])           # (BM,BN), 越界列自然变 0
        dp = dot(do, trans(v))                # (BM,BN)
        ds = p * (dp - D[:, None])            # (BM,BN)
        dq = dot(ds.to(fp16), k, dq)          # (BM,BD) 累加

    store dQ[offs_m] = dq * scale             # scale 只在这里乘一次

====================================================================
第五节  kernel B: 按 key 块算 dK 和 dV
====================================================================

把第四节转置过来: program 拥有 BN 行 key, K/V 驻留寄存器, 遍历所有 Q 块。

    pid_n = program_id(0);  offs_n = pid_n*BN + arange(BN)
    k = load K[offs_n];  v = load V[offs_n]   # (BN,BD) 全程不动
    dk = zeros((BN,BD));  dv = zeros((BN,BD))

    for m0 in range(cdiv(S, BM)):
        q, o, do = load Q/O/dO[offs_m]        # (BM,BD)
        lse = load LSE[offs_m]                # (BM,)
        D = sum(do.to(fp32) * o, axis=1)      # (BM,)

        qk = dot(k, trans(q)) * scale         # (BN,BM)  <- 注意是 k 在前
        qk = where(offs_m < S, qk, -inf)
        p  = exp(qk - lse[None, :])           # (BN,BM)  <- lse 在列上广播
        dp = dot(v, trans(do))                # (BN,BM)
        ds = p * (dp - D[None, :])            # (BN,BM)

        dv = dot(p.to(fp16),  do, dv)         # (BN,BM)(BM,BD) = (BN,BD)
        dk = dot(ds.to(fp16), q,  dk)         # (BN,BD)

    store dK[offs_n] = dk * scale             # dK 带 scale
    store dV[offs_n] = dv                     # dV 不带

====================================================================
第六节  坑 (每一条都对应一次可能的静默错)
====================================================================

1. **p = exp(qk - lse), 不是 exp(qk - max)。**
   lse = m + log l 已经把归一化的分母含进去了 (第 16 题讲过)。
   这里没有 online 递推, 因为前向的 lse 直接给了, 一步到位。
   若手写 max/qk 重新算一遍归一化, 结果一样但多一趟访存。

2. **dO 是 fp16, O 是 fp32, 点积前先统一 dtype。**
   `do.to(tl.float32) * o`。直接相乘 Triton 会按 fp16 算, D_i 误差能到 1e-2。

3. **dQ/dK 乘 scale, dV 不乘。**
   忘了或乘错, 表现是 "dQ/dK 整体差一个 √d 倍" (这里 √64=8, 误差巨大, 一眼能抓)。

4. **ds 转 fp16 才能进 dot。**
   和前向的 p 一样: `tl.dot` 只吃 fp16/bf16, ds 是 fp32 算出来的。

5. **kernel B 里 p / ds 直接 dot, 千万别 trans。**
   p 形状 (BN,BM) = (key, query), dO 是 (BM,BD), `dot(p, do)` 收缩的正是 query 维,
   得到 (BN,BD) —— 方向本来就对。我第一版想当然写了 `dot(trans(p), do)`,
   trans 成 (BM,BN) 后 dot 收缩的是 BM, 但因为 BM==BN==64, **形状检查照样过**,
   结果 dK/dV 差 0.29/0.21 (dQ 因为 kernel A 没 trans 是对的, 6.8e-05)。
   和第 14 题坑 6/7 同源: 维度相等时形状检查拦不住语义错, 靠 --odd hard
   (BM≠BN 的场景) 才露馅。两个 dot 的收缩维必须自己心里有数:
       A: dot(q, kᵀ)  (BM,BD)(BD,BN) -> (BM,BN)   收缩 d
          dot(ds, k)  (BM,BN)(BN,BD) -> (BM,BD)   收缩 n
       B: dot(k, qᵀ)  (BN,BD)(BD,BM) -> (BN,BM)   收缩 d
          dot(p, do)  (BN,BM)(BM,BD) -> (BN,BD)   收缩 m

6. **越界 mask 填 -inf, p 自然归零。**
   qk 在 query/key 越界处填 -inf, `exp(-inf - lse) = 0`, 该列/行对累加无贡献。
   若填 0, 会多出一个 exp(0-lse) 的假概率, 而且 dP 那边 V/dO 是 0 也救不回来
   (ds = p*(dp-D), D 不为 0 时 p*(-D) 是负的, 污染整行)。

7. **D_i 是逐行标量, 广播方向别反。**
   kernel A 里是 `D[:, None]` (沿 BN 列广播), kernel B 里是 `D[None, :]` (沿 BN 行广播)。
   两边对称, 写反了形状检查会过 (BM 和 BN 都是 64) 但结果全错 —— 又一个
   "维度相等时形状检查失效" 的坑。用 BM≠BN 的尺寸测一下最稳。

8. **lse 必须从前来, 不能在 backward 里重算。**
   重算 lse 要额外遍历一遍全部 KV 块, 等于把前向 qk 又算一次。存一个 (S,) 的
   lse 只有 8 KB, 重算要 2·S²·d = 33 MFLOP, 显然存着划算。

9. **d 不是 2 的幂时 shared memory 会爆。**
   `--odd hard` 里 d=72/96, BD 取 next_power_of_2 变 128, kernel B 同时驻留
   q/o/do/k/v 五个 (·,128) 张量, num_stages=2 要 106 KB, 超 3050 Ti 的 101 KB。
   按 d 降 num_stages 到 1 就能过 (主路径 d=64 不受影响)。这类"编译期资源超限"
   比结果错好抓 —— 至少报错信息直接给了 Required / Hardware limit 两个数。

====================================================================
第七节  实测 (单进程冷机, 口径见 README)
====================================================================

    18_attn_backward   0.4330 ms   6.20 TFLOPS   34.1%   启动 2 次

三个独立进程各跑一次: 0.4332 / 0.4321 / 0.4330 ms, 数字稳定。
--odd hard (d=96/72, S=513) 全部通过, 启动 2 次。

理论 FLOPs 口径: 前向 4·S²·d = 2 个 dot; backward 要重算 P (qk) + dp + dq + dk/dv,
合计 10·S²·d = 5 个 dot, 正好是前向的 2.5 倍 —— 与 FA2 论文里
"backward ≈ 2.5x forward" 的说法一致。

backward 是 compute-bound: 算术强度约 S·d / (每元素若干字节), 对 S=2048 远大于
平衡点 45.4。所以分母用 fp16 tensor core 实测峰值 18.2 TFLOPS。

= 34.1% 为什么比前向的 67.6% 低一半 =

理想情况下 backward 该是前向耗时的 2.5 倍 (0.0873 × 2.5 ≈ 0.22 ms),
实测 0.433 ms 是 5 倍。分 kernel 计时 (_probe_18_bwd.py, 已删) 找到了原因:

    dq   kernel  0.138 ms   (占 28%)
    dkdv kernel  0.358 ms   (占 72%)   <- 瓶颈

dkdv 慢是结构性的: 每个 program 的 K/V 驻留寄存器, 但要遍历全部 S/BM = 32 个
Q 块, 每轮重新加载 q/o/do 三块 + 做 4 个 dot, 访存轮数压不下来。这不是接线错
(输出和 autograd 参考 max|diff| 在 1e-4 量级), 是两-kernel 拆法的固有代价 ——
FA2 论文里 dK/dV 也比 dQ 慢, 工业实现要靠更精细的 warp 分工和原子分块来救。

tile 扫描结论 (BM/BN/warps/stages, 探针口径):

    BM64 BN64 w4 s2  dq 0.138  dkdv 0.358  <- 现在这个, 最优
    BM32 BN64 w4 s2  dq 0.673  dkdv 1.290  轮数翻倍 + 算术强度掉, 全线崩
    BM64 BN32 w4 s2  dq 1.102  dkdv 0.301  dq 那侧塌了
    BM64 BN64 w4 s1  dq 0.743  dkdv 1.725  关流水反而更慢 (流水线是有用的)
    BM64 BN64 w8 s2  dq 1.588  dkdv 0.711  warp 翻倍, 和前向 14 题一样腰斩
    BM128 BN64 w4 s2 dq 1.273  dkdv 3.865  block 太大, occupancy 掉光

反直觉的是 **BM=32 全线崩**: 前向 14 题 BM=32 只是慢一点 (55.9% vs 63%),
backward 这里直接翻 5 倍。因为 dq 循环里多了一整个 dot (dp), 每个 tile 的
定长开销被放大, 小 tile 的算术强度劣势被加倍惩罚。和 14 题是同一个机制,
只是 backward 的"每轮活更重"让天平更偏向大 tile。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dq_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, do_ptr, dq_ptr,
    S, d, scale,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
):
    """按 query 块并行, 遍历所有 KV 块, 累加 dQ 的 BM 行。"""
    pid_m = tl.program_id(0)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, BD)
    mask_m = offs_m < S
    mask_d = offs_d < d

    # 这个 program 负责的 BM 行 query / O / dO, 全程驻留寄存器
    q = tl.load(q_ptr + offs_m[:, None] * d + offs_d[None, :],
                mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    o = tl.load(o_ptr + offs_m[:, None] * d + offs_d[None, :],
                mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    do = tl.load(do_ptr + offs_m[:, None] * d + offs_d[None, :],
                 mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    lse = tl.load(lse_ptr + offs_m, mask=mask_m, other=0.0)

    # D_i = dO_i · O_i (第 1.4 节): 逐行一个标量
    D = tl.sum(do.to(tl.float32) * o, axis=1)

    dq = tl.zeros((BM, BD), tl.float32)

    for n0 in range(tl.cdiv(S, BN)):
        offs_nb = n0 * BN + offs_n
        mask_n = offs_nb < S
        k = tl.load(k_ptr + offs_nb[:, None] * d + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_ptr + offs_nb[:, None] * d + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale          # (BM,BN)
        qk = tl.where(mask_n[None, :], qk, -float("inf"))
        p = tl.exp(qk - lse[:, None])                # 越界列 -> exp(-inf) = 0
        dp = tl.dot(do, tl.trans(v))                 # (BM,BN)
        ds = p * (dp - D[:, None])                   # (BM,BN)
        dq = tl.dot(ds.to(tl.float16), k, dq)        # (BM,BD) 累加

    tl.store(dq_ptr + offs_m[:, None] * d + offs_d[None, :], dq * scale,
             mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _dkdv_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, do_ptr, dk_ptr, dv_ptr,
    S, d, scale,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
):
    """按 key 块并行, 遍历所有 Q 块, 累加 dK / dV 的 BN 行。"""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, BD)
    mask_n = offs_n < S
    mask_d = offs_d < d

    # K/V 驻留寄存器, 全程不动
    k = tl.load(k_ptr + offs_n[:, None] * d + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    v = tl.load(v_ptr + offs_n[:, None] * d + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0)

    dk = tl.zeros((BN, BD), tl.float32)
    dv = tl.zeros((BN, BD), tl.float32)

    for m0 in range(tl.cdiv(S, BM)):
        offs_mb = m0 * BM + offs_m
        mask_m = offs_mb < S
        q = tl.load(q_ptr + offs_mb[:, None] * d + offs_d[None, :],
                    mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        o = tl.load(o_ptr + offs_mb[:, None] * d + offs_d[None, :],
                    mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        do = tl.load(do_ptr + offs_mb[:, None] * d + offs_d[None, :],
                     mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        lse = tl.load(lse_ptr + offs_mb, mask=mask_m, other=0.0)

        D = tl.sum(do.to(tl.float32) * o, axis=1)    # (BM,)

        qk = tl.dot(k, tl.trans(q)) * scale          # (BN,BM)
        qk = tl.where(mask_m[None, :], qk, -float("inf"))
        p = tl.exp(qk - lse[None, :])                # (BN,BM)
        dp = tl.dot(v, tl.trans(do))                 # (BN,BM)
        ds = p * (dp - D[None, :])                   # (BN,BM)

        # p/ds 形状已经是 (BN,BM) = (key, query), 直接和 (BM,BD) 的 do/q 收缩 query 维。
        # 不要 trans —— trans 成 (BM,BN) 后收缩的是 BM, 而 BM==BN==64 时形状检查
        # 照样过, 语义却全错了 (实测 dK/dV 差 0.29/0.21)。
        dv = tl.dot(p.to(tl.float16), do, dv)     # (BN,BM)(BM,BD) -> (BN,BD)
        dk = tl.dot(ds.to(tl.float16), q, dk)     # (BN,BM)(BM,BD) -> (BN,BD)

    tl.store(dk_ptr + offs_n[:, None] * d + offs_d[None, :], dk * scale,
             mask=mask_n[:, None] & mask_d[None, :])
    tl.store(dv_ptr + offs_n[:, None] * d + offs_d[None, :], dv,
             mask=mask_n[:, None] & mask_d[None, :])


def run(q, k, v, o, lse, do, dq, dk, dv):
    """两个 kernel: dQ 按 query 块并行, dK/dV 按 key 块并行。"""
    S, d = q.shape
    scale = 1.0 / (d ** 0.5)
    BD = triton.next_power_of_2(d)

    BM, BN = 64, 64

    # d=64 (主路径) 时 BD=64, 两级流水刚好塞进 101 KB shared memory。
    # --odd hard 里 d=72/96 会把 BD 撑到 128, 同时驻留的 q/o/do/k/v 五个
    # 张量直接顶到 106 KB 超上限。这两组尺寸只验正确性不测速, 降一级流水
    # (num_stages=1) 换回空间即可; 主路径一个字节都不变。
    stages = 2 if d <= 64 else 1
    # Turing (sm_75, 如 T4) 的 MMA 布局展开后 shared 占用比 Ampere 大得多,
    # dkdv 在 Ampere 上 48.5 KB, 老卡上可能翻倍超 T4 的 64 KB 上限。
    # 老卡统一降一级流水换回空间, 主路径 (Ampere+) 一个字节都不变。
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        stages = 1

    _dq_kernel[(triton.cdiv(S, BM),)](
        q, k, v, o, lse, do, dq, S, d, scale, BM, BN, BD,
        num_warps=4, num_stages=stages,
    )
    _dkdv_kernel[(triton.cdiv(S, BN),)](
        q, k, v, o, lse, do, dk, dv, S, d, scale, BM, BN, BD,
        num_warps=4, num_stages=stages,
    )
