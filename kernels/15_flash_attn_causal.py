"""
15  FlashAttention + causal mask   看第 i 个 token 时不许偷看 i 之后的 token

Q, K, V : (2048, 64) float16
out     : (2048, 64) float32
契约: run(q, k, v, out)

和第 14 题算的是同一个 attention, 只多一条规矩:

    (i, j) 这个位置, 当 j > i 时 qk 必须被 mask 掉。

也就是 softmax 只在每行的**前缀**上做。第 0 行只有 1 个有效元素, 第 2047 行有 2048 个。
(训练 GPT 时每个 token 只能看见自己和过去, 不能看见未来。)

= 为什么这条 mask 不只是"多一个 where" =

naive 的 causal 实现会算完整的 SxS 再 mask —— 一半的算力白花了。
FlashAttention 分块之后能真的省掉:
第 pid_m 块 query 要看的 key 只有 0 .. start_m+BM-1, 后面的块**根本不用进循环**。
于是循环上界从 `cdiv(S, BN)` 变成 `cdiv(start_m+BM, BN)` —— 这是**运行期**的标量,
所以是动态循环, 不是编译期展开。

块数从 32*32 = 1024 掉到 32*33/2 = 528 (含对角块), 计算量减半。
这是 attention 训练里 "causal 能把 FA 拉快近 2 倍" 的真正原因, 不是 mask 本身便宜。

= 三个和 14 题不同的地方 =

1. **循环上界是动态的。**
   `hi = cdiv(min(start_m + BM, S), BN)` —— 两个都不能省:
   - 少了 `min(..., S)`: 最后一小块 pid_m 会多算一个越界块 (不影响正确性, 白算)
   - 少了 `cdiv`: 直接写 `(start_m+BM)//BN` 会漏掉不满一块的尾巴

2. **循环拆成两段。**
   `[0, start_m//BN)` 这一段整块都 <= start_m-1 < 所有 offs_m, 无条件通过 causal,
   而且 offs_n < start_m < S 也不越界 —— **where 和 load 的 mask 全都可以省掉**。
   只有对角块那段才需要 mask。实测省 4.4% (探针 V2)。

3. **FLOPs 口径变了。**
   13/14 题用 `4*S^2*d` (全量)。causal 的有效 (i,j) 对约 S^2/2, 所以按 `2*S^2*d` 报。
   注意分块实现**实际**算的比理论多: 对角块是一整块 BN 全算, 只有一半有用。
   S=2048/BM=BN=64 时实际 528 块 vs 理论 512 块 = 多算 3.1%。
   所以按 `2*S^2*d` 报出来的百分比是**偏保守**的。

= causal 特有的坑 =

**坑 1: 尾部 query 行不是问题, 别自己吓自己。**
S 不是 BM 整数倍时, offs_m 有 >= S 的 padding 行。直觉上这些行"整行都被 mask"→
m_new = -inf → alpha = exp(-inf - (-inf)) = exp(nan) → 全 nan。
但实际不会: padding 行的 offs_m 很大 (>= S), 而 mask 条件 `offs_m >= offs_n` 里
offs_n < S <= offs_m 恒成立, 于是 padding 行**全部通过 mask**, qk 是正常值 (q 灌 0 → qk=0),
m_new = 0, 没有 nan。这些行的输出最后被 store 的 mask 丢掉。
真正的危险是反过来: 如果哪天把 mask 写成 `offs_n <= offs_m & offs_m < S`, 就 nan 了。

**坑 2: mask 的方向。**
`offs_m[:, None] >= offs_n[None, :]` 是 (BM,1) 对 (1,BN) → (BM,BN), 行是 query 列是 key。
和 14 题尾部 mask `offs_n[None, :] < S` 一样要**升维**, 因为 qk 是二维的。
BM == BN 时写反了照样跑通 (形状对得上), 只有结果错 —— 和第 14 题坑 7 同一个机制。

**坑 3: 两个 mask 要合并。**
`(offs_n <= offs_m) & (offs_n < S)`。分两条 where 写也行, 但**都不能漏** ——
只留 causal 会读到尾部越界的 k/v (other=0 → qk=0 → 变成正常 logit)。

= 真正的瓶颈不是算力, 是负载不均 =

算完才发现: 计算量减半, 时间只少了 20% (0.0873 → 0.0702)。差的那 30% 去哪了?

先算清 14 题的时间构成。它 grid=32 (每 program 一个 m 块, 各跑 32 个 n 块),
20 个 SM 分两波, 每波 32 块 → 总 64 块的时间。0.0873 ms / 64 = **1.36 µs 一块**。

再看 15 题 (BM=64): 32 个 program 的块数是 1, 2, 3, ..., 32。
- 第一波 20 个 (块数 1..20): 由最长的决定 → 20 块
- 第二波 12 个 (块数 21..32): 而且第二波恰好是最长的那些 → 32 块
- 合计 **52 块**, 而 14 题是 64 块。52/64 = 0.81

**预测提速 19%, 实测 20%。对上了。**
也就是说 causal 省下的另一半算力, 全被"波次量化 + 长短不均"吃掉了。

试了三种补救 (探针 `_probe_causal_balance.py`, BM=64, 同一份 `_do_task` 代码):

    P0 现状, 一 program 一个 m 块              86.1 us   关键路径 52 块
    P1 配对, 16 个 program 各做 task i 与 31-i  76.0 us   每个恒 33 块
    P2 LPT, 20 个 program + host 端贪心装桶     78.7 us   关键路径还是 32 块

P2 为什么没赢? 因为 **LPT 也打不破 32**: 最大的那个 m 块自身就有 32 个 n 块,
只要它由一个 program 串行处理, 关键路径就 >= 32 块。表里确实有个桶是 [32]。

P1 (配对) 确实把 52 压到 33 块, 但它只用 16 个 SM (20 个 SM 有 4 个在歇),
且每个 program 要反复 load q / 重初始化状态, 摊掉了一部分收益 → 只快 12%。

**结论: 只要一个 m 块内部的 n 循环不拆开, 关键路径就卡在 32 块。
要再往下走必须 split-N (跨 program 分一段 n, 再二次归约) —— 那是第 16 题的事。**

= 固定开销有多大 (顺便证明它不是主因) =

探针 `_probe_causal_overhead.py`: 固定 grid=32, 让每块跑 NITER 次迭代, 扫 NITER:

    NITER= 1   10.70 us
    NITER= 2   13.58 us
    NITER= 4   19.45 us
    NITER= 8   29.88 us
    NITER=16   54.15 us
    NITER=32   94.88 us

拟合 t = F + c*NITER → **F ≈ 8 µs, c ≈ 2.71 µs/迭代** (两端点: 94.88-10.70) / 31。
F 就是 q load + 状态初始化 + 结尾除法/store 这些"每块一次"的成本, 占总时间 8%。
**8 µs 解释不了那 30% 的差距, 所以不是固定开销的问题** —— 排除了一个假设。

= BM 的结论和第 14 题相反 =

14 题: BM=64 最优 (63.0%), BM=32 只有 55.9%, 因为 tile 小了算术强度掉。
15 题: **BM=32 更快** (探针 69.7 vs 82.4), 因为 causal 下 BM 直接决定关键路径:

    BM=64 → 最长 program 跑 32 个标准块
    BM=32 → 最长 program 跑 32 个**半宽**块 = 16 个标准块

块总数不变 (528 个标准块), 但最长任务砍半, 20 个 SM 填得更平。
探针口径扫一遍 (含两段循环的 V2):

    BM=16  86.8 us   34.0%   太窄, 算术强度掉得比均衡收益快
    BM=32  66.6 us   44.3%   <- 现在这个
    BM=64  82.4 us   35.8%

**倒 U 形, 最优点在 32。** 这就是"负载均衡 vs 算术强度"的权衡, 和 GEMM 里
"tile 大算术强度高 vs tile 小填充率高"是同一场架, 只是这次均衡那一边赢了。

顺带两条没用的尝试: 逆序 pid (先派长任务) 只快 3% —— 说明 GPU 的 block 派发
本来就是动态的, 不是我先入为主的"同步波次"; 逆序 + 两段循环反而更慢 (68.8 us)。
**假设错了就记下来, 别只留成功的那个。**

= 实测 =

check.py 口径 (每配置一个独立进程, 冷机):

    14_flash_attn (全量, BM=64)  0.0873 ms  12.30 TFLOPS  67.6%
    15 BM=64, 单段循环            0.0702 ms   7.65 TFLOPS  42.0%
    15 BM=32, 单段循环            0.0615 ms   8.72 TFLOPS  47.9%
    15 BM=32, 两段循环            0.0594 ms   9.04 TFLOPS  49.7%   <- 现在这个
    torch SDPA (is_causal=True)  0.1075 ms   4.99 TFLOPS
    13_attn_naive (全量)         0.2008 ms   5.33 TFLOPS  29.3%

三次重跑 0.0594 / 0.0594 / 0.0602, 稳定。**超 SDPA 81%**。
(SDPA 的 causal 比它自己的全量还慢一点: 0.1075 vs 0.1068 —— 同一份 kernel 里
要额外处理掩码边界, 它的通用性在这里变成了成本。)

为什么停在 49.7% 而不继续: 上面算过, 不拆 n 循环的情况下关键路径下界是 32 个标准块,
即 32*1.36 = 43.5 µs → 上限约 68%。现在是 59.4 µs, 还有 1.37 倍的余地,
但要拿到就得 split-N。**这个上限不是调参能突破的, 是结构性的** ——
所以正确答案是留给第 16 题, 而不是在这里反复扫 BM/BN。

一个诚实的说明: causal 的 47.9% 不能和全量的 67.6% 直接比。
分母不一样 (2*S²d vs 4*S²d), 而且全量版有 1024 块喂 20 个 SM 喂得很饱,
causal 只有 528 块且长短悬殊。**同分母下 causal 每块更贵, 这是必然的。**
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_block(q, m_i, l_i, acc, k_ptr, v_ptr, offs_m, offs_n, offs_d, S, d, scale,
                BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
                MASKED: tl.constexpr):
    """一块 K/V 的在线 softmax 递推。

    MASKED=False 时 caller 已保证整块都在对角线左侧且不越界, 于是一条 mask 都不用写 ——
    这不是"优化技巧", 是 causal 的结构事实: 对角线左边的块本来就不需要判。
    """
    if MASKED:
        k = tl.load(k_ptr + offs_n[:, None] * d + offs_d[None, :],
                    mask=(offs_n[:, None] < S) & (offs_d[None, :] < d), other=0.0)
        v = tl.load(v_ptr + offs_n[:, None] * d + offs_d[None, :],
                    mask=(offs_n[:, None] < S) & (offs_d[None, :] < d), other=0.0)
    else:
        k = tl.load(k_ptr + offs_n[:, None] * d + offs_d[None, :])
        v = tl.load(v_ptr + offs_n[:, None] * d + offs_d[None, :])

    qk = tl.dot(q, tl.trans(k)) * scale

    if MASKED:
        # 一条 where 盖两个 mask: 右上三角 (causal) + 尾部越界 (S)。填 -inf 不是 0。
        qk = tl.where((offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < S),
                      qk, -float("inf"))

    m_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(qk - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]                # 旧账先折算, 必须在 dot 之前
    acc = tl.dot(p.to(tl.float16), v, acc)
    return m_new, l_i, acc


@triton.jit
def flash_attn_causal_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    S, d, scale,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
):
    """一 program 处理 BM 行 query, 沿 S 方向分 BN 块循环, 只循环到对角线为止"""
    pid_m = tl.program_id(0) 
    start_m = pid_m * BM
    offs_m = start_m + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)

    q = tl.load(q_ptr + offs_m[:, None] * d + offs_d[None, :],
                mask=(offs_m[:, None] < S) & (offs_d[None, :] < d), other=0.0)

    m_i = tl.full((BM,), -float("inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)      # 宽是 BD, 不是 BN

    # 第一段: 整块都 <= start_m-1 < 所有 offs_m, 无条件通过 causal, 也不越界。
    # 这一段连 load 的 mask 都省了, cp.async 更干净。
    for n0 in range(0, start_m // BN):
        offs_n = n0 * BN + tl.arange(0, BN)
        m_i, l_i, acc = _attn_block(q, m_i, l_i, acc, k_ptr, v_ptr, offs_m, offs_n,
                                    offs_d, S, d, scale, BM, BN, BD, False)

    # 第二段: 对角块 (可能出现一到两块), 必须 mask。
    hi = tl.cdiv(tl.minimum(start_m + BM, S), BN)
    for n0 in range(start_m // BN, hi):
        offs_n = n0 * BN + tl.arange(0, BN)
        m_i, l_i, acc = _attn_block(q, m_i, l_i, acc, k_ptr, v_ptr, offs_m, offs_n,
                                    offs_d, S, d, scale, BM, BN, BD, True)

    tl.store(o_ptr + offs_m[:, None] * d + offs_d[None, :], acc / l_i[:, None],
             mask=(offs_m[:, None] < S) & (offs_d[None, :] < d))


def run(q, k, v, out):
    S, d = q.shape
    scale = 1.0 / (d ** 0.5)
    BD = triton.next_power_of_2(d)

    # BM=32 不是随手调的: causal 下 BM 直接决定关键路径长度 (见 docstring
    # "BM 的结论和第 14 题相反"), 32 是负载均衡和算术强度的交叉点。
    BM, BN = 32, 64
    grid = (triton.cdiv(S, BM),)
    # Turing(sm_75): 4 warps 寄存器溢出严重 (见第14题 diag_turing.py 实测),
    # 8 warps 把每线程 acc 负载减半, spills 骤降。Ampere+ 保持 4 不变。
    wm = 8 if torch.cuda.get_device_capability(q.device)[0] < 8 else 4
    flash_attn_causal_kernel[grid](
        q, k, v, out, S, d, scale, BM, BN, BD,
        num_warps=wm, num_stages=2)
