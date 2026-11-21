"""
14  FlashAttention          out = softmax(Q Kᵀ / √d) V, 一趟算完

Q, K, V : (2048, 64) float16
out     : (2048, 64) float32
契约: run(q, k, v, out)

和第 13 题算的是**同一个东西**, 输入输出完全一样。区别只有一个:
第 13 题把 (S,S) 的中间矩阵写进 HBM 再读回来, 这题一次都不落盘。

= 为什么这件事不平凡 =

softmax 是个"整行"运算: 要除以 sum_j exp(s_j - m), 其中 m 是整行的最大值。
可是分块看的时候, 你手里只有一行的一部分 —— 算完第一块的时候, 你根本不知道
后面几块里会不会出现一个更大的值把 m 顶掉。顶掉了, 前面累加的 acc 就全废了。

第 13 题的解法是"等": 先把整行写完, 再回来扫。代价就是那个 (S,S) 矩阵。

FlashAttention 的解法是"改":
不要求 m 是**全局**最大, 只要求它"到目前为止最大"。
等来了更大的, 就把之前的账按比例折算一下 —— 这叫 **online softmax**。

= 核心恒等式 =

     exp(s - m_new) = exp(s - m_old) * exp(m_old - m_new)
                                    ~~~~~~~~~~~~~~~~~~~~~
                                     一个和 s 无关的标量 alpha

所以当最大值从 m_old 变成 m_new 时, 之前算的所有指数值只要**统一乘 alpha** 就修正了。
acc 是这些指数值的加权和, 所以 acc 也乘同一个 alpha。

= 三个状态变量 (每个 program 只管 BM 行 query) =

    m_i  (BM,)      这 BM 行各自到目前为止见过的最大值, 初值 -inf
    l_i  (BM,)      这 BM 行各自到目前为止的指数和, 初值 0
    acc  (BM, BD)  这 BM 行到目前为止的输出累加, 初值 0

= 每一轮的五个动作 (按这个顺序, 顺序错了就错) =

    1.  qk   = dot(q, k_blockᵀ) * scale            (BM, BN)
    2.  qk   在 S 方向越界的位置填 -inf            <- 不是 0!
    3.  m_new = max(m_i, max(qk, axis=1))           (BM,)
        alpha = exp(m_i - m_new)                    (BM,)
        p     = exp(qk - m_new[:, None])            (BM, BN)
        l_i   = l_i * alpha + sum(p, axis=1)
        acc   = acc * alpha[:, None]                <- 旧的账先折算
        acc   = dot(p.to(fp16), v_block, acc)       <- 再把新块加上去
        m_i   = m_new

最后 out = acc / l_i[:, None]。**除号在这里, 不在循环里** —— 这是和 13 题最大的结构差别:
13 题在 softmax kernel 里就把 p 归一化了, 这题一直带着未归一化的分子, 出门前才除。

= 七个坑 =

1. **mask 填 -inf, 不是 0。**
   S 方向尾部 (offs_n >= S) 的位置, qk 必须填 -inf。
   填 0 的话 exp(0 - m) 是个正数, 会混进 l_i 的分母, 整行被稀释。
   (第 04 题 `other=-inf` 的同一个教训, 这里换成 `tl.where`。)

2. **alpha 的广播方向。**
   alpha 是 (BM,), acc 是 (BM, BD), 要写成 `acc * alpha[:, None]`。
   写成 `acc * alpha` 在 Triton 里会按最后一维广播, 形状对不上或者静默错。

3. **p 要转 fp16 才能进 dot。**
   `tl.exp` 出来是 fp32, `tl.dot` 只吃 fp16/bf16。转的时候是 `p.to(tl.float16)`,
   不要 `p.to(tl.float32)` (第 11 题的 TF32 陷阱: 那是降级不是升级)。

4. **acc 的折算必须在 dot 之前。**
   顺序是 `acc = acc * alpha; acc = dot(p, v, acc)`。
   反过来的话, 新加进来的这块也被 alpha 乘了, 而它本来不该被折算。

5. **BD 必须 >= 16。**
   `tl.dot` 的三维都要求 >= 16, 而 d=64 没问题; 但 `--odd hard` 里有 d=72 这种,
   BD 要 `triton.next_power_of_2(d)` 再 mask, 不能直接 `tl.arange(0, d)`。

6. **第一个 dot 别漏 trans。**
   q 是 (BM,BD), k 是 (BN,BD), 写成 `tl.dot(q, k)` 不报错 —— 因为 d=64 时
   BM/BN/BD 全是 64, 形状检查过得了。但它收缩的是 q 的 d 和 k 的 n, 语义错位。
   实测 max|diff| = 54.8 (探针 `_probe_ktrans.py`)。
   这一条特别阴: **只有维度互不相等时才会报错**。`--odd hard` 里有 d=72
   (BD=128), 那时漏 trans 会直接编译失败 —— 所以边界尺寸是你的朋友。

7. **acc 的形状是 (BM, BD), 不是 (BM, BN)。**
   d=64 时 BM=BN=BD 全等于 64, 于是 qk / p / acc 三个张量形状一模一样,
   写成 `tl.zeros((BM, BN))` 照样跑通, 而且**结果完全正确**
   (实测 max|diff| = 7.67e-05, 和正确版一字不差)。
   因为那时 (BM,BN) 和 (BM,BD) 就是同一个形状, 不存在"错"这回事。
   只有 d 不是 2 的幂时才露馅: d=72 -> BD=128, acc 宽一格,
   `tl.dot(p, v, acc)` 直接编译失败。

   这条和坑 6 是同一个机制: **形状检查只能挡住"宽度不等",
   挡不住"宽度相等但语义不同"**。两个 dot 的收缩维必须自己心里有数:
       dot(q, kᵀ)  (BM,BD) x (BD,BN) -> (BM,BN)   收缩的是 d
       dot(p, v)   (BM,BN) x (BN,BD) -> (BM,BD)   收缩的是 n
   探针 `_probe_fa_shapes.py` 里用 BUG 开关复现了这两种表现。

= 一个容易忽略的细节 =

第一轮时 m_i = -inf, m_new 是个有限值, alpha = exp(-inf - 有限值) = 0。
于是 acc * 0 = 0、l_i * 0 = 0 —— 初值被自动清掉, 不需要特殊处理第一轮。
但如果你把 m_i 的初值写成 0, 那么当整行 logit 都是负数时, 第一轮的 alpha 就错了
(exp(0 - m_new) 会 > 1, 把不存在的历史放大)。**初值必须 -inf。**

= 目标与实测 =

    13_attn_naive   0.2013 ms   5.33 TFLOPS   29.3%   启动 3 次
    torch SDPA      0.1270 ms   8.46 TFLOPS   46.5%
    14_flash_attn   0.0873 ms  12.30 TFLOPS   67.6%   启动 1 次

一次都没落盘, 换来 2.3 倍于 13 题、1.45 倍于 SDPA。

**这个数字只在冷机单独跑时成立。** 全量连跑 14 道题时掉到 0.1000 ms / 59.0% ——
前 13 题已经把 3050 Ti 这笔记本卡的 45 W 功耗墙顶满了, SM 降频。有意思的是 13 题
几乎不受影响 (0.2013 -> 0.2007, 差 0.3%), 因为它是 memory-bound, 带宽不吃 SM 频率;
14 题是 compute-bound, 降频直接掉在 TFLOPS 上。
**同一个 check.py, 单独跑和全量跑能差 15%, 这是"一个配置一个进程"之外的第二条口径陷阱。**

这里要收回一个预测: 写骨架时我断言"别指望超过 SDPA, 能和它差 15% 以内就算写对了"。
错了 —— 这个尺寸下反超了 45%。但这不说明 Triton 比 CUDA 手写强, 而是
SDPA 要通吃 causal / 各种 mask / GQA / 变长 / fp32-bf16-fp8, 还要在十几代
架构上选 kernel, 通用性有代价; 这个 kernel 只认 d%16==0 的 single-head 无 mask
场景, BM/BN/warps/stages 又是专门为 S=2048 调过的。
**拿专用换通用, 数字好看是应该的, 不代表工程上更优。**

= 调参: 67.6% 基本就是这个结构的顶 =

探针口径 (比 check.py 低 8% 左右), S=2048 d=64, 每个配置独立进程:

    BM=32   BN=64   w4 s2   55.9%    grid=64, 3.2 轮
    BM=64   BN=64   w4 s2   63.0%    grid=32, 1.6 轮   <- 现在这个
    BM=128  BN=64   w4 s2   50.6%    grid=16, 0.8 轮, 4 个 SM 全程闲着
    BM=64   BN=32   w4 s2   54.3%
    BM=64   BN=128  w4 s2   52.5%
    BM=64   BN=64   w8 s2   29.3%    腰斩
    BM=64   BN=64   w4 s3   57.4%

两条反直觉的:

**BM=32 更慢。** block 数从 32 涨到 64, 波次填充率明明变好了 (3.2 轮 vs 1.6 轮),
但 tile 小了算术强度跟着掉, 循环轮数从 32 涨到 64, 每轮的定长开销全摊在更小的
计算量上。和 GEMM 那题"小 tile 掉一半"是同一个道理: **填充率和算术强度在打架。**

**warps=8 腰斩。** 64x64 的 tile 原来 4 个 warp 各分 32x32, 正好是 mma 片段的
整数倍; 拆成 8 个 warp 每个只拿 16x16, 单条 mma 的填充率掉下来, 调度开销反而
占比变大。**线程多不等于算得快**, 要看能不能整除到 mma 那块砖上。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    S, d, scale,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
):
    """一 program 处理 BM 行 query, 沿 S 方向分 BN 块循环 K/V

    参数里没有 L (中间矩阵), 因为没有中间矩阵。
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)

    # q 只搬一次: 这个 program 负责的 BM 行 query, 全程留在寄存器里
    q = tl.load(q_ptr + offs_m[:, None] * d + offs_d[None, :],
                mask=(offs_m[:, None] < S) & (offs_d[None, :] < d), other=0.0)

    # 三个状态变量, 每个 query 行各一份, 所以都是 (BM, ...)
    m_i = tl.full((BM,), -float("inf"), tl.float32)   # 到目前为止的行最大值
    l_i = tl.zeros((BM,), tl.float32)                 # 到目前为止的指数和
    acc = tl.zeros((BM, BD), tl.float32)              # 宽是 BD, 不是 BN

    for n0 in range(0, tl.cdiv(S, BN)):
        offs_n = n0 * BN + tl.arange(0, BN)

        # 每轮只把当前这一块 K/V 搬进 SRAM, 转头就用掉
        k = tl.load(k_ptr + offs_n[:, None] * d + offs_d[None, :],
                    mask=(offs_n[:, None] < S) & (offs_d[None, :] < d), other=0.0)
        v = tl.load(v_ptr + offs_n[:, None] * d + offs_d[None, :],
                    mask=(offs_n[:, None] < S) & (offs_d[None, :] < d), other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale       # (BM,BD) x (BD,BN) -> (BM,BN)

        # 尾块补 -inf, 不是 0: 填 0 的话 exp(0 - m) 是个正数, 会混进分母
        qk = tl.where(offs_n[None, :] < S, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))    # (BM,) 逐元素取大, 不是 tl.max
        alpha = tl.exp(m_i - m_new)               # 旧账的折算系数
        p = tl.exp(qk - m_new[:, None])            # 分子
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]                # 旧账先折算, 必须在 dot 之前
        acc = tl.dot(p.to(tl.float16), v, acc)    # (BM,BN) x (BN,BD) -> (BM,BD)
        m_i = m_new

    tl.store(o_ptr + offs_m[:, None] * d + offs_d[None, :], acc / l_i[:, None],
             mask=(offs_m[:, None] < S) & (offs_d[None, :] < d))


def run(q, k, v, out):
    S, d = q.shape
    scale = 1.0 / (d ** 0.5)
    BD = triton.next_power_of_2(d)

    BM, BN = 64, 64
    grid = (triton.cdiv(S, BM),)
    # Turing(sm_75): 4 warps 时 acc 顶满 255 regs 溢出 410 次到 local memory,
    # 8 warps 把每线程负载减半, spills 410→58、实测 2.97→1.82 ms (diag_turing.py)。
    # Ampere+ 保持 4 不变。8 warps 不改变 shared (都是 32 KB), stages=2 实测最优。
    wm = 8 if torch.cuda.get_device_capability(q.device)[0] < 8 else 4
    flash_attn_kernel[grid](
        q, k, v, out, S, d, scale, BM, BN, BD,
        num_warps=wm, num_stages=2)
