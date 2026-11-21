"""
17  GQA Decode Attention
    16 个 query head 分成 4 组, 每组 4 个 q head 共享同一组 K/V

输入  Q      (HQ=16, Sq=16, d=64)  float16
      K, V   (HKV=4, Sk=8192, d=64) float16
输出  out    (HQ=16, Sq=16, d=64)  float32
契约  run(q, k, v, out), 内部启动 2 个 kernel

====================================================================
第一节  优化的原理
====================================================================

--- 1.1 GQA 到底是什么 ---

MHA (第 16 题那种):  每个 q head 有自己独立的 K/V head
GQA:                 每 GROUP 个 q head 共享一组 K/V head

    q head:   0   1   2   3  |  4   5   6   7  |  8 ...
    kv head:       0           |       1           |  2 ...
              映射到 h//GROUP

HQ=16, HKV=4, GROUP=4。K/V 的参数量只有 MHA 的 1/4 —— 这就是 GQA 存在的理由:
decode 时 KV cache 随生成长度线性增长, 把它压到 1/4 是实打实的显存和带宽节省。

--- 1.2 关键: 省下的带宽必须真的被算子拿到 ---

K/V 只存 4 份, 但 16 个 q head 都要用它。如果这样写:

    for h in range(16):
        attn(q[h], k[h//4], v[h//4])     # 每个 head 启动独立 program

K/V 会被读 16 遍 —— 和 MHA 一样。**参数省了, 带宽没省。**

所以 kernel 必须让共享同一 kv head 的 GROUP 个 q head 待在**同一个 program** 里:
每块 K/V 只 load 一次, 分给 GROUP 个 q head 各算一次 qk。

    grid = (HKV, cdiv(Sq,BM), NSPLIT) = (4, 1, 16)   -> 64 个 block
    vs 第 16 题的 (1, 1, 16) = 16 个 block

顺带的好处: block 数从 16 涨到 64, 20 个 SM 全占满, 不再有 "19 个 SM 睡觉" 的问题。

--- 1.3 代价: partial buffer 变大 ---

每个 q head 都要单独存段总结, buffer 第一维从 NSPLIT 变成 (NSPLIT, HQ):

    op   (NSPLIT, HQ, Sq, d)  fp32   16*16*16*64*4 = 1 MB
    lse  (NSPLIT, HQ, Sq)     fp32   16 KB

比第 16 题的 64 KB 大 16 倍 —— 这是 GQA 的账: 省了 K/V 的读, 多了 partial 的写。
不过 1 MB 相对于 8 MB 的 K/V 访存仍然划算。

====================================================================
第二节  变量词典
====================================================================

    中文名           符号       形状              定义式 / 含义
    ----------------------------------------------------------------------
    组大小           GROUP      标量              = HQ / HKV, 这里 4
    kv head 号       kv_h       标量              pid0, 0..HKV-1
    q head 号        hq          (GROUP,)         kv_h*GROUP + arange(GROUP)
    query 块          q          (GROUP, BM, BD)  从 Q[hq, m, :] 读, 一次读 GROUP 个 head
    key 块            k          (BN, BD)        从 K[kv_h, n, :] 读, GROUP 个 head 共用
    本块分数          qk         (GROUP, BM, BN)  = q · k^T * scale
    组内最大分数      m_i        (GROUP, BM)      滚动量, = 到目前为止见过的最大
    组内分母          l_i        (GROUP, BM)      = Σ exp(qk - m_i)
    组内分子          acc        (GROUP, BM, BD)  = Σ exp(qk - m_i) · v
    折扣系数          alpha      (GROUP, BM)      = exp(旧 m_i - 新 m_i)
    段总结            o          (GROUP, BM, BD)  = acc / l_i, 已归一化
    对数分母          lse        (GROUP, BM)      = m_i + log l_i

注意 m_i / l_i / lse 都是**每个 q head、每行 query 各一个标量**, 不是一整块矩阵。
第 0 维是 GROUP, 因为这一组里的 4 个 head 各自独立算 softmax, 只是共享 K/V。

====================================================================
第三节  段内怎么算
====================================================================

和第 16 题唯一的算法差别: 状态多了一维 (GROUP,)。

    q  = tl.load(...)                       # (GROUP, BM, BD)
    m_i  = full((GROUP,BM), -inf)
    l_i  = zeros((GROUP,BM))
    acc  = zeros((GROUP,BM,BD))

    for n0 in range(b0, b1):
        k = tl.load(...)                    # (BN, BD)  <- 只读一次, 4 个 head 共用
        v = tl.load(...)                    # (BN, BD)

        qk = tl.dot(q, trans(broadcast(k))) * scale    # (GROUP, BM, BN)
        m_new = maximum(m_i, max(qk, axis=2))          # (GROUP, BM)
        alpha = exp(m_i - m_new)
        p = exp(qk - m_new[:, :, None])
        l_i = l_i * alpha + sum(p, axis=2)
        acc = acc * alpha[:, :, None]                  # 先折旧账
        acc = tl.dot(p.to(fp16), broadcast(v), acc)    # 再累加
        m_i = m_new

broadcast(k) 把 (BN,BD) 复制成 (GROUP,BN,BD), 让 tl.dot 做 batched matmul。
这是 4 次矩阵乘打包成一次调用, K/V 的数据在寄存器里实打实复用 4 次。

`m_i` 是滚动量: 处理第 0 块时只是那块的最大值, 循环结束才是整段最大值。
为什么必须打折: 分母是 Σ exp(x - m), m 变了基准就变了, 之前算的 exp 全偏大,
必须统一乘 exp(m_old - m_new)。分子分母乘同一个数, 商不变。

====================================================================
第四节  段间怎么合
====================================================================

和第 16 题完全一样, 只是多了一维 head: 每个 (hq, 行) 各自做一次合并。

    M    = max_s lse[s, hq, 行]             对所有段取最大对数分母
    w_s  = exp(lse[s, hq, 行] - M)          这一段的份额 = Z_s / exp(M)
    out  = Σ_s w_s * o_s / Σ_s w_s         分子加分子, 分母加分母

`max_s` 不是变量, 是"对 s 取最大值"的记号, 和 Σ_s 一样。
`lse` = log Z, 就是 softmax 分母取对数 —— 只在跨段相加分母时才用得上。

为什么份额和段总结必须一个含分母、一个不含:
exp(lse_s - M) = l_s * exp(m_s - M) 里已经带了 l_s, 如果 o_s 再用未归一化的 acc,
等于分母乘两次。第 16 题推错过一次, 症状是"误差 ∝ 段长", 这里直接照对的写。

====================================================================
第五节  实测与坑
====================================================================

--- 5.1 共享 K/V 到底值多少: 3.36 倍 ---

对照实验 (_probe_17_shared_vs_naive.py, CUDA Graph 口径, 顺序打乱三轮):

    A 共享 K/V (本题, grid=(HKV,1,16)=64 blocks)     70.8 us
    B 逐 head 独立读 (拿 16 题的 kernel 跑 16 次)     238.2 us
    两条路径输出 max|diff| = 1.86e-08  (同一份数学, 只差访存)

两者的 FLOPs 一模一样 (536.9 MFLOP), 唯一差别是 K/V 读 4 遍还是 16 遍:

    共享: 4 * 8192 * 64 * 2(K,V) * 2B   =  8.4 MB
    逐头: 16 * 8192 * 64 * 2 * 2B        = 33.6 MB   <- 4 倍流量

算术强度正好跨过 roofline 平衡点:

    A: 536.9 MFLOP / 8.4 MB  = 64.0 FLOP/Byte  > 45.4  -> compute-bound
    B: 536.9 MFLOP / 33.6 MB = 16.0 FLOP/Byte  < 45.4  -> memory-bound

**这就是 GQA 的全部意义**: 不是少算, 是把算术强度从带宽受限区推到算力受限区。
参数量少 4 倍只是前提; 算子必须真的复用 K/V, 否则那 4 倍一分拿不到。

--- 5.2 check.py 成绩与 SDPA 对照的陷阱 ---

    17_gqa_decode    0.0624 ms   8.61 TFLOPS   47.3%   启动 2 次  (NSPLIT=8)
    --odd / --odd hard 全部通过 (含 Sk=1000 / d=72 / Sq=3)

**"超 SDPA 3813%" 这个数不要当成绩讲。** PyTorch 的 SDPA 在 HQ != HKV 时
开了 enable_gqa=True 会掉到 math 后端 (2.67 ms), 那是"没为这个 shape 优化",
不是"优化得差"。真正的对手是自己手写的逐 head 版 (5.1 的 238 us)。

--- 5.3 NSPLIT 与 merge 的 BM ---

NSPLIT 选法和 16 题**不一样**, 因为 grid 第一维已经有 HKV=4 个 block:
    target = cdiv(nsm, cdiv(Sq,BM) * HKV) = cdiv(20, 1*4) = 5  -> 向上取 2 的幂 = 8
总 block = 4*8 = 32, 已经把 20 个 SM 填满。
实测 (_probe_17_nsplit.py, 三个独立进程各 4 次, 走 run(nsplit_override=...)):

    NSPLIT= 4 (16 blocks)    75.5 us
    NSPLIT= 8 (32 blocks)    64.3 us   <- 最优, 现在这个
    NSPLIT=16 (64 blocks)    71.1 us
    NSPLIT=32 (128 blocks)   80.0 us
    NSPLIT=64 (256 blocks)  104.5 us

16 段比 8 段慢 9.6%: 每条关键路径短一半, 但固定成本翻一倍, 而 32 blocks 已经
够填满 SM。**16 题选 16 段是因为只有 1 个 head 维; 17 题的 head 维自己就贡献了
4 倍并行, key 方向就不该再切那么碎** —— 同一套问题, 参数最优值又反过来了。

merge 的 BM 不跟 split 共用。注意 run() 的条件 `cdiv(Sq,BM_M*2)*HQ >= 4*nsm`
在 HQ=16 时会把 BM_M 从 1 顶到 2 (16*8=128 >= 80), 和 16 题 (HQ=1, BM_M=1)
不同。写探针直接调 kernel 时如果手填 BM_M=1, 会看到 7.7e-2 的系统偏差 ——
这不是 kernel 的 bug (单独测 1/2/4/8/16 全是 1.37e-05), 是探针没复现 run() 的
参数选择。**照 run() 调, 别手抄参数。**

--- 5.4 两个静默 bug 的坑 ---

空段的 0/0 = nan 必须兜: `tl.where(l_i > 0, l_i, 1.0)`, 因为 nan * 0 还是 nan,
一个空段能污染整个输出。

按块切不按元素切: 按元素切会让相邻段重叠, 那些 key 的 exp 被算两次, 分母偏大。
Sk 恰好是 NSPLIT*BN 的整数倍时会静默通过, 只有不整除的尺寸才露馅。

--- 5.5 一个只在 Colab T4 上炸的坑: 3D batched dot 的 shared memory ---

本地 3050 Ti (Ampere sm_86) 全量 7/7 通过, 丢到 Colab 的 T4 (Turing sm_75) 却在
第 17 题崩 `OutOfResources: Required 81920, Hardware limit 65536` —— 需要 80 KB,
T4 只有 64 KB。

诡异的是同一份 kernel 在本地编译只报 18 KB shared。差 4 倍多, 正好是 GROUP=4。
原因在 MMA 布局: Ampere 上 `broadcast(k)->(GROUP,BN,BD)` 是零拷贝 view, 4 个 head
的 batched dot 共用同一份 K/V 物理内存; 而 Turing 的 MMA 指令格式不支持这种
batch 维, 编译器把 GROUP 维**真的复制展开**成 4 份, shared 直接 ×4 = 72~80 KB,
超上限。

修法是换等价的 2D 写法 (`_split_kernel_2d`): 把 GROUP 个 head 的 query 行沿 M 维
拼成 (GROUP*BM, BD) = (64, 64), K/V 保持 (BN, BD) 走标准 2D dot。省带宽的语义
一点没丢 —— K/V 依然只读一次、4 个 head 共用, 只是从"一次 batched dot"变成
"一次更大的 2D dot", 编译器不再需要展开 batch 维。实测两边输出逐字节一致
(3D vs 2D max diff = 0; 2D+merge 对 fp32 参考 max diff = 1.4e-05), shared 从
80 KB 降到 24 KB。

run() 里按 `get_device_capability()[0]` 分派: Ampere+ 走更快的 3D 版,
Turing 及更早走 2D 版。**这是"同一份数学、不同硬件走不同编译路径"的典型案例**:
不是算法错了, 是编译器对同一写法的资源代价因架构而异。面试可讲。
"""

import torch
import triton
import triton.language as tl

# 模块级 buffer 缓存: 按 (hq, hkv, sq, sk, d, nsplit, dev) 建键。
# 注意是全局字典, 不线程安全 —— 接框架时要改成调用方传 buffer。
_BUF = {}


def _buffers(nsplit, hq, sq, d, dev):
    key = (nsplit, hq, sq, d, dev)
    b = _BUF.get(key)
    if b is None:
        op = torch.empty((nsplit, hq, sq, d), device=dev, dtype=torch.float32)
        lse = torch.empty((nsplit, hq, sq), device=dev, dtype=torch.float32)
        _BUF[key] = b = (op, lse)
    return b


@triton.jit
def _split_kernel(
    q_ptr, k_ptr, v_ptr, op_ptr, lse_ptr,
    HQ, HKV, Sq, Sk, d, scale,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    GROUP: tl.constexpr, NSPLIT: tl.constexpr,
):
    """一段 KV 的 partial attention。GROUP 个 q head 共享同一块 K/V。"""
    kv_h = tl.program_id(0)          # 第几组 K/V head
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_g = tl.arange(0, GROUP)
    hq_idx = kv_h * GROUP + offs_g   # 这一组里的 4 个 q head 编号
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)

    # (GROUP, BM, BD) —— 4 个 q head 一次性读进来
    q = tl.load(
        q_ptr + hq_idx[:, None, None] * (Sq * d)
        + offs_m[None, :, None] * d + offs_d[None, None, :],
        mask=(offs_m[None, :, None] < Sq) & (offs_d[None, None, :] < d),
        other=0.0,
    )

    m_i = tl.full((GROUP, BM), -float("inf"), tl.float32)
    l_i = tl.zeros((GROUP, BM), tl.float32)
    acc = tl.zeros((GROUP, BM, BD), tl.float32)

    nblk = tl.cdiv(Sk, BN)
    per = tl.cdiv(nblk, NSPLIT)
    b0 = pid_s * per
    b1 = tl.minimum(b0 + per, nblk)

    for n0 in range(b0, b1):
        offs_n = n0 * BN + tl.arange(0, BN)
        mask_n = offs_n < Sk

        # 关键: K/V 只按 kv_h 读一次, GROUP 个 head 共用 (broadcast 后做 batched dot)
        k = tl.load(
            k_ptr + kv_h * (Sk * d) + offs_n[:, None] * d + offs_d[None, :],
            mask=mask_n[:, None] & (offs_d[None, :] < d), other=0.0,
        )
        v = tl.load(
            v_ptr + kv_h * (Sk * d) + offs_n[:, None] * d + offs_d[None, :],
            mask=mask_n[:, None] & (offs_d[None, :] < d), other=0.0,
        )
        k3 = tl.broadcast_to(k[None, :, :], (GROUP, BN, BD))
        v3 = tl.broadcast_to(v[None, :, :], (GROUP, BN, BD))

        qk = tl.dot(q, tl.trans(k3)) * scale                  # (GROUP, BM, BN)
        qk = tl.where(mask_n[None, None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 2))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, :, None])
        l_i = l_i * alpha + tl.sum(p, 2)
        acc = acc * alpha[:, :, None]
        acc = tl.dot(p.to(tl.float16), v3, acc)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o = acc / l_safe[:, :, None]

    # op: (NSPLIT, HQ, Sq, d)
    op_ptrs = (op_ptr
               + (pid_s * HQ + hq_idx[:, None, None]) * (Sq * d)
               + offs_m[None, :, None] * d + offs_d[None, None, :])
    tl.store(op_ptrs, o,
             mask=(offs_m[None, :, None] < Sq) & (offs_d[None, None, :] < d))

    lse = tl.where(l_i > 0, m_i + tl.log(l_i), -float("inf"))
    tl.store(lse_ptr + (pid_s * HQ + hq_idx[:, None]) * Sq + offs_m[None, :],
             lse, mask=offs_m[None, :] < Sq)


@triton.jit
def _split_kernel_2d(
    q_ptr, k_ptr, v_ptr, op_ptr, lse_ptr,
    HQ, HKV, Sq, Sk, d, scale,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    GROUP: tl.constexpr, NSPLIT: tl.constexpr,
):
    """3D batched dot 的低共享内存版本 (给 Turing sm_75 等老卡)。

    上面的 _split_kernel 把 K/V broadcast 成 (GROUP,BN,BD) 做 3D batched dot。
    Ampere+ 上这是零拷贝 view, shared 只要 18 KB; 但 Turing 的 MMA 布局会把
    GROUP 维真的展开, 实测需要 80 KB —— 超过 T4 的 64 KB 上限直接崩。

    这里换一种等价写法: 把 GROUP 个 q head 的行沿 M 维拼成 (GROUP*BM, BD),
    K/V 保持 (BN,BD) 用标准 2D dot。K/V 依然只读一次、GROUP 个 head 共用,
    省带宽的语义完全不变; shared 降到 24 KB, 且与 3D 版数值逐字节一致。
    """
    kv_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)

    M: tl.constexpr = GROUP * BM
    offs_r = tl.arange(0, M)
    g = offs_r // BM                 # 组内第几个 q head
    m_loc = offs_r % BM
    hq = kv_h * GROUP + g
    offs_m = pid_m * BM + m_loc
    offs_d = tl.arange(0, BD)

    q = tl.load(
        q_ptr + hq[:, None] * (Sq * d) + offs_m[:, None] * d + offs_d[None, :],
        mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < d), other=0.0,
    )

    m_i = tl.full((M,), -float("inf"), tl.float32)
    l_i = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, BD), tl.float32)

    nblk = tl.cdiv(Sk, BN)
    per = tl.cdiv(nblk, NSPLIT)
    b0 = pid_s * per
    b1 = tl.minimum(b0 + per, nblk)

    for n0 in range(b0, b1):
        offs_n = n0 * BN + tl.arange(0, BN)
        mask_n = offs_n < Sk
        k = tl.load(
            k_ptr + kv_h * (Sk * d) + offs_n[:, None] * d + offs_d[None, :],
            mask=mask_n[:, None] & (offs_d[None, :] < d), other=0.0,
        )
        v = tl.load(
            v_ptr + kv_h * (Sk * d) + offs_n[:, None] * d + offs_d[None, :],
            mask=mask_n[:, None] & (offs_d[None, :] < d), other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * scale          # (M, BN), 标准 2D
        qk = tl.where(mask_n[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o = acc / l_safe[:, None]
    tl.store(op_ptr + (pid_s * HQ + hq[:, None]) * (Sq * d)
             + offs_m[:, None] * d + offs_d[None, :], o,
             mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < d))
    lse = tl.where(l_i > 0, m_i + tl.log(l_i), -float("inf"))
    tl.store(lse_ptr + (pid_s * HQ + hq) * Sq + offs_m, lse, mask=offs_m < Sq)


@triton.jit
def _merge_kernel(
    op_ptr, lse_ptr, o_ptr,
    HQ, Sq, d,
    BM: tl.constexpr, BD: tl.constexpr, NSPLIT: tl.constexpr, NSP: tl.constexpr,
):
    """按 (q head, query 行) 合并所有段。同一套 lse 加权, 只是多了一维 head。"""
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_s = tl.arange(0, NSP)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)
    mask_s = offs_s < NSPLIT
    mask_m = offs_m < Sq

    l_all = tl.load(
        lse_ptr + offs_s[:, None] * (HQ * Sq) + pid_h * Sq + offs_m[None, :],
        mask=mask_s[:, None] & mask_m[None, :], other=-float("inf"),
    )
    M = tl.max(l_all, 0)
    M = tl.where(M > -float("inf"), M, 0.0)
    w = tl.where(mask_s[:, None], tl.exp(l_all - M[None, :]), 0.0)

    acc = tl.load(
        op_ptr + (offs_s[:, None, None] * HQ + pid_h) * (Sq * d)
        + offs_m[None, :, None] * d + offs_d[None, None, :],
        mask=mask_s[:, None, None] & mask_m[None, :, None] & (offs_d[None, None, :] < d),
        other=0.0,
    )
    num = tl.sum(acc * w[:, :, None], 0)
    den = tl.sum(w, 0)
    den = tl.where(den > 0, den, 1.0)

    tl.store(o_ptr + (pid_h * Sq + offs_m[:, None]) * d + offs_d[None, :],
             num / den[:, None],
             mask=mask_m[:, None] & (offs_d[None, :] < d))


def run(q, k, v, out, nsplit_override=None):
    """nsplit_override 只给调参探针用, 正常路径永远走自动推断。"""
    HQ, Sq, d = q.shape
    HKV, Sk, _ = k.shape
    GROUP = HQ // HKV
    scale = 1.0 / (d ** 0.5)
    BD = triton.next_power_of_2(d)

    BM, BN = 16, 64
    nsm = torch.cuda.get_device_properties(q.device).multi_processor_count

    nblk = triton.cdiv(Sk, BN)
    # 和第 16 题的关键区别: grid 第一维已经有 HKV=4 个 block 了,
    # 所以 key 方向不必再切满 20 段。先除掉 head 维占用的份数, 再向上取 2 的幂。
    # 实测 8 段 (32 blocks) 比 16 段 (64 blocks) 快 9.6%: 段数多意味着每条关键路径
    # 短一半, 但固定成本多一倍, 而 32 blocks 已经把 20 个 SM 填满了。
    target = min(nblk, max(1, triton.cdiv(nsm, triton.cdiv(Sq, BM) * HKV)))
    target = min(nblk, triton.next_power_of_2(target))
    nsplit = target
    if triton.cdiv(nblk, target) * target > nblk:
        for cand in range(target, 0, -1):
            if nblk % cand == 0 and cand * 2 >= target:
                nsplit = cand
                break
    if nsplit_override is not None:
        nsplit = nsplit_override
    nsp = triton.next_power_of_2(nsplit)

    op, lse = _buffers(nsplit, HQ, Sq, d, q.device)

    # Turing (sm_75, 如 T4) 的 MMA 布局会把 3D batched dot 的 GROUP 维真的展开,
    # shared 从 Ampere 的 18 KB 暴涨到 80 KB, 超 T4 的 64 KB 上限直接崩。
    # 老卡改用等价的 2D 拼行版 (shared 24 KB), 数值逐字节一致。
    cap_major = torch.cuda.get_device_capability(q.device)[0]
    split_fn = _split_kernel if cap_major >= 8 else _split_kernel_2d
    split_fn[(HKV, triton.cdiv(Sq, BM), nsplit)](
        q, k, v, op, lse,
        HQ, HKV, Sq, Sk, d, scale,
        BM, BN, BD, GROUP, nsplit,
        num_warps=4, num_stages=2,
    )

    BM_M = 1
    while BM_M * 2 <= BM and triton.cdiv(Sq, BM_M * 2) * HQ >= 4 * nsm:
        BM_M *= 2

    _merge_kernel[(HQ, triton.cdiv(Sq, BM_M))](
        op, lse, out, HQ, Sq, d,
        BM_M, BD, nsplit, nsp,
        num_warps=4, num_stages=2,
    )
