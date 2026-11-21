"""
19  Paged Attention (decode, 分页 KV cache)
    K/V cache 切成固定大小的物理块, 逻辑块经 block_table 映射到物理块

输入  Q            (Sq, d)              fp16  新 token 的 query (decode, Sq 很小)
      K, V         (NB, BS, d)          fp16  物理块池: NB 个块, 每块 BS 个 token
      block_table  (NLB,)               int32 逻辑块 i -> 物理块 block_table[i]
      seqlen       (1,)                 int32 实际有效 key 数 (标量装在 tensor 里)
输出  out          (Sq, d)              fp32
契约  run(q, k, v, block_table, seqlen, out), 内部启动 2 个 kernel

====================================================================
第一节  PagedAttention 到底解决什么问题
====================================================================

--- 1.1 连续 KV cache 的痛 ---

第 16/17 题的 K/V 是一整块连续内存 (Sk, d)。decode 时序列一个 token 一个 token
变长, 连续 cache 必须一开始就按 max_len 预留一整段。三个后果:

    * 浪费: 序列实际只生成了 100 个 token, 却占着 8192 的位置
    * 碎片: 多个长短不一的序列共享显存池, 连续分配产生外部碎片, 大块申请不到
    * 无法共享: 两个序列有相同前缀, 连续布局下没法让它们指向同一份 K/V

--- 1.2 分页: 把操作系统虚拟内存搬过来 ---

PagedAttention 的思路和 OS 分页一模一样:

    连续 cache:  [ t0 t1 t2 ... t8191 ]                    一整段连续
    分页 cache:  物理块 #3  <- 逻辑块 0 的 token
                 物理块 #17 <- 逻辑块 1 的 token            物理块乱序散落
                 物理块 #8  <- 逻辑块 2 的 token
                 ...

    block_table = [3, 17, 8, ...]   逻辑块号 -> 物理块号

序列按需要一块一块领物理块, 用完才归还。好处:
    * 按需分配, 不预留 max_len
    * 没有外部碎片 (块大小固定)
    * 不同序列的物理块可以共享 (同一 prompt 的前缀块只存一份)

代价: K/V 不再连续, 每读一块都要先查 block_table 做一次间接寻址。
**算子必须真的用这张表** —— 这正是本题要练的东西。

--- 1.3 为什么 make 里要把物理块打乱 ---

这是本题正确性测试的命门。如果 block_table 恰好是 [0,1,2,...] 恒等映射,
一个"假装分页、实际按物理顺序连续读"的错误 kernel 也能过测试。

所以 make 里用一个随机排列分配物理块, 且物理池比需要的更大 (留空闲块当碎片)。
这样忽略 block_table 的 kernel 必然读错数据 —— 分页是真是假, 一测便知。

====================================================================
第二节  变量词典
====================================================================

    中文名         符号          形状            定义式 / 含义
    ------------------------------------------------------------------
    物理块大小     BS           标量            = K.shape[1], 每块多少 token
    有效长度       seqlen       标量(tl.load)   实际 key 数, 最后一块可能不满
    逻辑块数       nlb          标量            = cdiv(seqlen, BS)
    段内块范围     b0,b1        标量            pid_s 这一份负责 [b0,b1) 个逻辑块
    物理块号       phys         标量            = block_table[blk], 间接寻址的结果
    块内偏移       offs_b       (BS,)           0..BS-1, 物理块内的 token 位置
    全局 key 位置  offs_t       (BS,)           = blk*BS + offs_b, 用于 mask seqlen
    query 块       q            (BM, BD)        从 Q[m, :] 读
    key 块         k            (BS, BD)        从 K[phys, offs_b, :] 读 <- 分页关键
    本块分数       qk           (BM, BS)        = q · k^T * scale
    滚动最大       m_i          (BM,)           到目前为止见过的最大分数
    滚动分母       l_i          (BM,)           = Σ exp(qk - m_i)
    滚动分子       acc          (BM, BD)        = Σ exp(qk - m_i) · v
    折扣系数       alpha        (BM,)           = exp(旧 m_i - 新 m_i)

====================================================================
第三节  段内怎么算 (分页在哪一步介入)
====================================================================

和第 16 题的唯一算法差别, 就在循环开头那两行寻址:

    for blk in range(b0, b1):
        phys   = tl.load(block_table_ptr + blk)        # 查表: 逻辑块 -> 物理块
        offs_b = tl.arange(0, BS)
        offs_t = blk * BS + offs_b                     # 逻辑上的全局 key 位置

        # K/V 从物理块 phys 读, 用 offs_t (不是 phys!) 判断哪些 token 有效
        k = tl.load(k_ptr + phys*(BS*d) + offs_b[:,None]*d + offs_d[None,:],
                    mask=(offs_t[:,None] < seqlen) & (offs_d[None,:] < d), other=0)
        v = tl.load(... 同理 ...)

        qk    = tl.dot(q, tl.trans(k)) * scale
        qk    = tl.where(offs_t[None,:] < seqlen, qk, -inf)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(qk - m_new[:, None])
        l_i   = l_i * alpha + tl.sum(p, 1)
        acc   = acc * alpha[:, None]                   # 旧账先折算
        acc   = tl.dot(p.to(fp16), v, acc)             # 再累加新块
        m_i   = m_new

注意两个 mask 用的是**逻辑位置 offs_t**, 不是物理块号:
    * 物理块 phys 里 BS 个槽位都存在 (分配时就留好了)
    * 但最后一个逻辑块往往只用了一部分, seqlen 之后的槽位是垃圾
    * 必须按 offs_t < seqlen 盖掉, 否则垃圾数据混进 softmax 分母

====================================================================
第四节  段间怎么合
====================================================================

和第 16 题完全一样: 每个 query 行跨 NSPLIT 段合并, 用 lse 加权。

    M    = max_s lse[s, 行]
    w_s  = exp(lse[s, 行] - M)
    out  = Σ_s w_s * o_s / Σ_s w_s

段总结 o_s = acc / l_i 必须是**已归一化**的, lse = m_i + log(l_i) 带分母。
只存这两个就能合并, 不用把 (Sq, seqlen) 的中间结果落盘。

====================================================================
第五节  五个坑 (尤其第 1 个是本题的灵魂)
====================================================================

1. **不查 block_table 必错**。make 把物理块打乱了, 连续读物理块 #0,#1,...
   得到的是乱序 token 的 attention, 结果面目全非。这不是"性能没优化",
   是"算法就是错的"。分页的实质是寻址方式, 不是把 cache 换成三维形状。

2. **最后一块按逻辑位置 mask, 不是按物理块号**。seqlen 不是 BS 整数倍时
   (本题 seqlen=1000, BS=64 → 最后一块只有 40 个有效), 用错 mask 会读进垃圾。

3. **空段 0/0 = nan 要兜**。`tl.where(l_i > 0, l_i, 1.0)`。

4. **seqlen 是 tensor 不是 python 整数**。kernel 里 `tl.load(seqlen_ptr)` 得到
   标量, 不能直接当 constexpr 用, 循环上界/范围都得是运行时的。

5. **BD = next_pow2(d)**。d 非 2 的幂 (odd hard 里 d=72/96) 时 arange(0,d) 不合法,
   必须 BD 取整再 mask。

6. **run() 里绝不能写 `int(seqlen.item())`**。这是本题实测踩过的大坑: seqlen 是
   tensor, `.item()` 触发一次 GPU->CPU 同步, bench 调上千次就慢 32 倍 ——
   实测 0.4582 ms vs 修正后 0.0143 ms, 掩盖了 kernel 的真实耗时。
   逻辑块数改从 `block_table.shape[0]` 读, 那是 python 层已知的元数据, 零同步。
   kernel 内部仍 `tl.load(seqlen_ptr)` 拿真实值做 mask —— 那里是 GPU 内部读, 不同步。

====================================================================
第六节  实测
====================================================================

    check.py --only 19_paged_attention (Sq=16, seqlen=8192, d=64, BS=64):
        0.0143 / 0.0154 / 0.0151 ms   2.18~2.34 TFLOPS   12.0~12.9% 峰值
        启动 2 次 (split + merge), 和第 16 题的 2 次一致。

    --odd hard 三组全过: (1,1000,72) / (3,5000,96) / (16,8192,128)。
    其中 seqlen=1000 不是 BS=64 整数倍 (最后一块只有 40 个有效 token),
    d=72/96 非 2 的幂 —— 这两个边界正是坑 2 和坑 5 的针对性测试。

    分页 vs 第 16 题连续 cache: 0.0143 ms vs 0.0174 ms, 同一量级。
    分页多了查表开销, 但因为只查 nblk=128 次标量 load, 几乎可以忽略。

    **反直觉数字验证**: 首次测出 0.4582 ms (比 16 题慢 26 倍), 按项目规矩重跑,
    定位到 run() 里的 .item() 同步 (见坑 6), 修正后稳定在 0.014~0.015 ms。
    另一次全量连跑出现过 0.3430 ms 的离群值, 当轮同时有 autograd 的
    "no current CUDA context" warning (GPU 状态被前面题搅乱); 重跑全量即回到
    0.0134 ms。四次独立测量 0.0134/0.0143/0.0151/0.0154 ms, 离群值不采信。

    **分页真实性验证** (_probe_19_paging.py, 已清理):
        block_table 128 项映射到 160 块物理池, 前几项 [80,150,15,92,...], 非恒等;
        忽略分页、按物理块 #0..127 连续读的假结果, 与正确值 max|diff|=0.0400,
        远超 tol(1e-2,2e-2) —— 不真查表的 kernel 必挂, 测试设计成立。

    decode 是 memory-bound: 算术强度 = Sq = 16 < ridge 45.4, TFLOPS 天生难看,
    拿它和 compute-bound 的 GEMM 比 12% 没有意义。真正的对手是第 16 题的连续版。
"""

import torch
import triton
import triton.language as tl

# 模块级 buffer 缓存: 按 (nsplit, sq, d, dev) 建键。全局字典不线程安全,
# 接框架时要改成调用方传 buffer (和第 16/17 题同一个约定)。
_BUF = {}


def _buffers(nsplit, sq, d, dev):
    key = (nsplit, sq, d, dev)
    b = _BUF.get(key)
    if b is None:
        op = torch.empty((nsplit, sq, d), device=dev, dtype=torch.float32)
        lse = torch.empty((nsplit, sq), device=dev, dtype=torch.float32)
        _BUF[key] = b = (op, lse)
    return b


@triton.jit
def _paged_split_kernel(
    q_ptr, k_ptr, v_ptr, bt_ptr, seqlen_ptr, op_ptr, lse_ptr,
    Sq, d, scale,
    BM: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr, NSPLIT: tl.constexpr,
):
    """一段逻辑块的 partial attention。K/V 经 block_table 间接寻址。"""
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)
    offs_b = tl.arange(0, BS)

    seqlen = tl.load(seqlen_ptr)                  # 标量, 运行时值
    nlb = tl.cdiv(seqlen, BS)                     # 逻辑块总数
    per = tl.cdiv(nlb, NSPLIT)
    b0 = pid_s * per
    b1 = tl.minimum(b0 + per, nlb)

    q = tl.load(
        q_ptr + offs_m[:, None] * d + offs_d[None, :],
        mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < d), other=0.0,
    )

    m_i = tl.full((BM,), -float("inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)

    for blk in range(b0, b1):
        # ---- 分页的核心: 逻辑块 -> 物理块, 再按物理块号读 K/V ----
        phys = tl.load(bt_ptr + blk)
        offs_t = blk * BS + offs_b
        mask_t = offs_t < seqlen

        k = tl.load(
            k_ptr + phys * (BS * d) + offs_b[:, None] * d + offs_d[None, :],
            mask=mask_t[:, None] & (offs_d[None, :] < d), other=0.0,
        )
        v = tl.load(
            v_ptr + phys * (BS * d) + offs_b[:, None] * d + offs_d[None, :],
            mask=mask_t[:, None] & (offs_d[None, :] < d), other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * scale        # (BM,BS)
        qk = tl.where(mask_t[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o = acc / l_safe[:, None]
    tl.store(
        op_ptr + (pid_s * Sq + offs_m[:, None]) * d + offs_d[None, :],
        o, mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < d),
    )
    lse = tl.where(l_i > 0, m_i + tl.log(l_i), -float("inf"))
    tl.store(lse_ptr + pid_s * Sq + offs_m, lse, mask=offs_m < Sq)


@triton.jit
def _paged_merge_kernel(
    op_ptr, lse_ptr, o_ptr,
    Sq, d,
    BM: tl.constexpr, BD: tl.constexpr, NSPLIT: tl.constexpr, NSP: tl.constexpr,
):
    """按 query 行合并所有段, 和第 16 题的 merge 一字不差。"""
    pid_m = tl.program_id(0)

    offs_s = tl.arange(0, NSP)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)
    mask_s = offs_s < NSPLIT
    mask_m = offs_m < Sq

    l_all = tl.load(
        lse_ptr + offs_s[:, None] * Sq + offs_m[None, :],
        mask=mask_s[:, None] & mask_m[None, :], other=-float("inf"),
    )
    M = tl.max(l_all, 0)
    M = tl.where(M > -float("inf"), M, 0.0)
    w = tl.where(mask_s[:, None], tl.exp(l_all - M[None, :]), 0.0)

    acc = tl.load(
        op_ptr + (offs_s[:, None, None] * Sq + offs_m[None, :, None]) * d
        + offs_d[None, None, :],
        mask=mask_s[:, None, None] & mask_m[None, :, None]
        & (offs_d[None, None, :] < d), other=0.0,
    )
    num = tl.sum(acc * w[:, :, None], 0)
    den = tl.sum(w, 0)
    den = tl.where(den > 0, den, 1.0)

    tl.store(
        o_ptr + offs_m[:, None] * d + offs_d[None, :],
        num / den[:, None],
        mask=mask_m[:, None] & (offs_d[None, :] < d),
    )


def run(q, k, v, block_table, seqlen, out, nsplit_override=None):
    """nsplit_override 只给调参探针用, 正常路径永远走自动推断。"""
    Sq, d = q.shape
    BS = k.shape[1]
    scale = 1.0 / (d ** 0.5)
    BD = triton.next_power_of_2(d)

    BM = 16
    # 逻辑块数从 block_table 的形状读, 这是 python 层已知的元数据, 零 GPU 同步。
    # 千万别写 int(seqlen.item()) —— 那会在每次 run() 里强制同步一次,
    # bench 调上千次就慢 26 倍, 而且掩盖了 kernel 的真实耗时 (本题实测踩过)。
    nlb = block_table.shape[0]

    nsm = torch.cuda.get_device_properties(q.device).multi_processor_count
    # 逻辑块数 nlb 决定最多能切几段; 再按 SM 数和 query 块数定目标。
    target = min(nlb, max(1, triton.cdiv(nsm, triton.cdiv(Sq, BM))))
    target = min(nlb, triton.next_power_of_2(target))
    nsplit = target
    if nlb % target != 0:
        for cand in range(target, 0, -1):
            if nlb % cand == 0 and cand * 2 >= target:
                nsplit = cand
                break
    if nsplit_override is not None:
        nsplit = nsplit_override
    nsp = triton.next_power_of_2(nsplit)

    op, lse = _buffers(nsplit, Sq, d, q.device)

    _paged_split_kernel[(triton.cdiv(Sq, BM), nsplit)](
        q, k, v, block_table, seqlen, op, lse,
        Sq, d, scale, BM, BS, BD, nsplit,
        num_warps=4, num_stages=2,
    )

    BM_M = 1
    while BM_M * 2 <= BM and triton.cdiv(Sq, BM_M * 2) >= 4 * nsm:
        BM_M *= 2

    _paged_merge_kernel[(triton.cdiv(Sq, BM_M),)](
        op, lse, out, Sq, d, BM_M, BD, nsplit, nsp,
        num_warps=4, num_stages=2,
    )
