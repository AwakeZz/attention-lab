"""
16  Split-KV Decode Attention   把 8192 个 key 切给 16 个 program, 各算一份再合并

输入  Q      (16, 64)   float16    <- decode: 一次只来 16 个新 token
      K, V   (8192, 64) float16    <- KV cache, 长
输出  out    (16, 64)   float32
契约  run(q, k, v, out), 内部启动 2 个 kernel

------------------------------------------------------------------
怎么读这个文件

    第一节  优化的原理    问题在哪、手段是什么、代价是什么   <- 不懂就看这里
    第二节  变量词典      每个字母: 中文名 / 形状 / 定义式
    第三节  段内怎么算    一个 program 如何算完自己那一段
    第四节  段间怎么合    16 份结果怎么变成 1 份
    第五节  工程细节      grid / NSPLIT / buffer
    第六节  实测与踩过的坑

变量书写格式全文统一:

    中文名 (代码符号): 形状   = 定义式

    例:  段内分母 (l_i): (BM,) = Σ_n exp(本块分数_n - m_i)

中文名、代码符号、式子三样对齐, 对着代码看不会迷路。


====================================================================
第一节  优化的原理
====================================================================

--- 1.1 attention 本来是什么 ---

对第 i 行 query:

    out[i] = Σ_j  P[i,j] * v[j]                                      (式1)

    P[i,j] = exp(q[i]·k[j] * scale) / Σ_j' exp(q[i]·k[j'] * scale)
             └────── 分子, 这一格的权重 ──────┘
                                     └────── 分母, 整行加起来 ──────┘

分母是"整行 8192 个格子的 exp 之和"。必须看完 8192 个 key 才能定。
**这一点是后面所有麻烦的根源。**

--- 1.2 卡在哪: 16 行 query 撑不起并行度 ---

第 14/15 题的并行方式是**按 query 行切**:

    grid = (cdiv(Sq, BM),)          Sq=16, BM=16  ->  grid = (1,)

**整个 GPU 上只有 1 个 block 在跑。** 这张卡 20 个 SM, 19 个全程闲着。
同口径实测: 168.6 us, 1.1% 峰值, 12.5 GB/s —— 比完整 self-attention 还慢一倍多
(那个 0.0873 ms 是 32 个 block 一起跑出来的)。

为什么 decode 会这样: 每生成一个 token 都要跟整条 KV cache 做一次 attention,
query 永远只有 1 个 (或一个小 batch), 而 KV cache 随生成过程线性变长。
**算力一点都不缺, 缺的是活。**

--- 1.3 手段: 换一个方向切 ---

    query 方向:  16 行, 只能切 1 份      切不动
    key   方向:  8192 个, 可以切 16 份    每份 512 个

    grid = (1, 16)  ->  16 个 block, 20 个 SM 填进 16 个

这就叫 split-KV (把 K/V 切开)。

--- 1.4 代价: key 方向是"求和"方向, 切开就得再合起来 ---

attention 对 key 方向干的事是求和 (式1 里的 Σ_j)。切开之后每个 program
只看到 key 的一个子集, 它算出的是**部分和**:

    段 s 只看到 S_s 这些 key
      -> 算出的 o_s 是"假如全世界只有 S_s"的答案
      -> 16 个这样的答案**不能直接平均**: 各段看到的 key 多少不同,
         而且分母本该是全局的, 每段只拿到了自己那一小块

所以合并时必须"补上每段缺的那部分权重"。要让第二个 kernel 能补,
每段得额外交出一个数: **它自己的分母, 并且取了对数** —— 这就是 lse。
(第四节详细推)

--- 1.5 代价清单与收益 ---

    代价:  多 1 次 kernel 启动 (空 kernel 实测 2.7 us, 但会被 GPU 队列吃掉一部分,
           实际只多 2.3 us —— 见 5.4)
           多一个中间 buffer: 写 16*16*64 个 float 再读回来 (64 KB)
           多一套合并数学
    收益:  并行度 1 个 block -> 16 个 block
           171.6 us -> 21.8 us, **7.9 倍**

净赚, 因为原本 19 个 SM 在睡觉。真实推理框架还会用 CUDA Graph
把那次额外启动的开销压掉 (见 6.3)。


====================================================================
第二节  变量词典
====================================================================

【A 组  尺寸常量】整个 kernel 里不变

    符号       值        含义
    Sq         16        query 的 token 数, 即这一批新 token
    Sk         8192      key/value 的 token 数, 即 KV cache 长度
    d          64        每个 token 的向量维度
    BM         16        一个 program 负责几行 query (= Sq, 所以只需 1 个)
    BN         64        一块 key 覆盖多少个 token
    NSPLIT     16        key 轴切成几段 = 并行度
    BD         64        d 向上取到 2 的幂, 用来开 tl.arange
    scale      1/8       1/sqrt(d), 防 q·k 数值过大导致 softmax 饱和


【B 组  段内滚动状态】每个 program 私有, 每处理一块 key 就更新一次

    中文名           符号    形状       定义式 / 含义
    本块分数         qk      (BM, BN)   = q · k^T * scale
    段内最大分数     m_i     (BM,)      = max(已处理的所有分数)          <- 滚动量!
    段内分母         l_i     (BM,)      = Σ exp(已处理的所有分数 - m_i)   滚动累加
    段内分子         acc     (BM, BD)   = Σ exp(已处理的所有分数 - m_i) · v   滚动累加
    折扣系数         alpha   (BM,)      = exp(旧 m_i - 新 m_i)
    本块权重         p       (BM, BN)   = exp(本块分数 - 新 m_i)

    *** m_i 是滚动量, 这是最容易读错的地方 ***

    它**不是**"整段的最大值", 而是"到目前为止见过的最大值":

        处理第 0 块之前     m_i = -inf
        处理完第 0 块       m_i = 前 1 块的最大值
        处理完第 k 块       m_i = 前 k+1 块的最大值
        循环结束            m_i 才等于整段的最大值

    m_i 在循环里被反复覆盖, 每一轮吃的都是"上一轮的结果"。
    第三节用两组数字把这件事做实。

    l_i 和 acc 同理, 都是"到目前为止"的累加值, 不是一次算完的。
    这三个变量合起来就是 online softmax 的全部状态。


【C 组  段间结算】program 循环结束后算一次, 写出去给第二个 kernel

    中文名           符号    形状          定义式 / 含义
    段总结           o       (Sq, d)       = acc / l_i          <- 已归一化!
    对数分母         lse     (Sq,)         = m_i + log(l_i)     <- 就是 log Z, 见 4.1
    最大对数分母     M       (Sq,)         = max_s lse_s
    段份额           w       (NSPLIT, Sq)  = exp(lse_s - M)

    注意形状: C 组都是"每段、每行 query 各一个标量"。
    lse 的形状是 (NSPLIT, Sq) = 16*16 = 256 个数, 小得可以忽略。
    它不是 (Sq, Sk) 那种矩阵 —— 这点是 split-KV 便宜的全部原因。


【D 组  记号】这些不是变量

    pid_s        当前 program 负责第几段 key, 0 .. NSPLIT-1
    max_s lse_s  不是变量, 是"对所有段取最大值"的记号
                 Python 里等价于 max(lse[0], ..., lse[15]) 或 np.max(lse, axis=0)
    Σ_s w_s      同理, "对所有段求和", 等价于 np.sum(w, axis=0)


====================================================================
第三节  段内: 一个 program 怎么算完自己那一段
====================================================================

--- 3.1 目标 ---

输入: q (BM, d) + 一段 key (nblk 个块, 每块 BN 个)
输出: o (BM, d) 段总结,  lse (BM,) 对数分母

--- 3.2 为什么不能"先把所有分数算完再 softmax" ---

那样得把 BM * (nblk*BN) 个分数放进显存。8192 个 key 时是 16*8192*4B = 512 KB,
16 个 program 就是 8 MB, 而且读写这些分数的带宽比计算本身还贵。
这正是 FlashAttention 要解决的问题: **分数算出来立刻用掉, 绝不落地。**

于是只能一块一块算、边算边累加。但 softmax 需要两个"要看完才知道"的量:

    最大值  防 exp 溢出
    分母    整行的 exp 之和

解决: 用**滚动近似**。每次看到更大的分数, 就把之前累的账按比例缩小。

--- 3.3 每处理一块 key, 做四步 ---

    步骤 1  算本块分数
            本块分数 (qk): (BM, BN) = q · k^T * scale

    步骤 2  更新最大值
            本块行最大值 = max(qk, axis=1)                 形状 (BM,)
            新最大值 (m_new): (BM,) = max(旧 m_i, 本块行最大值)

    步骤 3  算折扣
            折扣 (alpha): (BM,) = exp(旧 m_i - 新 m_new)
            它的作用见 3.4, 这是 online softmax 的核心

    步骤 4  累加分母和分子
            本块权重 (p):   (BM, BN) = exp(qk - 新 m_new[:, None])
            段内分母 (l_i): (BM,)    = 旧 l_i * alpha + Σ_n p
            段内分子 (acc): (BM, BD) = 旧 acc * alpha[:, None] + p · v

            顺序要求: **acc 必须先乘 alpha, 再做 dot**。
            tl.dot(p, v, acc) 的第三个参数是"累加到 acc 上",
            所以 alpha 必须在调用它之前乘完, 否则这一块的 p·v 也会被折扣误伤。

--- 3.4 alpha 到底在干什么 (跟着两组数字走一遍) ---

一段 key 的分数假定是 [1.0, 2.0, 2.5, 1.5], 分两块算, 每块 2 个:

    块 0: [1.0, 2.0]        块 1: [2.5, 1.5]

    --- 处理块 0 ---
    本块最大值 = 2.0
    旧 m_i = -inf  ->  新 m_i = 2.0
    alpha = exp(-inf - 2.0) = 0                  <- 旧账本来是空的, 折成 0 正好
    p     = [exp(1.0-2.0), exp(2.0-2.0)] = [0.3679, 1.0000]
    l_i   = 0*0 + (0.3679+1.0000) = 1.3679
    acc   = 0*0 + p·v

    --- 处理块 1 ---
    本块最大值 = 2.5                             <- 比之前的大, 基准要换
    旧 m_i = 2.0  ->  新 m_i = 2.5
    alpha = exp(2.0 - 2.5) = 0.6065              <- 旧账全部打 6 折
    p     = [exp(2.5-2.5), exp(1.5-2.5)] = [1.0000, 0.3679]
    l_i   = 1.3679*0.6065 + (1.0000+0.3679)
          = 0.8296 + 1.3679 = 2.1975
    acc   = 旧 acc * 0.6065 + p·v

    --- 循环结束 ---
    m_i = 2.5,  l_i = 2.1975,  acc = 0.5908

    --- 对照: 4 个分数一次性算 ---
    m = 2.5,  l = Σ exp(x - 2.5) = 2.1975,  acc = 0.5908      <- 完全一致

**为什么必须打折**: 分母是 Σ exp(x - m), 它依赖 m 的取值。
块 1 把基准从 2.0 换成了 2.5, 之前按 2.0 算出来的 exp(1.0-2.0)、exp(2.0-2.0)
全部偏大, 必须统一乘 exp(2.0-2.5) 缩回去。
**分母和分子要缩同样的倍数**, 否则分子分母不成比例, 结果就错了。

**为什么打折就够了、不用回头重算**: 因为每个加数都乘同一个常数,
等价于整个和乘这个常数。所以不必重算前面的块, 只缩总和即可。

--- 3.5 循环结束后, 结算成两样东西 ---

    段总结 (o):     (BM, BD) = acc / l_i[:, None]
                    这段自己的答案, 合并时会被当成一个整体来加权

    对数分母 (lse): (BM,)    = m_i + log(l_i)
                    这段的 log(分母), 第四节说明为什么要存它

    空段保护: l_i 可能为 0 (这段一个有效 key 都没分到, 或全是 -inf)。
             直接 acc/0 出 nan, 而 nan 乘任何数还是 nan, 一个空段能污染整个输出。
             所以先 l_safe = where(l_i > 0, l_i, 1.0) 再除, 同时让 lse = -inf。
             份额 exp(-inf - M) = 0, 空段自然被排除, 但那个 1.0 不能省。


====================================================================
第四节  段间: 怎么把 16 份答案合成 1 份
====================================================================

--- 4.1 lse 就是 softmax 分母取对数 ---

式1 那个分母:

    Z = Σ_j exp(q·k_j * scale)                  配分函数 (partition function)

lse = log Z。名字就是 **L**og-**S**um-**E**xp 的缩写, 定义就是 log(Σ exp(...))。
所以它对应的就是 softmax 公式里"除以谁"的那个谁, 只不过取了对数。

代码里不直接算 log Z, 而是算 m_i + log(l_i):

    l_i   = Σ exp(x - m_i)              (缩放过的分母 = Z / exp(m_i))
    log Z = log( exp(m_i) * l_i ) = m_i + log(l_i)

两种算法结果完全相同 (实测 3.287339 vs 3.287339), 但后者不会溢出:
直接算 exp(x) 时 x 稍大就超 fp32 上限, 先减最大值再算就安全。

**为什么存 log 而不存 Z 本体**: 因为跨段要相加, 而 log 域能安全相加, 见 4.2。

--- 4.2 合并公式怎么推出来的 ---

先把全局分母拆成各段的贡献:

    Z_全局 = Σ_j exp(x_j) = Σ_s Σ_{j∈S_s} exp(x_j) = Σ_s Z_s

问题: 16 个 Z_s 之间的尺度不一样 (各自减的最大值不同), 不能直接相加。
换成 log 域就没这个问题, log 域的加法叫 log-sum-exp:

    log(Z_A + Z_B) = log( exp(lse_A) + exp(lse_B) )
                   = M + log( exp(lse_A - M) + exp(lse_B - M) )      M = max(lse_A, lse_B)
    ^ 减 M 纯粹为了防溢出, M 取任何值最终结果都一样 (分子分母会抵消)

把式中的两个 exp 项命名成"份额":

    段份额 (w_s): (NSPLIT, Sq) = exp(lse_s - M)

w_s 的物理意义 —— **这一段的 Z_s, 除掉公共因子 exp(M)**:

    w_s = exp(lse_s - M) = exp(log Z_s - M) = Z_s / exp(M)

于是"各段分母之和"就是 w_s 之和。输出就是:

    out = Σ_s w_s * o_s / Σ_s w_s                                     (式2)

-- 为什么式2 是对的: 它就是分数相加 --

    分子 = Σ_s w_s * o_s     各段的分子, 都缩放了 1/exp(M)
    分母 = Σ_s w_s           各段的分母, 都缩放了 1/exp(M)

分子分母同处一个缩放, 相除就抵消了, 得到真值。
段内做的是 (加权和 / 分母), 段间做的是 (分子之和 / 分母之和) —— 同一件事换个层级。

-- 我第一次推错了什么 --

第一版存的是**未归一化的分子 acc**, 权重仍用 w_s = exp(lse - M)。
错在: w_s 里已经含了段内分母 (它就是 Z_s 缩放), 再乘一个还带着分母的 acc,
等于**分母被乘了两次**。

症状很好认: NSPLIT=1 就错 27, 而且**误差随每段块数增长**
(每段 128 块错 27, 每段 7 块只错 2.5)。段越长, 段内分母越大, 多乘的那一份就放大得越狠。
正确实现里段长不该影响误差 —— "误差 ∝ 段长"本身就是线索。

所以 o 和 w_s 必须**一个含分母、一个不含**才配得上。
这就是 split kernel 结尾非要做 acc / l_i 那一步的原因。

--- 4.3 merge kernel 的实现要点 ---

    一次把所有段读进来, 不要 for s 循环:

        l_all: (NSP, BM)      每段每行的 lse        (NSP = NSPLIT 向上取到 2 的幂)
        M    : (BM,)          = max(l_all, axis=0)   <- 就是 max_s lse_s
        w    : (NSP, BM)      = exp(l_all - M)
        acc  : (NSP, BM, BD)  一次 3D load 读完全部段的 o
        num  : (BM, BD)       = sum(acc * w[:, :, None], axis=0)
        den  : (BM,)          = sum(w, axis=0)
        out  : (BM, BD)       = num / den[:, None]

    为什么必须一次读完: merge 的 grid 只有 cdiv(Sq, BM) = 1 个 block,
    用 for s 循环等于串行发 16 次 load, 那是**延迟**不是带宽。
    实测 NSPLIT=16 时: 串行 13.0 us -> 向量化 8.1 us (-38%)。


====================================================================
第五节  工程细节
====================================================================

--- 5.1 切分必须按块号, 不能按元素 ---

第一版按元素切: 每段起点 = 段号 * cdiv(Sk, NSPLIT), 再从这个起点向上取整到块边界。
**相邻段会重叠**: 起点 410 时 410//64 = 6, 第 6 块覆盖 384..447,
而 384..409 已经被上一段算过了。那些 key 的权重被算了两遍, 分母偏大, 输出偏小。

正确做法是按块号切:

    nblk = cdiv(Sk, BN)         一共多少块
    per  = cdiv(nblk, NSPLIT)   每段分几块
    b0   = pid_s * per          本段起始块号
    b1   = min(b0 + per, nblk)  本段结束块号 (开区间)
    for n0 in range(b0, b1): ...

这个 bug 在 Sk 恰好是 NSPLIT*BN 整数倍时**静默通过**
(8192 = 8*16*64, NSPLIT=8 时 per=16 正好整除), 只有不整除的尺寸才露馅。
所以 --odd 里除了 4096/8192 还得有一个奇怪的 Sk。

--- 5.2 NSPLIT 怎么选 ---

不能随便取: per = cdiv(nblk, NSPLIT), 若 per*NSPLIT > nblk 就有空段。
例: nblk=128, NSPLIT=20 -> per=7, 20*7 = 140 > 128, 第 20 段完全空转,
实际只有 19 段干活。

所以优先退到"能整除 nblk 且不小于目标一半"的最大值:

    nblk   = 128
    target = min(nblk, cdiv(SM 数 20, cdiv(Sq, BM)=1)) = 20
    20 会让 20*7 > 128, 不整除 -> 回退到 16 (per = 8, 正好整除)

**但别把理由说成"20 段让 4 个 SM 闲置, 16 段更浪费"** —— 那是把 SM 数当瓶颈想的,
这里不是。实测 (_probe_16_nsplit2.py, 一个配置一个进程):

    NSPLIT  有效段  per  nsp   split    merge    合计
      16      16     8   16   22.32    8.54    21.68
      20      19     7   32   22.07    8.36    22.20
      24      22     6   32   22.22    9.13    22.54
      32      32     4   32   23.66   10.58    24.49

三件事:

1. **多用 3 个 SM 只买到 1%**: split 22.32 -> 22.07。
   如果时间真是"每 program 块数 x 单块时间", 8 块变 7 块该快 12.5%,
   实际只快 1.1%。说明每个 program 的时间 = 固定成本 (加载 q / init / 写 lse / 启停)
   + 块数 x 单块时间, 而固定成本占了九成, 块数几乎不影响它。
2. **32 段把 20 个 SM 全占满 (还超订 12 个), 反而最慢** (24.49)。
   如果"闲置 SM 少 = 快"成立, 32 段应该最快。它不成立 —— 这就是瓶颈不在 SM 数的证据。
3. **20 段要多付一笔 merge 的账**: nsp = next_power_of_2(20) = 32, merge 的 tile
   从 (16,BM,BD) = 64 KB 变 (32,BM,BD) = 128 KB, 多读的 12 行全被 mask 掉。
   Sq=16 时 merge 只有 1 个 block, 这笔账是 0.45 us (5.43 -> 5.88);
   Sq=1024 时 merge 有 64 个 block, 涨到 3.7 us (35.3 -> 39.1, _probe_16_merge_scaling.py)。

净结果 16 赢 0.4~0.5 us (约 2%)。**它不是"调度更优", 而是"没有 padding 那笔账"。**
所以选 16 的真正理由是"不留空段 + 顺带是 2 的幂", 而不是"能省 3% 时间"。

--- 5.3 两块 buffer: 两个 kernel 之间唯一的交接 ---

    op   (nsplit, Sq, d)  fp32   每段已归一化的输出 o_s    split 写, merge 读
    lse  (nsplit, Sq)     fp32   每段的对数分母            split 写, merge 读

这两块是 split 和 merge **唯一的通信方式**。两个 kernel 不能共享寄存器/共享内存,
只能通过显存交接, 所以必须有一块地方放中间结果。第一维是 nsplit, 每个段各写自己那层。

流量账: 64 KB (op 写) + 64 KB (op 读) + 1 KB x2 (lse) ≈ 130 KB,
占全部 2.1 MB 的 6% —— 这正好对上 merge 那 0.25 us (21.78 - 21.53)。

为什么用 fp32: 实测把 op 换成 fp16, max|diff| 从 1.04e-5 涨到 1.74e-5 (1.7 倍)。
在这个量级上还能接受, 但 buffer 总共才 65 KB, 压成 fp16 只省 32 KB (总流量的 1.5%),
换 1.7 倍精度损失不划算。

模块级 _BUF 按 (nsplit, sq, d, dev) 缓存。

**但这个缓存没有测出提速, 别把它当优化。** 三种写法打乱顺序重复三轮:

    A 每次 torch.empty       22.3 / 24.4 / 24.0 / 26.9
    B _BUF 缓存              22.3 / 25.5 / 22.0 / 26.0
    C 预分配固定 buffer       24.6 / 26.8 / 22.2 / 24.4

全在噪声内。原因: 单次 torch.empty 的 16 us 是 **CPU 侧**开销, 和上一轮 kernel 的
GPU 执行重叠掉了 —— benchmark 里 CPU 跑得比 GPU 快, 这笔账根本不在关键路径上。
(最初写这段时我以为它会算进耗时, 实测否掉了。)

缓存留着只是为了"不制造 allocator 抖动、代码里不出现临时对象", 不是因为它快。
真想省启动开销, 正确做法是 CUDA Graph, 不是缓存 buffer。

**注意这个缓存是全局的, 不线程安全**: 同一个 (nsplit, Sq, d) 如果被两个 stream
或两个线程同时调 run, 会互相覆盖 op/lse。真接进框架要改成每次调用传进来或做池化。

--- 5.4 为什么必须是两次启动 (不能合成一个 kernel) ---

    _split_kernel   grid = (cdiv(Sq,BM), nsplit) = (1, 16)    16 个 block
    _merge_kernel   grid = (cdiv(Sq,BM_M),)      = (16,)      16 个 block (BM_M=1)

**硬原因: 一个 kernel 内部, block 之间没有同步原语。**
__syncthreads() 只在同一个 block 里有效; 不同 block 由硬件调度, 谁先跑完谁退出,
0 号 block 无法知道 15 号写完没有 —— 而这个 kernel 里 merge 恰恰需要**全部 16 段的 lse**
才能算出 M = max_s lse_s (式2 的第一步就是"取所有段里最大的")。

所以这不是"懒得合并", 是**跨 block 归约天然需要一个 kernel 边界**:
kernel 边界是 CUDA 里唯一"保证前一次 grid 的所有 block 都已结束"的东西。

想绕过它只有三条路, 每条都要自己负责正确性:

  1. atomic 计数 + 自旋等待
     每个 block 算完 atomic_add(counter, 1), 发现 counter == NSPLIT 的那个 block 做 merge。
     **要求所有 block 必须同时驻留**, 否则先完成的 block 占着 SM 空转, 后面的 block
     排不进来 -> 死锁。这里 16 block < 20 SM 刚好安全, 但 Sq=2048 时 grid 会变成
     2048 个 block, 必然死锁。**靠尺寸侥幸, 不是正确做法。**
  2. cooperative launch + grid.sync()
     要 cudaLaunchCooperativeKernel 启动, 且 grid 大小必须被 occupancy 证明能同时驻留。
     Triton 不暴露这个接口。
  3. persistent kernel
     只启动 SM 数那么多个 block, 内部用全局队列循环取任务, 活干完再做归约。
     死锁问题解决了, 但要自己写任务队列 + memory fence, Triton 的 while/atomic
     语义限制很多。vLLM/FlashInfer 里那么干的是手写 CUDA, 不是 Triton。

**经济原因: 这笔账本来就不大。**
实测(method: 空 kernel 与 CUDA Graph 对照, 见 _probe_16_onepass.py):

    空 kernel launch 固定成本 (CPU 侧)      2.7 us
    graph: split only                      19.3 us
    graph: split + merge                   21.7 us    -> merge 全部代价 2.3 us
    端到端 do_bench: split only            21.5 us
    端到端 do_bench: split + merge         21.8 us    -> 只差 0.25 us

最后一行是关键: **CPU 侧那 2.7 us 的 launch 成本, 已经被 GPU 执行上一段的时间重叠掉了。**
合并成一个 kernel 的理论收益上限就是这 0.25~2.3 us (1%~11%), 换来的却是
手搓 barrier + 死锁风险 + 丢掉可移植性。不值。

**真正能白拿的 5.4% 在别处: merge 的 tile 大小。**
原来 merge 跟 split 共用 BM=16, 一个 block 要一次读 (16, 16, 64) = 64 KB 的 3D tile。
merge 只是加权求和, 不需要 split 那种算术强度, tile 越大越是让单个 block 死扛。
改成 BM_M=1 (grid 变 16 个 block, 每个只读 4 KB):

    端到端 graph 口径   21.75 -> 20.71 us   (-4.8%)
    check.py 口径      0.0184 -> 0.0174 ms  (-5.4%, 三次一致)

零风险, 不需要任何同步技巧。**该调的是 tile, 不是 kernel 边界。**


====================================================================
第六节  实测与踩过的坑
====================================================================

--- 6.1 数字 ---

check.py 口径 (冷机单独跑), 16x8192x64:

    16_split_kv_decode    0.0174 ms   1.93 TFLOPS   10.6%   启动 2 次 / 理想 2 次
    torch SDPA 同尺寸     0.3604 ms   0.09 TFLOPS            (它也没有 split-KV)

同口径对比 (_probe_16_vs_nonsplit.py, 同一个 do_bench、同一份输入):

    方案                     us    TFLOPS  %峰值    GB/s  启动
    A 单 block (1 段)      171.6     0.20    1.1    12.3    1
    B split 部分 (16 段)    21.53    1.55    8.5    97.0    1
    C split+merge (16 段)   21.78    1.54    8.5    96.5    2

    A -> C 是 7.9 倍。merge 这一层只花 0.25 us。

--- 6.2 NSPLIT 扫描 ---

    NSPLIT  每段块数  有效段   split   merge(串行)   合计    GB/s
      4       32        4     46.9      7.4        46.2    45.5
      8       16        8     30.0      9.1        28.5    73.8
     16        8       16     22.3     13.0        23.1    91.2    <- 现在这个
     20        7       19     22.9     14.1        23.8    88.2
     32        4       32     22.9     19.8        26.9    78.1
     64        2       64     23.6     31.0        33.0    63.7

1. split 在 NSPLIT>=16 之后饱和 (22.3~23.6 us)。访存总量 2.1 MB 不变, 那是带宽下界,
   再切多少段也不会更快 —— 只是把同一块饼切得更碎。
2. merge 随 NSPLIT 线性涨 (约 0.35 us/段), 这是延迟不是带宽: 串行 for 循环发 N 次 load。
   改向量化后 NSPLIT=16 从 13.0 -> 8.1 us。
3. NSPLIT=20 反而不如 16: 不是"空了一块 SM", 而是 20 让 nsp 从 16 变 32,
   merge 的 tile 翻倍。上表是串行 merge 时代的数, 向量化之后 16 段合计 21.7 us、
   20 段 22.2 us, 差距缩到 2%。拆解见 5.2。

--- 6.3 为什么 10% 峰值看着低, 但这不是退步 ---

    算术强度:   4*16*8192*64 = 33.5 MFLOP / (8192*64*2*2 B = 2.1 MB) = 16.0 FLOP/Byte
    roofline 平衡点: 45.4 FLOP/Byte

**16 < 45.4, decode 铁定是 memory-bound。** 所以拿 14 题的 67.6% (compute-bound)
来要求它是错的 —— 两个百分比量的不是同一件事。

    访存下界: 2.1 MB / 180 GB/s = 11.7 us
    实测:     21.8 us
    中间那 10.1 us 拆成两笔: split 自身到 19.3 us 就触顶 (比下界多 7.6, 是启动+收尾),
    merge 再加 2.3。grid 只有 16 个 block, 20 个 SM 都填不满, 摊不薄。这是 decode 的本质:
    **活太少, 一次 launch 的成本和真正计算在一个量级。**

真实推理框架不靠继续优化 kernel, 而是用 CUDA Graph (把 launch 开销压到接近 0),
或把一整层 (甚至多层) 的 attention 打包成一次 launch。kernel 层面到这里基本到顶。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _split_kernel(
    q_ptr, k_ptr, v_ptr, op_ptr, lse_ptr,
    Sq, Sk, d, scale,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr, NSPLIT: tl.constexpr,
):
    """一段 KV 的 partial attention, 结算成两样东西写出去:

        段总结  (o):   (BM, BD) = acc / l_i        <- 已归一化, 合并时当整体加权
        对数分母 (lse): (BM,)    = m_i + log(l_i)   <- 就是 log(这段的 softmax 分母)

    推导见模块 docstring 第三节 (段内滚动) 和第四节 (为什么存 log)。
    """
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)

    q = tl.load(q_ptr + offs_m[:, None] * d + offs_d[None, :],
                mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < d), other=0.0)

    # online softmax 的三个滚动状态, 都是"到目前为止"的值, 每块更新一次
    m_i = tl.full((BM,), -float("inf"), tl.float32)   # 段内最大分数, 滚动
    l_i = tl.zeros((BM,), tl.float32)                 # 段内分母, 滚动累加
    acc = tl.zeros((BM, BD), tl.float32)              # 段内分子, 滚动累加

    # 按块号切, 不按元素切 —— 按元素切会让相邻段重叠 (见 5.1)
    nblk = tl.cdiv(Sk, BN)
    per = tl.cdiv(nblk, NSPLIT)
    b0 = pid_s * per
    b1 = tl.minimum(b0 + per, nblk)

    for n0 in range(b0, b1):
        offs_n = n0 * BN + tl.arange(0, BN)
        mask_n = offs_n < Sk

        k = tl.load(k_ptr + offs_n[:, None] * d + offs_d[None, :],
                    mask=mask_n[:, None] & (offs_d[None, :] < d), other=0.0)
        v = tl.load(v_ptr + offs_n[:, None] * d + offs_d[None, :],
                    mask=mask_n[:, None] & (offs_d[None, :] < d), other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(mask_n[None, :], qk, -float("inf"))   # 只看尾部越界, 没有 causal

        # 步骤 2: 发现更大的分数就换基准
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        # 步骤 3: 旧账要按新基准折算, 否则分子分母不成比例 (见 3.4)
        alpha = tl.exp(m_i - m_new)
        # 步骤 4: 累加分母和分子
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]         # 必须先折算旧账, 再做下面的 dot 累加
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_new

    # 段总结: acc 先归一化。空段 l_i=0 时用 1.0 兜住 —— 份额本来就是 0 (lse=-inf),
    # 但 0/0 会先变成 nan, 而 nan * 0 还是 nan, 一个空段能污染整个输出。
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o = acc / l_safe[:, None]
    op_ptrs = op_ptr + (pid_s * Sq + offs_m[:, None]) * d + offs_d[None, :]
    tl.store(op_ptrs, o, mask=(offs_m[:, None] < Sq) & (offs_d[None, :] < d))

    # 对数分母 = log(段内分母) = m_i + log(l_i), 见 4.1
    lse = tl.where(l_i > 0, m_i + tl.log(l_i), -float("inf"))
    tl.store(lse_ptr + pid_s * Sq + offs_m, lse, mask=offs_m < Sq)


@triton.jit
def _merge_kernel(
    op_ptr, lse_ptr, o_ptr,
    Sq, d,
    BM: tl.constexpr, BD: tl.constexpr, NSPLIT: tl.constexpr, NSP: tl.constexpr,
):
    """把 NSPLIT 份段总结按份额加权合并 (式2):

        M    = max_s lse_s                每行取所有段里最大的对数分母
        w_s  = exp(lse_s - M)             这一段的份额 = Z_s / exp(M)
        out  = Σ_s w_s * o_s / Σ_s w_s    分子加分子, 分母加分母

    各段结果用**一次** load 全部读进来 (NSP, BM) / (NSP, BM, BD), 不要 for s 循环 ——
    merge 的 grid 只有 cdiv(Sq,BM) 个 block, 循环 NSPLIT 次等于串行发 NSPLIT 次 load,
    那是延迟不是带宽。实测 NSPLIT=16 时串行 13.0 us / 向量化 8.1 us。
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)
    offs_s = tl.arange(0, NSP)
    mask_s = offs_s < NSPLIT
    mask_m = offs_m < Sq

    # (NSP, BM): 每段每行的 lse
    l_all = tl.load(lse_ptr + offs_s[:, None] * Sq + offs_m[None, :],
                    mask=mask_s[:, None] & mask_m[None, :], other=-float("inf"))
    M = tl.max(l_all, 0)                                # 就是 max_s lse_s
    # 全是 -inf 的行 (padding 行 / 所有段都空) 会被 mask 掉, 但 exp(-inf - -inf) 是 nan,
    # 先按 0 处理, 免得 nan 扩散
    M = tl.where(M > -float("inf"), M, 0.0)

    w = tl.where(mask_s[:, None], tl.exp(l_all - M[None, :]), 0.0)   # (NSP, BM) 份额

    # (NSP, BM, BD): 一次读完全部段总结
    acc = tl.load(op_ptr + (offs_s[:, None, None] * Sq + offs_m[None, :, None]) * d
                  + offs_d[None, None, :],
                  mask=mask_s[:, None, None] & mask_m[None, :, None]
                  & (offs_d[None, None, :] < d), other=0.0)

    num = tl.sum(acc * w[:, :, None], 0)      # Σ_s w_s * o_s
    den = tl.sum(w, 0)                        # Σ_s w_s
    den = tl.where(den > 0, den, 1.0)

    tl.store(o_ptr + offs_m[:, None] * d + offs_d[None, :], num / den[:, None],
             mask=mask_m[:, None] & (offs_d[None, :] < d))


# partial buffer 缓存: 每次调 run 都重新 torch.empty 会把 allocator 开销算进 kernel 时间里
_BUF = {}


def _buffers(nsplit, sq, d, dev):
    key = (nsplit, sq, d, dev)
    b = _BUF.get(key)
    if b is None:
        op = torch.empty((nsplit, sq, d), device=dev, dtype=torch.float32)
        lse = torch.empty((nsplit, sq), device=dev, dtype=torch.float32)
        _BUF[key] = b = (op, lse)
    return b


def run(q, k, v, out):
    Sq, d = q.shape
    Sk = k.shape[0]
    scale = 1.0 / (d ** 0.5)
    BD = triton.next_power_of_2(d)

    BM, BN = 16, 64
    nsm = torch.cuda.get_device_properties(q.device).multi_processor_count

    # 目标: Sq 方向的 block 数 x NSPLIT 至少能喂满 SM。
    # Sq 很小时 (decode), Sq 方向只有 1 个 block, 全靠 NSPLIT 撑起并行度。
    #
    # 但 NSPLIT 不能随便取: per = cdiv(nblk, NSPLIT), 若 per*NSPLIT > nblk 就有空段,
    # 比如 nblk=128 / NSPLIT=20 → per=7, 140>128, 第 20 段完全空跑, 实际只有 19 段干活。
    # 所以优先退到"能整除 nblk 且不小于目标一半"的最大值 (128 → 16, per=8 正好整除)。
    # 顺带的好处: 16 是 2 的幂, nsp 不用向上补, merge 的 tile 保持 64 KB 而不是 128 KB。
    # (多用的那 3 个 SM 只值 1%, 见 5.2 的实测 —— 别以为退到 16 是"少浪费 SM")
    nblk = triton.cdiv(Sk, BN)
    target = min(nblk, max(1, triton.cdiv(nsm, triton.cdiv(Sq, BM))))
    nsplit = target
    if triton.cdiv(nblk, target) * target > nblk:
        for cand in range(target, 0, -1):
            if nblk % cand == 0 and cand * 2 >= target:
                nsplit = cand
                break
    nsp = triton.next_power_of_2(nsplit)

    op, lse = _buffers(nsplit, Sq, d, q.device)

    _split_kernel[(triton.cdiv(Sq, BM), nsplit)](
        q, k, v, op, lse, Sq, Sk, d, scale,
        BM, BN, BD, nsplit, num_warps=4, num_stages=2)

    # merge 的 tile 不该跟 split 共用 BM —— 两者的 BM 含义完全不同:
    #   split 的 BM 是为了让 q·k^T 有足够算术强度喂 tensor core, 越大越好,
    #   merge 只是把 (NSP, BM, BD) 读进来做加权和, tile 越大 = 单个 block 死扛越多。
    # Sq=16 时 BM=16: 一个 block 要读 16*16*64*4B = 64 KB; 改成 BM=1: 16 个 block 各读 4 KB。
    # 实测(x 2) 端到端 21.75 -> 20.71 us, 快 4.8%。
    # 上限按 "block 数不超过 4*SM" 来放, 免得 Sq 很大时拆出几千个 block。
    BM_M = 1
    while BM_M * 2 <= BM and triton.cdiv(Sq, BM_M * 2) >= 4 * nsm:
        BM_M *= 2

    _merge_kernel[(triton.cdiv(Sq, BM_M),)](
        op, lse, out, Sq, d,
        BM_M, BD, nsplit, nsp, num_warps=4, num_stages=2)
