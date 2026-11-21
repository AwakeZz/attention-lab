# Attention 算子库：从三趟走到全场景闭环

一组手写的 Triton attention kernel，覆盖大模型推理的完整场景链：

```
prefill                causal prefill      decode              decode+GQA         decode+分页
13 三趟走(反面)  ──▶  14 FlashAttention ──▶  15 +causal      16 split-KV  ──▶  17 共享K/V ──▶ 19 PagedAttention
  0.201 ms              0.087 ms            0.059 ms          0.017 ms          0.062 ms          0.014 ms
                       (2.3×)                                 (7.5× vs 朴素)    (算术强度 16→64)   (vLLM 核心)
                                                                                             18 backward dQ/dK/dV (可训练闭环)
```

**这不是"我又写了一遍 FlashAttention"**。市面上这类项目大多止步于"跑通 forward、比 PyTorch 快一点"。这里的每个 kernel 都在回答同一个问题：

> **它离这张卡的硬件上限还有多远？中间那段差距是哪来的？**

每个优化手段的收益都被单独量化，归因到具体的硬件瓶颈——而不是笼统地说"我更快"。

---

## 为什么这个项目值得看

### 1. 有一条完整的 story，不是一个孤立的 kernel

从最慢的"三趟走"开始，每一步都针对上一版的真实瓶颈：

| 版本 | 上一版的瓶颈 | 采取的手段 | 收益 |
|---|---|---|---|
| 13 三趟走 | QKᵀ 的 (S,S) 中间矩阵落盘 32 MB，带宽用掉 88%、算力只用 29% | — | 基线 0.201 ms |
| 14 FlashAttention | 中间矩阵搬不动 | online softmax 滚动归一化，S 不落盘 | **2.3×**，0.087 ms |
| 15 causal | 算完一半无用块 | 循环上界从 `cdiv(S,BN)` 改成 `cdiv(start_m+BM,BN)`，右上三角根本不进循环 | 1.47×，0.059 ms |
| 16 split-KV decode | Sq=16 时 grid 塌成 1，20 个 SM 只有 1 个干活（1.4% 峰值） | 沿 KV 切 16 段并行，partial softmax 二次归并 | **7.5×**，0.017 ms |
| 17 GQA | 4 个 q head 各读一遍 K/V，算术强度只有 16 | 组内共享一次 K/V 加载，强度推到 64 | 带宽省 4× |
| 18 backward | 反向要穿过 softmax Jacobian，朴素做法要落 (S,S) | `D_i = dO·O` 化简 + 两个 kernel 反向并行 | 0.433 ms，闭环可训练 |
| 19 PagedAttention | 连续 KV cache 浪费显存、无法共享前缀 | block_table 乱序映射 + 非连续 gather | 0.014 ms，与连续版同开销 |

### 2. 数字用的是硬件实测分母，不是规格表

`bench.py` 在运行时实测当前 GPU 的**持续带宽**（copy kernel）和 **fp16 tensor core 峰值**（GEMM），所有 kernel 的 `% 峰值` 都除以这两个实测值。

这意味着：把项目丢到 Colab 的 A100 / L4 / T4 上，数字会**自动重新归一化**——不会出现"拿笔记本的峰值分母去算 A100 成绩"这种低级错误。练习集里那套 3050 Ti 的硬编码分母在这里一行都没有。

### 3. 正确性每一版都卡 fp32 参考

- forward 用 fp32 手写 `@`，不依赖 PyTorch SDPA（后者会挑 flash 后端，精度口径不固定）；
- backward 用 `torch.autograd.grad` 从 `O=softmax(QKᵀ/√d)V` 反推 dQ/dK/dV，避免手写 Jacobian 广播方向出错导致"参考值本身不可信"；
- PagedAttention 先按 block_table gather 成连续序列再算参考——测试用的物理块映射是**乱序 + 留空闲块**的，不真查表、按物理顺序连续读必然算错。

容差 `rtol=1e-2, atol=2e-2`，七题全过。

---

## 硬件与实测结果

主成绩用本地基线（RTX 3050 Ti Laptop，sm_86，20 SM，4 GB，45 W 功耗墙）。Colab 免费档的 T4(sm_75) 在 triton 3.8 上会触发后端退化；把 triton 降到 3.2 后同一份 kernel 能跑到两位数峰值——完整归因和验证见[「T4 上的性能异常」](#t4-上的性能异常从现象到版本分界线)。

| 项 | 值 |
|---|---|
| 实测持续带宽 | 180.5 GB/s |
| 实测 fp16 tensor core 峰值 | 18.4 TFLOPS |
| roofline 拐点 | 102 FLOP/Byte |

`python bench.py` 的输出（7 题全过，数字与练习集独立验收一致）：

| kernel | ms | TFLOPS | % 峰值 | SDPA 加速比 |
|---|---|---|---|---|
| 13 三趟走 | 0.201 | 5.35 | 29.0% | 0.63× |
| **14 FlashAttention** | **0.087** | **12.29** | **66.4%** | **1.45×** |
| 15 +causal | 0.060 | 8.99 | 48.6% | 1.81× |
| 16 split-KV decode | 0.017 | 2.18 | 11.8% | 23.7× |
| 17 GQA decode | 0.062 | 8.63 | 46.7% | — |
| 18 backward | 0.433 | 6.21 | 33.7% | — |
| 19 PagedAttention | 0.014 | 2.34 | 12.7% | — |

**Colab T4（仅把 triton 降到 3.2.0，kernel 代码一行没改）** — 2026-11-21 独立复跑，实测 fp16 峰值 25.0 TFLOPS / 带宽 220.9 GB/s：

| kernel | ms | TFLOPS | % 峰值 | SDPA 加速比 |
|---|---|---|---|---|
| 13 三趟走 | 0.2818 | 3.81 | 15.2% | 0.61× |
| **14 FlashAttention** | **0.1600** | **6.71** | **26.8%** | **1.07×** |
| 15 +causal | 0.1827 | 2.94 | 11.8% | 0.89× |
| 16 split-KV decode | 0.1162 | 0.29 | 1.2% | 4.43× |
| 17 GQA decode | 0.1143 | 4.70 | 18.8% | 5.11× |
| 18 backward | 0.4493 | 5.97 | 23.9% | — |
| 19 PagedAttention | 0.1878 | 0.18 | 0.7% | 0.62× vs #16 |

同样的 kernel，T4 从 triton 3.8 的 1–4% 跳到 11–27%，14 题反超 SDPA——**只降了一个编译器版本**。

> **两次实测都留着**：首次 14 题 28.7% / 1.10×，本次 26.8% / 1.07×，差 1–3 个点。来源是分母每次运行时重新实测、加上 Colab 会话间的功耗/频率状态不同。结论一致：两位数峰值、14 题反超 SDPA。这正是"同一个数不能只测一次就报"的原因——两次独立复跑结论站得住，才是能写进简历的数字。

> ⚠️ **两个不能当成绩讲的数字**：
> 1. 第 17 题 SDPA 对照 2.68 ms —— PyTorch 在 HQ≠HKV 时掉到 math 后端，43× 是"没优化过的对手"，不是我的加速。这一题真正该比的是"4 个 head 各读一遍 K/V"的朴素实现（见 kernel docstring 的共享 vs 逐 head 实测）。
> 2. decode 类（16/19）TFLOPS 天生只有 10% 出头 —— 它们是 memory-bound（算术强度 16 < 拐点 102），分母用算力本来就不合适。看带宽、看延迟，不要看 TFLOPS。

---

## 快速开始

### 本地

需要 CUDA GPU + Python 3.10+ + PyTorch + Triton：

```bash
pip install torch triton
python bench.py                  # 跑全部 + 硬件画像 + SDPA 对标
python bench.py --only 14 16     # 只跑指定题号
python bench.py --no-ref         # 跳过正确性校验, 只测速
```

### Google Colab（免费档 T4）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AwakeZz/attention-lab/blob/main/attention_lab.ipynb)

点上面徽章直接打开 notebook，Run All 即可。它会依次完成：环境检查 → 从本仓库 clone 代码 → 实测硬件峰值 → 跑全部 kernel → 输出结果表。运行时选 `T4 GPU`。

> **免费 Colab 只能选 T4（或 TPU，后者跑不了 CUDA kernel）**。notebook 会自动检测：T4(sm_75) 上把 triton 降到 3.2.0（Colab 默认装 3.8，会触发后端退化），再跑分。降级后七题 7/7 过 fp32 校验，14 题约 26.8–28.7% 峰值、反超 SDPA。pip 此时会报一条 `torch requires triton~=3.8.0` 的**依赖冲突警告**——是警告不是错误，手写 kernel 不走 `torch.compile`，忽略即可。完整归因见下面[「T4 上的性能异常」](#t4-上的性能异常从现象到版本分界线)。

> 注：Google AI Pro 的 Colab 算力文档上写可按量换 A100，但实际分配由 Colab 动态决定，免费/部分档位只开放 T4；不要假设一定能选到 A100。任何卡上 `bench.py` 的分母都是运行时实测的，数字口径统一。

---

## 目录结构

```
attention-lab/
├── bench.py                       # 统一 benchmark: 硬件画像 + 正确性 + 计时 + SDPA 对标
├── attention_lab.ipynb            # Colab 一键运行
├── diag_turing.py                 # 诊断 sm_75 的 shared / spills / 最优 warps×stages
├── diag_gemm.py                   # 对照实验: Triton 纯 GEMM vs cuBLAS, 定位后端退化
├── README.md
└── kernels/
    ├── 13_attn_naive.py           # 反面教材: QKᵀ→softmax→PV 三趟, 中间 (S,S) 落盘
    ├── 14_flash_attn.py           # FlashAttention: online softmax, 不落盘
    ├── 15_flash_attn_causal.py    # + 下三角 mask, 循环上界动态收缩
    ├── 16_split_kv_decode.py      # split-KV: 沿序列切段并行 + 二次归并
    ├── 17_gqa_decode.py           # GQA: 组内 4 q head 共享 K/V
    ├── 18_flash_attn_backward.py  # 反向 dQ/dK/dV, D_i = dO·O 化简
    └── 19_paged_attention.py      # PagedAttention: block_table 乱序映射
```

每个 kernel 文件的 docstring 都是一份独立教程：变量词典 → 数学推导 → 分块实现 → 踩过的坑 → 实测归因。中文注释，公式保留英文符号。

---

## T4 上的性能异常：从现象到版本分界线

把项目丢到免费 Colab 的 T4 上（triton 3.8），七题功能全对，但 `% 峰值` 只有 1–4%，第 14 题比 PyTorch SDPA 慢 6 倍。同一个 kernel 在本地 3050 Ti 是 66% 峰值、超 SDPA 1.45×。**同一个二进制，换个卡差 20 倍——这种反直觉的数字不能猜，要证。** 最后定位到一条编译器版本分界线，一条 pip 命令就把峰值拉回两位数。

### 第一步：排除"峰值分母算错"

`bench.py` 的分母是运行时实测的（GEMM 测算力、copy 测带宽），T4 报 26 TFLOPS 合理。分母没错。

### 第二步：怀疑是硬件架构差异，不是代码

T4 是 Turing(sm_75)，本地是 Ampere(sm_86)。先修掉了两个**真实存在**的架构兼容问题：

1. **第 17 题 shared memory 崩溃**：sm_75 的 MMA 布局不支持 3D batched dot 的 batch 维，编译器把 `(GROUP=4, BN, BD)` 真实复制 4 份，shared 从 18 KB 涨到 80 KB，超 T4 上限直接 `OutOfResources`。改成把 4 个 head 的 query 沿 M 拼成 2D dot（`_split_kernel_2d`），shared 降到 24 KB，输出逐字节一致。`diag_turing.py` 记录了这个过程。
2. **寄存器溢出**：诊断脚本测出 sm_75 上 `num_warps=4` 时每线程 255 寄存器还溢出 410 次到 local memory；改 8 warps 后溢出降到 58，第 14 题 2.97 → 1.82 ms。含 `tl.dot` 的 kernel 在 sm_75 上自动切 8 warps。

但这两步修完，全量重跑**仍然只有 4% 峰值**。说明还有更底层的原因。

### 第三步：用纯 GEMM 做对照实验，一锤定音

不去调 attention——直接测 Triton 的 `tl.dot` 在 T4 上**连最基础的矩阵乘**能跑多快，和 cuBLAS 比。`diag_gemm.py` 扫了 2048³ fp16 GEMM 的全部 tile × warps × stages 组合：

| | cuBLAS (`torch.mm`) | Triton 纯 GEMM 最高 | 寄存器溢出 |
|---|---|---|---|
| **T4 (sm_75)** | 27.2 TFLOPS | **2.9 TFLOPS（11%）** | `spill` 高达 **2774** 次 |
| **3050 Ti (sm_86)** | 15.1 TFLOPS | **16.7 TFLOPS（>100%）** | `spill = 0` |

结论没有歧义：**Triton 3.8 在 sm_75 上的 `tl.dot` 后端退化**——连纯矩阵乘都只有 cuBLAS 的 1/9，且大量配置把寄存器溢出到显存。T4 上再怎么调 attention 的 tile/warps 都翻不出这个天花板。**这是编译器后端缺陷，不是 kernel 逻辑错误**：同样的代码在 sm_86 上 spill=0、跑满甚至超过 cuBLAS。

> 到这里只能确定"不是代码、是后端"，但后端到底哪里错了？继续往下挖——见第四步。

### 第四步：定位到版本分界线，降级验证

纯 GEMM 的对照说明问题在后端，但后端错在哪？查 triton 的变更记录，找到一条明确的分界线：

- **PR #5066**（2024-11 合并，triton **3.3 首发**）让 `AccelerateMatmul` 对 `computeCapability < 80` 直接拒绝 MMA、回退普通 FMA。T4 是 sm_75，正好被排除。
- vLLM issue #60821 的最小复现：triton **3.1.0** 的 `tl.dot` 发出 `mma.sync`（tensor core）；**3.7.1** 一条 `mma.sync` 都没有，全是 `fma.rn.f32`，结果正确但极慢。该 issue 里 8k prefill 比 FlashInfer 慢 15×。
- 上游曾提 PR #9970 想恢复 Turing MMA，**被拒绝**。所以这是编译器政策的主动放弃，不是待修的 bug。

降级到 **triton 3.2.0** 后，同一份 `diag_gemm.py`：

| | triton 3.8.0 | triton 3.2.0 |
|---|---|---|
| Triton 纯 GEMM 最高 | 2.9 TFLOPS（cuBLAS 的 11%） | **21.6 TFLOPS（74%）** |
| 寄存器溢出 | 最高 2774 次 | **全部 0** |

全量 `bench.py` 重跑，kernel 代码一行没改：

| | triton 3.8.0 | triton 3.2.0 |
|---|---|---|
| 第 14 题 | 4.0% 峰值，0.17× SDPA | **26.8–28.7% 峰值，1.07–1.10× SDPA** |
| 第 17 题 | 1.9% 峰值 | **18.8% 峰值** |
| 第 18 题 | 2.5% 峰值 | **23.9–27.3% 峰值** |

> 上表 3.2 那列是两次独立实测的范围（首次 / 2026-11-21 复跑）。核心判断不变：Triton 3.8 → 3.2，14 题从 4% 跳到两位数并反超 SDPA。

> 这条经验比"把 T4 调到 60%"更值钱：性能问题被归因到**编译器版本**这一层，用一条版本分界线 + 一个降级实验闭环。面试讲的是"我怎么定位的"，不是"我把数字调高了"。

### 一个仍未定的小优化

`diag_gemm.py` 显示 3.2 下纯 GEMM 是 `w4` 更快（BM128×BN128：21.6 vs 18.2 TFLOPS），而当前 kernels 13/14/15/18 在 `cap major < 8` 时用的是 `w8`——那是为压 3.8 的 spill 加的，3.2 下 spill 已归零，`w8` 可能是负优化。但 attention 的 tile 里还有 online softmax 中间量，和纯 GEMM 不同构，**没有实测就不改**。想验证就把 `kernels/14_flash_attn.py` 里 Turing 分支的 `wm` 临时改回 4 跑一次 `bench.py --only 14` 对比；当前 w8 的 26.8–28.7% 是两次实测验证过的好成绩，没有收益证据不动它。

---

## 几个反直觉的实测结论（面试可讲）

1. **算力减半只换来 20% 提速**（第 15 题）：causal 砍掉一半 FLOPs，但耗时只从 0.087 降到 0.059。原因是块数从 1024 降到 528，每行工作量不同，SM 空载——不是算力问题，是负载不均。

2. **一个 `.item()` 让 kernel 慢 32 倍**（第 19 题）：`run()` 里读 `seqlen.item()` 会触发 GPU→CPU 同步，bench 调上千次就是上千次同步。改成从 `block_table.shape[0]` 读 Python 层已知元数据后，0.458 ms → 0.014 ms。

3. **形状检查会骗你**（第 18 题）：kernel B 里 `p` 本该直接 `dot(p, do)`，因为 `p` 已经是 `(key, query)` 转置姿态。加一个 `trans(p)` 在 BM==BN==64 时形状照样过，dK/dV 却差 0.29——靠 `--odd` 把 d 改成 72（BD=128≠BM）才暴露。

4. **同进程连测不可信**：笔记本 GPU 有功耗墙，前 13 题把 SM 顶到降频，第 14 题单跑 67.0%、全量连跑只有 59.0%。报成绩一律用单独冷机的数。

---

## 这些 kernel 的实验场

[triton-ops-lab](https://github.com/AwakeZz/triton-ops-lab) 是完整的 19 题练习集（含访存 8 题、GEMM 4 题、attention 7 题的 `--odd` 边界验收、探针脚本和逐题 docstring）。本仓库是其中 attention 线的独立打包，便于放到云上跑和展示。
