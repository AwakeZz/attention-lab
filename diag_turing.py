"""
T4 / Turing 专用诊断脚本 —— 排查"为什么在 Colab 上这么慢 / 某题崩了"。

用法 (Colab notebook 里, clone 代码之后):
    !python diag_turing.py

它做三件事:
  1. 打印 GPU 架构和 shared memory 上限
  2. 对第 14/17/18 题的核心 kernel 测 num_stages=1 vs 2 的实际 shared 占用
  3. 对第 14 题测不同 num_stages / num_warps 的耗时, 找出 Turing 上的最优配置

输出直接贴回来即可, 据此改 run() 里的分档。本脚本不改任何 kernel。
"""
import torch, triton, triton.language as tl

dev = 'cuda'
p = torch.cuda.get_device_properties(0)
cap = torch.cuda.get_device_capability(0)
print('=' * 70)
print(f'GPU: {p.name} | sm_{cap[0]}{cap[1]} | cap={cap}')
print(f'SM {p.multi_processor_count} | 显存 {p.total_memory/1024**3:.0f} GB')
print(f'shared/block: {getattr(p,"shared_memory_per_block","?")} B  '
      f'({getattr(p,"shared_memory_per_block",0)/1024:.1f} KB)')
print(f'shared/SM:    {getattr(p,"shared_memory_per_multiprocessor","?")} B')
print(f'torch {torch.__version__} | triton {triton.__version__}')
print('=' * 70)

# 一个最小的 matmul, 用来探 tl.dot 在 Turing 上的 shared 代价
@triton.jit
def _probe_dot(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m*BM + tl.arange(0, BM); rn = pid_n*BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:,None]*K + (k0+rk)[None,:])
        b = tl.load(b_ptr + (k0+rk)[:,None]*N + rn[None,:])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:,None]*N + rn[None,:], acc)

print('\n[1] tl.dot 基础 shared 占用 (BM=BN=64, BK=32, fp16):')
a = torch.randn(256, 256, device=dev, dtype=torch.float16)
b = torch.randn(256, 256, device=dev, dtype=torch.float16)
c = torch.empty(256, 256, device=dev, dtype=torch.float32)
for st in (1, 2, 3):
    try:
        kern = _probe_dot[(4, 4)](a, b, c, 256, 256, 256, 64, 64, 32,
                                  num_warps=4, num_stages=st)
        print(f'  num_stages={st}: shared={kern.metadata.shared} B '
              f'({kern.metadata.shared/1024:.1f} KB), regs={kern.n_regs}, spills={kern.n_spills}')
    except Exception as e:
        print(f'  num_stages={st}: ERR {type(e).__name__}: {str(e)[:90]}')

# 第 14 题: 用它的真实 kernel 测不同 stages/warps 的耗时
import importlib.util
def load(num, name):
    spec = importlib.util.spec_from_file_location(f'diag{num}', f'kernels/{num}_{name}.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

print('\n[2] 第 14 题 FlashAttention kernel: stages × warps 耗时 (S=2048,d=64,BM=BN=64):')
m14 = load(14, 'flash_attn')
S, d, BM, BN, BD = 2048, 64, 64, 64, 64
q = torch.randn(S, d, device=dev, dtype=torch.float16)
k = torch.randn(S, d, device=dev, dtype=torch.float16)
v = torch.randn(S, d, device=dev, dtype=torch.float16)
out = torch.empty(S, d, device=dev, dtype=torch.float32)
scale = d ** -0.5
grid = (triton.cdiv(S, BM),)
for st in (1, 2, 3):
    for nw in (4, 8):
        try:
            kern = m14.flash_attn_kernel[grid](q, k, v, out, S, d, scale, BM, BN, BD,
                                               num_warps=nw, num_stages=st)
            # 简单计时
            import time
            for _ in range(5):
                m14.flash_attn_kernel[grid](q, k, v, out, S, d, scale, BM, BN, BD,
                                            num_warps=nw, num_stages=st)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                m14.flash_attn_kernel[grid](q, k, v, out, S, d, scale, BM, BN, BD,
                                            num_warps=nw, num_stages=st)
            torch.cuda.synchronize()
            ms = (time.perf_counter()-t0)/20*1e3
            print(f'  stages={st} warps={nw}: shared={kern.metadata.shared/1024:.1f} KB, '
                  f'{ms:.3f} ms, regs={kern.n_regs}, spills={kern.n_spills}')
        except Exception as e:
            print(f'  stages={st} warps={nw}: ERR {type(e).__name__}: {str(e)[:90]}')

print('\n[3] 第 17 题 split 3D vs 2D (看 2D 版能否在 T4 编过):')
m17 = load(17, 'gqa_decode')
HQ, HKV, Sq, Sk, GROUP, NS = 16, 4, 16, 8192, 4, 8
BM2, BN2, BD2 = 16, 64, 64
q7 = torch.randn(HQ, Sq, d, device=dev, dtype=torch.float16)
k7 = torch.randn(HKV, Sk, d, device=dev, dtype=torch.float16)
v7 = torch.randn(HKV, Sk, d, device=dev, dtype=torch.float16)
op = torch.empty(NS, HQ, Sq, d, device=dev, dtype=torch.float32)
lse = torch.empty(NS, HQ, Sq, device=dev, dtype=torch.float32)
grid7 = (HKV, triton.cdiv(Sq, BM2), NS)
for label, fn in [('3D', m17._split_kernel), ('2D', m17._split_kernel_2d)]:
    try:
        kern = fn[grid7](q7, k7, v7, op, lse, HQ, HKV, Sq, Sk, d, scale,
                         BM2, BN2, BD2, GROUP, NS, num_warps=4, num_stages=2)
        print(f'  {label}: shared={kern.metadata.shared} B ({kern.metadata.shared/1024:.1f} KB) OK')
    except Exception as e:
        print(f'  {label}: ERR {type(e).__name__}: {str(e)[:100]}')

print('\n诊断结束。把以上输出贴回来即可。')
