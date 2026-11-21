"""决定性诊断: Triton tl.dot 在 T4(sm_75) 上到底能不能用 tensor core。
Part A: Triton 纯 GEMM vs cuBLAS — 若 Triton << cuBLAS, 则后端坏了, attention 怎么调都没用。
Part B: 若后端正常, 再扫 attention 的 tile 找瓶颈。
"""
import torch, triton, triton.language as tl, time

@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm*BM + tl.arange(0,BM); rn = pn*BN + tl.arange(0,BN); rk = tl.arange(0,BK)
    acc = tl.zeros((BM,BN), tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:,None]*K + (k0+rk)[None,:])
        b = tl.load(b_ptr + (k0+rk)[:,None]*N + rn[None,:])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:,None]*N + rn[None,:], acc)

def timeit(fn, n=50):
    for _ in range(30): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n*1e3

print('='*70)
p = torch.cuda.get_device_properties(0)
print(f'GPU {p.name} sm_{torch.cuda.get_device_capability(0)} | triton {triton.__version__}')
print('='*70)

M = N = K = 2048
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
c = torch.empty(M, N, device='cuda', dtype=torch.float16)
GEMM_FLOPS = 2*M*N*K

ms_cu = timeit(lambda: torch.mm(a, b))
print(f'[cuBLAS] torch.mm 2048^3: {ms_cu:.3f} ms = {GEMM_FLOPS/(ms_cu*1e-3)/1e12:.1f} TFLOPS')

print('\n[Triton 纯 GEMM] 2048^3 fp16 — 决定性问题: 能否接近 cuBLAS?')
for BM, BN, BK in [(64,64,32),(64,64,64),(128,64,32),(128,128,32)]:
    grid = (triton.cdiv(M,BM), triton.cdiv(N,BN))
    for w in (4, 8):
        for st in (2, 3):
            try:
                kern = mm_kernel[grid](a, b, c, M, N, K, BM, BN, BK,
                                       num_warps=w, num_stages=st)
                ms = timeit(lambda: mm_kernel[grid](a, b, c, M, N, K, BM, BN, BK,
                                                   num_warps=w, num_stages=st))
                tf = GEMM_FLOPS/(ms*1e-3)/1e12
                print(f'  BM{BM} BN{BN} BK{BK} w{w} s{st}: {ms:.3f}ms {tf:.1f}TF '
                      f'sh{kern.metadata.shared//1024}KB regs{kern.n_regs} spill{kern.n_spills}')
            except Exception as e:
                print(f'  BM{BM} BN{BN} BK{BK} w{w} s{st}: ERR {str(e)[:50]}')

print('\n判定: 若 Triton 最高 << cuBLAS 的 50%, 则是 sm_75 后端问题, 不是 attention 代码问题。')
