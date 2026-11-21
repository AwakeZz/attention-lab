"""
Attention 算子库 benchmark —— 运行时实测硬件峰值 + 全场景对标。

为什么要自己写 harness 而不是复用练习集的 check.py:
    练习集的峰值分母是硬编码的 3050 Ti 实测值 (180.5 GB/s / 18.2 TFLOPS)。
    这份项目要能上 Colab 的 A100/L4/T4 跑, 不同卡的峰值差一个数量级 ——
    分母必须在运行时实测出来, 否则 "% 峰值" 这列数字全是错的。

设计的三条口径 (和练习集 check.py 完全一致, 保证数字可比):
    1. 一个配置跑完再跑下一个, bench 内部预热 + 自适应迭代 + 中位数 (见 bench())。
    2. FLOPs 用"理想算力"口径: 前向不落 (S,S) 按 4*Sq*Sk*d, causal 按一半算,
       backward 按前向 2.5x。不是数 kernel 实际做了多少 dot。
    3. 正确性每个 kernel 都和 fp32 参考比对 (rtol=1e-2, atol=2e-2)。

用法:
    python bench.py                 # 全部 7 个 kernel + 硬件画像 + SDPA 对标
    python bench.py --only 14 16    # 只跑指定题号
    python bench.py --no-ref        # 跳过正确性校验, 只测速 (快一点)
    Colab: 直接跑 attention_lab.ipynb, 它 clone 本仓库后调用本脚本。
"""
import argparse
import importlib.util
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

DEV = "cuda"

# 标准尺寸 (和练习集一致, 保证数字可比)
AS, AD = 2048, 64       # prefill: S=2048, d=64
DSQ, DSK = 16, 8192     # decode: 16 个新 query, 8192 长 KV cache
GQ_HQ, GQ_HKV = 16, 4   # GQA: 16 q head 分 4 组共享 K/V
PAGED_BS = 64           # 分页: 每块 64 token, 有效长 8192 -> 128 逻辑块
PAGED_SL = DSK

# ---------------------------------------------------------------------------
# 计时: 预热把时钟拉满, 自适应迭代, 取中位数
# ---------------------------------------------------------------------------
def bench(fn, target_total_ms=80.0, min_iters=20, ramp_ms=800):
    fn()                                   # 首次含 JIT 编译, 排除
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    single = max((time.perf_counter() - t0) * 1e3, 1e-3)
    warm = max(1, min(int(ramp_ms / single), 4000))
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    floor = 3 if single > 50 else min_iters
    iters = min(max(floor, int(target_total_ms / single)), 5000)
    samples = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        samples.append(a.elapsed_time(b))
    return statistics.median(samples), iters


# ---------------------------------------------------------------------------
# 硬件画像: 运行时实测带宽峰值与 fp16 tensor core 峰值
# ---------------------------------------------------------------------------
@triton.jit
def _copy_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    x = tl.load(x_ptr + off, m, other=0)
    tl.store(y_ptr + off, x, m)


def measure_peak_bw():
    """copy (读 1 遍 + 写 1 遍) 实测持续带宽, GB/s。256 MB 够大, 绕过 L2。"""
    n = 256 * 1024 * 1024 // 4                 # 256 MB / fp32
    x = torch.randn(n, device=DEV, dtype=torch.float32)
    y = torch.empty_like(x)
    block = 4096
    grid = (triton.cdiv(n, block),)
    ms, _ = bench(lambda: _copy_kernel[grid](x, y, n, block, num_warps=4))
    return (2 * n * 4) / (ms * 1e-3) / 1e9      # 读+写各 n*4 字节


def measure_peak_flops():
    """fp16 GEMM 实测 tensor core 峰值, TFLOPS。cuBLAS 当基准, 3072³ 足够大。"""
    m = n = k = 3072
    a = torch.randn(m, k, device=DEV, dtype=torch.float16)
    b = torch.randn(k, n, device=DEV, dtype=torch.float16)
    ms, _ = bench(lambda: torch.mm(a, b))
    return (2 * m * n * k) / (ms * 1e-3) / 1e12


def gpu_profile():
    p = torch.cuda.get_device_properties(0)
    return dict(name=p.name, sm=p.multi_processor_count,
                mem_gb=p.total_memory / 1024**3,
                bw=measure_peak_bw(), flops=measure_peak_flops())


# ---------------------------------------------------------------------------
# 输入构造 (从练习集 check.py 搬来, 保证 make 逻辑一致)
# ---------------------------------------------------------------------------
def mk_attn(s=AS, d=AD):
    q = torch.randn(s, d, device=DEV, dtype=torch.float16)
    k = torch.randn(s, d, device=DEV, dtype=torch.float16)
    v = torch.randn(s, d, device=DEV, dtype=torch.float16)
    return dict(q=q, k=k, v=v, out=torch.empty(s, d, device=DEV, dtype=torch.float32))


def mk_decode(sq=DSQ, sk=DSK, d=AD):
    q = torch.randn(sq, d, device=DEV, dtype=torch.float16)
    k = torch.randn(sk, d, device=DEV, dtype=torch.float16)
    v = torch.randn(sk, d, device=DEV, dtype=torch.float16)
    return dict(q=q, k=k, v=v, out=torch.empty(sq, d, device=DEV, dtype=torch.float32))


def mk_gqa(sq=DSQ, sk=DSK, d=AD):
    hq, hkv = GQ_HQ, GQ_HKV
    q = torch.randn(hq, sq, d, device=DEV, dtype=torch.float16)
    k = torch.randn(hkv, sk, d, device=DEV, dtype=torch.float16)
    v = torch.randn(hkv, sk, d, device=DEV, dtype=torch.float16)
    return dict(q=q, k=k, v=v, out=torch.empty(hq, sq, d, device=DEV, dtype=torch.float32))


def mk_attn_bwd(s=AS, d=AD):
    q = torch.randn(s, d, device=DEV, dtype=torch.float16)
    k = torch.randn(s, d, device=DEV, dtype=torch.float16)
    v = torch.randn(s, d, device=DEV, dtype=torch.float16)
    sc = (q.float() @ k.float().T) * (1.0 / (d ** 0.5))
    lse = torch.logsumexp(sc, dim=1)
    o = torch.softmax(sc, dim=1) @ v.float()
    do = torch.randn(s, d, device=DEV, dtype=torch.float16)
    return dict(q=q, k=k, v=v, o=o, lse=lse, do=do,
                dq=torch.empty(s, d, device=DEV, dtype=torch.float32),
                dk=torch.empty(s, d, device=DEV, dtype=torch.float32),
                dv=torch.empty(s, d, device=DEV, dtype=torch.float32))


def mk_paged(sq=DSQ, seqlen=PAGED_SL, d=AD, bs=PAGED_BS):
    nblk = (seqlen + bs - 1) // bs
    nb = nblk + max(8, nblk // 4)
    perm = torch.randperm(nb, device=DEV)
    block_table = perm[:nblk].to(torch.int32)
    q = torch.randn(sq, d, device=DEV, dtype=torch.float16)
    k = torch.randn(nb, bs, d, device=DEV, dtype=torch.float16)
    v = torch.randn(nb, bs, d, device=DEV, dtype=torch.float16)
    seqlen_t = torch.tensor([seqlen], device=DEV, dtype=torch.int32)
    return dict(q=q, k=k, v=v, block_table=block_table, seqlen=seqlen_t,
                out=torch.empty(sq, d, device=DEV, dtype=torch.float32))


# ---------------------------------------------------------------------------
# fp32 参考实现
# ---------------------------------------------------------------------------
def ref_attn(t, causal=False):
    q, k, v = t["q"], t["k"], t["v"]
    d = q.shape[1]
    s = (q.float() @ k.float().T) * (1.0 / (d ** 0.5))
    if causal:
        mask = torch.tril(torch.ones(q.shape[0], q.shape[0], device=DEV, dtype=torch.bool))
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v.float()


def ref_gqa(t):
    q, k, v = t["q"], t["k"], t["v"]
    hq, sq, d = q.shape
    group = hq // k.shape[0]
    out = torch.empty(hq, sq, d, device=DEV, dtype=torch.float32)
    for h in range(hq):
        kh = h // group
        s = (q[h].float() @ k[kh].float().T) * (1.0 / d ** 0.5)
        out[h] = torch.softmax(s, dim=-1) @ v[kh].float()
    return out


def ref_bwd(t):
    q, k, v = t["q"], t["k"], t["v"]
    d = q.shape[1]
    qf = q.float().detach().requires_grad_(True)
    kf = k.float().detach().requires_grad_(True)
    vf = v.float().detach().requires_grad_(True)
    o = torch.softmax((qf @ kf.T) * (1.0 / d ** 0.5), dim=1) @ vf
    return list(torch.autograd.grad(o, (qf, kf, vf), t["do"].float()))


def ref_paged(t):
    q, k, v, bt = t["q"], t["k"], t["v"], t["block_table"]
    seqlen = int(t["seqlen"][0].item())
    d, bs = q.shape[1], k.shape[1]
    nblk = (seqlen + bs - 1) // bs
    # 按逻辑块顺序 index_select 整块物理块 -> 拉平 -> 截断到 seqlen。
    # 注意是选"整块"(dim=0), 不是按展平后的元素下标 gather —— 后者块号*bs 会越界。
    phys = bt[:nblk].long()
    kk = torch.index_select(k, 0, phys).reshape(-1, d)[:seqlen].float()
    vv = torch.index_select(v, 0, phys).reshape(-1, d)[:seqlen].float()
    s = (q.float() @ kk.T) * (1.0 / d ** 0.5)
    return torch.softmax(s, dim=-1) @ vv


# ---------------------------------------------------------------------------
# 动态加载 kernel (文件名以数字开头, 不能普通 import)
# ---------------------------------------------------------------------------
def load_kernel(num):
    here = Path(__file__).parent
    path = next(here.glob(f"kernels/{num}_*.py"))
    spec = importlib.util.spec_from_file_location(f"k{num}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path.name


# ---------------------------------------------------------------------------
# 题目定义: make / ref / args / FLOPs / SDPA 对标开关
# ---------------------------------------------------------------------------
def flops_attn(t):   return 4 * t["q"].shape[0] ** 2 * t["q"].shape[1]
def flops_causal(t): return 2 * t["q"].shape[0] ** 2 * t["q"].shape[1]
def flops_decode(t): return 4 * t["q"].shape[0] * t["k"].shape[0] * t["q"].shape[1]
def flops_gqa(t):    return 4 * t["q"].shape[0] * t["q"].shape[1] * t["k"].shape[1] * t["q"].shape[2]
def flops_bwd(t):    return 10 * t["q"].shape[0] ** 2 * t["q"].shape[1]
def flops_paged(t):  return 4 * t["q"].shape[0] * int(t["seqlen"][0].item()) * t["q"].shape[1]

PROBLEMS = [
    dict(num=13, name="attention 三趟走 (反面教材)", make=mk_attn,
         args=("q", "k", "v", "out"), ref=lambda t: ref_attn(t), flops=flops_attn, sdpa="full"),
    dict(num=14, name="FlashAttention (online softmax)", make=mk_attn,
         args=("q", "k", "v", "out"), ref=lambda t: ref_attn(t), flops=flops_attn, sdpa="full"),
    dict(num=15, name="FlashAttention + causal", make=mk_attn,
         args=("q", "k", "v", "out"), ref=lambda t: ref_attn(t, causal=True),
         flops=flops_causal, sdpa="causal"),
    dict(num=16, name="split-KV decode", make=mk_decode,
         args=("q", "k", "v", "out"), ref=lambda t: ref_attn(t), flops=flops_decode, sdpa="decode"),
    dict(num=17, name="GQA decode (共享 K/V)", make=mk_gqa,
         args=("q", "k", "v", "out"), ref=ref_gqa, flops=flops_gqa, sdpa="gqa"),
    dict(num=18, name="backward dQ/dK/dV", make=mk_attn_bwd,
         args=("q", "k", "v", "o", "lse", "do", "dq", "dk", "dv"),
         outs=("dq", "dk", "dv"), ref=ref_bwd, flops=flops_bwd, sdpa=None),
    dict(num=19, name="PagedAttention (vLLM)", make=mk_paged,
         args=("q", "k", "v", "block_table", "seqlen", "out"),
         ref=ref_paged, flops=flops_paged, sdpa=None),
]


def sdpa_ms(t, kind):
    """torch SDPA 对照, 用同一个计时口径。返回 ms, None 表示不适用。"""
    # 两个必须做对的点, 否则"加速比"是假的:
    #   1. dtype 和 kernel 一致 (fp16)。传 .float() 走 fp32 math 后端, 慢一个数量级。
    #   2. SDPA 要 4D (batch, heads, seq, d)。直接传 2D/3D 会掉到次优路径,
    #      实测 0.127 ms -> 0.99 ms (慢 8 倍), 那种数字不能拿去比。
    # GQA 用原生 enable_gqa=True, 不手动 repeat_interleave (那会凭空复制 K/V 4 份)。
    try:
        q, k, v = t["q"], t["k"], t["v"]
        if q.ndim == 3:                       # GQA: (HQ, Sq, d) -> (1, HQ, Sq, d)
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        else:                                 # 普通: (S, d) -> (1, 1, S, d)
            q, k, v = q.unsqueeze(0).unsqueeze(0), k.unsqueeze(0).unsqueeze(0), v.unsqueeze(0).unsqueeze(0)
        if kind in ("full", "causal"):
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=(kind == "causal"))
        elif kind == "decode":
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
        elif kind == "gqa":
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, enable_gqa=True)
        else:
            return None
        return bench(fn)[0]
    except Exception as e:
        print(f"    [SDPA 对照跳过: {e}]")
        return None


def run_one(p, prof, do_ref=True):
    mod, fname = load_kernel(p["num"])
    t = p["make"]()
    args = [t[a] for a in p["args"]]
    fn = lambda: mod.run(*args)

    ok = True
    if do_ref:
        try:
            fn()                               # 必须先跑一次填充 out (否则比对的是未初始化 empty)
            torch.cuda.synchronize()
            ref = p["ref"](t)
            got = [t[o] for o in p["outs"]] if "outs" in p else t["out"]
            if isinstance(got, list):
                ok = all(torch.allclose(g.float(), r, rtol=1e-2, atol=2e-2)
                         for g, r in zip(got, ref))
            else:
                ok = torch.allclose(got.float(), ref, rtol=1e-2, atol=2e-2)
        except Exception as e:
            ok = False
            print(f"    [参考比对异常: {e}]")

    ms, iters = bench(fn)
    flops = p["flops"](t)
    tflops = flops / (ms * 1e-3) / 1e12
    sms = sdpa_ms(t, p["sdpa"]) if p["sdpa"] else None
    return dict(num=p["num"], name=p["name"], file=fname, ok=ok, ms=ms,
                tflops=tflops, pct=tflops / prof["flops"] * 100,
                sms=sms, iters=iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, help="只跑指定题号, 如 --only 14 16")
    ap.add_argument("--no-ref", action="store_true", help="跳过正确性校验")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU。Colab 里请 运行时 -> 更改运行时类型 -> 选 T4/A100/L4。")

    print("=" * 78)
    print(f"Attention 算子库 benchmark  |  torch {torch.__version__}  |  triton {triton.__version__}")
    print("=" * 78)
    print("测硬件峰值 (copy 带宽 + fp16 GEMM, 约 10 秒)...")
    prof = gpu_profile()
    print(f"GPU: {prof['name']}  |  SM {prof['sm']}  |  显存 {prof['mem_gb']:.0f} GB")
    print(f"实测带宽 {prof['bw']:.1f} GB/s  |  实测 fp16 峰值 {prof['flops']:.1f} TFLOPS")
    print(f"roofline 拐点 = {prof['flops'] * 1e12 / (prof['bw'] * 1e9):.1f} FLOP/Byte")
    print()

    selected = [p for p in PROBLEMS if not args.only or p["num"] in args.only]
    results = []
    for p in selected:
        print(f"跑 {p['num']:>2} {p['name']} ...", flush=True)
        r = run_one(p, prof, do_ref=not args.no_ref)
        results.append(r)
        flag = "OK" if r["ok"] else "FAIL"
        print(f"    {flag:4}  {r['ms']:.4f} ms  {r['tflops']:.2f} TFLOPS  "
              f"{r['pct']:.1f}% 峰值   ({r['iters']} 次)"
              + (f"  SDPA {r['sms']:.4f} ms" if r["sms"] else ""))

    # Markdown 表, 直接可贴 README
    print("\n" + "=" * 78)
    print("结果表 (可直接贴 README)")
    print("=" * 78)
    print("| kernel | ms | TFLOPS | % 峰值 | SDPA(ms) | 加速比 | 校验 |")
    print("|---|---|---|---|---|---|---|")
    by_num = {r["num"]: r for r in results}
    for r in results:
        sp = f"{r['sms']:.4f}" if r["sms"] else "—"
        if r["sms"]:
            ratio = f"{r['sms'] / r['ms']:.2f}x"
        elif r["num"] == 19 and 16 in by_num:
            ratio = f"{by_num[16]['ms'] / r['ms']:.2f}x vs #16"
        else:
            ratio = "—"
        print(f"| {r['num']} {r['name']} | {r['ms']:.4f} | {r['tflops']:.2f} | "
              f"{r['pct']:.1f}% | {sp} | {ratio} | {'✓' if r['ok'] else '✗'} |")

    print(f"\n分母: {prof['name']} 实测 fp16 {prof['flops']:.1f} TFLOPS "
          f"/ 带宽 {prof['bw']:.1f} GB/s。换卡后分母自动重测, 数字口径统一。")
    if not all(r["ok"] for r in results):
        raise SystemExit("有 kernel 正确性校验失败, 见上表 ✗")


if __name__ == "__main__":
    main()
