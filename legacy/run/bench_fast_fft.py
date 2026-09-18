"""Benchmark grid.round-to-fast-fft: FFT/HAADF timing and GPU memory footprint.

Requires an abTEM checkout of the `fast-fft-grid` branch on PYTHONPATH (the
config option does not exist in older abTEM; with it missing, "fast" silently
equals "ceil" and the comparison is meaningless -- the preflight below aborts).

Usage: python bench_fast_fft.py [cpu] [gpu]   (default: both)

Small case: extent 10 A, sampling 0.03 -> gpts 334 (2x167) vs 336 (2^4*3*7)
Large case: tWSe2 cell 131.1 x 113.5 A, sampling 0.05
            -> gpts (2623, 2271) = (43*61, 3*757) vs (2625, 2304)
"""
import sys
import time
import warnings

import numpy as np
import ase
import abtem
from abtem.core import fft as abtem_fft

warnings.filterwarnings("ignore")

CASES = {
    "small": dict(cell=(10.0, 10.0, 8.0), sampling=0.03, batch=8, scan=(8, 8)),
    "large": dict(
        cell=(131.10557120122698, 113.54075523793207, 40.8),
        sampling=0.05,
        batch=4,
        scan=(2, 2),
    ),
}


def median_time(fn, repeats, sync=None, warmup=2):
    for _ in range(warmup):
        fn()
        if sync:
            sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def gpts_for(case, fast):
    with abtem.config.set({"grid.round-to-fast-fft": fast}):
        from abtem.core.grid import Grid

        g = Grid(extent=case["cell"][:2], sampling=case["sampling"])
        return g.gpts


def preflight():
    print(f"abtem in use: {abtem.__version__} <- {abtem.__file__}")
    if gpts_for(CASES["small"], True) == gpts_for(CASES["small"], False):
        sys.exit(
            "FATAL: grid.round-to-fast-fft has no effect -- this abTEM does not "
            "have the fast-fft-grid feature. Put the branch checkout first on "
            "PYTHONPATH."
        )


def bench_raw_fft(device, case, repeats):
    if device == "gpu":
        import cupy as xp

        sync = xp.cuda.Stream.null.synchronize
    else:
        xp = np
        sync = None

    out = {}
    for mode in ("ceil", "fast"):
        ny, nx = gpts_for(case, mode == "fast")
        a = (xp.random.rand(case["batch"], ny, nx) + 0j).astype("complex64")

        def roundtrip():
            b = abtem_fft.fft2(a.copy(), overwrite_x=True)
            abtem_fft.ifft2(b, overwrite_x=True)

        t = median_time(roundtrip, repeats, sync=sync)
        out[mode] = (t, (ny, nx))
    return out


def bench_haadf(device, case, repeats):
    out = {}
    for mode in ("ceil", "fast"):
        with abtem.config.set(
            {
                "grid.round-to-fast-fft": mode == "fast",
                "device": device,
                "precision": "float32",
            }
        ):
            atoms = ase.Atoms(
                "W4", positions=[(1, 1, z) for z in (5, 15, 25, 35)],
                cell=case["cell"], pbc=True,
            )
            potential = abtem.Potential(
                atoms, sampling=case["sampling"],
                slice_thickness=case["cell"][2] / 4,
            )
            probe = abtem.Probe(energy=60e3, semiangle_cutoff=31.5)
            probe.grid.match(potential)
            detector = abtem.AnnularDetector(inner=70, outer=200)
            scan = abtem.GridScan(
                start=(0, 0), end=(0.2, 0.2), gpts=case["scan"], fractional=True,
                potential=potential,
            )

            def run():
                probe.scan(potential, scan=scan, detectors=detector).compute(
                    progress_bar=False
                )

            t = median_time(run, repeats, warmup=1)
            out[mode] = (t, probe.gpts)
    return out


def bench_gpu_memory(case):
    """Pool footprint retained after one fft2+ifft2 round-trip per shape.

    The plan cache and pool are cleared first, so the difference between the
    array bytes and the pool total is FFT workspace retained by the cached
    plans (Bluestein padding on slow shapes). Requires an unbounded plan
    cache -- force it in case the abTEM in use bounds it by default.
    """
    import cupy as cp

    out = {}
    with abtem.config.set({"cupy.fft-cache-size": -1}):
        for mode in ("ceil", "fast"):
            ny, nx = gpts_for(case, mode == "fast")
            cp.fft.config.get_plan_cache().clear()
            cp.get_default_memory_pool().free_all_blocks()

            a = (cp.random.rand(case["batch"], ny, nx) + 0j).astype("complex64")
            b = abtem_fft.fft2(a.copy(), overwrite_x=True)
            abtem_fft.ifft2(b, overwrite_x=True)
            cp.cuda.Stream.null.synchronize()

            pool = cp.get_default_memory_pool().total_bytes()
            out[mode] = (a.nbytes, pool, (ny, nx))
            del a, b
    return out


def report(name, res):
    tc, gc = res["ceil"]
    tf, gf = res["fast"]
    pix = (gf[0] * gf[1]) / (gc[0] * gc[1])
    print(
        f"  {name:28s} ceil {gc}: {tc*1e3:9.1f} ms | fast {gf}: {tf*1e3:9.1f} ms"
        f" | speedup {tc/tf:5.2f}x (pixels x{pix:.3f})"
    )


def report_memory(name, res):
    for mode in ("ceil", "fast"):
        data, pool, gpts = res[mode]
        print(
            f"  {name:20s} {mode:4s} {str(gpts):13s} data {data/1e6:7.1f} MB"
            f" | pool after round-trip {pool/1e6:9.1f} MB ({pool/data:5.1f}x data)"
        )


if __name__ == "__main__":
    devices = [d for d in sys.argv[1:] if d in ("cpu", "gpu")] or ["cpu", "gpu"]
    preflight()
    abtem.config.set({"fftw.planning_effort": "FFTW_ESTIMATE"})

    for device in devices:
        print(f"== {device.upper()} timing ==")
        for cname, case in CASES.items():
            rep_fft = 20 if (cname == "small" or device == "gpu") else 5
            report(f"fft2+ifft2 [{cname}] (B={case['batch']})",
                   bench_raw_fft(device, case, rep_fft))
        for cname, case in CASES.items():
            rep = 5 if cname == "small" else 3
            npos = case["scan"][0] * case["scan"][1]
            report(f"HAADF scan [{cname}] ({npos} pos)",
                   bench_haadf(device, case, rep))

    if "gpu" in devices:
        print("== GPU memory (plan-cache workspace retention) ==")
        for cname, case in CASES.items():
            report_memory(f"[{cname}] B={case['batch']}", bench_gpu_memory(case))
