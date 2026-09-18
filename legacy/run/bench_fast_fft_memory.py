"""Does the FFT-friendly grid actually use less GPU memory? Measure it.

For ceil (2623, 2271) vs fast (2625, 2304) grids, at complex64 and complex128:

  A. cuFFT plan memsize for a 16-probe batch (unbounded cache, direct read) --
     the isolated plan-workspace claim.
  B. End-to-end 16-position HAADF scan under SHIPPED abTEM defaults
     (auto plan-cache bound), with a sampler thread recording the PEAK pool --
     the number that decides whether the claim survives in reality. Also
     reports after-run pool, wall time, and whether the oversized-plan
     fallback engaged.

Usage: python bench_fast_fft_memory.py <label> [--smoke] [--xl]
--smoke uses small grids (334/336) for a fast API check on any machine.
--xl runs the full production scan (Nyquist, ~101k positions) instead of 16
positions, float64 only -- the configuration whose memory behavior the claim
is actually about. Serial: budget ~1 h for the ceil case and ~30-45 min for
the fast case.
Run single-GPU (CUDA_VISIBLE_DEVICES=0); requires the multi-gpu-hardening
checkout on PYTHONPATH (auto cache bound + fallback).
"""
import sys
import threading
import time
import warnings

import numpy as np
import ase
import abtem
import cupy as cp

SMOKE = "--smoke" in sys.argv[1:]
XL = "--xl" in sys.argv[1:]
LABEL = next((a for a in sys.argv[1:] if not a.startswith("-")), "unlabeled")

if SMOKE:
    GRIDS = [("ceil", (334, 334), (10.0, 10.0)), ("fast", (336, 336), (10.0, 10.0))]
else:
    CELL = (131.10557120122698, 113.54075523793207)
    GRIDS = [("ceil", (2623, 2271), CELL), ("fast", (2625, 2304), CELL)]

PRECISIONS = ["float64"] if XL else ["float32", "float64"]


def plan_memsize(gpts, precision):
    cdtype = "complex64" if precision == "float32" else "complex128"
    cache = cp.fft.config.get_plan_cache()
    cache.clear()
    cache.set_memsize(-1)
    a = cp.zeros((16, *gpts), dtype=cdtype)
    cp.fft.ifft2(a)
    cp.cuda.Stream.null.synchronize()
    size = int(cache.get_curr_memsize())
    del a
    cache.clear()
    cp.get_default_memory_pool().free_all_blocks()
    return size


class PoolSampler:
    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        pool = cp.get_default_memory_pool()
        while not self._stop.is_set():
            self.peak = max(self.peak, pool.total_bytes())
            time.sleep(0.01)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, cp.get_default_memory_pool().total_bytes())


def scan_case(gpts, cell, precision):
    # Shipped defaults: auto plan-cache bound, auto batches; serial (1 GPU).
    abtem.config.set({"device": "gpu", "precision": precision})
    from abtem.core import fft as abtem_fft

    abtem_fft._CUFFT_CACHE_STATE = None  # re-resolve the bound for this case
    abtem_fft._warned_plan_cache_bypass = False
    cp.fft.config.get_plan_cache().clear()
    cp.get_default_memory_pool().free_all_blocks()

    atoms = ase.Atoms(
        "W4", positions=[(1, 1, z) for z in (5, 15, 25, 35)],
        cell=(*cell, 40.8), pbc=True,
    )
    potential = abtem.Potential(atoms, gpts=gpts, slice_thickness=10.2)
    probe = abtem.Probe(energy=60e3, semiangle_cutoff=31.5)
    probe.grid.match(potential)
    detector = abtem.AnnularDetector(inner=70, outer=200)
    if XL:
        scan = abtem.GridScan(
            start=(0, 0), end=(1, 1), fractional=True, potential=potential
        )
    else:
        scan = abtem.GridScan(
            start=(0, 0), end=(0.2, 0.2), gpts=(4, 4), fractional=True,
            potential=potential,
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        with PoolSampler() as sampler:
            result = probe.scan(
                potential, scan=scan, detectors=detector
            ).compute(progress_bar=False)
        elapsed = time.perf_counter() - t0

    fallback = any("uncached" in str(w.message) for w in caught)
    after = cp.get_default_memory_pool().total_bytes()
    checksum = float(np.asarray(result.to_cpu().array, dtype=np.float64).sum())
    return dict(
        peak_GB=sampler.peak / 1e9, after_GB=after / 1e9, time_s=elapsed,
        fallback=fallback, checksum=checksum,
    )


if __name__ == "__main__":
    print(f"[{LABEL}] abtem <- {abtem.__file__}")
    for precision in PRECISIONS:
        for name, gpts, cell in GRIDS:
            plan = plan_memsize(gpts, precision)
            prefix = f"[{LABEL}] {precision:7s} {name:4s} {str(gpts):13s} plan {plan / 1e9:6.2f} GB"
            try:
                r = scan_case(gpts, cell, precision)
            except Exception as exc:  # noqa: BLE001 -- OOM is a RESULT here
                pool = cp.get_default_memory_pool()
                print(
                    f"{prefix} | scan FAILED at pool {pool.total_bytes() / 1e9:6.2f} GB: "
                    f"{type(exc).__name__}: {str(exc)[:120]}"
                )
                cp.fft.config.get_plan_cache().clear()
                pool.free_all_blocks()
                continue
            print(
                f"{prefix} | scan peak {r['peak_GB']:6.2f} GB "
                f"after {r['after_GB']:6.2f} GB | {r['time_s']:7.1f} s | "
                f"fallback {'YES' if r['fallback'] else 'no '} | "
                f"checksum {r['checksum']:.6e}"
            )
