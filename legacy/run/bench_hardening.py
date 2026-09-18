"""Benchmark a branch (dev vs multi-gpu-hardening) on identical workloads.

Usage: python bench_hardening.py <label>
Prints one line per case: median wall time, result checksum (must match across
branches), and GPU pool retention after the case (plan-cache bound visibility).
Uses only APIs present on both branches (explicit gpts; no config keys that
exist on only one branch).
"""
import os
import sys
import time
import shutil
import warnings

import numpy as np
import ase
import abtem

warnings.filterwarnings("ignore")

LABEL = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"
ZARR = "/tmp/claude-1000/-workspaces/f3c99596-723a-4623-8943-790967a88c77/scratchpad/bench_hardening_out.zarr"

CASES = [
    # name, device, gpts, cell(xy), scan_gpts, repeats
    ("small-fast  gpu scan", "gpu", (336, 336), (10.0, 10.0), (8, 8), 5),
    ("large-fast  gpu scan", "gpu", (2625, 2304), (131.106, 113.541), (4, 4), 3),
    ("large-slow  gpu scan", "gpu", (2623, 2271), (131.106, 113.541), (4, 4), 3),
    ("small-fast  gpu to_zarr", "gpu", (336, 336), (10.0, 10.0), (8, 8), 3),
    ("small-fast  cpu scan", "cpu", (336, 336), (10.0, 10.0), (4, 4), 3),
    ("large-fast  cpu scan", "cpu", (2625, 2304), (131.106, 113.541), (2, 2), 3),
]


def build(device, gpts, cell_xy):
    atoms = ase.Atoms(
        "W4", positions=[(1, 1, z) for z in (5, 15, 25, 35)],
        cell=(*cell_xy, 40.8), pbc=True,
    )
    potential = abtem.Potential(atoms, gpts=gpts, slice_thickness=10.2)
    probe = abtem.Probe(energy=60e3, semiangle_cutoff=31.5)
    probe.grid.match(potential)
    detector = abtem.AnnularDetector(inner=70, outer=200)
    return potential, probe, detector


def run_case(name, device, gpts, cell_xy, scan_gpts, repeats):
    to_zarr = "to_zarr" in name
    with abtem.config.set(
        {"device": device, "precision": "float32",
         "fftw.planning_effort": "FFTW_ESTIMATE"}
    ):
        potential, probe, detector = build(device, gpts, cell_xy)
        scan = abtem.GridScan(
            start=(0, 0), end=(0.2, 0.2), gpts=scan_gpts, fractional=True,
            potential=potential,
        )

        def run():
            m = probe.scan(potential, scan=scan, detectors=detector)
            if to_zarr:
                shutil.rmtree(ZARR, ignore_errors=True)
                m.to_zarr(ZARR)
                return None
            return m.compute(progress_bar=False)

        result = run()  # warmup (plans, wisdom, numba)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = run() or result
            times.append(time.perf_counter() - t0)

        if result is None:
            checksum = float("nan")
        else:
            checksum = float(
                np.asarray(result.to_cpu().array, dtype=np.float64).sum()
            )

        pool_mb = float("nan")
        if device == "gpu":
            import cupy as cp

            pool_mb = cp.get_default_memory_pool().total_bytes() / 1e6

        print(
            f"[{LABEL}] {name:26s} {str(gpts):13s} "
            f"median {np.median(times)*1e3:9.1f} ms  "
            f"checksum {checksum:.6e}  pool {pool_mb:9.1f} MB"
        )



def _git_rev(module_file):
    """Best-effort commit of the abTEM checkout in use (provenance)."""
    import os
    import subprocess

    repo = os.path.dirname(os.path.dirname(os.path.abspath(module_file)))
    try:
        out = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", repo, "describe",
             "--always", "--dirty"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if out:
            return out
    except Exception:  # noqa: BLE001 -- fall through to file-based lookup
        pass
    # No usable git binary (e.g. inside a container): read .git directly.
    try:
        head = open(os.path.join(repo, ".git", "HEAD")).read().strip()
        if head.startswith("ref:"):
            ref = head.split(None, 1)[1]
            return open(os.path.join(repo, ".git", ref)).read().strip()[:8]
        return head[:8]
    except Exception:  # noqa: BLE001 -- provenance only
        return "unknown"


if __name__ == "__main__":
    print(f"[{LABEL}] abtem <- {abtem.__file__}")
    print(f"[{LABEL}] abtem commit: {_git_rev(abtem.__file__)}")
    # A/B hook for the cuFFT plan-cache bound, e.g. BENCH_FFT_CACHE=-1 to
    # lift it, BENCH_FFT_CACHE="4 GB" to widen it. Unset = branch default.
    _cache = os.environ.get("BENCH_FFT_CACHE")
    if _cache is not None:
        try:
            _cache_val: object = int(_cache)
        except ValueError:
            _cache_val = _cache
        abtem.config.set({"cupy.fft-cache-size": _cache_val})
        print(f"[{LABEL}] cupy.fft-cache-size = {_cache!r}")
    for case in CASES:
        run_case(*case)
    shutil.rmtree(ZARR, ignore_errors=True)
