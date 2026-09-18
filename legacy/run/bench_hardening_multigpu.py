"""Multi-GPU to_zarr benchmark: dev vs multi-gpu-hardening on a real workload.

Runs a to_zarr-terminated HAADF scan with dask.multi-gpu enabled. On dev the
flag is ignored on this path (whole scan serial on GPU 0); on the hardening
branch the cluster is bootstrapped and the scan distributes. Both branches
compute the same physics -- the printed checksum must match.

Sized to complete on ONE 40 GiB A100 (so dev finishes rather than OOMs):
2623x2271 float32 grid, 16x16 = 256 probe positions, 4 slices.

With --xl it instead reproduces the ORIGINAL production failure mode: float64,
the same Bluestein 2623x2271 grid, and a full Nyquist-sampled fractional scan
(~100k probe positions, ~2.9 GB complex128 auto-batches on dev). On dev this
is expected to OOM a 40 GiB card within minutes (serial GPU 0, unbounded plan
cache, 6x batch estimate, doubled input copy); on the hardening branch it
should complete, distributed. Expect ~15 min on 4 A100s. Do NOT run --xl on a
small local machine.

Usage: python bench_hardening_multigpu.py <label> [--xl] [--max-batch N]
                                          [--compute-first]
--compute-first triggers computation via .compute() before saving --
the path on which dev DOES start the multi-GPU cluster. With --xl this
tests whether dev's VRAM sizing survives distribution (fair comparison
isolating the sizing fixes from the save-path bootstrap fix).
--max-batch overrides the VRAM-aware auto batch size (probes per batch),
for measuring whether larger batches beat the conservative 12x Bluestein
estimate on the hardening branch.
Requires the respective abTEM checkout first on PYTHONPATH and >= 2 visible
GPUs. The __main__ guard is mandatory: dask-cuda spawns worker processes.
"""
import sys
import time
import shutil
import warnings

import numpy as np



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


def main():
    # Keep UserWarnings visible: the hardening branch warns with the REASON
    # when dask.multi-gpu is requested but declined.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    import ase
    import abtem
    import cupy as cp
    import zarr

    args = sys.argv[1:]
    xl = "--xl" in args
    compute_first = "--compute-first" in args
    max_batch = "auto"
    if "--max-batch" in args:
        max_batch = int(args[args.index("--max-batch") + 1])
    label = next(
        (a for a in args if not a.startswith("-") and not a.isdigit()),
        "unlabeled",
    )
    out = "bench_hardening_multigpu_out.zarr"

    print(f"[{label}] abtem <- {abtem.__file__}")
    print(f"[{label}] abtem commit: {_git_rev(abtem.__file__)}")
    print(f"[{label}] visible GPUs: {cp.cuda.runtime.getDeviceCount()}")
    try:
        import dask_cuda

        print(f"[{label}] dask_cuda {dask_cuda.__version__}")
    except ImportError as exc:
        print(f"[{label}] dask_cuda NOT importable: {exc}")

    abtem.config.set(
        {
            "device": "gpu",
            "precision": "float64" if xl else "float32",
            "dask.multi-gpu": True,
        }
    )

    atoms = ase.Atoms(
        "W4", positions=[(1, 1, z) for z in (5, 15, 25, 35)],
        cell=(131.10557120122698, 113.54075523793207, 40.8), pbc=True,
    )
    potential = abtem.Potential(
        atoms, gpts=(2623, 2271), slice_thickness=10.2
    )
    probe = abtem.Probe(energy=60e3, semiangle_cutoff=31.5)
    probe.grid.match(potential)
    detector = abtem.AnnularDetector(inner=70, outer=200)
    if xl:
        # No explicit scan gpts: Nyquist sampling over the full cell, ~339x294
        # positions -- the original production scan.
        scan = abtem.GridScan(
            start=(0, 0), end=(1, 1), fractional=True, potential=potential
        )
    else:
        scan = abtem.GridScan(
            start=(0, 0), end=(1, 1), gpts=(16, 16), fractional=True,
            potential=potential,
        )
    measurement = probe.scan(
        potential, scan=scan, detectors=detector, max_batch=max_batch
    )
    print(f"[{label}] max_batch: {max_batch} | "
          f"trigger: {'compute()+to_zarr' if compute_first else 'to_zarr'}")

    # For the Nyquist-sampled xl scan, gpts resolve during probe.scan (on a
    # validated copy of the scan object) -- read them off the measurement.
    scan_shape = measurement.shape[-2:]
    n_pos = int(np.prod(scan_shape))
    print(f"[{label}] scan {scan_shape} = {n_pos} positions, "
          f"precision {'float64' if xl else 'float32'}")

    # Cluster detection: public accessor on the hardening branch, private
    # attribute fallback on dev (where it stays None on this code path).
    def cluster_workers():
        try:
            from abtem.core.backend import get_cuda_cluster_client

            client = get_cuda_cluster_client()
        except ImportError:
            from abtem.core import backend

            client = getattr(backend, "_cuda_cluster_client", None)
        return len(client.nthreads()) if client is not None else 0

    # Two passes: "cold" includes one-time costs (cluster bring-up, worker
    # imports, JIT, FFT plan building); "warm" is the steady state, which is
    # the number comparable to the validated compute-path scaling. The xl
    # reproduction runs a single pass -- its point is completes-vs-OOMs.
    for phase in (("cold",) if xl else ("cold", "warm")):
        shutil.rmtree(out, ignore_errors=True)
        # Rebuild the lazy graph each pass: compute() materializes in place,
        # so reusing the object would make the second pass a no-op.
        lazy = probe.scan(
            potential, scan=scan, detectors=detector, max_batch=max_batch
        )
        t0 = time.perf_counter()
        if compute_first:
            # The fair-comparison mode: .compute() bootstraps the multi-GPU
            # cluster on BOTH branches (dev honors the flag on this path), so
            # this isolates the VRAM-sizing changes from the save-path fix.
            lazy = lazy.compute(progress_bar=False)
        lazy.to_zarr(out)
        elapsed = time.perf_counter() - t0

        root = zarr.open(out, mode="r")
        checksum = float(np.asarray(root["array0"], dtype=np.float64).sum())

        print(
            f"[{label}] to_zarr {n_pos} pos (2623, 2271) {phase}: {elapsed:8.1f} s"
            f" | cluster workers: {cluster_workers()} | checksum {checksum:.12e}"
        )

    # Pool footprint: per worker under a cluster, else this process's pool.
    try:
        from abtem.core.backend import get_cuda_cluster_client

        client = get_cuda_cluster_client()
    except ImportError:
        from abtem.core import backend

        client = getattr(backend, "_cuda_cluster_client", None)
    try:
        if client is not None:
            pools = client.run(
                lambda: __import__("cupy").get_default_memory_pool().total_bytes()
            )
            per_worker = ", ".join(f"{b / 1e9:.1f}" for b in pools.values())
            print(f"[{label}] worker pools after run [GB]: {per_worker}")
        else:
            import cupy as _cp

            pool = _cp.get_default_memory_pool().total_bytes()
            print(f"[{label}] pool after run: {pool / 1e9:.1f} GB")
    except Exception as exc:  # noqa: BLE001 -- reporting only
        print(f"[{label}] pool query failed: {exc}")
    shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
