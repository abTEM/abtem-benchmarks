#!/usr/bin/env python
"""Diagnose worker host/device memory growth in abTEM transition-potential scans.

Runs a *tiny* scan (a handful of probe positions) on the full production grid,
several times, and reports per-worker host RSS, CuPy pinned-host pool, CuPy
device pool and the worker's data container after each run. The production scan
stalls before it can be measured; a few positions exercise exactly the same
per-task allocations, so growth per run is what matters -- not absolute time.

Three configurations are compared:
  1. transition potentials as abTEM builds them today (complex128 -- a bug)
  2. cast to complex64 (the fix)
  3. complex64 + smaller max_batch

Usage (inside the container, 4 GPUs visible):
    python eels_memory_diagnose.py
    python eels_memory_diagnose.py --structure Im_22_defect_2026-07-01.json
    python eels_memory_diagnose.py --single          # 1 GPU, no cluster
    python eels_memory_diagnose.py --gpts 256 256    # quick smoke test
"""

import argparse
import os
import warnings


def worker_snapshot():
    """Collected on each worker: host RSS and the CuPy host/device pools."""
    import os

    import psutil

    out = {"rss_GB": psutil.Process(os.getpid()).memory_info().rss / 1e9}
    try:
        import cupy as cp

        # PinnedMemoryPool exposes no byte total, only a free-block count;
        # the decisive measurement is how much RSS free_all_blocks() returns.
        out["pinned_blocks"] = float(cp.get_default_pinned_memory_pool().n_free_blocks())
        pool = cp.get_default_memory_pool()
        out["dev_pool_GB"] = pool.total_bytes() / 1e9
        out["dev_used_GB"] = pool.used_bytes() / 1e9
    except Exception as exc:  # noqa: BLE001 -- reporting only
        out["cupy_error"] = str(exc)[:60]
    return out


def free_pools_and_snapshot():
    """Release both CuPy pools, then re-measure -- run on each worker."""
    try:
        import cupy as cp

        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:  # noqa: BLE001 -- reporting only
        pass
    return worker_snapshot()


def _format(snap):
    nan = float("nan")
    return (
        f"rss {snap.get('rss_GB', nan):6.2f} GB | "
        f"dev pool {snap.get('dev_pool_GB', nan):6.2f} GB "
        f"(used {snap.get('dev_used_GB', nan):5.2f}) | "
        f"pinned blocks {int(snap.get('pinned_blocks', 0)):6d}"
        + (f" | {snap['cupy_error']}" if "cupy_error" in snap else "")
    )


def report(client, label, free_pools=False):
    func = free_pools_and_snapshot if free_pools else worker_snapshot
    if client is None:
        print(f"  [{label}] {_format(func())}")
        return
    print(f"  [{label}]")
    for i, (_addr, snap) in enumerate(sorted(client.run(func).items())):
        print(f"    w{i}  {_format(snap)}")


def build_structure(path, cell_xy):
    """The user's structure if available, else an hBN sheet of the same cell."""
    import numpy as np
    from ase import Atoms
    from ase.io import read

    if path and os.path.exists(path):
        atoms = read(path)
        atoms.rotate(180, "y", rotate_cell=True)
        atoms.positions -= np.min(atoms.positions, axis=0)
        atoms.cell = np.max(atoms.positions, axis=0)
        atoms.rotate(3, "z", center=(0, 0, 0))
        print(f"structure: {path}  ({len(atoms)} atoms)")
        return atoms

    a = 2.504
    b = a * np.sqrt(3)
    base = Atoms(
        "BNBN",
        positions=[(0, 0, 0), (0, b / 3, 0), (a / 2, b / 2, 0), (a / 2, b / 2 + b / 3, 0)],
        cell=(a, b, 8.0),
        pbc=True,
    )
    atoms = base * (round(cell_xy[0] / a), round(cell_xy[1] / b), 1)
    atoms.cell = (*cell_xy, 8.0)
    print(f"structure: synthetic hBN sheet ({len(atoms)} atoms) -- {path} not found")
    return atoms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", default="Im_22_defect_2026-07-01.json")
    parser.add_argument("--gpts", type=int, nargs=2, default=(1344, 1440))
    parser.add_argument("--scan-gpts", type=int, nargs=2, default=(4, 4),
                        help="probe positions per run (keep tiny)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="runs per configuration, to expose growth")
    parser.add_argument("--single", action="store_true",
                        help="single GPU in-process, no dask-cuda cluster")
    parser.add_argument("--graph-size", action="store_true",
                        help="also pickle the task graph and report its size")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)

    import numpy as np
    import abtem
    from abtem.inelastic.core_loss import SubshellTransitions

    abtem.config.set({"device": "gpu", "dask.multi-gpu": not args.single})
    print(f"abtem {abtem.__version__} <- {abtem.__file__}")

    client = None
    if not args.single:
        from abtem.core.backend import ensure_cuda_cluster

        client = ensure_cuda_cluster()
        print(f"cluster: {len(client.nthreads())} workers")
        containers = client.run(lambda dask_worker: type(dask_worker.data).__name__)
        limits = client.run(
            lambda dask_worker: dask_worker.memory_manager.memory_limit / 1e9)
        print(f"  worker data container: {sorted(set(containers.values()))}"
              "   <- a device/host spilling container means dask-cuda spills to host")
        print(f"  worker memory limit  : "
              f"{sorted({round(v, 1) for v in limits.values()})} GB")

    report(client, "baseline (cluster just started)")

    atoms = build_structure(args.structure, (65.32314062797602, 71.0770016))
    potential = abtem.Potential(
        atoms, gpts=tuple(args.gpts), slice_thickness=atoms.cell[2, 2] / 4
    )

    transitions = SubshellTransitions(Z=5, n=1, l=0, xc="PBE", order=1, epsilon=10)
    tpots = transitions.get_transition_potentials(energy=60e3)
    tpots.grid.match(potential)
    print(f"building transition potentials on {tuple(args.gpts)} ...")
    arr128 = tpots.build()
    arr64 = arr128.copy()
    arr64._array = arr128.array.astype(np.complex64)
    print(f"  as built : {arr128.array.dtype}  {arr128.array.nbytes / 1e6:6.1f} MB")
    print(f"  cast     : {arr64.array.dtype}  {arr64.array.nbytes / 1e6:6.1f} MB")

    probe = abtem.Probe(semiangle_cutoff=32, energy=60e3)
    probe.grid.match(potential)
    scan = abtem.GridScan(
        start=(5 / 32, 11 / 32), end=(15 / 32, 24 / 32), gpts=tuple(args.scan_gpts),
        fractional=True, potential=potential, endpoint=False,
    )
    sites = atoms[atoms.numbers == 5]
    print(f"B sites: {len(sites)}   scan positions per run: "
          f"{args.scan_gpts[0] * args.scan_gpts[1]}")

    configs = [
        ("complex128 (as built), max_batch=20", arr128, 20),
        ("complex64  (cast),     max_batch=20", arr64, 20),
        ("complex64  (cast),     max_batch=4 ", arr64, 4),
    ]

    for label, tp, max_batch in configs:
        print(f"\n=== {label} ===")
        peak = 4 * max_batch * args.gpts[0] * args.gpts[1] * tp.array.itemsize
        print(f"  predicted scattered-wave allocation per scatter step: "
              f"{peak / 1e9:.2f} GB  (n_transitions x max_batch x grid x itemsize)")
        for run in range(args.repeats):
            measurement = probe.transition_potential_scan(
                scan=scan, potential=potential,
                detectors=abtem.FlexibleAnnularDetector(),
                transition_potentials=tp, double_channel=False, sites=sites,
                max_batch=max_batch, threshold=0.95, lazy=True,
            ).integrate_radial(inner=0, outer=48)

            if args.graph_size and run == 0:
                import pickle

                try:
                    n = len(pickle.dumps(measurement.array.__dask_graph__(), protocol=5))
                    print(f"  task graph: {n / 2**20:.2f} MiB")
                except Exception as exc:  # noqa: BLE001
                    print(f"  task graph: unmeasurable ({type(exc).__name__})")

            import time

            t0 = time.perf_counter()
            result = measurement.compute(progress_bar=False)
            dt = time.perf_counter() - t0
            total = float(np.asarray(result.to_cpu().array).sum())
            print(f"  run {run + 1}/{args.repeats}: {dt:6.1f} s   sum={total:.6e}")
            report(client, f"after run {run + 1}")
        report(client, "after releasing both CuPy pools", free_pools=True)

    print("\nReading the numbers:")
    print("  * rss climbs run over run, and drops sharply on 'after releasing")
    print("    both CuPy pools' -> the unmanaged memory is CuPy's pinned-host")
    print("    pool: every task re-uploads the transition potentials, and the")
    print("    pool never returns those buffers to the OS. Workaround: call")
    print("    cupy.get_default_pinned_memory_pool().free_all_blocks() between")
    print("    computes, or keep the potentials resident on the workers.")
    print("  * rss climbs but does NOT drop after releasing the pools, and the")
    print("    worker data container above is a device/host spilling type ->")
    print("    dask-cuda is spilling device data to host. Reduce device")
    print("    pressure: smaller max_batch, complex64 potentials.")
    print("  * dev_pool_GB approaching the card size -> device pressure is the")
    print("    driver either way; max_batch is the strongest lever.")
    print("  * rss flat across all three configs -> the stall is elsewhere;")
    print("    send this output and we look at the worker task stream next.")


if __name__ == "__main__":
    main()
