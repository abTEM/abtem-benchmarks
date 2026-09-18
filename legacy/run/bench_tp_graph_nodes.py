"""Benchmark the transition-potential graph transport (PR: tp-graph-nodes).

What this measures and why
--------------------------
The change does NOT make the multislice arithmetic faster. It removes a
*per-task copy* of the transition potential from the task graph. So the axes
that move are graph size, scheduler traffic and worker memory -- all of which
scale with the number of tasks -- while steady-state compute stays put. A
benchmark that only timed a small scan would show nothing and be misread as
"no benefit". Hence three modes:

  graph    client-only: build the lazy graph and weigh it. Cheap (nothing is
           computed), so it reaches production payloads and task counts in
           seconds. This is the headline measurement.
  cluster  real distributed execution under a worker memory cap: peak worker
           RSS, wall time, and whether the run survives at all. This is the
           failure mode the change exists to remove.
  compute  single-process end-to-end wall time. The honest control: it should
           be FLAT. A regression here would mean the change cost something.

Runs unchanged on both refs (no branch-only imports or config keys), so the
same file can be pointed at dev and at the branch:

    python bench_tp_graph_nodes.py dev
    PYTHONPATH=<worktree> python -P bench_tp_graph_nodes.py pr3

Env knobs:
    BENCH_MODE=graph|cluster|compute   (default graph)
    BENCH_TIER=quick|standard|production  (default quick)
    BENCH_DEVICE=cpu|gpu               (default cpu)
    BENCH_WORKERS=2                    (cluster mode, cpu)
    BENCH_MEMLIMIT=4GB                 (cluster mode, cpu, per worker)
    BENCH_SYNTHETIC=1                  skip GPAW; synthetic payload of the
                                       same shape/dtype (auto if GPAW absent)
    BENCH_REPEATS=1
    BENCH_CASES=<regex>                run only matching cases, e.g. "512|1024"
                                       matched against "<gpts> scan<n>x<n> mb<b>"

On GPU, `cluster` mode uses abTEM's own multi-GPU cluster (`dask.multi-gpu`)
rather than a hand-rolled one, and reports the per-worker CuPy pool.

Physics: hBN, B K edge, 60 kV, 32 mrad probe, eps=10 eV -- the Sonder
production configuration. Sites are left to abTEM (`sites=None`): explicitly
passed sites are not wrapped into the cell and can be silently dropped, and
the Z filter downstream keeps only boron regardless.
"""
import os
import sys
import threading
import time
import warnings

import numpy as np
import ase
import abtem

warnings.filterwarnings("ignore")

LABEL = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"
MODE = os.environ.get("BENCH_MODE", "graph")
TIER = os.environ.get("BENCH_TIER", "quick")
DEVICE = os.environ.get("BENCH_DEVICE", "cpu")
REPEATS = int(os.environ.get("BENCH_REPEATS", "1"))
SYNTHETIC = os.environ.get("BENCH_SYNTHETIC", "") not in ("", "0")
CASE_FILTER = os.environ.get("BENCH_CASES", "")

# gpts, scan side, max_batch. max_batch=1 maximises the task count, which is
# exactly the multiplier the change removes.
TIERS = {
    "quick":      [(256, 8, 2), (256, 16, 1)],
    "standard":   [(512, 16, 2), (512, 32, 1), (1024, 16, 1)],
    "production": [(1024, 32, 1), (2048, 32, 1)],
    # Calibrated for cluster mode on a workstation: a payload big enough that
    # the per-task copies dominate a capped worker, a scan small enough to
    # finish in a couple of minutes.
    "cluster": [(512, 12, 1)],
}


def cases():
    """The tier's cases, narrowed by BENCH_CASES."""
    import re

    selected = [
        c for c in TIERS[TIER]
        if not CASE_FILTER
        or re.search(CASE_FILTER, f"{c[0]} scan{c[1]}x{c[1]} mb{c[2]}")
    ]
    if not selected:
        raise SystemExit(
            f"BENCH_CASES={CASE_FILTER!r} matched no case in tier {TIER!r}: "
            + ", ".join(f"{g} scan{n}x{n} mb{b}" for g, n, b in TIERS[TIER])
        )
    return selected


def pin_config():
    """Pin everything the suite plan flags as result- or timing-affecting."""
    abtem.config.set(
        {
            "device": DEVICE,
            "precision": "float32",
            "fft": "fftw",
            "fftw.planning_effort": "FFTW_ESTIMATE",
            "fftw.threads": 1,
            "dask.lazy": True,
        }
    )


def hbn_slab(reps=(6, 4, 2)):
    """hBN slab with z padding -- a cell whose z equals max(z) silently folds
    the top plane onto z=0 and drops its atoms from the site list."""
    a = 2.504
    b = a * np.sqrt(3)
    c = 3.33
    base = ase.Atoms(
        "BNBN",
        positions=[
            (0, 0, 0.0), (0, b / 3, 0.0),
            (a / 2, b / 2, c), (a / 2, b / 2 + b / 3, c),
        ],
        cell=(a, b, 2 * c), pbc=True,
    )
    atoms = base * reps
    atoms.cell[2, 2] = float(atoms.positions[:, 2].max()) + c  # padding
    return atoms


_SETUP_CACHE = {}


def _synthetic_tp(gpts, extent, n_transitions=4):
    """A payload of exactly the shape and dtype the real B K edge produces.

    Graph transport and memory depend on the payload's size, not its values,
    so this is a faithful stand-in wherever GPAW is unavailable. Checksums
    from a synthetic run are only comparable to other synthetic runs.
    """
    from abtem.core.axes import OrdinalAxis
    from abtem.inelastic.core_loss import TransitionPotentialArray

    rng = np.random.default_rng(0)
    array = (
        rng.standard_normal((n_transitions, gpts, gpts))
        + 1j * rng.standard_normal((n_transitions, gpts, gpts))
    ).astype(np.complex64)
    return TransitionPotentialArray(
        Z=5, array=array, energy=60e3, extent=extent,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_transitions)))],
        metadata={"Z": 5, "n": 1, "l": 0},
    )


def setup(gpts):
    """Case setup -- excluded from every timing below."""
    key = (gpts, DEVICE)
    if key in _SETUP_CACHE:
        return _SETUP_CACHE[key]

    atoms = hbn_slab()
    potential = abtem.Potential(atoms, gpts=(gpts, gpts), slice_thickness=1.665)

    tp = None
    if not SYNTHETIC:
        try:
            from abtem.inelastic.core_loss import SubshellTransitions

            transitions = SubshellTransitions(
                Z=5, n=1, l=0, xc="PBE", order=1, epsilon=10
            )
            tp = transitions.get_transition_potentials(
                energy=60e3, extent=potential.extent, gpts=(gpts, gpts)
            ).build()
        except Exception as exc:  # noqa: BLE001 -- GPAW missing or unusable
            print(f"[{LABEL}] NOTE: real transition potentials unavailable "
                  f"({type(exc).__name__}: {exc}); using a synthetic payload "
                  "of the same shape. Checksums are then only comparable "
                  "between synthetic runs.")
    if tp is None:
        tp = _synthetic_tp(gpts, potential.extent)
    probe = abtem.Probe(energy=60e3, semiangle_cutoff=32)
    probe.grid.match(potential)
    _SETUP_CACHE[key] = (potential, tp, probe)
    return _SETUP_CACHE[key]


def build_lazy(gpts, scan_n, max_batch):
    potential, tp, probe = setup(gpts)
    scan = abtem.GridScan(
        start=(0, 0), end=(1, 1), gpts=(scan_n, scan_n),
        fractional=True, potential=potential,
    )
    return probe.transition_potential_scan(
        scan=scan, potential=potential,
        detectors=abtem.FlexibleAnnularDetector(),
        transition_potentials=tp, double_channel=False,
        sites=None, max_batch=max_batch, threshold=1.0, lazy=True,
    ), tp


class PeakRSS:
    """Sample this process's RSS in the background; report the peak."""

    def __init__(self, interval=0.02):
        self._interval = interval
        self._stop = threading.Event()
        self.peak = 0

    def __enter__(self):
        try:
            import psutil
        except ImportError:
            self._thread = None
            self.peak = float("nan")
            return self

        proc = psutil.Process()

        def poll():
            while not self._stop.is_set():
                try:
                    self.peak = max(self.peak, proc.memory_info().rss)
                except Exception:
                    pass
                self._stop.wait(self._interval)

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


class WorkerRSS:
    """Peak host RSS per worker, read from the scheduler's heartbeat state.

    Deliberately does NOT use ``client.run``: that executes code on every
    worker, and on a run already sitting on dask's 80 % pause threshold the
    extra load was enough to stop it recovering (observed on dev at 2048²,
    which livelocked for 20+ minutes where an unsampled run finished in 184 s).
    ``scheduler_info`` is served from state the scheduler already holds, so
    sampling costs the workers nothing -- and ``metrics["memory"]`` is the
    same figure dask itself uses to decide when to pause a worker.

    ``client_getter`` is polled rather than a client being passed in: abTEM
    bootstraps its multi-GPU cluster inside ``compute()``, so there is no
    client to sample until the run is already under way.
    """

    def __init__(self, client_getter, interval=1.0):
        self._get_client = client_getter
        self._interval = interval
        self._stop = threading.Event()
        self.peaks = {}

    def __enter__(self):
        def poll():
            while not self._stop.is_set():
                client = self._get_client()
                if client is not None:
                    try:
                        workers = client.scheduler_info()["workers"]
                        for addr, info in workers.items():
                            rss = info.get("metrics", {}).get("memory")
                            if rss:
                                self.peaks[addr] = max(self.peaks.get(addr, 0), rss)
                    except Exception:  # noqa: BLE001 -- scheduler busy/restarting
                        pass
                self._stop.wait(self._interval)

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)

    def report(self):
        if not self.peaks:
            return "n/a"
        return ", ".join(f"{v/1e9:.1f}" for v in sorted(self.peaks.values(), reverse=True))

    @property
    def max_gb(self):
        return max(self.peaks.values()) / 1e9 if self.peaks else float("nan")


def weigh_graph(measurement, payload_bytes):
    """Total serialized task payload -- what distributed actually ships -- and
    how many keys are payload-sized."""
    import cloudpickle

    graph = dict(measurement.array.__dask_graph__())
    sizes = [len(cloudpickle.dumps(value)) for value in graph.values()]
    big = [s for s in sizes if s > payload_bytes / 2]
    return len(graph), len(big), sum(sizes)


def run_graph():
    print(f"{'case':>22} {'payload':>9} {'tasks':>7} {'pKeys':>6} "
          f"{'graph MB':>10} {'build s':>8} {'peakRSS MB':>11}")
    for gpts, scan_n, max_batch in cases():
        setup(gpts)  # warm the cache so build time excludes it
        best = None
        for _ in range(REPEATS):
            with PeakRSS() as rss:
                t0 = time.perf_counter()
                measurement, tp = build_lazy(gpts, scan_n, max_batch)
                build_s = time.perf_counter() - t0
                n_tasks, n_big, total = weigh_graph(measurement, tp.array.nbytes)
            row = (tp.array.nbytes, n_tasks, n_big, total, build_s, rss.peak)
            best = row if best is None else min(best, row, key=lambda r: r[4])
        payload, n_tasks, n_big, total, build_s, peak = best
        name = f"{gpts}^2 scan{scan_n}x{scan_n} mb{max_batch}"
        print(f"[{LABEL}] {name:>21} {payload/1e6:8.1f}M {n_tasks:7d} {n_big:6d} "
              f"{total/1e6:10.1f} {build_s:8.2f} {peak/1e6:11.1f}")


def run_compute():
    # On GPU this is deliberately the synchronous scheduler on ONE device --
    # no graph serialization at all -- which is what makes it the control:
    # any wall-time difference here is arithmetic, not transport. It is also
    # why it is far slower than the 4-GPU cluster mode, and it runs
    # REPEATS + 1 passes per case (one warmup).
    print(f"{'case':>22} {'wall s':>9} {'peakRSS MB':>11}  checksum")
    for gpts, scan_n, max_batch in cases():
        setup(gpts)
        name = f"{gpts}^2 scan{scan_n}x{scan_n} mb{max_batch}"
        times = []
        result = None
        for i in range(REPEATS + 1):  # first is warmup
            phase = "warmup" if i == 0 else f"timed {i}/{REPEATS}"
            print(f"[{LABEL}] {name:>21}  {phase} ...", flush=True)
            measurement, _ = build_lazy(gpts, scan_n, max_batch)
            with PeakRSS() as rss:
                t0 = time.perf_counter()
                result = measurement.compute(progress_bar=False)
                times.append(time.perf_counter() - t0)
            print(f"[{LABEL}] {name:>21}  {phase} done in {times[-1]:.2f} s",
                  flush=True)
        checksum = float(np.asarray(result.to_cpu().array, dtype=np.float64).sum())
        print(f"[{LABEL}] {name:>21} {np.median(times[1:]):9.2f} "
              f"{rss.peak/1e6:11.1f}  {checksum:.8e}")


def _cuda_cluster_client():
    """abTEM's multi-GPU client; the private attribute is the dev fallback."""
    try:
        from abtem.core.backend import get_cuda_cluster_client

        return get_cuda_cluster_client()
    except ImportError:
        from abtem.core import backend

        return getattr(backend, "_cuda_cluster_client", None)


def run_cluster_gpu():
    """Use abTEM's own dask-cuda cluster, the way a production run does."""
    abtem.config.set({"dask.multi-gpu": True})
    print(f"{'case':>22} {'wall s':>9} {'workers':>8} {'peakRSS GB':>11}  "
          f"pools GB / checksum")
    for gpts, scan_n, max_batch in cases():
        setup(gpts)
        measurement, _ = build_lazy(gpts, scan_n, max_batch)
        with WorkerRSS(_cuda_cluster_client) as rss:
            t0 = time.perf_counter()
            try:
                result = measurement.compute(progress_bar=False)
                checksum = float(
                    np.asarray(result.to_cpu().array, dtype=np.float64).sum()
                )
                status = f"{checksum:.8e}"
            except Exception as exc:  # noqa: BLE001 -- an OOM here IS the result
                status = f"{type(exc).__name__}: {str(exc)[:60]}"
            wall = time.perf_counter() - t0

        client = _cuda_cluster_client()
        n_workers = len(client.nthreads()) if client is not None else 0
        try:
            if client is not None:
                pools = client.run(
                    lambda: __import__("cupy").get_default_memory_pool().total_bytes()
                )
                pool_str = ", ".join(f"{b/1e9:.1f}" for b in pools.values())
            else:
                import cupy as cp

                pool_str = f"{cp.get_default_memory_pool().total_bytes()/1e9:.1f}"
        except Exception:  # noqa: BLE001
            pool_str = "n/a"

        name = f"{gpts}^2 scan{scan_n}x{scan_n} mb{max_batch}"
        print(f"[{LABEL}] {name:>21} {wall:9.2f} {n_workers:8d} {rss.max_gb:11.1f}  "
              f"{pool_str} / {status}")
        print(f"[{LABEL}] {'':>21} per-worker peak host RSS [GB]: {rss.report()}")


def run_cluster():
    if DEVICE == "gpu":
        return run_cluster_gpu()

    from distributed import Client, LocalCluster

    n_workers = int(os.environ.get("BENCH_WORKERS", "2"))
    mem_limit = os.environ.get("BENCH_MEMLIMIT", "4GB")

    print(f"{'case':>22} {'wall s':>9} {'peakWorker MB':>14} {'status':>10}  checksum")
    for gpts, scan_n, max_batch in cases():
        setup(gpts)
        measurement, _ = build_lazy(gpts, scan_n, max_batch)
        with LocalCluster(
            n_workers=n_workers, processes=True, threads_per_worker=1,
            memory_limit=mem_limit, dashboard_address=None,
        ) as cluster, Client(cluster) as client:

            with WorkerRSS(lambda: client) as rss:
                t0 = time.perf_counter()
                try:
                    result = measurement.compute(progress_bar=False)
                    checksum = float(
                        np.asarray(result.to_cpu().array, dtype=np.float64).sum()
                    )
                    status = "ok"
                except Exception as exc:  # noqa: BLE001 -- the failure IS the result
                    checksum = float("nan")
                    status = type(exc).__name__
                wall = time.perf_counter() - t0

        name = f"{gpts}^2 scan{scan_n}x{scan_n} mb{max_batch}"
        print(f"[{LABEL}] {name:>21} {wall:9.2f} {rss.max_gb*1e3:14.1f} "
              f"{status:>10}  {checksum:.8e}")


def git_rev(module_file):
    import subprocess

    repo = os.path.dirname(os.path.dirname(os.path.abspath(module_file)))
    try:
        out = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if out:
            return out
    except Exception:  # noqa: BLE001 -- fall through to the file-based lookup
        pass
    # No usable git binary -- the common case inside a container. Read .git
    # directly; also handles a worktree, whose .git is a file, not a dir.
    try:
        dot_git = os.path.join(repo, ".git")
        if os.path.isfile(dot_git):
            with open(dot_git) as fh:
                dot_git = fh.read().split("gitdir:", 1)[1].strip()
        with open(os.path.join(dot_git, "HEAD")) as fh:
            head = fh.read().strip()
        if head.startswith("ref:"):
            ref = head.split(None, 1)[1]
            branch = ref.rsplit("/", 1)[-1]
            try:
                with open(os.path.join(dot_git, ref)) as fh:
                    return f"{fh.read().strip()[:8]} ({branch})"
            except FileNotFoundError:  # packed refs
                return f"unknown ({branch})"
        return head[:8]
    except Exception:  # noqa: BLE001 -- provenance only
        return "unknown"


if __name__ == "__main__":
    pin_config()
    print(f"[{LABEL}] abtem  <- {abtem.__file__}")
    print(f"[{LABEL}] commit <- {git_rev(abtem.__file__)}")
    print(f"[{LABEL}] mode={MODE} tier={TIER} device={DEVICE} repeats={REPEATS}")
    {"graph": run_graph, "compute": run_compute, "cluster": run_cluster}[MODE]()
