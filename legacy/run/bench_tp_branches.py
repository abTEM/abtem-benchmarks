"""Benchmark a branch (dev vs test/transition-potential-combined) on identical workloads.

Usage: python bench_tp_branches.py <label>
      BENCH_CASES=<regex>  run only matching cases (e.g. BENCH_CASES=haadf)
      BENCH_DEVICE=cpu     force CPU (default gpu)
      BENCH_SCATTER=off    disable dask.scatter-large-inputs (combined branch only; ignored on dev)
      BENCH_FFT_ENTRIES=N  set cupy.fft-cache-entries (combined branch only; ignored on dev)

Prints one line per case: median wall time, result checksum (must match across
branches bit-for-bit for HAADF/filter/detector cases; the core-loss cases are
NOT bit-identical across branches because the dtype fix intentionally stops the
complex128 upcast -- compare those checksums to ~1e-6 relative), and GPU pool
retention. Modeled on bench_hardening.py; uses only APIs present on both
branches, guarding config keys that exist on only one.

HAADF rationale: the detector-bin cache and the cuFFT plan-entry limit are hot
in plain STEM imaging too (FlexibleAnnularDetector rebuilds polar bin geometry
per wave-function chunk; frozen-phonon batching varies FFT batch shapes), and
the remaining branches must at minimum not regress elastic imaging.
"""
import os
import re
import sys
import time
import warnings

import numpy as np
import ase
import ase.build
import abtem

warnings.filterwarnings("ignore")

LABEL = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"
DEVICE = os.environ.get("BENCH_DEVICE", "gpu")
CASE_FILTER = os.environ.get("BENCH_CASES", "")
# "full" for cluster runs; "small" shrinks every case for a laptop/CI smoke run
SCALE = os.environ.get("BENCH_SCALE", "full")
SMALL = SCALE == "small"


def set_config_if_supported(key, value):
    """Set an abTEM config key that may not exist on the baseline branch."""
    try:
        abtem.config.set({key: value})
        print(f"[{LABEL}] {key} = {value!r}")
    except Exception:
        print(f"[{LABEL}] {key} not supported on this branch (ok)")


def sto_atoms(reps=(4, 4, 25)):
    a = 3.905
    atoms = ase.Atoms(
        "SrTiO3",
        cell=(a, a, a), pbc=True,
        scaled_positions=[
            (0.0, 0.0, 0.0),        # Sr
            (0.5, 0.5, 0.5),        # Ti
            (0.5, 0.5, 0.0),        # O
            (0.5, 0.0, 0.5),        # O
            (0.0, 0.5, 0.5),        # O
        ],
    )
    return atoms * reps


def bn_slab(reps=(6, 4, 2)):
    a = 2.504
    b = a * np.sqrt(3)
    base = ase.Atoms(
        "BNBN",
        positions=[(0, 0, 1.0), (0, b / 3, 1.0), (a / 2, b / 2, 1.0),
                   (a / 2, b / 2 + b / 3, 1.0)],
        cell=(a, b, 3.33), pbc=True,
    )
    return base * reps


def time_repeats(run, repeats):
    result = run()  # warmup: plans, wisdom, numba, caches under test
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = run()
        times.append(time.perf_counter() - t0)
    return result, times


def report(name, detail, times, result):
    checksum = float(np.asarray(result.to_cpu().array, dtype=np.float64).sum())
    pool_mb = float("nan")
    if DEVICE == "gpu":
        try:
            import cupy as cp
            pool_mb = cp.get_default_memory_pool().total_bytes() / 1e6
        except Exception:
            pass
    print(
        f"[{LABEL}] {name:34s} {detail:16s} "
        f"median {np.median(times) * 1e3:9.1f} ms  "
        f"checksum {checksum:.6e}  pool {pool_mb:9.1f} MB"
    )


# ---------------------------------------------------------------------------
# HAADF cases (bit-identical across branches; regression + PR2 gains)
# ---------------------------------------------------------------------------

def case_haadf_annular(repeats=3):
    """Plain HAADF: AnnularDetector, no phonons. Baseline elastic regression."""
    atoms = sto_atoms((2, 2, 4) if SMALL else (4, 4, 25))
    potential = abtem.Potential(atoms, sampling=0.1 if SMALL else 0.05,
                                slice_thickness=1.95)
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=25)
    probe.grid.match(potential)
    detector = abtem.AnnularDetector(inner=78, outer=110 if SMALL else 200)
    scan = abtem.GridScan(start=(0, 0), end=(0.25, 0.25),
                          gpts=(4, 4) if SMALL else (16, 16),
                          fractional=True, potential=potential)

    def run():
        return probe.scan(potential, scan=scan, detectors=detector).compute(
            progress_bar=False)

    result, times = time_repeats(run, repeats)
    report("haadf annular", f"{potential.gpts}", times, result)


def case_haadf_flexible(repeats=3):
    """HAADF via FlexibleAnnularDetector + integrate_radial: exercises the
    polar detector-bin geometry rebuilt per chunk (cache-detector-bins)."""
    atoms = sto_atoms((2, 2, 4) if SMALL else (4, 4, 25))
    potential = abtem.Potential(atoms, sampling=0.1 if SMALL else 0.05,
                                slice_thickness=1.95)
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=25)
    probe.grid.match(potential)
    detector = abtem.FlexibleAnnularDetector()
    scan = abtem.GridScan(start=(0, 0), end=(0.25, 0.25),
                          gpts=(4, 4) if SMALL else (16, 16),
                          fractional=True, potential=potential)

    def run():
        m = probe.scan(potential, scan=scan, detectors=detector,
                       max_batch=8).compute(progress_bar=False)
        if SMALL:
            # small grid: antialias cutoff caps the flexible detector ~83 mrad
            return m.integrate_radial(inner=50, outer=80)
        return m.integrate_radial(inner=78, outer=200)

    result, times = time_repeats(run, repeats)
    report("haadf flexible+radial", f"{potential.gpts}", times, result)


def case_haadf_phonons(repeats=3):
    """HAADF with frozen phonons: ensemble batching varies FFT batch shapes,
    exercising the cuFFT plan-entry limit (cufft-plan-cache-entries)."""
    atoms = sto_atoms((2, 2, 4) if SMALL else (4, 4, 25))
    phonons = abtem.FrozenPhonons(atoms, num_configs=2 if SMALL else 8,
                                  sigmas=0.078, seed=13)
    potential = abtem.Potential(phonons, sampling=0.1 if SMALL else 0.05,
                                slice_thickness=1.95)
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=25)
    probe.grid.match(potential)
    detector = abtem.AnnularDetector(inner=78, outer=110 if SMALL else 200)
    scan = abtem.GridScan(start=(0, 0), end=(0.25, 0.25),
                          gpts=(2, 2) if SMALL else (8, 8),
                          fractional=True, potential=potential)

    def run():
        return probe.scan(potential, scan=scan, detectors=detector).compute(
            progress_bar=False)

    result, times = time_repeats(run, repeats)
    report("haadf frozen-phonons", f"{potential.gpts}", times, result)


# ---------------------------------------------------------------------------
# Core-loss cases (dtype fix => checksums differ ~1e-6 rel across branches)
# ---------------------------------------------------------------------------

def eels_setup(reps=None, gpts=None):
    if reps is None:
        reps = (3, 2, 1) if SMALL else (6, 4, 2)
    if gpts is None:
        gpts = (96, 96) if SMALL else (256, 256)
    from abtem.inelastic.core_loss import SubshellTransitions

    atoms = bn_slab(reps)
    potential = abtem.Potential(atoms, gpts=gpts, slice_thickness=1.665)
    transitions = SubshellTransitions(Z=5, n=1, l=0, xc="PBE", order=1,
                                      epsilon=10)
    tp = transitions.get_transition_potentials(energy=100e3)
    tp.grid.match(potential)
    tp = tp.build()
    probe = abtem.Probe(energy=100e3, semiangle_cutoff=25)
    probe.grid.match(potential)
    scan = abtem.GridScan(start=(0, 0), end=(0.5, 0.5),
                          gpts=(2, 2) if SMALL else (8, 8),
                          fractional=True, potential=potential)
    sites = atoms[atoms.numbers == 5]
    return potential, tp, probe, scan, sites


def case_coreloss_multislice(double_channel, repeats=3):
    potential, tp, probe, scan, sites = eels_setup()
    detector = abtem.FlexibleAnnularDetector()

    def run():
        m = probe.transition_potential_scan(
            scan=scan, potential=potential, detectors=detector,
            transition_potentials=tp, double_channel=double_channel,
            sites=sites, threshold=0.5, lazy=True,
        ).compute(progress_bar=False)
        return m.integrate_radial(inner=0, outer=60)

    result, times = time_repeats(run, repeats)
    tag = "double" if double_channel else "single"
    report(f"coreloss multislice {tag}", f"{potential.gpts}", times, result)


def case_coreloss_prism(repeats=3):
    """PRISM-EELS driver: exercises the scattered-inputs wiring in PR 3."""
    potential, tp, _, scan, sites = eels_setup()
    s_matrix = abtem.SMatrix(potential=potential, energy=100e3,
                             semiangle_cutoff=25, interpolation=1)

    def run():
        m = s_matrix.transition_potential_scan(
            transition_potentials=tp, scan=scan,
            detectors=abtem.FlexibleAnnularDetector(), sites=sites,
            double_channel=False, lazy=True,
        ).compute(progress_bar=False)
        return m.integrate_radial(inner=0, outer=60)

    result, times = time_repeats(run, repeats)
    report("coreloss prism single", f"{potential.gpts}", times, result)


CASES = [
    ("haadf annular", case_haadf_annular),
    ("haadf flexible", case_haadf_flexible),
    ("haadf phonons", case_haadf_phonons),
    ("coreloss multislice single", lambda: case_coreloss_multislice(False)),
    ("coreloss multislice double", lambda: case_coreloss_multislice(True)),
    ("coreloss prism", case_coreloss_prism),
]


def _git_rev(module_file):
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
    print(f"[{LABEL}] device={DEVICE} scale={SCALE}")
    abtem.config.set({"device": DEVICE, "precision": "float32",
                      "fftw.planning_effort": "FFTW_ESTIMATE"})
    if os.environ.get("BENCH_SCATTER") == "off":
        set_config_if_supported("dask.scatter-large-inputs", False)
    entries = os.environ.get("BENCH_FFT_ENTRIES")
    if entries is not None:
        set_config_if_supported("cupy.fft-cache-entries", int(entries))
    for name, case in CASES:
        if CASE_FILTER and not re.search(CASE_FILTER, name):
            continue
        case()
