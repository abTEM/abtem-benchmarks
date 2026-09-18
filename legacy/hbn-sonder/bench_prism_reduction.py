"""GPU wall-clock and peak memory for the PRISM-EELS reduction chunking
(PR "fix-prism-eels-reduction-chunking", abtem_issues/
prism_eels_reduction_allocates_whole_scan.md).

    sbatch gpu_1gpu_shared.sbatch bench_prism_reduction.py

Needs the fix-prism-eels-reduction-chunking branch checked out at
/opt/src/abTEM (see gpu_1gpu_shared.sbatch) and a GPU.

Small and medium fit in memory unchunked too, so both a forced single
whole-scan block and the real auto-sized chunking are timed and compared --
that overhead (more, smaller GPU ops instead of one big one) is the cost of
the fix, worth knowing on top of the memory win. Large is the regime the
unfixed code cannot run at all -- that is the entire point of the fix -- so
there is no baseline to compare against there; only peak memory and
completion are reported, read from a live device query rather than an
intrusive probe (never run code on the thing being measured).

Warms up CUDA context / cuFFT plan / cuBLAS handle creation with a throwaway
call before any timed measurement -- first-call costs dominate and will
invert the comparison otherwise (this project's own "Measuring" convention:
an elastic-vs-core-loss comparison once read 453% cold, 35% warm). The
number of row-batches the auto-sizer actually chose is reported alongside
each timing, from a live minimum_crop call count -- this is the direct
answer to "does small skip chunking and large use it", not an inference
from timing alone.
"""
import time

import ase
import cupy as cp
import numpy as np

import abtem
import abtem.inelastic.core_loss as cl
import abtem.prism.utils as prism_utils
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import TransitionPotentialArray
from abtem.prism.utils import minimum_crop as _real_minimum_crop

if "estimate_scan_batch_size" not in dir(cl):
    raise SystemExit(
        f"this abTEM ({abtem.__version__}, {abtem.__file__}) has no "
        "estimate_scan_batch_size-based chunking in core_loss.py -- check "
        "out fix-prism-eels-reduction-chunking at /opt/src/abTEM, not dev."
    )

abtem.config.set({"device": "gpu"})


def make_tp(n=4):
    rng = np.random.default_rng(0)
    arr = (
        rng.standard_normal((n, 128, 128)) + 1j * rng.standard_normal((n, 128, 128))
    ).astype(np.complex64)
    return TransitionPotentialArray(
        Z=14, array=arr, energy=100e3, extent=(8.0, 8.0),
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n)))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )


def run(scan_gpts, interpolation, force_single_block=False, count_batches=False):
    atoms = ase.Atoms(
        "Si2", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 3.0)], cell=(8, 8, 8), pbc=True
    )
    potential = abtem.Potential(atoms, gpts=(128, 128), slice_thickness=2.0, exit_planes=1)
    s_matrix = abtem.SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=interpolation
    )
    scan = abtem.GridScan(
        start=(0, 0), end=scan_gpts, gpts=scan_gpts, fractional=False, potential=potential
    )
    tp = make_tp()

    orig_batch_size = cl.estimate_scan_batch_size
    if force_single_block:
        cl.estimate_scan_batch_size = lambda *a, **k: 10**9

    call_sizes = []
    if count_batches:
        orig_crop = prism_utils.minimum_crop

        def _recording(positions, shape, _sizes=call_sizes):
            _sizes.append(int(positions.shape[0]))
            return _real_minimum_crop(positions, shape)

        prism_utils.minimum_crop = _recording

    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    m = s_matrix.transition_potential_scan(
        transition_potentials=tp, scan=scan,
        detectors=abtem.FlexibleAnnularDetector(), sites=atoms,
        double_channel=False, lazy=False,
    )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = cp.get_default_memory_pool().total_bytes() / 1024**3

    if count_batches:
        prism_utils.minimum_crop = orig_crop
    if force_single_block:
        cl.estimate_scan_batch_size = orig_batch_size

    # The sizing pass (a shrinking sequence of candidate sizes) always
    # finishes before any site's real reduction call, and every one of
    # those real calls uses the same, final batch size -- so the last
    # recorded call size IS that final size, however many sizing probes
    # came before it. n_positions / that size is the row-batch count
    # (ceil, in case the scan's row count doesn't divide evenly).
    n_positions = int(np.prod(scan_gpts))
    batch_size = call_sizes[-1] if call_sizes else None
    n_row_batches = -(-n_positions // batch_size) if batch_size else None
    return (
        elapsed, peak_gb, np.asarray(abtem.core.backend.asnumpy(m.array)),
        n_row_batches,
    )


if __name__ == "__main__":
    print(f"cupy device: {cp.cuda.Device().id}")

    # Warm-up: a throwaway call on its own tiny geometry, discarded, so the
    # first *measured* call doesn't pay for CUDA context / cuFFT plan /
    # cuBLAS handle creation that has nothing to do with this fix.
    run((2, 2), 1)
    cp.get_default_memory_pool().free_all_blocks()
    print("(warm-up done)")

    for label, scan_gpts, interpolation in [
        ("small", (8, 8), 1),
        ("medium", (32, 32), 1),
    ]:
        t_chunked, mem_chunked, a_chunked, n_batches = run(
            scan_gpts, interpolation, count_batches=True
        )
        t_single, mem_single, a_single, _ = run(scan_gpts, interpolation, force_single_block=True)
        match = np.allclose(a_chunked, a_single, rtol=1e-5, atol=np.abs(a_single).max() * 1e-6)
        print(
            f"{label:6s} scan={scan_gpts}  row_batches={n_batches}  "
            f"chunked={t_chunked*1e3:8.2f} ms ({mem_chunked:.3f} GB peak)  "
            f"single-block={t_single*1e3:8.2f} ms ({mem_single:.3f} GB peak)  "
            f"ratio={t_chunked / t_single:5.2f}x  match={match}"
        )

    # Large: the OOM regime the unfixed code cannot run at all. Scaled up
    # toward (but below, to keep this a quick interactive check rather than
    # the full production job) the failing geometry in
    # abtem_issues/prism_eels_reduction_allocates_whole_scan.md.
    label, scan_gpts, interpolation = "large", (64, 64), 1
    t_chunked, mem_chunked, _, n_batches = run(scan_gpts, interpolation, count_batches=True)
    print(
        f"{label:6s} scan={scan_gpts}  row_batches={n_batches}  "
        f"chunked={t_chunked*1e3:8.2f} ms "
        f"({mem_chunked:.3f} GB peak) -- no unfixed baseline: that regime "
        "OOMs by design (see the issue file)"
    )
