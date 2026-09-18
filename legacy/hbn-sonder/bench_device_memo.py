"""GPU wall-clock for the transition-potential copy_to_device memo (PR
"fix-transition-potential-device-memo", abtem_issues/
transition_potential_uploaded_to_gpu_per_task.md).

    sbatch gpu_1gpu_shared.sbatch bench_device_memo.py

Needs the fix-transition-potential-device-memo branch checked out at
/opt/src/abTEM (see gpu_1gpu_shared.sbatch) and a GPU -- run it there, not
against plain dev, which has no memo to measure and no _device_array_cache
attribute to check for.

Every test written for this fix so far ran on CPU, where copy_to_device is a
no-op (get_array_module(array) is get_array_module(device) short-circuits
before any transfer) -- so the actual host-to-device PCIe cost the fix
avoids repeating has never been measured. This times it directly: small,
medium and large simulated task counts against one shared transition
potential, memoized against not.

Warms up CUDA context / cuFFT plan / cuBLAS handle creation with a throwaway
call before any timed measurement -- this project's own "Measuring"
convention: an elastic-vs-core-loss comparison once read 453% cold, 35%
warm. Without it, whichever size runs first (small, here) absorbs that
one-time cost on both its memoized and unmemoized timings, which is
indistinguishable from a real per-task cost until compared against a
warmed-up second and third size.
"""
import time

import cupy as cp
import numpy as np

import abtem
from abtem.array import ArrayObject
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import TransitionPotentialArray

_probe = TransitionPotentialArray(
    Z=1, array=np.zeros((1, 2, 2), dtype=np.complex64), energy=1e3, extent=1.0,
    ensemble_axes_metadata=[OrdinalAxis(values=(0,))], metadata={},
)
if not hasattr(_probe, "_device_array_cache"):
    raise SystemExit(
        f"this abTEM ({abtem.__version__}, {abtem.__file__}) has no "
        "TransitionPotentialArray._device_array_cache -- check out "
        "fix-transition-potential-device-memo at /opt/src/abTEM, not dev."
    )
del _probe


def make_tp(mb_payload=21):
    # ~21 MB matches the payload size quoted in the issue file.
    n_pix = int((mb_payload * 1024 * 1024 / 8) ** 0.5)  # complex64, 2 transitions
    rng = np.random.default_rng(0)
    arr = (
        rng.standard_normal((2, n_pix, n_pix))
        + 1j * rng.standard_normal((2, n_pix, n_pix))
    ).astype(np.complex64)
    return TransitionPotentialArray(
        Z=14, array=arr, energy=100e3, extent=(8.0, 8.0),
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )


def time_memoized(tp, n_tasks):
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_tasks):
        view = tp._task_local()
        view.copy_to_device("gpu")
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - t0


def time_unmemoized(tp, n_tasks):
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_tasks):
        view = tp._task_local()
        ArrayObject.copy_to_device(view, "gpu")
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - t0


if __name__ == "__main__":
    print(f"cupy device: {cp.cuda.Device().id}, "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name']}")

    # Warm-up: a throwaway call on its own tiny payload, discarded, so the
    # first *measured* call doesn't pay for CUDA context / cuFFT plan /
    # cuBLAS handle creation that has nothing to do with this fix.
    time_memoized(make_tp(), 1)
    print("(warm-up done)")

    for label, n_tasks in [("small", 4), ("medium", 64), ("large", 256)]:
        t_fixed = time_memoized(make_tp(), n_tasks)
        t_unfixed = time_unmemoized(make_tp(), n_tasks)
        print(
            f"{label:6s} n_tasks={n_tasks:4d}  "
            f"memoized={t_fixed*1e3:8.2f} ms  "
            f"unmemoized={t_unfixed*1e3:8.2f} ms  "
            f"speedup={t_unfixed / t_fixed:6.1f}x"
        )
