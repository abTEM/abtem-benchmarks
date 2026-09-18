"""Multi-GPU correctness check for the transition-potential copy_to_device
memo (PR "fix-transition-potential-device-memo", abtem_issues/
transition_potential_uploaded_to_gpu_per_task.md).

    salloc --nodes 1 --ntasks 1 --cpus-per-task=64 --gpus 2 \
        --qos shared_interactive --time 00:30:00 --constraint gpu --account m5395
    bash gpu_4gpu_node.sbatch bench_device_memo_multigpu.py

Needs the fix-transition-potential-device-memo branch checked out at
/opt/src/abTEM and at least 2 GPUs visible in the allocation.

bench_device_memo.py never varies which GPU is current: cp.cuda.Device()
defaults to 0 and nothing in that script changes it, so it exercises
exactly one device no matter how many the job requested (confirmed on
Perlmutter: 1-GPU and 2-GPU allocations produced near-identical output).
_device_cache_key's whole reason to exist -- see its docstring in
abtem/integrals.py -- is staying correct when one process drives several
GPUs, which is untested until a run actually switches the ambient device
context between calls. This script does that explicitly with
`with cp.cuda.Device(i):`, the same pattern abTEM's own multi-GPU dask
workers use.

Checks, across devices 0 and 1 sharing one graph node:
  1. the memoized array actually lands on the device named by the ambient
     context at call time, not always device 0
  2. the two devices get two distinct _device_array_cache entries, not one
     aliased slot
  3. values round-tripped from each device agree with the host source
  4. a repeat call under the same device context reuses the cached array
     object (identity, not merely equal values) instead of re-uploading

Then times a small/medium/large round-robin workload split across both
devices, memoized against not, warmed up per device first (this project's
own "warm up before timing" convention -- context/cuFFT/cuBLAS setup is
per-device, so warming up device 0 does not warm up device 1).
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

N_GPUS = cp.cuda.runtime.getDeviceCount()
if N_GPUS < 2:
    raise SystemExit(
        f"only {N_GPUS} GPU(s) visible -- request at least 2 (e.g. --gpus 2) "
        "to exercise more than one device."
    )


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
    cp.cuda.Device(0).synchronize()
    cp.cuda.Device(1).synchronize()
    t0 = time.perf_counter()
    for i in range(n_tasks):
        with cp.cuda.Device(i % 2):
            view = tp._task_local()
            view.copy_to_device("gpu")
    cp.cuda.Device(0).synchronize()
    cp.cuda.Device(1).synchronize()
    return time.perf_counter() - t0


def time_unmemoized(tp, n_tasks):
    cp.cuda.Device(0).synchronize()
    cp.cuda.Device(1).synchronize()
    t0 = time.perf_counter()
    for i in range(n_tasks):
        with cp.cuda.Device(i % 2):
            view = tp._task_local()
            ArrayObject.copy_to_device(view, "gpu")
    cp.cuda.Device(0).synchronize()
    cp.cuda.Device(1).synchronize()
    return time.perf_counter() - t0


if __name__ == "__main__":
    print(f"visible GPUs: {N_GPUS}")

    # --- Correctness: does the memo actually stay correct across devices? ---
    tp = make_tp()
    host_reference = tp.array.copy()

    results = {}
    for dev_id in (0, 1):
        with cp.cuda.Device(dev_id):
            view = tp._task_local()
            result = view.copy_to_device("gpu")
        results[dev_id] = result
        landed_on = int(result.array.device.id)
        print(f"device {dev_id}: array landed on device {landed_on} "
              f"[{'OK' if landed_on == dev_id else 'WRONG DEVICE'}]")
        assert landed_on == dev_id, (
            f"copy_to_device under `with cp.cuda.Device({dev_id})` put the "
            f"array on device {landed_on} instead"
        )

    cache_keys = sorted(tp._device_array_cache.keys())
    print(f"_device_array_cache keys: {cache_keys}")
    assert cache_keys == [("gpu", 0), ("gpu", 1)], (
        f"expected one cache entry per device, got {cache_keys} -- the two "
        "devices are aliasing onto the same cache slot"
    )

    for dev_id, result in results.items():
        with cp.cuda.Device(dev_id):
            back = cp.asnumpy(result.array)
        match = bool(np.allclose(back, host_reference))
        print(f"device {dev_id}: round-tripped value matches host source: {match}")
        assert match, f"device {dev_id}'s array does not match the source data"

    for dev_id in (0, 1):
        with cp.cuda.Device(dev_id):
            view = tp._task_local()
            second = view.copy_to_device("gpu")
        reused = second.array is results[dev_id].array
        print(f"device {dev_id}: repeat call reuses the cached array object: {reused}")
        assert reused, f"device {dev_id}'s second call re-uploaded instead of hitting the cache"

    print("PASS: distinct devices, distinct cache entries, no cross-device "
          "aliasing, memo holds independently per device\n")

    # --- Timing: round-robin small/medium/large across both devices ---
    # Warm up BOTH paths on BOTH devices, directly -- NOT by calling
    # time_memoized/time_unmemoized with n_tasks=1, since their internal
    # `i % 2` round-robin always resolves to device 0 for a single-task
    # call regardless of any outer `with cp.cuda.Device(...)` (a first
    # attempt at this warm-up made exactly that mistake: it silently only
    # ever touched device 0, leaving device 1 cold on both paths). The
    # memoized path never runs an FFT (a cache hit just copies
    # already-computed arrays); the unmemoized path rebuilds through
    # __init__, which recomputes _local_potential via ifft2 -- its own
    # per-device cuFFT plan.
    for dev_id in (0, 1):
        with cp.cuda.Device(dev_id):
            make_tp()._task_local().copy_to_device("gpu")
            ArrayObject.copy_to_device(make_tp()._task_local(), "gpu")
    print("(per-device, per-path warm-up done)\n")

    # Per-task timing for the memoized path, first miss then hit on each
    # device -- the small/medium/large batch below measures the same
    # memoized path but its single end-of-batch synchronize (rather than
    # per-task) reads anomalously high specifically at n_tasks=4 in a way
    # that does not reproduce at n_tasks=64/256 and does not match this
    # per-task breakdown; treat that batch number as unreliable at n=4 and
    # use this one instead, which is stable across repeated runs.
    tp_costs = make_tp()
    for i in range(4):
        with cp.cuda.Device(i % 2):
            cp.cuda.Device(i % 2).synchronize()
            t0 = time.perf_counter()
            view = tp_costs._task_local()
            view.copy_to_device("gpu")
            cp.cuda.Device(i % 2).synchronize()
            kind = "miss" if i < 2 else "hit"
            print(f"  device {i % 2} cache {kind}: {(time.perf_counter() - t0) * 1e3:.2f} ms")
    print()

    for label, n_tasks in [("small", 4), ("medium", 64), ("large", 256)]:
        t_fixed = time_memoized(make_tp(), n_tasks)
        t_unfixed = time_unmemoized(make_tp(), n_tasks)
        print(
            f"{label:6s} n_tasks={n_tasks:4d} (round-robin over 2 devices)  "
            f"memoized={t_fixed*1e3:8.2f} ms  "
            f"unmemoized={t_unfixed*1e3:8.2f} ms  "
            f"speedup={t_unfixed / t_fixed:6.1f}x"
        )
