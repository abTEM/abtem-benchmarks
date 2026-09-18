"""Probe cuFFT plan memsizes: plain process vs dask-cuda worker, by layout.

Chases the observed difference where the serial abTEM client builds an 11 GB
plan for the (4, 4, 2623, 2271) complex128 ifft2 while distributed workers
resolve the same nominal transform under the 10.7 GB cache bound.

Measures, in BOTH contexts (this process, and inside a LocalCUDACluster
worker), the cached plan memsize for:
  A. flat contiguous      (16, 2623, 2271)
  B. 4D contiguous        (4, 4, 2623, 2271)
  C. 4D non-contiguous    (a strided view of B)

Same memsizes in both contexts -> planning is layout-driven, not
context-driven -> next step is tracing which layouts each mode executes.
Different memsizes for identical layouts -> context-dependent planning ->
minimal repro + CuPy upstream issue.

Requires >= 1 GPU and dask-cuda. Run inside the container with the usual
mounts; needs no abTEM.
"""
import json


def measure():
    import cupy as cp

    results = {}
    cache = cp.fft.config.get_plan_cache()
    for name, make in [
        ("flat-contig (16,2623,2271)",
         lambda: cp.zeros((16, 2623, 2271), dtype="complex128")),
        ("4d-contig (4,4,2623,2271)",
         lambda: cp.zeros((4, 4, 2623, 2271), dtype="complex128")),
        ("4d-noncontig (view)",
         lambda: cp.zeros((4, 8, 2623, 2271), dtype="complex128")[:, ::2]),
    ]:
        cache.clear()
        cache.set_memsize(-1)
        a = make()
        cp.fft.ifft2(a)
        cp.cuda.Stream.null.synchronize()
        plans = []
        # show_info prints; walk the cache dict-style via its iterator instead
        node = None
        try:
            for key in list(getattr(cache, "__iter__", lambda: [])()):
                plans.append(int(cache[key].work_area.mem.size))
        except Exception:
            pass
        if not plans:
            # Fallback: total memsize of the cache = sum of cached plans.
            plans = [int(cache.get_curr_memsize())]
        results[name] = {
            "n_plans": len(plans),
            "memsizes_GB": [round(p / 1e9, 2) for p in sorted(plans, reverse=True)],
            "contiguous": bool(a.flags.c_contiguous),
        }
        del a
        cp.get_default_memory_pool().free_all_blocks()
    cache.clear()
    return results


def main():
    print("== plain process ==")
    print(json.dumps(measure(), indent=1))

    from dask_cuda import LocalCUDACluster
    from distributed import Client

    with Client(LocalCUDACluster(n_workers=1)) as client:
        print("== inside dask-cuda worker ==")
        worker_results = client.run(measure)
        for addr, res in worker_results.items():
            print(addr)
            print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
