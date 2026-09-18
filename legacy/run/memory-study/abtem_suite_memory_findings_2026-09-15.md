# abTEM suite memory: diagnosis

**Written:** 2026-09-15. Answers the question posed in `handoff_abtem_suite_memory_2026-09-15.md`: is the >20 GB accumulation benign caching, or a real leak worth reporting upstream?

**Answer: both, and the split is the useful part.** Most of the bulk is bounded first-touch cost — imports, kernel compilation, caches filling. On top of that sits a genuine unbounded leak of roughly 40 MB per repetition of identical work, confined to the GPU path, in native memory that no Python-level tool can see.

**The handover's leading hypothesis is refuted as stated.** Accumulation is not specific to machines with a GPU and GPAW. The CI-equivalent configuration — cupy and GPAW both unimportable — still accumulates 613 MB across 18 files. What the GPU changes is the rate, roughly doubling it per test, not whether it happens.

## Method

`abtem_accum_experiment.py` runs a fixed 18-file subset in **one** process under four conditions, recording resident memory after a `gc.collect()` at every module boundary (`abtem_mem_probe.py`). Two departures from the experiment as originally sketched:

**A subset, not the whole suite.** Accumulation is a slope; 18 files show it as clearly as 53, at a fraction of the peak. Running the full suite in one process is also what the discipline rule forbids, and what killed the browser.

**The floor between modules, not peak RSS.** Peak conflates one module's transient with everything retained before it. The floor after a collect is what was actually kept, and it is what separates a plateauing cache from a leak. `test_measure.py` is excluded as a 3.9 GB transient outlier that says little about retention.

Worktree: `dev` @ `468e3be`. Box: 46 GB, 24 cores, ROCm cupy, GPAW installed.

## Results

Resident MB at each module boundary, and total growth:

| | base | nocupy | nogpaw | neither |
|---|---|---|---|---|
| start | 518 | 416 | 501 | 335 |
| end | 2489 | 1119 | 2217 | 948 |
| **growth** | **+1971** | **+702** | **+1716** | **+613** |
| per test | 1974 KB | 1088 KB | 1741 KB | 969 KB |
| tests run | 1022 | 661 | 1009 | 648 |

Per-test normalisation matters: `nocupy` skips 521 tests, so its lower total is partly just less work. Normalised, cupy roughly doubles the rate rather than tripling it.

GPAW is a minor contributor (1971 → 1716, about 13 %). cupy is the dominant one (1971 → 702).

Growth concentrates in a few modules, and in two distinguishable groups:

| module | base Δ | neither Δ | reading |
|---|---|---|---|
| test_waves.py | +672 | +14 | almost purely cupy |
| test_prism.py | +323 | +57 | mostly cupy |
| test_ionization.py | +241 | +226 | not cupy at all |
| test_array.py | +237 | +103 | mixed |

No condition plateaus, and all accelerate — in `base` the back half grows +1440 MB against the front half's +530 MB. By the handover's own criterion, monotonic growth against modules processed is the defect signature, not the cache signature.

## What it is not

**Not device memory.** The cupy device pool peaks at 329 MB reserved, 17 MB in use, against 1971 MB of host growth.

**Not pinned host memory.** The pinned pool ends the run at 9 free blocks.

**Not allocator fragmentation.** With `malloc_trim(0)` at every boundary, the trimmed floor still climbs +1499 MB, and only 15 MB of the final 2032 MB comes back. Trimming throughout does help — it brings the final untrimmed floor from 2489 MB to 2032 MB, so about 457 MB (23 %) of the climb is glibc holding freed arenas — but the remaining 77 % is genuinely retained.

**Not a Python object-count explosion, and not reachable at all.** `test_waves.py` retains 672 MB while adding 4,878 objects. Run solo it goes from 494 MB at interpreter start to 1180 MB, 1036 MB after a trim — and a full walk of the object graph at session end finds **2 distinct numpy buffers totalling 0 MB**. Half a gigabyte of host memory is retained with nothing in Python referring to it. That is native allocation, consistent with cupy/HIP runtime kernel-compilation caches, which is also why CI never sees it: CI compiles zero GPU kernels.

## Cache or leak: the repeat test

Running the same module repeatedly in one process separates a compile-once cost from unbounded growth. With `--hypothesis-seed=0` so every pass draws the same examples and does identical work:

| | with cupy | cupy blocked |
|---|---|---|
| after pass 1 | 950 MB | 384 MB |
| pass 2 adds | +51 | +12 |
| pass 3 adds | +35 | +10 |
| pass 4 adds | +42 | +5 |

**The shape is the result.** With cupy blocked, growth decays toward zero — a bounded cache filling up. With cupy, it does not decay: identical work keeps costing ~40 MB a pass, indefinitely. The first attempt at this test, without a fixed seed, was confounded — `test_waves.py` is `@given(data=st.data())` throughout, so each pass drew new grids and compiled new kernels, and the growth was legitimate cache fill rather than a leak.

Caveat: the cupy-blocked arm also runs fewer tests, so the two columns are not matched on work. The decay-versus-flat shape is robust to that; the absolute sizes are not.

## Verdict

**Do not file this as "abTEM leaks memory".** The dominant term is native memory in the GPU stack, and it is not yet established whether abTEM causes it (compiling fresh kernels per call instead of reusing module-level ones would produce exactly this) or whether it is inherent to cupy on ROCm. Filing against abTEM without that distinction would be a bug report the maintainers cannot act on.

**The remaining step is attribution**, and it is a different kind of work from everything above: instrument kernel creation to count distinct compiled kernels across passes. If the count grows while the work repeats, it is a caller bug and belongs upstream in abTEM. If the count is flat while memory grows, it is cupy/HIP and belongs there.

**The CPU-side component is a bounded cache** and needs nothing beyond a note. It is ~969 KB/test, it decays on repetition, and CI absorbs it.

**Practically, nothing changes for this box.** `abtem_safe_suite.py`'s per-file isolation already handles it, and the handover's conclusion — that a bespoke runner should not be pushed upstream for a problem CI does not have — stands.

## Separate defect found on the way

`abtem_issues/gpu_tests_run_without_a_device.md` — the suite gates `device="gpu"` tests on whether cupy *imports*, not on whether a device exists, so on a machine with cupy and no usable GPU they run and crash instead of skipping. One-line fix; the correct predicate already exists in the same file.

## Artefacts

- `abtem_mem_probe.py` — boundary probe: RSS, optional `malloc_trim` floor, cupy pools, object counts; `MEM_PROBE_BLOCK` makes a module unimportable
- `abtem_retention_probe.py` — object-graph attribution at session end
- `abtem_accum_experiment.py` — the four-condition driver, with an RSS cap and a system-memory floor
- `abtem_accum_results/` — the JSONL curves behind every number above
