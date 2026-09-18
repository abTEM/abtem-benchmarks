# abTEM regression benchmark suite: design (v2)

Status: draft for review, 2026-09-18. Supersedes `DESIGN-v1-2026-09-10.md` (kept for the record). Revised after Toma Susi's reply of 2026-09-17 on [abTEM discussion #380](https://github.com/abTEM/abTEM/discussions/380) and the lessons recorded from earlier benchmark work (see `../legacy/PROVENANCE.md`). Milestones: `M0-consolidation.md` to `M4-ci-and-retirement.md`.

Repository split: the harness (`benchmarks/abtem_bench`) and the case definitions (`benchmarks/cases`) live in `abTEM/abTEM`; reference bundles, this design, and the frozen legacy scripts live in `abTEM/abtem-benchmarks`.

## 1. Purpose

One command runs a fixed matrix of user-facing abTEM workloads against a candidate checkout and reports, per workload, device and size tier: whether the results changed relative to a reference, how wall time changed, and how peak memory changed.

Three uses, in order of urgency:

1. Release notes for v1.1: which workloads differ numerically from v1.0.10, by how much, and which of those differences were intended (the exact Fresnel propagator of #298 will move nearly every case).
2. Pull-request gate: did this change alter results, speed or memory relative to its merge base, on identical case code and identical environment.
3. Internal consistency: do equivalent code paths agree (CPU vs GPU, lazy vs eager, PRISM vs multislice, one GPU vs the multi-GPU path with one worker, float32 vs its own float64), and did any gap widen.

"Result change" means: arrays differ beyond the case's stated tolerance, or shape, dtype, axes sampling, offset or units changed. A flagged change is not automatically a bug; it must be accepted deliberately, and the acceptance record is the changelog entry.

Non-goals: validation of the physics against other codes or experiment; kernel micro-benchmarks; absolute performance claims across machines; a public leaderboard.

## 2. What changed relative to v1 (2026-09-10)

| topic | v1 | v2 | driver |
|---|---|---|---|
| ordering | live A/B first, references later | reference mode first (capture at v1.0.10, diff dev), live A/B second, one compare layer | Toma, v1.1 release notes |
| accuracy precision | float32, rtol 1e-5 | float64, bit-exact expectation on CPU; float32 checked against its own float64 | Toma: 1e-5 sits on the float32 reassociation noise floor |
| tolerance | one rtol/atol pair | a per-output vector: identical flag, max abs / max ref, relative error above 1e-6 of max, integrated intensity, shape/axes/dtype equality | Toma; #261 would have been caught by integrated intensity |
| drift policy | report or block | fail, cleared by `accepted_changes.toml` in the same PR | Toma |
| reference data | none in git, "attach to release" | release assets on abtem-benchmarks, fetched and cached; machine-fingerprint policy | Toma; Paul 2026-09-18 |
| propagator | not addressed | `[order1]` variant on the candidate isolates #298 from all other changes | verified: `algorithm=FourierMultislice(order=1)` accepted on v1.0.10 and dev |
| GPAW in CI | GPAW-tagged cases skipped | no built-in transition potentials exist in abTEM; CI uses a seeded synthetic `TransitionPotentialArray`, local tiers use a precomputed real O K asset | verified 2026-09-18 |
| case identity | none | sha256 of case sources in every manifest; compare refuses mismatches | Toma |
| peak RSS | sampler thread + `ru_maxrss` | `os.wait4` rusage of the case subprocess; sampler optional and off by default | memory study 2026-09-15 |
| memory repeats | N repeats in one process | one shot per process, N processes, median | GPU path leaks ~40 MB per identical repeat |
| cache-keying bugs | not covered | `consistency.cache_reuse` runs two grids and two elements in one process | issue history (integral-table and scattering-factor caches keyed on symbol only) |
| graph transport | not covered | optional meter: task count and pickled graph bytes for lazy cases | PR #386 lesson |
| config | pin a list | pin the list and dump the fully resolved `abtem.config.config`, dask config and thread env into the manifest | Toma |
| legacy scripts | delete after porting | keep in `abtem-benchmarks/legacy` with provenance; delete from abTEM per reproduced case | Toma |
| comparison helpers | own implementation | `array_is_close` promoted to `abtem/core/testing.py` and reused; `close_stats()` added beside it | Toma |

## 3. Toma's points and where they are addressed

| point (2026-09-17) | section |
|---|---|
| reference-array mode first, A/B second, shared compare | 4.1, M1, M2 |
| float64 bit-exact accuracy tier; float32 tracks its own float64 | 6.1, 6.2 |
| drift fails, cleared by `accepted_changes` in the same PR | 8.3 |
| chunked potential building as its own case, pinned chunk size | 6.1, 11 (`potential.chunked`) |
| CrystalPotential with frozen phonons (pool and tiling) | 11 (`potential.crystal`) |
| single GPU vs `dask.multi-gpu` with one worker | 11 (`gpu.multigpu_one_worker`) |
| energy ensembles and focal spread | 11 (`energy.ensemble`, `hrtem.focal_spread`) |
| non-orthogonal grids after #282 | 11 (deferred; #282 is open) |
| multi-detector scan prominent | 11 (`stem.multidetector`, M1) |
| GPAW not on hosted runners | 11 (`coreloss.*`), 13 |
| tolerance vector; reuse `array_is_close`; integrated intensity | 8.1, 8.2 |
| phonon seeds consumed identically | 11 (`phonons.haadf` emits `displacements`) |
| harness in main repo, references elsewhere | 12, 10 |
| `python -m abtem_bench`, ruff, mypy | 12 |
| case-definition hash in every result | 4.3, 7 |
| do not delete legacy until reproduced; mine `benchmark_potential_chunking.py` | `../legacy/PROVENANCE.md`, M2, M4 |
| CI label-triggered, required only after a month of zero false positives | 13 |
| pin and record resolved config | 6.1, 7 |

## 4. Architecture

### 4.1 Modes

Reference mode. `abtem-bench capture --ref v1.0.10 --tier quick --device cpu --preset accuracy --out <bundle-dir>` runs every case under the reference checkout and writes a bundle. `abtem-bench compare <bundle-A> <bundle-B>` diffs two bundles; a bundle may be fetched by tag (`--ref-bundle v1.0.10`) from the release assets on abtem-benchmarks.

Live A/B mode. `abtem-bench run --ref A --ref B ...` captures both bundles in one invocation, interleaving refs per case (A case 1, B case 1, A case 2, ...) so that thermal and background drift hits both refs alike, then calls the same compare.

Self-check. `abtem-bench run --self-check --ref X` captures X twice in interleaved order. Under the accuracy preset the two CPU bundles must be bit-identical; the speed and memory spreads become the per-machine noise floor that compare uses for its flags. The self-check is also the false-positive monitor for CI.

The three modes share the registry, worker, meters, store and compare. Only the runner's ref list differs.

### 4.2 Layers

- `cases/`: declarative case definitions, pure abtem API calls plus parameter tables. No timing, no measurement.
- `abtem_bench/registry.py`: `@case` decorator, `Case`, `Tier`, `Variant`, `Tolerance`, case-id rules, case hash.
- `abtem_bench/presets.py`: determinism presets applied inside the worker before `import abtem` side effects matter (config keys, dask config, env, seeds).
- `abtem_bench/worker.py`: runs exactly one case id in a fresh process: apply preset, build (untimed), cold run, warm repeats, collect outputs, write the case record.
- `abtem_bench/meters.py`: wall clock with GPU synchronisation, process CPU time, VRAM sampler (pool and driver), optional RSS sampler, optional graph meter.
- `abtem_bench/runner.py`: worktrees, environment assembly, subprocess launch, `os.wait4` peak RSS, timeouts, stderr classification, interleaving, rounds.
- `abtem_bench/store.py`: schema, manifest, case records, `.npz` outputs, machine fingerprint, config dump, bundle read/write.
- `abtem_bench/compare.py`: pairing, metric vector, verdicts, `accepted_changes`, noise-floor thresholds, Markdown and JSON reports, exit code.
- `abtem_bench/refs.py`: bundle naming, upload and fetch via `gh release`, local cache under `~/.cache/abtem-bench/refs`.
- `abtem_bench/fixtures.py`: structures (Si, SrTiO3, graphene, hBN), synthetic transition potentials, helpers that must work on every ref.
- `abtem_bench/cli.py`: `list`, `capture`, `run`, `compare`, `refs`, `self-check`.

### 4.3 Invariants

1. Harness and cases always come from the invoking checkout; only the `abtem` package is swapped per ref. Both refs execute byte-identical case code.
2. Every manifest records `case_hash` = sha256 over the sorted contents of `cases/*.py` plus `abtem_bench/__version__`. Compare refuses to pair bundles with different hashes unless `--allow-case-mismatch` is passed, and then marks the report.
3. A case whose API is missing on a ref declares `requires=<callable>`; the worker records `UNSUPPORTED` for that ref and the run continues.
4. Compare never imports the ref's abtem. It reads the commit-independent store format only. It may import the invoking checkout's `abtem.core.testing` for `array_is_close`.
5. One case id runs in one fresh subprocess. Nothing measured in one case can leak into the next (VRAM, FFTW wisdom, numba compilation, the GPU-path native-memory leak).
6. The instrument never runs code on the thing it measures: no `client.run` on dask workers; distributed memory is read from `client.scheduler_info()["workers"][addr]["metrics"]["memory"]`; missing metrics are reported as `n/a`.
7. The worker prints `ref`, `sha`, `abtem.__file__` and `abtem.__version__` as its first lines and records them; the label passed on the command line is never trusted alone.
8. Store schema changes are additive only and versioned; compare reads every earlier schema.

### 4.4 Swapping refs

abtem is pure Python (numba and CuPy kernels compile at runtime), so refs share one environment when their dependency pins resolve there. The runner creates `git worktree add .worktrees/bench/<sha> <ref>` in the abTEM checkout and launches `python -P -m abtem_bench.worker ...` with `PYTHONPATH=<invoking>/benchmarks:<worktree>`. The `-P` flag is mandatory: without it a cwd inside a checkout shadows `PYTHONPATH` (verified 2026-09-10). The worker asserts that `abtem.__file__` starts with the worktree path and aborts otherwise.

v1.0.10 is the tip of `main` and carries the same numpy >= 2, zarr >= 3.1 and dask pins as dev, so the v1.1 reference capture runs in shared mode. Tags v1.0.9 and older have unpinned dependencies and get isolated mode (M3): `uv venv` per ref plus `uv pip install -e <worktree> --constraint <freeze of the invoking environment>` so only abtem differs. The shared venv is never modified; a plain `uv pip install abtem==<tag>` would replace the editable install and is forbidden.

### 4.5 Process model and failure handling

Per case id and ref the runner spawns one worker with `cwd` set to the bundle directory (never a checkout). It waits with `os.wait4` and stores `ru_maxrss` as the authoritative peak RSS of that case. Timeouts per tier (quick 120 s, standard 900 s, large none by default) kill the process group and record `TIMEOUT`. Stderr is classified (`CUDA_ERROR_OUT_OF_MEMORY` and `CUDA_ERROR_ILLEGAL_ADDRESS` as `OOM`, cgroup kills as `OOM`, other CUDA codes as `ERROR` with the code) using the logic ported from `legacy/abtem-repo/benchmark_potential_chunking.py`. `--rounds N` repeats the whole interleaved matrix to expose drift over time.

## 5. Case contract

```python
from abtem_bench.registry import case, Tier, Variant, Tolerance, Output

@case(
    "stem.multidetector",
    tags={"stem", "multislice", "detectors", "v1.1"},
    tiers={
        "quick":    Tier(gpts=(256, 256),   reps=(4, 4, 8),    scan=(6, 6),   max_batch=8,  chunk=4, timeout=120),
        "standard": Tier(gpts=(1024, 1024), reps=(8, 8, 15),   scan=(16, 16), max_batch=8,  chunk=8, timeout=900),
        "large":    Tier(gpts=(4096, 4096), reps=(20, 20, 75), scan=(8, 8),   max_batch=8,  chunk=8, devices=("gpu",)),
    },
    variants={
        "order1":  Variant(algorithm_order=1, compare_as="default"),   # isolates #298
        "eager":   Variant(lazy=False),
        "auto":    Variant(max_batch="auto", chunk="auto", flag=False), # reported, never flagged
    },
    outputs=("bf", "adf", "segmented"),
    tolerance=Tolerance(rel=1e-10, above_rel=1e-6, intensity=1e-12),   # float64 accuracy preset
    consistency=[("default", "eager"), ("cpu", "gpu")],
    requires=lambda abtem: hasattr(abtem, "SegmentedDetector"),
    nominal_bytes=lambda p: p.gpts[0] * p.gpts[1] * (4 * p.n_slices + 16 * p.max_batch),
    warmup="cold_and_warm",
)
def stem_multidetector(p, device):
    from abtem import AnnularDetector, GridScan, Potential, Probe, SegmentedDetector
    from abtem.multislice import FourierMultislice
    from abtem_bench.fixtures import silicon
    potential = Potential(silicon(p.reps), gpts=p.gpts, slice_thickness=2.0, device=device)
    probe = Probe(energy=200e3, semiangle_cutoff=20, device=device)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=p.scan)
    a = 0.95 * min(probe.cutoff_angles)
    detectors = [AnnularDetector(0, 0.2 * a), AnnularDetector(0.5 * a, 0.9 * a),
                 SegmentedDetector(0.2 * a, 0.5 * a, nbins_radial=2, nbins_azimuthal=4)]
    def run():
        return probe.scan(potential, scan=scan, detectors=detectors, lazy=p.lazy, max_batch=p.max_batch,
                          potential_chunk_size=p.chunk, algorithm=FourierMultislice(order=p.algorithm_order)).compute()
    return run
```

Rules for case authors, enforced by the registry where possible:

- Setup happens in the function body (untimed); `run()` is a zero-argument callable and is the only timed region. `run()` returns one `ArrayObject`, a list, or a dict of them; `outputs` names them.
- Explicit `gpts` always; `sampling=` is rejected by a registry check, because `grid.round-to-fast-fft` changes derived grids between commits. Explicit `max_batch` and `potential_chunk_size` except in `auto` variants.
- Every case declares the three tiers; a tier may restrict `devices`. Tier names are `quick`, `standard`, `large`; the middle tier is never called "production" or "default".
- A variant with `compare_as="default"` is paired against the other bundle's default variant (this is how `hrtem.exitwave[order1]@dev` is compared to `hrtem.exitwave@v1.0.10`).
- `warmup` is a per-case decision, stated in the case: `cold_and_warm` (default; cold recorded, verdict on warm), `cold_only` (one-shot workloads such as potential builds, where the cold cost is what users pay), or `warm_only`.
- Stochastic cases fix `seed` in the case and emit the consumed random state as an output (frozen phonons emit `displacements`), so a refactor of the sampling order shows up as a result change rather than as physics.
- Outputs are moved to host (`.to_cpu()`), kept in native dtype, and stored with `axes_metadata` serialised through `axis_to_dict`, the JSON-safe part of `metadata`, and the invariants (sum, mean, min, max, abs max). `data_origin` is dropped since it embeds the version string.

Case id: `name[variant]@tier/device`, e.g. `stem.multidetector[order1]@quick/cpu`.

## 6. Presets and determinism

### 6.1 `accuracy` (float64, the bit-exact tier)

Pinned before any abtem computation, then the fully resolved `abtem.config.config`, `dask.config.config`, the thread environment and the seeds are dumped into the manifest:

| key | value | reason |
|---|---|---|
| `precision` | `float64` | bit-exact comparison is meaningful only here |
| `fft` | `fftw` | one backend across refs |
| `fftw.planning_effort` | `FFTW_ESTIMATE` | FFTW_MEASURE selects plans by timing; plans differ per process and rounding follows (measured: 1 flake in 7 across processes at rtol 1e-5) |
| `fftw.threads` | `1` | reduction order |
| `mkl.threads` | `1` | as above |
| dask scheduler | `synchronous`, `num_workers 1` | reduction order and memory |
| `dask.lazy` | per variant, default `true` | the lazy and eager paths are both under test |
| `dask.chunk-size`, `dask.chunk-size-gpu` | `128 MB`, `512 MB` (explicit) | shipped defaults may change |
| `potential.slice-chunk-size` | integer from the tier | `auto` depends on free memory |
| `grid.round-to-fast-fft` | `false` | cases pass explicit gpts anyway |
| `cupy.fft-cache-size` | `1 GB` (explicit bytes) | `auto` is 25 % of whatever card is present |
| `cupy.fft-cache-entries` | `64` | shipped default may change |
| `diagnostics.progress_bar`, `diagnostics.task_progress` | `false` | the 0.5 s tqdm delay perturbs short runs |
| `warnings.overspecified-grid` | `false` | noise |
| env | `TQDM_DISABLE=1`, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=1`, `PYTHONHASHSEED=0` | reduction order, hashing |
| `max_batch`, `potential_chunk_size` | explicit from the tier | `auto` depends on free memory |

Expectation on CPU: bit-identical outputs for the same case on the same machine and the same ref, across processes and across days (verified by the self-check; CLAUDE.md records that FFTW_ESTIMATE plus one thread and one worker is the necessary combination). Numba kernels compiled with `fastmath=True` are deterministic on one machine but not across CPU generations, so bit identity is claimed only for matching machine fingerprints (section 10).

Expectation on GPU: reductions in cuFFT and CuPy are not order-stable across runs, and the finite-projection integrator is known to differ at ~1e-7 relative between runs. GPU float64 is therefore compared with the tolerance measured by the self-check on that machine, and the `identical` flag is reported but not required.

### 6.2 `float32-tracking`

The same cases at `precision: float32`, run beside their float64 results in the same bundle. The metric is not float32 versus the reference's float32 (the seventh digit is noise) but the distance of float32 from its own float64 result. Compare flags a case when that distance grows by more than the noise floor relative to the reference bundle's distance. This is the check that catches a float32 code path degrading (the complex128 upcast bug of #364 was the inverse failure and is caught by the `dtype` output).

### 6.3 `speed`

Shipped FFTW settings (`FFTW_MEASURE`, wisdom per process) because that is what users pay, but fixed and recorded thread counts and worker counts. `cold` (first call in the fresh process: numba compilation, FFTW planning, CuPy kernel compilation, cuFFT plans) is always recorded separately from `warm` repeats (default 5, CuPy pools freed between repeats). The speed verdict uses the warm median with the warm minimum as a robustness check; `cold` ratios are reported and flagged only for cases declaring `warmup="cold_only"` or `cold_sensitive=True`. Warm-up is generic per case, not per shape: a case comparing several shapes warms each shape.

### 6.4 `memory`

One repeat per process, N fresh processes (default 3, 1 for `large`), median of the per-process `wait4` peak. Rationale: the GPU path retains about 40 MB of native memory per identical repeat and FFTW_MEASURE trial plans produced a 2.7× run-to-run spread in peak RSS on unchanged code; a single measurement is never a regression. `auto` variants (batch and chunk sizes from free memory) are reported and never flagged.

## 7. Store format

```
<bundle>/
  manifest.json
  cases/<case-id>.json
  cases/<case-id>/<output>.npz
```

`manifest.json`: `schema_version`, `harness_version`, `case_hash`, `ref` (label, sha, `git describe`, dirty flag), `abtem_version`, `abtem_file`, `preset`, `tier`, `device`, `timestamp_utc`, `fingerprint` (python, numpy, scipy, dask, pyfftw, numba, cupy, gpaw versions; CPU model and count; GPU name and driver; OS), `config` (resolved abtem config), `dask_config`, `env` (thread and seed variables), `command`.

`cases/<id>.json`: `status` (`OK`, `UNSUPPORTED`, `ERROR`, `OOM`, `TIMEOUT`), `error`, `timings` (`setup`, `cold`, `warm` list, `median`, `min`, `cpu_time`), `memory` (`peak_rss_bytes` from `wait4`, `peak_vram_pool_bytes`, `peak_vram_device_bytes`, `nominal_bytes`), `graph` (`n_tasks`, `pickled_bytes`, optional), `outputs` (name, shape, dtype, invariants, file), `params` (resolved tier and variant parameters), `requires_result`.

`<output>.npz`: `array` (native dtype), `axes` (JSON string of `axis_to_dict` list), `metadata` (JSON string).

The format depends on numpy and json only. Compare works on any two bundles regardless of which abtem produced them.

## 8. Compare

### 8.1 Metric vector per output

For candidate `a` and reference `r`, both host numpy arrays:

- `identical`: `a.shape == r.shape and a.dtype == r.dtype and np.array_equal(a, r)` (NaN-aware).
- `max_abs_norm`: `max|a - r| / max|r|`.
- `rel_above`: `max |a - r| / |r|` over elements with `|r| > above_rel * max|r|` (default `above_rel = 1e-6`), i.e. the `check_above_rel` semantics of `array_is_close`. Complex arrays compare real and imaginary parts and also `|a|`.
- `intensity`: `(sum(a) - sum(r)) / sum(r)` for real measurements; for complex waves `sum|a|²`.
- `shape_ok`, `dtype_ok`, `axes_ok` (sampling, offset, units, labels, ordinal values compared exactly).

Implementation: `array_is_close` is promoted from `test/utils.py` into `abtem/core/testing.py` unchanged (test/utils.py re-exports it), and a sibling `close_stats(a, r, above_rel=1e-6)` returning the vector is added next to it. Compare calls `close_stats` and uses `array_is_close(rel_tol=..., check_above_rel=...)` for the pass/fail decision, always with explicit tolerances (both default to infinity and pass silently otherwise).

### 8.2 Verdicts

`IDENTICAL` (bit-identical), `OK` (within the case tolerance for this preset and device), `DRIFT` (beyond tolerance), `ACCEPTED` (drift listed in `accepted_changes.toml`), `SHAPE` (shape, dtype or axes changed), then the run statuses `UNSUPPORTED`, `ERROR`, `OOM`, `TIMEOUT`, and `ONLY-A` / `ONLY-B` for unpaired ids.

Consistency pairs are evaluated within one bundle with their own tolerance (for example CPU vs GPU float64 at the self-check floor, PRISM vs multislice at 1e-3 relative) and additionally reported as "gap widened" when the candidate's gap exceeds the reference's gap by more than the noise floor.

Speed: ratio of warm medians candidate/reference, flagged when `|ratio - 1| > max(threshold, 3 × noise floor)`; default threshold 10 %. Memory: ratio of peak RSS and of peak VRAM, default threshold 5 %, `auto` variants never flagged.

### 8.3 `accepted_changes.toml`

```toml
[[accepted]]
case = "hrtem.exitwave*"          # glob over case ids, tier and device optional
since = "v1.0.10"                 # reference the acceptance applies against
reason = "Exact Fresnel propagator is the default (#298); order-1 phase error removed."
pr = 298
```

A `DRIFT` that matches an entry becomes `ACCEPTED`; the report renders every matched entry as a changelog table (case, reason, PR, measured drift vector). Entries are validated: unknown case globs fail the run, and an entry that no longer matches a drift is reported as stale so the file cannot rot.

### 8.4 Reports

Markdown table, one row per paired case id, columns: verdict, `identical`, `max_abs_norm`, `rel_above`, `intensity`, time ratio (with the two medians), RSS ratio, VRAM ratio, status notes. Every table states what varies down the rows (case ids), across the columns (metrics) and what each cell reports relative to which reference, per the house rule. Also `compare.json` (machine-readable, includes the noise floor used) and an exit code controlled by `--fail-on drift|shape|speed:<pct>|memory:<pct>`.

## 9. Meters

- Wall clock: `time.perf_counter` around `run()`, with `cp.cuda.Stream.null.synchronize()` before stopping on GPU; `time.process_time` alongside.
- Peak RSS: `ru_maxrss` from the parent's `os.wait4` on the worker (per-child rusage). `RUSAGE_CHILDREN` is a running maximum over every child ever reaped and is never used. An in-worker `/proc/self/status` sampler exists for time-resolved curves and is off by default because sampling perturbs memory-pressured runs.
- Peak VRAM: in-worker thread at 5 ms reading `cp.get_default_memory_pool().used_bytes()` and `total - free` from `cp.cuda.Device().mem_info`. The second captures cuFFT workspace outside the pool. Both are driver or allocator queries and run no code on the computation.
- Graph transport (lazy cases, optional): number of tasks and `len(pickle.dumps(graph))` of the built graph before compute. Cheap, CPU-only, catches per-task payload copies (PR #386 shipped `scan_positions / max_batch` copies of a transition potential).
- Distributed cases: worker memory read from `client.scheduler_info()` only.
- Every meter records what it measured and its overhead estimate in the case record.

## 10. Reference bundles

Bundle name: `refs-<tag>-<tier>-<device>-<preset>-<fingerprint-short>.tar.zst`, where `fingerprint-short` is the first 8 hex digits of a sha256 over CPU model, GPU name, python, numpy, pyfftw, numba and cupy versions. One GitHub release per tag on abtem-benchmarks (`refs-v1.0.10`), one asset per bundle, `refs/INDEX.json` in git listing assets with their manifests' key fields.

`abtem-bench refs fetch v1.0.10 --tier quick --device cpu` picks the asset whose fingerprint matches the local machine; if none matches it takes the closest (same device class) and compare downgrades the expectation from bit-exact to the float64 tolerance and says so in the report header. `abtem-bench refs upload <bundle>` attaches an asset and updates `INDEX.json` (a commit in abtem-benchmarks). Local cache: `~/.cache/abtem-bench/refs/<name>/`.

Size estimate: the quick tier of the M2 matrix is a few hundred float64 arrays of at most 256² plus one 4D-STEM stack, under 100 MB per bundle compressed; the standard tier is a few GB and is captured only on demand. Release assets keep the git history of abtem-benchmarks at its current 1 MB.

## 11. Case matrix

M1 (feeds the v1.1 release notes): `potential.infinite` (prerequisite, seconds), `hrtem.exitwave` (Si and SrTiO3 plane wave through 20 to 60 slices; outputs `exitwave` complex, `saed` diffraction pattern; `[order1]`), `stem.multidetector` (BF + ADF + segmented; `[order1]`, `[eager]`, `[auto]`), `diffraction.cbed` (probe CBED, six decades of dynamic range; `[order1]`).

M2: `potential.finite` (cpu/gpu pair, non-bitwise), `potential.crystal` (CrystalPotential with `num_frozen_phonons` and pool sizes, emits pool seeds and displacements), `potential.chunked` (explicit `potential_chunk_size` 1, 5, all; outputs must be identical to each other), `stem.haadf[batch1|batch8|auto|eager]`, `stem.4d` (PixelatedDetector with explicit `max_angle`), `stem.flexible` (FlexibleAnnularDetector plus radial integration), `stem.thickness_series` (`exit_planes`), `prism.scan` (SMatrix interpolation 1 and 2; pair vs `stem.haadf` at 1e-3), `phonons.haadf` (FrozenPhonons, fixed seed, outputs `image`, `displacements`, `seeds`), `energy.ensemble` (plane wave and probe with a three-energy list and a gaussian spread), `hrtem.focal_spread` (CTF with `focal_spread`, both `ensemble_mean` settings), `coreloss.image` (synthetic seeded `TransitionPotentialArray` on all tiers; real O K edge from a precomputed asset in abtem-benchmarks on local tiers; outputs `image`, `dtype`), `coreloss.prism` (PRISM-EELS interpolation 1 to 3, `[crop]`), `bloch.diffraction` (StructureFactor + BlochWaves thickness series), `grid.fastfft_pair` (2623×2271 vs 2625×2304 on the tWSe2 cell; standard and large tiers; the speed and VRAM case for #347), `consistency.cache_reuse` (one process builds Si at grid A, Au at grid B, Si at grid A again, on cpu then gpu; third equals first bit-exactly), `gpu.multigpu_one_worker` (`requires` dask-cuda and at least two visible devices; `dask.multi-gpu-devices=[0]`; pair vs single-GPU run).

Later: non-orthogonal grids once #282 merges (an entire grid class every cache has to respect), plasmons, magnetism, C-PRISM and BiP-PRISM cases contributed by their authors.

Tier budgets: quick under 5 s per case on CPU so the whole quick matrix runs in well under 10 minutes on a hosted runner; standard 30 to 120 s; large the sizes from `legacy/PROVENANCE.md` (2048² to 4096² grids, 3 to 36 GB potentials, the 16384² stress cases), GPU only, Perlmutter.

## 12. Repository layout, packaging, lint

abTEM, branch `benchmark-suite` (PR to dev):

```
benchmarks/
  pyproject.toml            # package abtem_bench, console script abtem-bench, deps: numpy, tabulate; extras: none
  README.md
  accepted_changes.toml
  abtem_bench/
    __init__.py  cli.py  registry.py  presets.py  worker.py  meters.py  runner.py  store.py  compare.py  refs.py  fixtures.py
  cases/
    __init__.py  potentials.py  hrtem.py  diffraction.py  stem.py  prism.py  phonons.py  energy.py  coreloss.py  bloch.py  consistency.py  gpu.py
  tests/                    # harness unit tests: store round-trip, compare verdicts, accepted_changes validation, case hash, registry checks
abtem/core/testing.py       # array_is_close (moved), close_stats (new); test/utils.py re-exports
```

`python -m abtem_bench` works with `PYTHONPATH=benchmarks` or after `uv pip install -e benchmarks`; the runner always uses the `PYTHONPATH` form so the worker sees the invoking checkout's harness. The main `pyproject.toml` gains `[tool.setuptools.packages.find] exclude = ["benchmarks*", "test*"]` so nothing ships in the wheel. ruff already covers `benchmarks/` (only `test` is excluded); `mypy.ini` gains `files = abtem, benchmarks/abtem_bench`. The stale `.pre-commit-config.yaml` (black and flake8 at 120 columns) is aligned to ruff in M4. Harness tests run under `pytest benchmarks/tests` in the normal CI job.

abtem-benchmarks: `design/`, `legacy/`, `refs/` as in `../README.md`.

## 13. CI

Workflow `benchmark.yml`, `workflow_dispatch` and pull requests labelled `benchmark`; ubuntu, Python 3.12, `uv sync --group test`, no GPAW, no GPU:

1. A/B gate: quick tier, CPU, `accuracy` preset, PR head vs merge base with dev, interleaved, in the same job. Expectation bit-identical; `DRIFT` or `SHAPE` fails unless matched by `accepted_changes.toml` changed in the same PR. This answers "did this PR change results".
2. Informational: compare the PR head against the latest tag bundle fetched from abtem-benchmarks; posted to the job summary, never failing. This is the running v1.1 release-notes table.
3. Self-check on dev, weekly, to accumulate the false-positive record. The check becomes required only after about a month with zero false positives.

Speed and memory are never judged on hosted runners. They run on the dev box and on Perlmutter with the same CLI (M3 runbook; the weekly GPU test job in `dev-env/cron` is the model, including its `ABTEM_CI_VENV` plus `PYTHONPATH` convention).

## 14. Machines

Dev box (24 cores, 46 GB, one Radeon 8050S shared with the display): CPU tiers only when unattended; GPU runs deliberately and never two suites at once; the runner refuses to start a case when `MemAvailable` is below 8 GB and records `SKIPPED-MEMORY`. Timings on the iGPU wobble and rocFFT dominates plan memory, so GPU speed and VRAM numbers from this box are informational.

Perlmutter (A100, account per `dev-env/cron/README.md`): the citable GPU machine. Short checks run interactively in `salloc`; the standard and large tiers run under `sbatch` with `--gpu-bind=none` when multi-GPU cases are included. Wall-clock repeatability measured there is ±0.2 % on a quiet node; never compare across nodes.

## 15. Risks and open questions

- Bit identity across machines is not achievable (numba `fastmath`, BLAS, FFTW codelets); the fingerprint policy makes this explicit rather than silently tolerant.
- GPU float64 tolerance will be case-dependent; the self-check floor per case is stored in the bundle so compare can use per-case floors rather than one global number.
- `accepted_changes` globs can be written too broadly; the stale-entry check and the reviewer see the matched drift vector in the report.
- The synthetic transition potential exercises the scatter machinery but not the radial solver; the real O K asset covers physics locally, and both are labelled as such in reports.
- Reference bundle captures must run on a quiet machine; the manifest records load average and `MemAvailable` at start so a noisy capture can be recognised later.
- The `[order1]` variant makes the propagator change attributable, but two changes in one PR can still hide behind one accepted entry; the reason field must name the PR.
