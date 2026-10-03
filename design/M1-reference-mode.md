# M1: Reference mode and the v1.1 drift table

Target: about two weeks of work; one PR on abTEM (`benchmark-suite` branch, base `dev`), one release on abtem-benchmarks.

Status 2026-10-03: implemented on the `benchmark-suite` branch, rebased onto dev `a562ce4a`, with the fixes from three review rounds (fresh records per run, the worker's own peak RSS with the meter recorded, case hash over every module that shapes a case, accepted changes scoped by `since` and bounded, with `max_abs_norm` required, a strict `--fail-on` with exit 2 for any input error or crash, the `--min-delta` rule, floors for memory, VRAM and accuracy, `--overwrite` limited to bundles). Every drift against v1.0.10 is explained and accepted. Remaining for M1: the PR itself.

## Results (workstation CPU, quick tier, accuracy preset, 2026-10-03)

Captured on a 24-thread AMD Ryzen AI MAX PRO 390 at dev `a562ce4a`. Bundles and report outside the repository until M3 hosts them.

- Self-check (dev captured twice, interleaved): all 9 case ids bit-identical; speed spread at most 3.3 %, peak-RSS spread at most 0.8 %. An earlier capture showed dev `a562ce4a` bit-identical to dev `fba42a98` on all 9 ids.
- dev vs v1.0.10, default variants, the exact propagator (#298), per output (relative error / integrated intensity): exit wave 5.2e-3 / 4.8e-6, its SAED pattern 1.6e-1 / 2.0e-4, CBED 1.2 / 1.2e-4, ADF 5.1e-3 / 3.9e-3, BF 5.4e-5 / 8.0e-6, segmented 1.4e-3 / 1.0e-4. Accepted under PR 298, bounded in intensity at about twice the largest value measured on the quick and standard tiers.
- dev `[order1]` vs v1.0.10 (propagator held at order 1): potential 7.4e-8 / 2.8e-8, exit wave 6.8e-8, SAED 2.5e-7 / 9.9e-8, CBED 1.3e-6 / 2.5e-8, segmented 1.0e-6 / 2.0e-7, ADF 1.4e-7, BF 1.6e-8. Cause: two float32 roundings in v1.0.10 that #269 (merge `8fa77bdd`) made float64: `pi` and `pi**2` in `projected_scattering_factor` (`abtem/parametrizations/functions/lobato.py`), and the accumulator of `DiffractionPatterns._radial_binning` (`abtem/measurements.py:3061` at v1.0.10), which the segmented detector uses. The first-parent bisect lands on `8fa77bdd` (its first parent is bit-identical to v1.0.10). With both restored to float32 in dev, the `[order1]` outputs of `potential.infinite`, `hrtem.exitwave` and `stem.multidetector` are bit-identical to v1.0.10's and the CBED pattern differs by 7e-14 (in the SrTiO3 potential, every even slice by 1.5e-14; mechanism not isolated). The accumulator is most of the segmented drift. Accepted under PR 269, bounded in all three metrics. A changed summation order cannot explain it: dev's potential is bit-identical for slice chunk sizes 1, 4, 22 and auto.
- Speed and memory: dev 1 % to 19 % faster than v1.0.10 and peak RSS within 5 %, but v1.0.10 was captured after the self-check rather than interleaved with dev, on a machine with a load average of 3 to 5; read as "no slowdown" only.
- Slices per case (from `Potential.num_slices` at the tier parameters): quick: potential 22, hrtem 28, cbed 40, stem 22; standard: potential 30, hrtem 157, cbed 157, stem 41.

## Results (workstation CPU, standard tier, accuracy preset, 2026-10-03)

Captured per case group (`rest` = potential, HRTEM and CBED cases; `stem`; `stem-order1`) with the harness at `98675b45`, one cold run per case. Rows: accepted-changes entry; cells: the largest value over the case's outputs of rel / |intensity| / max_abs_norm, dev against v1.0.10.

| entry | quick tier | standard tier |
|---|---|---|
| `hrtem.exitwave` (#298) | 0.16 / 2.0e-4 / 2.8e-3 | 6.5 / 3.1e-3 / 0.25 |
| `diffraction.cbed` (#298) | 1.2 / 1.2e-4 / 2.2e-3 | 9.0 / 6.4e-4 / 7.7e-2 |
| `stem.multidetector` (#298) | 5.1e-3 / 3.9e-3 / 4.9e-3 | 4.0e-3 / 7.9e-4 / 4.0e-3 |
| `potential.infinite` (#269) | 7.4e-8 / 2.8e-8 / 5.1e-8 | 5.3e-8 / 2.8e-8 / 5.2e-8 |
| `hrtem.exitwave[order1]` (#269) | 2.5e-7 / 9.9e-8 / 9.5e-8 | 4.8e-6 / 1.6e-7 / 6.0e-7 |
| `diffraction.cbed[order1]` (#269) | 1.3e-6 / 2.5e-8 / 2.2e-7 | 2.6e-5 / 3.9e-8 / 2.8e-7 |
| `stem.multidetector[order1]` (#269) | 1.0e-6 / 2.0e-7 / 5.5e-7 | 3.7e-6 / 7.5e-7 / 3.2e-6 |

- The non-STEM standard values agree with the A100's to two digits; STEM at standard had not been measured before (v1.0.10's STEM fails on the A100 image).
- `accepted_changes.toml` bounds are about twice the larger of the two columns, rounded up to one significant digit; the #298 HRTEM and CBED entries are split per tier because their `max_abs_norm` grows about a hundred-fold. `[eager]` and `[auto]` were not captured at standard (their quick-tier outputs equal the default's) and share the default's bounds.
- The standard `stem.multidetector` takes 2618 to 2822 s cold on CPU (four captures running at once), beyond its former 1800 s timeout; the tier now declares 5400 s.

## Results (Perlmutter A100, GPU, quick tier, accuracy preset, 2026-09-24)

At dev `fba42a98`, with the harness revision of that day (peak RSS from `os.wait4`), in a container image with CuPy 13.5.1 and the CUDA 12.8 runtime. Not re-measured since; #453 changed lazy GPU paths in between.

- Self-check on GPU: all 9 case ids bit-identical in float64. The design's assumption that GPU float64 needs a tolerance did not materialise at quick-tier sizes on the A100; the self-check floor still widens the tolerance where a machine is not bit-stable.
- Accuracy against v1.0.10: identical to the CPU numbers to two digits for every case and variant, so the float32-constant residual is device-independent.
- `stem.multidetector` (all variants) is `ERROR` on v1.0.10: its detector kernels are `numba.cuda` (`abtem/core/_cuda.py` at v1.0.10, `@cuda.jit`) and need `libnvvm`, absent from the CUDA runtime image. dev's CuPy `RawModule` kernels need no NVVM. Measuring that case against v1.0.10 on GPU needs a `devel` image or a host CUDA toolkit; note for the M3 runbook and a release-notes line for v1.1 (no numba.cuda dependency on GPU).
- Host RSS of dev GPU workers 730-830 MB vs v1.0.10 530-600 MB: a constant offset of about 240 MB on every comparable case, absent on CPU. Measured with the `os.wait4` meter, which carries the runner's high-water mark; re-measure with the worker's `VmHWM` before chasing the cause. Import-only RSS of both checkouts is equal on the workstation's ROCm stack, so the offset, if it holds, is CUDA-specific.
- GPU quick-tier medians are 0.07-0.66 s; differences of tens of milliseconds are launch jitter. compare marks a speed ratio whose medians differ by less than `--min-delta` (0.05 s) as `short` instead of flagging it. GPU speed is judged on the standard tier.
- Cold ratios on GPU (down to 0.15 in the self-check) are not comparable across processes: CuPy's on-disk kernel cache under `$HOME` warms every later process. A true cold needs `CUPY_CACHE_DIR` pointed at a fresh directory per capture; M2 decides whether that belongs in the accuracy preset.

## Results (Perlmutter A100, GPU, standard tier, accuracy preset, 2026-09-24)

- Self-check: all 9 ids bit-identical in float64 at 1024² with 30 to 157 slices and a 16×16 scan; speed spread ≤ 3 %.
- v1.0.10 vs dev: potential 5.3e-8 (SrTiO3); `hrtem.exitwave[order1]` 4.8e-6 and `diffraction.cbed[order1]` 2.6e-5 relative (largest output of each), against 2.5e-7 and 1.3e-6 on the quick tier. The tiers differ in grid and number of slices (28 and 40 vs 157 each), and the HRTEM case also in structure (Si vs SrTiO3) and slice thickness (2 vs 1 Å), so this is growth with the size of the calculation, not a fitted law. Default HRTEM and CBED variants 6.5 / 9.0 relative, 3.1e-3 / 6.4e-4 intensity (#298, inside the bounds). dev 1–5 % faster on every measurable case. `stem.multidetector` still `ERROR` on v1.0.10 (libnvvm).
- Host-RSS offset constant (~150–230 MB, `os.wait4` meter): ratio 1.11 on `potential.infinite` (1257 → 1401 MB, 1 GB array) vs 1.29–1.39 on small-array cases.
- For M2's consistency pairs: lazy default `stem.multidetector` (`max_batch=8`) 6.46 s vs eager 2.58 s on the A100 with bit-identical outputs; `[auto]` (lazy, auto batch) 2.66 s. The cost is in small lazy batches. On the workstation CPU at quick the same pair is 4.5 vs 3.5 s.
- Tier calibration: standard runs in about 6 minutes per ref on one A100 (accuracy preset, 3 repeats). Adequate as the GPU speed tier.

## Perlmutter procedure that worked (for the M3 runbook)

Independent clone under `$PSCRATCH` (never the CI checkout); `UV_CACHE_DIR` on scratch (now in `hpcenvs/perlmutter/env.sh`); refs resolved on the host with `uv run --no-project python -P -m abtem_bench.prepare --ref origin/dev --ref v1.0.10` because the runtime image has no git (git added to the image on 2026-09-24, effective at the next rebuild); then one `srun ... podman-hpc run --rm --group-add keep-groups --gpu -v $CFS:$CFS -v $SCRATCH:$SCRATCH -v $HOME:$HOME -e PYTHONPATH=<clone>/benchmarks:<clone> -e OUT --workdir "$PWD" idrobolab:latest bash -c '...'` running self-check, both captures and compare. Whole sequence well under 15 minutes on one A100 in the shared QOS.


## Goal

Answer Toma's immediate question: how does dev differ numerically from v1.0.10, workload by workload, and how much of that is the exact Fresnel propagator. Everything else in M1 exists to make that answer trustworthy and repeatable.

## Scope

Harness core, in `benchmarks/abtem_bench` on abTEM:

- `registry.py`: `@case`, `Case`, `Tier`, `Variant`, `Tolerance`, case ids, `requires=`, `compare_as`, the `sampling=` rejection, and `case_hash()` (sha256 over the case modules and the harness modules that shape a case, plus harness version).
- `presets.py`: the `accuracy` preset of DESIGN section 6.1 (float64, FFTW_ESTIMATE, one thread, synchronous scheduler, explicit sizes, seeds, env); dump of resolved `abtem.config.config`, `dask.config.config` and env into the manifest.
- `worker.py`: one case id per process; prints and records ref, sha, `abtem.__file__`, `abtem.__version__`; asserts the import came from the intended worktree; applies the preset; builds (untimed); cold run; warm repeats; VRAM sampler; collects outputs to host; writes the case record.
- `meters.py`: wall clock with GPU synchronisation, process CPU time, VRAM sampler (pool and driver), ported from `legacy/abtem-repo/benchmark_potential_chunking.py`.
- `runner.py`: `git worktree` per ref under `.worktrees/bench/<sha>`, `PYTHONPATH=<benchmarks>:<worktree>` and `python -P`, fresh case records, per-tier timeouts, stderr classification (ported), `MemAvailable` floor of 8 GB, interleaving (needed already for the self-check).
- `store.py`: the schema of DESIGN section 7, `schema_version = 1`, fingerprint, bundle read and write, `.tar.zst` pack and unpack.
- `compare.py`: pairing (including `compare_as`), the metric vector of section 8.1, verdicts of 8.2, `accepted_changes.toml` handling of 8.3 with validation and stale-entry detection, noise floor from a self-check bundle, Markdown and JSON reports, `--fail-on`.
- `cli.py`: `list`, `capture`, `self-check`, `compare`. (`run --ref A --ref B` is M2, `refs` is M3.)
- `fixtures.py`: `silicon(reps)`, `srtio3(reps)`; nothing that needs GPAW.
- `abtem/core/testing.py`: `array_is_close` moved from `test/utils.py` (which re-exports it, so nothing in the test suite changes), plus `close_stats()`.
- `benchmarks/tests/`: store round trip, compare verdict matrix on synthetic arrays (identical, drift, shape, accepted, stale entry), case hash stability, registry rejections, `accepted_changes` validation.
- `benchmarks/pyproject.toml`, `benchmarks/README.md`, main `pyproject.toml` package exclusion, a CI step running the harness tests. (The `mypy.ini` files entry is dropped: following imports into abtem reports 295 errors in abtem's own modules; the harness is checked by path with `--follow-imports=silent`.)

Cases (quick tier first, standard tier parameters declared but calibrated in M2), each with the `[order1]` variant:

- `potential.infinite`: Si and SrTiO3, lobato, `projection="infinite"`, `.build()`; output `potential`.
- `hrtem.exitwave`: plane wave 200 keV through Si (20 slices) and SrTiO3 (40 slices); outputs `exitwave` (complex), `saed` (diffraction pattern with `block_direct=True`).
- `stem.multidetector`: BF + ADF + segmented as in DESIGN section 5; outputs `bf`, `adf`, `segmented`; variants `order1`, `eager`, `auto`.
- `diffraction.cbed`: probe CBED at 200 keV on SrTiO3; output `cbed`; tests the `rel_above` masking over six decades.

Capture and compare on the dev box:

1. `abtem-bench self-check --ref dev --tier quick --device cpu --preset accuracy` twice on different days: all `IDENTICAL`.
2. `abtem-bench capture --ref v1.0.10 --tier quick --device cpu --preset accuracy --out refs/work/v1.0.10-quick-cpu`.
3. `abtem-bench capture --ref dev ...` and `abtem-bench compare` both ways: default vs default (everything drifts, expected) and `dev[order1]` vs `v1.0.10` default (expected: identical or within the float64 floor for the cases whose only change is the propagator; anything else is a finding for the release notes).
4. Fill `accepted_changes.toml` with the propagator entry and any other explained drift; unexplained drift is investigated before the PR is called ready.
5. GPU on the dev box: informational only (iGPU); recorded in the PR body as such.

## Deliverables

- PR on abTEM: harness, four cases, tests, `accepted_changes.toml` with the #298 entry, README with the three commands above.
- The dev vs v1.0.10 quick-tier CPU report (Markdown) attached to the PR and to discussion #380.
- The v1.0.10 quick-tier CPU bundle kept locally under `abtem-benchmarks/refs/work/` until M3 publishes it as a release asset.

## Exit criteria

- Self-check on CPU under `accuracy`: every output `IDENTICAL` on two separate days.
- `dev[order1]` vs `v1.0.10`: every case `IDENTICAL` or `OK` at the float64 floor, or the difference is explained and listed in `accepted_changes.toml` with a PR number.
- `ruff check benchmarks`, `mypy benchmarks/abtem_bench`, `pytest benchmarks/tests` and the existing `pytest test` are clean.
- `python -P -m abtem_bench list` works with `PYTHONPATH=benchmarks` and after `uv pip install -e benchmarks` into an environment that has abtem's dependencies (the workers run on the CLI's Python).
- Adversarial pass on the diff before handoff: a case with `sampling=` is rejected; a tampered `cases/` file changes the hash and compare refuses; a ref whose `abtem.__file__` is not the worktree aborts; a killed worker records `OOM` or `ERROR`, not a missing row.

## Verification commands

```
git -C <abTEM checkout> worktree add <abTEM checkout>/.worktrees/benchmark-suite benchmark-suite   # never switch a shared checkout
export PYTHONPATH=<abTEM checkout>/.worktrees/benchmark-suite/benchmarks
python -P -m abtem_bench list --tier quick
python -P -m abtem_bench self-check --ref dev --tier quick --device cpu --preset accuracy --out /workspaces/run/bench-out/selfcheck
python -P -m abtem_bench capture --ref v1.0.10 --tier quick --device cpu --preset accuracy --out /workspaces/run/bench-out/v1.0.10
python -P -m abtem_bench capture --ref dev --tier quick --device cpu --preset accuracy --out /workspaces/run/bench-out/dev
python -P -m abtem_bench compare /workspaces/run/bench-out/v1.0.10 /workspaces/run/bench-out/dev --noise /workspaces/run/bench-out/selfcheck --md report.md --fail-on drift,shape,error,missing
```

## Risks

- Bit identity on CPU may fail for a case because of a hidden nondeterminism (thread pool in scipy, numba parallel kernels in `finite_difference.py`). The self-check finds it before the reference capture; the fix is a pin, or a documented non-bitwise tolerance for that case.
- v1.0.10 lacks `potential_chunk_size`; the `chunk` parameter is passed only where accepted (registry probes the signature, the old ref records the parameter as `n/a`).
- The `[order1]` comparison may reveal drift from the integral-table cache rework of September; that is a finding, not a blocker, and goes to Toma with the report.
