# M1: Reference mode and the v1.1 drift table

Target: about two weeks of work; one PR on abTEM (`benchmark-suite` branch, base `dev`), one release on abtem-benchmarks.

Status 2026-09-23: harness, four cases, tests and docs implemented on the `benchmark-suite` branch; quick-tier CPU results below. Remaining for M1: the PR itself and Toma's word on the #269 finding.

## Results (dev box, CPU, quick tier, accuracy preset, 2026-09-23)

- Self-check (`dev` captured twice, interleaved): all 9 case ids bit-identical; speed spread at most 1.1 %, peak-RSS spread at most 0.3 %.
- `dev` vs `v1.0.10`, default variants: every multislice case drifts as expected from the exact propagator (#298): `hrtem.exitwave` rel 1.6e-1 / intensity 2.0e-4, `diffraction.cbed` rel 1.2 / intensity 1.2e-4, `stem.multidetector` rel 5.1e-3 / intensity 3.9e-3. These are accepted in `accepted_changes.toml` with PR 298.
- `dev[order1]` vs `v1.0.10` (propagator held at order 1): a residual drift remains: `hrtem.exitwave` 2.5e-7, `diffraction.cbed` 1.3e-6, `stem.multidetector` 1.0e-6 relative, and `potential.infinite` itself 7.4e-8 relative (intensity 2.8e-8). Bisecting `potential.infinite` over the first-parent merges between v1.0.10 and dev with the harness itself points at `8fa77bdd`, PR #269 (potential chunking and GPU update, 2026-07-15), as the first merge that changes the CPU float64 infinite-projection potential (5.1e-8 relative). Not accepted: it is Toma's call whether this is an intended numerical change (it is far above float64 round-off) and it needs a reason line.
- Speed and memory: no flags; dev is between 1 % slower and 9 % faster than v1.0.10 across the cases at this size, peak RSS within 5 %.

## Results (Perlmutter A100, GPU, quick tier, accuracy preset, `idrobolab` container, 2026-09-24)

- Self-check on GPU: all 9 case ids bit-identical in float64. The design's assumption that GPU float64 needs a tolerance did not materialise at quick-tier sizes on the A100; keep the fingerprint-scoped tolerance policy for larger tiers until measured.
- Accuracy against v1.0.10: identical to the CPU numbers to two digits for every case and variant, so the #269 residual is device-independent (algorithmic, not reduction order).
- `stem.multidetector` (all variants) is `ERROR` on v1.0.10: its detector kernels are `numba.cuda` (`abtem/core/_cuda.py` at v1.0.10, `@cuda.jit`) and need `libnvvm`, absent from the CUDA runtime image. dev's CuPy `RawModule` kernels need no NVVM. Measuring that case against v1.0.10 on GPU needs a `devel` image or a host CUDA toolkit; note for the M3 runbook and a release-notes line for v1.1 (no numba.cuda dependency on GPU).
- Host RSS of dev GPU workers 730-830 MB vs v1.0.10 530-600 MB: a constant offset of about 240 MB on every comparable case, absent on CPU. Import-only RSS of both checkouts is equal on the ROCm dev box, so the cause is CUDA-specific and still open (candidates: CUDA-side libraries loaded by dev's `cupyx.scipy.signal`/RawModule paths). Follow up in M2 with the `nominal_bytes` field, which separates a constant offset from a scaling one.
- GPU quick-tier wall times are 0.07-0.6 s; differences of tens of ms are launch jitter. compare marks speed ratios on runs shorter than `--min-time` (0.5 s) as `short` instead of flagging. GPU speed is judged on the standard tier.
- Cold ratios on GPU (down to 0.15 in the self-check) are not comparable across processes: CuPy's on-disk kernel cache under `$HOME` warms every later process. A true cold needs `CUPY_CACHE_DIR` pointed at a fresh directory per capture; M2 decides whether that belongs in the accuracy preset.

## Perlmutter procedure that worked (for the M3 runbook)

Independent clone under `$PSCRATCH` (never the CI checkout); `UV_CACHE_DIR` on scratch (now in `hpcenvs/perlmutter/env.sh`); refs resolved on the host with `uv run --no-project python -P -m abtem_bench.prepare --ref origin/dev --ref v1.0.10` because the runtime image has no git (git added to the image on 2026-09-24, effective at the next rebuild); then one `srun ... podman-hpc run --rm --group-add keep-groups --gpu -v $CFS:$CFS -v $SCRATCH:$SCRATCH -v $HOME:$HOME -e PYTHONPATH=<clone>/benchmarks:<clone> -e OUT --workdir "$PWD" idrobolab:latest bash -c '...'` running self-check, both captures and compare. Whole sequence well under 15 minutes on one A100 in the shared QOS.


## Goal

Answer Toma's immediate question: how does dev differ numerically from v1.0.10, workload by workload, and how much of that is the exact Fresnel propagator. Everything else in M1 exists to make that answer trustworthy and repeatable.

## Scope

Harness core, in `benchmarks/abtem_bench` on abTEM:

- `registry.py`: `@case`, `Case`, `Tier`, `Variant`, `Tolerance`, case ids, `requires=`, `compare_as`, the `sampling=` rejection, and `case_hash()` (sha256 over sorted `cases/*.py` contents plus harness version).
- `presets.py`: the `accuracy` preset of DESIGN section 6.1 (float64, FFTW_ESTIMATE, one thread, synchronous scheduler, explicit sizes, seeds, env); dump of resolved `abtem.config.config`, `dask.config.config` and env into the manifest.
- `worker.py`: one case id per process; prints and records ref, sha, `abtem.__file__`, `abtem.__version__`; asserts the import came from the intended worktree; applies the preset; builds (untimed); cold run; warm repeats; VRAM sampler; collects outputs to host; writes the case record.
- `meters.py`: wall clock with GPU synchronisation, process CPU time, VRAM sampler (pool and driver), ported from `legacy/abtem-repo/benchmark_potential_chunking.py`.
- `runner.py`: `git worktree` per ref under `.worktrees/bench/<sha>`, `PYTHONPATH=<benchmarks>:<worktree>` and `python -P`, `os.wait4` peak RSS, per-tier timeouts, stderr classification (ported), `MemAvailable` floor of 8 GB, interleaving (needed already for the self-check).
- `store.py`: the schema of DESIGN section 7, `schema_version = 1`, fingerprint, bundle read and write, `.tar.zst` pack and unpack.
- `compare.py`: pairing (including `compare_as`), the metric vector of section 8.1, verdicts of 8.2, `accepted_changes.toml` handling of 8.3 with validation and stale-entry detection, noise floor from a self-check bundle, Markdown and JSON reports, `--fail-on`.
- `cli.py`: `list`, `capture`, `self-check`, `compare`. (`run --ref A --ref B` is M2, `refs` is M3.)
- `fixtures.py`: `silicon(reps)`, `srtio3(reps)`; nothing that needs GPAW.
- `abtem/core/testing.py`: `array_is_close` moved from `test/utils.py` (which re-exports it, so nothing in the test suite changes), plus `close_stats()`.
- `benchmarks/tests/`: store round trip, compare verdict matrix on synthetic arrays (identical, drift, shape, accepted, stale entry), case hash stability, registry rejections, `accepted_changes` validation.
- `benchmarks/pyproject.toml`, `benchmarks/README.md`, main `pyproject.toml` package exclusion, `mypy.ini` files entry.

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
- `python -P -m abtem_bench list` works with `PYTHONPATH=benchmarks` and after `uv pip install -e benchmarks` (in a throwaway venv, never the shared one).
- Adversarial pass on the diff before handoff: a case with `sampling=` is rejected; a tampered `cases/` file changes the hash and compare refuses; a ref whose `abtem.__file__` is not the worktree aborts; a killed worker records `OOM` or `ERROR`, not a missing row.

## Verification commands

```
cd /workspaces/code/abTEM && git switch benchmark-suite
export PYTHONPATH=$PWD/benchmarks
python -P -m abtem_bench list --tier quick
python -P -m abtem_bench self-check --ref dev --tier quick --device cpu --preset accuracy --out /workspaces/run/bench-out/selfcheck
python -P -m abtem_bench capture --ref v1.0.10 --tier quick --device cpu --preset accuracy --out /workspaces/run/bench-out/v1.0.10
python -P -m abtem_bench capture --ref dev --tier quick --device cpu --preset accuracy --out /workspaces/run/bench-out/dev
python -P -m abtem_bench compare /workspaces/run/bench-out/v1.0.10 /workspaces/run/bench-out/dev --noise /workspaces/run/bench-out/selfcheck --md report.md --fail-on drift
```

## Risks

- Bit identity on CPU may fail for a case because of a hidden nondeterminism (thread pool in scipy, numba parallel kernels in `finite_difference.py`). The self-check finds it before the reference capture; the fix is a pin, or a documented non-bitwise tolerance for that case.
- v1.0.10 lacks `potential_chunk_size`; the `chunk` parameter is passed only where accepted (registry probes the signature, the old ref records the parameter as `n/a`).
- The `[order1]` comparison may reveal drift from the integral-table cache rework of September; that is a finding, not a blocker, and goes to Toma with the report.
