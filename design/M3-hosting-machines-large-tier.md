# M3: Reference hosting, machines, memory mode and the large tier

Target: two weeks; one PR on abTEM (`refs`, isolated mode, memory mode), one release on abtem-benchmarks, one runbook.

## Goal

Make the reference bundles a shared resource rather than files on one laptop, make GPU numbers citable, and make the suite reach the sizes at which the memory bugs of the last six months actually appeared.

## Scope

- `refs.py` and `abtem-bench refs upload|fetch|list`: naming with machine fingerprint, `gh release` assets on abtem-benchmarks, `refs/INDEX.json`, local cache, the fingerprint-mismatch downgrade in compare.
- Isolated environment mode for tags at or below v1.0.9: `uv venv` per ref plus editable install of the worktree with a constraints file frozen from the invoking environment; never touches the shared venv.
- `memory` preset: N fresh processes per case, median of `wait4` peaks, `auto` variants unflagged; report shows the per-process spread.
- Timeout and OOM handling verified on purpose with a case sized to OOM the iGPU.
- Large tier: the sizes from `legacy/PROVENANCE.md` (2048² and 4096² grids with 3 to 36 GB potentials, 8192² and 16384² stress, the tWSe2 2623×2271 Bluestein pair, PRISM-EELS reduction at the #423 size), GPU only.
- Perlmutter runbook `benchmarks/docs/perlmutter.md`: `salloc` command for short checks, `sbatch` template for standard and large tiers (`--gpu-bind=none` for the multi-GPU case), the `ABTEM_CI_VENV` plus `PYTHONPATH` convention from `dev-env/cron/abtem_gpu_tests.sh`, and the rule that numbers are never compared across nodes.
- Weekly hook: the local `cron/abtem_gpu_tests.sh` gains an optional `abtem-bench self-check` step so the false-positive record accumulates without anyone remembering to run it.

## Deliverables

- Release `refs-v1.0.10` on abtem-benchmarks with the quick-tier CPU bundle from the dev box and quick plus standard GPU bundles from a Perlmutter A100, `INDEX.json` committed.
- dev vs v1.0.10 report on GPU (A100) attached to #380 alongside the CPU report from M1.
- `gpu.multigpu_one_worker` verified on Perlmutter: single GPU vs one-worker dask-cuda cluster, bit-identical at float64 or the documented floor.

## Exit criteria

- `abtem-bench refs fetch v1.0.10 --tier quick --device cpu` on a fresh clone downloads and caches the bundle and `compare` against it reproduces the M1 report.
- Isolated mode captures v1.0.9 on the dev box and compare pairs it with v1.0.10.
- Memory mode on the dev box CPU: per-process spread of peak RSS under the accuracy preset below 5 % for every quick case (FFTW_ESTIMATE removes the trial-plan spread); if not, the case is investigated.
- Large tier on Perlmutter completes for every case that is expected to complete, and records `OOM` for the stress cases that are expected to OOM.

## Verification commands

```
python -P -m abtem_bench refs upload /workspaces/run/bench-out/v1.0.10 --tag v1.0.10
python -P -m abtem_bench refs fetch v1.0.10 --tier quick --device cpu
python -P -m abtem_bench capture --ref v1.0.9 --isolated --tier quick --device cpu --preset accuracy --out /workspaces/run/bench-out/v1.0.9
python -P -m abtem_bench run --ref dev --ref dev --tier quick --device cpu --preset memory --processes 5 --out /workspaces/run/bench-out/mem
# Perlmutter, interactive:
salloc -C gpu -q interactive --gpus 1 -A m5395 -t 01:00:00
```

## Risks

- Release assets need a token with `contents: write` on abtem-benchmarks; Paul has push, so `gh release upload` from his account works. CI fetches are anonymous (public repository).
- Bundle sizes for the standard tier on GPU may reach several GB; assets accept that, but `INDEX.json` should mark them optional and the CI job fetches `quick` only.
- The iGPU cannot answer cuFFT plan-memory questions (rocFFT strategy differs); the VRAM meter is validated on the A100 with `legacy/run/fft_plan_probe.py` as the check.
