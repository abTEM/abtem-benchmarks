# M2: Live A/B mode and the full case matrix

Target: two to three weeks; two PRs on abTEM (mode and meters; cases).

## Goal

Make the suite answer "did this change alter results, speed or memory" for any two checkouts, and cover the workloads Toma listed plus the ones the issue history says are fragile. Every legacy script ends this milestone with a reproduced case or an explicit "not needed" line.

## Scope

Runner and compare:

- `run --ref A --ref B [--ref C ...]`: ABAB interleaving per case, `--rounds N`, one bundle per ref, compare called automatically; `--baseline` chooses the reference side.
- `UNSUPPORTED` guards exercised for real (cases that need `potential_chunk_size`, `SegmentedDetector`, PRISM-EELS, `transition_potential_scan`).
- Consistency pairs within a bundle: `(default, eager)`, `(cpu, gpu)`, `(prism, multislice)`, `(batch1, batch8)`, `(float32, float64)`; "gap widened" reporting.
- `float32-tracking` preset (DESIGN 6.2) and the `speed` preset (6.3) with the per-case `warmup` policy; `cold` reported separately.
- Graph-transport meter for lazy cases: task count and pickled graph bytes.
- Per-case noise floors stored in the self-check bundle and used by compare.

If the #380 follow-up of 2026-09-19 is accepted (DESIGN 15.2), M2 also gains `abtem-bench profile <case-id>` and the first `routine.*` case, `routine.radial_bin_sum`; both are scoped there and not below.

Cases (see DESIGN section 11 for parameters): `potential.finite`, `potential.crystal`, `potential.chunked`, `stem.haadf[batch1|batch8|auto|eager]`, `stem.4d`, `stem.flexible`, `stem.thickness_series`, `prism.scan`, `phonons.haadf` (with `displacements` and `seeds` outputs), `energy.ensemble`, `hrtem.focal_spread`, `coreloss.image` (synthetic seeded transition potential everywhere; real O K asset on local tiers, produced once with GPAW and stored in abtem-benchmarks under `refs/assets/`), `coreloss.prism` (interpolation 1 to 3, `[crop]`), `bloch.diffraction`, `grid.fastfft_pair`, `consistency.cache_reuse`, `gpu.multigpu_one_worker` (guarded, runs only where dask-cuda and two devices exist).

Legacy mining: a table in `benchmarks/README.md` (mirrored in `legacy/PROVENANCE.md`) mapping each of the 13 abTEM scripts and the run-directory scripts to the case that reproduces it, with a note on what was deliberately not carried over (the tracemalloc profilers, the pool-tracing diagnostic).

Standard tier calibrated on the dev box CPU to 30 to 120 s per case, using the size tables from `benchmark_potential_chunking.py` and `bench_hardening.py` scaled down.

## Deliverables

- PR 1: A/B mode, consistency pairs, presets, graph meter, per-case floors, tests for each.
- PR 2: the cases, the O K asset generator script (`scripts/make_ok_asset.py`, GPAW required, run once), the mining table.
- Report: dev self-check noise floor per case on the dev box (CPU and iGPU), attached to #380.

## Exit criteria

- `run --ref dev --ref dev~20 --tier quick --device cpu` produces a report in one command; `UNSUPPORTED` rows appear where expected and nowhere else.
- Every legacy script has a "reproduced by" or "not needed" entry.
- `phonons.haadf` self-check is bit-identical including `displacements`; changing the seed changes `displacements` and the image; changing `directions` from `xyz` to `xy` leaves x and y displacements unchanged (documented behaviour, DESIGN facts).
- `consistency.cache_reuse` passes on dev and is shown to fail on a checkout before the September cache re-keying (`git bisect`-style check on one old sha), proving it detects the bug class.
- `coreloss.image` `dtype` output equals `complex128` under the accuracy preset and `complex64` under float32-tracking.
- Quick matrix runtime on the dev box CPU under 10 minutes per ref.

## Verification commands

```
python -P -m abtem_bench run --ref dev --ref dev~20 --tier quick --device cpu --preset accuracy --out /workspaces/run/bench-out/ab
python -P -m abtem_bench run --ref dev --ref dev --tier quick --device cpu --preset speed --rounds 3 --out /workspaces/run/bench-out/noise
python -P -m abtem_bench run --ref <pre-cache-rework-sha> --ref dev --tier quick --device cpu --preset accuracy --only consistency.cache_reuse
```

## Risks

- The standard tier may not fit the hosted-runner budget; it is not meant to, and the CI job selects `quick` only.
- The synthetic transition potential has no physics; the report labels it, and the real asset covers the local tiers.
- The multi-GPU case cannot be run before M3 (needs Perlmutter); it ships guarded and is verified in M3.
