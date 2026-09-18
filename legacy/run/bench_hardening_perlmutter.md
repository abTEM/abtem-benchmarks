# Benchmarking `multi-gpu-hardening` on Perlmutter

Two scripts, run once per branch (`dev` = baseline, `multi-gpu-hardening`):

- **`bench_hardening.py <label>`** — single-GPU comparison table (small/large,
  fast/Bluestein grids, scan + to_zarr, CPU + GPU). Same script as the local
  run; A100/cuFFT numbers replace the noisy iGPU column.
- **`bench_hardening_multigpu.py <label>`** — the headline case: a
  to_zarr-terminated 256-position HAADF scan on the 2623×2271 production grid
  with `dask.multi-gpu: true`. On `dev` the flag is ignored on this path
  (serial on GPU 0, prints `cluster workers: 0`); on the branch it distributes
  (prints `cluster workers: 4`). Sized to complete serially in one 40 GiB
  A100, so dev finishes and a speedup ratio comes out. Checksums must match
  between branches.

## What the large-scale run answers that the local box could not

1. **The to_zarr distribution fix** — expect ~4× on 4 GPUs (the compute path
   measured 3.90× on 2026-08-19; the save path now inherits it).
2. **The 12× Bluestein batch factor** — 256 positions exceed one auto-batch on
   a 40 GiB card, so the halved batches actually engage.
3. **Open question: the 2 GB plan-cache bound on cuFFT.** If a Bluestein
   plan's workspace exceeds 2 GB, the bound forces replanning every batch —
   watch the `large-slow gpu scan` row: if the branch is *slower* than dev
   there, the `cupy.fft-cache-size` default needs raising (e.g. "4 GB") before
   the PR merges. ROCm could not answer this.

## Setup (mirrors the multigpu verification launchers)

The container's `/opt/src/abTEM` must be shadowed via `PYTHONPATH`; check the
first output line (`abtem <- ...`) points at the mounted checkout.

```bash
# from the dev box
scp /workspaces/run/bench_hardening.py /workspaces/run/bench_hardening_multigpu.py \
    perlmutter:/global/cfs/cdirs/m5395/projects/DanDan/2026_tWSe2_Cao_Strat/

# on a Perlmutter login node, in that directory (one clone, switch branches)
git clone https://github.com/abTEM/abTEM.git abTEM-bench
git -C abTEM-bench checkout multi-gpu-hardening

salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu \
    --gpus 4 --account m5395
```

On the compute node, from the same directory (`--gpu-bind` defaults are fine
under `salloc --gpus 4`; all 4 devices reach the single task):

```bash
run_bench() {  # $1 = script, $2 = label
  srun --cpu-bind=none --gpu-bind=none podman-hpc run --rm --group-add keep-groups --gpu \
      -v $CFS:$CFS -v $SCRATCH:$SCRATCH -v $HOME:$HOME \
      -e PYTHONPATH=$PWD/abTEM-bench \
      -e OMP_NUM_THREADS=16 -e OPENBLAS_NUM_THREADS=16 \
      --workdir "$PWD" \
      ghcr.io/pzeiger/idrobolab:latest \
      python "$@"
}

git -C abTEM-bench checkout multi-gpu-hardening
run_bench bench_hardening.py hardening
run_bench bench_hardening_multigpu.py hardening

git -C abTEM-bench checkout dev
run_bench bench_hardening.py dev
run_bench bench_hardening_multigpu.py dev     # serial by design -- allow ~5-10 min
```

Follow-ups after the first 4-GPU results (2026-08-22):

- `bench_hardening_multigpu.py` now times TWO passes: "cold" (includes
  cluster bring-up, per-worker imports, JIT, FFT plan building) and "warm"
  (steady state). The warm number is the one comparable to the validated
  3.90×/4 GPUs compute-path scaling; only rerun the hardening side if time is
  short (dev is serial on both passes).
- Plan-cache A/B for the reproducible +8 % on the large fast-grid GPU row:
  run `bench_hardening.py hardening` once more with `-e BENCH_FFT_CACHE=-1`
  added to the podman line (lifts the 2 GB bound). If the row drops back to
  the dev time, the bound is the cause and the default should rise to ~4 GB.

## XL: reproducing the original production failure (`--xl`)

`bench_hardening_multigpu.py <label> --xl` recreates the exact failure mode
that started this work: **float64**, the Bluestein 2623×2271 grid, and the
full Nyquist-sampled fractional scan (343×297 = 101,871 positions),
terminated in `to_zarr` with `dask.multi-gpu: true`. Flag-gated so it never
runs by default — do NOT run it on a local machine.

```bash
run_bench bench_hardening_multigpu.py dev --xl          # expected: OOM
run_bench bench_hardening_multigpu.py hardening --xl    # expected: completes
```

Expected outcomes:

- **dev**: serial on GPU 0 with the unbounded plan cache, 6× batch estimate
  and doubled input copy — expected to die with
  `cupy.cuda.memory.OutOfMemoryError` within minutes, as the production run
  did (pool ~38.5 GiB of 40 GiB). An OOM here is the *successful* result.
- **hardening**: distributes over 4 workers and completes — this is also the
  first measurement where the 12× Bluestein batch factor actually engages
  (101,871 positions ≫ one auto-batch). Expect very roughly ~15 min on
  4×A100 (single pass; there is no warm pass in xl mode). The final
  `worker pools after run [GB]` line shows the per-GPU footprint — all four
  should sit well below 40 GB.
- Checksums are float64 here, so they will differ from the float32 cases —
  but must match between any two runs that both complete.

The **serial (1-GPU) variant** completes the 2×2 matrix (serial/distributed ×
dev/hardening): define `run_bench_1gpu` as `run_bench` plus
`-e CUDA_VISIBLE_DEVICES=0` on the podman line, then

```bash
run_bench_1gpu bench_hardening_multigpu.py hardening-serial --xl  # sizing fixes standalone test
run_bench_1gpu bench_hardening_multigpu.py dev-serial --xl        # clean-labeled OOM baseline
```

Header markers: `visible GPUs: 1` + the multi-GPU-declined warning. dev is
expected to OOM (~1 min); hardening completing (~15–20 min, workers: 0) would
demonstrate the VRAM-sizing fixes independently of distribution; hardening
OOMing would attribute serial survival to distribution alone (both outcomes
PR-relevant).

The **fair-comparison** variant triggers computation via `.compute()` before
saving — the path on which dev DOES bootstrap the cluster:

```bash
run_bench bench_hardening_multigpu.py dev-computefirst --xl --compute-first
```

This isolates the VRAM-sizing fixes (12× Bluestein batches, removed pristine
copy, bounded plan cache) from the save-path bootstrap fix: dev's workers get
6×-sized 30-probe (2.9 GB) batches, BOTH multislice copies, and an unbounded
plan cache. Expected: per-worker OOM even distributed — but either outcome is
informative (a completion would mean compute-then-save was a viable dev
workaround at this scale, and the sizing fixes are about margin). The header
must read `trigger: compute()+to_zarr`.

## Notes

- `bench_hardening_multigpu.py` has the mandatory `if __name__ == "__main__"`
  guard (dask-cuda spawns workers); keep it if you modify the script.
- Writes/removes a scratch zarr store in the working directory — run from
  `$CFS`/`$SCRATCH`, not `$HOME`.
- Checksums must be identical across branches per case; if not, stop and
  investigate before quoting any timing.
- Afterwards: put the A100 table and the multi-GPU ratio into
  `abtem_multi_gpu_hardening_pr.md` (replacing the iGPU table's caveats), and
  resolve the plan-cache question one way or the other in the PR text.
