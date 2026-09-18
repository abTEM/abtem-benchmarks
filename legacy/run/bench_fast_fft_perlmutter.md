# Running `bench_fast_fft.py` on Perlmutter

Benchmarks the `fast-fft-grid` branch (`grid.round-to-fast-fft`) on an A100:
FFT/HAADF **timing** (CPU + GPU) and the **GPU memory** footprint retained by
cached cuFFT plans — the number that substantiates the PR's memory claim,
which could not be validated on the local ROCm iGPU (rocFFT's buffer strategy
dominates there; the small-grid iGPU timings also wobble run-to-run, so the
A100 GPU column is the citable one).

## What the memory section measures

Per shape (ceil vs fast, small vs large) it clears the cuFFT plan cache and
the CuPy pool, runs one `fft2 + ifft2` round-trip, and reports pool bytes
retained versus array bytes. The excess is workspace held by the cached FFT
plans; on cuFFT the Bluestein (ceil) shapes are expected to retain
power-of-two-padded workspaces several × the transform size, the fast shapes
far less. It forces `cupy.fft-cache-size: -1` during the measurement so the
result is independent of the plan-cache default in the abTEM in use.

## Critical: mount the branch over the container's abTEM

The container's baked-in `/opt/src/abTEM` predates the feature — there the
config knob is silently ignored and "fast" would equal "ceil". The script's
preflight hard-aborts in that case (it verifies the knob actually changes
gpts and prints `abtem in use: … <- path`, same discrimination as the
multigpu verification launchers). The `fast-fft-grid` checkout must be first
on `PYTHONPATH`.

## Commands

```bash
# from the dev box
scp /workspaces/run/bench_fast_fft.py \
    perlmutter:/global/cfs/cdirs/m5395/projects/DanDan/2026_tWSe2_Cao_Strat/

# on a Perlmutter login node, in that directory
git clone --depth 1 --branch fast-fft-grid \
    https://github.com/abTEM/abTEM.git abTEM-fast-fft-grid

# one GPU is enough -- no multi-GPU involved
salloc --nodes 1 --qos interactive --time 00:30:00 --constraint gpu \
    --gpus 1 --account m5395

# on the compute node, from the same directory
ABTEM_SRC=$PWD/abTEM-fast-fft-grid
srun --cpu-bind=none podman-hpc run --rm --group-add keep-groups --gpu \
    -v $CFS:$CFS -v $SCRATCH:$SCRATCH -v $HOME:$HOME \
    -e PYTHONPATH=$ABTEM_SRC \
    -e OMP_NUM_THREADS=16 -e OPENBLAS_NUM_THREADS=16 \
    --workdir "$PWD" \
    ghcr.io/pzeiger/idrobolab:latest \
    python bench_fast_fft.py
```

## Notes

- Runtime ~5–10 minutes.
- `python bench_fast_fft.py gpu` skips the CPU rows (use this for A100-only
  numbers, or if pyfftw turns out to be absent in the container).
- The first output line must read `abtem in use: … <- $ABTEM_SRC/...`; the
  preflight exits with a clear message otherwise.
- The script suppresses warnings; without that, the ceil GPU shapes would
  fire the new Bluestein `UserWarning` from the shared commit — the feature
  announcing itself.

## Afterwards

Swap the GPU timing rows in `abtem_fast_fft_grid_pr.md` for the Perlmutter
numbers and replace the memory caveat with the measured workspace ratios —
that removes both caveats currently written into the PR's benchmark section.
