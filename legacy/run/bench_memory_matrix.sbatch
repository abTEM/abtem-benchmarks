#!/bin/bash
#SBATCH --job-name=abtem_bench_memory_matrix
#SBATCH --account=m5395_g              # GPU accounts carry the _g suffix
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none                # single task must see all 4 GPUs (step 4)
#SBATCH --exclusive
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

# Runs every pending measurement in one job:
#   1. hardening: quick memory sweep (plan sizes + 16-pos peaks, c64+c128)  ~10 min
#   2. hardening: xl memory pair (ceil vs fast, f64, serial)               ~80 min
#   3. dev:       xl memory pair (ceil expected OOM -> reported; fast ???) ~45 min
#   4. hardening: f64 DISTRIBUTED xl rerun (4 GPUs, post config fix)       ~15 min
# Submit from the directory holding bench_*.py and the abTEM-bench clone.

set -uo pipefail   # no -e: each step may fail independently; we log and go on

HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$HERE"

IMAGE=${IMAGE:-ghcr.io/pzeiger/idrobolab:latest}
ABTEM_SRC="$HERE/abTEM-bench"

run_step() {  # $1 = extra env (or ""), rest = command
    local extra="$1"; shift
    srun --cpu-bind=none --gpu-bind=none \
        podman-hpc run --rm --group-add keep-groups --gpu \
        -v "$CFS:$CFS" -v "$SCRATCH:$SCRATCH" -v "$HOME:$HOME" \
        -e PYTHONPATH="$ABTEM_SRC" \
        -e OMP_NUM_THREADS=16 -e OPENBLAS_NUM_THREADS=16 \
        $extra \
        --workdir "$HERE" \
        "$IMAGE" "$@" || echo "== step failed (rc=$?): $* =="
}

ONE_GPU="-e CUDA_VISIBLE_DEVICES=0"

echo "== preflight =="
git -C "$ABTEM_SRC" fetch origin
git -C "$ABTEM_SRC" checkout multi-gpu-hardening
git -C "$ABTEM_SRC" reset --hard origin/multi-gpu-hardening
git -C "$ABTEM_SRC" log --oneline -1

echo "== step 1: hardening quick memory sweep =="
run_step "$ONE_GPU" python bench_fast_fft_memory.py hardening

echo "== step 2: hardening xl memory pair =="
run_step "$ONE_GPU" python bench_fast_fft_memory.py hardening-xl --xl

echo "== step 3: dev xl memory pair (ceil OOM expected, reported as result) =="
git -C "$ABTEM_SRC" checkout dev
git -C "$ABTEM_SRC" reset --hard origin/dev
git -C "$ABTEM_SRC" log --oneline -1
run_step "$ONE_GPU" python bench_fast_fft_memory.py dev-xl --xl

echo "== step 4: hardening f64 distributed xl (config-fix verification, 4 GPUs) =="
git -C "$ABTEM_SRC" checkout multi-gpu-hardening
run_step "" python bench_hardening_multigpu.py hardening-f64 --xl

echo "== done =="
