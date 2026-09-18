# Running `bench_tp_graph_nodes.py` on Perlmutter

Benchmarks the `tp-graph-nodes` branch (transition potentials shipped as one graph node instead of a copy per task) against `dev`. One script, three modes, run once per branch.

The change does not touch the multislice arithmetic — it removes a per-task copy of the transition potential — so the axes that move are graph size, scheduler traffic and worker memory, all proportional to task count. Timing a scan alone would show nothing.

| mode | what it answers | needs a GPU? |
|---|---|---|
| `graph` | how many payload copies the graph carries, and how many bytes distributed would ship | no |
| `cluster` | peak worker memory, and whether a production-scale run survives at all | yes (uses abTEM's own `dask.multi-gpu` cluster on GPU) |
| `compute` | the control: end-to-end wall time must be **flat** and checksums identical | no, but GPU is the interesting one |

Local (CPU, dev box) results for reference: `dev` ships `scan_positions / max_batch` copies — 9.7 GB of task payload for one 512², 32×32 scan — where the branch always ships one (26.6 MB). Peak worker RSS 1293 → 621 MB on a capped 2-worker cluster. Compute wall time and checksums unchanged.

## What Perlmutter adds that the local box could not

1. **A100 + real VRAM.** The `cluster` mode on GPU uses abTEM's `dask.multi-gpu` cluster and reports the per-worker CuPy pool, so the memory claim is measured on the hardware the production runs use.
2. **Production payloads.** The Sonder-scale case (1300²–2048² grids, 54–134 MB transition potentials) is where the extrapolated ~55 GB of dev task payload becomes a real OOM rather than arithmetic.
3. **A genuine failure to point at.** The decisive result is a configuration where `dev` spills, pauses or dies and the branch completes.

## Setup (mirrors the hardening / fast-fft launchers)

The container's baked-in `/opt/src/abTEM` must be shadowed via `PYTHONPATH`. **Check the first line of every run** — the script prints `abtem  <- …` and `commit <- …`; if that path is not the mounted checkout, the numbers are meaningless.

```bash
# from the dev box
scp /workspaces/run/bench_tp_graph_nodes.py \
    perlmutter:/global/cfs/cdirs/m5395/projects/<your-dir>/

# on a Perlmutter login node, in that directory (one clone, switch branches)
git clone https://github.com/abTEM/abTEM.git abTEM-bench
git -C abTEM-bench checkout tp-graph-nodes

salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu \
    --gpus 4 --account m5395
```

On the compute node, from the same directory:

```bash
run_bench() {  # $@ = env assignments then script args, e.g. BENCH_MODE=graph ... label
  srun --cpu-bind=none --gpu-bind=none podman-hpc run --rm --group-add keep-groups --gpu \
      -v $CFS:$CFS -v $SCRATCH:$SCRATCH -v $HOME:$HOME \
      -e PYTHONPATH=$PWD/abTEM-bench \
      -e OMP_NUM_THREADS=16 -e OPENBLAS_NUM_THREADS=16 \
      -e BENCH_MODE -e BENCH_TIER -e BENCH_DEVICE -e BENCH_REPEATS \
      -e BENCH_WORKERS -e BENCH_MEMLIMIT -e BENCH_SYNTHETIC \
      --workdir "$PWD" \
      ghcr.io/pzeiger/idrobolab:latest \
      python bench_tp_graph_nodes.py "$@"
}
```

`-e VAR` with no value forwards it from the calling shell, so `BENCH_MODE=graph run_bench pr3` works.

## Preflight (30 s, do this first)

The payload is the real B K edge, which needs GPAW and its setups. idrobolab has GPAW, but confirm the setups resolve inside the container before spending an allocation — a silent fallback to the synthetic payload would make the checksums non-comparable with any earlier real run:

```bash
BENCH_MODE=graph BENCH_TIER=quick run_bench preflight
```

Check three things in the output: the `abtem  <- …` line points at `abTEM-bench` (not `/opt/src/abTEM`), the `commit <- …` line is the branch you meant, and there is **no** `NOTE: real transition potentials unavailable` line. If that NOTE appears, fix `GPAW_SETUP_PATH` in the container invocation rather than accepting the fallback.

**Confirm the checkout before each pair of runs.** The label you pass is just a string — nothing stops a run labelled `dev` from measuring the branch. In `graph` mode the `pKeys` column gives it away (1 on the branch, `scan_positions / max_batch` on dev), but in `compute` and `cluster` mode there is no such tell, so a mislabelled pair is silently worthless. Either rely on the `commit <- …` line, or run this on the node (outside the container, where git exists) between checkouts:

```bash
git -C abTEM-bench rev-parse --short HEAD && git -C abTEM-bench branch --show-current
```

## The run matrix

Run each line once per branch (`git -C abTEM-bench checkout dev` in between; the label is the only argument).

```bash
# 1. Graph transport -- the headline. No GPU needed, seconds on the branch.
BENCH_MODE=graph BENCH_TIER=production run_bench pr3
BENCH_MODE=graph BENCH_TIER=production run_bench dev     # see timing note below

# 2. Multi-GPU execution -- the decisive one.
BENCH_MODE=cluster BENCH_TIER=production BENCH_DEVICE=gpu run_bench pr3
BENCH_MODE=cluster BENCH_TIER=production BENCH_DEVICE=gpu run_bench dev

# 3. Control -- must be flat, checksums must match across branches.
BENCH_MODE=compute BENCH_TIER=standard BENCH_DEVICE=gpu BENCH_REPEATS=3 run_bench pr3
BENCH_MODE=compute BENCH_TIER=standard BENCH_DEVICE=gpu BENCH_REPEATS=3 run_bench dev
```

Tiers: `quick` = 256² (smoke), `standard` = 512²/1024², `production` = 1024²/2048².

## What to expect

- **Graph mode, branch**: `pKeys = 1` on every row, `graph MB` ≈ the payload plus a few MB. Seconds.
- **Graph mode, dev**: `pKeys` equals `scan_positions / max_batch` (1024 for the production rows) and `graph MB` runs to tens of GB. **This mode is slow on dev by construction** — weighing the 2048² row means serializing ~137 GB, so allow a few minutes. That slowness is the finding, not a hang.
- **Cluster mode on GPU**: the script prints the number of cluster workers (should be 4) and the per-worker CuPy pool. Expect dev to show a much larger pool, and at the 2048² row to spill, pause or raise `OutOfMemoryError` — **an OOM on dev there is the successful result**, exactly as in the `--xl` hardening reproduction. The branch should complete with a modest pool.
- **Compute mode**: times within noise of each other and **bit-identical checksums** between branches. A difference here would mean the change is not numerically inert, which locally it is.

## The payload, and the synthetic fallback

The payload is the real B K edge (hBN, 60 kV, 32 mrad, ε = 10 eV — the Sonder production configuration) built through `SubshellTransitions`: 4 transitions, complex64, 2–5 s to build at 256²–2048². idrobolab ships GPAW, so this is the path you will take; the preflight above confirms it.

The fallback exists only as insurance. If GPAW or its setups cannot be reached, the script prints a NOTE and substitutes a payload of exactly the same shape and dtype — graph transport and memory depend on the payload's size, not its values, and the two give identical graph figures (verified locally). `BENCH_SYNTHETIC=1` forces it. Checksums from a synthetic run are comparable only to other synthetic runs, so if you ever use it, use it on **both** branches.

## Time budget

Graph mode is minutes (dominated by dev's serialization). Compute and cluster modes are the real cost: the `standard` compute rows took ~35 s each for a 256² 16×16 scan on one CPU core locally, so budget generously on the production tier and start with `BENCH_TIER=standard` to calibrate before committing to `production`. Two hours of an interactive 4-GPU allocation should cover the whole matrix; if it does not, the priority order is 2 → 1 → 3.
