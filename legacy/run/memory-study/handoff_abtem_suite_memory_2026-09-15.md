# Handover: why does abTEM's test suite accumulate >20 GB, and is any of it a real leak?

**Written:** 2026-09-15. **For:** a fresh Claude session picking up this investigation alone.
**Status:** **DIAGNOSED 2026-09-15** — see `abtem_suite_memory_findings_2026-09-15.md` for the answer and the evidence. In short: the hypothesis below is refuted as stated (accumulation is not GPU/GPAW-specific; the CI-equivalent configuration still accumulates 613 MB over 18 files), but a real unbounded leak of ~40 MB per repetition of identical work does exist, confined to the GPU path and living in native memory invisible to the Python object graph. One step remains before anything is filed upstream: attributing that native growth to abTEM or to cupy/HIP. The experiment design below is kept for the record; the two departures actually made from it are documented in the findings.

A separate defect surfaced while doing this and is written up in `abtem_issues/gpu_tests_run_without_a_device.md`.

## Why this exists

While verifying an abTEM PR, a full-suite run (`pytest test/`, one process, ~50 files) climbed past 20 GB resident on the dev box and the OOM killer took **the user's browser**. The immediate harm is fixed — see *Discipline* below, which is binding, not advisory. What is unresolved is the underlying question: **is the accumulation benign caching, or a real resource leak worth reporting upstream?**

Do not skip to writing a fix. The first deliverable is an answer to that question.

## What is already established (measured, not assumed)

**No single test file is heavy.** Peak RSS, one pytest process per file, on branch `integrator-cache-unification`:

```
test_measure.py           3.90 GB   238 passed
test_ionization.py        1.86 GB    47 passed
test_prism_upsample.py    1.01 GB    78 passed
test_potentials.py        0.81 GB    56 passed
test_atomic_potential.py  0.73 GB    50 passed
...most files             ~0.5 GB
```

So >20 GB is **accumulation across files within one process**, not any one test. Between files, nothing releases: module-level caches, session-scoped fixtures, and (on this box) cupy's memory pool all persist for the life of the process.

**abTEM's own CI does not have this problem.** `.github/workflows/tests.yml:53` runs `uv run pytest test` in a single process and passes on a GitHub runner (~7–16 GB). If the suite genuinely needed 20 GB everywhere, CI would fail constantly.

**Therefore the leading hypothesis:** the accumulation is specific to machines with a GPU and GPAW installed, because those exercise paths CI skips entirely — `test_core_loss.py` skips without GPAW, and cupy paths are inert without a device. This box has both. **This hypothesis is untested.** It is the first thing to check, because if true the upstream story is much narrower than "abTEM leaks memory".

**`pytest-xdist` is installed but does not solve it.** `-n N --dist loadfile` keeps a file's tests on one worker, but each worker still accumulates across every file it receives — you divide accumulation by N while running N of them concurrently. `pytest-forked` (not installed) forks per *test*, which isolates but is heavier than needed. One process per file is the targeted answer; there is no drop-in plugin that does exactly that.

## The experiment to run

1. **Is it GPU/GPAW-specific?** Run the suite in one process three ways, sampling RSS over time: (a) as-is, (b) with the GPU hidden (`CUDA_VISIBLE_DEVICES=""` / `HIP_VISIBLE_DEVICES=""`), (c) with GPAW made unimportable. Compare growth curves. If (b) or (c) flattens it, the CI-vs-here discrepancy is explained and the scope narrows accordingly.
2. **Host or device memory?** cupy's pool holds *device* memory, which does not show in RSS — but pinned host allocations do. Separate the two: sample `cupy.get_default_memory_pool().used_bytes()` / `.total_bytes()` and `cupy.get_default_pinned_memory_pool()` alongside RSS.
3. **What survives between modules?** Between test files in one process, snapshot with `gc.get_objects()` / `tracemalloc` and look for large ndarray-holding structures that persist. Specifically suspect anything module-level: parametrization tables, FFT plan caches (FFTW plans are cached and are not small), and any registry keyed by grid.
4. **Is it a leak or a cache?** The distinction that matters: a bounded cache that plateaus is fine and should be documented; something that grows monotonically with the number of distinct grids/elements seen is a defect. Plot growth against files processed and see whether it plateaus.

## Verdict already reached on one sub-question

**Do not upstream `/workspaces/run/abtem_safe_suite.py`.** It is Linux-only (parses `/proc`), untested, and encodes this machine's assumptions. Maintainers of a scientific library should not inherit a bespoke test runner for a problem their CI does not have. If the diagnosis turns out to be "ordinary fixture and cache retention", the right artefact is a sentence in the contributor docs for people on small machines — not a tool.

If the diagnosis finds a genuine leak, that goes in the issues library (`/workspaces/run/abtem_issues/`, see its `MODEL.md` for the file format) and then upstream as an ordinary bug report.

## The tool that exists

`/workspaces/run/abtem_safe_suite.py <worktree> [--cap 10] [--floor 8] [--log PATH]`

Runs each test file in its own pytest process, with an RSS watchdog that kills a run exceeding `--cap` and a system-memory floor that aborts the whole sweep below `--floor`. Peak stays ~4 GB. Use it for any full-suite verification.

**Caveat, and it matters if you extend this:** the watchdog samples `/proc` every 0.25 s, so it *misses spikes shorter than the interval* — it reported 2.44 GB for a file whose true high-water mark, per `ru_maxrss`, was 3.90 GB. The reported peak now comes from `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`, which is the kernel's exact mark. The **kill decision still samples** and is therefore best-effort: a fast enough allocation can exceed the cap between samples. Treat `--cap` as a backstop against runaway growth, not a hard guarantee.

A cgroup cap via `systemd-run` is not available here (no session bus), and `RLIMIT_AS` was rejected deliberately: it caps *virtual* address space, which cupy reserves heavily, so it false-trips on GPU tests without bounding resident memory.

## Discipline — binding

- **Never** run `pytest test/` over the whole directory in one process on this box.
- One file, or a `-k` selection, at a time. Check `free -g` first; if available is under ~8 GB, wait.
- Never run two suites concurrently. That caused an earlier memory shortage the same day.
- **"Available" during a still-allocating run is runway, not headroom.** Reporting 20 GB available at 65 % through a climbing run is what preceded the browser being killed. Judge against the trajectory.
- Ask before starting any long unattended run that could contend for RAM.

## Environment

- Repo: `/workspaces/code/abTEM`. **Do not `git checkout` there** — another session holds it on branch `tp-graph-nodes`. Make your own worktrees: `git worktree add --detach <path> <commit>`, under the session scratchpad, with a unique name. Do not remove worktrees you did not create.
- Run as `PYTHONPATH=<worktree> python -P`; plain `python` is `/home/ubuntu/venv/bin/python` (uv-managed, no `pip` module — use `uv pip -p`).
- GPU present; cupy is a **ROCm/HIP** build, not CUDA. GPAW is installed.
- `fft="fftw"` is the default and uses FFTW_MEASURE, so it is run-to-run nondeterministic on some grids — pin `abtem.config.set({"fft": "numpy"})` for any bit-identity assertion.
- Box: 46 GB RAM, 24 cores.

## Not in scope

The abTEM defect-fixing work (PRs #386–#393, the issues library, the three PRs being prepared for the PRISM scan-axis squeeze, shared-`Atoms` mutation, and `Potential` equality) stays in the originating session. This handover covers only the suite-memory question.
