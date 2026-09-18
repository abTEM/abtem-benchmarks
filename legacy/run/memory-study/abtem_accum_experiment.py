#!/usr/bin/env python3
"""Is abTEM's in-process memory accumulation specific to GPU/GPAW paths?

The open question from the suite-memory handover: a full `pytest test/` in one
process climbs past 20 GB here, while abTEM's CI runs the same command on a
7-16 GB runner and passes. The leading hypothesis is that the difference is
this box having both a GPU and GPAW, exercising paths CI skips entirely.

This runs a FIXED subset of files in ONE process under three conditions --
baseline, GPU hidden, GPAW unimportable -- and compares the growth curves.

Two deliberate departures from the experiment as originally sketched, both of
which make it cheaper and safer without weakening the answer:

  It uses a subset, not the whole suite. Accumulation is a slope. If it is
  real, ~18 files show it as clearly as 53 do, at a third of the runtime and a
  fraction of the peak. Whether the slope flattens when the GPU is hidden does
  not require reaching 20 GB to observe -- and reaching 20 GB on this box is
  what killed the browser.

  It measures the floor between modules, not peak RSS. Peak conflates one
  module's transient with everything retained before it; the floor after a
  gc.collect() is what was actually kept. The floor is also what distinguishes
  the two outcomes that matter: a bounded cache plateaus, a leak does not.

`test_measure.py` is excluded by default -- at 3.9 GB it is a transient outlier
that dominates the peak while saying little about accumulation.

Safety: an RSS watchdog kills a condition that exceeds --cap, and the sweep
aborts if system MemAvailable falls below --floor. As in abtem_safe_suite.py
the kill decision samples /proc and so is best-effort against a fast enough
allocation; --cap is a backstop, not a guarantee.

Usage:
  abtem_accum_experiment.py <worktree> [--conditions base,nogpu,nogpaw]
                            [--cap 12] [--floor 10] [--outdir DIR] [--files a.py,b.py]
  abtem_accum_experiment.py --analyze <outdir>      # re-read results, no run
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time

PROBE_DIR = "/workspaces/run"

# Ordered, and identical across conditions -- the comparison is only valid if
# every condition sees the same modules in the same order.
DEFAULT_FILES = [
    "test/test_import.py",
    "test/test_grid.py",
    "test/test_axes.py",
    "test/test_distributions.py",
    "test/test_ensemble.py",
    "test/test_array.py",
    "test/test_fft.py",
    "test/test_potentials.py",
    "test/test_atomic_potential.py",
    "test/test_potential_chunking.py",
    "test/test_waves.py",
    "test/test_prism.py",
    "test/test_prism_upsample.py",
    "test/test_detect.py",
    "test/test_ionization.py",
    "test/test_core_loss.py",
    "test/test_gpaw.py",
    "test/test_multigpu_logic.py",
]

# Blocking the *import* is what mirrors CI, and hiding the device is not a
# substitute for it. abTEM keys GPU availability on cupy being importable, not
# on a device being present, so HIP_VISIBLE_DEVICES="" leaves every
# device="gpu" test collected and running -- they then die in xp.array() with
# no device rather than skipping. Measured, not assumed: that is exactly how
# the first attempt at this experiment aborted, in test_array.py.
CONDITIONS = {
    "base": {},
    "nocupy": {"MEM_PROBE_BLOCK": "cupy"},
    "nogpaw": {"MEM_PROBE_BLOCK": "gpaw"},
    "neither": {"MEM_PROBE_BLOCK": "cupy,gpaw"},
    # Same as base, but records the floor either side of malloc_trim(0), to
    # tell real retention from the allocator simply not giving pages back.
    "base_trim": {"MEM_PROBE_TRIM": "1"},
}


def children(pid):
    try:
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
        kids = [int(x) for x in out.split()]
        return kids + [g for k in kids for g in children(k)]
    except Exception:
        return []


def rss_gb(pid):
    total = 0
    for p in [pid] + children(pid):
        try:
            with open(f"/proc/{p}/statm") as f:
                total += int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except OSError:
            pass
    return total / 1024**3


def avail_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return 0.0


def run_condition(name, worktree, files, cap, floor, outdir):
    jsonl = os.path.join(outdir, f"{name}.jsonl")
    env = {
        **os.environ,
        "PYTHONPATH": f"{worktree}:{PROBE_DIR}",
        "MEM_PROBE_OUT": jsonl,
        **CONDITIONS[name],
    }
    cmd = [sys.executable, "-P", "-m", "pytest", *files,
           "-q", "--no-header", "-p", "no:randomly", "-p", "abtem_mem_probe"]
    if CONDITIONS[name].get("MEM_PROBE_BLOCK"):
        # Blocking gpaw makes ASE's entry-point registration warn, and the
        # suite promotes warnings to errors, so the run dies at collection.
        # This has to be a pytest flag: an ini line added from pytest_configure
        # lands too late for a warning raised during collection. The warning is
        # an artefact of blocking a module that is installed -- on a machine
        # without gpaw there is no entry point and no warning.
        cmd += ["-W", "ignore:Failed to register external IO format:UserWarning"]

    print(f"\n=== condition: {name} ===")
    print(f"    {' '.join(f'{k}={v!r}' for k, v in CONDITIONS[name].items()) or '(unmodified)'}")
    print(f"    avail before: {avail_gb():.1f} GB")

    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    proc = subprocess.Popen(cmd, cwd=worktree, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    state = {"peak": 0.0, "killed": None}

    def watch():
        while proc.poll() is None:
            r = rss_gb(proc.pid)
            state["peak"] = max(state["peak"], r)
            if r > cap:
                state["killed"] = f"RSS {r:.1f} GB over cap {cap} GB"
                proc.kill()
                return
            if avail_gb() < floor:
                state["killed"] = f"system available under floor {floor} GB"
                proc.kill()
                return
            time.sleep(0.25)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    out, _ = proc.communicate()
    t.join(timeout=1)

    # ru_maxrss is the kernel's exact high-water mark; the sampler above misses
    # spikes shorter than its interval.
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    exact_peak = max(after, before) / 1024**2

    tail = out.strip().splitlines()[-4:] if out else []
    print("    " + "\n    ".join(tail))
    print(f"    peak (sampled): {state['peak']:.2f} GB   peak (ru_maxrss): {exact_peak:.2f} GB")
    if state["killed"]:
        print(f"    !! KILLED: {state['killed']}")
    return {"condition": name, "killed": state["killed"], "peak_gb": exact_peak, "jsonl": jsonl}


def boundaries(jsonl):
    rows = []
    try:
        with open(jsonl) as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") == "boundary":
                    rows.append(r)
    except OSError:
        pass
    return rows


def analyze(outdir):
    print("\n" + "=" * 78)
    print("ACCUMULATION CURVE -- resident MB after gc.collect(), at each module boundary")
    print("=" * 78)

    per_condition = {}
    for name in CONDITIONS:
        rows = boundaries(os.path.join(outdir, f"{name}.jsonl"))
        if rows:
            per_condition[name] = rows

    if not per_condition:
        print("no results found in", outdir)
        return

    names = list(per_condition)
    width = max(len(os.path.basename(r["module"])) for rows in per_condition.values() for r in rows)
    header = "module".ljust(width) + "".join(f"{n:>14}" for n in names)
    print(header)
    print("-" * len(header))

    modules = []
    for rows in per_condition.values():
        for r in rows:
            m = r["module"]
            if m not in modules:
                modules.append(m)

    for m in modules:
        line = os.path.basename(m).ljust(width)
        for n in names:
            hit = next((r for r in per_condition[n] if r["module"] == m), None)
            line += f"{hit['rss_mb']:>13.0f}" + " " if hit else f"{'-':>14}"
        print(line)

    print()
    for n in names:
        rows = per_condition[n]
        first, last = rows[0]["rss_mb"], rows[-1]["rss_mb"]
        growth = last - first
        per_module = growth / max(len(rows) - 1, 1)
        # A bounded cache stops growing; a leak keeps going. Compare the growth
        # over the back half against the front half.
        mid = len(rows) // 2
        front = rows[mid]["rss_mb"] - rows[0]["rss_mb"]
        back = rows[-1]["rss_mb"] - rows[mid]["rss_mb"]
        shape = "plateauing" if back < 0.5 * front else "still climbing"
        # A condition that stopped early has a shorter curve, not a flatter one.
        # Saying so here is the difference between a result and a wrong result.
        n_full = max(len(r) for r in per_condition.values())
        incomplete = "  << INCOMPLETE, not comparable" if len(rows) < n_full else ""
        print(f"{n:>8}: {first:.0f} -> {last:.0f} MB over {len(rows)} modules "
              f"(+{growth:.0f} MB, {per_module:.0f} MB/module, {shape}; "
              f"front half +{front:.0f}, back half +{back:.0f}){incomplete}")

        cupy_rows = [r for r in rows if r.get("cupy")]
        if cupy_rows:
            c_first, c_last = cupy_rows[0]["cupy"], cupy_rows[-1]["cupy"]
            print(f"{'':>8}  cupy device pool: {c_first['dev_total_mb']:.0f} -> "
                  f"{c_last['dev_total_mb']:.0f} MB reserved "
                  f"({c_last['dev_used_mb']:.0f} MB in use at end)")

    print("\nReading this (only for conditions that ran to the SAME final module --")
    print("a short run is not a flat curve):")
    print("  base climbs, nocupy/nogpaw flat   -> hypothesis holds, scope is narrow")
    print("  all climb alike                   -> not GPU/GPAW; CI differs for another reason")
    print("  all plateau                       -> bounded caches; document, do not report")
    print("  cupy device pool grows, RSS flat  -> device-side, invisible to the OOM killer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("worktree", nargs="?")
    ap.add_argument("--conditions", default="base,nocupy,nogpaw,neither")
    ap.add_argument("--cap", type=float, default=8.0, help="kill a run over this RSS (GB)")
    ap.add_argument("--floor", type=float, default=6.0, help="abort if system available drops below (GB)")
    ap.add_argument("--outdir", default="/workspaces/run/abtem_accum_results")
    ap.add_argument("--files", default=None, help="comma-separated, overrides the default subset")
    ap.add_argument("--analyze", metavar="OUTDIR", default=None, help="analyze existing results and exit")
    args = ap.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return 0
    if not args.worktree:
        ap.error("worktree is required unless --analyze is given")

    worktree = os.path.abspath(args.worktree)
    files = args.files.split(",") if args.files else DEFAULT_FILES
    missing = [f for f in files if not os.path.exists(os.path.join(worktree, f))]
    if missing:
        print("missing in worktree:", ", ".join(missing), file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    print(f"worktree : {worktree}")
    print(f"files    : {len(files)}")
    print(f"cap      : {args.cap} GB per run, abort below {args.floor} GB available")
    print(f"outdir   : {args.outdir}")

    results = []
    for name in args.conditions.split(","):
        if name not in CONDITIONS:
            print(f"unknown condition {name!r}", file=sys.stderr)
            return 2
        if avail_gb() < args.floor:
            print(f"\nstopping: only {avail_gb():.1f} GB available, floor is {args.floor} GB")
            break
        results.append(run_condition(name, worktree, files, args.cap, args.floor, args.outdir))

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    analyze(args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
