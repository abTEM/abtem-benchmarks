#!/usr/bin/env python3
"""Run the abTEM suite file-by-file with a hard RSS cap and a system-memory floor.

Per-file isolation is what keeps peak low (the suite in one process accumulates
past 20 GB; the heaviest single file is ~3.9 GB). The watchdog is the backstop:
it kills a run that exceeds --cap rather than letting the OOM killer choose a
victim, which last time was the user's browser.

Usage: abtem_safe_suite.py <worktree> [--cap GB] [--floor GB] [--log PATH]
"""
import argparse, os, re, resource, subprocess, sys, threading, time, glob

def rss_gb(pid):
    total = 0
    for p in [pid] + children(pid):
        try:
            with open(f"/proc/{p}/statm") as f:
                total += int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except OSError:
            pass
    return total / 1024**3

def children(pid):
    try:
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
        kids = [int(x) for x in out.split()]
        return kids + [g for k in kids for g in children(k)]
    except Exception:
        return []

def avail_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    return 0.0

def run_file(path, wt, cap):
    env = {**os.environ, "PYTHONPATH": wt}
    proc = subprocess.Popen(["python", "-P", "-m", "pytest", path, "-q", "-p", "no:randomly",
                             "--no-header"], cwd=wt, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak = [0.0]; killed = [False]
    def watch():
        while proc.poll() is None:
            r = rss_gb(proc.pid)
            peak[0] = max(peak[0], r)
            if r > cap:
                killed[0] = True
                proc.kill()
                return
            time.sleep(0.25)
    t = threading.Thread(target=watch, daemon=True); t.start()
    # os.wait4 returns rusage for THIS child alone. RUSAGE_CHILDREN does not:
    # it is a monotonic high-water mark across every child this process has
    # ever reaped, so using it made each file report the running maximum rather
    # than its own peak -- the very error this line replaced (a sampling
    # watchdog that undercounted) traded for a different one.
    out = proc.stdout.read()
    _, status, ru = os.wait4(proc.pid, 0)
    proc.returncode = status
    t.join(timeout=1)
    exact = ru.ru_maxrss / 1048576
    return out, max(exact, peak[0]), killed[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("worktree"); ap.add_argument("--cap", type=float, default=10.0)
    ap.add_argument("--floor", type=float, default=8.0)
    ap.add_argument("--log", default="/tmp/abtem_suite.log")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.worktree, "test", "test_*.py")))
    tot = dict(passed=0, failed=0, skipped=0, error=0)
    worst = 0.0; problems = []
    with open(a.log, "w") as log:
        for f in files:
            if avail_gb() < a.floor:
                print(f"ABORT: {avail_gb():.1f} GB available < floor {a.floor} GB"); return 2
            out, peak, killed = run_file(f, a.worktree, a.cap)
            worst = max(worst, peak)
            last = (out.strip().splitlines() or ["(no output)"])[-1]
            name = os.path.basename(f)
            if killed:
                line = f"{name:34s} KILLED at {peak:.2f} GB (cap {a.cap})"
                problems.append(name)
            else:
                for k in tot:
                    m = re.search(rf"(\d+) {k}", last)
                    if m: tot[k] += int(m.group(1))
                if "failed" in last or "error" in last.lower():
                    problems.append(name)
                line = f"{name:34s} {peak:5.2f} GB  {last[:56]}"
            print(line); log.write(line + "\n"); log.flush()
        summary = (f"TOTAL: {tot['passed']} passed, {tot['failed']} failed, "
                   f"{tot['skipped']} skipped, {tot['error']} error | peak RSS {worst:.2f} GB")
        print(summary); log.write(summary + "\n")
        if problems:
            print("PROBLEM FILES: " + ", ".join(problems)); log.write("PROBLEM FILES: " + ", ".join(problems) + "\n")
    return 1 if problems else 0

if __name__ == "__main__":
    sys.exit(main())
