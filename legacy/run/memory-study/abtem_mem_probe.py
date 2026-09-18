"""pytest plugin: record the memory accumulation curve across a run.

Load with `-p abtem_mem_probe` (its directory on PYTHONPATH). It answers the
question the per-file runner cannot: within ONE process, does resident memory
between test modules *plateau* (a bounded cache, fine) or climb monotonically
with the number of modules seen (a leak)?

What it records, as JSONL to $MEM_PROBE_OUT:

  per test      cheap VmRSS sample, tagged with its module
  per boundary  when the module changes: gc.collect() first, then VmRSS plus
                cupy's device and pinned pools

The boundary rows are the signal. Peak RSS during a module is that module's
transient; the *floor between* modules is what accumulated and was never
released. Comparing floors is what distinguishes a cache from a leak.

Two things it deliberately does not do:

  It never imports cupy. Importing it would initialise a device and allocate,
  changing the thing being measured; pools are read only if something under
  test has already imported it.

  It does not call gc.collect() per test -- only at boundaries. Per-test
  collection would both slow the run and mask the retention being looked for.

Env:
  MEM_PROBE_OUT    JSONL output path (default: ./mem_probe.jsonl)
  MEM_PROBE_BLOCK  comma-separated modules to make unimportable, e.g. "gpaw".
                   Used to test whether accumulation is specific to the paths
                   that CI skips. Blocking happens at plugin import, which is
                   before conftest, so importorskip sees it.
"""

import ctypes
import ctypes.util
import gc
import json
import os
import sys
import time

_out = None
_current_module = None
_t0 = time.monotonic()


# --- making a module unimportable -------------------------------------------


class _BlockFinder:
    """meta_path finder that refuses a module and everything under it.

    Raising from find_spec (rather than returning None) is deliberate: it stops
    the search outright, so a module that is genuinely installed still looks
    absent to `import gpaw`, `importlib.util.find_spec` and
    `pytest.importorskip` alike.
    """

    def __init__(self, names):
        self.names = tuple(names)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.names:
            raise ModuleNotFoundError(
                f"No module named {fullname!r} (blocked by MEM_PROBE_BLOCK)",
                name=fullname,
            )
        return None


_blocked = [n.strip() for n in os.environ.get("MEM_PROBE_BLOCK", "").split(",") if n.strip()]
if _blocked:
    for _name in list(sys.modules):
        if _name.split(".")[0] in _blocked:
            del sys.modules[_name]
    sys.meta_path.insert(0, _BlockFinder(_blocked))


# --- sampling ----------------------------------------------------------------


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _cupy_pools():
    """Device/pinned pool bytes, or None if cupy was never imported.

    Device memory does not appear in RSS, so a run whose RSS is flat may still
    be accumulating on the GPU -- and pinned host memory *does* show in RSS and
    is easy to mistake for a host-side leak. Both are needed to tell them apart.
    """
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return None
    try:
        dev = cupy.get_default_memory_pool()
        pin = cupy.get_default_pinned_memory_pool()
        return {
            "dev_used_mb": dev.used_bytes() / 1024**2,
            "dev_total_mb": dev.total_bytes() / 1024**2,
            "pinned_free_blocks": pin.n_free_blocks(),
        }
    except Exception:
        return None


def _write(row):
    if _out is None:
        return
    row["t"] = round(time.monotonic() - _t0, 2)
    _out.write(json.dumps(row) + "\n")
    _out.flush()


_TRIM = os.environ.get("MEM_PROBE_TRIM") == "1"
_libc = None
if _TRIM:
    try:
        _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        _libc.malloc_trim.argtypes = [ctypes.c_size_t]
        _libc.malloc_trim.restype = ctypes.c_int
    except Exception:
        _libc = None


def _boundary(module, when):
    """Record the floor: what the finished module left behind after a full gc.

    With MEM_PROBE_TRIM=1 the floor is recorded twice, either side of
    malloc_trim(0). That separates the two things a rising RSS can mean:

      memory still referenced by something          -> both values stay high
      memory freed but not returned by the allocator -> only the untrimmed one

    The second is glibc holding freed arenas, which looks exactly like a leak
    in RSS and in the OOM killer's accounting, but is not one. The gap between
    the two series is how much of the climb is that.
    """
    gc.collect()
    row = {
        "kind": "boundary",
        "when": when,
        "module": module,
        "rss_mb": round(_rss_mb(), 1),
        "cupy": _cupy_pools(),
        "n_objects": len(gc.get_objects()),
    }
    if _libc is not None:
        _libc.malloc_trim(0)
        row["rss_trimmed_mb"] = round(_rss_mb(), 1)
    _write(row)


# --- hooks -------------------------------------------------------------------


def pytest_configure(config):
    global _out
    path = os.environ.get("MEM_PROBE_OUT", "mem_probe.jsonl")
    _out = open(path, "w")
    if _blocked:
        config.addinivalue_line(
            "filterwarnings", "ignore:Failed to register external IO format:UserWarning")
    _write({
        "kind": "start",
        "rss_mb": round(_rss_mb(), 1),
        "blocked": _blocked,
        "python": sys.version.split()[0],
    })


def pytest_runtest_logfinish(nodeid, location):
    global _current_module
    module = nodeid.split("::")[0]
    if _current_module is not None and module != _current_module:
        # First test of a new module has finished, so the previous module's
        # fixtures have been torn down. This is the earliest honest floor.
        _boundary(_current_module, "after")
    _current_module = module
    _write({"kind": "test", "module": module, "rss_mb": round(_rss_mb(), 1)})


def pytest_sessionfinish(session, exitstatus):
    if _current_module is not None:
        _boundary(_current_module, "final")
    _write({"kind": "end", "rss_mb": round(_rss_mb(), 1), "exitstatus": int(exitstatus)})
    if _out is not None:
        _out.close()
