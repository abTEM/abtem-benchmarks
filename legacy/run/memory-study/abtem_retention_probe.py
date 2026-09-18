"""pytest plugin: at session end, name what is still holding memory.

The accumulation experiment establishes *that* memory is retained across
modules and that it survives gc.collect() and malloc_trim(0). It does not say
*what* holds it, which is what an upstream report needs to be actionable.

This walks the live object graph once the session is over and attributes
retained bytes to the structures holding them: array totals by flavour, the
largest individual buffers, and the types of the objects referring to them --
a dict of big arrays that nothing else explains is what a grid-keyed cache
looks like from the outside.

Load with `-p abtem_retention_probe`, alongside or instead of the memory probe.
Output goes to $RETENTION_OUT (default: ./retention.txt).
"""

import gc
import os
import sys
from collections import Counter, defaultdict


def _fmt(b):
    return f"{b / 1024**2:.0f} MB"


def _owner_label(obj):
    """A short, stable name for whatever is holding a buffer."""
    t = type(obj)
    name = getattr(t, "__module__", "?") + "." + getattr(t, "__qualname__", t.__name__)
    if t is dict:
        # A bare dict is the interesting case -- it is what a cache looks like.
        # Its referrers say whose cache it is.
        for r in gc.get_referrers(obj):
            if hasattr(r, "__qualname__"):
                return f"dict held by {getattr(r, '__module__', '?')}.{r.__qualname__}"
            if isinstance(r, type):
                return f"dict held by class {r.__module__}.{r.__qualname__}"
        return "dict (holder unidentified)"
    return name


def pytest_sessionfinish(session, exitstatus):
    out_path = os.environ.get("RETENTION_OUT", "retention.txt")
    gc.collect()
    gc.collect()

    np = sys.modules.get("numpy")
    cp = sys.modules.get("cupy")
    if np is None:
        return

    objs = gc.get_objects()
    lines = [f"live objects: {len(objs)}"]

    # Bytes held, counting each distinct buffer once: a view and its base share
    # memory, and summing both would double-count the thing being looked for.
    seen = {}
    for o in objs:
        # isinstance itself can raise here: the list contains dead weakref
        # proxies, whose type check dereferences a gone object.
        try:
            if not isinstance(o, np.ndarray):
                continue
            base = o if o.base is None else o.base
            seen[id(base)] = (getattr(base, "nbytes", 0), o)
        except Exception:
            continue
    total = sum(n for n, _ in seen.values())
    lines.append(f"numpy buffers: {len(seen)} distinct, {_fmt(total)} total")

    if cp is not None:
        try:
            dev = cp.get_default_memory_pool()
            lines.append(f"cupy device pool: {_fmt(dev.total_bytes())} reserved, "
                         f"{_fmt(dev.used_bytes())} in use")
        except Exception:
            pass

    # Who holds the big ones
    by_owner = defaultdict(int)
    counts = Counter()
    big = sorted(seen.values(), key=lambda t: -t[0])[:400]
    for nbytes, arr in big:
        if nbytes < 1024 * 256:
            continue
        for r in gc.get_referrers(arr):
            if r is objs or isinstance(r, list) and len(r) > 10000:
                continue
            by_owner[_owner_label(r)] += nbytes
            counts[_owner_label(r)] += 1
            break

    lines.append("")
    lines.append("retained bytes by holder (buffers > 256 KB):")
    for owner, nbytes in sorted(by_owner.items(), key=lambda kv: -kv[1])[:25]:
        lines.append(f"  {_fmt(nbytes):>10}  x{counts[owner]:<5} {owner}")

    lines.append("")
    lines.append("largest individual buffers:")
    for nbytes, arr in big[:15]:
        lines.append(f"  {_fmt(nbytes):>10}  shape={getattr(arr, 'shape', '?')} "
                     f"dtype={getattr(arr, 'dtype', '?')}")

    # Anything memoized is a cache by construction, so report its size directly
    # rather than inferring it from the graph.
    lines.append("")
    lines.append("memoize-style caches found:")
    for modname, mod in list(sys.modules.items()):
        if not modname.startswith(("abtem", "cupy")):
            continue
        for attr in dir(mod):
            try:
                v = getattr(mod, attr)
            except Exception:
                continue
            c = getattr(v, "cache_info", None)
            if callable(c):
                try:
                    info = c()
                    if getattr(info, "currsize", 0):
                        lines.append(f"  {modname}.{attr}: {info}")
                except Exception:
                    pass

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
