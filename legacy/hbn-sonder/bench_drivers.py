"""Which core-loss driver is faster for a given system?

    python bench_drivers.py

Needs an abTEM with PRISM-EELS, so it cannot be a `uv run --script` job: the
PyPI release has no SMatrix.transition_potential_scan. Run it against the dev
checkout, or inside the container the simulations use --

    sbatch gpu_1gpu_shared.sbatch bench_drivers.py

Times multislice against PRISM-EELS over two thicknesses and both channelling
settings, on a small hBN slab that runs on CPU in a few minutes. The point is
the *ratios*, not the absolute times -- see NOTES.md section 6 for what they
mean and for the one way this benchmark understates PRISM.

Edit `gpts`, `interp` and the scan below to model a different system; the
quantity that decides the answer is sites-per-slice, and after that the ratio
of scan positions to plane waves.

The transition potential is synthetic -- see synthetic_tp() -- because timing
depends on its shape, not its values."""
import time
import numpy as np, ase, ase.build, abtem
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import TransitionPotentialArray
abtem.config.set({"device": "cpu"})

if not hasattr(abtem.SMatrix, "transition_potential_scan"):
    raise SystemExit(
        f"this abTEM ({abtem.__version__}, {abtem.__file__}) has no PRISM-EELS "
        "-- SMatrix.transition_potential_scan is missing. Run against a dev "
        "checkout or the project container; the PyPI release will not do."
    )


def synthetic_tp(Z, gpts, extent, energy, n_transitions=4, seed=0):
    """A stand-in transition potential of the right shape.

    This benchmark measures where time goes, and that depends on the array's
    shape -- n_transitions, gpts -- not on its values. Building the real thing
    via SubshellTransitions would pull in GPAW for the atomic wavefunctions,
    which the isolated `uv run --script` environment does not have and which
    would change no timing. abTEM's own core-loss tests use the same trick.
    """
    rng = np.random.default_rng(seed)
    arr = (rng.standard_normal((n_transitions, *gpts))
           + 1j * rng.standard_normal((n_transitions, *gpts))).astype(np.complex64)
    return TransitionPotentialArray(
        Z=Z, array=arr, energy=energy, extent=tuple(extent),
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_transitions)))],
        metadata={"Z": Z, "n": 1, "l": 0},
    )

d = 3.3537
base = ase.build.graphene(a=2.504, vacuum=None); base.symbols = "BN"
base = ase.build.make_supercell(base, np.diag([6, 4, 1]))

def build(n_layers):
    cell = np.diag([2.504*6, 2.504*np.sqrt(3)*4, n_layers*d])
    at = ase.Atoms(cell=cell, pbc=True)
    for i in range(n_layers):
        L = base.copy(); L.positions[:, 2] = i*d + d/2; at += L
    at.cell = cell; at.pbc = True; at.wrap()
    return at

E, alpha, outer, gpts, interp = 60e3, 32.0, 48.0, (192, 192), 2
det = abtem.FlexibleAnnularDetector()
print(f"gpts {gpts}  interpolation {interp}  window {tuple(np.array(gpts)//interp)}")
print(f"{'slices':>6} {'sites':>6} {'dc':>5} {'multislice':>11} {'PRISM':>9} {'speedup':>8}")

for n_layers in (4, 12):
    atoms = build(n_layers)
    pot = abtem.Potential(atoms, gpts=gpts, slice_thickness=d)
    tpa = synthetic_tp(5, tuple(pot.gpts), pot.extent, E)
    probe = abtem.Probe(semiangle_cutoff=alpha, energy=E); probe.grid.match(pot)
    scan = abtem.GridScan(start=(0, 0), end=(12*0.38, 12*0.38), gpts=(12, 12),
                          endpoint=False, potential=pot)
    sm = abtem.SMatrix(potential=pot, energy=E, semiangle_cutoff=alpha,
                       interpolation=interp, downsample=False)
    n_sites = int((atoms.numbers == 5).sum())
    n_beams = len(sm.wave_vectors); n_pos = int(np.prod(scan.gpts))
    for dc in (False, True):
        t0 = time.time()
        probe.transition_potential_scan(
            potential=pot, transition_potentials=tpa, scan=scan, detectors=det,
            sites=None, double_channel=dc, threshold=1.0, lazy=False,
        ).integrate_radial(0, outer).compute()
        t_ms = time.time() - t0
        t0 = time.time()
        sm.transition_potential_scan(
            transition_potentials=tpa, scan=scan, detectors=det, sites=None,
            double_channel=dc, lazy=False,
        ).integrate_radial(0, outer).compute()
        t_pr = time.time() - t0
        print(f"{n_layers:>6} {n_sites:>6} {str(dc):>5} {t_ms:10.1f}s {t_pr:8.1f}s "
              f"{t_ms/t_pr:7.2f}x")
    print(f"        ({n_pos} positions, {n_beams} beams)")
