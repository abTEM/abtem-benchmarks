# Excerpt: 2026_hBN_defects_Sonder/NOTES.md, section 6 (PRISM-EELS and BiP-PRISM)

Copied verbatim on 2026-09-18 from git@github.com:pzeiger/2026_hBN_defects_Sonder.git @ 1cdeca7c. Calibrated PRISM-EELS / BiP-PRISM knowledge (beam counts vs interpolation, edge delocalization radii, retention vs PRISM cell, multislice-vs-PRISM driver timings). Source of truth stays in that repo.

## 6. PRISM-EELS and BiP-PRISM

`SMatrix.transition_potential_scan` (on `dev`) replaces one multislice per probe position with one per plane wave. Beam counts for this cell at 32 mrad / 60 kV:

| interpolation | beams | PRISM cell [Å] |
|---|---|---|
| 1 | 6453 | 65.3 × 71.1 |
| 2 | 1641 | 32.7 × 35.5 |
| 3 | 745 | 21.8 × 23.7 |
| 4 | 431 | 16.3 × 17.8 |
| 6 | 207 | 10.9 × 11.9 |
| 8 | 115 | 8.2 × 8.9 |

against ~9800 probe positions at Nyquist.

**The PRISM cell also caps the transition-potential window**, and the B K edge at 60 kV is delocalized. Cumulative |H_n0|² within radius r, band-limited to 48 mrad, ε = 10 eV:

| fraction | B K | N K |
|---|---|---|
| 50 % | 1.27 Å | 0.75 Å |
| 90 % | 4.57 Å | 2.45 Å |
| 95 % | 6.09 Å | 3.16 Å |
| 99 % | 9.89 Å | 5.04 Å |

Measured on a 4-layer hBN test cell (192², 60 kV, 32 mrad, 0–48 mrad, single channel), integrated B K signal relative to the multislice reference:

| PRISM cell | retained | max &#124;ΔI&#124; / max(I) |
|---|---|---|
| 15.0 × 17.3 Å (f=1) | 1.0000 | 1.6e-06 |
| 7.5 × 8.7 Å (f=2) | 0.9289 | 6.3e-02 |
| 5.0 × 5.8 Å (f=3) | 0.8220 | 1.7e-01 |
| 3.8 × 4.3 Å (f=4) | 0.7319 | 2.2e-01 |
| 2.5 × 2.9 Å (f=6) | 0.5619 | 4.0e-01 |

`f=1` reproduces multislice to 1.6e-6 of the peak, so the translation itself is exact; everything below that line is window truncation. On the real 65 × 71 Å cell, keeping the loss near 1 % means **interpolation ≤ 3** (21.8 × 23.7 Å window). Do not push to 6 or 8 for absolute intensities.

**BiP-PRISM (PR #338, branch `bip-prism-eels`, not on `dev`)** removes that trade. It holds `interpolation = 1` — the PRISM cell is the whole supercell, so nothing is truncated — and sparsifies the *beams* instead, building S1 (and S2) on hex-ring parent columns reconstructed per ionized atom. The parent count depends only on `partitions_*` and `n_angular`, not on the aperture or cell size:

| `partitions_s1` | parents | S1 columns at f=1 |
|---|---|---|
| 3 | 37 | 6453 → 37 (174×) |
| 4 | 61 | 6453 → 61 (106×) |
| 6 | 127 | 6453 → 127 (51×) |
| 8 | 217 | 6453 → 217 (30×) |

Verified against the multislice reference on a 4-layer hBN cell (10.0 × 13.0 Å, 96², 60 kV, 32 mrad, 0–48 mrad), max |ΔI| / max(I), all with integrated signal 1.0000× the reference:

| reduction | `double_channel=False` | `double_channel=True` |
|---|---|---|
| PRISM `real_space`, f=1 | 8.2e-07 | 8.1e-07 |
| `beam_basis` exact | 7.1e-07 | 8.1e-07 |
| BiP `partitions_s1=3` | 1.0e-03 | 9.9e-04 |
| BiP `partitions_s1=6` | 2.1e-04 | 2.1e-04 |
| BiP `s1=6, s2=6` | — | 1.9e-04 |

So `partitions_s1=6` costs ~2e-4 of the peak and nothing measurable in absolute scale. Compare like for like on the real cell: 127 parent columns is about the beam count plain PRISM reaches at `interpolation = 8` (115 beams), where the PRISM cell is 8.2 × 8.9 Å and the measured retention was 0.93. Same number of columns, ~7 % of the B K signal versus 0.02 %. That is the whole case for this driver.

`collection_angle=48` restricts S2 to the detector disk, which is what makes `double_channel=True` reachable at all — the exact beam-basis S2 spans the full reciprocal grid at O(prod(gpts)²) per site per slice. Keep `mag_preserve=True` for absolute intensities.

At `interpolation = 1` the default transition-potential window is the whole 65 × 71 Å cell, which is far more than the edge needs; `BIP_INELASTIC_CROP = 24.0` Å holds >99 % of |H_n0|² for both K edges and keeps the per-atom reconstruction affordable.

Backend limitations to plan around: **eager only** (no Dask, so no multi-GPU fan-out — use `gpu_1gpu_shared.sbatch`), no frozen-phonon ensemble (`EELS_sim_bipprism.py` loops the configurations by hand), no `downsample`, single exit plane. Being eager and single-GPU, BiP-PRISM is *not* automatically faster in wall-clock than the lazy 4-GPU multislice for this problem — 3249 B sites × 4 slices × 9800 positions is a lot of per-atom reconstruction. Its value here is (a) exactness at a small beam count, and (b) being the only tractable route to `double_channel=True` on a 672 × 720 grid. **Benchmark it on a cropped scan before committing a 24 h job.**

### When each driver is actually faster — measured

PRISM-EELS gave **no speedup at all** on this system: 2.03 s per scan position against multislice's 1.92, i.e. the two differ only in fixed cost. That is not a defect, and it is worth understanding when it *would* pay, because the conditions are specific.

PRISM replaces one multislice per probe position with one per plane wave. Here the elastic propagation is **0.85 % of the runtime** — job 58259458 reports `elastic reference: 16.3 s` against `core loss: 1890.1 s` on the same 930 positions. There was only ever 1 % to win, because both algorithms pay `n_positions × n_sites` in the transition-potential reduction and PRISM does not touch that term.

Two things change that, and only one of them is thickness. Measured with `bench_drivers.py` (hBN, 192² grid, interpolation 2, 144 positions, 111 beams):

| slices | B sites | double-channel | multislice | PRISM | speedup |
|---|---|---|---|---|---|
| 4 | 96 | no | 10.2 s | 8.2 s | 1.25× |
| 4 | 96 | yes | 26.8 s | 13.4 s | 2.00× |
| 12 | 288 | no | 28.4 s | 24.6 s | 1.15× |
| 12 | 288 | yes | 225.1 s | 81.4 s | 2.76× |

Run-to-run spread on a shared CPU box is about ±15 % — an earlier run of the same script gave 1.09 / 1.71 / 1.20 / 2.84. Read the pattern, not the digits: single-channel sits near 1.2× and does not improve with thickness, double-channel is roughly 2× and climbs.

**Thickness on its own does almost nothing** (1.25 → 1.15×, i.e. flat within noise). Adding layers to a uniform crystal raises both terms together, since `n_sites = n_slices × sites_per_slice`, so

```
elastic : inelastic  =  n_slices : n_sites  =  1 : sites_per_slice
```

and the thickness cancels. What sets PRISM's headroom is **sites per slice** — 812 here, hence 800:1. A dilute emitter (a dopant, a few sites per slice) is the regime PRISM-EELS was demonstrated on. This specimen is made of the element being mapped, which is the worst case for it.

**Double channelling is where PRISM wins, and there thickness compounds** (2.00 → 2.76×). Multislice propagates the scattered wave to the exit per `(site, position)` on the full grid; the PRISM driver propagates it in the beam basis on the crop window and reduces to positions only at exit planes:

```python
batched = stack([...])          # (n_sites, n_T × n_beams, *window)
batched *= cropped_t
batched = fft2_convolve(batched, kernel)
```

so it swaps `n_positions → n_beams` *and* full grid → window. The cost of that term is `n_sites × n_slices/2 × (n_pos or n_beams)` — quadratic in thickness, visible above as multislice going 26.8 → 225.1 s (×8.4) for only 3× the sites.

**The table understates PRISM.** It ran 144 positions against 111 beams, a ratio of 1.3; the production case is 9898 against 745, a ratio of 13.3. PRISM's inner propagation is independent of position count and multislice's is proportional to it, so almost all of that factor accrues to PRISM. Decomposing the 12-slice row and rescaling puts double-channel PRISM nearer 8× — an extrapolation from two points, not a measurement.

**What this means here.** Single channel, 4 slices, 812 sites per slice is PRISM's worst case and the ~1 % will not improve. But if a `double_channel=True` check is ever wanted — and that is the least-justified approximation in the current setup — PRISM is the cheaper way to get it even at 4 slices. It also reframes the BiP-PRISM failure: BiP's pitch is accuracy at low beam count, and the regime where that matters is the same double-channel regime that is currently out of reach.

