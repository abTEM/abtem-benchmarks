# abtem-benchmarks

Benchmarks for abTEM simulations.

This repository has three areas.

- `design/`: the design and milestone plan for the abTEM regression benchmark suite (results, speed and memory compared across commits), proposed in [abTEM discussion #380](https://github.com/abTEM/abTEM/discussions/380). Start with `design/DESIGN.md`. The harness itself lives in the main abTEM repository under `benchmarks/`; this repository holds what must not live there.
- `refs/`: reference result bundles captured at release tags (for example v1.0.10), published as GitHub release assets of this repository so that the main repository stays small. `refs/README.md` documents naming, fetching and the machine-fingerprint policy.
- `legacy/`: frozen copies of every one-off benchmark script written for abTEM before the suite existed, with `legacy/PROVENANCE.md` recording where each came from and which suite case reproduces it. Nothing in `legacy/` is maintained.

The notebooks in `fft/`, `hrtem-local/` and `prism-distributed/` are the 2022/2023 scaling studies this repository was created for and are kept as they were.
