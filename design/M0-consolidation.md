# M0: Consolidation

Status: done. Tree committed and pushed 2026-09-18 (three commits); design summary and two follow-ups posted on #380 (2026-09-18, 2026-09-19). Only item 3 under "Left for Paul" remains.

## Goal

One place for everything benchmark-related that is not the harness itself: the design, the milestone plan, the frozen legacy scripts with provenance, and the home for reference bundles. Nothing is lost, nothing is deleted from abTEM, and every legacy script has a named successor.

## Scope

- Clone `abTEM/abtem-benchmarks`. `/workspaces/code` is a root-owned overlay in the devcontainer (the other repositories are individual bind mounts), so the clone lives at `/workspaces/run/abtem-benchmarks`; a bind mount line in `dev-env/.devcontainer/dev-ubuntu24.04_rocm7.0/devcontainer.json` exposes it under `/workspaces/code/` after the next rebuild.
- Layout: `design/`, `legacy/{abtem-repo,run,hbn-sonder}/`, `refs/`; the 2022/23 notebooks stay in place.
- Copy the 13 scripts from `abTEM/benchmarks` (dev @ c1d18e9f), verified byte-identical with `diff -rq`.
- Move the loose benchmark files from `/workspaces/run` (12 scripts and procedure docs, plus the suite-memory study with its JSONL curves) so nothing benchmark-related remains there.
- Copy the four PR-specific GPU benchmarks from `2026_hBN_defects_Sonder` (@ 1cdeca7c) and excerpt section 6 of its NOTES.md; the originals stay, that repository is a science record.
- `legacy/PROVENANCE.md`: per file, source, date, what it measures, knowledge to mine, reproducing component or case.
- `design/DESIGN.md` (v2), `design/DESIGN-v1-2026-09-10.md` (superseded, kept), `design/M0` to `M4`, `refs/README.md`, top-level `README.md`.

## Deliverables

All present in the tree; see `git -C /workspaces/run/abtem-benchmarks status`.

## Exit criteria

- `diff -rq /workspaces/code/abTEM/benchmarks legacy/abtem-repo` is empty (met).
- `ls /workspaces/run | grep -i bench` shows only `abtem-benchmarks` (met).
- Every file in `legacy/` has a row in `PROVENANCE.md` (met).
- Every point in Toma's reply maps to a section in `DESIGN.md` section 3 (met).
- Paul has read `DESIGN.md` and replied on #380 (pending).

## Left for Paul

1. Done 2026-09-18: design summary posted on #380 (it corrects the built-in transition-potential point), followed by the repository-roles and routine-tier follow-ups.
2. Done 2026-09-18: tree committed in three commits (legacy + provenance, design, README).
3. Add the devcontainer mount so the clone appears under `/workspaces/code/abtem-benchmarks`:
   `"source=${localWorkspaceFolder}/../code/abtem-benchmarks,target=/workspaces/code/abtem-benchmarks,type=bind"` after moving the clone on the host into `code/`.

## Risks

None technical. The only judgement call is that the 2023 notebooks were left untouched rather than moved under a `studies/` directory; moving them is a one-line change if Toma prefers.
