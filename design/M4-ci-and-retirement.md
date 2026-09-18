# M4: CI, required-check decision, docs, legacy retirement

Target: two weeks elapsed, mostly waiting on the false-positive record; three small PRs on abTEM, one on the docs repository.

## Goal

Turn the suite into a gate that people trust, and remove the 13 one-off scripts from abTEM once every one of them has a reproduced case.

## Scope

- `.github/workflows/benchmark.yml`: label `benchmark` or `workflow_dispatch`; ubuntu, Python 3.12, `uv sync --group test`; job 1 A/B gate (quick, CPU, accuracy, head vs merge base, `--fail-on drift,shape`, `accepted_changes.toml` diff shown in the summary); job 2 informational compare against the latest tag bundle fetched from abtem-benchmarks; job 3 weekly self-check on dev writing its result to the job summary. `pytest benchmarks/tests` joins the normal test workflow.
- Required-check decision: after about a month, count self-check and gate outcomes; propose to Toma that the gate becomes required only if there were zero false positives.
- `.pre-commit-config.yaml` aligned to ruff (drop black and flake8, add ruff and ruff-format), because pre-commit currently runs flake8 at 120 columns over `benchmarks/`.
- Docs page in the abTEM docs repository: what the suite measures, how to add a case, how to accept a change, how to read the report (rows, columns, cells), how to run on your own machine.
- Legacy retirement: one PR deleting the 13 scripts from `abTEM/benchmarks`, with the PR body listing script, reproducing case and the M2 mining table; `legacy/PROVENANCE.md` in abtem-benchmarks is the permanent record. Scripts whose knowledge is diagnostic only (`profile_scan_detailed.py`, `diagnose_scan_vram.py`) are deleted with a pointer to the frozen copy.
- Contribution path for other authors' cases (C-PRISM, BiP-PRISM, plasmons, magnetism): a template case file and the checklist from DESIGN section 5 in `benchmarks/README.md`.

## Deliverables

- Three PRs on abTEM: workflow, pre-commit alignment, legacy deletion (separate, per the one-logical-change rule; the deletion PR waits for Toma's explicit go).
- Docs PR.
- A short summary on #380 closing the loop: what exists, how to run it, what is required.

## Exit criteria

- A pull request labelled `benchmark` produces a job summary with the A/B table within the hosted-runner budget (target under 15 minutes wall).
- The informational compare shows the v1.1 release-notes table for every PR against `refs-v1.0.10`.
- Zero false positives over the observation window, or each false positive has a root cause and a fix (a pin or a tolerance) before the required-check proposal is made.
- `abTEM/benchmarks` contains only the harness, cases, tests and docs.

## Verification commands

```
gh workflow run benchmark.yml --ref benchmark-suite
gh run watch
gh pr view <n> --json statusCheckRollup
```

## Risks

- Hosted runners are slow and shared; the quick tier must stay under the budget, so any new case that pushes the quick matrix past 10 minutes on the dev box is demoted to `standard`.
- Fetching the tag bundle from another repository in CI is a network dependency; job 2 is informational precisely so a fetch failure never blocks a PR.
- Deleting the legacy scripts removes `benchmark_finite_projection.py` from `main` as well at the next release; the PR body says so.
