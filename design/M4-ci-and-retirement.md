# M4: CI, required-check decision, docs, legacy retirement

Target: two weeks elapsed, mostly waiting on the false-positive record; three small PRs on abTEM, one on the docs repository.

## Goal

Turn the suite into a gate that people trust, and remove the 13 one-off scripts from abTEM once every one of them has a reproduced case.

## Scope

- `.github/workflows/benchmark.yml`, as DESIGN section 13 describes: job 1 per pull request into dev that changes `abtem/` (quick, CPU, `accuracy`, head against merge base, interleaved, `--fail-on drift,shape,error,missing`, the `accepted_changes.toml` diff shown in the summary); job 2 weekly on dev (a small set against the last checked commit, a dev-against-dev self-check in the same job, and an informational compare against the latest tag bundle fetched from abtem-benchmarks, all written to the job summary); a `workflow_dispatch` trigger for both. The harness's own tests already run in every cell of the `tests` job (#704). Timing and memory runs on a fixed machine (M3) carry their own self-check in the same job.
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

- A pull request into dev that changes `abtem/` produces a job summary with the A/B table within the hosted-runner budget (target under 15 minutes wall); one that does not change `abtem/` runs no capture.
- The weekly job's informational compare shows the v1.1 release-notes table against `refs-v1.0.10`.
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
- Fetching the tag bundle from another repository in CI is a network dependency; it runs only in the weekly job and is informational, so a fetch failure never blocks a pull request.
- A dependency change (a numpy or scipy bump) is not covered by the per-PR check: both refs run in one environment.
- Deleting the legacy scripts removes `benchmark_finite_projection.py` from `main` as well at the next release; the PR body says so.
