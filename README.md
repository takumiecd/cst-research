# cst-research

English (canonical) | [日本語](README.ja.md)

Research on CST scaling, accuracy, learning dynamics, speed and memory under
reproducible conditions. We freeze study designs and experiment identities before
running comparisons, and distinguish observations, uncertainty and untested
hypotheses. Licensed under [Apache-2.0](LICENSE).

## Relationship to torchcst

| Repository | Responsibility |
|---|---|
| [torchcst](https://github.com/takumiecd/torchcst) | Public modules, mathematical operator contracts, backends, autograd and kernel correctness |
| cst-research | Study designs, models and tasks, comparison protocols, learning curves, statistics, experiment provenance and reproduction instructions |

This repository does not duplicate the torchcst runtime. Research consumers and
workflow controllers live here; their pinned operator implementations remain
external. When an experimental Graph adapter needs
private interfaces, record those dependencies, the exact source commit and the
scope of independent validation. Use within an experiment and adoption into a
public API are separate decisions.

## Current study

**The low-density HIGGS CST study is RUNNING. Results are incomplete, and its
training runner has not migrated here.**

The campaign runs from torchcst [PR #141](https://github.com/takumiecd/torchcst/pull/141),
pinned to commit `97580582ed28c4e494bf5c2656bbe94a85c544c3`.
It replaces only the middle layer of a 28→N→N→2 network with CST, comparing
N=1024/2048 and α=0.001/0.002/0.005/0.01 against a tuned dense baseline.
The input and output layers remain dense, and all layers train. Three confirmation
seeds are used, with margins fixed before execution: −0.2 percentage points in
accuracy and −0.002 in ROC-AUC.

See the [machine-readable protocol](src/cst_research/studies/higgs-alpha-confirmation-20261105.json)
and [reproduction and evidence guide](docs/reproducibility.md).
Earlier studies and raw evidence remain in their original torchcst history and
external archives; they have not been copied wholesale into this repository.

The included consumer implementations and their measured histories are listed below.
They retain their original source/data identities; integration is not a rerun or
revalidation. See the [integration provenance](docs/integration-provenance.md).

| Study | Status | Scope |
|---|---|---|
| [HIGGS primary](docs/reproducibility.md) | RUNNING | External975; final30-model assessment pending |
| [HIGGS static/RigL comparison](docs/higgs-dst-comparison.md) | RUNNING | Stage1/2 consumers; common-horizon78-model gate pending |
| [A100 MIG comparison](docs/higgs-gpu-comparison.md) | COMPLETED | Bounded one-epoch environment comparison |
| [Coordinate LR](docs/higgs-coordinate-lr.md) | COMPLETED | Short pilot plus bounded two-seed followups; retain stock q1 |
| [Depth/layer position](docs/higgs-depth-position.md) | COMPLETED |13 development-only cases; scope-limited stall evidence |
| [Fixed output gain](docs/higgs-depth-gain.md) | COMPLETED |3 exploratory arms; partial rescue remains below Dense |
| [Covertype transfer](docs/covertype-transfer-results.md) | COMPLETED | Nine sealed models, one heldout assessment; both CST densities below Dense |
| [Eight-update normalization diagnostic](docs/higgs-depth-normalization-diagnostic-results.md) | COMPLETED | Nine cases ×8 task updates; mechanism evidence, no quality selection or default adoption |
| [Three-epoch normalization followup](docs/higgs-depth-normalization-learning.md) | PLANNED / PREPARED | Nine development-only cases; inherited qualification plus fresh-prefix admission required, no learning result yet |
| [Normalization load-contract recovery](docs/higgs-normalization-learning-recovery.md) | PLANNED / PREPARED | Parent remains incomplete; admitted300k reused, selected-state correction before only375k remaining updates |
| [Linear Atom count](docs/higgs-linear-atom-scaling.md) | PLANNED / PREPARED | New16 native qualification first; A=cN, H4/H8; no400k task training yet |

The [observations and limitations](docs/bounded-validation-results.md) bind the
completed coordinate/depth/gain results to their actual measured commits and raw
audits. Development-selected pilots establish neither a minimum alpha nor formal
noninferiority. HIGGS primary and DST stay RUNNING until their final fixed-family
admission and assessment finish. Existing frozen jobs continue using their
original worktrees, not this combined source tree.

## Quick start

Python 3.11 or later is required. The registry CLI uses only the standard library
and does not use a GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cst-research list
cst-research show higgs-alpha-confirmation-20261105
cst-research validate --json
python -m unittest discover -s tests -v
```

To inspect the registry without installing, use
`PYTHONPATH=src python -m cst_research list`.
`validate` checks registry metadata and fixed protocol conditions. It does not
validate training results, the existence of external evidence or GPU correctness.
The formal HIGGS campaign still runs in the pinned external source. A separate
[bounded A100 MIG environment-comparison adapter](docs/higgs-gpu-comparison.md)
consumes that unchanged benchmark after independent numerical admission.

If you need the torchcst library, install the optional extra pinned to the study's
source commit:

```bash
python -m pip install -e '.[torchcst]'
```

This installs a library dependency. To use the current benchmark runner, follow
the [instructions for obtaining the pinned torchcst source](docs/reproducibility.md).

## External evidence

Keep datasets, checkpoints, raw logs, pool state and credentials out of Git.
Place a directory named after the study ID under `CST_RESEARCH_EVIDENCE_ROOT`, or
configure the location in an untracked `research.local.toml`:

```toml
[evidence]
root = "./evidence"
```

```bash
cst-research --config research.local.toml show higgs-alpha-confirmation-20261105 --evidence
```

The configuration resolves a location; it does not copy, download or delete
anything. Output from `--evidence` includes the local storage path and should not
be included in public reports. When set, the environment variable takes precedence
over the TOML root.

Add studies and changes through a PR following [CONTRIBUTING.md](CONTRIBUTING.md).

The [width-aware amplitude initialization study](docs/higgs-width-amplitude-init.md) compares raw `0.1/sqrt(A/N)` without gain/LayerNorm against the old initializer and the prior fixed-gain control. GPU evidence is pending.

The [raw Dense-RMS initialization follow-up](docs/higgs-rms-amplitude-init.md) tunes initial amplitudes before training while keeping runtime gains1 and normalization NONE; GPU evidence pending.

The [He/Xavier initialization pilot](docs/higgs-fan-initialization.md) uses data-free Atom moment targets, configurable lowercase `q`, and fresh paired Dense/CST comparisons. CPU preparation is separate from native qualification, learning results and public default adoption.
