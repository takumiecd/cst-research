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

This repository does not duplicate the torchcst runtime. Future research runners
using its public API will live here. When an experimental Graph adapter needs
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

A newly authorized [static/RigL comparison](docs/higgs-dst-comparison.md) adds
independent LR tuning and three paired training seeds for both sparse references,
with a matched-horizon control gate. Its first implementation scope is
qualification and tuning; final freeze/confirmation/assessment stay blocked until
all required controls are ready. The existing CST primary study is unchanged,
and the earlier seed-17 DST pilot remains exploratory.

The next authorized direction is a [layer-position study plan](docs/higgs-layer-position-plan.md):
compare early/middle/late hidden-layer replacements at a fixed width and depth,
then test predictions formed from bounded density exploration. It starts after
the current campaign; its runner and execution protocol are not yet frozen.

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
