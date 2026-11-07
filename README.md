# cst-research

English (canonical) | [日本語](README.ja.md)

Research on making CST usable in neural networks. The current priority is to
establish whether CST works in feed-forward networks (FFNs) and Attention,
starting with a small language model. We record reproducible conditions, learning
behavior, failures and fixes, and distinguish observations from hypotheses.
Licensed under [Apache-2.0](LICENSE).

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

## Current direction: FFN and Attention feasibility

**Atom-count lower-bound and Big-Ω research is stopped. The priority is to
establish practical use in FFNs and Attention.** This direction was adopted on
2026-11-06. FFN development experiments are now available; Attention validation
has not started.

The approved [FFN feasibility plan](docs/ffn-feasibility.md) uses
`A=ceil(8*sqrt(d_in*d_out))`, first at width256, then width128/512. It checks
usefulness with a linear Atom budget and bounded aspect-ratio changes; it does
not search for the minimum coefficient.

[Initial results and learning curves](docs/ffn-feasibility-results.md) separate
the three-arm pilot, radius diagnosis, numerical qualification and remaining
work. The selected recipe improves on no-FFN in audited one-seed development
runs at width128/256/512 and bounded aspect ratios. The owner stopped the
main-width three-seed family before completion to prioritize short field-LR
experiments. Seed29 results are retained as incomplete confirmation evidence. Current eager FP32
runs compress FFN parameters but do not show speed or GPU-memory advantages.
The [short field-LR comparison](docs/ffn-field-lr-r12-20261107.md) found slower
learning when only amplitude LR was reduced from16× to1× at fixed center rates
and Polar outer time. The [audited follow-ups](docs/ffn-field-lr-r14-20261107.md)
include amplitude8/32/64 at fixed centers/outer16. Amplitude8 passed the
engineering gate for API implementation. The public field_lrs API subsequently
passed25 CPU/8 CUDA tests and all8 full-LM update/resume configurations with64
trained projection VJPs; it merged through [torchcst PR151](https://github.com/takumiecd/torchcst/pull/151).
See the [validation and API record](docs/ffn-field-api-r16-20261107.md).
The owner clarified that whole-Atom32/64 is the intended high-rate comparison;
the [audited whole-Atom comparison](docs/ffn-whole-lr-r15-20261107.md) is now complete.
Whole32/64 improved some early points but did not improve the final short-budget
DEV value over16; no optimality or convergence claim is made.

1. **FFN first.** Check rectangular input/output dimensions, multiple tokens,
   forward/backward correctness and complete optimizer steps. Then replace the
   FFN projections in a small language model with CST and check whether it learns
   the task with stable training and meaningful validation performance.
2. **Attention next.** Start with CST replacements for the query, key, value and
   output projections. Keep the token-to-token attention computation and masking
   explicit; projection replacement alone does not establish a new attention
   operator. Investigate further changes after this first application works.
3. **Diagnose and repair failures.** Record the failing configuration and inspect
   initialization, activation/gradient scales, geometry and implementation
   correctness. Isolate the cause, fix it in the appropriate repository, and
   repeat the affected check before expanding the experiment.

A working application is the immediate goal. Dense remains a sanity control for
the task and training setup. Exhaustive Dense/DST comparisons, broad tuning and
Atom-count asymptotics are deferred. Learning curves, validation quality, speed
and memory remain useful observations, rather than a requirement to outperform
another method in this phase. Select concrete tasks, configurations and success
criteria in the next study before running it; a decreasing training loss alone
is not sufficient evidence of task usefulness.

Use the public torchcst initialization API, with the initializer and lowercase
`q` recorded explicitly. Initializer changes and other fixes remain hypotheses
until checked on the affected workload. Keep successful runs, negative results
and fixes traceable so the README can develop into a record of what CST can do.

## Previous HIGGS work

**The HIGGS campaign has unfinished scientific admission/assessment and is
stopped under the new research priority. Its training jobs are finished; no
primary/DST training is running.**

The campaign used torchcst [PR #141](https://github.com/takumiecd/torchcst/pull/141),
pinned to commit `97580582ed28c4e494bf5c2656bbe94a85c544c3`.
It replaced only the middle layer of a 28→N→N→2 network with CST, comparing
N=1024/2048 and α=0.001/0.002/0.005/0.01 against a tuned dense baseline.
The input and output layers remained dense, and all layers trained. Three confirmation
seeds were used, with margins fixed before execution: −0.2 percentage points in
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
| [HIGGS primary](docs/reproducibility.md) | STOPPED | External975; final30-model assessment pending |
| [HIGGS static/RigL comparison](docs/higgs-dst-comparison.md) | STOPPED | Stage1/2 consumers; common-horizon78-model gate pending |
| [A100 MIG comparison](docs/higgs-gpu-comparison.md) | COMPLETED | Bounded one-epoch environment comparison |
| [Coordinate LR](docs/higgs-coordinate-lr.md) | COMPLETED | Short pilot plus bounded two-seed followups; retain stock q1 |
| [Depth/layer position](docs/higgs-depth-position.md) | COMPLETED |13 development-only cases; scope-limited stall evidence |
| [Fixed output gain](docs/higgs-depth-gain.md) | COMPLETED |3 exploratory arms; partial rescue remains below Dense |
| [Covertype transfer](docs/covertype-transfer-results.md) | COMPLETED | Nine sealed models, one heldout assessment; both CST densities below Dense |
| [Eight-update normalization diagnostic](docs/higgs-depth-normalization-diagnostic-results.md) | COMPLETED | Nine cases ×8 task updates; mechanism evidence, no quality selection or default adoption |
| [Three-epoch normalization followup](docs/higgs-depth-normalization-learning.md) | STOPPED | Original source ea0 remains at exit 1 and incomplete after 300,000 main updates; retained evidence reused only through separate recovery |
| [Normalization load-contract recovery](docs/higgs-depth-normalization-learning-results.md) | COMPLETED | Nine exploratory cases through mixed lineage; A104 RMS/LN rescue remains below Dense; no normalization default adopted |
| [Linear Atom count: Stage B results](docs/higgs-linear-atom-results.md) | COMPLETED (bounded Stage B) | Sixteen one-epoch seed-17 endpoints; H8 CST bias-only, no Atom-order or minimum-c claim |
| [Scale versus LayerNorm](docs/higgs-initialization-results.md) | COMPLETED | Two independently admitted six-arm GPU cohorts |
| [Width / TRAIN-RMS initialization](docs/higgs-initialization-results.md) | COMPLETED | Old and width-aware stall; empirical Dense-RMS rescue, one seed |
| [He / Xavier moment initialization](docs/higgs-initialization-results.md) | COMPLETED | Paired q=0 pilot; explicit public API, no universal quality claim |

The [observations and limitations](docs/bounded-validation-results.md) bind the
completed coordinate/depth/gain results to their actual measured commits and raw
audits. Development-selected pilots establish neither a minimum alpha nor formal
noninferiority. HIGGS primary and DST are STOPPED with final fixed-family admission and
assessment unfinished; stopping does not imply completion. Measured jobs retain
their original source snapshots; integration does not restart them or complete
the outstanding scientific assessment.

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
The stopped formal HIGGS campaign used the pinned external source. A separate
[bounded A100 MIG environment-comparison adapter](docs/higgs-gpu-comparison.md)
consumes that unchanged benchmark after independent numerical admission.

For current FFN work, use the separately pinned library extra (merged
rectangular/initializer/field-LR APIs):

```bash
python -m pip install -e '.[ffn]'
```

The exact library commit is `c3b68dcf6275f96d29a21ff2aa6e989d42de2cff`.
This does not reproduce the numeric runtime or migrate historical runs; follow
[the frozen-source guide](docs/ffn-reproduction.md) for recorded quality results.

The historical HIGGS library extra retains its original study commit. Use a
separate environment for that dependency:

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

The [width-aware amplitude initialization study](docs/higgs-width-amplitude-init.md) compares raw `0.1/sqrt(A/N)` without gain/LayerNorm against the old initializer and the prior fixed-gain control. Both bounded GPU cohorts are complete and independently audited; see the [results](docs/higgs-initialization-results.md).

The [raw Dense-RMS initialization follow-up](docs/higgs-rms-amplitude-init.md) tunes initial amplitudes before training while keeping runtime gains1 and normalization NONE; Bounded GPU execution and independent audit are complete; see the [results](docs/higgs-initialization-results.md).

The [He/Xavier initialization pilot](docs/higgs-fan-initialization.md) uses data-free Atom moment targets, configurable lowercase `q`, and fresh paired Dense/CST comparisons. CPU preparation is separate from native qualification, learning results and public default adoption.

Completed [initialization results](docs/higgs-initialization-results.md) cover old/width-aware/TRAIN-RMS/data-free He/Xavier methods and normalization controls. The explicit public API is documented in [torchcst initialization](https://github.com/takumiecd/torchcst/blob/main/docs/initialization.md).

The [campaign disposition](docs/ffn-campaign-disposition-20261107.md) maps each
requirement to its evidence, including the owner-stopped three-seed family.

A [post-hoc radius review](docs/ffn-radius-review-20261107.md) conditions on physical
amplitude and matches Atom IDs: weaker/decreased Atoms do widen, but weak support
stays near4 under the current low-q/default-no-dormant-expansion recipe.
