# Reproducing source and evidence

English (canonical) | [日本語](reproducibility.ja.md)

## The current HIGGS campaign runs in torchcst

The registered study is running from this
[pinned torchcst source](https://github.com/takumiecd/torchcst/tree/97580582ed28c4e494bf5c2656bbe94a85c544c3).
Its authoritative study protocol is
[`higgs-alpha-confirmation-20261105.ja.md`](https://github.com/takumiecd/torchcst/blob/97580582ed28c4e494bf5c2656bbe94a85c544c3/docs/research-history/cuda-linear/higgs-alpha-confirmation-20261105.ja.md)
at that commit. This repository's JSON provides an operational entry point and
index; it does not rewrite the source identity of an existing run.

```bash
git clone https://github.com/takumiecd/torchcst.git torchcst-higgs-source
cd torchcst-higgs-source
git checkout --detach 97580582ed28c4e494bf5c2656bbe94a85c544c3
python -m pip install -e '.[dev,cuda]'
python -m benchmarks.training.higgs_confirmation_campaign --help
```

Installing only the optional library extra does not install `benchmarks` from the
source checkout. A new reproduction run must use separate outputs, state and
identities, and must not be mixed into the running campaign.

Before GPU work, read torchcst's CONTRIBUTING document, research operating rules
and the applicable resource skill. Submit shared L4 work through the shared pool.
Do not allocate a new VM or directly edit existing pool state as part of these
instructions.

## Data and stages

The original HIGGS dataset has 11,000,000 rows. Standardization is fitted only on
the 10,400,000 training rows. The experiment trains on the first 100,000 rows and
splits the official validation set into the first 50,000 examples for development
and the last 50,000 for confirmation. The official 500,000-example test set is not
evaluated in this study.

The required prepared NPZ's SHA256 is fixed in the study JSON. This repository
contains neither the original CSV nor the NPZ. Follow the data preparation and
verification path in the pinned source's existing runner and external
data-manifest. A different split or standardization does not reproduce the same
result.

The pinned source provides this entrypoint:

```text
python -m benchmarks.training.higgs_confirmation_campaign
  --stage qualify|tune|freeze-width|confirm|assess
  --data PREPARED_NPZ --output FRESH_OUTPUT --campaign-state CAMPAIGN_STATE
  --source-identity FROZEN_SOURCE_MANIFEST_SHA256
```

This is an option reference. The actual pool driver manages the immutable source
cache, dependencies, state bound to source/data/runtime, and retrieval with hash
verification. Create new campaign state, pass `qualify`, then execute the stages
in order:

1. `qualify`: Check independent oracles and eager/Graph complete steps.
2. `tune`: Tune dense and four CST densities at each width with development seed 17. Run every LR candidate for 10 epochs.
3. `freeze-width`: Use the five configurations' development results to choose a common 10- or 20-epoch horizon. If required, rerun the top two LRs for 20 epochs.
4. `confirm`: Train fresh models with seeds 29/43/71 using the frozen LR and budget. Each job contains dense and all four CST densities.
5. `assess`: Freeze all 30 checkpoints first, then evaluate the 50,000 confirmation examples. Do not open the test set.

See the source's `--help` and the study JSON for `--width`, `--kind`, `--divisor`
and `--seed`. `confirm` names the independent-training-seed stage. Its learning
curves still contain **development metrics**, not held-out confirmation evaluation.

## Scope of comparisons and statistics

The IUT requires both the accuracy margin of −0.2 percentage points and the
ROC-AUC margin of −0.002 to pass, with Bonferroni correction over eight
configurations: two widths × four densities. The simultaneous two-sided intervals
for 16 endpoints are separate diagnostics.

Three seeds may leave comparisons inconclusive. Distinguish the smallest supported
tested point from an identified density boundary. Even if α=0.001 passes, required
densities below it remain untested.

Measure synchronized complete training steps and report time including preparation
separately. Training time excludes evaluation, checkpoint and hash diagnostics.
Separate HPO jobs may run on different L4 VMs, and the CPU host model was not
recorded, so overlays of those timing curves are exploratory. Primary speed and
time-to-quality comparisons use paired dense-versus-four-CST measurements within
the same job for each width and independent seed.

Training/development loss curves and improvements at the end of a run do not prove
mathematical optimization convergence. Distinguish allocator allocated/reserved
peaks from total GPU process memory as well.

## External storage and migration

`CST_RESEARCH_EVIDENCE_ROOT/<study-id>/` locates a study's external evidence.
Set the environment variable or use `[evidence] root` in an untracked TOML file.
A relative root is resolved from the TOML file's directory.
`CST_RESEARCH_CONFIG` can select that file; `CST_RESEARCH_EVIDENCE_ROOT` takes
precedence. The CLI reports the location only. It does not guarantee evidence
availability, hashes or completion of the experiment.

Retain the original pool job result manifests, recovery archives and receipts,
source/data identities, curves, and checkpoint/prediction hashes. Verify the outer
archive and internal member hashes before retrieval or reuse. Public notes should
contain relative locators, source/result SHAs, job IDs and summaries, without local
absolute paths or secrets.

When moving a research runner here in the future, identify its public torchcst API
dependencies and private experimental Graph adapter separately. Revalidate
independent oracles, complete steps, optimizer state, and time/memory comparisons
under matching conditions. Public API changes require a separate torchcst PR.
Do not relabel an existing private adapter as a public API.

Older experiments remain in their original torchcst commits, records and ignored
archives. Copying their history does not turn them into experiments or verified
evidence produced in this repository. When organizing external evidence, preserve
its association with the original source, data, hardware, runtime, workload and job,
along with recovery paths.
