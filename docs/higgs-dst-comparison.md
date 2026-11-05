# HIGGS: tuned static and RigL-style references

English (canonical) | [日本語](higgs-dst-comparison.ja.md)

## Status and relationship to the running study

This is a newly authorized comparison protocol, not a completed result. The
existing seed-17 DST pilot is exploratory development evidence that motivated
this addition; it is not a fresh held-out result or a tuned DST baseline.

The original CST study remains pinned to torchcst commit
`97580582ed28c4e494bf5c2656bbe94a85c544c3`, with its original eight-configuration
noninferiority family unchanged. Its confirmation evaluation is held until the
additional DST and comparison-control configurations and checkpoints are frozen.
No confirmation-tail result may guide their LR, horizon or checkpoint selection.
The official test set stays unopened.

The research consumer uses the external, pinned DST implementation at torchcst
commit `f37729e61dc11a57e76103aa2457c2be11c4c306`
([PR #143](https://github.com/takumiecd/torchcst/pull/143)). It does not vendor the
runtime or turn private experimental CUDA Graph adapters into public API.
Qualification and tuning are the first implementation scope. Confirmation and
assessment must fail closed until matching controls and the complete comparison
freeze are available; recording this protocol does not claim those stages have
run or are implemented.

## Consumer and exchange boundary

The consumer entrypoint is `cst_research.experiments.higgs_dst_comparison`; the
pool wrapper is `src/cst_research/experiments/higgs_dst_comparison_driver.py`.
The wrapper accepts a pinned consumer commit and stage plus width/method/divisor
selectors. Use the authorized shared-pool workflow; these modules do not allocate
GPUs themselves. The first delivery implements `qualify` and `tune`.
`freeze-width` can expose verified original-horizon/common-horizon readiness but
must refuse to publish a final comparison freeze until the full control workflow
is implemented and reviewed. `confirm` and `assess` are unavailable and fail closed
in that delivery.

External source archive/manifest, prepared data/data-manifest and consumer-source
manifest are independently supplied and bound to the run. Source manifests carry
full commits and member hashes; recovery requires its archive and receipt SHA.
The controller receives external-source/data/output/state/identity locations; an
optional original-selection index supplies original-study artifacts. Import the
actual original `frozen-width.json` with a bound file SHA, including identity,
width, horizon, all five configuration entries and their LR/candidate inventories.
Do not substitute a manually copied LR or infer a frozen horizon from a curve.
Retain the original confirmation seeds and development-only selection flags.

```bash
python -m cst_research.experiments.higgs_dst_comparison --help
```

## Arms, data and budgets

All models are 28→N→N→2 with two ReLUs, dense input/output layers, a bias-free
middle and every layer trainable. Use N={1024,2048}, divisors
{1000,500,200,100}, and α={0.001,0.002,0.005,0.01}. For each width/density,
compare a static mask and the pinned row-balanced RigL-style variant: 16 DST
configuration groups in total.

$$
A=\lfloor N^2/\mathrm{divisor}\rfloor,\qquad K=4A,\qquad
P_{\mathrm{active}}=K+31N+2.
$$

K is the middle layer's active scalar-weight **slot** count, including newly
grown zero-valued slots. It matches the CST atom-coordinate scalar count 4A;
it is not the number of stored Tensor elements, nonzero values or mask/index
bytes. Static and RigL start from the same mask/values. Their row budgets differ
by at most one and remain unchanged during rewiring. The pinned row-fan-in
initialization, ordinary-layer initial values and seed-dependent data order are
retained. Mask search and CST geometry impose different constraints; matching
active slots does not establish equivalent function classes.

Use the same prepared archive SHA256:
`62d32a1789ed737a5997ca6a0630548e6211af86c9cd6a0a605cd057d8bd32ac`.
Training uses the first 100,000 official training rows. Standardization was fitted
on the 10,400,000 official training rows only. Official validation indices
[0,50000) are development and [50000,100000) are confirmation; intervals are
zero-based and half-open. No official test example is evaluated.

Keep true batch 4, FP32, TF32 off, CPU math threads 1 and the pinned AdamW
contract: LR is tuned, betas=(0.9,0.999), eps=1e-8 and WD=0.01 for DST weights
and ordinary layers. The dense masked reference retains its pinned optimizer
implementation; do not imply bitwise-equivalent arithmetic across methods.

The RigL-style variant keeps cadence 250, initial drop fraction 0.3, cosine decay
ending at 0.75 of the run's minibatch budget, row-balanced growth and stable
column-index tie breaking. Events occur at positive minibatch steps divisible by
250 and strictly below floor(0.75 times total minibatches). Pruning uses a global
weight-magnitude threshold, retaining at least one survivor per row, then regrows
each row to its initial slot budget. A topology minibatch replaces the all-layer AdamW
update. It preserves surviving weights/moments and resets all pruned/grown
weights and moments, including immediate regrowth. Global Adam clocks do not
advance on topology minibatches. Record both minibatches/examples seen and actual
optimizer updates. Equal budget means equal example presentations, not equal
Adam update counts. A 20-epoch run starts fresh: extending a 10-epoch checkpoint
would change the intended horizon-dependent rewiring schedule.

## Equal LR search and a common comparison horizon

Development seed is 17. For each of the 16 DST groups, run every LR in
{0.0001,0.0003,0.001,0.003} for 10 complete epochs and evaluate all development
50,000 examples after each epoch. Pick the checkpoint with minimum development
cross-entropy, earliest epoch on an exact tie. Pick LR by minimum best-development
cross-entropy, lower LR on an exact tie.

If the selected base LR is at an edge, add exactly one candidate in that
direction: 0.00003 below 0.0001, or 0.01 above 0.003. The extra also receives
10 epochs. Retain the boundary flag if the final selection is still at an
evaluated edge. Do not drop failures or replace performance losses with reruns.

For each width N, obtain the original frozen horizon T_original(N), which was
chosen from dense plus four CST groups. Define the DST extension trigger using
all eight DST groups at that width: **any** selected 10-epoch candidate has
best_epoch≥9 and CE(epoch8)−CE(epoch10)>0.001. Then

$$
T_{\mathrm{DST}}(N)=\begin{cases}20&\text{if the DST trigger fires},\\10&\text{otherwise},\end{cases}
\qquad T_{\mathrm{comparison}}(N)=\max\bigl(T_{\mathrm{original}}(N),T_{\mathrm{DST}}(N)\bigr).
$$

If the comparison horizon is 20, rerun the top two 10-epoch LRs for **every** DST
group at that width from initialization for 20 epochs, then select by the same
development rule. This also applies when only the original study triggered 20.
All static, RigL, dense and CST comparison arms at a width use that common horizon.
Record best_epoch≥19 and CE(epoch18)−CE(epoch20)>0.001 as an unresolved improving
tail; do not silently extend the budget after confirmation.

Control handling is explicit:

| Original / comparison horizon | Dense and four CST comparison controls |
|---|---|
| 10 / 10 or 20 / 20 | Reuse the original frozen configurations and three-seed checkpoints only after source/data/configuration/hash verification. |
| 10 / 20 | In a separate comparison namespace, rerun the original development inventory's top two LRs per group for 20 epochs using the unchanged `97580582...` cell runner. Select there and train three fresh confirmation seeds for all five groups. |

The second case adds 10 control LR runs and 15 fresh control models per affected
width. It does not revise the original study's horizon, selection records,
checkpoints or eight-configuration primary analysis. If these extra controls
cannot be completed, the matched-horizon comparison is not ready; do not mix
10-epoch controls with 20-epoch DST or claim fair confirmation.

## Freeze before opening confirmation

Use paired confirmation seeds 29,43,71, initialized afresh. DST contributes
16×3=48 best-development checkpoints. Freeze all 48 checkpoint/data/configuration
identities, and all 30 dense/CST comparison-control checkpoint identities or
verified reuse references, before **any** confirmation-tail evaluator runs.
Also freeze the original study's 30-checkpoint manifest before its assessment.
Unfinished HPO, unresolved control provenance or an incomplete freeze blocks
assessment. Per-seed checkpoint selection uses development only.

The confirmation half remains held out only if no earlier pilot or assessment
has exposed it. Record that prerequisite explicitly. If it has already been
viewed, stop and amend the protocol before formal evaluation; it must not be
relabeled fresh held-out confirmation.

## Statistical claims

For each of 16 (width, density, DST method) configurations, compare with the
matched-horizon dense control using paired-seed differences. Both accuracy and
ROC-AUC margins are 0.002 in probability units; accuracy 0.002 is 0.2 percentage
points. Use the same df=2 one-sided paired-seed t tests and co-primary IUT:

$$
p_{\mathrm{configuration}}=\max(p_{\mathrm{accuracy}},p_{\mathrm{AUC}}),
\qquad16p_{\mathrm{configuration}}\le0.05.
$$

For zero seed standard error, p=0 only if mean_difference+margin>0; otherwise
p=1, including the exact boundary. Distinguish support, strong evidence of
inferiority and inconclusive outcomes. If simultaneous two-sided seed-t bands
are used for an inferiority diagnostic, Bonferroni covers this family's
32 endpoints; keep them separate from the primary IUT. Pointwise intervals and
paired-example bootstrap intervals are diagnostics with their declared scopes.
Three seeds have limited power and do not validate the approximately normal IID
paired-seed assumption. No joint seed/example confidence guarantee is implied.

The original CST family of eight and this new DST family of 16 each have their
own 0.05 familywise error level; this is not a joint 95% guarantee across all
24 claims. Direct CST–static/RigL contrasts and learning-curve differences are
descriptive unless a separate fixed inferential family is registered before
opening confirmation. Report the smallest supported tested density only, with
missing/inconclusive lower points and no monotonicity or universal minimum claim.

## Qualification, curves and interpretation

Before training, verify the pinned external source/data identity, masked CPU FP64
Y/dX/dW references, and static/RigL eager-versus-Graph trajectories at both widths
and density endpoints. Check actual topology budgets, all states/gradients,
dense growth gradients, moment resets, clocks and event counts. Retain failures
and qualification evidence. Do not bypass guards or widen a public capture API.

Retain every HPO, extended-horizon and confirmation-training curve, not only
selected runs. Plot training loss and development CE/accuracy/AUC against
minibatches/examples seen and cumulative training time; also record actual Adam
updates, preparation/evaluation costs and available end-to-end wall time.
Keep HPO seed 17 and independent confirmation-training seeds visually distinct.
These curves contain development metrics, not repeated confirmation evaluations.

The implementation stores dense weights/moments/masks and computes dense growth
gradients. Its times include topology work and event audit hashes. Record
synchronized complete-step timing, preparation-inclusive timing and allocator
allocated/reserved peaks, but make no claim about the speed or memory efficiency
of a sparse-kernel DST implementation. Separate VM HPO timings are exploratory;
retain job/hardware/runtime provenance for within-job comparison controls.

Raw source/result archives, selection inventories, masks, checkpoints, topology
hashes and curve data remain in ignored external evidence. The registry records
relative evidence locators, never datasets, credentials or absolute user paths.

## Stage 2 assessment boundary and analysis

Stage 1 remains the frozen qualification/tuning consumer at commit
`22f2810b1bc51a599e51c8936acac310ca9455ef`. A separate Stage 2 consumer adds
control import/retuning, independent training and the global freeze. Its
`cst_research.experiments.higgs_dst_assessment` module provides CPU admission,
once-only evaluation scheduling and CPU statistics. These paths have synthetic
CPU coverage; this documentation does not claim their GPU worker has been run.

The comparison-owned `original-confirmation-freeze-proof.json` is the exact
30-model payload the unchanged original `97580582...` assessor will create,
constructed from verified original training jobs and width freezes. It is bound
by raw-file and canonical-JSON SHA256 in the global 78-model freeze and explicitly
labeled a comparison-owned proof. Constructing it must not call the original
assessor: that command creates its freeze and immediately scores the tail.

The required order is:

1. Finish all original and comparison training/selection; seal the 78-model manifest and the original30 proof. Run the new assessment `audit` stage, which verifies all model/data/source files without importing a benchmark or reading confirmation examples.
2. Run the unchanged original975 `assess` once, after that global admission. Its actual30-model freeze must be canonically identical to the proof.
3. Run the new `evaluate` stage with `--original-assessment-job`. Verify the original job's source/result archives, receipt, manifests, freeze and saved prediction metrics. Import those once-scored predictions for reused controls; never evaluate their models again. Only the 48 new DST models and any fresh alternate20-epoch controls are scored here.
4. Run the CPU `analyze` stage on the complete saved assessment index. The original eight-configuration analysis remains separate and unchanged.

A missing original assessment bundle blocks new scoring when controls are reused.
A missing/mutated selected checkpoint, mismatched common horizon, source manifest,
producer/reuse reference or incomplete family blocks admission. Before loading
confirmation data, each fresh worker rechecks the complete global gate and its
pending attempt marker. Imports are restricted to byte-verified external source
directories for the two pinned commits. Source code from result archives is never
executed. The official test is not loaded.

The freeze seals 48 DST identities plus 30 matched-control identities, fixed
checkpoint/metadata/learning filenames, source-manifest digests, dataset identity,
per-width horizons and the original30 proof. `--frozen-index-sha256` must match
that immutable artifact. `--source-index` maps each pinned commit to its verified
cache directory and external manifest; it is not permission to run arbitrary
source paths.

Predictions, metrics, raw/array hashes and evaluation provenance are cached by
model identity and freeze SHA. A completed result is validated and copied on
resume without rescoring. An incomplete started attempt is preserved and refused
rather than automatically retried; recover it explicitly before continuing.
Reused original evaluation metadata is retained alongside the imported predictions.

```bash
python -m cst_research.experiments.higgs_dst_assessment --help
```

The stages accept these option groups:

| Stage | Inputs |
|---|---|
| `audit` / `evaluate` | `--frozen-index`, `--frozen-index-sha256`, `--campaign-state`, `--data`, `--source-index`, `--output` |
| `evaluate` with reused controls | Also `--original-assessment-job`, the complete verified original975 pool job |
| `analyze` | `--assessment-index`, `--output`; reads saved predictions only |

Use the authorized job driver and resource workflow for evaluation; the module
does not allocate GPUs. CPU prediction validation/analysis uses NumPy from the
recorded experiment environment, while registry commands remain stdlib-only.

Analysis implements the fixed16 IUT and separate joint32-endpoint inferiority
diagnostic. Direct CST–DST output includes each seed difference, mean, standard
deviation/error and an **unadjusted pointwise two-sided95% seed-t interval**
(df2), explicitly descriptive and conditional on the fixed examples/normal-seed
approximation. No multiplicity-adjusted CST–DST superiority claim is made.
Paired-example bootstrap intervals are not computed by this implementation;
the output declares that omission rather than implying joint seed/example coverage.
