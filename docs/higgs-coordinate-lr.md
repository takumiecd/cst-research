# Bounded HIGGS position learning-rate pilot

English (canonical) | [日本語](higgs-coordinate-lr.ja.md)

This separate research consumer uses unchanged torchcst
`97580582ed28c4e494bf5c2656bbe94a85c544c3`, its private HIGGS benchmark and the
same SHA256-pinned100k training/100k validation archive. It changes neither the
public runtime nor the frozen CST/DST comparison. Status: preparation; no new
GPU or model-quality result is implied by CPU checks.

## Question and fixed budget

Test whether the positions need a different LR from the coupled Polar fields.
Use middle-layer CST with N1024, divisor1000 (1048 atoms), batch4, FP32, TF32off,
all layers trainable, and identical initialization/example order. All three arms
run eager: stock control1 and position multipliers0.1/10, each3 full epochs,
seed17,100k training examples and the first50k validation examples for development.
Ordinary and coupled-Polar LR remain0.0003; amplitude decay remains0.01. There
is no new LR grid, architecture/density change, holdout or official test access.
The Graph exact-type qualification is preserved; this subclass is not admitted
as a public Graph optimizer. Timings describe an eager scientific reference with
extra position computation and sampled diagnostics, not an optimization.

## Exact update boundary

The packed atom is **[s,t,output center,input center]**. The pair s,t jointly
encodes bounded amplitude and activity, from which the physical support radius
is decoded. This experiment does not independently tune physical amplitude and
radius. For each physical center μ:

\[
\hat m_t=m_t/(1-\beta_1^t),\quad
\hat v_t=v_t/(1-\beta_2^t),\qquad
\mu_{t+1}=\operatorname{wrap}_N\!\left(
\mu_t-\eta_\mu\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\right),
\]
\[
\eta_\mu=q\eta_{\rm polar},\quad q\in\{0.1,1,10\},\qquad
\eta_{\rm polar}=\eta_{\rm ordinary}=0.0003.
\]

The Adam gradients and moment recurrences are unscaled. The master clock advances
once. Normal Polar retraction/activity and old-amplitude decay use the unchanged
base LR. Positions have raw decay0 and retain their physical torus coordinates.

Scaling the displacement after the original FP32 proposal is **not** the chosen
method: at physical centers near512, rounding can double the effective motion
of the0.1 arm. Instead the private subclass snapshots contiguous position
parameters/gradients/moments and the old clock, calls the unchanged native AdamW
proposal, then recomputes positions from that original state with the direct
position LR. It copies only position parameters/moments back before the normal
retraction. Fused AdamW never receives strided packed-column views. The1 path
immediately delegates with no subtraction, multiplication or extra copy; the
pilot control uses the actual stock class. No per-step GPU scalar read is added.
Checkpoint contracts record the multiplier and reject a mismatch on load.

## Admission and retained evidence

Verify both consumer and external Git inventories, archive/data hashes and actual
L4 runtime. Reuse the actual receipt-bound975 qualification (136 tests,10 numerical
oracles,16 eager/Graph epoch pairs) only with its matching source/data/runtime.
Do not rerun the unchanged kernel's entire suite or weaken its existing guards.

New CPU checks cover independent scalar Adam plus Polar/torus retraction, actual
FP32-center ULPs, all1/native equality, a forced-recompute1 oracle, unchanged
ordinary/Polar proposals and moments, once-only clocks, boundary crossing,
serialized resume, different-constructor-LR restoration and multiplier rejection.
Before the pilot, a short new CUDA eager gate verifies direct LR/ULPs, contiguous
moment/column integrity,8-step trajectories and4-step save/reload continuations;
all1 also matches the stock tensor trajectory. Failure stops training.

Reuse the external runner through explicit factory/checkpoint/step instrumentation,
restoring every benchmark global in `finally`. Retain initialization, every epoch,
selected-best and final raw model/optimizer/gradient checkpoints, selected dev
predictions, curves and artifact hashes. Save sparse raw before/after update
samples and correctly labeled physical output/input centers, decoded amplitude
and radius, signed minimum-torus displacements, zero-motion fractions and clocks.
The coupled radius may evolve with the learned task; its update law is unchanged.

## Prespecified conditional step

Select each arm's earliest minimum-dev-CE checkpoint. Among AUC-eligible arms,
prefer minimum CE, with the control winning an exact control tie. Advance only
if a noncontrol improves CE by at least0.001 versus the matched control, loses
at most0.001 AUC, and passes finite/correctness/resume gates. Between noncontrols
with exactly equal eligible CE, choose the lower multiplier.

If it advances, the selected ratio and matched stock control run **10 full epochs
at seeds29 and43**, four runs total, with unchanged base LR/data/architecture.
Each fresh job runs one seed's pair if needed for the runtime limit. The owner
must verify the pilot evidence and stage its exact decision-file SHA before any
conditional submission; the runner never expands automatically. This is bounded
exploratory replication using development data, not formal noninferiority or a
new unseen evaluation. Otherwise stop: the conclusion is only no evidence within
this short pilot budget, not that position LR can never help.

The pool script is `src/cst_research/experiments/higgs_coordinate_lr_driver.py`.
Its `--stage pilot` runs admission then the three arms. `--stage followup` also
requires `--seed 29|43`, `--selected-ratio 0.1|10`, and the reviewed
`--pilot-decision-sha256`. Supply flags and values as separate CLI arguments.
Freeze the consumer manifest and open a draft PR before GPU execution.
