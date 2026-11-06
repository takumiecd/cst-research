# HIGGS normalization learning: bounded three-epoch followup

English (canonical) | [日本語](higgs-depth-normalization-learning.ja.md) · [Exact JSON](../src/cst_research/studies/higgs-depth-normalization-learning-20261106.json)

**PREPARED; implementation and CPU checks only, no new GPU execution or learning result.** Study ID:
`higgs-depth-normalization-learning-20261106`. This separate three-epoch followup
was chosen **after** the completed 4f eight-update observations and earlier f6/69
development results. It asks whether the early motion differences persist into
bounded fitting/quality differences. Previously inspected development data are
reused deliberately; they are not a fresh test. All primary HIGGS/DST protocols
and the completed eight-update study remain unchanged. There is no automatic
adoption, case advancement or further run after this finite comparison.

## Fixed cases and recipe

All nine cases run in exactly this order on the same reserved RTX4090/runtime.

| Order | Case | Square maps | Before each square-output ReLU |
|---:|---|---|---|
| 1 | dense-stock | Dense | None; unit gains |
| 2 | dense-rms | Dense | No-affine RMS, fixed Dense pre-ReLU RMS target |
| 3 | dense-ln | Dense | No-affine LN, fixed Dense pre-ReLU RMS target |
| 4 | cst104-stock | CST A104 ×3 | None; unit gains |
| 5 | cst104-gain | CST A104 ×3 | Fixed Dense pre-ReLU RMS target |
| 6 | cst104-postgain | CST A104 ×3 | Fixed Dense post-ReLU RMS target |
| 7 | cst104-rms | CST A104 ×3 | No-affine RMS, fixed Dense pre-ReLU RMS target |
| 8 | cst104-ln | CST A104 ×3 | No-affine LN, fixed Dense pre-ReLU RMS target |
| 9 | cst1048-stock | CST A1048 ×3 | None; unit gains |

Architecture: 28→1024→1024→1024→1024→2; three **bias-free** square maps at
sites2/4/6, ReLUs at1/3/5/7, trainable ordinary Dense IO. A104 means divisor10000
(α=.0001), and A1048 means divisor1000 (α=.001), with the inherited integer floor.
Every case starts fresh at seed17 and trains exactly **3 epochs ×25000 true B4
updates =75000 updates**; total learning budget675000. Epoch permutation seeds
are117/118/119, using unchanged `torch.randperm(100000)` semantics and the same
permutation across cases. A1048 is a distinct budget anchor, not a threshold test.

LR=.0003, q1 stock public975 CSTAdamW (ordinary AdamW for Dense),
β=(.9,.999), Adam ε=1e−8, ordinary WD=.01, amplitude WD=.01, raw atom WD=0,
fused=True, capturable=False, FP32/TF32 off/**eager**. All trainable parameters
remain trainable. No HPO, LR/ε/density search, changed optimizer or early stopping
based on quality. The timed loop calls unchanged **unobserved** `ordinary_step`.

For each example's1024 features, after each square and before ReLU:

\[
R_\epsilon(z)=\frac{z}{\sqrt{\operatorname{mean}(z^2)+\epsilon}},\qquad
L_\epsilon(z)=\frac{z-\operatorname{mean}(z)}
{\sqrt{\operatorname{mean}((z-\operatorname{mean}(z))^2)+\epsilon}},
\qquad\epsilon=10^{-5}.
\]

No learned norm affine/bias. The existing sequential CPU-FP64 public
`dense_weight()` calibration uses the first256 original-order **TRAIN features
only**, no labels. Positive gains are nontrainable FP32 buffers in[1e−6,1e6].
Targets and the postgain exception are exactly those in the table. Zero/nonfinite
signals/targets, out-of-range gains or changed trainables stop admission; no
clipping, alternate ε or substitute target is allowed.

## Data and identity

Use only the existing TRAIN100k/development50k artifact, SHA
`58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`, manifest
`530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`.
Raw TRAIN rows[1000000,1100000), development[1100000,1150000). Preprocessing fits
only this TRAIN; prior aggregate-feature-statistic exposure remains disclosed.
Future-confirmation[1150000,1200000), original HIGGS confirmation and official
test payloads are excluded. Do not claim never-opened bytes merely from an
evaluation restriction; here the deployment itself excludes those payloads.

Preserve all24 existing scientific modules of measured consumer
`4f777a5f1ef38cca7031cf9c637d0fa6fc0d5e7b` byte-for-byte; manifest
`aa942f52f88d9a26a950eb51203df948216bbb50fb437c410b9dc8d9b86bbd53`.
The exact24-file inventory is taken from that measured85-file manifest, not
reconstructed from the current checkout; see the [model contract](higgs-depth-normalization-diagnostic.md). Public source remains
`97580582ed28c4e494bf5c2656bbe94a85c544c3`, manifest
`17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`.
A **separate learning loop/driver/tests** is prepared; its reviewed source commit
and manifest must be frozen before execution. Keep the inherited model-contract identity separate
from this new learning-study identity rather than relabelling old results.

Exact execution target: NVIDIA GeForce RTX4090, CC8.9,128SM,25262096384 bytes,
authorized profile `gpu4090-tf2-18-0-pytorch2-6-0-cu126-keras3-8-0`.
Actual runtime pin
`1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b`:
Python3.12.9, Torch2.11.0+cu126/CUDA12.6, Triton3.6.0, NumPy2.0.2,
driver570.133.20. Root remains the sole operator; no L4 campaign/source/queue
change is part of this proposal.

## Inherited qualification and new loop admission

The previous **independent CPU raw admission** is
`89238d2bece6f2537c68e3053e4da839a138dba7f4f78cd8c0bc92fe54faeba8`;
151-file admission
`3d857e153906395c76cf516f0afeb54508bf532acfa375b99e0c69f6e916ab3a` and
outer receipt
`ae0c2bbb7f000f5465c41b24e66c08223ffdd722a45382d70251db71f905c6ab`.
Reference existing immutable raw evidence in place, without another2.64GB
upload/copy. Relative external locator:
`higgs-depth-normalization-diagnostic-20261106/evidence-index.json`.

Report this as **INHERITED 4f qualification**, not a newly executed native/FP64
oracle. It covers all-nine normal Y/dX/all-gradient, norm/packed-column/material,
8 versus4+resume+4, scalar-Adam and observer/native checks in their original
scope. Independent retained observer/native comparison was final8; producer
live checks covered every synthetic step. Amplified componentwise failures,
including Dense, remain failed and preserved. It does not validate a new epoch
loop, metric/selection code, final quality or throughput.

Before any three-epoch case learns:

1. Verify all source/data/runtime/profile identities and every preserved24-file
   hash. New CPU checks must cover epoch orders/counters/budget/no-drop/reset,
   earliest-min-CE ties, scoring/serialization, eval/train mode restoration and
   absence of heldout paths. Root independently reviews the frozen new source.
2. Construct **all nine fresh zero-update models** and calibrate features-only;
   compare their trainable tensor fields, FP32 gains, ordinary IO and calibration
   records against actual prior4f **state0/calibration**. Compare tensor fields,
   not whole checkpoint-file bytes containing different study metadata. Never
   use fresh local2.13 constructor assumptions as the producer-byte oracle.
3. Compact new-loop admission: fresh bare `ordinary_step` replays the
   exact first32 TRAIN indices for each case and matches actual prior4f states
   **1/2/4/8** on all model/gradient/moment/clock tensor bytes. These72 repeated
   task-prefix admission updates are separate from the675000 learning updates;
   they are not synthetic or a rerun FP64 qualification. Save compact equality/
   hash results, then discard/reset fresh0. **Never continue state8.**

## Selection, fitting and resource evidence

Evaluate full development50k after each epoch. Choose the **earliest minimum
development CE among epochs1..3 only**; no epoch0, case or LR selection. Retain
state0 plus all three raw epoch model/optimizer/gradient/clock/order states, and
a best-checkpoint SHA reference to that retained epoch (avoid duplicate tensors).

At that **same selected checkpoint**, retain full TRAIN100k and development50k
FP32 logits, INT64 labels and original-row order/provenance. Existing selected
epoch dev logits can be reused only with exact checkpoint/prediction hashes;
otherwise score selected dev separately outside the training clock. Rebuild/load the
selected state and require exact train/dev logit-array reload equality at fixed
batching/runtime. Recompute
CE/accuracy (argmax tieclass0)/AUC/logitRMS independently from retained arrays
using pinned existing conventions. Report devCE−trainCE at the same fixed state,
and train/dev accuracy/AUC; online training CE is a different trace, not a
fixed-checkpoint train evaluation. All nine cases must appear in the report.

Case `preparation_seconds` is an inclusive total for construction, state0
read/admission, state0 serialization and initial diagnostics. The separately
reported `initial_diagnostic_seconds` is already included, not additive. Observe
the declared scopes of other preparation/evaluation/checkpoint components rather
than summing overlapping totals and breakdowns.

Plot all nine online trainCE/devCE/devaccuracy/AUC histories against true
minibatches and cumulative synchronized **unobserved complete-step** training
seconds, including matched Dense-stock/RMS/LN controls. Record preparation,
calibration, admission, evaluation and checkpoint/hash time separately; no
proposal observer in timed steps. Retain the fixed256 TRAIN-feature activation/RMS/ReLU-energy probe and last actual
minibatch retained parameter gradients plus m/v/clock statistics outside timing.
Do not infer an actual one-step proposal/displacement from epoch endpoints; any
saved-state epoch movement would be a separately labelled diagnostic. Each epoch resets allocated/reserved peaks immediately before the synchronized
training loop and captures them before checkpoint/evaluation. They include
resident train/dev arrays, model, optimizer and native buffers; reserved allocator
blocks from earlier evaluation may persist. These are not process-memory peaks.
The old observed eight-step walltime/memory has a different diagnostic scope.
Same sequential device/runtime still has order/cache/thermal effects: no general
throughput, hardware-causality or independent statistical claim follows.

## Fixed stop and interpretation

Hard learning budget:9 cases,3 epochs/75000 updates each,675000 total. Proposed
foreground wallclock cap7200s is an **operational proposal**, not a runtime
prediction; root must seal it and an available-storage budget before execution.
Source/init/calibration/gate mismatch, nonfinite epoch loss/state/moments/gradients,
ownership/clock failure, OOM/storage/timeout/transport failure stops the batch,
preserves partial evidence and triggers no automatic retry/fallback. A finite
stalled case is a valid negative result and completes its three prescribed epochs.
Root may preempt for primary needs, leaving this study incomplete. Success ends
after all nine bounded cases and evidence recovery/audit: **no automatic extra
epochs, seeds, densities, ε/LR search or advancement**.

For gain-only cases, positive gains and bias-free squares/ReLU satisfy, in ideal
real arithmetic with unbounded trainable Dense output:

\[
G=g_1g_2g_3>0,\qquad F_{\rm gain}(x;W_{\rm out},b_{\rm out})
=F_{\rm stock}(x;G W_{\rm out},b_{\rm out}).
\]

The global gain-only function family is unchanged, but gradient scales,
parameterization/Adam ε/regularization mapping and FP32 trajectories change;
this is not pure initialization-only causality. RMS/LN are **input-dependent
nonlinear maps**, with nonzero ε and LN centering, so the gain-only identity does
not establish equivalence for normalized models. Matched Dense norm controls
help separate a generic normalization effect from this CST recipe; one seed and
three epochs cannot prove centering alone caused a rescue, intrinsic capacity,
minimum α, a scaling law, formal NI/significance/CI or default adoption.

## Prepared direct-Hub adapter

The new adapter accepts exactly ten staged inputs: `consumer-source-manifest.json`,
`original-source-manifest.json`, `original-source.tar.gz`, `train-development.npz`,
`data-manifest.json`, `runtime-pin.json`, `prior-full-CPU-admission.json`,
`prior-file-admission.json`, `prior-outer-receipt.json` and
`prior-launch-intent.json`. All are SHA-pinned; the new consumer manifest binds
its future reviewed full commit. The prior 151-file result tree is a read-only
path argument. Only the two small old receipt/intent files may be relocated by
exact byte hash, because the remote parent-level paths differ. All149 remaining
files stay in their original relative result locations and are streamed in place.

From the reviewed consumer checkout, with those inputs in `research-inputs/`:

```bash
python -B src/cst_research/experiments/higgs_depth_normalization_learning_driver.py \
  --consumer-commit FULL_REVIEWED_COMMIT \
  --prior-root PRIOR_ATTEMPT_ROOT --cache-root OWNED_COMPILE_CACHE \
  --output FRESH_ATTEMPT_OUTPUT --timeout 7200
```

This is an execution recipe, not authorization or a completed run. The foreground
child process group is reaped on timeout/failure; partial actual files and their
SHA inventory remain. There is no automatic resume or retry. State0/epoch states
are retained once (selected best is a SHA reference), expected total new evidence
below2GiB; the owner must verify at least10GiB remote and25GiB local free storage
before launch. A separate deployment envelope pins these operational receipts;
this adapter does not provision hardware or alter cached account/session state.
