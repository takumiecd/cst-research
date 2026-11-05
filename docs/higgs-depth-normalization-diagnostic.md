# HIGGS eight-update scale-versus-centering diagnostic

English (canonical) | [日本語](higgs-depth-normalization-diagnostic.ja.md)

Study `higgs-depth-normalization-diagnostic-20261106` is **PREPARED, not executed**
(registry status PLANNED). This post-observation mechanism probe adds no learning
curve, density/LR search, quality-selection claim or automatic followup to the
completed f6/69 pilots or the separate running HIGGS confirmation families.
Current evidence motivates scale and sign-energy questions, not a proven cause
of the A104 stall or a guaranteed LayerNorm remedy.

## Fixed scope

28→1024→1024→1024→1024→2, three bias-free square middle maps and four ReLUs.
Ordinary input/output layers train; all trainable tensors are initialized by the
unchanged f6 factory and match the corresponding stock case. Public operators
and stock multi-site CSTAdamW come from external975; q1, LR=.0003, ordinary WD=.01,
amplitude decay=.01 and raw atom WD=0 remain unchanged. All cases are FP32,
TF32 off, B4, eager. No public Graph guard/default is altered or claimed.

Exactly **eight true training updates per case**, seed17, from the first32 indices
of the first epoch's `torch.randperm(100000, Generator(seed=117))`. Synthetic
qualification uses fresh models separately and never contributes task states.
No epochs, HPO, development evaluation, heldout opening or case selection occurs.
The nine cases and their execution order are fixed:

| Case | Squares | Intervention before each hidden ReLU |
|---|---|---|
| dense-stock | Dense | None |
| dense-rms | Dense | No-affine RMS + calibrated fixed positive gain |
| dense-ln | Dense | No-affine LayerNorm + calibrated fixed positive gain |
| cst104-stock | Three CST, A104 each | None |
| cst104-gain | Three CST, A104 each | Calibrated fixed positive gain, Dense pre-ReLU target |
| cst104-postgain | Three CST, A104 each | Calibrated fixed positive gain, Dense post-ReLU target |
| cst104-rms | Three CST, A104 each | No-affine RMS + calibrated fixed positive gain |
| cst104-ln | Three CST, A104 each | No-affine LayerNorm + calibrated fixed positive gain |
| cst1048-stock | Three CST, A1048 each | None; positive learning anchor from earlier budget only |

Both normalization operators act independently over the1024 features of each
example, after each square and before its ReLU, with **no learned affine or bias**:

\[
R_\epsilon(z)=z/\sqrt{\operatorname{mean}_{features}(z^2)+\epsilon},\quad
L_\epsilon(z)=\frac{z-\operatorname{mean}_{features}(z)}
{\sqrt{\operatorname{mean}_{features}((z-\operatorname{mean}_{features}z)^2)+\epsilon}},
\quad\epsilon=10^{-5}.
\]

Runtime RMS uses PyTorch `functional.rms_norm` with explicit ε; LN uses
`functional.layer_norm`. Independent CPU-FP64 references use manual mean/square
formulas and public `dense_weight()`, rather than calling those norm operators.

## Preparation amendment before execution

Before any GPU run, the user's output-contribution-scale hypothesis added the
ninth `cst104-postgain` case. Existing `cst104-gain` keeps its PRE-ReLU target.
Only the new case sequentially matches ReLU(raw square after preceding gains)
RMS to Dense-stock's initial POST-ReLU targets at3/5/7. Apply each positive gain
before ReLU, using ideal positive homogeneity. The same bounds/ε/failclosed rules
apply. The unexecuted eight-case prototype and SHA manifest are preserved
separately; this is not a response to a new result. True task updates total9×8.

## Calibration, without task labels

Use the first256 original-order TRAIN feature rows and Dense-stock's initial
three square-output RMS targets, as in the earlier gain pilot. For each norm
variant, sequentially apply its norm and preceding calibrated gains, then fix a
positive FP32 post-norm multiplier to match the corresponding PRE-ReLU target (POST-ReLU only for postgain). Dense norm
controls undergo the same procedure. The gain-only A104 case has no norm.
Stock/anchor cases keep unit gains. The norm ε is not varied. Zero/nonfinite
signals or required gains outside the inherited fixed [1e−6,1e6] bounds fail
closed; no clipping, substitute target or automatic fallback is permitted.
Calibration may not change any trainable tensor or consume labels/development.

## Admission before the first task update

All nine cases must pass before any task probe starts:

1. Independent CPU-FP64 full-network Y/dX/all-parameter gradients, with the
   inherited f6/69 normal elementwise tolerance3e−5+3e−4|reference| and strict
   relative norm/max-reference criteria3e−4, including each packed CST column.
   The inherited amplified-cotangent material/norm gates remain required; failed
   amplified componentwise stress remains disclosed and is not relabelled passed.
2. Native eight synthetic steps versus4+serialized-resume+4, exact full tensor
   state/gradient/moment/clock equality, finite values and seven owned parameters.
3. Observation-instrumented versus **unobserved native** eight synthetic steps,
   every step byte-equal on model/gradients/moments/clocks, for every case. The
   original CSTAdamW type/step remains intact; only its instance proposal seam is
   temporarily observed and restored in `finally`. Dense uses ordinary AdamW.
4. A native fused AdamW scalar fixture spanning gradients1e−10/1e−8/1e−6 and
   changing signs must validate the explicit denominator reconstruction against
   actual proposals, using the existing normal and relative-norm criteria.

The last diagnostic reads actual param-group β/ε/LR/WD and native m/v/clock.
Its reporting formula is mhat=m/(1−β₁^t), sqrt(vhat)=sqrt(v/(1−β₂^t)),
denominator=sqrt(vhat)+ε, with FP32-opmath coefficient semantics. Source975
performs PyTorch functional AdamW first, then CST retraction; this is its **raw
Adam proposal** denominator, not a claim about a different radius policy's
physical update. The numbers are explicitly reconstructed and scalar-qualified,
not exposed native kernel intermediates or an exact-bitwise operand claim.
No guessed literal β values replace actual group settings.

## Retained observations and interpretation limits

Save initial state0 and states1/2/4/8. StateN is owned by
`update-N.pt['accepted']`; avoid duplicate full checkpoints. Each retained raw
update also contains actual task B4 inputs/labels/logits/CE, per-layer forward
and backward tensors, pre-step parameters/raw task gradients, actual prepared
CST gradients, raw proposal parameters/moments and accepted/retracted state.
`updates.json` records every step's CE, row IDs, RMS/zero fractions, per-output-channel RMS quantiles and current-B4 nonzero/positive channel fractions, ReLU energy
retention, packed polar_s/polar_t/mu_output/mu_input statistics, actual m/v and
bias corrections, sqrt(vhat)/ε quantiles and fraction<ε, proposal/accepted/
retraction displacements and relative update RMS. Positions use minimum signed
torus displacements; the polar pair jointly encodes amplitude/activity and is
not labelled as two independently trained physical amplitude/radius parameters.

These are task-batch diagnostics, not the old256-row probe or a full-data metric.
Calibration uses the256-row feature probe separately. Wall time includes tensor
copies/statistics and is not throughput evidence. Allocated/reserved peaks are
allocator quantities including the diagnostic work, not whole-process memory.
Expected raw size is approximately1.5–3GB, dominated by Dense parameter/moment
copies; retain one set of raw state files rather than duplicating checkpoints.

The model's initial sign-energy ratios, actual task gradients and observed Adam
steps may support or refute narrow links. Eight steps cannot determine final
accuracy, an intrinsic capacity floor, a minimum α or an optimal N/depth scaling
law. A learnability floor is recipe-dependent; it differs from a Dense-quality
floor. No result in this probe authorizes a longer run or normalization adoption.

## Source, data, resource and recovery boundary

Freeze consumer commit/full file manifest and all inputs before execution.
External975 is `97580582ed28c4e494bf5c2656bbe94a85c544c3`; external manifest
`17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`.
Reuse f6 train/development artifact
`58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`
and manifest `530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`.
Train rows are[1000000,1100000); normalization fits only them. Historical aggregate
feature-statistic exposure remains disclosed. The unchanged reader mechanically
admits all four train/development arrays; no development tensor/evaluation is
constructed. Future-confirmation, original HIGGS confirmation and official test
payloads are excluded.

The direct driver accepts exactly six `research-inputs` files:
consumer-source-manifest.json, original-source-manifest.json,
original-source.tar.gz, train-development.npz, data-manifest.json, runtime-pin.json.
It verifies full source/data/runtime identities, uses isolated immutable source
paths for compilation and the already installed read-only job environment,
then runs a foreground child with bounded timeout. No provision/install/queue/
background process or retry is implemented. Only the root resource owner can
execute after review, on the exact reserved RTX3090, RTX4090 or A100 MIG3g.40gb profile;
the expected runtime-file hash is explicit. No Colab resource is acquired here.


### Pre-GPU resource amendment: RTX4090

The user's explicitly started/authorized RTX4090 is added as an execution target.
Root verified Hub profile `gpu4090-tf2-18-0-pytorch2-6-0-cu126-keras3-8-0`,
GPU `NVIDIA GeForce RTX 4090`, CC8.9,128SM,25262096384 bytes. The profile's
PyTorch2.6 name is not permission to use its base runtime: a separate isolated
job environment must provide Torch2.11.0+cu126/CUDA12.6/Triton3.6.0. Root has verified/reviewed the actual isolated runtime; its pin SHA is
`1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b`
(Python3.12.9/NumPy2.0.2/driver570.133.20). The driver verifies the
exact hardware tuple and actual runtime pin; root's launcher separately verifies
the requested Hub profile ID. Device properties alone do not attest that profile.
The scientific nine cases, eight updates each, calibration, math, tolerances and
all admission gates are unchanged. The preceding review inventory/source are
preserved; no GPU execution has occurred.

```sh
python -B src/cst_research/experiments/higgs_depth_normalization_driver.py --consumer-commit FULL_FROZEN_SHA --runtime-sha256 ACTUAL_RUNTIME_FILE_SHA --cache-root ISOLATED_JOB_CACHE --output FRESH_ATTEMPT_OUTPUT --timeout 3600
```

This is a recipe with symbolic pins, not an execution receipt. `progress.json`
and the foreground log report qualification/case/update progress. On success or
failure, the driver retains an SHA inventory `recovery-index.json` for the outer
result archive. Recover and inspect that exact attempt; never relabel a partial
run or reuse qualification/task state to continue beyond eight updates. Source
changes require a new frozen identity and reviewed attempt. Registration/CPU
checks are not GPU admission or completion.
