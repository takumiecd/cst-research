# HIGGS multilayer conditioning: fresh A100 six-arm diagnostic

English (canonical) | [日本語](higgs-multilayer-conditioning.ja.md)

**PLANNED / PREPARED, not executed.** The question is what needs correction when
multiple CST maps stall, rather than assuming LayerNorm is necessary. Earlier
normalization cases combined LN with calibrated fixed gains, so they do not
identify bare-LN efficacy. This separate study preserves the original numerical
consumers and registries. It changes neither public torchcst nor a running study.

## Six fresh paired cases

All cases use `28 → 1024 × 8 → 2`, seven bias-free square maps, eight ReLUs,
trainable Dense IO, seed17 and unchanged legacy initialization. Each CST map has
exactly A=4N=4096 Atoms; no Atom/width/depth/LR/epsilon sweep is included.

| Case ID | Square maps | Normalization | Fixed gains |
|---|---|---|---|
| dense-none | Dense | NONE |1|
| dense-bare-ln | Dense | no-affine LN |1|
| cst-none | CST | NONE |1|
| cst-fixed-gain | CST | NONE | calibrated |
| cst-bare-ln | CST | no-affine LN |1|
| cst-ln-gain | CST | no-affine LN | calibrated |

All six models are freshly trained on A100. Old RTX4090 states/curves are not
relabeled paired A100 controls. Within each method raw ordinary/Atom tensors and
IO seed streams are identical before learning. LN adds no trainable affine and
acts over1024 features per example with population variance and epsilon1e−5.

Both CST gain arms sequentially match the **same initial Dense-stock pre-ReLU
RMS** at the seven square outputs, including all preceding candidate gains/norms.
Use only the fixed first256 TRAIN feature rows; no labels/DEV selection. Compute
with CPU FP64 public chunked references, round each multiplier to FP32, freeze it
as a nontrainable buffer in[1e−6,1e6]. Zero/nonfinite targets/signals or out-of-bound
ratios fail without clipping, fallback or an epsilon sweep. Raw P/ordinary tensors
must remain byte-identical. This is not an amplitude-initializer intervention.

\[
\operatorname{LN}(z)=\frac{z-\bar z}{\sqrt{v+10^{-5}}},\qquad
\operatorname{Var}(\operatorname{LN}(z))=\frac{v}{v+10^{-5}}.
\]

Tiny inputs can remain tiny: at the epsilon floor LN has finite gain about316.23.
In the NONE/no-normalization architecture, fixed positive gains preserve the global
ideal-real function family with bias-free/ReLU/unbounded-head maps, while initial
functions and Adam coordinates/epsilon behavior change. Do not extend simple
product-gain/head absorption to LN+gain: intervening LN epsilon breaks positive
homogeneity. LN itself changes the input-dependent function map. Neither
intervention is presumed necessary, sufficient or equivalent.

## Fixed budget and actual curves

Use B4, FP32, TF32 off, eager stock q1 CSTAdamW, LR0.0003, betas(0.9,0.999),
Adam epsilon1e−8, ordinary/amplitude weight decay0.01, raw Atom weight decay0.
The public975 physical radius-cell-mass M8/merge2/spacing1/birth4 settings remain
unchanged. One full100k epoch per arm is25000 actual task updates,150000 total.
Save actual `randperm(100000)` order seed117, shared across six cases.

Evaluate **full TRAIN100k and DEV50k at the same state** at updates0,1000,5000,
10000,15000,20000,25000. Retain logits, labels, raw row IDs and recomputed gap.
These are prespecified diagnostic curves, not HPO. Selection is **fixed final25000**,
never minimum DEV, early stopping or automatic extension. Reload that full state
with the exact native seven-owner generation transition and require TRAIN/DEV
array bytes and metrics to equal cached final predictions. Compare subsequent
mutation against the actual post-load baseline; do not drop metadata fields.

## New numerical and observation admission before task training

For each actual H8 arm, test full Y/dX/all11 parameter gradients against a new
manual CPU FP64 Dense/ReLU/LN/gain reverse and unchanged975 one-chunk256-Atom CST
VJPs. It evaluates all actual Atoms/widths without retaining a full-network
CST autograd graph or `[A,N,N]` tensor. Independent tiny public dense-export/full
autograd and functional-LN tests check the new reverse, including the epsilon floor.
Do not bypass the old NONE-only reference or weaken its guards.

Normal cotangent1 and amplified1e24 require prospective CPU representability/
material checks before GPU freeze. Normal elementwise tolerance is3e−5+3e−4|ref|;
both scales require per-gradient/per-packed-column stable FP64 relative L2 and
maxerror/refmax≤3e−4, with exact-zero reference requiring exact-zero actual.
Amplified dX and every packed column need some |ref|>3e−5+3e−4|ref| and some
nonzero actual. Preserve amplified componentwise failures as diagnostics and
normal FP32 rounded-zero tails without exemptions. Amplified gradients never
enter an optimizer. Retain every actual/expected VJP array, input and cotangent.

Add independent48-point quadrature on31 actual Atoms, native scalar-Adam checks,
eight normal synthetic CE updates versus four+native-load+four, and observed
first+seven bare updates versus eight unchanged `ordinary_step` updates. Every
model/gradient/m/v/clock and native owner store must match; only exact seven owners
at2/4/6/8/10/12/14 may advance generation by `max(before+1,serialized)` at load.
That is20 synthetic qualification updates per case,120 total, **not** task training.
All task models restart fresh0 with exact qualified initial0 bytes.

Actual task update1 retains real B4 features/labels/order, raw backward gradients,
accepted P/m/v/clocks, post-step gradients, per-layer forward/upstream gradients
and stable torus/FP32 displacements. It is part of25000, not an extra update.
Use Tensor.retain_grad only in this grad-enabled traced forward; no optimizer
monkeypatch or module/full-backward hooks. Raw/post-step equality plus canonical
FlatTorus identity semantics are reported; no prepared-buffer capture is claimed.
Denominators are **accepted-state/source-FP32-opmath reconstructions**, separately
scalar-qualified, not captured native intermediate buffers. Accepted displacement
includes retraction/amplitude decay. Report epsilon fractions on actual nonzero
FP32-gradient entries separately from all entries; do not turn them into causal proof.

Fixed256 TRAIN-feature forward diagnostics run initially/finally outside learning.
A final same-first-B4 backward probe runs on an independent fully loaded copy,
with no optimizer step; only declared copy gradients may change. Original model,
optimizer and retained gradients stay exact. Bare remaining steps use unchanged
`ordinary_step`, with descriptive rolling loss logs every1000 updates.

## Source, runtime, retention and execution

External source is `97580582ed28c4e494bf5c2656bbe94a85c544c3`, manifest
`17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`.
Data artifact is `58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`,
manifest `530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`.
Historical feature exposure and prior adaptive DEV use remain disclosed; this is
not fresh formal confirmation. No future/original confirmation or official test is staged.

A100 runtime pin is `35016be129960614d829affbea959797ae8d035b5ea41197b6803ae963709f55`:
Python3.12.9/Torch2.11.0+cu126/CUDA12.6/Triton3.6.0/NumPy2.0.2,
driver570.86.15, exact NVIDIA A10080GB PCIe MIG3g.40gb, CC8.0,42SM,
42144366592 GPU bytes. No generic parent80GB/4090 substitution. The owner alone
operates the Hub. The foreground consumer installs nothing and allocates no session.

```bash
python -B -m cst_research.experiments.higgs_multilayer_conditioning_driver \
  --consumer-commit FULL_REVIEWED_COMMIT \
  --cache-root /path/to/source-runtime-verified-cache \
  --output /path/to/fresh-diagnostic-output --timeout 7200
```

Exact six regular inputs are consumer-source-manifest.json, original-source-manifest.json,
original-source.tar.gz, train-development.npz, data-manifest.json and runtime-pin.json.
A reviewed clean source snapshot and all input bytes are checked before execution.
The cache fixes the public source/runtime compilation path and rejects changed bytes.
Timeout reaps the owned child process group; failures preserve evidence, with no retry.

This is **DIAGNOSTIC timing**, not a performance comparison to old B. Separate
calibration/admission/first observation/evaluation/snapshot/I/O/probe clocks.
Training-block walltime includes declared1000-step logging; component timers are
nonadditive. Reset/capture training peaks before checkpoint/eval; report resident
arrays/model/optimizer/native buffers and retained allocator reservations.

Retain qualified0/4/8/resumed8/observer8, actual first-step raw and full final state.
Intermediate points retain predictions plus SHA of the **complete typed core**,
including every row ID/m/v/clock/metadata field; this is a producer digest, not
independent raw-midstate replay. Load proofs retain seven-owner maps and complete
core SHAs, never repeated giant typed metadata trees. Conservative caps are6GiB
raw and64MiB JSON, with actual inventory/RSS measurements before launch.
Immutable full raw may stay on A100. Retrieve compact proofs/predictions and either
a local compressed archive if actual free space fits, or run a separately pinned
peer CPU saved-evidence auditor server-side with CUDA disabled and download its
report/selected predictions. No mandatory whole local decompressed duplicate or
existing evidence deletion. Root reviews actual storage/runtime before launch.

All prior scientific modules and11 registries are byte-identical;119/120 old files
are unchanged. The sole metadata exception updates the registry-count test11→12,
with its original bytes reconstructible. No validator gate is weakened.
No formal NI, Atom-order/capacity verdict, LN necessity, public default adoption or
automatic cross-device followup is implied. Evidence locator:
`$CST_RESEARCH_EVIDENCE_ROOT/higgs-multilayer-conditioning-20261106/`.
