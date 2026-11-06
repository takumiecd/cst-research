# HIGGS multilayer conditioning: separate RTX4090 cohort

English (canonical) | [日本語](higgs-multilayer-conditioning-rtx4090.ja.md)

**PLANNED / PREPARED; no execution result.** This cohort repeats the exact
[paired six-arm A100 design](higgs-multilayer-conditioning.md) on RTX4090. The user
requested parallel use of the available4090. N=1024, H=8 and A=4N=4096 are held
fixed so width and device are not confounded. No N2048 or extra intervention is
added. Each device has its own six fresh paired models and new native admission;
no A100/old4090 state, calibration or qualification marker is relabeled or reused.

## Unchanged scientific contract

The seven original conditioning implementation/driver files from
`a43d6a938db7ff1bbf3522343cd35f10f3dcac1c` remain byte-identical and are SHA-checked
before model construction and numerical execution. New source only orchestrates the4090 cohort and strict
foreground deployment. It directly calls unchanged `qualification.qualify_all`,
`base.run_case`, `base.calibration_features`, activations and budget helpers.
There is no monkeypatch, attribute override, copied numerical runtime or Graph change.
The raw model-configuration cohort retains its original numerical-contract ID;
new run identity, runtime and protocol identify this independent4090 experiment.

| Case | Square maps | No-affine LN | Gains |
|---|---|---|---|
| dense-none | Dense | no |1|
| dense-bare-ln | Dense | yes |1|
| cst-none | CST | no |1|
| cst-fixed-gain | CST | no | fixed calibrated |
| cst-bare-ln | CST | yes |1|
| cst-ln-gain | CST | yes | fixed calibrated |

All models are28→1024×8→2 with seven bias-free square maps, eight ReLUs and
trainable Dense IO. Seed17, unchanged legacy initialization, eager FP32/TF32 off,
B4, q1 AdamW/LR0.0003, betas(0.9,0.999), Adam epsilon1e−8, ordinary/amplitude
WD0.01 and raw Atom WD0 remain fixed. Within each method initial raw Atom and
ordinary/IO tensors use the same streams across arms. LN is featurewise over1024,
nonaffine, epsilon1e−5. Both CST gain arms sequentially match the same initial
Dense-stock pre-ReLU RMS on the first256 TRAIN features, using unchanged CPU FP64
references. Positive FP32 gains in[1e−6,1e6] are frozen; no labels/DEV, zero/nonfinite
fallback or clipping. Raw trainables are unchanged. This is not amp initialization.

One full100k epoch gives25000 true updates per case,150000 total. Retain the
actual shared seed117 TRAIN order. FullTRAIN100k/DEV50k evaluation occurs at
updates0,1000,5000,10000,15000,20000,25000. Selection is fixed final25000, not minimum
DEV. Final restore requires exact prediction-array/metric equality and the
unchanged seven-owner native generation rule; all other full core bytes remain
exact against the actual post-load baseline. No HPO, early stop, tail/test or
new confirmation data are used; prior adaptive DEV/feature exposure is disclosed.

Fresh qualification for all six arms precedes any task update: full Y/dX/all11
parameter gradients against unchanged manual FP64 LN/gain/CST reverse and public
chunked reference, normal1/amplified1e24 cotangents with the existing exact
zero/material/relative/elementwise gates, independent31-Atom quadrature and scalar
Adam, 8ordinary vs4+load+4 and observed-first+7 vs8ordinary fullstate equality.
The120 synthetic updates are separate from150000 task updates; amplified
cotangents never enter the optimizer. Task models restart fresh0. No tolerance,
material coverage or numerical acceptance changes.

Actual first B4 observation, retained gradients/m/v/clocks, accepted-state
FP32 denominator reconstruction, seven milestone predictions, fixed256 forward
probes and final independent-copy backward probe retain their original scopes.
No kernel-intermediate capture, full intermediate raw replay or epsilon-causality
claim is added. See the A100 design for the exact shared formulas and safeguards.
Fixed-gain function-family equivalence applies only to NONE/bias-free-ReLU with
an unbounded head, not to intervening LN+gain with epsilon.

## Strict resource/source boundary

RTX4090 runtime pin:
`1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b`.
Python3.12.9/Torch2.11.0+cu126/CUDA12.6/Triton3.6.0/NumPy2.0.2,
driver570.133.20; NVIDIA GeForce RTX4090, CC8.9,128SM,25262096384 device bytes.
A100/3090 substitution or any runtime-pin mismatch fails closed. Compiler target
is8.9. The operator alone verifies the approved Hub profile and launches the job;
this consumer allocates/installs nothing, retries nothing and runs foreground.

External public source remains `97580582ed28c4e494bf5c2656bbe94a85c544c3`, manifest
`17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`.
TRAIN/DEV NPZ is `58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`,
data manifest `530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`.
The exact six regular inputs are consumer-source-manifest.json,
original-source-manifest.json, original-source.tar.gz, train-development.npz,
data-manifest.json and runtime-pin.json. No future/official/original confirmation
payload is staged. All committed consumer files and all inputs are verified.

```bash
python -B -m cst_research.experiments.higgs_multilayer_conditioning_rtx4090_driver \
  --consumer-commit FULL_REVIEWED_COMMIT \
  --cache-root /path/to/source-runtime-verified-cache \
  --output /path/to/fresh-diagnostic-output --timeout 7200
```

The existing4090 public975 cache can be reused only if its external archive,
manifest and full runtime binding match exactly. It contains compilation source,
not task state or qualification; source bytes are reverified. New consumer paths
are isolated by commit. Output must be fresh and separate. Timeout reaps the
owned process group and preserves partial evidence; no automatic replay/resume.

Raw cap6GiB and JSON cap64MiB apply independently to this new cohort. Preserve
immutable raw remotely and retrieve compact proofs/predictions plus a compressed
archive only if actual storage permits, or use a separately pinned peer CPU
saved-evidence auditor with CUDA off. No mandatory whole local decompressed copy.
Timing is DIAGNOSTIC, with preparation/calibration/admission/observations/eval/I/O
separated and inclusive components nonadditive. Block peaks are measured before
eval and include resident arrays/model/optimizer/nativebuffers/reserved cache;
not process memory. Cross-device comparison is descriptive, not hardware-only
speed adoption, Atom-order/capacity, LN necessity or formal NI. No automatic
followup. Evidence locator:
`$CST_RESEARCH_EVIDENCE_ROOT/higgs-multilayer-conditioning-rtx4090-20261106/`.

All131 inherited files except a single registry-count12→13 test metadata change
remain byte-identical, including all12 prior registry files. Historical running
A100 source/evidence and earlier A/B consumers are untouched.
