# Atom capacity and epsilon pilot

English (canonical) | [日本語](lm-geometry-epsilon-capacity-20261110.ja.md)

Status: durable RTX3090 job launched, not a learning result. Frozen consumer
`76e80b6816269f63b02ff85ad262746f14754726`; [draft PR38](https://github.com/takumiecd/cst-research/pull/38).
CPU:234 passed/61 optional skips. Actual GPU correctness/qualification precede learning. This pilot tests two possible contributors
to all-FFN CST degradation: capacity and epsilon attenuation of small geometry
moments. It does not presume a remedy or a unique cause. M8 already exhibited a
quality gap; that does not exclude an additional M16 effect.

## Fixed task and arms

Use one NVIDIA GeForce RTX 3090 24GB, CC8.6. Do not combine its timing with earlier
4090 measurements. The public torchcst source remains at
`95f647f97749f61f5f5c15ac29932f908fb334ce`; no runtime changes/default adoption.
Research source is frozen before execution, with full file/archive SHA receipts.

TinyStories GPT2 tokens, manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
Use story-disjoint existing TRAIN and8192-target DEV; confirmation remains unused.
Four blocks, width512, heads8, context128, FFN512→2048→512. Replace all eight FFN
maps; attention, embeddings, tied vocabulary output and normalization remain Dense.
Seed17, named ordinary initialization and shuffle905 are shared.

| Arm | Profile | c | Atom Adam epsilon |
| --- | --- | --- | --- |
| dense | Dense | — | — |
| dst-c8 | row-balanced RigL reference | 8 | ordinary1e-8 |
| m16-c8-e8 | connected M16 | 8 | 1e-8 |
| m16-c8-e10 | connected M16 | 8 | 1e-10 |
| m16-c16-e8 | connected M16 | 16 | 1e-8 |
| m16-c16-e10 | connected M16 | 16 | 1e-10 |
| legacy-m8 | unconnected M8 | 8 | 1e-8 |
| dst-c16 | row-balanced RigL reference | 16 | ordinary1e-8 |

A=c√(d_in d_out):8192 or16384 atoms/map, four raw scalars/atom. DST receives exactly
4A active weights/map with active-fan variance correction. DST uses the source-
qualified row-balanced prune/grow algorithm, topology every250 updates, and sparse
LR.003 transferred from previous DEV tuning. No new tuning claim, especially c16;
it is a dense-storage accuracy reference, not a sparse performance implementation.

Each arm runs976 updates/999424 targets, with a fixed20M-token LR/topology horizon.
Four microbatches of two contexts give1024 targets/update. Ordinary LR.0003;
all four CST raw fields32×, AdamW(.9,.95), decay0, clip1, FP32 and TF32 disabled.
Only the packed Atom group's epsilon changes; ordinary epsilon remains1e-8.
Amplitude also sees that epsilon change. Do not claim its updates are unchanged.
Envelope tau.003, fraction0, radius[.5,50], amplitude cap32. Same coefficient
means exact raw initial bytes across epsilon variants/M8; different coefficients
use the same variance target and seeded family, without post-hoc RMS calibration.

## Pre-execution gates and evidence

On actual3090, rerun the four relevant native suites, then check representative
block1 up/down with full-A FP64 factor reference, B3. Require maxabs≤3e-4 AND
relativeL2≤1e-4 separately for Y,dX and each raw field. Retain counts of the older
componentwise criterion's failures; do not relabel that criterion as passed.
This inherits the previously amended admission rule but fixes it before this
pilot. Do not relax it again. Retain any failed run.

All eight arms require exact serialized next-update equality including optimizer
moments, epsilon and clocks, with observational hooks enabled on one branch.
DEV CE/PPL/top1 at0,200,256,400,600,800,976. Retain fixed TRAIN activations and
unscaled/unclipped cotangents at0,256,976 for offsets0,4096,8192,12288; tensor
retention at offset0. Record radius/fraction/geometry moment scales and final
native correctness, finite FP32 moments, expected Adam clocks976 (DST973), and
reload-audited final DEV. Timings synchronize complete steps, exclude first10,
and exclude evaluation/probes/checkpoint/oracle preparation. Report both peak
allocated and reserved bytes; no instrumentation speedup claim.

A separate CPU FP64 2×2 comparison uses identical saved raw rows and original
TRAIN X/cotangent from the prior final all-cst and pair12 checkpoints: M8/M16 ×
unconnected/connected. Use first up to2 existing deterministic witnesses per
geometry category at all12 selected sites; reproduce old connected-M16 VJPs.
This isolates local derivative sensitivity, not a retrained M8 causal attribution.

Plots must include Dense plus both explicit compatible DST inputs, with active
budget labels. A single-seed short pilot cannot establish convergence, equivalence,
a universal Atom lower bound or a new default. Lower epsilon can amplify FP32
noise and cannot create an absent true derivative; native/FP64 evidence stays
separate. Retain checkpoints and raw tensors remotely, retrieve compact hashed
reports, and preserve source/config identities and failures.

Entrypoints: `cst_research.experiments.lm_geometry_trial_run` (`qualify`, `train`,
`audit`) and `scripts/compare_lm_profile_sensitivity.py`. Supply the exact data
manifest SHA, explicit arm, qualification receipt and an unused output directory.

## First3090 attempt stopped before learning

Job `lm-geometry-epsilon-capacity-3090-20261110-01` passed84 native tests
and connected-M16 c8 full-A up/down oracles. Dense/DST-c8 serialized next-update
checks passed. The M16 c8 model completed its first256-row update, then failed
in the128-row probe backward with `CUDA error: unspecified launch failure`.
Afterward nvidia-smi reported no devices while the Hub remained ready. The
cause is unresolved; neither a kernel fault nor a host/driver fault is established.
The user restarted the3090 server. A synchronized qualification-prefix
reproduction is prepared in `scripts/reproduce_lm_geometry_probe.py`.

No learning arm started. Do not call the full-model gate passed or infer quality.
Original failed evidence remains unchanged; seven compact payload hashes were
verified. Failed archive SHA256:
`5166cfe813cd7284c4d8b0a775613e00832d78d0a4b3560be65114287d767501`.
