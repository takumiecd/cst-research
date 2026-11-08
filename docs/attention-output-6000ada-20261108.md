# CST replacement pilot on RTX 6000 Ada

**English** | [日本語](attention-output-6000ada-20261108.ja.md)

The owner authorized this four-arm learning comparison after the separate
Linear hardware pilot. Use one actual NVIDIA RTX 6000 Ada Generation, 48 GB,
CC 8.9, Torch 2.9.1+cu128 / CUDA 12.8. Freeze operator
`b393ac653f3b3893c378acda1c1752919c1539d1` and the consumer commit in the
deployment manifest. Use the existing durable launcher, qualification,
`ffn_run` learning loop and trained-checkpoint auditor. Do not change an active
snapshot, restart unrelated jobs, search hyperparameters or evaluate confirmation.

## Fixed comparison

Fresh models, in this fixed execution order:

| Arm | Attention projections | FFN | CST sites |
| --- | --- | --- | ---: |
| Dense | Dense | Dense | 0 |
| FFN | Dense | CST up/down | 8 |
| QKVO | CST Q/K/V/O | Dense | 16 |
| QKVO+FFN | CST Q/K/V/O | CST up/down | 24 |

All four Transformer blocks participate. "All CST" in this pilot means all
linear projections inside the blocks. Token/position embeddings and the tied
vocabulary output stay Dense; LayerNorm, biases, activations and causal SDPA
retain their operations. This is not an embedding/vocabulary replacement test.

Width 512, 4 pre-LN blocks, 8 heads, context 128, FFN expansion 4, batch 2,
accumulation 4, seed 17. Each arm requests 1M tokens: 976 updates / 999424 actual
tokens, with the same fixed 20M-token data-order and learning-rate horizon.
Use the frozen TinyStories/GPT2 manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
Train and DEV only. Common ordinary named Parameters use the same initialization.

Atom budget is `ceil(8 sqrt(d_in d_out))`: 4096 for square maps and 8192 for FFN
up/down. Use levels, FP32 / TF32 off, variance scaling with target std 0.02 for
Q/K/V/up and `0.02/sqrt(8)` for O/down, q=0,
rational_zero_opening / radius_birth=0.5 / zero_opening_scale=0.001,
amplitude_max=1 and w_c=0.3. AdamW: base LR 3e-4, betas (0.9, 0.95), decay 0,
global clip 1; the whole Atom group uses LR multiplier 16. This is the existing
selected recipe, with no per-arm tuning. The single width does not establish
linear scaling or a minimum Atom count.

## Admission and preservation

Reuse the previously retrieved, hash-verified 223-regression record from the
same RTX 6000 Ada only after verifying exact operator source files, runtime and
log hash against its receipt. Keep that provenance explicit; this is not a new
regression run. Use a fresh job-local native build and dependency environment.
Run all three required model qualifications and strict learner admission anew
on this device before any learner: actual R256 shapes against independent CPU
FP64 Y/dX/all-Atom VJPs and profile quadrature, manual causal Attention, exact
future-token isolation, every CST site updated, and serialized next-update and
optimizer-state checks. Dense uses the qualified common operator recipe.

After each arm, recompute DEV exactly from its checkpoint, check every trained
CST map against bounded-row FP64 VJPs, and compare the next real task update
from two independent restorations, including optimizer state. Preserve failure
and partial evidence; stop the attempt on a failed gate, without automatic retry
or tolerance relaxation. Allow up to 60 minutes per learner and 25 minutes per
qualification/audit; these are execution ceilings, not completion estimates.

Keep datasets, source, checkpoints and native builds on the cluster. Retrieve
and hash-verify compact receipts, curves, geometry, summaries and audit logs.
Record exact source/data/configuration identities, checkpoint hashes, command
times and owned processes. Do not infer completion from a detached launcher.

## Measurements

Overlay every arm's train/DEV CE and DEV perplexity against actual tokens and
recorded elapsed time; retain every recorded point. Report median update time,
tokens/second, all post-warmup samples, and peak allocated/reserved memory.
The existing learner timer excludes zero_grad/LR assignment and its initial
synchronization, but includes transfer, four forward/backwards, clipping,
AdamW and final synchronization. Exclude the first 10 updates. Evaluation,
geometry, probes and checkpoint IO are outside update timing; recorded wall
time includes them. Setup and qualification are separate. Memory is allocator
training peak, not total process GPU usage. Do not equate this timer with the
isolated Linear benchmark's timer.

Fixed execution order on one device is an exploratory resource observation,
not independent hardware replication. One seed / approximately 1M tokens shows
learning behavior; it does not establish convergence, noninferiority or the
speed of a fully trained LM. Publish all four arms and any failure. Keep this
cohort separate from the unavailable earlier 4090 attempt and the stopped L4
learning plan.
