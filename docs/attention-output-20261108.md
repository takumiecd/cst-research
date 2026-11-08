# Full Attention projection and FFN feasibility at width512

**English** | [日本語](attention-output-20261108.ja.md)

Prepared after merging torchcst PR154 and cst-research PR34, before GPU execution.
Use torchcst `b14db87d34c1d6ba788517780528932473970505` (explicit LM levels and
stream64 cohorts, unchanged existing CUDA arithmetic). Consumer source is pinned
in the deployment manifest before execution. Use one owner-authorized RTX4090,
verified actual hardware/runtime, isolated source and SM8.9 build, standard nohup,
owned-process receipts and hash-verified collection. No A100 or Colab allocation.

## Fixed family

Width512,4 pre-LayerNorm blocks,8 heads (head dimension64),FFN expansion4,
context128,batch2,accumulation4,seed17. Fresh arms: Dense,FFN-only,QKV-only,
QKV+FFN,QKVO,QKVO+FFN. Their CST site counts are0,8,12,20,16,24 respectively.
All blocks participate. Q/K/V are separate logical maps, not one fused3N map.
Each Atom budget is `ceil(8 sqrt(d_in d_out))`:4096 per square projection,
8192 per FFN up/down. This checks the proposed linear budget, not a minimum
Atom count or asymptotic lower bound. Width1024/2048 are later stages.

QKVO includes Attention's residual output projection O. Its initialization target
variance is`.02^2/(2*4)`; Q/K/V and FFN-up target`.02^2`, FFN-down uses the same
residual variance as O. Use the public variance-scaling initializer at q0,
rational_zero_opening/birth.5/tau.001, amplitude_max1,w_c.3. Match every
unreplaced named ordinary Parameter across arms. Token embedding and vocabulary
output remain tied Dense, position embedding Dense; LayerNorm, activations,
causal SDPA, softmax and biases retain their operations. The other all-matrix
stages in the width plan are not declared implemented or validated here.

FP32 IEEE/TF32off, AdamW base LR3e-4, betas(.9,.95),decay0,clip1,whole Atom
LR multiplier16; fixed20M-token LR/data horizon. Requested1M tokens yields
976 updates/999424 actual tokens per arm. Use the exact prior TinyStories/GPT2
stream manifest SHA256
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
Train and DEV only; no confirmation scores, generation or LR/radius search.
Run failures stop the frozen attempt; no automatic retry or tolerance relaxation.

## Admission before training

Run scope/all-gradient GPU tests for6 shapes and both retention policies.
Existing `attention_qualification` is extended rather than adding a second
qualification or learner loop. For each training model kind, validate full CPU
FP64 Fourier Y/dX/all-Atom VJPs on its actual shapes at256 token rows, independent
bounded profile quadrature, same-gradient optimizer/serialization checks,
manual causal Attention Y/dX/all-QKVO gradients and exactly-zero future gradients,
then a real full-model update and resumed next update. Every CST site must receive
finite nonzero gradients and change. Qualify levels_stream64 on the24-site model
separately; actual learning uses levels. Require exact full config/head/backend
identity in addition to existing model-source/shape/kernel/recipe admission.

After training, independently restore every checkpoint and recompute DEV exactly;
check every trained CST layer against the full CPU FP64 VJP on bounded token rows
and replay the next task update from two independent restorations. Preserve
checkpoints on the GPU and record their hashes; compact logs are downloaded.

## Measurements and conclusions

Overlay train/DEV CE and PPL versus actual tokens and wall time. Record all update
timing samples after10 warmups and allocated/reserved peaks. The existing learner
clock starts after zero_grad/LR assignment and initial synchronization; it includes
data transfer, four forward/backwards, clipping, optimizer and final sync. Keep
that clock definition; do not equate it with the isolated Linear timer.
Evaluation, geometry/gradient probes and checkpoint IO are outside update timing;
wall time includes them. Setup/build cost is separate. Allocator peaks are not
whole-process GPU memory. Diagnostics are recorded for every Q/K/V/O/FFN role.

This is a single-seed development feasibility study. Loss decreasing establishes
learning, not convergence, quality equivalence or noninferiority. Keep fresh
Dense as a control, disclose all failures and all arms. Long3seed confirmation,
large width and vocabulary/embedding replacement require subsequent fixed stages.
