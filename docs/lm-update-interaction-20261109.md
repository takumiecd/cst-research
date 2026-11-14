# Task update direction and interaction protocol

English (canonical) | [日本語](lm-update-interaction-20261109.ja.md)

The owner authorized investigating multiple suspected CST learning limitations.
Use immutable, already audited task checkpoints, original TinyStories targets,
the exact width512 four-block model and frozen operator0d277ae. Dense is a control,
not a teacher. This is exploratory diagnosis and DEV screening, not confirmation.

Stage A: Dense, CST single-up, paired-middle4, scattered-mixed4, all8 and DSTall8;
checkpoint steps976/4882; three1024-target TRAIN batches each. Offset0 is the
actual next shuffled training batch, offsets32/64 use other batches at the same
restored optimizer clock. There are36 independent restored one-step counterfactuals.
Keep the20M schedule, Adam moments/clipping, public field-rate scheduler and DST
regular-step semantics. No selected step is a DST topology event.

Measure local projection output changes on **actual saved inputs**, weighted by
the original task cotangents. Report -sum(g*delta_y), descent cosine and relative
output change. The cotangents include accumulation1/4. They describe a local
first-order task effect, not Dense reconstruction or parameterization equivalence.
Evaluate exact same-batch CE after installing each projection's update alone,
the Dense remainder alone, allFFN, allup, alldown and allParameters. The difference
between joint gain and summed individual gains is finite-step nonlinear
interaction; its sign alone does not prove harmful long-term coadaptation.
Apply amplitude/fraction/output-center/input-center endpoints separately for CST,
and inspect boundary/unchanged fractions and per-field directions. Periodic
centers use wrapped tangent displacement. Validate a small Dense-remainder
direction by finite difference (epsilon.01,absolute2e-4/relative12% diagnostic
consistency bound, unchanged native derivative gates). Instrumented and unhooked
steps must reproduce Parameters/moments/counters exactly on every offset0 case.
Restore Parameters,buffers,optimizer and CPU/CUDA RNG after each temporary step.
Persistent source checkpoints are never overwritten.

Stage B: a bounded512-update continuation screen from the audited step976 state.
Retain Dense and DST controls and eight all8-CST configurations, using the same
training order and original20M horizons. Each starts with its own method's same
checkpoint/moments. CST multipliers (amplitude,fraction,output-center,input-center):
(32,32,32,32), (8,8,8,8), (16,16,16,16), (64,64,64,64), (8,32,32,32),
(32,8,8,8), (32,8,32,32), (32,32,8,8). Ordinary rate remains unchanged. Use the
public absolute field_lrs API and its scheduler anchor, without rescaling gradients
or moments. Control continuation must reproduce original DEV CE/top1 at1200/1400.
Retain all candidates, TRAIN curves, DEV CE/top1 at1200/1400/1488, final checkpoint,
moment/counter audits, geometry and every trained CST map's independent oracle.
These are same-state continuation interventions, not from-scratch LR selection or
evidence of an optimal field recipe. Do not automatically launch long retraining.

Use the reserved RTX4090, verify actual GPU/runtime/source/data/order identities,
fresh68 native regression tests and independent checkpoint oracles. Serialize
work; no new resource allocation. Timings and peak memory in this diagnostic are
instrumented and are not benchmark claims. Save to persistent NFS and retrieve
hash-verified compact evidence. On ambiguous/repeated Hub failures stop retries;
preserve partial results instead of changing source or tolerances. Negative and
inconclusive findings remain in the report. No public runtime/default changes.

Evidence:CST_RESEARCH_EVIDENCE_ROOT/lm-update-interaction-4090-20261109-01.

## Owner scope correction after execution

The owner clarified that the objective is analysis of CST behavior, not an improvement search. At the correction check the bounded ten-condition LR screen had already finished. Preserve those raw outcomes separately; do not select/adopt a recipe or launch more optimization. The [behavior report](lm-task-update-behavior-20261109-results.md) analyzes the36 restored task-update probes and existing signal evidence. The original stage-B plan above remains as execution history.
