# Fixed-c64 M8 FFN width and duration pilot

English (canonical) | [日本語](lm-width-scale-20261110.ja.md)

## Question and fixed design

Does the validation-quality gap to Dense shrink with width when each map has
A=64 sqrt(d_in d_out), and does longer task training change that gap? This is
an exploratory task-feasibility study, not an Atom lower-bound investigation.

| Width | FFN up/down dimensions | Atoms or DST mask-one entries per map | Atom density |
| --- | --- | ---: | ---: |
| 512 | 512↔2048 | 65,536 | 6.25% |
| 1024 | 1024↔4096 | 131,072 | 3.125% |
| 2048 | 2048↔8192 | 262,144 | 1.5625% |

Four pre-LN decoder blocks, eight heads, context128, GPT2 vocabulary50257;
all eight FFN maps are replaced. Attention, embeddings, normalization, biases
and tied vocabulary output stay ordinary Dense. Keep the task and next-token
labels; no Dense-matrix teacher. Seed17, named initialization, shuffle905,
1024 targets/update (two contexts × four microbatches), FP32, TF32off.

Reuse the story-disjoint TinyStories token manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
TRAIN has20,000,001 tokens; DEV8192 next-token targets is validation, run in
model.eval()/no_grad(). Confirmation32768 targets remains unevaluated. Selection
and diagnostic hypotheses use DEV; no formal equivalence or independent-test claim.

M8, physical spacing1, radius-envelope tau.003, fraction0, radius[.5,50],
amplitude cap32. Same variance initializer as the preceding width512 pilot:
ordinary standard deviation.02, residual output maps divided by sqrt(2*layers).
This fixed recipe is not width-invariant He scaling or a pure capacity intervention:
width and Atom count can change activation/gradient and initial-radius scales.
CST ordinary peak LR.0003, all four fields32×, AdamW(.9,.95),eps1e-8,
weight decay0, global clip1. Keep the inherited fraction-field LR policy.
No kernel redesign or CST LR search in this study.

## Short-run selection and long continuation

Every width gets fresh Dense LR candidates {.0001,.0003,.001}, CSTc64,
and dense-storage DST sparse LR candidates {.0003,.001,.003,.01}.
DST's ordinary parameters use peak LR.0003, so the selected Dense rate need
not be the ordinary rate used by CST/DST. Tune sparse LR independently.
Retain dense weights plus Boolean masks; no COO/CSR. Match mask-one count=A
per map, including newly grown zero entries. Record raw scalars and measured
memory separately. Keep row-balanced RigL-style250-update topology, active-fan
initial variance, survivor moments and zero-growth/reset rules.

Short stop976 updates=999,424 training targets. Select each width's Dense and
DST candidate by minimum final DEV CE, tie smaller LR. Retain all candidates
and flag boundary winners; this does not establish optimal learning rates.
Extend the two selected controls and CST at **all three widths** to4882 updates
=4,999,168 targets. Long runs restore short976 model/optimizer/RNG state and
use the exact same data permutation,20M-token cosine horizon and20M-token
DST topology horizon. Never reset warmup, LR or topology at continuation.
Short-selected LRs are frozen for long curves, not reoptimized at5M.

DEV at0/200/256/400/600/800/976, then every400 steps from1200, and4882.
Save checkpoints256/976/2441/4882. Four fixed TRAIN-context probes outside
step timing record per-layer output/gradient RMS and feature/residual summaries;
retain offset0 tensors. Save per-layer amplitude/radius/fraction and Adam scales.

Overlay Dense/CST/DST validation CE, PPL and next-token Top1 by training targets,
retain separate TRAIN curves and all short LR candidates. Report complete-step
synchronized time and peak allocated/reserved memory, excluding validation,
probes, checkpoint/oracle I/O and first ten steps of each process segment.
The +.01 CE and -.5pp Top1 margins remain descriptive guides. One seed/DEV
selection does not establish convergence, equivalence or a universal scaling law.

## Admission, preservation and execution

Use one owner-reserved RTX4090 (actual CC8.9 and memory recorded), no new GPU
allocation. A private width-specific route admits exactly the six directional
shape/count cohorts above, rows1..256; ordinary/c16/budget/public dispatch remains
unchanged. Numerical kernels are shared, not retuned. Run metadata round trips,
native retained/noncontiguous tests, full-A independent FP64 Y/dX/all-four-field
oracles initially and after short/long training. Fixed gate per tensor/field:
maxabs<=3e-4 AND relativeL2<=1e-4; retain original component failures too.
Check serialized next update exactly with an inert diagnostic probe; audit
checkpoint hashes, Adam clocks, finite FP32 moments, DST masks/zeros/counts,
final model digest and reloaded validation metrics.

Freeze tracked joint source and protocol before remote execution. Preserve raw
remote checkpoints/tensors and retrieve compact hash-verified evidence. Fail
closed on a gate failure and retain the failure; do not relax tolerances or
silently fall back to another operator. Per-arm16-hour safety bound. Prepare
all widths/short comparisons before long continuation; publish partial curves
with explicit incomplete status. No default adoption or main merge implied.

Runner: `python -m cst_research.experiments.lm_width_run --help`.
Plotter: `python scripts/report_lm_width.py --run RUN --output PLOTS`.
Study registry and ignored external evidence locator bind exact source commits,
archive hashes, remote directory and durable driver PID. Existing studies stay frozen.
