# Task-LM output and gradient diagnostics

English (canonical) | [日本語](lm-signal-diagnostics-20261109.ja.md)

The owner requested per-layer output/gradient analysis for the task-learning gap.
This read-only post-hoc study inspects the17 Dense/CST/DST site-count and placement
models. It does not train a Dense teacher or reconstruct Dense weight matrices.
Initial observations motivated the probe: all8 CST block outputs initially near
Dense, later roughly2x, cotangents roughly0.5–0.7x on the original TRAIN prefix.
These observations alone do not identify a cause or prove gradient death.

At initial,976 and4882 steps, evaluate the original128-target TRAIN probe; at the
final state evaluate four separate128-row TRAIN contexts at offsets0/4096/8192/12288
(512 targets total); never exceed the qualified R<=256 native row contract. Record all
blocks,norms,QKV,O,FFN up/down and residual FFN: input/output distributions and
input/output cotangents. Statistics include mean,std,RMS,max,exact/relative zero
fractions,absolute quantiles from at most8192 sampled values. Covariance
participation/stable ranks use at most128 centered token rows and are not full
weight rank or representational-capacity bounds. Record explicit causal attention
logit dispersion,entropy/available-key entropy,max probabilities and Q/K/V RMS;
native SDPA forward remains unchanged. There are51 model states and102 probes.

All learning checkpoints and task labels are unchanged. Require completed,
audited source cohorts,matching GPU/runtime/data,checkpoint hashes,unchanged
Parameters and CPU/CUDA RNG. Reproduce original128 loss within1e-6 and FFN
raw gradient RMS within absolute1e-10/relative1e-5. These are diagnostic consistency
checks,not changes to native derivative tolerances. CPU tests check known uniform
attention/centered-rank answers and hook noninterference with task gradients.

Activation cotangents at matching layer positions have comparable units. Raw
Dense-weight versus CST-coordinate gradients do not; combine them with the
previously recorded realized function updates. Output scale,gradient attenuation,
update scale and feature diversity are separate candidate mechanisms. Zero
fractions for GELU are not ReLU dead-unit proofs. One seed/fixed TRAIN probes and
post-hoc observation do not establish causality. No optimizer update,confirmation
or new HPO. Save evaluator/identity/log/statistics on NFS and verify compact recovery.

Historical QKV evidence used width256 and an older Polar recipe,while QKVO used
width512 and a different operator/runtime. The O contribution was not isolated
in a same-cohort O-only comparison. A subsequent matched Q/K/V/O-only and QKV/
QKVO study with the current recipe is needed before claiming an intrinsic O/FFN
weakness. QKV select/encode attended content; O and FFN write transformed features
back into the residual stream in the actual model code. Dense complements,
normalization,input statistics,budgets and coupled optimization are hypotheses,
not an explanation already established. This FFN-cohort observer is not that
new attention-role training experiment.

Evidence:CST_RESEARCH_EVIDENCE_ROOT/lm-signal-diagnostics-4090-20261109-01.

The current radius-envelope physical native adapter is explicitly qualified for
width512 expansion4 rectangles and1..256 rows. Square attention maps require a
separate backend qualification before a current-envelope role trial; do not relax
that shape gate based on FFN evidence. The supplementary TRAIN contexts are
separate128-row calls and stay within the existing contract.
