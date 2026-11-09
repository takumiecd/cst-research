# Short task learning and finite-update fidelity: connected M16

**English** | [日本語](lm-connected-short-20261109.ja.md)

Prespecified before execution. The user authorized a short learning comparison
and checking the coordinate/radius update mismatch. This is an exploratory
one-seed task experiment, not confirmation or an Atom-count lower-bound study.

## Fixed comparison

TinyStories, the existing story-disjoint GPT2-tokenized data manifest, width512,
4 pre-norm causal blocks, 8 heads, context128, FFN expansion4, Dense attention,
tied Dense embedding/output. All eight FFN maps are replaced in the CST/DST arms.
A=8 sqrt(d_in*d_out)=8192 atoms per CST map. Dense and row-balanced RigL-style
DST (dense storage, 4A active scalars per map) are fresh controls using the
existing source-qualified DST algorithm and selected sparse LR0.003. No new LR
search, teacher matrix/output reconstruction or data selection.

Four fresh arms: Dense, DST-all, legacy unconnected M8-envelope, connected
M16-envelope with the qualified contiguous-row FP32 executor. The two CST arms
copy exactly the same raw initial Parameters, including amplitude and fraction;
all shared ordinary Parameters must match byte-for-byte across all arms. There
is no RMS recalibration of M16. Record initial output/CE differences. This
comparison changes both Fourier degree and connection; do not attribute its
learning differences to connection alone or to the executor's speed change.

Seed17, tau.003, radius[.5,50], fraction0, amplitude cap32. Ordinary AdamW LR.0003,
CST all four field LRs32x (including fraction), betas(.9,.95), epsilon1e-8, decay0,
global gradient clip1. Keep the existing 20M-token LR/topology horizon and shuffle
seed905. Four microbatches of2 contexts yield1024 targets/update. Stop at976
updates/999424 targets. DEV8192 targets at0/200/256/400/600/800/976; no confirmation
split or test-set selection. Track TRAIN CE, DEV CE/perplexity and next-token
accuracy. Keep actual synchronized uninstrumented whole-LM update times and
allocated/reserved peaks separate from setup, evaluation, checkpointing and probes.
DST topology events do not count as Adam updates. Its dense storage timings are
not sparse-kernel efficiency claims.

## Update prediction diagnostics

At actual updates1/256/976, record the first32 token rows of the first TRAIN
microbatch at block1 up/down (zero based). Save the local input, original task
cotangent (including accumulation1/4), pre/post raw Atom rows and native output.
Hooks must preserve the next complete update, model and optimizer state exactly
in serialized-resume qualification. They do not define a new update rule.

With bias fixed and that input X held fixed, compare Jv and
f(theta+v,X)-f(theta,X), separately for raw amplitude, radius_fraction,
output_center, input_center and all Atom fields. Amplitude's induced radius
change remains live. Use shortest torus displacements, actual post-projection
endpoints, and scales1/.1/.01. FP64 physical factors receive the exact encoded
FP32 physical map, retaining both map derivatives; casting/rounding can dominate
small finite differences and is recorded as a limitation. Evaluate native finite
changes separately on a scratch layer; require agreement with the reference
(max abs<=2e-4 or relative L2<=.01), without changing these gates after results.
Record prediction/finite cosine, finite/prediction norm ratio,
||finite-Jv||/||Jv||, and local task inner products/cosines. Zero prediction yields
null ratios. Task alignment and prediction fidelity are distinct measurements;
a local probe is not a whole-loss causal attribution.

## Gates and provenance

Before training: independent rectangular primitive quadrature Y/dX/all raw field
gradients for connected envelope; actual FFN full-A oracle at both directions;
serialized full-model/optimizer next-update equality and hook invariance for all
arms. Repeat the full-A oracle at each CST endpoint. Native full-model oracles
retain atol1e-4/rtol4e-3; independent primitive gates3e-4 are separate. Source,
archives, data manifest/order, hardware/runtime, commands, PIDs, checkpoints and
raw curves/probes are fixed and hashed. Preserve failed attempts and old evidence.
Use the reserved4090, run arms sequentially, no new resource allocation or public
default adoption. Integrate through separate PRs; leave main and old runs unchanged.
Results are pending.
