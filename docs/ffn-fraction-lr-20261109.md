# Initial fraction .5: direct field-LR comparison

English (canonical) | [日本語](ffn-fraction-lr-20261109.ja.md)

The user approved holding initial radius_fraction=.5 while changing only its LR
multiplier1,4,16. No RMS calibration is used. Probe-RMS initialization remains an
unexecuted optional preparation. This follows the prior observation that all
Atoms moved from.5 but their signed changes canceled, with mean absolute change
.00384 at256updates. That is not a vanishing-gradient result.

Freeze tau.003,amplitude_max32,explicit levels_wide_amplitude backend,width512,
four pre-norm blocks,8heads,context128,expansion4,A8192 per FFN map. Keep Dense
attention/embedding/tied head, public moment initializer up variance.0004/down
.00005, all named seeds and input order. Require exact initial Parameter hashes
across the three CST arms. Initial output RMS need not match Dense; it must be
identical across the LR arms. Do not reinterpret the old broad-start diagnostic
as a scale-matched comparison with Dense.

Seed17 only. Fresh1x/4x/16x runs plus Dense each976updates/999424tokens; additionally
observe/evaluate/checkpoint step256/262144tokens. Same20M-token order/schedule,
200update warmup,batch2 x128 x accumulation4,AdamW baseLR3e-4,betas(.9,.95),eps1e-8,
zero decay,globalclip1. Amplitude/centers16x in every CST arm; fraction1x/4x/16x
means peak absolute rates.0003/.0012/.0048 and the same schedule. FP32 Parameters,
moments and physical rows; FP64 map intermediates,TF32off. No LR selection changes
training budgets, no automatic further sweep/three-seed confirmation/default change.

Use the verified token manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
Only8192 DEV targets, never confirmation. Treat old1x curves as context; rerun1x
and Dense under the same new source. Reuse the operator's68-check GPU result only
after exact source/GPU/runtime/hash verification. Run fresh consumer qualification
for each LR, including full map/Jacobian, outstanding saves, actual fraction
updates and serialized exact next model/Adam state. Observer must preserve
Parameters/gradients/moments/RNG and source-bound admission must include it.

Record actual per-Atom post-projection fraction delta at every update, net change
from initialization, positive/negative/unchanged shares and cumulative absolute
path. At step0,1..5,every20,256,976, also record physical radius deltas and the exact
identity r(a_t,f_t)-r(a_0,f_0)=[r(a_t,f_t)-r(a_t,f_0)]+[r(a_t,f_0)-r(a_0,f_0)].
The first term is the fraction component at current amplitude; the second is the
amplitude-bound component at initial fraction. This is an algebraic decomposition,
not a causal attribution of the training trajectory. Record endpoint occupancy,
fixed-probe field gradients and a FP64 analytic Adam proposal estimate from the
stored FP32 moments/current field LR. That estimate precedes bounds/rounding;
do not label it an exact raw-proposal or clipping-rate trace.

Keep learning curves against updates/tokens/wall time, radius/fraction distributions,
amplitude bins and actual activation RMS. Observer copies/decoding/reductions run
outside scoped update timers; wall time includes them and timing/memory are
exploratory, not primary kernel adoption. Timing retains the preceding scope
(accumulation/transfers/forward/backward/clip/optimizer, excluding zero_grad/LR
setup/probes/evaluation). No Graph or GPU-process-wide memory claim.

Audit all four final checkpoints and the256-update checkpoints: reproduce DEV
within1e-6, validate counters/finite FP32 moments/actual cap bounds, and trained
up/down encoded-row FP64 Y/dX/all-Atom VJP at7 noncontiguous rows with the unchanged
atol1e-4/rtol4e-3 gate. Preserve raw/checkpoint/source/result hashes and failures;
retrieve verified compact evidence, retain large checkpoints remotely and verify
owned processes inactive. This is one-seed mechanism exploration, not convergence,
universal best LR, noninferiority or a public default recommendation.
