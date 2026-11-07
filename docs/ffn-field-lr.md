# FFN field learning-rate pilot

English (canonical) | [日本語](ffn-field-lr.ja.md)

The owner stopped the unstarted long seed43/71 confirmation cases and requested
a short amplitude/center learning-rate comparison before implementing a public
API. All three seed29 widths completed, passed their trained audits and were
retrieved; that retained observation is not three-seed confirmation.

This new development experiment keeps the selected model/data recipe fixed:
four blocks, eight CST FFN projections, width256, expansion4,
`A=ceil(8*sqrt(d_in*d_out))`, seed17, initial q0, amplitude bound1, w_c0.3,
birth radius4, contiguous features. Attention, embeddings and head remain Dense.
Each fresh arm runs 976 updates / 999,424 actual tokens. Data order and LR retain
the common planned20M-token horizon, batch2/context128/accumulation4, FP32,
TF32 off, AdamW betas(.9,.95), eps1e-8, zero decay and global gradient clip1.
Dense and no-FFN controls run fresh under the same budget. This is a development
pilot; it does not reopen the Atom lower-bound search or the long seed family.

| Arm | Ordinary peak LR | Amplitude-direction peak LR | Input/output center peak LR | Polar outer time |
| --- | ---: | ---: | ---: | ---: |
| all16 | 0.0003 | 0.0048 | 0.0048 | 0.0048 |
| amplitude1-centers16 | 0.0003 | 0.0003 | 0.0048 | 0.0048 |

All rates follow the same warmup/cosine factor. The research-only
`FieldRateOptimizer` wraps exact PyTorch AdamW with the public `CSTOptimizer`.
Its base pre/post hooks scale Adam's normalized proposal before the declared
Polar retraction, retaining one Parameter and one moment table per Atom site.
The joint `(s,t)` displacement is scaled together; multiplying the raw gradient
would not express this rate change under Adam. Unit-ratio blocks are untouched.
Moment accumulation and ordinary parameter rates remain unchanged.

The activity law uses the same outer time in both arms. Activity still depends
on amplitude motion and amplitude itself, so equal outer time does not freeze
q/radii. This comparison isolates the configured proposal rate, not every
downstream consequence of that rate.

Before learning, require the independent CPU Adam equations/moment tests,
all16 agreement with stock CSTAdamW, native-vs-CPU parameter/moment/counter checks,
and actual four-block/eight-projection task serialization and next-step checks.
Existing exact-source rectangular Y/dX/all-Atom VJP qualification is also required.
After learning, audit each trained projection's Y/dX/all-Atom VJP, recompute CE,
and check serialized next-task-step parameters/moments/counters on disposable
copies. Preserve failed evidence; do not automatically retry/resume a learner.

Overlay all train/DEV curves, report final DEV and confirmation CE, synchronized
complete-update time and allocated/reserved peaks. The first two are mechanism
observations; confirmation is not used to pick the next rate. This short pilot
cannot establish convergence, an optimal rate, seed repeatability or a speed or
memory advantage. The legacy fixed-recipe reporter explicitly rejects these
optimizer variants, which require a separate observation report.

If the separated arm passes all correctness gates, remains useful versus no FFN,
and has final DEV CE within0.03 of all16 (or better), proceed to public API
implementation and its broader contract tests. Otherwise retain the negative
result and diagnose before adoption; a bounded intermediate amplitude rate may
be considered using DEV evidence. No public `field_lrs` API or default is adopted
by this research adapter. The intended public interface uses absolute field
rates, semantic fields declared by parameterization/geometry, and explicit
scheduler/checkpoint semantics rather than raw tensor column names.

## Audited short result and bounded follow-up

R12 completed all four arms and their trained audits. Final DEV CE was Dense
3.58196, no FFN4.03015, all16 CST3.67351 and amplitude1/centers16 CST3.84731.
Reducing amplitude proposal LR to1× worsened DEV CE by0.17381 relative to all16;
the prespecified0.03 API gate did not pass. This is slower learning within the
observed budget, not proof of a worse converged optimum. Both CST arms remain
better than no FFN. Correctness passed: sixteen trained projection VJPs, four
serialized next-task-step audits and all four parameter resume errors0. See the
[observation report](ffn-field-lr-r12-20261107.md) for curves and provenance.

Next test one intermediate condition: amplitude8× (absolute peak0.0024), both
centers16× and outer time16×. Keep N256/seed17/976updates and the same20M horizon.
Freeze new source and qualify this exact recipe before learning; retain failed
R12 evidence and the negative amplitude1 result. Run fresh Dense/no-FFN controls;
reuse the completed R12 all16 quality baseline only after proving unchanged
model/data/optimizer class implementations. Treat cross-job timing as an
observation, not a paired performance claim. No need to rerun another all16
training merely for unchanged code. The same DEV0.03/usefulness/correctness gate
applies before API implementation; this is a bounded development follow-up,
not an exhaustive LR search or a default/optimality claim.

## R13 result and owner-requested higher amplitude rates

R13 amplitude8/centers16 completed999,424 tokens and passed the trained eight
projection VJPs, CE and all three arm resume audits (maximum parameter error0).
DEV CE3.694387 differs from the retained all16 baseline3.673505 by0.020882,
passing the prespecified0.03 engineering gate. Fresh Dense/no-FFN DEV scores
exactly reproduced3.581960/4.030150. Ten archive members including the manifest
were retrieved/hash verified. This authorizes public API implementation; it is
not statistical equivalence or an optimal-rate/default claim.

The owner then requested32×/64× checks (2026-11-07). Run two additional short
amplitude-only conditions with peaks0.0096 and0.0192, keeping both centers and
Polar outer peak time0.0048. Same width256/c8/seed17/976updates/20M horizon;
fresh Dense/no-FFN controls. Reuse the audited all16/amp8 quality baselines after
unchanged model/data/optimizer-class source-byte checks. Qualify both exact
recipes before training; audit trained VJPs/CE/resume afterward. Record curves,
finite failures and resource observations; choose from DEV only and do not call
this an optimum or a paired cross-job speed comparison. The owner's explicit
request extends the earlier bounded follow-up. API implementation proceeds
locally while the single4090 runs these new arms sequentially; frozen operator
2b587 and past experiment sources stay unchanged.

## Correction: whole-Atom 32×/64× comparison (R15)

The owner clarified that the higher-rate question concerns the whole Atom, not
amplitude alone. R14 was an assistant interpretation error: retain its audited
amplitude-only results, but do not use them to answer the whole-Atom question.
R14 DEV CE was3.689207 at amplitude32/centers16/outer16 and3.698881 at
amplitude64/centers16/outer16; all16 was3.673505. These are separate contrasts.

Next run stock CSTAdamW with the entire Atom group LR multiplied by32 or64:
joint amplitude coordinates, both centers and the Polar outer clock all share
that rate. Keep ordinary parameters, including FFN biases, at the Dense rate.
No field-rate adapter or new public API is used. N256/c8/seed17,999,424 actual
tokens (976 updates), the same20M planned horizon, data/order/initialization,
and fresh Dense/no-FFN controls remain fixed. Qualify each exact rate against
CPU optimizer/moments/counters and serialized task resume before training;
audit trained eight projection VJPs, CE and resume afterward. Compare curves
with the retained all16 baseline. This is a bounded one-seed development check,
not an optimum or a paired cross-job speed claim. Use only the existing4090,
with one sequential scientific child; do not resume cancelled long cases.

## Completed whole-Atom follow-up (R15)

Both32/64 stock-CSTAdamW arms completed976updates and passed all16 trained
projection VJPs plus4arm next-task-step resume audits(maxerror0). DEV CE was
3.682039/3.687784 vs retained all16 baseline3.673505. Fresh controls exactly
reproduced prior quality;13archive members were retrieved/hash verified.
See the [whole-Atom curves/report](ffn-whole-lr-r15-20261107.md). This answers the
corrected question separately from amplitude-onlyR14; no optimal-rate or
convergence claim. Public field_lrs native/full-LM qualification is the next gate.

## Public API gate completed (R16)

The public absolute field_lrs API subsequently passed25 remoteCPU/8nativeCUDA
cases with0skips and8 actualfour-block/eight-projection configurations with64
trained projection VJPs, public/prototype task parity and default-constructor
checkpoint restoration. It merged through torchcst PR151 as
c3b68dcf6275f96d29a21ff2aa6e989d42de2cff.
[The API record](ffn-field-api-r16-20261107.md) distinguishes these disposable
correctness fixtures from the historical learning runs. No optimal-rate/default
or Graph/speed/memory claim is added. No further learning is launched here.
