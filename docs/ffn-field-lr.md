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
