# CST query/key projection pilot

English (canonical) | [日本語](attention-qk-20261107.ja.md)

Following the completed CST-V pilot, the owner authorized Q and K experiments.
Compare fresh Dense, Q-only CST and K-only CST on the reserved RTX4090. Replace
the selected projection in all four blocks; keep the other two projections,
attention output, FFN, embedding and vocabulary output Dense. The causal
scaled-dot-product/softmax operation is unchanged. Q-only and K-only each use
c8/2048 Atoms per256×256 map, rational Polar, initial q0 and whole-Atom LR16.

Use exactly the [V pilot task/budget](attention-v-20261107.md): N256,4 pre-LN
blocks,4 heads,context128,FFN4N,seed17,batch2/accumulation4,eager FP32/TF32off,
AdamW(.9,.95),decay0,clip1,baseLR3e-4,20M-token LR/data-order horizon,
976 updates/999424 tokens and the same fixed hashed TinyStories tokens/DEV.
No confirmation/generation selection, automatic LR sweep or combined QKV run.

Copy complementary Dense rows from the original packed named-stream weights
exactly, preserving shared Parameters. Recalibrate each selected CST map to
Dense std.02 second moment within.5%, retaining clipping reports. Qualify each
Q/K recipe separately: independent scalar quadrature, actual square native
Y/dX/all-Atom VJPs, explicit masked-softmax integration gradients, future-token
gradient zeros, optimizer and actual four-block task/resume. Run Dense freshly
under the same frozen source/runtime. Previous V observations remain tied to
their original source and are shown as historical context only.

Save train/DEV curves, per-site amplitude-conditioned radius/q, preparation,
synchronized full-update timing and allocated/reserved peaks. Diagnostics are
outside update timing but inside instrumented wall time. Audit every trained
selected map, exact DEV/final geometry and restored next updates, retrieve and
hash-verify compact results, retain bulky checkpoints remotely. One-seed short
observations do not establish convergence, noninferiority or a universal gain.
