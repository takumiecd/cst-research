# Combined CST QKV and FFN feasibility pilot

English (canonical) | [日本語](attention-combined-20261107.ja.md)

After individual Q/K/V runs trained successfully, the owner authorized combined
QKV and then QKV+FFN experiments. Compare three fresh RTX4090 arms: all Dense;
independent CST Q/K/V in every block with Dense FFN; the same QKV plus both FFN
maps CST in every block. Attention output O, LayerNorm, embedding, vocabulary
output and causal scaled-dot-product/softmax stay Dense/unchanged.

Use the [individual projection protocol](attention-qk-20261107.md): N256,four
pre-LN blocks,four heads,context128,FFN4N,c8,initial q0,seed17,batch2/accumulation4,
FP32/TF32off,AdamW(.9,.95),decay0,clip1,baseLR3e-4,whole-Atom LR16,
976 updates/999424 tokens and a fixed20M-token LR/data-order horizon. Same hashed
TinyStories tokens and8192-token DEV; no confirmation/generation selection,
LR sweep, output-O replacement or longer automatic extension.

Each square Q/K/V map has2048 Atoms; each rectangular FFN map has4096.
There are12 CST sites in QKV-only and20 in QKV+FFN. Use the existing rational
Polar envelope and Dense-matched initialization: std.02 for Q/K/V and FFN up,
std.02/sqrt8 for FFN down, error<=.5%, retaining clipping reports. Preserve
shared ordinary weights and named per-projection Atom streams exactly.

Qualify each full model separately. A receipt for12 sites cannot admit20 sites:
its explicit (attention_kind,ffn_kind) pair and both rectangular/square cases must
match. Before learning, check independent quadrature, actual native/all-Atom
VJPs for all three square seeds and both FFN directions, masked-softmax Y/dX/all
QKV Atom gradients and future-token gradient zeros, optimizer state and actual
four-block task/resume with all12/20 sites receiving finite nonzero gradients.
Reject initialization or correctness failures without silently changing targets.

Save complete-step time, preparation, allocated/reserved peaks, train/DEV and
per-site amplitude-bin radius/q histories outside update timing. Audit all
trained sites, exact DEV/final geometry and restored next-update Parameters and
optimizer state; retrieve/hash-verify compact evidence and preserve checkpoints
remotely. Earlier individual Q/K/V are historical context, not fresh timing
controls. These one-seed short observations concern feasibility, not convergence,
formal noninferiority, an Atom lower bound or a universal speed/memory advantage.
