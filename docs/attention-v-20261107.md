# CST V-projection feasibility pilot

English (canonical) | [日本語](attention-v-20261107.ja.md)

On one reserved RTX4090 compare a fresh all-Dense control with CST V projections
in all four attention blocks. Keep Q/K/output projections and the FFN Dense.
The causal scaled-dot-product/softmax operation is unchanged. Split Q/K weights
and biases are copied from the exact named-stream packed Dense initialization;
all other shared ordinary Parameters match exactly. V uses the public CSTLinear
with the existing rational Polar law, initial q0 and c8:2048 Atoms per256×256 map.
Recalibrate V to the same Dense second moment (std.02, error<=.5%). No new radius
shape or movement-dependent update is introduced on this device.

Keep TinyStories token manifest SHA256
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`,
N256,4 pre-LN blocks,4 heads,context128,FFN4N,seed17,batch2/accumulation4,
eager FP32/TF32off,AdamW(.9,.95),decay0,clip1,baseLR3e-4,whole-Atom LR16,
and the20M-token LR/data-order horizon. Each arm stops at976 updates/999424 tokens.
DEV uses the fixed8192-token development prefix. No confirmation/generation
selection, Attention-wide replacement, longer extension or automatic LR sweep.

Before training, qualify shared initialization, independent scalar quadrature,
native Y/dX/all-Atom VJPs for the actual square V map, explicit masked-softmax
attention Y/dX/Atom VJPs, exact future-token gradient zeros, optimizer moments,
and full four-block task/resume. Then audit trained V sites, exact DEV/final
geometry and restored next-update Parameters/optimizer state. Run the fresh
Dense control on the same GPU/runtime; historical curves are context only.

Overlay train/DEV curves. Record current-amplitude-bin radius/q/count histories
outside synchronized update timing, full-update time, preparation time and
allocated/reserved peaks. Diagnostics contribute to total instrumented wall time.
Freeze source/data/config before execution, retrieve/hash-verify compact raw
results and retain bulky checkpoints remotely. One-seed short observations do
not establish convergence, optimal capacity, universal scaling or speed advantage.
