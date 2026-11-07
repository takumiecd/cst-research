# Minimum-based Polar lower-envelope screening

English (canonical) | [日本語](polar-minimum-a100-20261107.ja.md)

The owner selected the same TinyStories small-LM FFN task for two Polar-derived
research stages: lower-envelope shape first, then a movement-dependent update.
This first stage changes only the bounds. It does not implement stage two.

The existing public declaration already expresses the proposed formula: use
`radius_envelope="rational_zero_opening"`, `birth=minimum=.5`, and
`zero_opening_scale=epsilon`. Then
`L=.5+(U0-.5)*epsilon²/(a²+epsilon²)`, `U=U0`. Zero amplitude closes exactly;
farther away L rapidly approaches .5. This is a new parameter recipe, not a new
native kernel or a copied runtime implementation.

On the reserved A100 MIG3g.40GB, compare fresh rational/birth4 control against
epsilon=.001 and .003 at birth.5, all q0. Each gets400 updates/409600 tokens,
with the same20M-token LR/data-order horizon. Keep N256,4 pre-LN blocks,FFN4N,
c8/4096 Atoms per projection,8 CST sites,seed17,context128,batch2/accumulation4,
FP32/TF32off,AdamW(.9,.95),decay0,clip1,baseLR3e-4 and whole-Atom LR16.
Attention/embed/output remain Dense. Same token manifest SHA256:
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
Use DEV only; no confirmation/generation selection or automatic long extension.

A100's current Torch2.6.0+cu126 is a separate runtime from RTX4090's2.11.0.
Qualify every arm on A100 before learning: independent quadrature, actual both
FFN directions/all-Atom VJPs, optimizer moments/resume and full task/resume.
Recalibrate all eight maps to identical Dense second moments within.5%, keeping
amplitude/clipping reports. Reject an unreachable target without fallback.

Record learning/DEV, every-update shortest torus center displacement divided by
the previous physical radius, current-amplitude-bin radius/q/counts, gradients,
complete-update timing, preparation and allocated/reserved peaks. CPU diagnostics
run after synchronized timing and affect instrumented total elapsed time.
The center-motion trace is observation only; it changes no update or RNG state.
Audit trained checkpoints and restored updates; retrieve/hash-verify compact
results, retaining checkpoints/data remotely. One-seed screening is not evidence
of optimal epsilon, convergence, minimum capacity or a cross-device speed gain.
Stage two needs its own formula, fixed comparator and qualification record.

Attempt01 passed both native linear directions but stopped before learning at
the Torch2.6 AdamW private-signature seam. Attempt02 retains Torch2.6 and uses
the explicit version-compatible CSTAdamW initialization bridge, tested against
PyTorch AdamW with AMSGrad on/off and serialized resume before all recipe gates.
The attempted isolated2.11 install was stopped after external-index DNS failures;
it did not change the system environment or produce a qualified runtime.
