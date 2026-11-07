# Gradual zero-opening FFN pilot

English (canonical) | [日本語](ffn-zero-opening-20261107.ja.md)

Three owner-authorized short learners vary only tau: .01, .03 and .1, all with
initial q=0. For the effective old rational bounds L0,U0, use
`g=a²/(a²+tau²)`, `L=U0-g*(U0-L0)`, `U=U0`; retain geometric q interpolation.
At zero L=U, at |a|=tau the relative gap is half. Strong amplitudes approach
the old interval; closure at amplitude_max is not imposed. Radius bounds remain
minimum .5, birth4, maximum50, upper_floor1, a_max1 and w_c.3.

Keep the [prior short protocol](ffn-closed-envelope-20261107.md): N256, four
pre-LN blocks, eight FFN projections, FFN4N, c8, seed17, context128,
batch2/accumulation4, FP32/TF32off, AdamW(.9,.95), decay0, clip1, baseLR3e-4,
whole-Atom LR16 times base. Stop at976 updates/999424 tokens, retain20M-token
LR/data-order horizon and existing hashed TinyStories data. Use DEV only.
Recalibrate every arm to identical Dense second moments, up std.02 and down
std.02/sqrt8, with error <=.5% and retained clipping diagnostics. Reject an
unreachable target; do not alter the target silently.

Freeze source/data/config before GPU execution. Verify baseline initial parameter
bytes/reports and reuse audited R12 Dense and rational-q0 curves without training
them again. Qualify actual both-direction native Y/dX/all-Atom VJPs, independent
quadrature, optimizer and full-task resume separately for each tau before learning.
Audit trained checkpoints, recompute DEV and verify retrieval afterward.

Save55 geometry samples, at initialization, updates1–5 and every20 through976.
Record current-amplitude-bin counts, radius/q means and quantiles and L/U for all
eight projections. Pool blocks by count within up/down direction; membership can
change. Plot learning curves and global/amplitude-conditioned radius and q.
The old q0 control lacks intermediate bin traces: retain only its recorded global
history and bin radius endpoint, with missing data explicit. Historical timings
are not fresh paired measurements. One-seed short DEV observations establish
neither convergence, optimal tau nor an Atom scaling law. No confirmation/HPO,
long extension or automatic extra arms are authorized by this protocol.
