# Closed-envelope FFN pilot

English (canonical) | [日本語](ffn-closed-envelope-20261107.ja.md)

[Completed observations and learning/radius/q curves](ffn-closed-envelope-results-20261107.md).

This exploratory follow-up tests the envelope, not the Atom lower bound or an
Attention replacement. It is separate from the completed/stopped prior campaign.

Keep N256, four pre-LN blocks, FFN4N, eight CST projections, c8 in
`A=ceil(c*sqrt(d_in*d_out))`, seed17, context128, batch2/accumulation4,
FP32/TF32off, AdamW(.9,.95), zero decay, global clip1, ordinary peak LR3e-4 and
whole-Atom/outer peak LR4.8e-3. Keep the20M-token LR/data-order horizon but stop
each new learner after976 updates/999,424 tokens. Use the existing hashed
TinyStories tokens, tokenizer and story-disjoint DEV split. Do not use the
confirmation split for pilot selection.

New intended arms are rational/q1, closed/q0 and closed/q1. The closed law is
`b=clamp(1-(a/a_max)^2,0,1)`, `L=.5+49.5*b^4`, `U=.5+49.5*b`. Keep a_max1,
and the existing Polar activity and detached-radius task differentiation.
Reinitialize each arm to the same target second moment (up std.02, down
std.02/sqrt8); retain error and clipping diagnostics, do not force identical
physical amplitudes across different radius maps.

Pre-execution admission found rational/q1's up projection did not reach the
target in the bounded amplitude search. Record that rejection with no learner
and no silent unmatched fallback. The two closed recipes initialized all eight
actual projections within0.5% on CPU. Those preparation records alone are not GPU or learning evidence.
Each admitted recipe requires independent quadrature, both actual native
directions/all-Atom VJPs, controlled optimizer state/resume and actual four-block
task/resume checks before training.

Reuse the audited R12 Dense and rational/q0/all16 curves; do not rerun those
baselines. The saved stock-vs-wrapper comparison has zero differences in53
common train/gradient/LR points and six DEV values. Verify the data, model,
schedule, seed, initial reports and source boundary before overlaying. Reused
timings are historical observations, not a fresh paired speed comparison.

Save current-amplitude-bin radius/q mean,p10,p90,count, bound values and amplitude
at initialization, updates1–5 and every20 updates through976. Bin membership
changes over time. Bins: [0,.001), [.001,.01), [.01,.03), [.03,.1), [.1,.3),
[.3,.6), [.6,1]. Empty bins remain missing, not zero. Record each of eight sites
and pool only with count weighting. Old q0 has no intermediate bin trace: show
its saved global history/endpoints only; do not fabricate a bin trajectory.

The diagnostic pass runs outside synchronized update samples. Record full
update samples and allocated/reserved peaks, but treat instrumented wall time
and cross-job historical overlays as exploratory. Retain raw checkpoint/source
evidence and exact hashes. Select further shape or duration experiments only
after reviewing this bounded pilot; no automatic LR/capacity/three-seed sweep.
