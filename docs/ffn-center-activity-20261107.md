# Center-driven Polar activity: bounded FFN pilot

English (canonical) | [日本語](ffn-center-activity-20261107.ja.md)

The owner authorized a first A100 comparison after questioning whether physical
radius should be driven by amplitude-side activity. Preserve (s,t), the angular
amplitude proposal, L/U and moment-matched initialization. Change only the
activity source and its explicitly screened gain. This is not adoption of a
new coordinate embedding or sigmoid/clip/smoothstep target controller.

Use the existing TinyStories task: width256, four pre-LN blocks/four heads,
context128, FFN4N, seed17, A=ceil(8 sqrt(d_in d_out))=4096 per FFN map,
batch2/accumulation4 (1024 tokens/update), eager FP32/TF32off, AdamW(.9,.95),
weight decay0, global clip1, baseLR3e-4/whole-Atom LR16 and the fixed20M-token
LR/data-order horizon. Train each arm400 updates/409600 tokens. Use DEV8192
tokens only; no confirmation or generation selection.

The fixed envelope is the previously development-selected minimum-based
rational_zero_opening recipe, birth=minimum=.5, upper_floor1, maximum50,
epsilon=.001, initial p0. Its prior400-update observation motivated this choice;
it is not a prespecified heldout discovery or an optimal envelope claim.

Run a fresh Dense control and four CST arms: amplitude activity/gain1, then
center_chord/gain1,64,4096. All CST arms use the same explicit Torch radius eager
update Algorithm v1 and the same native Linear Plan, avoiding an update-backend
confound. The three gains span activity scales; none changes the base or field
learning rates. Gains are exploratory, fixed before new learning. Do not extend
this batch automatically or choose a heldout winner.

Center activity is E=.5[(shortestDeltaMu_out/r_out_old)^2+
(shortestDeltaMu_in/r_in_old)^2], with S*=clip(S+gain*E,1,4) and original exact
radial decay; S=||s,t||² and p=(S-1)/3. No extra EMA/dormant expansion or
optimizer-time division. Small sustained movement need not send p to one. This
retains the representation while changing its update policy. Raw displacement,
learning rate and support normalization affect its interpretation.

Before learning, verify actual GPU identity/memory, source and data hashes,
initial Parameter/calibration equality across the four CST arms, all CPU/CUDA
update tests, independent quadrature, native rectangular Y/dX/all-Atom VJPs,
mixed optimizer moments/resume and actual four-block task/resume for each recipe.
Any failure stops the frozen attempt; do not edit a snapshot in place.

Record training/DEV curves, synchronized complete optimizer-update time,
allocated/reserved peaks, setup/wall time, activation/gradient probes and
amplitude-conditioned radius/p/L/U counts/quantiles. Observe shortest center
motion every update read-only, outside timing. Retain26 geometry samples per CST
arm. Distinguish p/q naming in existing diagnostics from Atom density.

After learning, check actual trained native Y/dX/all-Atom VJPs at all eight CST
sites per arm, exact DEV recomputation, exact restored next-task Parameters and
optimizer state, and final geometry/checkpoint agreement. Retrieve compact raw
evidence and verify its manifest/hash while leaving source/checkpoints intact.
No Graph claim: this research update explicitly rejects capture. Time/memory
are descriptive measurements of the bounded eager cohort, not an adoption result.

Source revisions and the immutable data SHA are in the
[registry record](../src/cst_research/studies/ffn-center-activity-20261107.json).
Public runtime changes remain in torchcst; consumers/qualification/controllers
remain here. Resource commands and raw receipts stay in ignored evidence.
