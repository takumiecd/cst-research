# All CST field learning rates:16/32/64/128×

English (canonical) | [日本語](ffn-all-fields-lr-20261109.ja.md)

The user authorized extending CST learning rates beyond16×. Change amplitude,
input/output centers and radius_fraction together to16/32/64/128× the base LR.
Keep all non-Atom parameters at the base LR in every arm, including Dense
attention, embedding, tied vocabulary head, norms and FFN biases. Fresh Dense
and16× controls in the same job; no old unpaired timing baseline.

Use the same initial fraction=.5,tau=.003,amplitude cap32,radius[.5,50],M8,
levels_wide_amplitude,public moment initialization as the prior direct fraction
study. No RMS calibration. TinyStories,width512,four pre-norm blocks,eight FFN
maps with8192 Atoms each(A=8√(d_in d_out)),heads8,context128,expansion4. Require
identical exact initial CST Parameter digests across all four LR arms.

Seed17 only, all five arms976updates/999424tokens, plus evaluate/checkpoint at
256updates/262144tokens. Batch2×128×accumulation4; AdamW baseLR3e-4,
betas(.9,.95),eps1e-8,decay0,globalclip1,200update warmup and common20M-token
schedule horizon. Peak CST field LRs=.0048/.0096/.0192/.0384. FP32 Parameters,
Adam moments and physical rows; FP64 map intermediates,TF32off,eager execution.
Same token order and data manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
DEV8192 targets only, no confirmation evaluation or automatic further sweep.

The existing operator gate may be reused only after exact torchcst bytes,
GPU/runtime and evidence-hash parity. Fresh consumer qualification must use each
actual declared amplitude/center/fraction LR; require outstanding-forward and
full-map VJP equality, actual fraction motion and exact serialized next update.
The prior helper hardcoded16× amplitude/center: extend it before freezing source.
CPU regression verifies all field rates and single schedule scaling at16–128×.

Observe losses, fixed-probe per-field gradients, activation RMS, amplitude bins,
radius/fraction distributions and actual per-Atom fraction net/path/update values
under the existing read-only observer. Record global pre-clip norm at sampled
updates. Compare actual changes, not raw gradient sizes alone. No causal claim
that larger LR repairs vanishing gradients, no assumption that Dense quality is
attainable with this Atom budget. A nonfinite run is retained as a failed arm;
do not retune/restart it as if prespecified. Stop on numerical qualification
failure and preserve evidence.

Audit all five final and256 checkpoints: exact DEV reproduction within1e-6,
finite FP32 Adam moments,counters,bounds and16 trained up/down oracle
checks (four CST arms×two checkpoints×two maps),Y/dX/all-Atom derivatives,7
noncontiguous rows,unchanged atol1e-4/rtol4e-3. Independently reproduce eight
CST observer net-change frames. Preserve checkpoint/source/result hashes and
verify retrieved compact evidence plus inactive owned PIDs.

Scope timing as before: accumulation/transfers/forward/backward/clip/optimizer,
excluding zero_grad/LR setup/probes/evaluation/observer. Record setup/wall and
PyTorch allocated/reserved peaks; no Graph/process-wide memory or universal
speed claim. Higher LR changes learned workload. One development seed gives
mechanism evidence, not convergence/noninferiority/universal optimum/default.
