# Actual-output-RMS matched broad-start study

English (canonical) | [日本語](ffn-probe-rms-start-20261109.ja.md)

The previous moment-matched broad-start run is retained as an initialization
diagnostic, not evidence against broad initialization: despite matching expected
Atom energy, actual FFN activation RMS reached4x Dense at fraction.5 and42x at1.
Structured weights and correlated/noncentered hidden features can invalidate a
trace-only independent-weight variance approximation. This is a mechanism
hypothesis consistent with the observed cascade, not a completed covariance audit.

The user's requested scale control is implemented directly. Starting with the
same public moment initializer, adjust only amplitude values at initialization.
Use a fixed TRAIN window (tokens0..127, no labels) and the same-seed Dense model's
eight FFN up/down output RMS targets. Calibrate in forward execution order.
For each projection, hold its actual inputs fixed, bracket/bisect a positive
amplitude multiplier with the same cap32. Fraction and centers remain bit-exact;
ordinary Parameters are untouched. After all eight sites, require every measured
RMS within.1% of its Dense target. A strict calibration failure aborts before
learning and restores all amplitudes/reports; do not relax the gate. Encoded
hierarchy transitions can make some scalar targets unreachable. No normalization
layer, runtime gain, persistent controller or label-dependent fitting is added.

Record per-site multipliers, before/after RMS, actual amplitudes and the original
moment report. Expected weight moments need not remain at their original target
after calibration; do not claim both controls simultaneously. Measure a second
nonoverlapping TRAIN window (tokens128..255) without using it to choose multipliers.
The control matches the fixed calibration window, not all future inputs or
unreplaced attention outputs. Preserve calibration inputs/data/source hashes.

Use common cap32, width512/four pre-norm blocks/8heads/context128/expansion4,
A8192 per FFN map, Dense attention/embedding/tied head. Six calibrated primary
(tau,fraction) arms: (.003,0),(.001,0),(.0003,0),(.003,.25),(.003,.5),(.003,1).
Add fresh Dense and fresh cap32/.003/fraction0 MOMENT-only bridge (no calibration).
Seed17. Phase1 all8 arms256updates/262144tokens. Before learning, fix selection:
best finite DEV among the two smaller-tau arms; separately best among the three
broad-fraction arms (ties: smaller tau/fraction). Restart those two, the calibrated
common baseline and Dense for976updates/999424tokens.12 total bounded runs;
no convergence, three-seed confirmation, universal optimum or causal-radius claim.

Keep the preceding20M-token schedule/order horizon,200update warmup,batch2 x128
x accumulation4, baseLR3e-4,betas(.9,.95),eps1e-8,decay0,clip1; amplitude/centers16x,
fraction1x. The factor may still move slowly; initialization and optimizer-rate
questions remain distinct. All raw Parameters/moments/physical rows are FP32;
map intermediates FP64; TF32off. Use the same verified token manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
DEV8192 targets only; never inspect confirmation for selection.

Operator0d277ae6 stays byte-identical to its68-check GPU qualification on the same
RTX6000 Ada/Torch2.9.1/cu128. Verify every operator source byte and prior result
hash before reusing that gate. Run fresh exact consumer/config qualifications,
including actual calibration, .1% final RMS checks, map/Jacobian parity, distinct
outstanding forwards, finite updates/fraction movement, and serialized exact
continuation for fractions.5/1. The new source/initialization mode must appear in
admission receipts; moment-only receipts do not authorize calibrated runs.

Keep all learning/radius/fraction/amplitude-bin/field-gradient plots and measured
activation RMS, including second-window ratios. Scoped update timing and allocated/
reserved peaks retain the previous definition, excluding zero_grad/LR setup/probes/
evaluation; record preparation/calibration wall time separately. No Graph or
process-wide memory claim. Audit all12 final checkpoints: reproduce DEV within1e-6,
check counters/finite FP32 moments/actual cap bounds and trained up/down encoded-row
FP64 Y/dX/all-Atom VJP at7 noncontiguous rows, unchanged atol1e-4/rtol4e-3. Retrieve
verified compact evidence; large checkpoints stay remote; verify owned jobs ended.
