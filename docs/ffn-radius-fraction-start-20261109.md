# Smaller tau and broader initial radius fractions

English (canonical) | [日本語](ffn-radius-fraction-start-20261109.ja.md)

New exploratory study after the completed cap1 tau pilot; do not overwrite or
reinterpret its frozen runs. The user requested smaller tau and broad initial
fractions, then explicitly chose a common wider amplitude bound with the same
Dense initial moment targets. Cap1 cannot realize the old up variance target at
fraction.5/1 with8192Atoms. Cap32 can; initial up amplitude RMS may be about10
at fraction1/radius50. This domain change requires its own native qualification.

Primary arms all use amplitude_max32, explicit levels_wide_amplitude backend:
(tau,fraction_init)=(.003,0),(.001,0),(.0003,0),(.003,.25),(.003,.5),(.003,1).
Add a fresh cap1/.003/fraction0 bridge and fresh Dense control. Seed17 only.
Width512,four pre-norm blocks,8heads,context128,expansion4,A8192 per FFN map;
attention/embedding/tied vocabulary head Dense. Same named ordinary weights,
center streams and Gaussian amplitude-noise streams. Match the public moment
initialization targets up variance.0004 and down.00005 within.005 relative error;
this matches expected energy, not exact per-example activation RMS. Record the
actual fixed-probe output RMS, loss and field gradients to expose correlations.

Phase1: all8 arms256 updates/262144tokens. Before examining learning, select the
lowest finite final DEV CE among the two smaller-tau arms; separately select the
lowest among the three broader-fraction arms (ties: smaller tau/fraction). Phase2:
restart those two plus the common cap32/.003/fraction0 baseline and Dense for
976updates/999424tokens each. This is12 bounded learning runs, not three-seed
confirmation. A failed correctness gate stops execution before learning; retain
all failures. A failed/nonfinite learning arm is ineligible and must be diagnosed
before expanding its family. No automatic extra sweep or public-default change.

Keep the previous20M-token order/LR horizon,200update warmup,batch2/context128/
accumulation4,AdamW baseLR3e-4,betas(.9,.95),eps1e-8,decay0,globalclip1.
Amplitude/centers LR16x and fraction LR1x remain fixed; broad fractions may move
slowly, so a short failure does not prove that a broad start is intrinsically bad.
FP32 Parameters, moments and physical rows; FP64 map intermediates; TF32off.
Reuse only the exact verified token manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
Use8192 DEV targets for selection, never confirmation. Old curves are context;
new cap32 and cap1 controls are fresh, not spliced historical speed baselines.

Before learning: CPU initialization feasibility, full operator GPU regression at
|a|<=32/both directions/rows7and256/radius caps4.7and50, encoded-row FP64
Y/dX/all-Atom VJP with unchanged atol1e-4/rtol4e-3, retained backward and exact
serialized next update. Qualify every exact task recipe; additionally check
serialization for broad fractions.5 and1. New source-bound receipts are required.

Record train/DEV learning curves versus updates/tokens/wall time, actual update
allocated/reserved peaks and scoped timings as in the old study (zero_grad/LR
setup/probes/evaluation excluded). Record fraction, radius, amplitude bins
extended through32, endpoint occupancy, field gradients and actual activation
RMS. Compare initial amplitude saturation too; RMS matching alone need not match
feature diversity or hidden correlations. Current-amplitude bins are not fixed
cohort trajectories. No convergence, causal optimal-radius or scale-order claim.

After training, load all four long checkpoints and reproduce DEV CE within1e-6;
check optimizer counters/finite FP32 moments and all Atom bounds using the actual
amplitude_max. For each CST long checkpoint, compare trained block0 up/down
Y/dX/all Atom gradients at7 noncontiguous rows against the encoded-row FP64
reference, preserving the same gate. Verify source/data/checkpoint/result hashes,
retrieve compact evidence, keep large checkpoints remotely and verify owned
processes inactive. Record source/operator/job IDs and first failures separately.
