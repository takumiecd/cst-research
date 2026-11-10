# M8 capacity with active-count-matched masked DST

English (canonical) | [日本語](lm-m8-active-count-20261110.ja.md)

Prepared, not a result. The owner's primary criterion is **CST Atom count A =
DST mask-one count per Linear**. Keep the existing dense weight+boolean-mask DST
implementation. Do not convert to COO/CSR. Raw parameter matching4A is a separate
auxiliary comparison; earlier results are preserved under their original criterion.

Fresh arms: M8 c16 (A16384/map); DST nnz8192/map and nnz16384/map, each with sparse
LR.001/.003/.01. Within each DST count, select minimum final DEV CE, breaking ties
by smaller LR, retain every candidate and label DEV selection. No confirmation
is used. Do not reuse the old4A-optimized LR as proof of tuning for nnz=A.

Reuse verified read-only Dense and M8 c8 A8192/map from
`lm-geometry-epsilon-capacity-3090-20261110-02`; all old4A DST is auxiliary.
Compare M8 A8192 with DST mask-one8192, and M8 A16384 with DST mask-one16384.
CST still has4A raw values (amplitude, fraction, two centers); the criterion
matches active representation units, not raw trainable scalar counts.
DST active positions include newly regrown zero-valued weights, because the
owner's criterion is the number of ones in the mask, not numerical nonzeros.

Keep4 blocks,width512,FFN512→2048→512, all8 FFN maps CST/DST, Dense attention,
embeddings and tied vocabulary. Unconnected M8,tau.003,fraction0,radius[.5,50],
cap32, ordinary LR.0003,CST4field LR32×, all epsilon1e-8, AdamW(.9,.95),decay0,
clip1, seed17,shuffle905,FP32/TF32off stay fixed. DST uses the unchanged row-balanced
RigL-style update at250/500/750, with active-fan variance correction and exact
row/global mask counts. Same DST masks/values at initialization across LR trials;
all ordinary weights match old controls. Larger-A CST keeps the same variance
target and seeded family, not duplicated atoms. Verify c8 raw identity with old
M8, c16 raw identity with the prior c16 family, and source/loop identity.

Data manifest5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01.
976 updates/999424 targets;20M schedule horizon; context128;2context×4microbatches;
DEV8192 targets. Evaluate0/200/256/400/600/800/976. Four fixed TRAIN contexts at
0/256/976 retain per-layer outputs/cotangents/gradients, geometry/moments and
checkpoints; confirmation untouched. Actual Adam clocks are976 CST and973 DST.

The runtime correction preserves ordinary c8 binary guards and adds a dedicated
RADIUS_LM_C16 build plus extension-binding hook. Only packed_rows.cuh's
conditional host guard and packed.py's extension binding differ among old runtime
files; prove default preprocessing/AST equivalence and unchanged numerical
kernels, initialization and optimizer. The earlier rejected source stays frozen.
Before learning require98 native/regression tests, actual3090/24GB/CC8.6, initial
full-A up/down FP64 Y,dX,raw-field gates (each maxabs≤3e-4 AND relativeL2≤1e-4,
older component failures retained,no relaxation), and exact serialized-next-update
and observer invariance for all7 fresh arms. Repeat final full-A gates for M8,
finite FP32 moments, clocks, topology invariants and256/976 checkpoint/DEV audits.

Compare speed/memory of the actual dense-mask implementation, including dense
weights, masks, dense-gradient buffers and optimizer state; record full-step
allocated/reserved peaks and timing, excluding preparation/probes/first10.
Do not present measured dense-mask usage as a COO/CSR implementation or a proof
about optimized sparse storage. Old/new-job timing is exploratory. Keep all
source/data/result hashes and failed evidence. No public/default adoption.

Entry: cst_research.experiments.lm_m8_active_count_run qualify/train/audit.
Explicit old controls and explicit --dst-run inputs with nnz checks are required
for plots. Learning curves show selected DST plus all LR candidates in a separate
HPO overlay. One seed and short DEV selection support exploratory capacity
comparisons, not sufficient-Atom, convergence, equivalence or unique-depth-cause
claims. More atoms helping cannot by itself establish why stacking deteriorates.
