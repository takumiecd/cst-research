# Raw-amplitude initialization matched to Dense RMS

English (canonical) | [日本語](higgs-rms-amplitude-init.ja.md)

This follow-up changes only initial Atom amplitudes to match initial Dense
pre-ReLU RMS, keeping runtime gains equal to1 and normalization NONE. It is a
separate exploratory initialization condition, not a public default change.
The previous fixed0.1 width-aware study remains frozen at its original source.

Use the same N1024/H8/seven CST maps/A4096/seed17, ordinary legacy input/output,
TRAIN100k/DEV50k, batch4 FP32 TF32-off stockq1 learning rate0.0003 and fixed-final
one-epoch budget. Compare with the already admitted same-GPU Dense, old initial,
width0.1 initial and old-plus-gain observations. Different jobs' times remain
descriptive; one seed/epoch gives no convergence or noninferiority claim.

Before any task learning, use only the first256 TRAIN feature rows. For each
internal map in order, retain its original centers and standardized normal draws.
Choose a layer coefficient k so the amplitude std before clipping is
`k/sqrt(A/N)`. Apply the original symmetric ratio clip and zero-alpha polar
encoding. Amplitude-induced radius changes are intentional and clipping counts
are reported. Match the same initial Dense pre-ReLU RMS on these256 features;
the following ReLU uses the actual calibrated CST output. Post-ReLU RMS is
recorded but is not the matching criterion. No labels, DEV or heldout values
participate in initialization.

Search evaluates0,.05,.1,.2,.4,.8,1.6,3.2 until a sign bracket is found and uses
at most32 bisections, accepting actual measured relative RMS error <=0.5%.
It assumes no global monotonicity. An unreachable/discontinuous target fails
with retained search history; no gain fallback, clipping-rule change or runtime
bound expansion is performed. CPU FP64 public references evaluate actual FP32
candidate Atom values. Native FP32 B4 witnesses on all256 feature rows must also
match initial Dense within1%. Store coefficients, raw initial Atoms, physical
clipping/radius diagnostics, target/reference/native arrays and partial progress.

After initialization, freeze the initialization procedure and train the Atom
Parameters normally. RMS is not continuously constrained during training.
Fresh full-network VJP, independent quadrature, scalarAdam, normal/resume and
observer parity qualification precedes task learning. The20 synthetic network
updates and separate scalar8 are not part of the25,000 task updates. The new
runner supplies an explicit model factory to the unchanged numerical training
and qualification flow; the existing default factory behavior remains intact.

Save full TRAIN/DEV predictions and fixed first-B4 copy-only layer/parameter
gradient probes at0,1000,5000,10000,15000,20000,25000. Probe copies make no optimizer
updates and leave the source fullcore unchanged. Keep the real first-step
gradients, moments and accepted displacements. Preserve full evidence remotely;
independent saved-array/state verification is required before admitting results.
No primary/future confirmation or official test is scored.

Entrypoint: `python -m cst_research.experiments.higgs_rms_amplitude_run --help`.
This bounded follow-up uses the reserved RTX4090 while the original A100
width-initializer comparison continues under its unchanged source.
