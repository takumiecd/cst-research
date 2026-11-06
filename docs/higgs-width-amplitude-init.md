# Width-aware amplitude initialization

English (canonical) | [日本語](higgs-width-amplitude-init.ja.md)

Completed bounded execution and independent saved-evidence audit. See [results and limitations](higgs-initialization-results.md). The protocol below retains its original design; preparation/future-tense statements are historical.

This separate exploratory study compares the requested raw amplitude standard
deviation `0.1/sqrt(A/N)` against `0.1/sqrt(A)` and the previously measured fixed
gain recipe. It does not use the unexecuted He/self-energy prototype.

The model has eight hidden layers of width1024, seven bias-free internal CST
maps with4096 Atoms each, and trainable ordinary Dense input/output layers.
Each GPU runs four fresh paired arms: Dense legacy control, old CST initializer,
new CST initializer, and old CST initializer plus fixed gain. All arms use
seed17, legacy ordinary initialization, FP32/TF32 off, eager true batch4,
stockq1 AdamW/CSTAdamW, learning rate0.0003 and the same saved seed117 order.
The budget is one epoch,25,000 updates per arm. Selection is fixed final,
without HPO, early stopping or best-DEV selection. RTX4090 and A100MIG3g.40gb
are separate hardware cohorts with exact runtime pins.

## Intervention

The unchanged public975 factory supplies the paired normal draws and centers.
In the new arm, multiply each initial polar first component by sqrt(N), apply
the same symmetric amplitude-ratio clip, and reconstruct the second component
as sqrt(1-ratio²). At N1024 the factor32 is exact in FP32: the normal standard
deviation changes from0.0015625 to0.05. The initialized alpha remains the same
zero-alpha encoding, up to its normal FP32 encoding error. Coordinates and all
ordinary tensors are unchanged. Radius changes induced by amplitude are
intentional; this is not a geometry-preserving intervention.

The new and old raw-initializer arms have unit gains and no normalization.
The gain control uses exactly the prior sequential calibration on the first256
TRAIN feature rows, matching the same initial Dense pre-ReLU RMS. No labels or
DEV are used for calibration. This gain is not the later feature-free He gain.
The public runtime, operator, differentiation and optimizer are unchanged.

The runtime architecture contracts retain their original case names; every
checkpoint and report's experiment identity separately records its actual arm
and initialization policy. New initialized values are restored through the
unchanged native model/optimizer load path, not a special restored initializer.

## Measurements and admission

Fresh full-network Y/dX/all-parameter FP64-reference VJPs, unchanged numerical
gates, independent31-Atom quadrature, scalarAdam, eight ordinary versus4+load+4,
and observed-first+7 versus ordinary8 full-state equality must pass for all four
arms before task learning. Amplified cotangents are diagnostic, never updates.
The80 synthetic qualification updates are separate from100,000 task updates.

TRAIN100k and DEV50k are evaluated at updates0,1000,5000,10000,15000,20000,25000.
At every point, the same first TRAIN batch4 is probed on an independently loaded
copy: save every layer's raw forward values/upstream gradients and parameter
gradients, perform no optimizer update, and check the original full core remains
unchanged. The real first training step additionally retains raw gradients,
accepted Parameters, moments, clocks and displacements. Accepted-state Adam
denominators are reporting reconstructions, not captured native intermediates.

Synchronized bare-step intervals record time and peak allocated/reserved memory;
probe, evaluation, calibration and IO costs are separate. Keep complete raw
evidence on the server under a6GiB raw/64MiB JSON cap, with source/data/runtime
hashes, final checkpoints and all prediction arrays. No primary confirmation,
future confirmation or official test is scored. One seed/epoch establishes no
noninferiority, convergence, optimal initializer or Atom-count scaling law.

The entrypoint is `python -m cst_research.experiments.higgs_width_amplitude_run`;
`--help` lists explicit immutable consumer/external-source/input paths and GPU
selection. Freeze the consumer commit/manifest and inputs before GPU execution.
