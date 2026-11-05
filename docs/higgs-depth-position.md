# HIGGS depth and CST layer position pilot

English (canonical) | [日本語](higgs-depth-position.ja.md)

This is a separate, initially PLANNED research consumer of unchanged torchcst
`97580582ed28c4e494bf5c2656bbe94a85c544c3`. Its cohort is
`higgs-depth-position-v1`, not the older single-middle-layer HIGGS cohort.
No public runtime or Graph eligibility rule changes. CPU checks and prepared data
are preparation, not a deep-model GPU result.

## Fixed architecture and initial screen

Use **28→N→N→N→N→2**, N1024: dense input/output with biases, three bias-free square
hidden operators, four ReLUs, and all parameters trainable. Compare one dense
control with CST replacing only the early, middle or late square layer; after
reviewing those curves/gradients, run all three square layers as CST. For each
CST configuration use alpha0.0001/0.001/0.01 (divisors10000/1000/100), giving
104/1048/10485 atoms per CST layer. This is13 configurations, each seed17 and
3 full epochs over100k examples, batch4, FP32, TF32off, all arms eager.

The single LR is0.0003 for ordinary and packed CST parameters; positions have
multiplier1. Amplitude decay0.01 and the stock coupled-Polar update remain.
No per-density four-LR grid, automatic width sweep, gain calibration, or Graph
qualification bypass. Adoption of a positive coordinate pilot would require
reviewed evidence and a new explicit source freeze; the one-site private
position optimizer is not reused for the three-site model.

Within `torch.random.fork_rng`, input/output seeds are seed+1/+2. Dense square
layer j uses seed+100+j; CST atoms at that position use seed+1000+j. Therefore
all shared dense weights and every same-position CST initialization agree across
arms regardless of which other positions are replaced. Public CST initialization
is preserved. Activation/gradient collapse is a possible observation, not grounds
for silently rescaling the model.

## New data and exposure boundary

Use these zero-based half-open raw HIGGS TRAIN ranges:

| Role | Raw rows | Count |
|---|---|---|
| Training | [1000000,1100000) |100k|
| New development |[1100000,1150000)|50k|
| Future confirmation, separate payload |[1150000,1200000)|50k|

Raw source SHA256 is
`ea302c18164d4e3d916a1e2e83a9a8d07069fa6ebc7771e4c0540d54e593b698`.
Fit population mean/std in float64 **only on the new100k training features**,
std floor1e-6, then apply `(raw_float32-mean_float32)/std_float32` identically
for every arm. No new-development/future-confirmation fitting.

The fixed pilot file `train-development.npz` contains exactly
`x_train,y_train,x_development,y_development`, SHA256
`58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`.
Its manifest SHA256 is
`530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`.
The separate `future-confirmation.npz` is **not staged or opened by the pilot**.
Official test rows[10500000,11000000) remain reserved.

These ranges avoid the documented prior retained first1M and prior validation
[10400000,10500000). They are not globally untouched raw features: prior
normalization included[0,10400000), including the new rows. The manifest records
raw model/label-use boundaries separately from that historical aggregate feature
exposure. It makes no universal claim about every historical CST label use.
The old100k validation/confirmation payload is not reused for new formal claims.
This coarse adaptive pilot has no formal confirmation claim; a future claim
requires a separately fixed protocol and a completed exposure audit.

## Implementation and gates

`higgs_depth_study.py` owns the new architecture, optimizer groups and training
loop. It reuses only external975 public CST construction and generic
`ordinary_step`, `evaluate`, configuration and tensor helpers. Stock CSTAdamW
owns every CST table; ordinary/dense parameters remain in ordinary AdamW groups.
The dense control uses ordinary AdamW. No single-middle-index assumption or
old cohort label is reused.

Before each case's training, an actual-device gate compares new-network Y/dX and
all parameter gradients to a public CPU FP64 `dense_weight` composition, then
checks8 eager steps and4-step save/reload continuation, finite values, complete
optimizer ownership and clocks. This also covers the newly smaller104-atom case;
prior L4 qualification is not a substitute for new-device/depth admission.
Device model, capability, SM count, memory and Torch/CUDA are recorded. The owner
must use the approved profile; source preparation grants no resource ownership.

Retain initialization, all epochs, selected-best/final raw model/optimizer/gradient
checkpoints, new-development predictions with raw row IDs, complete tensor hashes,
order hashes, train/dev curves, timing and allocated/reserved peaks. Fixed first256
new-training rows supply per-layer activation RMS/max/zero fractions; all parameter
gradients receive the same diagnostics. Evaluation/probes must leave model and
optimizer tensors unchanged. Full gradients and moment clocks are checked each
epoch. Tiny/zero signals and negative cases remain evidence.

Selection is earliest minimum new-development CE. Timings are synchronized eager
complete training steps; evaluation/diagnostic/I/O time is excluded and preparation
is separate. Memory includes resident data/model/optimizer and probe effects,
not total GPU process memory. This is quality-first exploratory work, not a
cross-device speed-adoption claim. No new heldout evaluator exists in this runner.

Run one case per reviewed job, with verified consumer/external manifests and an
output directory outside the immutable consumer source. Inspect the dense and
single-position cases before the all-three cases. Additional seeds, longer
training, wider N or new density/LR search require a subsequent explicit design.
