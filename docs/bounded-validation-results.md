# Bounded CST validation observations, 2026-11-06

**English** | [日本語](bounded-validation-results.ja.md)

Three exploratory studies have completed execution, retrieval and independent raw
checks. The primary L4 three-seed HIGGS noninferiority campaign remains running;
these observations do not complete or amend that campaign. No confirmation-tail
or official test metric was used in these studies.

## Coordinate learning rate: early improvement did not persist consistently

The model is 28→1024→1024→2, with dense trainable input/output layers and one CST
middle layer containing 1048 atoms (nominal alpha .001). Let
`q = position learning rate / ordinary-and-coupled-polar learning rate`.
The packed fields are polar_s, polar_t, output_center and input_center; this
experiment does not give physical amplitude and radius independent Adam rates.
The base LR is .0003. A research-local optimizer applies the actual position LR
to contiguous position and moment state before the unchanged polar retraction;
q=1 is checked against the stock proposal. This is not a public default change.

All runs used B4, FP32, TF32 off and eager execution on the same A100 80GB PCIe
MIG 3g.40gb profile (42 SM, 42,144,366,592 bytes), Python 3.12.9,
Torch 2.11.0+cu126 and Triton 3.6.0. Use the primary HIGGS train/development
partition, not its confirmation tail. The seed17 three-epoch pilot compared
q=.1/1/10. q=10 improved selected development CE by .00274229 and AUC by .00232536
over q=1, meeting the frozen pilot continuation rule. q=.1 performed worse.
Only q=1/10 advanced to the bounded ten-epoch followups at seeds29/43.

| Seed | q10 − stock selected CE | Accuracy difference, percentage points | AUC difference |
|---|---:|---:|---:|
| 29 | +.0006189942 | −.2580 | −.0008525581 |
| 43 | −.0000753999 | −.1920 | +.0004390437 |
| Mean of two | +.0002717972 | −.2250 | −.0002067572 |

Each arm selects the earliest minimum-development-CE checkpoint. These paired
development differences are descriptive, without a noninferiority decision or
confidence claim. The early advantage did not produce a consistent final gain.
Keep stock q=1; no further ratio or seed sweep follows automatically.

CPU scalar arithmetic, actual CUDA moments/clocks, q1 stock byte equality and
eight-update/four-update-resume checks passed. An earlier qualification attempt
failed before learning because its moment oracle used a double-precision beta
coefficient instead of the actual stored FP32 coefficient. The corrected oracle
and failed attempt are both retained; training math and tolerances were unchanged.

## Depth and layer position: compression depends on its placement

Use 28→1024→1024→1024→1024→2: three bias-free square hidden layers, four ReLUs,
trainable dense input/output. Compare Dense, one CST at early/middle/late, and
all three square layers CST, at nominal alpha .0001/.001/.01 (104/1048/10485
atoms per replaced layer). All 13 cases completed at seed17, three epochs,
LR .0003, stock optimizer q1, B4 FP32/TF32 off/eager.

The actual GPU was RTX3090, CC8.6, 82 SM, 25,296,044,032 bytes, driver580.95.05,
Python3.12.9, Torch2.11.0+cu126 and Triton3.6.0. Train rows[1000000,1100000),
development[1100000,1150000), and separate unused future rows[1150000,1200000)
are zero-based half-open ranges. Fit this study's normalization on its training
features only. Historical aggregate feature statistics had included these rows;
the study does not claim globally untouched features. Future labels were not
used for model selection or evaluation.

| Replaced layers | alpha .0001: CE / AUC | alpha .001: CE / AUC | alpha .01: CE / AUC |
|---|---:|---:|---:|
| Early | .5627934 / .7783969 | .5484946 / .7982233 | .5482507 / .7994730 |
| Middle | .5485868 / .7982039 | .5429138 / .8024335 | .5433032 / .8029886 |
| Late | .5450923 / .8013952 | .5437870 / .8026811 | .5427784 / .8035260 |
| All three | .6913939 / .5000000 | .5524291 / .7916114 | .5487499 / .7977808 |

Dense selected CE/AUC was .5483097/.8014886. These are development-selected
single-seed points after only three epochs. They establish neither a minimum
alpha nor a monotone depth law. All-three alpha .0001 remained a constant
classifier; the .001/.01 cases learned and were still improving. With alpha
.0001, a single middle/late replacement could learn while replacing all three
failed to do so within this budget.

Single-CST synchronized training took approximately 102–108 seconds versus
Dense 46.292 seconds for the full three epochs. Allocated peaks were
approximately 87.8–88.6 MB versus Dense 104,513,024 bytes. These are allocator peaks,
not process memory; no CUDA Graph was used. Preparation, evaluation, diagnostics
and checkpoint I/O are outside training time. The first CST case's cold compile
made preparation approximately 130 seconds versus 4.10–6.41 seconds for later
single-CST cases;
this is not a density effect or an L4 speed comparison.

Independent full-network CPU FP64 Y/dX/all-gradient checks and eight-update,
four-update-resume checks passed. A separate amplified-cotangent componentwise
stress diagnostic failed at all three densities; preserve that negative result.
A scale-normalized field-and-packed-column check passed, with zero/sign-flip
negative controls and material-gradient coverage. Its alpha .0001 admission was
post hoc; the same criterion was frozen before the .001/.01 probes. Do not
relabel the original amplified componentwise failure as a pass.

The first replaced layer's initial CST output RMS was about .0005 versus Dense
.2428 on the training probe. This motivates a conditioning hypothesis; it does
not isolate capacity, initialization, optimization or activation-mask effects.
A separately frozen finite gain experiment is reported below. Its per-site bounds
and optimization scales change, but its unrestricted output weight preserves the
global ideal-real function family. This is not an initialization-only causal test.

## Fixed gains: the smallest all-layer case learns, but remains below Dense

On the A100 MIG/runtime above, compare a new Dense control, stock all-three CST,
and fixed-gain all-three CST at alpha .0001, seed17, three epochs and LR .0003.
Use the depth study's exact train/development rows and stock q1 optimizer. The
actual Dense baseline belongs to this A100 cohort; do not substitute the RTX3090
result. This density was chosen after observing the stock depth pilot's stall,
before seeing any gain result, and is explicitly exploratory.

For each replaced square layer, a fixed positive output gain matches the
corresponding Dense layer's initial RMS on the first256 training features.
Calibration is sequential CPU FP64 public `dense_weight` composition, uses no
labels or development rows, and performs no clipping or target search. Gains are
nontrainable FP32 buffers fixed throughout learning. Actual executed gains are
521.2747192382812, 258.95751953125 and 945.0846557617188. The actual saved stock
and gain arms have byte-identical trainable initialization; the stock tables also
match the prior verified depth study.

| Arm | Selected development CE | Accuracy | AUC |
|---|---:|---:|---:|
| Dense | .5476290584 | .7199800014 | .7952768061 |
| Stock all-three CST | .6913939118 | .5295999646 | .5000000000 |
| Fixed-gain all-three CST | .5910087228 | .6868199706 | .7539428246 |

All selected checkpoints are epoch2. Gain versus stock improves CE by .1003851891,
accuracy by15.7220 percentage points and AUC by .2539428246. Gain still trails
Dense by .0433796644 CE,3.3160 accuracy points and .0413339815 AUC. This intervention
breaks the observed constant-classifier stall within this budget, while leaving
a substantial quality gap. It does not isolate a capacity limit, establish a
minimum alpha, or justify adopting gain calibration as a default. No extra gain
epochs or seeds follow automatically.

Independent raw checks passed for all52 archive members, actual sources/data/
runtime, calibration, ordinary/stock/gain initialization, normal Y/dX/all-gradient
and field/column norm gates, eight-update/four-update resume, every epoch's
parameters/gradients/moments/clocks/CST optimizer stores, and selected full-train
100k/development50k predictions. All three amplified componentwise diagnostics,
including Dense, failed and remain labelled failed. Their field/column relative
L2 errors passed the frozen3e-4 norm criterion. No confirmation/test data was used.

Two reader-only failures are preserved: a fresh local Torch2.13/Mac CST constructor
was not byte-identical to the actual Torch2.11 initializer (maximum differences
below9e-9), and a helper assumed50k rows when scoring the100k training predictions.
The final reader binds the actual saved initialization to the prior measured
stock, independently recomputes calibration from those tensors, and uses the same
binary metric equations for either row count. Producer math, tolerances and raw
results were unchanged. Exact cross-runtime initializer bytes are not claimed.

## Positive gains preserve the global ideal function family

For the actual three bias-free square maps, ReLUs and positive fixed gains,
positive homogeneity gives the invertible reparameterization

\[
F_g(\theta)=F_{\rm stock}(T_g\theta),\qquad
W'_{\rm out}=G W_{\rm out},\quad b'_{\rm out}=b_{\rm out},
\qquad G=\prod_{j=1}^{3}g_j\approx127575095.34510244.
\]

All hidden parameters stay fixed under this mapping. Because the output weight
is unrestricted, rescaled per-site amplitude bounds do not expand the global
ideal-real function family. The partial rescue therefore cannot be explained
solely by a different representable function class or unchanged initial zero
columns. Identical saved tensors still begin at different functions; gradients,
Adam epsilon and relative-update trajectories differ. Multiplicative decoupled
weight decay itself commutes with constant rescaling, so its isolated causal role
is not established. Finite FP32 bitwise equivalence is not claimed.

The [full derivation](positive-gain-equivalence.md) binds actual saved initialization
and unchanged69/975 source. A public CPU-FP64 comparator on fixed synthetic linspace
and zero4×28 inputs gave maximum absolute errors8.67e-19 and0 after transforming
the stock output weight. No dataset/labels, task evaluation, GPU operation or new
qualification was involved. Analytic report
`higgs-depth-gain-20261106/stats-positive-gain-equivalence-01/analysis.json` has SHA
`a90fa2323c706095b4975e8b81bc29c745322ae13a56fbb467dd8c8f8c05f5c7`.
Existing seed17/three-epoch limits and failed diagnostics remain unchanged.

## Source and evidence identities

All numerical runtime comes from unchanged torchcst
`97580582ed28c4e494bf5c2656bbe94a85c544c3`, external manifest
`17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`.
Measured research consumers remain on their exact commits:

| Study | Measured consumer | Full manifest | Passed summary audit |
|---|---|---|---|
| Coordinate LR | `83cb77fe84e91ae296c24f2144bd56ca9398f404` | `7ce3d76f3ea3e09084e5c2dcb37ade7581eebffbf0f22c1b74dae08b467511bd` | `b5be7229474a29e182b7978ba5270e665a941f34870d0ac9b27fcecf2ca45532` |
| Depth13 | `f6d08ad76001a69070b0965cefe7614f192ddf3f` | `aec42b259f37bd3922a3763f4c4f5e3af9b9a926263b10128861d09b18295a15` | `95b1bf35095028623900732dfd60eca41cf80d0a896a891dc3b0f2ab75050dd3` |
| Depth gain3 | `69d7f0636ce47db892096d7b7aacc2c2cb731b8b` | `b7b6428e80bbf18fc0cff22af2388e6818c3c9c28ad9221c98c352123666d193` | `4bfb2d41ed402fcce01b05648d52d601cee3fde820300edd6f1a15d8f80cbb7d` |

Coordinate source: [PR7](https://github.com/takumiecd/cst-research/pull/7).
Depth source: [PR8](https://github.com/takumiecd/cst-research/pull/8).
Gain source: [PR9](https://github.com/takumiecd/cst-research/pull/9).
Reproduce with those commits' protocols and pinned inputs, separately from this
documentation commit. Use fresh output/state directories and the required
resource instructions. CPU tests or registry validation alone do not reproduce
the GPU results.

Under the [external evidence convention](reproducibility.md), retain:

- `higgs-coordinate-lr-20261105/stats-actual-a100-followup-seed29-01` and
  `stats-actual-a100-followup-seed43-01`: independent raw audits;
  `stats-verified-two-followups-02`: derived all-curves CSV, epoch-aligned means,
  PNG/PDF overlays and SHA-bound audit. The combined report reuses passed audits.
- `higgs-depth-20261105/stats-verified-all13-01`: all13 curves, case/activation
  summaries, density/position/resource PNG/PDF plots and audit; earlier raw audit
  lineage and all failed stress artifacts remain retained.
- `higgs-depth-gain-20261106/stats-actual-a100-gain-03`: full raw audit, all curves,
  activation/gradient summaries, same-checkpoint train/development and resource
  plots. `presentation-final-01` binds the scope-labelled learning PNG/PDF to that
  audit and the preserved reader-failure lineage. Result archive SHA
  `0ed924e6511005a8a780764392dcd6d6d04f8cc48b25c7e725dfbb78da4cd92f`.

Coordinate result archive SHAs: seed29
`6300c1efd82f4804a2b4de7ea7aea654f55f574d6a8e7bcc1520697e0672ba67`,
seed43 `653471726c8c714d6d873f743d33a82666b9a3a07cad46cf3227d8a45121ac8b`.
Depth all-three archive SHAs in density order .0001/.001/.01:
`c2db2ee4f8565929664b6989fefb4758ff0b13056f366c3362f08b7bdf4531ae`,
`31c8cb10aae104ded6bb4dab56cbf5615b8824cefd67799191c119f23bf5c10b`,
`53b19026996fe6ecd1520743dc8dafc016a3dd907bdb038ca01696cdfbc20d6a`.
These results use Hub job receipts. Source, operational deployment,
raw state, selection and derived-report identities remain distinct.
