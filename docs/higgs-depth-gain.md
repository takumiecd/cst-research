# HIGGS depth: fixed output-gain intervention

English (canonical) | [日本語](higgs-depth-gain.ja.md)

**COMPLETED.** All three bounded arms ran at measured consumer69 with independent
raw admission. Fixed gains partly rescued the stall but remained below matched
Dense. See [observations, audit identities and limitations](bounded-validation-results.md).
The following protocol retains its adaptive-choice and failed-diagnostic disclosures.
The original depth source `f6d08ad76001a69070b0965cefe7614f192ddf3f`, its
13 cases, and the formal single-middle CST/DST studies remain unchanged.

## Question and adaptive choice

The stock depth screen used Dense, nine single-position replacements and three
all-hidden replacements, one seed and three epochs. Its development curves showed
all-hidden α=0.0001 at AUC 0.5, while α=0.001 and 0.01 learned. This motivates asking
whether fixed forward-scale conditioning helps the low-density all-hidden model.
The first source preparation used α=0.001 without execution; the chosen density
changed to **α=0.0001 before any gain experiment ran**, based on those observed
stock curves. This is an adaptive exploratory choice, not independent confirmation.
The earlier source preparation is retained in ignored evidence.

Exactly three arms run in one process on the same admitted GPU/runtime:

| Arm | Three square hidden layers | Fixed gains |
|---|---|---|
| `dense` | Dense | 1, 1, 1 |
| `stock-all` | CST, α=0.0001 | 1, 1, 1 |
| `gain-all` | Same CST trainable initialization | Features-only calibrated gains |

Architecture is 28→1024→1024→1024→1024→2, four ReLUs, bias-free square layers,
dense input/output, all parameters trainable. Each CST layer has
A=floor(1024²/10000)=104 atoms. Dense and CST use the original independent
per-layer seeds; stock/gain **all named trainable tensors are byte-identical**,
and ordinary input/output tensors also match Dense. Gains are three nontrainable
FP32 values in one vector buffer; unit multiplications are present in all three consumer arms.

All arms use seed17, three full epochs, batch4, FP32, TF32 off, eager execution,
and stock CSTAdamW (AdamW for Dense), LR0.0003, position multiplier1,
ordinary weight decay0.01, atom raw decay0, logical amplitude decay0.01.
There is no gain grid, LR sweep, adaptive extra epoch, automatic followup or extra
seed. This remains a distinct consumer around unchanged public external
`torchcst` source `97580582ed28c4e494bf5c2656bbe94a85c544c3`.

## One-shot calibration

Only the **first256 normalized new-training feature rows**, raw indices
[1000000,1000256), enter calibration. No labels, development, confirmation or test
values are used to compute gains. Independent untrained copies and the public
CPU FP64 `dense_weight()` composition determine the values. Dense target RMS
uses the Dense network's own preceding Dense/ReLU path. Candidate RMS uses the
candidate's already calibrated preceding CST/ReLU path.

For square site j, sequentially:

\[
t_j=\operatorname{RMS}_{X_{256}}(z_j^{\mathrm{dense}}),\qquad
r_j=\operatorname{RMS}_{X_{256}}(C_j(h_{j-1}^{\mathrm{gain}})),\qquad
g_j=\operatorname{FP32}(t_j/r_j),\qquad
h_j^{\mathrm{gain}}=\operatorname{ReLU}(g_j C_j(h_{j-1}^{\mathrm{gain}})).
\]

Every target and observed RMS must be finite and strictly positive, and both the
unrounded ratio and stored FP32 gain must lie within inclusive [1e-6,1e6].
A zero target, zero observed signal, nonfinite value or infeasible bound yields
**inconclusive, no training**. There is no clipping, alternate target or tuning.
The raw feature matrix, its hash/row binding, initial models, computed gains,
sequential RMS transcript and unchanged trainable hashes are saved. Buffers stay
fixed through optimizer steps and checkpoint restoration.

This is a **forward-scale/conditioning intervention**, not an initialization-only
causal test. Per-site effective amplitude bounds change from |a|≤1 to |g_j a|≤g_j.
For this exact bias-free square/ReLU chain with fixed positive gains and an
unrestricted final Dense weight, however, the product can be absorbed into that
output weight. Thus the **global ideal-real function family stays the same**:

\[
F_g(\theta)=F_{\rm stock}(T_g\theta),\qquad
G=\prod_{j=1}^{3}g_j,\qquad W'_{\rm out}=G W_{\rm out},
\quad b'_{\rm out}=b_{\rm out}.
\]

All other trainable parameters stay unchanged, and the inverse divides the output
weight byG. This is a function-family identity, not an optimizer-trajectory or
FP32 byte-equivalence claim. Identical saved tensors generally start at different
functions; gradient scales, Adam epsilon and relative updates differ. The
multiplicative part of decoupled weight decay itself commutes with constant
rescaling; weight decay alone is not established as the cause. Positive gains
preserve a layer's ReLU signs for the same pre-gain input, not a shared trained
activation pattern. See the [analytic derivation and source-bound synthetic-only
CPU comparator](positive-gain-equivalence.md). The original product-gain
Y/all-gradient algebra gate and all training/protocol/source bytes remain intact.

## Data and admission

Reuse only the frozen new-depth train/development payload
SHA256 `58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`,
manifest `530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`.
Training is raw [1000000,1100000), development [1100000,1150000).
Normalization fits only training100k in FP64 and is applied in FP32. Historical
aggregate feature statistics previously included these rows; this is disclosed,
not a universal claim of unseen features or labels. The future depth confirmation
[1150000,1200000) and official test [10500000,11000000) remain unopened and are
absent from the pilot payload. See the [depth protocol](higgs-depth-position.md).

Before **any** arm trains, all three fresh actual-device qualifications must pass:

- Full network Y/dX/all seven parameter-gradient fields against a separate public
  CPU FP64 dense-weight composition, using exact widened FP32 upstream vectors.
- Mandatory original normal-scale elementwise tolerance 3e-5+3e-4|reference|.
- At normal and amplified 1e12 upstream scale, relative L2 and max-error/reference-max
  ≤3e-4 for every gradient field and every packed atom column. Normalization has
  no epsilon; an exactly zero reference requires exactly zero actual values.
  Zero/signflip substitutions are explicitly checked, including tiny columns.
- Amplified dX and each CST field require materially nonzero reference and actual
  signal. Forward outputs must be unchanged by upstream scaling. Raw tensors are
  saved before gates; amplified componentwise results are **diagnostic only**.
- Eight stock optimizer updates versus four+restore+four must match all numerical
  tensor bytes, moments, buffers and clocks, with seven independent owned states,
  finite values and fixed gains. AtomState layout-generation metadata may advance
  on load; numerical tensor equality remains mandatory.

The earlier stock all-hidden amplified componentwise stress **failed**. It stays
failed. The later field/column norm admission was post hoc for D10000 and frozen
before probes for the other two densities. This preparation uses the explicitly
scoped norm/coverage rule; it makes no arbitrary-cotangent per-component claim.
Fresh failures stop this pilot rather than silently changing tolerances.

## Outputs and reproduction

The entrypoint is `cst_research.experiments.higgs_depth_gain`. Run only after
source, inputs, operational envelope and exact GPU/runtime are frozen and reviewed:

```bash
PYTHONPATH="$CONSUMER/src:$TORCHCST_SOURCE/src:$TORCHCST_SOURCE" \
python -B -m cst_research.experiments.higgs_depth_gain \
  --consumer-commit "$CONSUMER_COMMIT" --consumer-manifest "$CONSUMER_MANIFEST" \
  --external-source "$TORCHCST_SOURCE" --external-manifest "$EXTERNAL_MANIFEST" \
  --data "$DATA" --data-manifest "$DATA_MANIFEST" --output "$FRESH_OUTPUT"
```

This consumer accepts only the three declared arms/budget, serially on the same
GPU. It does not provision resources. Follow the repository/resource rules.
Retain operational receipts, exact source/input manifests and result archive SHA.
Store all raw artifacts outside Git under the external evidence locator.

Every epoch saves raw model/optimizer/gradients, clocks, order hash, activation
and gradient diagnostics, synchronized complete-step training seconds and
allocated/reserved allocator peaks. Gain-aware activation traces include each
fixed multiplier at the same location as the full model forward; CPU checks
require final trace output and reported statistics to match the full forward.
Separate synchronized seconds are saved for full-train evaluation, development
evaluation and diagnostics (including initial diagnostics/development evaluation).
Peaks include resident data/evaluation and
are not process memory. Preparation/calibration/all-arm qualification is recorded
separately. Evaluation/diagnostics/checkpoint I/O are outside training seconds.
Online training metrics describe changing weights; a separate full-training100k
and development50k evaluation uses the **same post-epoch checkpoint**. Earliest
minimum development CE selects each arm; save both selected train and development
predictions, final checkpoints and all curves. There is no selection by plotted
curves, no minimum-alpha inference, formal NI, speed adoption or cross-device
comparison. A learning improvement would support investigating this combined
intervention; a negative three-epoch screen cannot exclude a longer-run effect.
