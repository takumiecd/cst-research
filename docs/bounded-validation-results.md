# Bounded CST validation observations, 2026-11-06

**English** | [日本語](bounded-validation-results.ja.md)

Two exploratory studies have completed execution, retrieval and independent raw
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
A separately frozen finite gain experiment is running and has no result claimed
in this note. Its output gains also change effective weight bounds and gradient/
regularization scaling, so it cannot be a pure initialization-only causal test.

## Source and evidence identities

All numerical runtime comes from unchanged torchcst
`97580582ed28c4e494bf5c2656bbe94a85c544c3`, external manifest
`17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`.
Measured research consumers remain on their exact commits:

| Study | Measured consumer | Full manifest | Passed summary audit |
|---|---|---|---|
| Coordinate LR | `83cb77fe84e91ae296c24f2144bd56ca9398f404` | `7ce3d76f3ea3e09084e5c2dcb37ade7581eebffbf0f22c1b74dae08b467511bd` | `b5be7229474a29e182b7978ba5270e665a941f34870d0ac9b27fcecf2ca45532` |
| Depth13 | `f6d08ad76001a69070b0965cefe7614f192ddf3f` | `aec42b259f37bd3922a3763f4c4f5e3af9b9a926263b10128861d09b18295a15` | `95b1bf35095028623900732dfd60eca41cf80d0a896a891dc3b0f2ab75050dd3` |

Coordinate source: [PR7](https://github.com/takumiecd/cst-research/pull/7).
Depth source: [PR8](https://github.com/takumiecd/cst-research/pull/8).
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

Coordinate result archive SHAs: seed29
`6300c1efd82f4804a2b4de7ea7aea654f55f574d6a8e7bcc1520697e0672ba67`,
seed43 `653471726c8c714d6d873f743d33a82666b9a3a07cad46cf3227d8a45121ac8b`.
Depth all-three archive SHAs in density order .0001/.001/.01:
`c2db2ee4f8565929664b6989fefb4758ff0b13056f366c3362f08b7bdf4531ae`,
`31c8cb10aae104ded6bb4dab56cbf5615b8824cefd67799191c119f23bf5c10b`,
`53b19026996fe6ecd1520743dc8dafc016a3dd907bdb038ca01696cdfbc20d6a`.
These results use Hub job receipts. Source, operational deployment,
raw state, selection and derived-report identities remain distinct.
