# HIGGS first-update normalization diagnostic — bounded results

English (canonical) | [日本語](higgs-depth-normalization-diagnostic-results.ja.md)

**The nine-case, eight-update diagnostic completed and its retained evidence
passed independent CPU admission.** In the first real B4 update, A104 stock had
nonzero but tiny coordinate gradients and no physical coordinate motion; fixed
gains and normalization produced coordinate RMS movements around1e−4. This is
initial-step evidence, not an explanation of a full learning curve or proof that
normalization closes a quality gap. No development/heldout evaluation, HPO, longer
training, minimum-α/scaling result, throughput comparison or default adoption was
performed. Earlier negative/partial-rescue results remain in force.

The executed consumer is **4f777a5f1ef38cca7031cf9c637d0fa6fc0d5e7b**, not this
result-note branch. Its85 files and external975's1188 files were verified. Public
runtime remains97580582ed28c4e494bf5c2656bbe94a85c544c3. See the unchanged
[scientific protocol](higgs-depth-normalization-diagnostic.md),
[earlier bounded results](bounded-validation-results.md), and the
[machine-readable derived summary](higgs-depth-normalization-diagnostic-summary.json).

## Execution and admission scope

One root-owned normal-exit foreground RTX4090 attempt ran all nine fixed cases,
**eight true task updates each** (72 total), B4/FP32/TF32 off/eager, seed17/LR.0003.
The architecture is28→1024→1024→1024→1024→2, three bias-free square middles,
trainable Dense IO, stock q1 CSTAdamW. No-affine RMS/LN share ε=1e−5; stock
Adam ε=1e−8, β/decay semantics and all numeric gates remain unchanged.

The device is NVIDIA GeForce RTX4090, CC8.9,128SM,25262096384 bytes, authorized
Hub profile `gpu4090-tf2-18-0-pytorch2-6-0-cu126-keras3-8-0`. Actual isolated
runtime: Python3.12.9, Torch2.11.0+cu126, CUDA12.6, Triton3.6.0, NumPy2.0.2,
driver570.133.20. Runtime pin SHA is
`1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b`.

All151 retained files (2,643,520,928 raw bytes) were SHA-admitted before semantic
review. All nine normal Y/dX/all-gradient, norm/packed-column/material gates,
8 versus4+resume+4 states, finite/ownership/clocks and task states0/1/2/4/8 passed.
Producer observation/native checks passed at every synthetic step; the independent
retained-raw comparison is **final8**, not an independently retained all-step
history. Task inputs/labels/order and forward/backward summaries were checked.
Native scalar-Adam reconstruction maximum relative error was9.7044e−8.
Synthetic qualification is separate from the72 task updates and resets before
probing. Every case's amplified componentwise failures remain saved, including
Dense controls; the predefined mandatory normal/norm/material rules passed.
This does not qualify arbitrary amplified componentwise cotangents.

## First real task batch, not the256-row calibration probe

The displayed forward pass uses the first **four** examples in the common
seed117 first-epoch order, **before update1**. The same batch supplies CE/backward
and the update below. Fixed gains were calibrated on a separate256 TRAIN-feature
probe without labels; matching that probe's RMS does not force equality on this
B4. There are no confidence intervals or full-data quality scores.

\[
\rho=\frac{\sum_i\max(z_i,0)^2}{\sum_i z_i^2}
=\left(\frac{\operatorname{RMS}(\operatorname{ReLU}(z))}{\operatorname{RMS}(z)}\right)^2.
\]

| Case | Last pre-ReLU RMS | Last post-ReLU RMS | Positive energy retained (%) |
|---|---:|---:|---:|
| dense-stock | 0.0362681 | 0.026206 | 52.2099 |
| dense-rms | 0.0402553 | 0.0290711 | 52.1527 |
| dense-ln | 0.0402539 | 0.0285256 | 50.2174 |
| cst104-stock | 2.56882e-10 | 7.71244e-12 | 0.0901398 |
| cst104-gain | 0.0327717 | 0.000983915 | 0.0901398 |
| cst104-postgain | 0.650299 | 0.0195241 | 0.0901398 |
| cst104-rms | 0.0330892 | 0.00100207 | 0.0917109 |
| cst104-ln | 0.0324452 | 0.00554607 | 2.92193 |
| cst1048-stock | 4.81402e-10 | 2.74249e-10 | 32.4544 |

A104 stock, pre-target gain and post-target gain retain approximately0.09014%
of final pre-ReLU energy. Positive constant gains preserve the ideal sign/energy
partition. Post-target calibration makes this batch's post-ReLU RMS.0195241
versus pre-target.000983915 and Dense.0262060; it need not match Dense exactly
outside the calibration probe. RMS gives0.09171% retention, LN2.92193%, still
below Dense52.2099%. These measure energy distribution, not an activation mean,
a count of permanently dead neurons, or eventual accuracy.

## Actual coordinate update and ε scope

Ranges below cover **six field blocks** per CST case: μ_output/μ_input at three
sites. Gradient and displacement RMS each pool that field's atoms; displacement
is the minimum signed torus movement in physical N1024 coordinate units.
ε-dominant means sqrt(vhat)<ε, reconstructed from actual native proposal m/v,
clock and group settings and independently scalar-checked. Its percentage here
**excludes coordinates whose prepared gradient is zero**; it counts components,
not gradient energy. The summary also retains all-component percentages.

| CST case | Coordinate prepared-gradient RMS | Nonzero gradient (%) | ε-dominant among nonzero gradients (%) | Accepted physical μ-displacement RMS |
|---|---:|---:|---:|---:|
| cst104-stock | 2.83e-14–4.27e-14 | 60.6–73.1 | 100–100 | 0–0 |
| cst104-gain | 3.61e-06–5.46e-06 | 58.7–75 | 79–86.9 | 7.98e-05–0.000104 |
| cst104-postgain | 7.19e-05–0.000109 | 62.5–77.9 | 76.9–81.5 | 9.72e-05–0.00012 |
| cst104-rms | 3.66e-06–5.56e-06 | 86.5–100 | 78.8–88.5 | 9.29e-05–0.000128 |
| cst104-ln | 8.82e-06–0.000125 | 65.4–100 | 0–65.4 | 0.000164–0.000304 |
| cst1048-stock | 7.94e-13–1.27e-12 | 99.6–100 | 100–100 | 0–7.36e-09 |

The first A104-stock update has physically zero movement in all six coordinate
fields, despite one encoded0↔1024 change; those encodings are the same torus
point. Corrections increase the measured gradients/accepted movement, supporting
a narrow scale-to-first-step link. They do not establish the cause of the earlier
three-epoch stall. In particular, A1048-stock is also100% ε-dominant on its
nonzero coordinate gradients initially, yet it learned in the **separate RTX3090
f6 three-epoch cohort**. This is not a controlled cross-device causal proof;
initial ε-dominant percentage alone does **not explain the A104 stall**.
FP32 coordinate resolution, gradient distribution, subsequent moments, changing
gates and training budget are not resolved by this first batch.

## Function maps and preserved reporting limits

For fixed positive gains, bias-free squares/ReLU and unrestricted final Dense,
the previously established ideal-real reparameterization is

\[
F_g(\theta)=F_{stock}(T_g\theta),\qquad
W'_{out}=\Bigl(\prod_{\ell=1}^{3}g_\ell\Bigr)W_{out},\quad b'_{out}=b_{out}.
\]

RMS/LN instead introduce input-dependent normalization; LN additionally centers.
Equal A and identical initial trainable tensors do **not** establish the same
function family for those variants. No gain-style equivalence is proved for
them, and their first-step differences are not a capacity-controlled quality
comparison. No unique atom count follows from matching an activation scale.

Two reporting limitations are preserved rather than changing the measured source:

- The frozen displacement reporter uses a +N/2 wrap that can round tiny in-range
  movements away. A separate stable shortest-torus derivation checked37,632 actual
  deltas against libm IEEE remainder. Seven legacy differences were at most
  3.6374e−14. A104 stock has an actual9.5407e−15 coordinate delta at retained
  step8, so the result is **not literal zero motion through all eight updates**.
  Both legacy and corrected reports, raw values and frozen4f remain unchanged;
  this is a derived-report erratum, not a scientific rerun/tolerance revision.
- CPU reader01 stopped on a non-Tensor `_extra_state` comparison. Reader02
  recursively checked exact metadata and tensors and passed the same raw attempt.
  That failed reader and all amplified componentwise failures remain available;
  no second GPU attempt or inference was substituted.

## Reusable provenance

Consumer manifest:
`aa942f52f88d9a26a950eb51203df948216bbb50fb437c410b9dc8d9b86bbd53`.
Scientific protocol:
`0a267cc7fe634426ada6a515bdddf3459b7f7fa2d34797b84d38f65860cc3d33`.
Data/artifact and exposure boundaries remain those of f6: train-only-fit data SHA
`58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`;
no official or original HIGGS confirmation payload was evaluated.

External locator:
`$CST_RESEARCH_EVIDENCE_ROOT/higgs-depth-normalization-diagnostic-20261106/evidence-index.json`.
That ignored index binds original retained reports, source-bound PNG/PDF/CSVs and
receipts; no raw checkpoint/data/log archive is committed with this note.
Whole result gzip SHA:
`882fa8a3e96b9214793fa4bf68f0c203de8bab83e98b4750d0d6e21e5764959b`.
The figure/CSVs display update1 only; the evidence records all8 updates.
This note aggregates passed JSON/CSV, without a new tensor/data read or GPU run.

- full: `89238d2bece6f2537c68e3053e4da839a138dba7f4f78cd8c0bc92fe54faeba8`
- coordinates: `046c2e03cd9c2166379c38972cf3efb2075b35f00f903a5e40b4bc7073706fb6`
- analysis: `cd6c8311ba354f674e8abedbafc8630d57688057ecd22e584990cd0beccecc93`
- files: `3d857e153906395c76cf516f0afeb54508bf532acfa375b99e0c69f6e916ab3a`
- torus: `f95c32fc0c2b8c0875464ca1f4cd74d2fe990dc4854dde854cf4550a30502950`
