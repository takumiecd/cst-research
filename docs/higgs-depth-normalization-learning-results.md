# HIGGS depth normalization learning: bounded results

English (canonical) | [日本語](higgs-depth-normalization-learning-results.ja.md)

**The bounded nine-case study is complete through an explicit recovery lineage.**
In this pilot with one seed and three epochs, calibrated RMS and LayerNorm (LN)
recovered learning in three hidden CST layers with 104 atoms each, which stalled
under the stock parameterization. They did **not** reach Dense quality: selected
development ROC-AUC was 0.772532 with RMS and 0.775996 with LN, versus 0.802750 for
Dense stock. Unnormalized A1048 stock reached 0.792666. These observations do not
establish that LN is necessary at larger atom budgets or justify adopting a
default normalization recipe. No results from a later linear-atom study are
included here.

## Fixed scope and selection

The architecture is 28→1024→1024→1024→1024→2, with three bias-free square hidden
layers, Dense input/output layers and ReLU between layers. All model parameters
remain trainable and are owned by their registered optimizers. Each case used
seed 17, three complete epochs over 100,000 training rows, batch size 4 and
75,000 actual Adam updates. The learning rate was 0.0003, with the stock q1
optimizer, FP32 arithmetic, TF32 disabled and eager execution. Development
contains 50,000 rows. This adaptive exploratory followup was chosen after the
eight-update diagnostic and earlier development observations. Development had
already been inspected; it is not a fresh heldout test. There was no
confirmation/test scoring, learning-rate, epsilon or density search, additional
seed, or formal noninferiority analysis.

The source is the UCI HIGGS compressed CSV, SHA256
`ea302c18164d4e3d916a1e2e83a9a8d07069fa6ebc7771e4c0540d54e593b698`.
Zero-based half-open row ranges are TRAIN [1,000,000, 1,100,000) and DEV
[1,100,000, 1,150,000). The 14,285,659-byte training artifact contains only those
cohorts. Reserved future confirmation [1,150,000, 1,200,000) is excluded from that
artifact and was not scored. Input standardization was fitted only on the new
TRAIN rows; this is separate from the within-network RMS/LN interventions.
Historical aggregate feature statistics over [0, 10,400,000) had already included
these rows. We therefore make no claim that they were unseen throughout all prior
research, or about all historical label use. The exact data manifest is identified
below.

Nominal alpha 0.0001 gives 104 atoms per hidden CST layer, about 0.102 N; nominal
alpha 0.001 gives 1,048 atoms, about 1.023 N. RMS/LN have no affine parameters and
use epsilon 1e-5. Positive frozen output gains were calibrated sequentially using
only the first 256 training feature rows, without labels, to match initial Dense
pre-ReLU RMS. The post-gain arm instead targets post-ReLU RMS. These are the
inherited fixed calibration values; there was no gain search. The earliest
minimum full-development CE among epochs 1–3 selects the checkpoint. Full TRAIN
and DEV evaluations use that same checkpoint; TRAIN evaluation never selects it.

\[
e^*=\arg\min_{e\in\{1,2,3\}}(\mathrm{CE}_{\rm dev}(\theta_e),e),\qquad
\Delta_{\rm CE}=\mathrm{CE}_{\rm dev}(\theta_{e^*})-\mathrm{CE}_{\rm train}(\theta_{e^*}).
\]

## Selected-checkpoint quality

The table records producer-reported scalars from FP32 logits. They were checked
independently against retained prediction arrays with unchanged tolerances:
3e-6 for CE, 2e-7 for accuracy and 1e-12 for AUC. AUC ranks the FP32 logit
differences; an FP64 sigmoid transformation does not replace that convention.
Accuracy is a percentage. The CE gap compares the same checkpoint on both cohorts.

| Case | Epoch | TRAIN CE | DEV CE | DEV accuracy (%) | DEV AUC | DEV − TRAIN CE |
|---|---:|---:|---:|---:|---:|---:|
| Dense stock | 3 | 0.529078 | 0.544667 | 72.418 | 0.802750 | +0.015589 |
| Dense RMS | 2 | 0.536822 | 0.546350 | 71.786 | 0.796577 | +0.009528 |
| Dense LN | 2 | 0.539684 | 0.547439 | 71.600 | 0.795264 | +0.007755 |
| CST A104 stock | 2 | 0.691587 | 0.691394 | 52.960 | 0.500000 | -0.000193 |
| CST A104 pre-gain | 2 | 0.591317 | 0.590339 | 68.812 | 0.754402 | -0.000978 |
| CST A104 post-gain | 3 | 0.586959 | 0.587594 | 69.638 | 0.764557 | +0.000635 |
| CST A104 RMS | 2 | 0.566468 | 0.567374 | 69.986 | 0.772532 | +0.000906 |
| CST A104 LN | 2 | 0.561341 | 0.565227 | 70.402 | 0.775996 | +0.003886 |
| CST A1048 stock | 3 | 0.547023 | 0.551018 | 71.518 | 0.792666 | +0.003995 |

CST A104 LN/RMS reduced both TRAIN and DEV CE relative to A104 stock. Their
remaining TRAIN losses, 0.561341 and 0.566468, are still above Dense stock's
0.529078. The respective gaps of 0.003886 and 0.000906 do not explain the remaining
shortfall solely through a large train–development gap. Three epochs establish
neither optimization convergence nor long-run generalization. Dense RMS/LN also
scored below Dense stock in this pilot.

For the fixed-gain-only arms, positive ReLU homogeneity allows the product of the
gains to be absorbed into the trainable final Dense weight. The ideal real-valued
global function family is unchanged, while parameterization, gradient scales,
Adam epsilon and the regularization mapping differ. Input-dependent RMS/LN change
the function map. This is therefore not a causal proof about initialization or
conditioning alone, or a claim about the minimum atom count.

## Recorded resources

The approved RTX 4090 profile has compute capability 8.9, 128 SMs and
25,262,096,384 device-memory bytes. The runtime was Python 3.12.9,
PyTorch 2.11.0+cu126, CUDA 12.6, Triton 3.6.0, NumPy 2.0.2 and driver 570.133.20.
The two attempts retain separate source and timing provenance.

Training seconds sum the three synchronized complete-step eager loops. They
exclude preparation, evaluation, diagnostics, checkpoint I/O and hashing.
Allocated and reserved memory below are the maximum recorded training peaks
across the three epochs, captured before post-epoch evaluation. They include
resident TRAIN/DEV arrays, model, optimizer and native buffers. Reserved cache
may persist from an earlier development evaluation. These are not process-memory
or CPU-oracle peaks, do not cover later selected-evaluation peaks, and imply no
optimized Graph or sparse-throughput result.

| Case | Trainable scalars | Training, 3 epochs (s) | Allocated MiB | Reserved MiB | Case preparation (s) | 3 DEV evaluations (s) |
|---|---:|---:|---:|---:|---:|---:|
| Dense stock | 3,177,474 | 31.767 | 83.934 | 124.000 | 0.056 | 2.439 |
| Dense RMS | 3,177,474 | 36.630 | 83.689 | 124.000 | 0.060 | 3.019 |
| Dense LN | 3,177,474 | 37.551 | 83.689 | 124.000 | 0.055 | 3.092 |
| CST A104 stock | 32,994 | 96.255 | 37.557 | 64.000 | 0.063 | 12.232 |
| CST A104 pre-gain | 32,994 | 96.552 | 38.646 | 70.000 | 0.064 | 12.397 |
| CST A104 post-gain | 32,994 | 98.870 | 37.557 | 64.000 | 0.063 | 12.840 |
| CST A104 RMS | 32,994 | 104.610 | 37.589 | 64.000 | 0.068 | 13.894 |
| CST A104 LN | 32,994 | 103.421 | 37.590 | 64.000 | 0.064 | 13.960 |
| CST A1048 stock | 44,322 | 99.533 | 37.814 | 64.000 | 0.068 | 12.890 |

Trainable counts follow the equations below and the admitted configuration shapes.
The three fixed gains are buffers; RMS/LN have no trainable affine parameters.

\[
K_{\rm Dense}=3N^2+31N+2,\qquad K_{\rm CST}=12A+31N+2.
\]

Case preparation includes initial diagnostics and `state0` admission/serialization.
The separately reported initial diagnostic time is already included and must not
be added again. Detailed components are in [resources.csv](assets/higgs-depth-normalization-learning/resources.csv).
Missing parent restore timings remain unknown, not zero. Recovery shared
preparation was 2.989716 seconds, including source/runtime guards, inherited and
parent admission, and data/device preparation. That inclusive total must not be
added to its components. Driver archive/cache bootstrap appears only within outer
walltime; the first selected TRAIN evaluation may include lazy native compilation
or placement initialization. These records are not a complete adoption-time
comparison or a hardware-only throughput claim.

## Recovery lineage

The source ea0 parent trained four cases for 300,000 main updates and passed all
nine prefix reproductions, which added 72 separate admission updates. The three
Dense selected evaluations completed. CST stock selected epoch 2, but the
consumer's literal check of the restored layout generation stopped its selected
evaluation. The parent remains exit 1 and incomplete; all 75 raw members remain
unchanged.

The separate source 32428 adapter requires the public source 975 transition
`loaded=max(before_load+1,serialized)` only for the three actual CST owners, with
all other tensors and metadata exact. Evaluation then uses exact post-load state
mutation and reload checks. The adapter first completed stock's selected
evaluation, then trained only the remaining five cases from fresh `state0` for
375,000 updates. Combined main updates total 675,000, with zero new prefix updates.
The original 300,000 updates were not repeated or relabelled as a completed run.

The combined audit passed on its first actual attempt. Earlier parent-reader
failures concerning JSON tuple/list encoding and optimizer-store representation,
and the root reviewer's FP64-score AUC mistake, remain preserved as reader-only
failures. Training, operators, arrays and numerical tolerances were unchanged.
Qualification remains inherited from source 4f; no new independent native FP64
oracle was claimed. The tables and figures were derived solely from admitted
metadata, without rereading tensors, predictions, features or datasets, or running
another GPU or inference pass.

[All nine curves](assets/higgs-depth-normalization-learning/all-nine-learning-curves.png) · [Matched normalization controls](assets/higgs-depth-normalization-learning/matched-normalization-learning-curves.png) · [Gain/density curves](assets/higgs-depth-normalization-learning/gain-density-learning-curves.png) · [PDF/CSV index](assets/higgs-depth-normalization-learning/README.md)

## Source and evidence identities

| Item | SHA / revision |
|---|---|
| External public operators | `97580582ed28c4e494bf5c2656bbe94a85c544c3` |
| Inherited model/calibration/qualification | `4f777a5f1ef38cca7031cf9c637d0fa6fc0d5e7b` |
| Parent training | `ea0d5701c45b188d619853ebd303767f224e36b8` |
| Explicit recovery | `32428aa8112ea6f0e206f0e1c12cda0206d920a0` |
| Recovery source manifest | `a2995731b62c601cadf7e7baafa86390bf54991bac7bd9f935e1dbe5e73008db` |
| Unchanged training recipe protocol | `c2e27deeed740cc299471a2fc1b7f13c03864a678a293d45f05986353911fd4d` |
| Recovery protocol | `80afd04557b70b13b82df9efb2104864562e9d5f638b9bdb8bae6b829e7be629` |
| Train/development artifact | `58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3` |
| Data manifest | `530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526` |
| Failed parent independent CPU admission | `7d69415094ec9bdf31e5ecd66654d3aac2d792e46d92836dd016c2abe780b6b7` |
| Combined independent CPU audit | `eefdef7e8b7598232c137a6f8589c74f1a511051d13bcbd8fe0b5a9a01dc8e84` |
| Root independent science review | `ac19830c54f67610319a02fc334c74322a5ddf0077239cf11aefd476b8d2daa6` |
| Normal recovery outer receipt | `32f30113088d08e030522016062a8d5e3176a467a8d790715a8a024a1e2d0ef9` |
| Root all89-file release | `0bce347e75b7ed9ab666c97a1233028723ed918abd09aa8bd736fe937836741e` |
| Combined producer summary | `b1e4f20f6f83b37efa7a4de4fdda752dda04c57258e0ea28518ffe0fa21c0f33` |
| Derived metadata report | `b23a58b490c0d852fc18cd96928b27d5443ef667588e95f14ad9cb5089987b92` |

| Additional provenance | SHA256 |
|---|---|
| Parent source manifest | `68bf4e7eb80ed9c48ce4c27fc26bd549f55a309276ddd519d63d00dcf2860fb0` |
| Parent failed outer receipt | `18403283955ab2a4d6f5dbdfe9f2094cec54efe9aadbeb3bff15e5285275bcab` |
| Parent 75-file admission | `dbe3081a726d6e175f880d483b298ce86319eb28cf4bdbca98090769553ea4fb` |
| External source 975 manifest | `17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931` |
| Inherited source 4f manifest | `aa942f52f88d9a26a950eb51203df948216bbb50fb437c410b9dc8d9b86bbd53` |
| Inherited 151-file CPU admission | `89238d2bece6f2537c68e3053e4da839a138dba7f4f78cd8c0bc92fe54faeba8` |
| Inherited 151-file outer receipt | `ae0c2bbb7f000f5465c41b24e66c08223ffdd722a45382d70251db71f905c6ab` |
| Combined CPU reader | `83158c46134bdadc4d93b7136f2065af90e5effdb5cb545d0bdb17b832896c6f` |
| Frozen parent reader 03 | `e2d119153dc816c34cb6876f226a368a0337fc847418520092be29f6942caa0a` |
| Root derived 27-row review | `9c81c79b42e7088c887562d87d93dfd16397b2daec76854835a41436ddae56d3` |

## Reproduction and external evidence

Use the exact measured source revisions above, rather than this documentation
checkout. The commands below record the actual frozen entrypoints. They do not
automatically rerun anything, authorize additional GPU use or instruct the reader
to repeat the known failed parent. The original ten input files and recovery's
eleventh `parent-admission.json` must match their pinned source, data, runtime and
proof hashes. See the [parent recipe](higgs-depth-normalization-learning.md) and
[explicit correction](higgs-normalization-learning-recovery.md).

```bash
# Historical failed-parent entrypoint at ea0; not an instruction to retry it.
python -B src/cst_research/experiments/higgs_depth_normalization_learning_driver.py \
  --consumer-commit ea0d5701c45b188d619853ebd303767f224e36b8 \
  --prior-root READONLY_4F_ATTEMPT_ROOT --cache-root OWNED_PARENT_CACHE \
  --output FRESH_PARENT_OUTPUT --timeout 7200

# Explicit recovery entrypoint at 32428, requiring the exact admitted failed parent.
python -B src/cst_research/experiments/higgs_normalization_learning_recovery_driver.py \
  --consumer-commit 32428aa8112ea6f0e206f0e1c12cda0206d920a0 \
  --parent-admission-sha256 7d69415094ec9bdf31e5ecd66654d3aac2d792e46d92836dd016c2abe780b6b7 \
  --prior-root READONLY_4F_ATTEMPT_ROOT --parent-root READONLY_FAILED_EA0_ROOT \
  --cache-root OWNED_CORRECTION_CACHE --output FRESH_RECOVERY_OUTPUT --timeout 7200
```

Under the [external evidence convention](reproducibility.md), resolve
`higgs-normalization-learning-recovery-20261106/evidence-index.json` below
`CST_RESEARCH_EVIDENCE_ROOT`. The external index maps these roles to retained
immutable artifacts: the inherited 151-file source 4f evidence, the failed
75-file parent, the normal 89-file recovery, both outer receipts, independent
parent/combined audits, root science/derived reviews, frozen CPU readers and
earlier failed-reader lineage, and the original derived result pack. It may
contain local storage paths and is not public data. Documentation integration
neither moves nor revalidates those raw artifacts, and does not publish them.

All six public figures preserve the original derived bytes. The three public
CSVs normalize line endings only; original and public hashes are in
[provenance.json](assets/higgs-depth-normalization-learning/provenance.json).
No new result or normalization requirement for later larger-atom studies follows.
