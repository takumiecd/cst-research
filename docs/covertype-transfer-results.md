# Covertype bounded transfer result

English (canonical) | [日本語](covertype-transfer-results.ja.md)

Both tested CST densities underperform Dense on the separately held-out Covertype
rows in this finite regime. Across all three paired training seeds, accuracy is
lower for alpha=0.001 and 0.01. This is a negative task-transfer result at N=1024,
not a universal statement about CST or sufficient alpha.

The architecture is 54→1024→1024→7 with Dense input/output, one bias-free square
middle layer and two ReLUs. All layers train. Alpha is the atom budget A/N²
(integer floors A=1048/10485), not the dense-weight nonzero fraction. All models
use eager FP32, TF32 off, B=4 and 10 complete epochs (250,000 minibatch/Adam
updates). Runtime: one RTX 3090, CC 8.6/82 SM/25,296,044,032 bytes;
Torch 2.11.0+cu126/CUDA 12.6, Python 3.12.9, driver 580.95.05,
Triton 3.6.0, NumPy 2.0.2 and NVCC 12.6.85.
See the [fixed protocol](covertype-transfer.md) and
[registered execution identities](../src/cst_research/studies/covertype-transfer-20261106.json).

The split, fixed before execution, uses 100k train/50k development/50k heldout;
normalization fits the ten continuous columns on training only. Each method had
seed 17 LR screening at 1e-4/3e-4/1e-3 for 3 epochs, at most one prescribed edge LR (3e-5 or 0.003),
then its true top2 LRs restarted for 10 epochs. Dense selected 0.0003; both CST
groups selected 0.001. Checkpoints use earliest minimum development CE. New paired
seeds 29/43/71 then trained the frozen 10 epoch configurations. All nine selected
checkpoints were globally sealed and independently audited before the one heldout
assessment. Actual total: 26 training cells/183 epochs, with all curves retained.

## Heldout results

Means average the three individually scored seeds on the same 50k heldout examples.
These descriptive values are not formal
noninferiority/significance tests or geographic-generalization claims.

| Method | Mean CE | Mean accuracy % | Mean Macro-F1 | Mean Macro OvR AUC | Paired mean accuracy delta vs Dense (pp) | Paired mean AUC delta |
|---|---|---|---|---|---|---|
| Dense | 0.303929 | 87.9207 | 0.804058 | 0.986685 | 0 | 0 |
| CST alpha=0.001 | 0.399907 | 83.3613 | 0.688082 | 0.976923 | -4.5593 | -0.009762 |
| CST alpha=0.01 | 0.371187 | 84.6973 | 0.734105 | 0.980509 | -3.2233 | -0.006176 |

All nine selected checkpoints (no heldout checkpoint reselection):

| Method | Seed | Selected epoch | Heldout CE | Accuracy % | Macro-F1 | Macro OvR AUC | Train s (10 epochs) |
|---|---|---|---|---|---|---|---|
| Dense | 29 | 9 | 0.305178 | 87.8280 | 0.812309 | 0.986473 | 126.775 |
| Dense | 43 | 9 | 0.304472 | 87.8000 | 0.801412 | 0.986803 | 132.435 |
| Dense | 71 | 10 | 0.302136 | 88.1340 | 0.798453 | 0.986779 | 124.913 |
| CST alpha=0.001 | 29 | 9 | 0.394169 | 83.5260 | 0.728886 | 0.977612 | 305.716 |
| CST alpha=0.001 | 43 | 7 | 0.406623 | 82.9900 | 0.670625 | 0.975728 | 300.621 |
| CST alpha=0.001 | 71 | 10 | 0.398929 | 83.5680 | 0.664735 | 0.977428 | 298.131 |
| CST alpha=0.01 | 29 | 9 | 0.368974 | 84.8140 | 0.762324 | 0.980720 | 313.569 |
| CST alpha=0.01 | 43 | 9 | 0.375746 | 84.5100 | 0.745164 | 0.980640 | 313.507 |
| CST alpha=0.01 | 71 | 10 | 0.368841 | 84.7680 | 0.694826 | 0.980166 | 318.843 |

Mean paired CE/F1 differences are +0.095978/−0.115976 for alpha=0.001 and
+0.067259/−0.069954 for alpha=0.01.

All paired accuracy differences are negative: alpha=0.001 ranges from−4.8100 to
−4.3020 percentage points, alpha=0.01 from−3.3660 to−3.0140. Full precision,
per-seed CE/F1/AUC contrasts and mean/SD/ranges are retained in the independent
audit; [all-nine-quality-resources.csv](covertype-all-nine-quality-resources.csv) contains every resource/quality row.

## Recorded resources

The preceding table reports synchronized **training seconds for all 10 epochs**, even
when the selected checkpoint is earlier. These separate exclusions are means of
the three corresponding seed jobs, in seconds; they are not included in that
training clock:

| Method | Startup s | Qualification s | Preparation s | 10 dev evals s | Selected train eval s | Selected dev eval s | Heldout eval s |
|---|---|---|---|---|---|---|---|
| Dense | 1.841 | 0.518 | 0.080 | 7.523 | 1.500 | 0.752 | 0.789 |
| CST alpha=0.001 | 1.844 | 0.980 | 0.086 | 31.384 | 6.130 | 3.060 | 3.095 |
| CST alpha=0.01 | 1.847 | 2.591 | 0.092 | 39.101 | 7.715 | 3.852 | 3.826 |

Training covers complete minibatch forward/backward/optimizer work and excludes
startup, qualification, preparation, evaluation and raw checkpoint/file I/O.
Evaluation timings include forward/copy/metric work. The same physical RTX 3090
cohort/runtime was used, but methods were separate serial phase jobs, so timing
comparisons are exploratory; HPO timing is not a primary speed result.

| Method | Trainable scalars | Atoms | Allocated MiB mean [min,max] | Reserved MiB mean [min,max] |
|---|---|---|---|---|
| Dense | 1,112,071 | — | 67.05 [67.05, 67.05] | 88.00 [88.00, 88.00] |
| CST alpha=0.001 | 67,687 | 1048 | 52.12 [52.12, 52.12] | 60.00 [60.00, 60.00] |
| CST alpha=0.01 | 105,435 | 10485 | 53.12 [53.12, 53.12] | 62.00 [62.00, 62.00] |

Trainable counts sum the actual saved initialization parameter shapes, including
four packed scalars per atom and all ordinary input/output weights/biases;
buffers/optimizer state are not counted as trainable parameters. Memory is the
maximum recorded per-epoch PyTorch allocated/reserved peak, sampled after
synchronized training. It includes resident train/development arrays, model and
optimizer; reserved storage may reflect previous evaluation allocator caching.
It is not process memory or a separately measured heldout-evaluation peak.

## Evidence and limits

- Consumer: `bfa58dce18da85f9d2cb3a791db6d6a5126805d7`; public torchcst: `97580582ed28c4e494bf5c2656bbe94a85c544c3`.
- Training-data SHA256: `010f150bab3a700fbcde1106daf21e83cf28fe9024732d29c25756144a07e416`; heldout: `7220f0d7282c83732c203325f523da180e4701d8a001647ea0eff4cdc0dbee4f`.
- All-nine freeze: `029a5e9dd7e8f71357ef49f9e31a22b26a945a94fa6c9cd8f301640dd8468030`.
- Strict training/seal audit: `ebfb2a99a973aef6fc68dde0b731d42e21b9d995b5711489126f67629be5870d`.
- Once-only assessment archive: `6949031e3d7a4a88ba15ff7af7108982d39a5637d89d4a06ea1079c247fd6760`.
- Independent prediction/metric audit: `6534b81fb590e23b49a14c5e810b24ac467d682c7221445ce33d6529d10cd1eb`.

All 9 prediction arrays, paired labels/row IDs and CE/accuracy/macro-F1/softmax-
probability macro-OvR-AUC were independently recomputed. The reader performed no
GPU forward and opened no heldout feature payload or HIGGS tail/test. The one
assessment exited with code 0, without retry or inference repetition. Failed reader
attempt 01 is preserved: it stopped before predictions because its command guard
assumed direct module execution; the verified frozen driver `--mode assess`
dispatch was then admitted with unchanged sources/models/numerical tolerances.
Its correction lineage is retained in the presentation provenance.

The result is limited to the fixed task split, architecture, two densities,
finite LR search, 10 epoch budget and three seeds. It does not identify a general
minimum alpha, prove optimization convergence, establish a speed benefit or
authorize further experiments. The HIGGS primary/comparison studies remain
separate. `paired-seeds.png` retains all nine development learning curves plus
epoch-aligned means; `heldout-summary.png` shows selected models only, with seed
SD explicitly distinguished from a confidence interval.

Under the [external evidence convention](reproducibility.md), retain
`covertype-transfer-20261106/stats-verified-through-phase17-01`,
`stats-heldout-independent-02` (including `presentation-01` figures),
`stats-covertype-result-note-03` and `assessment-after-all9-01` with their
raw receipt/archive and failed-reader lineage. The published nine-row CSV is a
derived summary; raw checkpoints and data remain external. Consumer manifest:
`8a3c45f7677001c66cc666009d3dfa4a9027d0e1a7571eb571032be75df59d48`;
external975 manifest: `17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`.
Assessment envelope: `7e4210589a17f8d3a2cdf96d679cd8731baab2710df4caebdae5df251f75f040`.
