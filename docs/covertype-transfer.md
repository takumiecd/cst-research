# Covertype transfer study (prepared; GPU not executed)

**English** | [日本語](covertype-transfer.ja.md)

This bounded exploratory study tests whether low-density CST behavior transfers
to another task. It does not replace the frozen HIGGS studies, establish formal
noninferiority or a universal optimal alpha, reproduce an official benchmark,
or test geographic generalization. GPU execution waits until the current deep
pilot finishes and the owner reviews resources and source/data pins.

## Data fixed before results

[UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype) has 581,012 rows,
54 features (10 continuous, 44 binary) and 7 classes. Credit: Blackard, J. (1998),
[Covertype, DOI 10.24432/C50K5N](https://doi.org/10.24432/C50K5N), CC BY 4.0.
The official zip SHA256 is
`89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d`;
its covtype.data.gz member SHA is
`614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771`.

Permute all rows using NumPy PCG64 seed 20261106. The first 100,000 rows train,
the next 50,000 select development checkpoints, and the next 50,000 form the
separate future heldout. The remaining rows are unused. This is a random row
split; nearby geographic observations may cross partitions. No claim of globally
unseen labels across all historical research is made.

Parse raw integers as FP32. Fit means and population standard deviations for only
the first 10 features on the selected training rows, in float64, flooring std at
1e-6. Apply that transform in float64 and cast to FP32. Preserve all 44 binary
features. Map labels 1..7 to 0..6. Save exact original row IDs with each split.
The mechanical preparation may extract future rows, but computes no future
quality metric. Training receives only train-development.npz; future-heldout.npz
is not staged until an explicit sealed-family evaluation.

Prepared data pins:

| Artifact | SHA256 |
|---|---|
| train-development.npz | `010f150bab3a700fbcde1106daf21e83cf28fe9024732d29c25756144a07e416` |
| data-manifest.json | `4a6026b41b0abe9594a0320e6455dee17e3fc8ecddcb3aff97657a75b7ae4bcd` |
| separate future-heldout.npz | `7220f0d7282c83732c203325f523da180e4701d8a001647ea0eff4cdc0dbee4f` |

## Model and bounded selection

Use 54→1024→1024→7 with two ReLUs and a bias-free square middle layer. Input and
output layers remain dense and all layers train. Compare Dense, CST alpha .001
(A1048) and CST alpha .01 (A10485). Use unchanged public torchcst975 operators,
stock CSTAdamW and position multiplier 1; no coordinate-LR pilot adoption or
private Graph adapter. Public registry-loaded existing linear/update Plans are
explicit; dispatcher guards remain intact. Ordinary AdamW decay is .01; raw atom
decay 0 and polar amplitude decay .01, with the unchanged radius policy.

All cells use B4, FP32, TF32 off and eager execution. Ordinary input/output seeds
are seed + 1 / + 2; the middle uses seed + 3. Every epoch uses the same CPU randperm
stream seed + 100 + epoch − 1 across methods and LRs. Each epoch presents all 100k rows,
25k updates. One runtime/device profile is locked for the campaign.

1. At seed 17, each method trains fresh at LR 1e-4 / 3e-4 / 1e-3 for 3 epochs. If its
   winner is at an edge, run exactly one extra 3-epoch LR 3e-5 or 3e-3 respectively.
2. Every method restarts its true top 2 LRs fresh for 10 epochs. Choose the minimum
   best-development CE, earliest checkpoint epoch, then lower LR on an exact tie.
   Epoch 0 is not a candidate. Freeze all three configurations.
3. Train all methods at paired seeds 29/43/71 for 10 epochs, using their frozen LR.
   Select each checkpoint by the same development-only rule. Seal all 9 raw
   selected checkpoints and configurations before any future heldout evaluation.
4. A separate assessor requires the exact sealed JSON SHA. It verifies all 9 raw
   source/configuration/file hashes, finite states and optimizer clocks before
   opening any future heldout bytes. Evaluation remains exploratory/descriptive.

HPO and final cells are individually resumable only after their full raw artifact
manifest completed. Partial/uncertain cells fail closed and require inspection;
no experiment is automatically retried. Execute one phase at a time per state.

## Admission and retained evidence

Before learning on each actual GPU: compare full Y, dX and every parameter
VJP against an independent CPU FP64 network composed with public dense_weight;
then compare an eager trajectory of 8 updates with save/reload at 4 updates, including parameters,
all gradients, moments and clocks. Retain raw qualification tensors. CPU smoke
tests do not constitute GPU admission or model-quality results.

Raw admission requires all five parameters' moments/gradients with exact
shape/dtype/ownership and matching CST model-side optimizer store.
Retain raw initialization, every epoch, best/final checkpoints, all optimizer
moments/clocks and gradients, logits/labels/row IDs, learning curves and file
hashes. Record synchronized complete-step train time, startup, qualification,
preparation and evaluation separately, plus training allocated/reserved peaks
(not total process memory). Kernel compilation belongs to admission/preparation,
not training. Cross-VM HPO timing remains exploratory.

Metrics are CE, accuracy, macroF1 and macro one-vs-rest AUC. CE uses float64
log-sum-exp. With $p_{ik}=\operatorname{softmax}(z_i)_k$, report
$\mathrm{CE}=-n^{-1}\sum_i\log p_{i,y_i}$ and
$\mathrm{macroAUC}=7^{-1}\sum_k\mathrm{AUC}(p_{:k},1[y=k])$. Compute softmax in
float64 from saved FP32 logits and use external975 binary_auc per class; never
rank raw per-class logits as if they were probabilities. Evaluate the SAME
selected checkpoint on full train and development sets outside the training
clock and save both prediction arrays and their CE gap. Recomputed development logits,
labels, row IDs and metrics must equal the saved best predictions/metadata using
array bytes; evaluating either split must leave selected model tensors unchanged. Online epoch train loss
averages changing model states and is not a same-state generalization gap.

## Reproduction after source review

Use a clean Git archive of the reviewed consumer and exact unchanged external
`97580582ed28c4e494bf5c2656bbe94a85c544c3` source. Verify its 1188-file manifest
SHA `17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`.
Dependencies stay in an isolated job environment. Freeze the full consumer
manifest after commit. Keep campaign state and outputs outside immutable source.
The direct-script driver disables bytecode before importing consumer/runtime.

```bash
python src/cst_research/experiments/covertype_driver.py --mode campaign \
  --consumer-commit "$CONSUMER_COMMIT" --consumer-manifest "$CONSUMER_MANIFEST" \
  --external-source "$TORCHCST975" --external-manifest "$EXTERNAL_MANIFEST" \
  --data "$DATA/train-development.npz" --data-manifest "$DATA/data-manifest.json" \
  --data-manifest-sha256 4a6026b41b0abe9594a0320e6455dee17e3fc8ecddcb3aff97657a75b7ae4bcd \
  --state "$CAMPAIGN_STATE" --stage screen --method dense
```

Repeat screen for cst-1000/cst-100, retune for all 3, then freeze; confirm requires
one method and seed per call, and seal requires all 9. No provisioning/submission
is included. The owner remains responsible for GPU authorization, allocation,
artifact retrieval and stopping. The separate assessor uses `--mode assess`,
common pinned inputs, `--state`, `--frozen-index`, `--frozen-index-sha256`,
`--heldout` and fresh `--output`. Do not stage the future file for earlier stages.
