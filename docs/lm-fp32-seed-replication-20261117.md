# Paired TinyStories FFN-down replication with FP32 radius decode

English | [日本語](lm-fp32-seed-replication-20261117.ja.md)

This replication fixes the width at 1024 with four blocks, eight heads, context
128, and FFN expansion 4. Each run trains from initialization for 4,882 updates
of 1,024 targets. The optimizer, 19,531-update learning-rate schedule, q
initialization, Atom count and data order follow the
[seed-17 placement study](lm-rational-projection-20261116.md).

Seeds 23 and 42 each pair an all-Dense model with FFN down replaced in four
blocks by 196,608 Atoms per projection. Each pair shares its Dense initialization
outside the replaced slices and sees identical batches in identical order.
Data ordering uses the frozen seed 905 across training seeds. Seed 17 repeats
the FFN-down condition on the new implementation. Its previous Dense run and
previous FFN-down run provide historical anchors with checked initialization and
result hashes. The run order is seed 23 Dense then down, seed 42 down then
Dense, followed by the seed-17 down replay.

The private FP32 radius and shared-weight training adapters are the ones used
in the checkpoint timing work. The native Fourier weight and CPU-rounded
amplitude cap come from the merge of torchcst PR #168 into
kernel/lm-rational-projection-admission. The four weights are built once per
optimizer update, and their cotangents accumulate across microbatches before
the Atom VJP. Evaluation uses the same FP32 decoder and weight construction.
The independent FP64 factor oracle and initial q-update checks qualify the
fresh seed-23 model before its full FFN-down run. Every FFN-down run records
the first and terminal q updates.

The primary outcome is paired FFN-down minus Dense development CE at step
4882 on 8,192 targets. The fixed, story-disjoint confirmation set supplies a
second CE on 32,768 targets. Record development top-1 and intermediate curves,
step timing, and peak allocated and reserved memory. Comparisons are
descriptive by seed, with no tuning or formal equivalence claim.

Set `REFERENCE_RUN` to the retained `lm-rational-projection-20261116-02`
directory, including its trainer, result and source archive. Set `SNAPSHOT`
to an isolated copy of the archived training source with the original
`cst-research` snapshot alongside `torchcst`. The frozen data paths come from
the reference manifest. Overlay the torchcst files changed by commit
`6703a18a55cd949564be8a0ceca1ca1b813f7fc5` relative to
`31f40e64241eb866e95305401b94cc8da4ac5a7e` and pass their SHA-256 manifest.
In the PyTorch 2.9.1+cu128 environment, build the extension from that snapshot
or reuse a verified build with the same source digest. The runner checks the
frozen dataset, source files, native binary identity, output identities, and
restored checkpoints.

~~~sh
export PYTHONPATH="$SNAPSHOT/torchcst/src:$SNAPSHOT/torchcst:$SNAPSHOT/cst-research/src"
export TORCH_CUDA_ARCH_LIST=8.9
export TORCH_EXTENSIONS_DIR="$EXTENSIONS"
mkdir -p "$TORCH_EXTENSIONS_DIR"
python - <<'PY'
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_width import width_extension

width_extension()
PY
python scripts/run_lm_fp32_seed_replication.py \
  --reference-run "$REFERENCE_RUN" --source "$SNAPSHOT" \
  --merged-manifest "$MERGED_MANIFEST" --output "$OUTPUT"
~~~

Resume from the latest recorded checkpoint with the same inputs and
the additional flag `--resume`. The registry records the evidence location.

## RTX 4090 results

All five new runs completed update 4,882 and reproduced development CE and
top-1 after restoring the final checkpoints. Seed 17 uses the previous Dense
run. The optimized replay and previous down run share the same initial
parameter digest. A negative CE difference favors FFN down.

| Seed | Dense development CE | FFN down development CE | Δ | Dense confirmation CE | FFN down confirmation CE | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | 2.53073 | 2.51144 | -0.01929 | 2.45098 | 2.44584 | -0.00514 |
| 42 | 2.50373 | 2.52609 | +0.02235 | 2.43854 | 2.44764 | +0.00910 |
| 17 (archived Dense) | 2.51733 | 2.50175 | -0.01559 | 2.44804 | 2.43966 | -0.00838 |

The two new seeds split one each on both data splits. The mean paired CE
differences over those seeds were +0.00153 on development and
+0.00198 on confirmation. The seed-17 replay moved the
previous down CE from 2.49768 to 2.50175 on development
(+0.00407) and from 2.43741 to
2.43966 on confirmation (+0.00225).
Each comparison uses the same model width and data order.

The independent CPU FP64 radius cell mass check for the full seed-23 down
projection (196,608 Atoms, three inputs) covered Y, dX and the four Atom
gradients. The maximum relative L2 error was 1.66e-06 and the maximum
absolute error was 4.65e-06. The first task update produced positive q
in all four projections in each down run. Terminal q and optimizer states
passed the checkpoint checks.

| Seed | Arm | Development top-1 (%) | Median step (ms) | Initialization (s) | Peak allocated / reserved (GiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 23 | Dense | 46.77 | 36.65 | 0.35 | 2.161 / 2.441 |
| 23 | FFN down | 46.66 | 44.01 | 297.70 | 4.654 / 4.838 |
| 42 | Dense | 46.53 | 36.74 | 0.28 | 3.433 / 3.723 |
| 42 | FFN down | 46.59 | 43.89 | 314.49 | 2.819 / 3.006 |
| 17 | FFN down | 46.84 | 44.01 | 324.68 | 3.453 / 3.723 |

Timing uses synchronized eager training for updates 11–4,882, including
forward, backward and the optimizer update. Allocator peaks span the training
process and can include allocations carried over from a prior arm or the
qualification copy. The identical Dense architecture reported different
peaks across seeds in this sequence, so these values describe this process.

The retained run ID is `lm-fp32-seed-replication-20261117-01` on the
RTX 4090 with PyTorch 2.9.1+cu128, CUDA 12.8,
Triton 3.5.1 and TF32 disabled. The retrieved summary, five
curves, seed results and five final checkpoint hashes were verified.
Checkpoints remain on the training host. The summary SHA-256 is `6a4a062cd6b7d5628b8cbdd6a0c4f823cd91d3b52b376b045c13d22661db5c41`.
