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

Use isolated copies of the archived training source and data. Overlay the
torchcst files changed by commit 6703a18a55cd949564be8a0ceca1ca1b813f7fc5
relative to 31f40e64241eb866e95305401b94cc8da4ac5a7e and pass their SHA-256
manifest. The archived rational adapter and original trainer are verified
against the previous run. The extension is built from the merged source
with TORCH_CUDA_ARCH_LIST=8.9. The runner checks the frozen dataset, source
files, native binary identity, output identities, and restored checkpoints.

~~~sh
export PYTHONPATH="$SNAPSHOT/torchcst/src:$SNAPSHOT/torchcst:$SNAPSHOT/cst-research/src"
export TORCH_CUDA_ARCH_LIST=8.9
python scripts/run_lm_fp32_seed_replication.py \
  --reference-run "$REFERENCE_RUN" --source "$SNAPSHOT" \
  --merged-manifest "$MERGED_MANIFEST" --output "$OUTPUT"
~~~

Resume from the latest recorded checkpoint with the same inputs and
the additional flag --resume. The registry records the evidence location.
