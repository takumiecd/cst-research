# Width-2048 QKV Atom allocation toward V

English | [日本語](lm-qkv-v-allocation-20261117.ja.md)

The [allocation pilot](lm-qkv-role-budget-20261117.md) found the lowest
seed-17 CE with extra Atoms at V. Test V-heavy QKV at the uniform c64
budget and at a lower budget. Each condition replaces Q, K, and V in
four blocks. Train all three conditions from initialization at seeds
17, 23, and 42 through update 4,882 on one RTX 4090.

| Arm | Q | K | V | Atoms per model |
| --- | ---: | ---: | ---: | ---: |
| Uniform control | 64 | 64 | 64 | 1,572,864 |
| Matched budget | 56 | 56 | 80 | 1,572,864 |
| Reduced budget | 48 | 48 | 64 | 1,310,720 |

Use the [uniform sweep](lm-qkv-local-sweep-20261117.md) model width, context,
frozen TinyStories splits and data order, FP32 radius decoder, GPU initializer,
optimizer and learning-rate schedule. Check each of the three replayed c64
initial parameter hashes against its original run and check the common
Dense initialization across all nine runs. The seed-17 qualification compares Y,
dX and the four Atom gradient fields of the first Q to an independent reference.
Verify CPU/GPU bitwise initialization at the first Q for the two new
allocations and check first and terminal q updates at all twelve sites.

For each candidate, report per-seed and mean paired differences in final
confirmation CE, development CE and complete-update latency versus the
replayed c64 arm in the same job. Confirmation CE uses the story-disjoint
split. Report how many seeds favor each allocation on both splits.
Interpret the three-seed results as exploratory. Record development top-1,
model preparation time and PyTorch allocator peaks. Run order rotates
across the three seeds as declared in the runner.

## Reproduction

Use the same frozen source snapshot, original projection archive, merged
manifest, native extension and paired anchor as the uniform sweep. Set
TORCH_EXTENSIONS_DIR to the existing SM89 extension directory, and use a
new OUTPUT directory.

~~~sh
export PYTHONPATH="$SNAPSHOT/torchcst/src:$SNAPSHOT/torchcst:$SNAPSHOT/cst-research/src"
python scripts/run_lm_qkv_v_allocation.py --reference-run "$REFERENCE_RUN" \
  --paired-anchor "$PAIR_ANCHOR" --source "$SNAPSHOT" \
  --merged-manifest "$MERGED_MANIFEST" --output "$OUTPUT"
~~~

Resume the same output directory with the frozen inputs and --resume.
