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

## RTX 4090 results

All nine runs reached update 4,882. The full training curves and final
checkpoint hashes matched the run records. All three initial c64 hashes
matched the original runs. Both new allocations passed the bitwise
initializer check at the first Q. The independent output and gradient
checks had maximum relative L2 error 2.25e-06. The first and terminal
q audits passed at all twelve sites per run.

| Seed | Q / K / V | Development CE | Confirmation CE | Development top-1 (%) | Median update (ms) | Model preparation (s) | Peak allocated / reserved (GiB) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | (64, 64, 64) | 2.46175 | 2.39464 | 47.55 | 115.12 | 1.19 | 5.657 / 6.057 |
| 17 | (56, 56, 80) | 2.44493 | 2.39101 | 47.75 | 114.99 | 1.09 | 5.662 / 6.109 |
| 17 | (48, 48, 64) | 2.45411 | 2.38740 | 47.67 | 113.35 | 1.05 | 5.618 / 6.105 |
| 23 | (64, 64, 64) | 2.47421 | 2.40967 | 47.09 | 115.14 | 1.10 | 5.657 / 6.057 |
| 23 | (56, 56, 80) | 2.45187 | 2.38778 | 47.94 | 115.00 | 1.12 | 5.662 / 6.109 |
| 23 | (48, 48, 64) | 2.47349 | 2.39716 | 46.86 | 113.36 | 1.08 | 5.618 / 6.105 |
| 42 | (64, 64, 64) | 2.48922 | 2.41238 | 46.58 | 115.25 | 1.07 | 5.657 / 6.057 |
| 42 | (56, 56, 80) | 2.46718 | 2.39886 | 46.88 | 115.03 | 1.14 | 5.662 / 6.109 |
| 42 | (48, 48, 64) | 2.45460 | 2.39110 | 47.33 | 113.42 | 1.04 | 5.618 / 6.105 |

The following paired differences subtract the uniform c64 replay at the
same seed. Lower CE is better. The uniform replay's mean of median update
times was 115.17 ms.

| Seed | Q / K / V | Δ development CE | Δ confirmation CE | Δ median update (ms) |
| ---: | --- | ---: | ---: | ---: |
| 17 | (56, 56, 80) | -0.01681 | -0.00363 | -0.12 |
| 17 | (48, 48, 64) | -0.00763 | -0.00724 | -1.77 |
| 23 | (56, 56, 80) | -0.02234 | -0.02189 | -0.14 |
| 23 | (48, 48, 64) | -0.00072 | -0.01251 | -1.79 |
| 42 | (56, 56, 80) | -0.02204 | -0.01352 | -0.21 |
| 42 | (48, 48, 64) | -0.03462 | -0.02127 | -1.83 |

| Q / K / V | Mean Δ development CE | Seeds with lower development CE | Mean Δ confirmation CE | Seeds with lower confirmation CE | Mean Δ median update (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| (56, 56, 80) | -0.02040 | 3/3 | -0.01301 | 3/3 | -0.16 |
| (48, 48, 64) | -0.01432 | 3/3 | -0.01368 | 3/3 | -1.80 |

Runtime uses PyTorch 2.9.1+cu128, CUDA 12.8 and Triton 3.5.1.
Frozen research driver commit `dd4b3921c895359fad8aa7b01c44bea2b00ab8c4`.
Script archive SHA-256 `2eb17a932ceb9c6f2ffbf90559ddf5bb2f338d63e82464a3e34dbc1c599ba7e9`.

Verified summary SHA-256 `8dcee587925476f5c03ad3b696d87f971f272b0c5ddaa7c3ae820f1ea4144224`.
