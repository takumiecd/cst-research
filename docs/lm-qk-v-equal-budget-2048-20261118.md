# Equal-Atom QK and V placement at width 2048

English | [日本語](lm-qk-v-equal-budget-2048-20261118.ja.md)

Pair two placements at each of two QKV Atom budgets on the RTX 4090.
The [partial placement study](lm-qk-v-partial-2048-20261118.md) supplies verified V c64 and
Q/K c48 arms. Add Q/K c32 and V c96 at seeds 17, 23 and 42, each to
update 4,882. Each new placement uses the selected projections in all
four blocks, with Dense complementary rows copied byte for byte.

At seed 17, also train V c128 through update 4,882 with Dense Q and K.
Report the V-only c64/c96/c128 response at that seed.

| Atom budget per model | Q / K | V | New arm |
| ---: | --- | --- | --- |
| 524,288 | 32 / 32 | 64 | Q / K |
| 786,432 | 48 / 48 | 96 | V |

Keep the earlier TinyStories model, batches, optimizer, and FP32 radius
decoder. Add only the named Atom counts to the width extension's
admission table. Match the Dense initial parameters. Verify independent
Y, dX and Atom gradients for the first projection and first and terminal
q updates for all replaced sites. For each budget report the paired
QK minus V differences on development and story-disjoint confirmation
CE at each seed and their mean. Also report synchronized full-update
timing and allocator peaks, identifying runs from the earlier job.

For the width-4096 pilot, choose the placement with lower mean confirmation
CE over the two budgets and three seeds, with lower mean update time as
the tie breaker. Carry both per-role coefficient settings for that
placement into the wider pilot.

## Reproduction

Build the registered source snapshot for SM89 and set TORCH_EXTENSIONS_DIR.
Use the prior rational-projection archive and original width-2048 summary
as REFERENCE_RUN and DENSE_ANCHOR respectively.

~~~sh
export PYTHONPATH="$SNAPSHOT/torchcst/src:$SNAPSHOT/torchcst:$SNAPSHOT/cst-research/src"
python scripts/run_lm_qk_v_equal_budget.py --reference-run "$REFERENCE_RUN" \
  --dense-anchor "$DENSE_ANCHOR" --source "$SNAPSHOT" \
  --merged-manifest "$MERGED_MANIFEST" --output "$OUTPUT"
~~~

Resume with --resume and the same inputs and output directory.

## RTX 4090 results

All seven new arms completed 4,882 updates. First-projection independent
output and gradient checks passed with maximum relative L2 error
2.47e-06. Initial Dense slices, initial and terminal q audits, complete
curves and final checkpoint hashes were verified. The low-budget V and
high-budget QK controls are from the partial-placement study. Matching
Atom counts pairs two replaced matrices with one, so the number of Dense
trainable projection weights differs.

| Seed | Atom budget | QK development CE | V development CE | QK confirmation CE | V confirmation CE | QK − V confirmation CE | QK update (ms) | V update (ms) | QK peak alloc. / res. (GiB) | V peak alloc. / res. (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 524,288 | 2.38278 | 2.46557 | 2.32991 | 2.40804 | -0.07812 | 106.45 | 102.46 | 5.596 / 6.062 | 5.723 / 6.172 |
| 17 | 786,432 | 2.40249 | 2.46417 | 2.35462 | 2.40406 | -0.04944 | 107.52 | 103.68 | 5.652 / 6.023 | 5.777 / 6.170 |
| 23 | 524,288 | 2.38255 | 2.49352 | 2.31953 | 2.41131 | -0.09178 | 106.47 | 102.51 | 5.596 / 6.062 | 5.723 / 6.172 |
| 23 | 786,432 | 2.41023 | 2.47988 | 2.34095 | 2.40796 | -0.06702 | 107.53 | 103.73 | 5.652 / 6.023 | 5.777 / 6.170 |
| 42 | 524,288 | 2.38536 | 2.48927 | 2.32454 | 2.39924 | -0.07470 | 106.43 | 102.49 | 5.596 / 6.062 | 5.723 / 6.172 |
| 42 | 786,432 | 2.40769 | 2.48854 | 2.34065 | 2.40885 | -0.06819 | 107.59 | 103.74 | 5.652 / 6.023 | 5.777 / 6.170 |

| Atom budget | Mean QK − V development CE | Mean QK − V confirmation CE |
| ---: | ---: | ---: |
| 524,288 | -0.09923 | -0.08153 |
| 786,432 | -0.07073 | -0.06155 |

The across-budget mean confirmation CE was
2.33504 for QK only and
2.40658 for V only.
The predeclared selection chose QK only for the width-4096 pilot.

Within QK, c32/c32 had lower confirmation CE than c48/c48 in all three
seeds, with a mean difference of −0.02074. Within V, c96 improved on c64
in two seeds, with a mean difference of +0.00076.

## V-only budget sweep at seed 17

| V coefficient | Development CE | Confirmation CE | Median update (ms) | Peak alloc. / res. (GiB) |
| --- | ---: | ---: | ---: | ---: |
| c64 | 2.46557 | 2.40804 | 102.46 | 5.723 / 6.172 |
| c96 | 2.46417 | 2.40406 | 103.68 | 5.777 / 6.170 |
| c128 | 2.46219 | 2.40485 | 104.84 | 5.818 / 6.180 |

- Runner commit `75d4d81c3f72088099ccdf069837290e6484e656`
- Source archive SHA-256 `dfcad9777a5539f9b5c17b5d19d65bf2c3f36769ecd98c50819a61fbe3201a99`
- Verified summary SHA-256 `0e260ddb99cbdf0fcbebbde157f53d1d9fa49974a2d090e9f8c56614a50c8312`
