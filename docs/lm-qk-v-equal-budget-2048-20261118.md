# Equal-Atom QK and V placement at width 2048

English | [日本語](lm-qk-v-equal-budget-2048-20261118.ja.md)

Pair two placements at each of two QKV Atom budgets on the RTX 4090.
The [partial placement study](lm-qk-v-partial-2048-20261118.md) supplies verified V c64 and
Q/K c48 arms. Add Q/K c32 and V c96 at seeds 17, 23 and 42, each to
update 4,882. Each new placement uses the selected projections in all
four blocks, with Dense complementary rows copied byte for byte.

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
