# QKV Atom budget with FFN down at width 2048

English | [日本語](lm-qkv-down-v-budget-20261118.ja.md)

Replay the [combined arm](lm-qkv-down-2048-20261118.md) from initialization
with Q/K/V coefficients (56, 56, 80). Compare it with that study's
archived (48, 48, 64) combined run. Use seed 17, 4,882 updates, the same
RTX 4090, frozen TinyStories splits, batch order and optimizer schedule.
Each of the four down projections keeps 393,216 Atoms and the same
initial parameter digest. The larger QKV arm uses 1,572,864 QKV Atoms
per model, up from 1,310,720. Check its QKV initialization against the
[earlier QKV-only arm](lm-qkv-v-allocation-20261117.md) and check the shared
Dense initialization.

Reuse the exact prior width extension, FP32 radius decoder and GPU
variance calibration. Check the first Q projection against the
independent output and gradient oracle before training. Verify the
first and terminal q updates at all sixteen sites.

Report the final development and story-disjoint confirmation CE
differences against the archived combined arm. Record synchronized
complete-update medians and PyTorch peak allocated and reserved memory.
This is a one-seed comparison using an archived control.

## Reproduction

Use the source snapshot and extension from the width-2048 combined study.
Take its completed summary as PILOT_SUMMARY and use its other anchors.
Set TORCH_EXTENSIONS_DIR to the prior SM89 build and use a fresh OUTPUT
directory.

~~~sh
export PYTHONPATH="$SNAPSHOT/torchcst/src:$SNAPSHOT/torchcst:$SNAPSHOT/cst-research/src"
python scripts/run_lm_qkv_down_v_budget.py --reference-run "$REFERENCE_RUN" \
  --paired-anchor "$PAIR_ANCHOR" --qkv-anchor "$QKV_ANCHOR" \
  --pilot-summary "$PILOT_SUMMARY" --source "$SNAPSHOT" \
  --merged-manifest "$MERGED_MANIFEST" --output "$OUTPUT"
~~~

Resume with --resume and the same inputs and output directory.

## RTX 4090 results

The (56, 56, 80) combined run reached update 4,882 at seed 17.
The four down initializations matched the archived combined arm and the
twelve QKV initializations matched the larger QKV-only control. The first
Q output and gradient oracle passed with maximum relative L2 error
2.25e-06. All sixteen first and terminal q audits, the complete training
curve and the final checkpoint hash passed verification.

| Q / K / V with down | Development CE | Confirmation CE | Development top-1 (%) | Median update (ms) | Preparation (s) | Peak allocated / reserved (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 48 / 48 / 64 | 2.63859 | 2.56407 | 45.36 | 135.13 | 1.35 | 5.398 / 5.986 |
| 56 / 56 / 80 | 2.65907 | 2.57685 | 44.76 | 136.82 | 1.55 | 5.442 / 6.025 |

The larger allocation changed development CE by +0.02048
and confirmation CE by +0.01278 relative to the archived
combined arm. Relative to the (48, 48, 64) QKV-only run (confirmation CE
2.39050), the confirmation gap was +0.17358 with the original
combined allocation and +0.18635 with the larger allocation.
The archived (56, 56, 80) QKV-only confirmation CE was 2.39101.

The runner reused the prior native extension on an RTX 4090 with
PyTorch 2.9.1+cu128, CUDA 12.8 and Triton 3.5.1.
The frozen runner commit was `78524a819f3114b10fe54e77dd0e45ff1f2f9b5d` and the script archive
SHA-256 was `fc69b03262caaa0a99e9fbfd6c12c23a6ee2cf6ad5aa262b4bcbcbe36206119d`.

Verified summary SHA-256 `2850900ae00f2deffc950268ea59ba0783dabd5cb3c90caa19e34de1022bc837`.
