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
Create a fresh OUTPUT directory. Invoke the previous command with
scripts/run_lm_qkv_down_v_budget.py and --pilot-summary "$PILOT_SUMMARY"
alongside the source, manifest and anchor arguments. Resume with --resume
and the same inputs and output directory.
