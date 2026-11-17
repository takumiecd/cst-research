# Fixed-budget allocation within width-2048 QKV

English | [日本語](lm-qkv-role-budget-20261117.ja.md)

Test three placements of the extra Atom budget on seed 17. Each of the
four blocks replaces Q, K, and V, with 524,288 Atoms across the three
maps (2,097,152 per model). Finish 4,882 updates for each condition.

| Arm | Q coefficient | K coefficient | V coefficient |
| --- | ---: | ---: | ---: |
| Extra Q | 128 | 64 | 64 |
| Extra K | 64 | 128 | 64 |
| Extra V | 64 | 64 | 128 |

Use the width, TinyStories splits and frozen batch order from the
[uniform sweep](lm-qkv-local-sweep-20261117.md), along with the same
optimizer rates, schedule and GPU initialization. Prespecified training
order is extra V, extra Q, extra K. Verify the same Dense initial
weights outside the QKV maps and the exact Atom allocation. The first
Q map in each arm is compared with the independent CPU forward/gradient
reference and its CPU initializer. The first q update covers all twelve
sites at their actual counts. Record restored development and
story-disjoint confirmation CE, top-1, synchronized update time, and
allocator peaks.

The primary comparison is final development CE across the three
same-budget arms. The uniform c64 and c128 values at seed 17 serve as
historical references at different total Atom counts. Treat this as
a one-seed placement pilot.

Use the same torchcst source snapshot and binary as the uniform sweep.
Run the same command from that study with
scripts/run_lm_qkv_role_budget.py and a fresh OUTPUT. The prior width-2048
summary is the PAIR_ANCHOR. Resume with --resume and the same inputs.
