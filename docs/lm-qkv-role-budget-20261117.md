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

## RTX 4090 results

All three arms completed 4,882 updates at seed 17 and retained final
checkpoints. The first Q check was bitwise equal for CPU versus GPU
initialization in each arm. The maximum relative L2 error across the
independent output and gradient checks was 2.18e-06. All twelve
q sites per arm passed the first and terminal update audits.

| (Q, K, V) | Development CE | Confirmation CE | Development top-1 (%) | Median update (ms) | Peak allocated / reserved (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (128, 64, 64) | 2.68622 | 2.59546 | 44.57 | 118.48 | 5.752 / 6.115 |
| (64, 128, 64) | 2.74670 | 2.64953 | 43.81 | 118.43 | 5.748 / 6.191 |
| (64, 64, 128) | 2.46447 | 2.39321 | 47.78 | 117.27 | 5.749 / 6.111 |

Allocating c128 to V gave the lowest development and confirmation CE
at seed 17. For context, the prior same-seed
development / confirmation CEs at their respective budgets were
2.43763 / 2.37238 (Dense),
2.46632 / 2.39698 (uniform c64),
and 3.15618 / 3.00745 (uniform c128).

Verified summary SHA-256 `a08b283b75c0cc52ff668e386bff8a3174c3a5c964a7aabe73985afb8fc5df6a`.
