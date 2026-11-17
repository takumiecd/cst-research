# Local QKV Atom-count sweep at width 2048

English | [日本語](lm-qkv-local-sweep-20261117.ja.md)

Compare uniform QKV c48, c56, c72, and c80 against the same-seed c64 and
Dense controls from the [width-2048 study](lm-qkv-2048-20261117.md). Train
all four counts from initialization at seeds 17, 23, and 42 through update
4,882. Replay c64 at seed 17 with the new initialization path. This makes
13 new training runs. Only Q, K, and V are replaced in the four blocks.

| Coefficient | Atoms per projection | Atom count / Dense weights |
| ---: | ---: | ---: |
| 48 | 98,304 | 2.344% |
| 56 | 114,688 | 2.734% |
| 64 | 131,072 | 3.125% |
| 72 | 147,456 | 3.516% |
| 80 | 163,840 | 3.906% |

Keep width 2048, eight heads, context 128, and FFN expansion four. Keep the
frozen TinyStories GPT-2 tokens and order seed 905 at 1,024 targets per
update. Use the prior per-field learning rates, ordinary 0.0001 and Atom
0.0096, through the same 19,531-update schedule, and the same radius
envelope and tau schedule. The Dense initialization digest is checked
against the prior run for every seed. The c64 replay also checks the full
initial parameter digest. The GPU initializer keeps the CPU random stream
and FP64 radius bank selection and uses FP32 energies on the GPU.

Before training, check bitwise CPU/GPU initialization of the first Q
projection at every new count on seed 17. Qualify Y, dX and the four Atom
gradient fields against independent CPU factors and verify the first q
update in all twelve sites. The first and terminal q updates are checked
again for each training run.

Use all thirteen results. The primary comparison is the development CE
difference versus c64 for the same seed. Record CE on the story-disjoint
confirmation set and development top-1. Report all four conditions and
all three seeds, as well as the seed-17 replay difference. Timing and
memory use the prior study's synchronized complete eager update and
PyTorch allocator measurements. Arm order is prespecified in the runner.

## Reproduction

Extend the prior width-2048 source snapshot with the torchcst changes in
the registered combined manifest. Reuse the original rational projection
archive, frozen dataset, and FP32 adapters. Build the width extension for
SM89 and set TORCH_EXTENSIONS_DIR. PAIR_ANCHOR points to the prior
width-2048 evidence/summary.json.

~~~sh
export PYTHONPATH="$SNAPSHOT/torchcst/src:$SNAPSHOT/torchcst:$SNAPSHOT/cst-research/src"
python scripts/run_lm_qkv_local_sweep.py --reference-run "$REFERENCE_RUN" \
  --paired-anchor "$PAIR_ANCHOR" --source "$SNAPSHOT" \
  --merged-manifest "$MERGED_MANIFEST" --output "$OUTPUT"
~~~

Resume the same output directory with the frozen inputs and --resume.

## RTX 4090 results

All 13 new runs reached update 4,882. Restored checkpoints, complete curves
and first and terminal q updates were checked. The seed-17 forward/backward
gates across the five shapes had maximum relative L2 error 2.25e-06.
The four new shapes passed bitwise CPU/GPU initialization of the first Q.

| Seed | Arm | Development CE | Confirmation CE | Development top-1 (%) | Median update (ms) | Peak allocated / reserved (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 17 | Dense (prior) | 2.43763 | 2.37238 | 48.16 | 96.97 | 5.754 / 6.193 |
| 17 | c64 (prior) | 2.46632 | 2.39698 | 47.49 | 115.24 | 5.657 / 6.057 |
| 17 | c64 (replay) | 2.47220 | 2.40649 | 47.08 | 115.25 | 5.657 / 6.057 |
| 17 | c48 | 2.48145 | 2.40721 | 46.80 | 112.76 | 5.598 / 6.070 |
| 17 | c56 | 2.44947 | 2.38576 | 48.22 | 114.07 | 5.626 / 6.115 |
| 17 | c72 | 2.47923 | 2.41551 | 47.66 | 116.76 | 5.702 / 6.139 |
| 17 | c80 | 2.56047 | 2.49119 | 46.20 | 117.85 | 5.734 / 6.109 |
| 23 | Dense (prior) | 2.43700 | 2.37684 | 47.56 | 97.01 | 5.754 / 6.193 |
| 23 | c64 (prior) | 2.47425 | 2.40955 | 47.44 | 115.20 | 5.657 / 6.057 |
| 23 | c48 | 2.47953 | 2.40475 | 47.60 | 112.87 | 5.598 / 6.070 |
| 23 | c56 | 2.48170 | 2.41120 | 47.44 | 114.22 | 5.626 / 6.115 |
| 23 | c72 | 2.49935 | 2.42107 | 47.57 | 116.73 | 5.702 / 6.139 |
| 23 | c80 | 2.51608 | 2.44257 | 46.72 | 117.77 | 5.734 / 6.109 |
| 42 | Dense (prior) | 2.46177 | 2.37810 | 47.83 | 96.99 | 5.754 / 6.193 |
| 42 | c64 (prior) | 2.48145 | 2.40921 | 46.97 | 115.18 | 5.657 / 6.057 |
| 42 | c48 | 2.45977 | 2.39776 | 47.24 | 112.83 | 5.598 / 6.070 |
| 42 | c56 | 2.47994 | 2.38999 | 47.13 | 114.17 | 5.626 / 6.115 |
| 42 | c72 | 2.49000 | 2.42494 | 47.05 | 116.78 | 5.702 / 6.139 |
| 42 | c80 | 2.63986 | 2.55832 | 44.67 | 117.89 | 5.734 / 6.109 |

The three-seed means use the prior same-seed c64 and Dense controls.
Positive CE differences indicate higher cross-entropy. The controls had
mean median update times of 115.21 ms (c64) and 96.99 ms (Dense).

| Coefficient | Δ development vs c64 | Δ confirmation vs c64 | Δ development vs Dense | Δ confirmation vs Dense | Median update, mean (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| c48 | -0.00042 | -0.00201 | +0.02812 | +0.02747 | 112.82 |
| c56 | -0.00364 | -0.00960 | +0.02491 | +0.01988 | 114.15 |
| c72 | +0.01552 | +0.01526 | +0.04406 | +0.04473 | 116.76 |
| c80 | +0.09813 | +0.09212 | +0.12668 | +0.12159 | 117.83 |

The c64 replay reproduced the full seed-17 initial parameter hash and
used the same frozen batch order. Its development and confirmation CE
differed by +0.00588 and +0.00951 from the earlier c64 run. Its model
preparation took 1.23 seconds, compared with
304.20 seconds previously. Model
preparation across the new runs took 1.01–1.23 seconds.
The seed-17 replay difference is a reference for interpreting small
CE differences.

Verified summary SHA-256 `7628959c08456024929199e88479a398655aad19b100ef94bc487a688fe3879f`.
