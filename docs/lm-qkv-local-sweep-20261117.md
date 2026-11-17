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
