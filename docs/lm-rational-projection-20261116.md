# TinyStories rational CST projection comparison

English | [日本語](lm-rational-projection-20261116.ja.md)

Run Dense, FFN down only, attention output only, and QKV only from seed 17
on the RTX 4090. Keep the four width-1024 blocks, eight heads, context 128,
expansion four, 1,024 targets per update, and 4,882 updates. The original
19,531-update learning rate schedule and data order apply to all arms.

| Arm | Replacements | Atoms per projection | Atom count / replaced Dense weights |
| --- | ---: | ---: | ---: |
| Dense | 0 | 0 | — |
| FFN down | 4 | 196,608 | 4.6875% |
| Attention output | 4 | 49,152 | 4.6875% |
| QKV c64 | 12 | 65,536 | 6.25% |

The frozen QKV recipe fits initialization moments using the exponential law,
then switches to the rational lower bound for execution. Share this setup
order across CST arms. The radius range is 0.5 to 512, with
(L(a)=0.5+3.5/(1+3(a/0.3)^2)) and the original logarithmic interpolation.
CSTAdamW learns the radius fraction starting at zero. The original log-space
tau schedule runs from 0.03 to 0.003 in 1,536 updates. Audit all four sites in
each new placement on the first task update and check a full-size forward and
VJP against the existing FP64 factors on one layer per shape.

The primary metric is development CE at update 4,882 on 8,192 targets.
Evaluate the story-disjoint 32,768-target confirmation split after restoring
the final checkpoint. Report per-arm synchronized training step medians and
Torch allocated/reserved peaks after the first ten steps. The run provides one
seed per arm.

Use the source and data manifests referenced in the study entry. Copy the
archived rational QKV runtime into an isolated directory and apply the
two-file CUDA admission delta from torchcst commit
`31f40e64241eb866e95305401b94cc8da4ac5a7e` relative to its parent. Keep
the archived `cst-research` sources alongside the runtime under `source/`.
Use PyTorch 2.9.1+cu128 and the RTX 4090 environment.

```sh
export PYTHONPATH="$SNAPSHOT/torchcst/src:$SNAPSHOT/cst-research/src"
python scripts/run_lm_rational_projection.py \
  --archive-root "$ARCHIVES" --source "$SNAPSHOT" --output "$OUTPUT"
```

The runner verifies the input files and both original initialization digests
before using the archived comparison. On interruption, use the same command with
`--resume` to check and restore the latest recorded checkpoint.
