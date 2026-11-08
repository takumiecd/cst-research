# LM width scaling and complete matrix replacement

English (canonical) | [日本語](lm-width-scaling-plan-20261108.ja.md)

Status: design/preparation only. The owner selected LM width scaling after the
RTX 4090 kernel comparison and expanded the target to the output maps and other
learned matrices. This document does not register a running study, resume the
stopped confirmation campaign, or claim that the required backends exist.

## Question and fixed family

Test whether CST remains useful as model width grows, with a fixed coefficient
`A = ceil(c * sqrt(d_in * d_out))`, initially c=8. This is a fixed-budget-family
experiment, not a search for the minimum Atom count or an Omega claim.

Proposed widths: 512, 1024, 2048. Keep four pre-LN blocks, FFN expansion4,
context128, batch2, accumulation4 (1024 tokens/update), tokenizer/vocabulary,
training/development streams, training token budget and c fixed. A constant
head dimension64 (8/16/32 heads) is proposed to define one width family; this
requires new explicit model configuration and fresh Dense controls. It must
not inherit the old width512/four-head observations as a matched baseline.

Use the existing TinyStories/GPT2-tokenizer data protocol and verified token
manifest. Starting optimization recipe: ordinary LR3e-4, whole-Atom multiplier16,
AdamW(.9,.95), decay0 and global clip1. Freeze the qualified Polar/initialization
recipe before learning; the minimum-001/q0 recipe is a starting candidate, not
an already validated all-matrix LM recipe. Match each Dense target's second
moment with the public CST initializer, including residual output scaling.
Keep FP32/TF32-off as the first precision family. Source, data, LR schedule,
actual token budget and selection rules must be fixed before launch.

## Replacement conditions

| Condition | CST matrices |
| --- | --- |
| Dense | None |
| FFN | FFN up/down in every block |
| QKV | Q/K/V in every block |
| QKV+FFN | Both groups above |
| +O | Q/K/V/O and FFN up/down in every block |
| +shared vocabulary | Above plus the shared token embedding/vocabulary output |
| all learned matrices | Above plus the learned positional embedding |

The first four conditions separate FFN and QKV effects. Additional conditions
are cumulative: establish feasibility at width512 before adding the successful
fixed family to1024/2048. Do not immediately expand every unqualified condition
into a large multi-seed grid. LayerNorm, softmax, causal masking, activations,
residual addition and one-dimensional biases/normalization coefficients retain
their operations; “all matrices” is not a replacement of every computation.

## Tied vocabulary and backend gates

The current LM computes logits with `embedding.weight`; token embedding and
vocabulary output are tied. Preserve one shared CST Parameter/AtomState for
lookup and decoding, one optimizer state/update per shared Parameter, combined
gradient accumulation and checkpoint identity. A separate untied comparison
would need its own matching Dense control and is outside this first design.
The positional embedding remains learned and is a separate matrix-replacement
condition; do not silently switch to a different positional architecture.

Current measured operator source095684f0 qualifies square N2048/8192/16384 and
small FFN shapes. It does not qualify all width-study rectangles, a50,257-row
vocabulary projection or CST embedding lookup. Runtime/operator/backend work
belongs in torchcst, not copied into this research repository. Before training:

- Admit and independently qualify the actual square and up/down rectangular
  shapes, A8-geometric-mean budgets, R256 and both retention choices.
- Implement/test vocabulary shape handling and direct embedding lookup without
  materializing a full dense weight or token one-hot matrix during training.
  Preserve the actual vocabulary, softmax domain and weight tying. Internal
  padding must preserve operator semantics and be disclosed/qualified.
- Check forward, dX and all Atom gradients against CPU FP64/profile oracles;
  include repeated token IDs, boundaries, shared encode+decode gradients,
  deduplicated optimizer state, serialization and exact next-update resume.
- Match initializer targets and record per-row energy/coverage, zero rows,
  activations and gradients. A highly rectangular vocabulary map may expose
  issues not seen in balanced FFN/attention maps; record them before changing
  kernel, c, radius or LR. Such changes become a separate version.

## Execution stages and decision boundary

1. Finish/audit/retrieve the4090 component comparison. Keep A100 MIG and4090
   timings separate, each with its own Dense control.
2. CPU declarations and independent GPU operator/optimizer gates; then two real
   LM updates and serialized next-update/DEV reproduction for each new family.
3. Short development learning at width512 first. Proposed initial budget is
   about1M tokens/condition, seed17, with the existing fixed20M-token LR/data
   horizon. This is a proposal; no training is launched by this document.
4. Freeze working conditions and equal token budgets, then run the width family.
   Overlay train/DEV CE and perplexity versus both tokens and wall time. Record
   actual steps/tokens, full-update time, allocated/reserved peaks and setup.
5. Longer three-seed confirmation follows a separately fixed budget/protocol;
   do not relaunch the owner's stopped old confirmation or select from final
   confirmation scores. No universal optimum/noninferiority claim is assumed.

Use existing research learners and qualifications, extending their explicit
configuration/admission boundaries rather than creating another training loop.
Full-update timing must state whether zero_grad/LR assignment, transfers,
accumulation, clipping and optimizer are inside the timer; preserve historical
clock definitions on their original sources. Geometry/rank/coverage diagnostics,
evaluation and checkpoint IO stay outside primary timing. Any failed, partial,
OOM or unsupported-shape case remains part of the record.

Q/K/V remain three logical maps, with the Atom rule applied to each, as in the
existing pilots. They are not silently replaced by one3N-output CST map with a
different budget. With four blocks, CST site counts are8 (FFN),12 (QKV),20
(QKV+FFN),24 (+O), then25 distinct learned matrices with tied vocabulary and26
with position. Tied vocabulary has two uses but one learned matrix/state.
