# CST FFN feasibility

English (canonical) | [日本語](ffn-feasibility.ja.md)

## Question and status

Can CST replace both FFN projections in every block of a small causal language
model, with a fixed coefficient in `A=ceil(c sqrt(d_in d_out))`? This study does
not search for a minimum Atom count or establish a Big-Omega bound. Preparation,
native qualification, learning and independent result verification are separate
milestones. Registration is not evidence of a successful language model.

The initial recipe is four pre-LayerNorm Transformer blocks, four attention
heads, width256, FFN `256→1024→256` with GELU, context128 and no dropout. All
eight FFN projections are CST in the CST arm. Embeddings, attention and tied
output weights remain Dense. Dense-FFN and no-FFN arms retain exactly the same
shared ordinary initialization streams, task data and token budget. The no-FFN
arm omits the FFN residual contribution: attention alone can learn, so loss
reduction by itself cannot demonstrate useful CST FFNs.

## Atom budget and initialization

Initially c=8 and q=0. For an FFN `N→4N→N`, each projection has A=16N Atoms.
Each Polar Atom has four trainable coordinates. Including the two FFN biases,
each block therefore has `128N+5N=133N` FFN parameters, versus `8N²+5N` for Dense.
The width cohorts are N128/256/512, with FFN widths512/1024/2048. This is linear
FFN parameter accounting, not an O(N) runtime, GPU-peak or whole-model claim.
Dense attention still has quadratic parameter terms.

Use public `torchcst.init.variance_scaling_` before optimizer construction or
CUDA transfer. Target Dense std0.02, with residual-output projections scaled by
`1/sqrt(2*layers)` in both arms. Record all initialization reports, actual
activation RMS and gradients. The Polar radius cell-mass contract detaches task
gradients through derived radii; preserve that declared semantics. No CST-only
runtime gain or normalization is added. q is an initial radius fraction, not an
Atom density and not a frozen training coordinate.

## Qualification before task learning

1. Independent raw-polynomial quadrature checks a bounded rectangular fixture,
   including amplitude/center VJPs under the declared detached-radius contract.
2. CPU FP64 factored/chunked references check native Y, dX and every Atom
   coordinate for both projection directions, noncontiguous inputs and multiple
   tokens. This reference shares mathematical factor code, but no CUDA hierarchy
   or native backward. It complements, rather than replaces, quadrature.
3. Controlled updates compare native-model CSTAdamW with the public optimizer
   wrapper reference, including Parameters, moments and counters. A serialized
   checkpoint must reproduce the next complete update exactly.
4. Model checks cover causal masking, common initialization across arms,
   real CST-FFN gradient flow, and complete task updates before a long run.

Native selection initially uses the internal `GLOBAL_ACTIVE_GRAD4_NATIVE` Plan
and explicit `RADIUS_FLAT_TORUS_FUSED` coordinate-update Plan with public
CSTLinear/CSTAdamW; the default radius update reference is CPU-only.
This is recorded as an experimental dependency,
not a new public dispatcher default. Its current 1024-site limit blocks N512's
2048-wide projection; extend and independently qualify an explicit route before
that cohort, without interpreting an unsupported shape as a capacity failure.
Bounded aspect checks include 1:1, 1:4/4:1 and 1:16/16:1. Fixed-ratio width
experiments alone cannot select geometric mean over other homogeneous budgets.

## Learning and evidence plan

Use TinyStories with a frozen dataset revision, tokenizer identity and token-file
hashes. Split by stories before packing tokens; keep training and validation
separate. Fix next-token shifting, evaluation windows and prompts before runs.
Initial data is pinned to revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`,
the original train/valid text files and tiktoken0.12.0 GPT-2 encoding. Retain
20,000,001 training tokens,8,193 development tokens and32,769 confirmation tokens,
with exact-story duplicates excluded across splits before packing. Hash the
actual token files, encoding and consumed normalized-story prefixes; a prefix
digest is not a retrieved full upstream file digest. The prefix and story-ID
files stay external. Confirmation is evaluated at the end of each bounded run.

Initial training uses FP32, TF32off, microbatch2×context128 with accumulation4
(1,024 tokens/update), AdamW base LR3e-4, betas(.9,.95), eps1e-8, decay0,
global gradient clipping1 and warmup `min(200, updates//10)` (at least1), then
cosine decay to3e-5. CST uses the explicit fused coordinate-update Plan.
Keep identical data-order streams across arms. Round token budgets down to
whole updates. Development evaluation runs at step0, every200 updates and the
last step; activation/gradient probes use separate backwards at initialization,
steps10/100 and the end. Generate from the two prompts encoded in the runner,
64 extra tokens, temperature0.8/top-k40, seeds7000/7001.

Start with repeated small batches and a one-seed one-million-token pilot. After
validating behavior and cost, use at most20million training tokens per run and
three seeds on working configurations. An enlarged budget or changed recipe
must have a separately recorded revision, not silently extend a failed run.

Record training/validation CE and perplexity against steps, seen tokens and
elapsed time; fixed-prompt generations; layer activation/gradient diagnostics;
FFN and total parameter counts; and synchronized complete-step speed with
allocated/reserved GPU peaks. Exclude diagnostic probes from timed samples and
report compilation/setup separately. Cross-width measurements require the same
batch/context/measurement scope. A working result requires stable learning and
heldout improvement, useful FFN contribution relative to the no-FFN arm, and
repeatable behavior. Dense superiority is not required; failed or inconclusive
arms remain visible. Do not claim formal noninferiority or a general scaling law.

Save raw logs, states and source/data hashes in ignored or external evidence.
Preserve failures, analyze whether initialization, gradients, geometry or code
caused them, and requalify fixes before expanding. Runtime/backend fixes belong
in torchcst; models, data and study tooling belong here. Attention projection
replacement is a later study.

## Entry points

This study requires the merged initialization API at torchcst commit
`9cb6de727ddbf5c2178fe18db6d580c9e181b926`. The repository's historical
`[torchcst]` extra still pins the old HIGGS dependency; do not use that older
source for this study. Install the explicit study dependency in an isolated
environment, plus a matching CUDA build toolchain, ninja and tiktoken0.12.0:

```bash
python -m pip install 'torchcst @ git+https://github.com/takumiecd/torchcst.git@9cb6de727ddbf5c2178fe18db6d580c9e181b926'
PYTHONPATH=src python -m cst_research.experiments.ffn_qualification --cohort small --output evidence/qualification-small
PYTHONPATH=src python -m cst_research.experiments.ffn_qualification --cohort target --output evidence/qualification-target
PYTHONPATH=src python -m cst_research.experiments.ffn_data --output evidence/tokens
```

Inspect the actual receipts and token manifest, freeze their SHA256s, then pass
them explicitly to `ffn_run --qualification ... --qualification-sha256 ...
--data ... --data-manifest-sha256 ... --kind cst --width 256 --seed 17
--tokens 1000000 --output evidence/pilot-cst`. The pilot uses a one-hour wall
budget by default, preserves incomplete states on failure and does not resume
or retry itself. No training result is admitted from metadata validation alone.
