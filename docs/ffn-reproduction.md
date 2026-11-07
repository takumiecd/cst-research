# Reproducing the selected FFN recipe

English (canonical) | [日本語](ffn-reproduction.ja.md)

This describes a fresh reproduction of the selected R8/R10/R11 recipe. It does
not resume the owner's running campaign or admit another GPU automatically.
The [study plan](ffn-feasibility.md) and [results](ffn-feasibility-results.md)
retain the exploratory history and unfinished confirmation status.

## Freeze both repositories and the runtime

The scientific learners use these immutable sources:

- cst-research: `70e02d864e661d3a87cd8aea322b49ae3f4a1f8f`.
- torchcst: `2b587e0f04ddf64e2bfb0c2e3db591e87a6f6440`.
- Observed runtime: Python3.12.9, NumPy2.0.2, Torch2.11.0+cu126, CUDA12.6, NVIDIA GeForce
  RTX4090,24GB class, CC8.9; TF32 disabled, FP32 training, four CPU threads.

In an isolated environment with that Torch/CUDA build, matching CUDA compiler
and headers, install the explicit operator and consumer sources:

```bash
python -m pip install 'torchcst[cuda] @ git+https://github.com/takumiecd/torchcst.git@2b587e0f04ddf64e2bfb0c2e3db591e87a6f6440'
python -m pip install 'cst-research @ git+https://github.com/takumiecd/cst-research.git@70e02d864e661d3a87cd8aea322b49ae3f4a1f8f'
python -m pip install 'tiktoken==0.12.0' 'numpy==2.0.2' ninja
```

Do not install this study through the historical `[torchcst]` consumer extra;
it pins the old HIGGS source. Keep the recorded compiler/header environment too:
package versions alone do not reproduce native extension compilation. The frozen
bootstrap scripts in the evidence locator retain those environment details.
Record actual GPU/runtime identity and qualify it before learning on another
machine. Never edit the source tree of an existing learner.

## Prepare and verify the data

```bash
python -m cst_research.experiments.ffn_data --output evidence/new-ffn-tokens
```

The TinyStories revision is `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, using
original train/valid text, GPT-2 encoding, whole-story deduplication before
packing and little-endian uint32 files. This preparation streams consumed
normalized prefixes; it does not claim a full upstream file download/digest.
Check the manifest AND actual token/story-ID file hashes before execution.

| Split | Saved tokens | Token SHA256 |
| --- | ---: | --- |
| Train | 20000001 | `9bfb0a333b8deb13d9f0aa9bcd64aeca4c7d512c9ac548acb8fdbe72bf3cc9ca` |
| Development | 8193 | `56656e2d90717be6919bc4d26f0b0095b3739c614ae4c071ab629142e4d0fd51` |
| Confirmation | 32769 | `2bcc516a29e7aa941edf633576720cf38da120266e094f6772b6f19e4fce17a8` |

The recorded manifest SHA256 is
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
The independent data audit verified all saved hashes, disjoint story sets and
replayed the first3 unique upstream training stories/548 tokens. That prefix
replay is not verification of the entire upstream corpus.

## Qualify one exact shape and recipe

This example reproduces width256/seed29. Use a fresh output root:

```bash
python -m cst_research.experiments.ffn_qualification \
  --cohort recipe --width 256 --expansion 4 --coefficient 8 \
  --q 0 --amplitude-max 1 --w-c 0.3 --radius-birth 4 \
  --feature-order contiguous --atom-lr-multiplier 16 \
  --output evidence/new-n256-seed29-qualification
```

Inspect `qualification.json`: both projection directions, actual target
standard deviations, Y/dX/all-Atom VJPs, quadrature, optimizer and model-resume
checks must pass. Compute and retain its SHA256; the next command deliberately
requires it. A qualification for another recipe/shape/count is insufficient.
N512 expansion4 and N128 expansion16 require the explicit2048 operator above;
its accepted Atom counts and eager FP32 boundaries are not broadened by this
recipe. The stored campaign additionally requires a two-update full-model
CST preflight before learning.

## Train all three arms at the same budget

Replace `QUALIFICATION_SHA256` with the verified digest, then run each kind
`dense`, `none`, `cst` separately, with a distinct fresh output directory:

```bash
python -m cst_research.experiments.ffn_run \
  --data evidence/new-ffn-tokens \
  --data-manifest-sha256 5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01 \
  --qualification evidence/new-n256-seed29-qualification/qualification.json \
  --qualification-sha256 QUALIFICATION_SHA256 \
  --output evidence/new-n256-seed29-cst --kind cst --seed 29 \
  --width 256 --expansion 4 --coefficient 8 \
  --q 0 --amplitude-max 1 --w-c 0.3 --radius-birth 4 \
  --feature-order contiguous --atom-lr-multiplier 16 \
  --batch 2 --accumulation 4 --tokens 5000000 \
  --schedule-tokens 20000000 --max-wall-seconds 7200
```

Before the5M arms, run the same arguments with `--tokens 2048
--repeat-fixed-batches` and a separate preflight output directory for two CST
updates. This fixed-batch preflight skips confirmation/generation and must not
enter heldout learning reports.

The model has four pre-LayerNorm blocks, four heads, context128, GELU and no
dropout. All eight FFN projections are CST in that arm; attention, embedding and
tied output remain Dense. `none` removes the FFN residual contribution. Initial
Dense std is0.02, with residual-output std0.02/sqrt(8); public variance-scaling
initialization matches the CST weight-energy target. There is no CST-only
runtime gain or normalization. Keep every scalar recipe flag identical across
arms. Atom-LR16 applies only in the CST optimizer to all four coordinates per
Atom; ordinary parameters keep base LR3e-4. Betas(.9,.95), eps1e-8, decay0,
global clipping1 and the scheduled warmup/cosine are fixed by the frozen runner.

Five million requested tokens round down to4882 updates/4,999,168 seen tokens.
The20M schedule also fixes the data permutation horizon; changing it changes the
early learning trajectory. Do not enable repeated fixed batches or per-update
motion sampling for this quality/resource family. Retain partial states and
failures without silently resuming, retrying or increasing budgets.

The actual runtime uses NumPy2.0.2/PCG64. Data order is generated by
`np.random.default_rng(seed + 888).permutation(156250)`, because the planned20M
tokens contain156250 context128 windows. SHA256 of the full permutation encoded
as little-endian int64 is recorded below. Each5M arm consumes its first39056
windows (eight per update), identically across the three FFN kinds at that seed.
These order-generation checks were reconstructed on the original host with the
learner's import paths; they do not constitute another training run.

| Seed | Full permutation SHA256 (int64 little-endian) |
| ---: | --- |
| 17 | `927ba5aad87c2f72f89d416c4ee33c7f7d8ec183e3419827258b1b868089a324` |
| 29 | `296658bd662fd307b5b83150976b9a39ede0f7462c738e9d05014325eeb9e43d` |
| 43 | `1c439d39af4529f43c63225ed3eaeeb58c49490fa273e01aac07adf91410cf41` |
| 71 | `7ea85e999c4a2e2d7fa6490e4969e8531a95a597418219b58bec50115537b688` |

The fixed confirmation grid is seeds29/43/71 × widths128/256/512, expansion4,
c8, all three arms,5M each:27 fresh learning runs. Run sequentially, with each
case's qualification/preflight and terminal audit before the next case. Aspect
checks N128 expansion1/16 are separate seed17,1M development pilots. Retain every
seed; do not tune the recipe from confirmation scores. This is a bounded budget,
not a promise of convergence. The fixed usefulness criterion and claim limits
are in the study plan.

## Audit, retrieve and report

Resolve `$CST_RESEARCH_EVIDENCE_ROOT/ffn-feasibility-20261106` for the archived
source manifests, exact launch/audit argv, raw summaries/curves, checkpoint
hashes and audit scripts. `source-r8.tar` SHA256 is
`d0ad97fbf6954f428b786f8cbd8ebbef23bb8310f18ccef1f4af7ffac139fdc3`;
all224 consumer source files match its manifest. The original host wrappers are
retained evidence, not a portable public cluster launcher; a new deployment root
requires a separately versioned wrapper, never editing a live frozen copy.

The recorded trained-checkpoint auditor
`audit-ffn-checkpoints-r8-v2.py` SHA256 is
`25096a1faad73023a3ef70362e188d7a4f31cbd64359fcfea51d0676914040bc`.
It checks saved hashes/counters, load semantics, independently recomputed CE and
all eight trained CST VJPs, then two updates only in disposable copies to verify
the serialized next task step/optimizer. It never resumes or overwrites an
original saved run. The no-FFN control intentionally has unused norm2 parameters;
the corrected auditor permits missing gradients/state only for that exact set.
Bulk checkpoints stay on the GPU host; retrieved metadata archives and every
bundled file must match their manifests before reporting.

Use the separate [receipt-linked reporting CLI](ffn-reporting.md) at consumer
`6eec101cf5092d319d280bd819ac4a97867e2004` or its unchanged reporter descendant,
in a separate reporting environment with the optional `[reporting]` extra.
Reporting does not migrate/revalidate the learner source. Its request must name
the actual learner/operator commits above, complete cohort and individual
receipt/audit hashes. Keep all curves/seeds, paired CE differences, both prompt
samples and complete-step allocated/reserved observations. Review every rendered
plot. Parameter compression is not a measured GPU-memory or speed advantage.
