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
   checkpoint must reproduce the next controlled update exactly. Validate the
   declared AtomState cache-generation advance on load separately; all payload
   tensors, row IDs and other metadata remain exact. A full small causal model
   also checks serialized state and the next CE update within FP32 tolerances.
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

The prepared extension is the explicit
`GLOBAL_ACTIVE_GRAD4_NATIVE_RECTANGULAR_2048` route at torchcst commit
`2b587e0f04ddf64e2bfb0c2e3db591e87a6f6440` ([PR150](https://github.com/takumiecd/torchcst/pull/150)).
It preserves old recipe fields/Plan guards and the existing native arithmetic.
Qualify all12 expanded fixtures, including N512 and N128 expansion16, before
using that source for learning. The initial R2 width256 pilots retain their
original research/operator snapshots; they are not silently migrated.

Beyond the main expansion4 width cohorts, run bounded one-million-token,
development-seed17 pilots at N128 expansion1 and16 with the same c=8. These
check aspect-ratio applicability, not an optimal formula or minimum coefficient.
The owner subsequently restricted execution to RTX4090. A100 small qualification
and CPU token preparation ended; no further A100 work is part of this campaign.
Keep any retained device-specific preparation separate from RTX4090 results.

## Current mechanism followup

The initial R2 one-million-token, seed17 pilot completed. Its development CE was
Dense3.7273, CST4.1798, no-FFN4.2307. CST's initial FFN activation RMS was close
to Dense, but its down-projection RMS stayed near0.04–0.05 while Dense grew to
0.12–0.17. This does not establish whether capacity, optimization or training
duration caused the gap. Its976-update cosine schedule already decayed to3e-5,
so it cannot isolate a duration effect from a longer schedule.

The initial followup plan proposed comparing N256 coefficients c8/16/32
(4096/8192/16384 Atoms per projection) under the same q0, target variance, base
LR, data and seen-token budget. Subsequent owner discussion prioritized the
bounded R7 radius/order/Atom-LR recipe family below before increasing Atom count.
The coefficient alternatives remain available if that family is insufficient.
Reinitialize each Atom budget to the same Dense moment target and report actual
initial activations. This is bounded practical recipe selection, not renewed
lower-bound/Big-Omega research.

Use a common planned20million-token learning-rate AND data-order horizon, even
for an early checkpoint. `--schedule-tokens 20000000` controls warmup/cosine and
the full data permutation; `--tokens` controls the bounded execution length.
Thus1M/5M/10M/20M observations of that design have matching early LR/data streams.
Retain the old R2 short-horizon pilot as a separate observation. Record current
c8's longer trajectory before concluding Atom shortage; compare Atom budgets
at matched checkpoints. An Atom-budget intervention also changes coordinate
geometry and per-Atom amplitude, so improvement alone does not prove a pure
expressivity limit.

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
The fixed-batch diagnostic uses100updates/102,400 seen tokens with the same
eight sequence windows repeated, seed17, at N256 in each arm. It skips
confirmation and generation, and is not a heldout-quality result.

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

For the selected R8/R10/R11 recipe, use the [frozen-source reproduction
guide](ffn-reproduction.md). The commands below describe the initial R2 entry
points; selected2048 cohorts require the later explicit operator in that guide.

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


## Radius diagnosis after the initial pilot

The owner stopped the c8 long run to prioritize radius coverage. A radius
comparison uses width256, c8 (4096 Atoms per projection), seed17 and q=0/0.25/0.5.
Keep learning rate, data order and a common20M-token planned horizon identical;
first observe512 updates (524288 tokens) per arm, with a1200-second wall cap.
These are exploratory comparisons, not confirmations. Recompute moment-matched
initialization for each q using the same Dense target variance; increasing q
can also change amplitude and spatial correlation. It is not an isolated
radius-only intervention. q is the initial logarithmic fraction between
amplitude-dependent bounds, not a physical radius or a value frozen in training.
Retain actual radius/amplitude distributions and clipping diagnostics. Use the
new `radius` qualification cohort (both FFN directions, full B2 T128 and all
Atom VJPs at each q) before `ffn_run --q 0.25` or `--q 0.5`. The declared maximum
radius remains50; actual radii can be substantially smaller. No result is
implied by preparing these controls. Atom-count and LR interventions are
deferred while this radius hypothesis is tested.

CPU preparation found that q=1 cannot meet the up-projection variance with
4096 Atoms and the current amplitude bound1; this is an initializer rejection,
not a failed training run. q=.5 passed the second-moment search but clipped
277/4096 up Atoms and3/4096 down Atoms in the seed17 preparation fixture. q=.25
clipped14/4096 up Atoms and no down Atoms. These fixtures use `make_cst` seed17;
actual model projections use their named seeds and require separate reports.
Their average radii were up/down2.49/3.21 (q0),4.35/5.58 (q.25),7.14/9.71
(q.5). This constrains broadening under the current amplitude/radius coupling;
it does not establish that Polar or the backend is incorrect.

The stopped-long c8 periodic checkpoint retains2800 updates/2867200 tokens,
SHA256 `9e82dd7985d773294c59cfac0f34412a1fe31648fd55a8ed1a5f6ee6a0be6745`.
Matched Atom IDs show aggregate up amplitude RMS.2280→.2359 and down.1171→.1237.
The final strongest-amplitude10% have average radii.914/1.637; weakest10% are
3.967/3.992. q stays near zero. Wide weak Atoms coexist with narrow strong Atoms.
Endpoints do not measure cumulative amplitude oscillation. The last periodic
checkpoint precedes the SIGTERM stop; unsaved updates are not reconstructed.

## Matched-duration mechanism followup

After the radius pilot, run fresh width256/seed17 Dense, no-FFN and CST q0/q.25/q.5
observations capped at5M tokens per arm, retaining the20M-token planned LR and
data-order horizon. Keep c8, data, initialization targets and optimizer settings
fixed. Include all three CST radii even if early losses are close; this checks
whether a radius effect emerges later. This is a separately recorded followup,
not a silent continuation of R5 or the stopped R2 run. Do not select a scaling
recipe from half a million tokens.

For these CST diagnosis runs, `--track-atom-motion` samples physical amplitude
after every optimizer update, recording accumulated absolute motion separately
from net endpoint change. CPU buffers retain the diagnostic state. Radius, q,
and near-amplitude-bound fractions are recorded at validation checkpoints.
Sampling does not modify Parameters or optimizer state. Synchronized update
timing excludes sampling; total elapsed time includes it. Instrumented runs
are not primary speed/memory comparisons: use separately recorded uninstrumented
passes for those claims. All original R2/R5 artifacts remain pinned.

## Bounded recipe followup (R7 preparation)

After R6 finishes and its evidence is audited, test the following width256,
c8, seed17 recipes. Every FFN projection stays CST; hard radius minimum.5,
maximum50, four blocks and the Dense moment targets stay fixed. These are
practical recipe checks, not an Atom lower-bound search.

| Recipe | q | Amplitude bound / w_c | Radius birth | Feature order | Atom LR multiplier |
| --- | ---: | --- | ---: | --- | ---: |
| Reference | 0 | 1 / .3 | 4 | Contiguous | 1 |
| Narrow birth | 0 | 1 / .3 | 1 | Contiguous | 1 |
| Calibrated wide | 1 | 4 / 1.2 | 4 | Contiguous | 1 |
| Fixed ordering | 0 | 1 / .3 | 4 | Permuted | 1 |
| Faster Atom updates | 0 | 1 / .3 | 4 | Contiguous | 16 |

`radius_birth` changes the weak-amplitude lower curve; it does not lower the
hard minimum. `permuted` stores fixed independent hidden/output channel orders
per block, without extra trainable parameters. The same ordering option applies
to Dense; include a permuted-Dense control. Dense can absorb these permutations
into its matrices, as checked algebraically on the CPU. Reordering a covariance
matrix alone was not a trained-model result.

`atom_lr_multiplier` applies to each complete Atom Parameter, including angle
and center coordinates; ordinary weights/biases retain the base LR. The actual
group LR also controls the Polar policy's outer time, so this is not an isolated
amplitude-LR intervention. Record optimizer-group rates, physical motion and
radius history. Do not infer equal functional updates from equal nominal LR.

First use one-million-token observations with a common20M-token LR/data horizon,
plus fresh Dense/no-FFN controls; use development CE for exploratory choices.
Before learning, `ffn_qualification --cohort recipe` must pass both256↔1024
directions at B2T128, all Atom VJPs, polynomial quadrature, and optimizer/model
resume for the requested q/bounds/order/LR recipe. Admission checks both actual
Atom counts and physical scalars, and the full-model recipe. Capacity cohorts
also check their requested coefficients in optimizer/model resume. The raw
polynomial oracle now reads declared per-axis scalar bounds rather than assuming
birth4/max8; an initial preparation failure exposed that old fixture assumption.
No runtime/backend mathematical contract was changed by correcting that oracle.

R7 remains preparation until these GPU gates and bounded task updates actually
pass. Keep R6's running source and outputs unchanged. New scalar settings,
feature ordering or LR multipliers require new qualification; older receipts
cannot silently admit them. Width/aspect and three-seed checks remain required
after selecting a working recipe.

## Qualifying a selected width and aspect ratio

The recipe cohort also accepts `--width 128|256|512`, `--expansion 1|4|16`
and `--coefficient 8|16|32`. It qualifies both actual FFN directions at B2T128,
using std.02 for up and .02/sqrt(8) for down, including expansion1 where the
two square projections have different initialization targets. Other cohorts
reject nondefault values of these flags rather than silently ignoring them.
Keep the selected q/bounds/order/Atom-LR flags identical in qualification and
learning. A receipt for one shape or Atom count cannot admit another.

For example, after selecting a recipe and using the qualified explicit2048
operator source, qualify width512 with
`ffn_qualification --cohort recipe --width 512 --expansion 4 --coefficient 8`
plus that recipe's flags. Follow with actual four-block task updates and a
checkpoint audit before a long run. These flags do not extend backend support;
unsupported combinations still fail qualification. Preparing this entry point
does not demonstrate width/aspect learning or validate a new Atom count.

## Selected development followup (R8)

After inspecting R7 development CE, select c8/q0/bound1/w_c.3/birth4,
contiguous feature order and Atom LR16 for a fresh width256/seed17 comparison
with Dense and no-FFN, capped at5M tokens per arm. This is a post-R7 selection,
not a recipe preregistered before those results or a public-default decision.
Keep the20M-token planned LR/data horizon and ordinary-parameter LR unchanged.
Disable per-update motion sampling; retain separate activation/gradient probes,
complete-update timing and allocated/reserved peaks.

The frozen consumer is `70e02d864e661d3a87cd8aea322b49ae3f4a1f8f`, with operator
`2b587e0f04ddf64e2bfb0c2e3db591e87a6f6440`. Both actual directions and the full
four-block preflight passed before this followup began. Retrieve and audit the
terminal evidence before adopting a recipe or making three-seed claims. Main
width128/256/512, bounded aspect-ratio learning and seeds29/43/71 remain required
on working configurations. Keep c fixed when checking width behavior; do not
restart an Atom lower-bound search.

Use the [receipt-linked reporting entry point](ffn-reporting.md) to aggregate
completed, audited cohorts and overlay all learning curves. Report preparation
and synthetic metadata tests do not constitute model-quality evidence.

## Main-width confirmation family

After the selected5M and bounded shape pilots finish their terminal audits,
confirm the same c8/q0/bound1/w_c.3/birth4, contiguous-order, Atom-LR16 recipe at
width128/256/512, expansion4, seeds29/43/71. Every width/seed has fresh Dense,
no-FFN and CST arms capped at5M tokens (4882 updates/4,999,168 actual tokens),
with the common planned20M-token LR/data horizon and no motion instrumentation:
27 learning runs in total. Keep the frozen R8 consumer70e02d8/operator2b587
sources. This family and budget are fixed after development, before confirmation.

Retain every seed and control. Report paired CE differences and per-seed
perplexities, all learning curves, complete-step time and allocated/reserved
memory. A useful-FFN confirmation claim requires CST confirmation CE below
no-FFN at every main width in all three seeds; report effect sizes without a
formal Dense noninferiority or universal scaling-law claim. Do not exclude a
seed or tune the recipe from confirmation scores. Preserve failures and inspect
numerical failures before further execution; do not silently retry or resume.
Bounded aspect pilots remain one-seed development checks, not three-seed claims.

The long family uses standard `nohup` in an independent POSIX session, recording
PID/start ticks, process group, exact argv, source hashes and per-case receipts.
An observer/VPN timeout is not completion or permission to restart a job. Poll
the recorded live process and journal; keep one scientific GPU child at a time
on the approved RTX4090. Leave the user-owned Hub running after owned workers
finish.

### Owner-directed stop and short field-LR exploration

The owner subsequently cancelled the six unstarted seed43/71 cases to prioritize
short mechanism experiments before committing to a longer confirmation or public
API. N128/N256 seed29 completed and were audited/retrieved; the in-flight N512
seed29 learner is allowed to finish unchanged and receive the same terminal audit.
The original nine-case request, source snapshots and raw supervisor journal are
preserved. Only that supervisor was terminated; a separate finish-only worker
waits for the current learner, then audits/bundles it without launching more
training. The original journal's last `running` value is stale after this stop;
the retained owner-stop record controls its disposition.

This is a stopped, incomplete three-seed family, not a numerical failure or a
successful three-seed result. Keep all seed29 evidence and the unexecuted planned
cases visible. The owner redirected the work for time and API/mechanism priorities;
do not select or exclude confirmation outcomes as if the planned family finished.

Next, prepare a separate N256/seed17 one-million-token development comparison:
all Atom directions at16× versus the amplitude direction at1× and both centers
at16×, keeping Polar outer time and the20M LR/data horizon identical. First
qualify the research adapter against stock behavior and independent optimizer/
Polar references, moments/counters, serialization and native task steps. The
physical amplitude is derived from the Polar pair; multiplying gradients alone
is not the intended LR intervention. Activity still depends on accepted motion
even with the same outer clock. Public `field_lrs` remains an unimplemented API
proposal until the mechanism check; direct per-field rates are the owner's
preferred input shape, with scheduler/checkpoint semantics still to be designed.

## Final execution disposition

The owner amendments control the campaign: all seed29 widths finished5M and
were audited, while the six unstarted seed43/71 cases stay stopped and the
formal three-seed family remains incomplete. Separate short field-rate and
whole-Atom32/64 comparisons completed and were audited. Public field_lrs also
passed native/full-LM gates and merged through torchcst PR151.
The [campaign disposition](ffn-campaign-disposition-20261107.md) maps the full
requirements to current evidence and limits. Earlier preparation/future-tense
sections retain their design history; they do not authorize cancelled jobs or
imply that an unperformed capacity/Attention experiment is complete.
