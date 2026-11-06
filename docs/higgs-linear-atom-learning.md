# Linear Atom-count pilot: Stage B learning proposal

English (canonical) | [日本語](higgs-linear-atom-learning.ja.md)

**CPU preparation only; not frozen, executed or admitted.** This is the second
implementation delivery of the same registered linear-Atom study. The immutable
[Stage A protocol](higgs-linear-atom-scaling.md), all114 Stage A files, its registry
and its historical qualification-only entrypoint remain unchanged. The separate
Stage B entrypoint is not usable until actual native16 qualification, full-file
release and independent scientific CPU admission have passed and the owner has
sealed their exact hashes and the new source/runtime/storage/launch.

## Updated research objective: identify the order

The user's latest objective is to empirically identify the required Atom count
A*(N,L), conditional on an explicit task, training recipe, budget and quality
criterion, **testing rather than assuming A∝N**. Showing that a chosen cN budget
works would provide an upper-bound example, not determine the minimum or its
scaling order. This unchanged sixteen-case one-epoch pilot is only a first screen.

Future separately reviewed studies should distinguish a learnability frontier
from a tuned Dense-quality frontier at fixed depth, task, horizon and margins.
Bracket each frontier at each width/depth, then examine widths1024/2048/4096/8192
as feasible, with appropriate independent LR tuning and paired seeds. Inspect
A/N and an effective exponent, including constant and sublinear alternatives:

\[
p_{\mathrm{eff}}=\frac{\log(A^*(N_2,L)/A^*(N_1,L))}{\log(N_2/N_1)}.
\]

Finite uncertain brackets, few widths, short horizons and training failures cannot
establish an asymptotic law. Separate initialization-scale/conditioning stalls
from density/capacity limits; a failed training recipe is not an intrinsic lower
bound on A*. No extra widths, Atom budgets, normalization, updates or automatic
followup are authorized by this wording change. Subsequent ranges and protocols
must be specified separately after this pilot's actual evidence is admitted.

## Fixed scope

Execute all16 fixed cases, in unchanged Stage A order: N1024/2048, hidden H4/H8,
Dense-stock and CST c1/4/16 with exact A=cN at each of H−1=3/7 internal squares.
Input/output Dense maps remain trainable. Normalization NONE, gains1, no
calibration, explicit legacy initialization, q1, LR0.0003, B4, FP32/TF32 off/eager
all remain unchanged. No LN/gain fallback, density/LR sweep or initialization repair.

Each case starts fresh0, then one full100k-training epoch with25000 true B4
updates; total400000 task updates. Epoch1 order is PCG-independent PyTorch
`randperm(100000)` with CPU generator seed117, retained as actual tensor bytes.
There are no early stops based on quality or automatically extended cases.
The four Dense controls measure H8 adequacy under this same recipe; a failed
Dense reference cannot establish CST parity. This is an exploratory screen, not
formal noninferiority, a minimum sufficient c, or measured linear cost.

## Inherited admission and new initial-state gate

Stage A producer is `446f25f1ec339f1e3acb4cf84d96ac5825306ab7`, source manifest
`717bcbcb88dde83fe619cad547bd9ffbc9092e35be5787fdcad7f1eda2859db4`, protocol
`9a34b7fc5a30d58b47c36db25ed15b148dae7c347fbd4898beb0fda8e7bc7b28`.
Stage B never relabels that producer as its own numerical qualification.

Before any task-array decoding, verify the owner-sealed independent report and
root file-release hashes, actual normal-exit0/completed outer receipt, the full
raw inventory and all bytes in place, exact16 case/qualification/initial0 SHA
bindings, all114 old source files, runtime and data. Preparation or partial
qualification reports fail closed. The report status must be
`passed-independent-StageA16-qualification-only-NOT-task-training`.
No report or native pass is fabricated during preparation.

For every task case, construct a new model and optimizer and compare the complete
initial core literally with actual qualified initial0. Require epoch0/updates0,
empty optimizer history and all gradients None. No saved state is loaded into
the main model, and qualification state8 cannot be a training starting point.
Its initial0 remains a SHA-bound producer reference to avoid a duplicate checkpoint.
The inherited16×8 normal and16×4 resumed synthetic qualification updates are192
qualification updates, not task training. Stage B adds no qualification/replay updates.

## Evidence, selected state and measurements

Retain epoch1 full model/optimizer/gradients/moments/clocks, exact order, full50k
DEV logits/labels/raw row IDs and the selected checkpoint SHA reference. Selection
is trivial: fixed epoch1 is the only candidate. Also evaluate the **same selected
state** on all100k TRAIN and50k DEV outside the training clock. Two independent
native full-state restores must yield identical TRAIN array bytes; restored DEV
array bytes and metrics must equal cached epoch1. Retain selected TRAIN arrays,
DEV SHA reference and load proofs. Only actual3/7 owners receive the native
`max(before+1,serialized)` generation rule; all other state remains literal,
and evaluation mutation checks use actual post-load baselines.

Reuse the unchanged bare `ordinary_step` epoch loop without per-step observation
copies. Reset memory peaks immediately before the synchronized training interval,
capture them before checkpoint/evaluation, and label resident arrays/model/optimizer/
native buffers and persistent allocator reservations. Report preparation, full raw
admission, diagnostics, snapshot, prediction/checkpoint I/O, restore and evaluation
separately; inclusive totals/components are nonadditive. Fixed256 TRAIN-feature
initial/end activation RMS and ReLU retention plus last actual B4 gradients and
accepted m/v/clocks are diagnostics outside the clock, not extra backward probes.
No endpoint difference is called a single-update displacement.

## Source, data and execution boundary

Public975 source and source/runtime compilation path stay identical to Stage A.
The RTX4090 runtime pin remains
`1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b`.
The train/development artifact remains
`58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`
and manifest `530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`.
No original/future confirmation or official test payload is staged. Development
already informed previous recipe choices and is exploratory, not fresh confirmation.

Exact10 staged inputs are the original eight (with the new consumer manifest)
plus `stage-a-admission.json` and `stage-a-file-release.json`. The actual Stage A
raw root is read only in place; no multi-GiB evidence copy is made.

```bash
python -B -m cst_research.experiments.higgs_linear_atom_learning_driver \
  --consumer-commit NEW_REVIEWED_STAGE_B_COMMIT \
  --stage-a-root /path/to/actual-readonly-StageA-attempt \
  --stage-a-admission-sha256 OWNER_SEALED_ACTUAL_REPORT_SHA \
  --stage-a-file-release-sha256 OWNER_SEALED_ACTUAL_RELEASE_SHA \
  --cache-root /path/to/same-verified-linear-cache \
  --output /path/to/fresh-learning-output --timeout 7200
```

This command is a proposal, not authorization or a ready execution bundle.
The combined Stage A qualified raw plus Stage B new raw planning cap is8GiB.
Before launch actual local free space minus conservative compressed total,
maximum owned scratch and primary-study reserve must leave at least8GiB.
Preserve all compressed raw locally and immutable remote files; sequential
case-owned temporary CPU admission avoids a whole persistent decompressed copy.
The owner checks actual runtime/profile/storage and seals the finite launch.
Foreground timeout/failure retains evidence and reaps owned processes; no retry,
automatic resume, additional study, or evidence deletion is authorized.
