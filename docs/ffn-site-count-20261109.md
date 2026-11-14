# Selective FFN composition diagnostics

English (canonical) | [日本語](ffn-site-count-20261109.ja.md)

The full8-map CST LM learned but remained above matched DST. Earlier single-map
HIGGS success does not isolate depth because the task and architecture changed.
Keep this TinyStories LM at4 blocks and change only which FFN maps are replaced.
This is a new exploratory study; preparation is not GPU execution or a result.

## Fixed comparison

Width512, heads8, context128, expansion4, same seed17/data/order, ordinary named
initialization, batch2 and accumulation4. Train fresh5M-token prefixes (4882steps,
4999168tokens), retaining the previous20M-token LR and topology horizons. No new
LR/tau/Atom search, no stopping/selection by DEV, no confirmation evaluation.

Block indices are zero based. At each site set run both CST and matched DST:

| Site set | Replaced maps | Count |
|---|---|---:|
| up | block1 up |1|
| down | block1 down |1|
| block | block1 up+down |2|
| two_blocks | blocks1+2 up+down |4|
| all | all4 blocks up+down |8|

Add a fresh Dense control (11 runs total). All8-map CST/DST controls are rerun
in this cohort. The old20M job finished training but its container-home artifacts
disappeared after server stop before verified local retrieval; do not borrow them. Keep attention, embedding, tied head and every
unreplaced map Dense; all are trainable. The factory starts from the Dense model
and replaces selected maps using their original named seeds. With all8 sites it
must reproduce the historical full models' state exactly. No public runtime change.

CST keeps A8192/map, envelope tau.003, fraction0, amplitude cap32, radius.5..50,
levels_wide_amplitude, moment initialization and all4 field LR32x. DST keeps
K32768/map, sparse LR.003, ordinary LR3e-4. Active total FFN scalar counts match
within each site-set pair; total compression changes across site sets. FP32,
TF32off, Adam betas(.9,.95), eps1e-8, WD0, clip1, warmup200,20M cosine as before.
DST has19 topology events through4750 and4863 actual Adam updates at4882steps.

## Mechanism evidence and gates

At initial,10,100,256,976, every200 and final steps, retain a fixed TRAIN probe:
input/output RMS and output-cotangent RMS at all4 blocks and8 projections, raw FFN
gradients, per-field gradients and amplitude-binned radius/fraction. Raw gradients
in different parameterizations are not directly equivalent optimization scales.
Measure actual one-step projection changes on the same7 fixed Gaussian inputs,
plus per-Atom field movement (center differences wrapped by chart period), so
function changes can be compared independently of coordinate encoding.
Record preclip norm, clocks and effective field LRs. DEV8192-target CE/perplexity
and teacher-forced top1 use the same fixed targets; TRAIN CE is logged separately.

Qualification must pass independent FP64 initial map derivatives, exact serialized
regular/topology updates and an instrumented-versus-uninstrumented update check.
CPU checks protect exact Dense complements, all8 historical parity, active scalar
budgets, RNG and read-only probes. Every final/prefix checkpoint (33 total) must
reproduce DEV CE/top1 and finite FP32 moments/counters; trained map oracles and
source/checkpoint/compact hashes must pass before scientific admission.

Timing and memory use synchronized uninstrumented training steps after10; fixed
probes are outside those samples. Keep setup and wall time separately. Historical
all8 timing is not a paired speed control. Save evidence on persistent NFS, retain large checkpoints there and periodically
retrieve/verify compact evidence locally. Freeze source before launch.
Evidence locator: CST_RESEARCH_EVIDENCE_ROOT/ffn-site-count-4090-20261109-01.

The hypotheses are signal/gradient attenuation, inappropriate realized update
scale, and restricted composition of structured maps. Their diagnostics guide
subsequent interventions; correlations alone do not prove one cause. Central
sites do not establish position-independent behavior, and a fixed-budget one-seed
5M pilot cannot prove representational impossibility or a universal depth law.
