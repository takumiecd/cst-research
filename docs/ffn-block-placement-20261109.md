# Distributed CST maps and within-FFN composition

English (canonical) | [日本語](ffn-block-placement-20261109.ja.md)

The owner rejected the proposed Dense-teacher/weight-reconstruction detour.
The objective is to solve the original language-model task using its real token
labels. Dense remains a comparison control. Earlier HIGGS experiments also showed
single-versus-multiple replacement differences, but their cause is not assumed
to be identical after changing task, normalization and initialization.

Keep the same4-block width512 TinyStories LM, initial ordinary weights, seed17,
5M-token prefix/20M LR and topology horizons, CST per-mapA8192 and LR32x,
tau.003/fraction0/cap32, matched DST K32768 and sparse LR.003. No new HPO,
confirmation, RMS calibration or additional layers. Replacing a map keeps shape.

The earlier immutable site-count cohort already covers one up map, one down map,
both maps in one/two blocks, and both maps in all4 blocks. Add six fresh arms:

| Placement (zero-based) | FFN layout | Maps | Methods |
|---|---|---:|---|
| scattered_up | every block:CST→GELU→Dense |4|CST/DST|
| scattered_down | every block:Dense→GELU→CST |4|CST/DST|
| scattered_mixed | blocks0/2:up, blocks1/3:down |4|CST/DST|

Compare single maps against one map in each block; compare those distributed
maps against both FFN maps in every block. Also compare scattered_mixed to
both maps in blocks1/2: both have4 replaced maps,2up+2down and identical active
scalar counts, allowing an equal-count distribution-versus-pairing diagnostic.
CST/DST have equal active FFN scalar budgets at each placement. Total compression
changes between4 and8 replacements and is not silently treated as constant.

Train on task labels from fresh named initialization. Retain TRAIN/DEV CE,
perplexity, DEV token top1, fixed-TRAIN activation/cotangent/field-gradient traces,
preclip norms, realized random-action updates, amplitude/radius/fraction and
effective field rates. Initial/serialized/oracle/instrumentation gates and all
18 checkpoint audits follow the site-count runner. Reuse the68 operator gate
only after exact operator bytes/runtime/GPU/log parity. Queue after the previous
cohort has finished GPU work; do not change a live frozen source.

Source core/model/data/runtime/order/seed/budget must match the prior verified
controls. The only math-source edits are an appended site catalog and explicit
qualification-family selection; existing arm initialization is protected by tests.
Preserve both source archives and mark this as an owner-directed new study rather
than adding arms to the old frozen family. NFS evidence and immutable progressive
compact archives are backed up and verified locally.

Analyze task performance against the same train-token budget, then align signal,
cotangent and realized update traces between paired and distributed cases. A
small single-map gap with a larger paired-map gap motivates composition analysis;
it does not itself prove vanishing gradients or a kernel bug. The mixed comparison
matches direction counts but still changes positions. Position counterbalancing,
LR/gain interventions and extra seeds require a separately recorded next step.
Evidence:CST_RESEARCH_EVIDENCE_ROOT/ffn-block-placement-4090-20261109-01.
