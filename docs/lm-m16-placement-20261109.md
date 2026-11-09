# Connected M16: matched-count FFN placement

**English** | [日本語](lm-m16-placement-20261109.ja.md)

Prespecified before execution. The user requested placement analysis, not a new
optimizer, kernel, normalization or learning-rate improvement experiment.

Eight primary configurations contain four CST maps, two up and two down, with
identical active FFN scalar counts: paired blocks 01,02,03,12,13,23; and one map
per block, up/down/up/down or down/up/down/up. Block indices are zero based.
`pair12` reproduces the previously successful central-pair layout using new M16.
Fresh Dense, DST-pair12 (same four selected sites and active scalar budget), and
all-eight-CST are context controls. Dense/all-eight-CST do not belong to the
matched-count placement family. Other attention, normalization, embedding and
vocabulary maps remain Dense. No claim that this is an all-CST Transformer.

Use the frozen connected ordered M16 envelope implementation from torchcst
95f647f97749f61f5f5c15ac29932f908fb334ce (runtime tree unchanged from the previous
qualified trial). TinyStories existing story-disjoint GPT2 data manifest
5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01;
width512, four pre-norm blocks, eight heads, context128, expansion4,
A=8sqrt(d_in*d_out)=8192 per selected map. Same named seed17 initial rows, with
M16 receiving the legacy initializer's raw rows without RMS recalibration.
Common sites/ordinary Parameters and all retained Dense matrices must match
initial bytes. Coefficient, shape, depth and CST field LRs are fixed across placements.

Train every arm fresh for4882 updates/4,999,168 targets, with four microbatches
of2 contexts,1024 targets/update. Retain the20M-token schedule/topology horizon,
shuffle905, AdamW ordinary LR.0003, all CST fields32x, tau.003, fraction0,
radius[.5,50], amplitude cap32, betas(.9,.95), eps1e-8, decay0, clip1.
DST uses the existing row-balanced RigL-style dense-storage reference and previously
selected sparse LR.003; topology every250 updates skips Adam at those updates.
No new HPO. DEV8192 targets at every200 updates,256,976,2441,4882 and initial.
Report CE, perplexity and next-token top1; confirmation/test is untouched.
Single seed is exploratory; no equivalence, convergence or causal-compensation claim.

At0,976,2441,4882, probe the same four128-token TRAIN contexts at offsets
0,4096,8192,12288 with original labels and single-context unscaled/unclipped CE.
Capture input/output RMS and their cotangents for every FFN up/down, FFN branch,
norm2, attention output and block. Record Parameter/field gradient RMS. Raw field
gradients across different parameterizations are not equivalent functional steps.
Centered token-feature Gram eigenvalues describe stable rank, entropy rank and
leading energy share; these describe128 contextual rows, not full matrix rank.
Retain offset0 tensors for same-token cross-arm representation/cotangent cosines
and centered linear CKA. These measure similarity, not correctness or a desired
Dense teacher target. At that context also record FFN/residual RMS ratio, their
cosine and FFN/negative-task-cotangent cosine. This last quantity is a local
sensitivity, not evidence of an optimizer update's descent. Parameter, buffer and
RNG bytes must remain unchanged by each probe; serialized next-update equality
must hold with/without the observer. No probe modifies the live update rule.

Before training, rerun native envelope/coalesced/regression gates and independent
quadrature. Full8192-Atom FP64-factor Y/dX/raw-dP oracle on representative pair12
up/down at initial/final, with existing atol1e-4/rtol4e-3. Qualify all eleven arms
by exact serialization and next-update model/optimizer-state equality. Verify all
three saved milestone checkpoint hashes, FP32 finite optimizer moments/clocks,
final digest and reproduced final DEV CE. Freeze and hash sources/data/config,
keep checkpoints and tensors remote and retrieve verified compact evidence.
Synchronized full-step timing excludes input preparation, DEV, probes and saving;
measure peak allocated/reserved separately and exclude first10 updates. Observer
allocation can affect reserved-memory history. DST includes topology events;
its dense storage is not a sparse-kernel speed claim.

Run sequentially on the reserved RTX4090, never interfere with unrelated jobs.
The primary analysis asks whether layout changes task loss and where scale,
feature diversity and cotangent transport diverge. A good hybrid supports a
layout-dependent interaction; observation alone cannot establish that Dense
causally repairs CST or distinguish all optimization/capacity explanations.
No speculative improvements or additional sweeps are authorized by this protocol.
