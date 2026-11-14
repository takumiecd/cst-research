# Radius-envelope tau search in TinyStories FFN

English (canonical) | [日本語](ffn-radius-fraction-tau-20261109.ja.md)

Prospective exploratory protocol, fixed before looking at new learning results.
The public torchcst radius-envelope family learns amplitude, radius_fraction and
centers independently, with full gradients through the amplitude-dependent lower
bound: U=50, L=.5+49.5 exp(-(a/tau)^2), r=L^(1-f) U^f, f=radius_fraction.
The new explicit cached-map Plan preserves the old map and every task derivative;
its small speed pilot is separate from a quality result or public-default adoption.
Operator commit443ee84c is frozen for this study; the runtime lives in torchcst.

Use the same TinyStories GPT2 token files as the previous FFN pilot, manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
Verify the actual token bytes and story-disjoint declaration. Development8192
next-token targets only; do not inspect confirmation for tau selection.
Width512,4pre-norm blocks,8heads,context128,expansion4,8CST FFN maps,
A=8 sqrt(512*2048)=8192 per map. Attention, embedding and tied vocabulary head
remain Dense. FP32 Parameters/physical rows, FP64 map intermediates, TF32 off.
All arms share named ordinary weights, center streams and Gaussian amplitude
noise seed. Each tau uses the public moment initializer, matching the same
weight-second-moment target: upstd.02, downstd.02/sqrt(8). Actual amplitude and
radius distributions may differ; this is not identical realized physical weights.
Record initializer target/realized errors for every layer. Initial fraction0,
amplitude cap1, radius[.5,50]. No changes to source during a running snapshot.

Prespecified screening: tau .003,.01,.03,.1,.3, seed17,256updates/262144tokens
per arm, plus a fresh Dense control at the same stop point. Select the two lowest
finite final development CE values (tie: smaller tau). A failed/nonfinite arm is
ineligible and retained. Re-run the selected two from scratch for976updates/
999424tokens, seeds17,29,43; fresh Dense at the same long budget for all3 seeds.
This is longer follow-up, not convergence or noninferiority confirmation. Three
seeds and a development-selected family cannot establish a universal optimal tau.
The second stage runs only if two arms finish without correctness failures;
otherwise preserve the failure and diagnose before expanding the batch.

Batch2 x128 x accumulation4 =1024tokens/update. All runs retain the SAME20M-token
order/schedule horizon (200update warmup). AdamW baseLR3e-4, betas(.9,.95),
eps1e-8,decay0,globalclip1; amplitude/centers16x baseLR, radius_fraction1x
baseLR via public CSTAdamW field_lrs, decayed by the same schedule. No joint LR,
Atom count, depth or fraction-init sweep. Dense uses ordinary AdamW at baseLR.

Before training, run actual CUDA envelope/physical regression, full-model map
value/VJP parity for every tau, two outstanding forwards with separate saved
backwards, finite all-site steps and actual fraction movement. At tau.03,
serialize model/AdamW and verify exact next Parameters, moments and step counters.
Use a fresh source-bound admission receipt for the new consumer. Existing model
qualification receipts cannot admit changed model/config or this new family.

Record train/development CE against updates/tokens and elapsed time; synchronized
complete-update time and allocated/reserved peaks exclude evaluation/probes but
include accumulation, transfers, clipping and optimizer. Record preparation and
total wall time separately. After step0,1..5,every20 andfinal, record per-layer
radius, radius_fraction/endpoints, amplitude bins and latent field gradients
on a fixed probe. Gradient field1 is dLoss/df, not the physical partial dLoss/dr.
No Graph or GPU-process-wide peak claim. Plot all tau curves overlaid with Dense;
plot amplitude-conditioned radius and fraction trajectories, not only means.

Raw sources, logs, checkpoints and token data stay in ignored/external storage.
Register job/source/data/archive hashes and verify retrieved compact files before
claiming completion. Old Polar curves remain historical context with their exact
source; do not splice them into new paired speed comparisons.


The post-training read-only audit criteria were fixed before running that audit:
reproduce every long checkpoint's development CE within1e-6, verify checkpoint
hashes and all976 Adam step counters/finite FP32 moments, and compare trained
block0 up/down Y,dX and all Atom gradients with the encoded-row FP64 physical
reference at7 noncontiguous rows. Retain the existing atol1e-4/rtol4e-3 gate.
Check all eight maps' finite values and amplitude/fraction bounds. Loading may
invalidate placement generations; it must not alter numerical continuation.
The first qualification attempt incorrectly demanded equality of those management
generations. It stopped before learning, was preserved, and was replaced by a
new source-bound attempt; no numerical tolerance or learning recipe was relaxed.
The read-only audit script is separately hashed and never modifies frozen sources.
