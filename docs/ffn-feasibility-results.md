# FFN feasibility: initial observations

English (canonical) | [日本語](ffn-feasibility-results.ja.md)

These are development-seed17 observations, not three-seed confirmations or a
scaling-law result. The [study plan](ffn-feasibility.md) remains in progress.
All eight FFN projections in a four-block width256 causal Transformer use
CST with4096 Atoms each, `A=ceil(8 sqrt(d_in d_out))`; attention, embeddings
and tied vocabulary output stay Dense. FFNs are256→1024→256, context128.

## Initial three-arm pilot (R2)

Each arm trained for976 updates/999424 tokens. Data order and cosine LR used
that short horizon. Dense target std is.02, residual-output std is.02/sqrt(8),
and CST uses the public second-moment initializer with q0. Parameters shared
between arms have identical initialization streams. FP32, TF32off, RTX4090,
Torch2.11.0+cu126,1024 tokens per complete update.

| FFN | Development CE | Perplexity | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| Dense | 3.72727 | 41.57 | 482.50 |
| CST | 4.17978 | 65.35 | 687.28 |
| None | 4.23073 | 68.77 | 440.39 |

![Initial pilot learning curves](assets/ffn-feasibility-20261107/pilot-r2-learning-curves.png)

CST learns, but remains much closer to no-FFN than Dense at this budget. Its
FFN logical parameter count is6.48% of Dense; its measured GPU peak is larger.
This run does not demonstrate a practical speed or memory advantage. Native
qualification and a separate trained-checkpoint audit checked Y, dX and all
Atom-coordinate VJPs, finite optimizer moments/counters and recomputed heldout
CE. Those checks support numerical correctness within their recorded scope;
they do not explain the learning gap.

Initial FFN activation RMS was close to Dense. CST's down-projection RMS stayed
near.04–.05 while Dense grew to.12–.17. A later q0 trajectory with a20M-token
planned horizon was stopped by the owner. Its last periodic checkpoint contains
2800 updates/2867200 tokens; do not reconstruct the unsaved tail or compare its
CE directly with the short-horizon Dense result.

## Radius family (R5)

Fresh q0/.25/.5 arms each trained512 updates/524288 tokens with the same planned
20M-token LR and data permutation. Moment matching was repeated per q. Changing
q changes amplitude, spatial correlation and clipping as well as physical
radius; this is not an isolated radius-only intervention.

| Initial q | Development CE | Median complete update ms | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| 0 | 4.40716 | 419.35 | 687.28 |
| .25 | 4.39447 | 396.89 | 687.28 |
| .5 | 4.41436 | 222.69 | 687.28 |

![Radius-family learning curves](assets/ffn-feasibility-20261107/radius-r5-learning-curves.png)

The loss curves nearly overlap. These early results do not show a large benefit
from broadening. Times are observed synchronized updates from one sequential
job, ordered q.5/0/.25, without repeated order controls. A changed mathematical
workload is not a kernel-only speedup; do not extrapolate these observations.

In a separate seed17 `make_cst` CPU fixture, average up/down radii were2.49/3.21
(q0),4.35/5.58 (q.25) and7.14/9.71 (q.5). Actual model layers use named seeds and
retain their individual initialization reports. The q1 up fixture could not
reach the target second moment with amplitude bound1; no q1 learning run was
admitted. Broadening can encounter the current amplitude/radius coupling.

Matched Atom IDs in the stopped q0 checkpoint show amplitude RMS.2280→.2359
(up) and.1171→.1237 (down). Strongest-amplitude10% have average radii.914/1.637,
while weakest10% have3.967/3.992. q remains near zero. Broad weak Atoms coexist
with narrow strong Atoms; an overall mean radius obscures this distinction.
Endpoint comparisons cannot establish cumulative amplitude oscillation.

## Native scope and remaining work

R5 qualification passed both FFN directions at B2 T128, q0/.25/.5, Y/dX/all Atom
VJPs, polynomial quadrature and optimizer/model resume checks. Separately, the
scoped2048 route passed all12 expanded R3 native cases on RTX4090, including
512↔2048 and128↔2048. This does not establish full language-model feasibility
at width512, or qualify a different consumer/q recipe automatically.

Fresh5M-token Dense/no-FFN/CST q0/.25/.5 observations retain a20M planned horizon.
CST diagnostics sample amplitude after each update to separate cumulative
motion from endpoint change, and retain radius/q histories. They are not primary
timing/memory runs. Width128/256/512, bounded aspect-ratio learning, three-seed
confirmation and final performance/memory validation remain incomplete.

## Additional CPU mechanism checks

These are synthetic/calibration observations, not language-model results or
changes to the default recipe. For Polar norm Q=s²+t², public q=(Q−1)/3, tangent
proposal δu and step size h, the default activity clock uses

$$
Q_* = \operatorname{clip}_{[1,4]}(Q+\gamma\|\delta u\|^2),\qquad
Q' = \left[1-\frac{Q_*-1}{Q_*}e^{-4\lambda h}\right]^{-1}.
$$

With fixed tangent velocity v, δu=h v, its activity increment is proportional
to h². The existing `time_energy` option divides that increment by h. A 1000-step
CPU fixture at h=.0003, ||v||=.15, gamma1 and lambda.1 produced q≈6.36e-7 in
FP64 with `finite_chord`, matching an independent scalar recurrence; FP32 gave
q≈2.65e-9. `time_energy` gave q≈.00212. Thus a near-zero q can coexist with
small repeated angular updates, and FP32 rounding can further reduce activity.
This does not establish the actual Adam tangent velocity or the learning-gap
cause; no clock mode was changed in the R6 learning runs.
These local CPU checks used Torch2.13.0/Python3.11.15 against the same operator
source; they do not validate the remote CUDA updater's FP32 rounding behavior.

A separate bounded CPU initialization grid varied amplitude bound and w_c
together. With bound4, w_c1.2 and q1, the seed17 up/down fixtures matched the
same Dense targets with average radii31.53/40.54 and350/10 clipped Atoms out of
4096. These much wider settings also change coordinate conditioning and require
new native qualification before learning. They are prepared alternatives, not
successful models. Raw scripts/results are `polar-activity-clock-diagnostic*`
and `broad-radius-initialization-grid*` under the evidence locator.

A separate initial-state Gaussian-stimulus check used512 independent normalized
inputs per block. Summing the four FFN output covariance matrices gave
participation-ratio statistics184.13 (Dense),126.63 (CST q0) and62.09 (CST q.5);
mean adjacent-channel correlations were−.003,.419 and.839. Participation ratio
is a covariance statistic, not the algebraic rank or a task-accuracy measure.
Independent output-channel permutations changed the combined CST covariance
statistics to144.15/100.58; this was a covariance-only counterfactual, not a
trained-model intervention. Larger radius can increase feature correlation,
so matching global RMS does not establish Dense-like feature diversity. Narrower
birth-radius settings and fixed feature orderings remain hypotheses to test,
alongside broadening and Atom-budget checks. No remedy is established by this
fixture. Retain `ffn-feature-covariance-initial*` in the external evidence.

## Provenance and retained evidence

TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, tiktoken0.12.0
GPT-2 encoding, whole-story deduplicated splits. Token manifest SHA256:
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.

- Operator for R2/R5: `9cb6de727ddbf5c2178fe18db6d580c9e181b926`.
- R2 consumer: `f75c1d2efe54767f6b9f65804c234924e6f6e191`.
- R5 consumer: `09e15b759fd796ad02c436e3580f1a223e098a92`.
- R5 native receipt SHA256: `4268f0d8ee42f4febd036e1f9846ecdd565231edaca330ef33fd5f4ddb49222b`.
- R5 terminal manifest SHA256: `17d42c7f9722373da24ab23ab3c4e3244e700ab38cb84791f4a64d88309e2b40`.
- Expanded operator: `2b587e0f04ddf64e2bfb0c2e3db591e87a6f6440`;
  consumer `036d0c5beb4a6f675ce1dbeafd4224e6c270f342`.
- Expanded native receipt SHA256: `419ab7750182bca2302d7152c6fc01c1a961efe69f19bf5bb22355f3aa299ce6`.

Resolve `$CST_RESEARCH_EVIDENCE_ROOT/ffn-feasibility-20261106` for raw receipts,
per-arm summary/curves, independent audit results and plotting scripts. R5 raw
JSON/curve hashes were verified against the terminal manifest; trained R5
checkpoint audit remains separate. Bulk checkpoints stay on the GPU host with
their hashes. PNG and SVG here are derived figures, not raw evidence.
