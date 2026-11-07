# FFN campaign disposition

English (canonical) | [日本語](ffn-campaign-disposition-20261107.ja.md)

The bounded campaign demonstrates useful CST FFNs on the recorded small
language-model task with a fixed linear Atom budget. It does not demonstrate
Dense superiority, convergence, three-seed repeatability, a minimum coefficient,
asymptotic runtime, or speed/GPU-memory advantages. Scientific execution is
finished under the owner's amendments; research-repository integration is
completed only when this PR is merged.

| Requirement | Recorded disposition and evidence |
| --- | --- |
| Rectangular/multi-token mathematics | Independent polynomial quadrature plus CPU FP64/native Y, dX and all-Atom VJP checks; explicit2048 route passed12 expanded cases. R3's small task fixture had2 blocks/4 CST projections; it is not the actual4-block task. [Backend record](https://github.com/takumiecd/torchcst/blob/c3b68dcf6275f96d29a21ff2aa6e989d42de2cff/docs/research-history/cuda-linear/ffn-rectangular-2048-20261107.md). |
| All FFNs in a causal LM | Four pre-LN blocks,4 heads, context128, eight CST projections, Dense attention/embed/output; fixed common initialization/data/order and causal masking checks. [Model/protocol](ffn-feasibility.md). |
| Useful task learning and controls | Fresh Dense/no-FFN/CST; selected N256 seed17 5M plus the bounded width/aspect family, each terminal-audited. CST improves over no FFN at all retained shapes, remains behind Dense. [R8 results](ffn-feasibility-results.md), [R10 shapes](ffn-feasibility-r10-shapes-20261107.md). |
| Linear Atom budget across widths | Fixed c8 in A=ceil(c*sqrt(d_in*d_out)); N128/256/512 and expansion1/4/16 observed. Atom/FFN parameter count is linear by construction; task usefulness is empirical only at these finite points. Dense attention stays quadratic. |
| Agreed long seed family | All seed29 widths completed5M and passed24 trained projection VJPs/9 resume audits. Owner stopped the six unstarted seed43/71 cases; planned3seed confirmation remains incomplete. [Retained report](ffn-feasibility-r11-seed29-20261107.md). No cancelled learner is restarted. |
| Curves, samples, time and memory | Both fixed-prompt generation examples, train/DEV curves, preparation/elapsed times, synchronized update samples, allocated/reserved peaks and parameter counts retained for every admitted run. Current eager CST is slower and uses more measured allocation than Dense. |
| LR mechanism follow-ups | R12–R14 amplitude-only interventions are separate from corrected R15 whole-Atom32/64. All are audited and plotted. Higher whole rates did not improve the final short-budget DEV value over16. [Amplitude-only](ffn-field-lr-r14-20261107.md), [whole-Atom](ffn-whole-lr-r15-20261107.md). |
| Public API | Absolute field_lrs, semantic declarations, common scheduler factor, checkpoint restore; remote CPU25/native CUDA8 and actual4-block8-case/64-VJP parity/resume gates passed. Merged torchcst PR151; parameters/losses matched the prototype exactly in recorded task updates. [R16 API record](ffn-field-api-r16-20261107.md). |
| Source/evidence preservation | Original source snapshots/data hashes, failed pre-learning attempts, negative results, owner stop, raw archives and remote checkpoints retained; derived PNG/SVG and equivalent English/Japanese notes committed. [Frozen reproduction guide](ffn-reproduction.md). |
| Resource/scope limits | After the owner restricted execution, sole approved RTX4090 with sequential scientific children; validation workers finished and user-owned Hub stays up. No A100/Colab reallocation, HIGGS/DST restart, Atom lower-bound/Big-Omega search or Attention replacement. |

The final evidence re-audit checked8 retained R8/R10/R11 archives and88 raw
members,64 trained projection VJPs/24 resume audits,24 observed runs with two
samples each, final token budgets, curve/time/memory fields and the Atom formula.
R12–R16 have separate hash-verified terminal archives and audits. All58 committed
derived figure files match their manifest. A report/verifier is evidence only
within these contracts; it does not authenticate arbitrary third-party data or
establish a stronger quality claim.

The owner-stop records and formal reader gates remain intact. The original
three-seed plan is visible and incomplete rather than relabeled as successful.
A separate current-FFN library extra pins the merged public runtime; old HIGGS
pins and measured source snapshots are unchanged. No historical learning result
is silently migrated to the new API.

Remaining research questions include why the Dense gap persists and whether more
Atoms at fixed width reduce it. Existing width128/256/512 seed29 DEV gaps are
0.1062/0.1069/0.1126, so width alone has not erased the gap. Larger final projection
RMS under high LR did not improve the endpoint, so low output magnitude alone
is insufficient. Atom capacity, feature diversity, coordinate conditioning and
finite training duration remain hypotheses. A fixed-width c8/16/32 comparison
would be a separate bounded capacity test, not a completed result or a resumed
lower-bound search. This campaign does not run that new test or Attention work.
