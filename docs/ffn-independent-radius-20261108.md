# Independent task-gradient radius: RTX 6000 Ada FFN pilot

English (canonical) | [日本語](ffn-independent-radius-20261108.ja.md)

This prospective, single-seed pilot asks whether directly learning physical
radius improves the existing TinyStories FFN result. It does not estimate the
optimal radius, convergence, noninferiority, or an Atom-count lower bound.

## Frozen comparison

Use the existing language model and learner: width512, four pre-norm blocks,
eight heads, context128, FFN expansion4, tied Dense embedding/vocabulary head.
Attention remains Dense. All eight FFN maps use CST with
`A = 8 sqrt(d_in d_out) = 8192` Atoms per map, M8 cell-mass profile,
base tile1/merge2, periodic unit grid. Three arms, in fixed order:

1. `polar`: existing rational-zero-opening Polar, minimum scale0.001,
   radius birth0.5, q0, amplitude maximum1.
2. `learned`: decode the same realized Polar initialization to physical
   `(a, r, mu_out, mu_in)` and use ordinary AdamW on all four fields.
3. `fixed`: exactly the same physical Parameters and optimizer, masking only
   the radius gradient before global clipping. Its radius moments stay zero.

Conversion rounds the decoded amplitude/radius once to FP32. Centers and
ordinary model Parameters are shared exactly; learned/fixed initial tensors are
identical. Initial Polar/physical logits must agree within the prespecified
FP32 gate. Polar versus physical also changes amplitude update geometry;
**learned versus fixed isolates radius learning in the physical representation**.

Independent physical radius has no clamp in forward. After AdamW, project
radius to `[0.5,50]` and amplitude to `[-1,1]`. Centers remain periodic through
the operator. These bounds define this experiment, not an unrestricted optimum.
AdamW: base LR0.0003, all Atom fields LR x16, betas(0.9,0.95), decay0,
global gradient clip1, FP32 with TF32 disabled. No LR selection follows results.

Seed17; batch2 x context128 x accumulation4; 976 updates/999424 tokens,
unchanged 20M-token schedule horizon and fixed training prefix. Token manifest:
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
Only the 8192-token development split is evaluated; confirmation remains unused.
No new Dense run: the historical matched-recipe Dense result is context only.

## Admission and measurements

Freeze full source commits and per-file hashes before launch. Use one reserved
RTX6000 Ada 48GB (verify actual identity/runtime); run serially under the existing
durable controller. A private explicit physical LM Plan reuses the full native
physical radius derivative, including both input/output factors. It does not
change public defaults or reuse Polar's intentionally radius-free reduction.

Before learning, require CUDA regression tests, an independent Gauss-quadrature
physical derivative check, all-8192-Atom CPU-reference VJPs for both FFN shapes
with 256 noncontiguous rows, retained-backward parity, full-model initialization
parity, finite/nonzero learned-radius gradients, fixed-gradient masking, and
complete-step/checkpoint/optimizer-resume gates for each arm.

At steps0,1..5,every20,and976 (55 records), record each of eight layers' amplitude
and radius mean/RMS/p10/p50/p90 and boundary occupancy. Measure task gradients
on the same fixed probe after the update, before clipping, outside the update
timer. Polar has no task radius gradient; fixed masks it to zero. These are
probe gradients, not accumulation-averaged training gradients. Boundary
occupancy is not a count of proposals that hit the bounds.

Overlay learning curves against updates/tokens, radius trajectories and radius
gradient RMS. Record synchronized full-update time, preparation time and peak
allocated/reserved memory; diagnostic overhead belongs to wall time, not the
update timer. Keep the fixed run order explicit when interpreting timing.
After training, independently restore every checkpoint, recompute DEV, compare
trained VJPs, verify fixed radii/moments and deterministic next-step resume.
Retrieve compact evidence and verify hashes before marking COMPLETED. Preserve
checkpoints, data, source archives and native builds on the cluster. Do not
continue or modify a failed frozen attempt.

Runner: `cst_research.experiments.ffn_radius_qualification`, followed by the
existing `cst_research.experiments.ffn_run --kind cst --attention-kind dense
--linear-backend levels --radius-mode {polar,learned,fixed} --track-radius
--development-only` with the fixed settings above and hashed admission receipts.

Execution/result identities are recorded separately after the source freeze.
