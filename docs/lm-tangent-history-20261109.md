# Local tangent geometry and Adam-history attribution

English (canonical) | [日本語](lm-tangent-history-20261109.ja.md)

Owner-authorized **analysis only**. No continuation, LR/kernel/Atom-count change,
DEV selection, improvement experiment or public-default adoption. Use original
task gradients and unchanged audited step4882 checkpoints of the width512 LM.
Analyze block1 up/down in Dense,DSTall,CSTall, and block1 up in CSTsingleton:
seven local map states. Observe actual next1024-target TRAIN batch and offsets32/64
at the same restored clock:21 records. No held-out data or additional seed.

For fixed actual input X, z=f(theta,X), task cotangent g and J=dz/dtheta:

* Compare infinitesimal Jv with the realized finite output change. Retain
  original Adam/clipping/field rates and include the ordinary projection bias.
* Holding the post-current-gradient second moment fixed, replace the first
  moment by the current clipped gradient. Separately replace the denominator
  by abs(current gradient)+epsilon. Record the full2x2 history factorial plus
  raw current-gradient direction and actual quantized displacement. These are
  direction measurements only, never applied as training steps. Boundary-tangent
  projections are reported separately from unconstrained directions.
* On offset0, measure the best unrestricted linear-map tangent using the SVD
  of [X,1] at the **same X,g**, not Dense teacher weights. Use relative singular
  cutoff1e-7. DST's active-mask tangent is measured by per-output SVD.
* For CST, recover factor partials with four CPU public-factor JVPs per atom
  chunk. Atom independence makes each column its own derivative. Use these
  factors for matrix-free Jv/J-transpose-g. Column-normalized CGLS estimates the
  projection of -g into the numerically resolved local Atom/bias tangent, with
  max256 iterations, normal-residual target1e-4 and column cutoff1e-10. Record
  convergence/trace and recompute the residual explicitly. A nonconverged result
  is a reachable lower bound, **not a structural ceiling**. The unconstrained
  tangent ignores one-sided fraction/amplitude bounds and is not a finite-step
  capacity or an Atom-count lower bound. Conditioning can make raw-gradient
  alignment low even when range(J) contains the task direction.
* Evaluate temporary finite displacements at epsilon1/.1/.01, with the original
  displacement scaled inside its feasible segment and periodic centers wrapped.
  Record tangent/finite angle, linearization defect and declared route-signature
  changes. This distinguishes local tangent behavior from finite nonlinear,
  routing-boundary or rounding effects; no model trajectory is retained.

Native outputs versus factor-reference relative L2 error <=.005; cached task VJP
versus native clipped task gradient relative L2 <=.005; epsilon1 reference/native
output-change relative error <=.01. These new analysis consistency gates do not
relax native Y/dX/dP tolerances1e-4/4e-3. CPU known-answer tests cover explicit
Jacobians, conditioned-gradient versus tangent projection, exact masked/free
subspaces, the no-history Adam limit and actual public radius-factor derivatives.

Use the reserved RTX4090 after identity/runtime/source/data/order checks and
fresh68 native tests. Restore Parameters,buffers,optimizer and CPU/CUDA RNG after
every temporary actual step; deep-copy optimizer loads and verify original clocks
and checkpoint file hashes. Public operators remain frozen; this research code
consumes their factors rather than changing CUDA/runtime math. Save to persistent
NFS and retrieve hash-verified compact evidence, including failures and
nonconvergence. Timing is instrumented and not a performance benchmark.

Evidence:CST_RESEARCH_EVIDENCE_ROOT/lm-tangent-history-4090-20261109-01.

## Recorded analysis extension

The primary21 observations passed;256-iteration Atom projections did not converge, and history removal also raised Dense within-batch alignment. A separate immutable follow-up therefore raises the analysis-only CGLS bound to2048, adds finite epsilon.001/.0001, partitions the linearization defect into changed-route and unchanged-route Atom contributions, and evaluates each source-batch direction on all three target TRAIN batches. This3x3 cross-batch check distinguishes fitting the current batch from history that tracks shared directions. Same checkpoints,data,operators and numerical gates; no training or improvement intervention. Primary artifacts remain unchanged.
