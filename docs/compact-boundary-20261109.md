# Compact-profile boundary diagnostic

English (canonical) | [日本語](compact-boundary-20261109.ja.md)

Owner-requested CPU analysis only. Consume frozen torchcst0d277ae6dc0ae14fe515c7559f39535ed0238950;
no model, optimizer, GPU, production backend or speed/quality experiment.
The primary question is whether public Triweight remains continuous at support,
coordinate and width boundaries, including discrete L2 normalization.

Use128 periodic unit-grid sites and a single [width,center] pair. Evaluate public
Torch Triweight point/raw and discrete L2(floor1e-6), with public raw Biweight and
WendlandC2 as related shape controls. Independent controls are Gaussian fixed-cell
integrals from the prior study and a normalized Triweight density integrated over
fixed cells. The latter is a mathematical reference only, not a new public
cell_mass binding. Its density is35/(32R)*max(0,1-((x-mu)/R)^2)^3. Integrate with
the exact polynomial primitive and a stable endpoint-tail expression. Use fixed
periodic images -2..2 for compact integrals. Different contracts/norms and widths
are not matched for task quality.

Freeze24 fixtures: radius0.5/1/2 changes in both directions at center7.3;
support-radius1 at center7, empty-birth radius0.5 at center7.5, support-center7
at radius1 and half-center7.5 at radius0.5, all in both directions; prior
window-center7.5 at radius0.5001; within-branch width/center at(0.75,7.3);
empty-interior width/center at(0.4,7.5); periodic-center0 at radius0.75;
and both directions across the Triweight L2 floor in center and width.
The floor locations solve (1-(d/R)^2)^3=1e-6 for a single active site.
Eight displacement scales1e-2..1e-12 match the Gaussian study. Six FP64 profiles
and four FP32 primary profiles give240 fixture/profile/dtype records and1920
displacements. This is an axis-factor diagnostic, not full LM or matrix timing.

Record finite/predicted norms, absolute/relative Taylor residual, cosine,
encoded no-ops, raw support counts and raw Triweight norms. Vanishing predicted
directions at compact support edges can make cosine/relative ratios undefined
or nonconvergent even for a smooth function; judge absolute residual order too.
Inspect the L2 floor separately: a slope kink is different from a finite value
jump. Independent tests check public polynomial/analytic derivatives and zero
first/second derivatives at support, exact-integral quadrature/mass/periodicity,
and continuity with different one-sided slopes at the L2 floor. Require finite
values/JVPs. Preserve source/runtime/result hashes and both positive and negative
observations, without a new task tolerance or kernel adoption.

Runner:scripts/inspect_compact_boundaries.py.
Evidence:CST_RESEARCH_EVIDENCE_ROOT/compact-boundary-cpu-20261109-01.
