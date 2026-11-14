# Gaussian and Fourier-window Taylor continuity diagnostic

English (canonical) | [日本語](kernel-taylor-continuity-20261109.ja.md)

Owner-authorized CPU mathematical analysis only. No training, optimizer,
production kernel/default change, speed benchmark or Gaussian adoption. The
question is whether actual center/width changes approach their first-order
predictions as displacements shrink, and whether increasing current Chebyshev M
changes the routing jumps. Freeze this protocol and runner before execution.

Use one amplitude1 Atom on periodic128x256 unit grids. The independent Gaussian
control integrates a wrapped Gaussian over every fixed cell, using CDF differences
with stable erfc tails, sigma>0 and a fixed image sum -4..4. All widths are at
most2.01, so omitted images are far below FP64 resolution; no adaptive cutoff or
minimum-image distance is used. Compare autograd with independently derived PDF
partials,64-point Gauss-Legendre quadrature, total mass, positive masses and
periodic-center parity. CDF definition:
[SciPy ndtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtr.html).

Compare the frozen public physical-radius cell-mass reference at Chebyshev
M4/8/16/32/64. M changes the density itself, not approximation accuracy for a
fixed Gaussian. Matching numeric sigma/radius probes the same geometric
boundaries; it does not match shape, output RMS or model quality. Reference-only
M32/64 results do not establish CUDA support.

Predeclared19 fixtures: physical-width boundaries0.5/1/2 in both directions;
input/output-center boundaries at (width,center)=(0.5,7),(0.5001,7.5),
(0.75,7.5),(1.5,7.5),(0.75,0); and within-branch width/output/input controls at
(1,0.75,7.3,13.7). Half-displacements1e-2/1e-3/1e-4/1e-5/1e-6/1e-8/1e-10/1e-12.
Use encoded endpoint differences, not nominal displacements, in Jv. Record
finite/predicted norms, absolute and relative Taylor residual, cosine, routing
metadata and FP32 no-op markers. Primary FP64 covers all6 profiles; secondary
FP32 covers Gaussian and current M8:152 fixture/profile/dtype records and1216
finite displacements. Include same-window controls to distinguish truncation
boundaries from smooth-branch arithmetic.

Gaussian independent JVP relative error must be<1e-11 in FP64. Known-answer tests
require quadrature absolute error<=2e-15, mass error<=3e-15, periodic parity,
cosine>0.99999 at h1e-4 and approximately quadratic Taylor remainder over
h1e-3→1e-4. At tiny h, roundoff or near-zero derivatives can invalidate relative
ratios; retain absolute errors and mark encoded no-ops rather than claiming
smoothness from them. Do not interpret M trends as matched-shape approximation
convergence or extrapolate a small fixture to learned task quality.

Consumer script:scripts/compare_kernel_taylor.py; operator0d277ae6dc0ae14fe515c7559f39535ed0238950.
Raw evidence:CST_RESEARCH_EVIDENCE_ROOT/kernel-taylor-continuity-cpu-20261109-01.
Keep exact source/runtime hashes, outcomes, failures and bilingual results. No GPU
is allocated and no existing training or evidence is changed.
