# HIGGS layer positions: explore, refine, then test predictions

English (canonical) | [日本語](higgs-layer-position-plan.ja.md)

**Authorized follow-up; planned, not running or preregistered.** GPU execution
starts after the current frozen HIGGS alpha-confirmation campaign finishes.
Its source, eight-configuration family and selection/evaluation rules remain
unchanged. This planning note is not a runnable experiment or completed result;
no GPU job has been submitted for it. Register a new study once its real source,
runner and executable protocol exist.

## Controlled architecture

Use the same model for every layer-position arm:

```text
28 -> N1 -> N2 -> N3 -> N4 -> 2
      N1=N2=N3=N4=N, initially1024
```

The input projection and output head stay dense, all layers train, and each of
the four hidden activations is ReLU. All three square hidden transformations
are bias-free in every arm. Compare fully dense with exactly one CST replacement
at N1->N2 (early), N2->N3 (middle), or N3->N4 (late). Width, depth, activations,
data and update budget stay identical across positions. This deeper model is
separate from the current28->N->N->2 architecture. Simultaneous replacement of
several layers is a subsequent experiment.

Use independent per-layer initialization streams and shared data orders. Shared
ordinary layers start byte-identically; use the same initial CST parameter draw
when moving the replacement among equal-width positions. Preserve public CST
initialization and log activation-scale differences without hidden gain
calibration. Fair LR tuning and a common training-horizon rule are required:
failure at one arbitrary LR is not evidence of insufficient capacity.

## Exploration and confirmation have different jobs

1. **Explore coarsely.** First check that same-depth dense trains. A proposed
   starting set is alpha={0.0001,0.001,0.01} per position, seed17. These are
   planning anchors; fix the actual grid and compute budget after the current
   campaign and short correctness/budget pilots. Retain all trials and curves.
2. **Refine observed transitions.** Add intermediate points such as0.0002/0.0005
   or0.002/0.005 where useful. If the lowest tested density remains sufficient,
   report an open boundary and extend downward within a declared budget.
   Repeat important irregular/transition cases. Do not assume monotonic quality
   or treat an inconclusive noninferiority test as proof of inferiority.
3. **Form explicit hypotheses.** Candidate explanations include constant A,
   A proportional to N, and position-specific c*N^p. Preserve fit inputs and
   predictions before the next test. At one fixed N, c=A/N is just a relabeling;
   it cannot identify a scaling law. Two widths do not establish an exponent.
4. **Test predictions independently.** Freeze a new comparison family,
   margins, LR/checkpoint/horizon rules, new seed IDs, evaluation split and
   analysis before scoring. Test predicted counts and smaller/larger neighbors
   with new seeds and a width excluded from the fit (for example2048 after1024,
   or4096 after1024/2048). New seeds test repeatability; an excluded width tests
   extrapolation. A successful prediction supports the tested scope, rather
   than proving a universal law. Revise contradicted hypotheses and test them
   in a separate subsequent phase.

Use the accepted absolute margins accuracy0.002 and ROC-AUC0.002 as the starting
quality criterion. The eventual multiplicity/seed analysis must cover the new
family; the current eight-configuration test does not cover adaptive additions.
One possible bounded confirmation selects a candidate and one observed neighbor
per position, plus dense, with three paired seeds. Freeze the actual family
after exploration. Such a subset supports only its tested points, not a minimum
over all exploratory densities. Report the smallest supported tested alpha,
adjacent inferior/inconclusive observations and unresolved boundaries.

For square layers,

```math
A=\lfloor\alpha N^2\rfloor,\quad c=A/N,\quad
A_{\mathrm{required},\ell}(N;Q^*)\approx c_\ell N^{p_\ell}.
```

Fixed alpha gives quadratic atoms; constant A requires alpha proportional to
N^-2; linear atoms require alpha proportional to N^-1. A coefficient c<1 at one
width does not establish a sublinear exponent. The current product kernel has
W=sum_a amplitude_a*u_a*v_a^T and mathematical rank(W)<=min(A,N). A<N need not
produce zero rows/columns because each profile covers multiple sites. Coverage
and independent linear directions are separate. Fixed spacing/radius settings
also change relative support with width and remain part of any scaling claim.

## Measurements and data

Overlay every evaluated density/position/LR/seed's train/development loss,
accuracy and AUC against updates and synchronized training seconds. Retain
individual seeds. Record preparation, evaluation, checkpoint, HPO and process
wall-time costs separately, with capture-inclusive allocator allocated/reserved
peaks. Compare quality at equal updates and time/memory at a common prespecified
quality target. Keep useful smaller-width dense/CST controls for utility claims.

Record initial/per-layer activation and gradient scales, zero/active
activations, profile coverage and zero rows/columns. Declare near-zero
thresholds and any effective-rank diagnostic. A B4 activation matrix has rank
at most4 trivially: use a larger fixed untimed development probe or an operator
spectral diagnostic. Sampled inactivity, exact support and rank are different;
these diagnostics suggest mechanisms rather than establish causality.

Use development data for exploration. The current confirmation50k will already
have been evaluated when this study starts. New seeds/study IDs do not make
those cases a fresh holdout. Before a formal claim, audit prior data exposure
and freeze either a separately reserved, previously unused evaluation set with
preprocessing fitted outside it, or explicitly report reused-validation scope.
A new holdout from the original training partition needs new preprocessing
provenance: the existing standardizer used all10.4M training rows. Retaining
that standardizer requires explicit disclosure of feature exposure. The official
test remains unopened until a separately fixed final evaluation.

## Readiness before execution

Implement in cst-research against an explicit torchcst revision, preserving the
running campaign. Independently qualify each position and the deeper model:
Y/dX/all atom gradients, optimizer states and complete-step trajectories. The
current private Graph cohort admits its original model and density grid; this
architecture and smaller densities need new admission, never a bypass. Use
public eager or a separately qualified experiment Graph cohort consistently
for controls. Freeze source/data/runtime hashes, resource budget and adaptation
rules before the relevant phase. Use the shared L4 pool within actual allocation
limits. Documentation alone does not grant GPU resources or validate a runner.
