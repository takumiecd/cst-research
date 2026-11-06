# He/Xavier-compatible CST initialization pilot

English (canonical) | [日本語](higgs-fan-initialization.ja.md)

Completed bounded execution and independent saved-evidence audit. See [results and limitations](higgs-initialization-results.md). The protocol below retains its original design; preparation/future-tense statements are historical.

This is an experimental research initializer, not a public torchcst default.
Use lowercase `q`, `s`, `t`. The configurable fraction `q` in [0,1] is the
Polar `alpha_init`, distinct from Atom density alpha=A/N². Before training,
set rho=sqrt(1+3q), s=rho*a/amplitude_max and
t=rho*sqrt(1-(a/amplitude_max)²). Physical radius is
R=L(a)^(1-q)*U(a)^q: q=0 is the amplitude-dependent lower bound; q=1/2 is the
geometric midpoint; q=1 is the upper bound. Existing radius bounds and the
public975 runtime remain unchanged.

## Mathematical contract

For W=sum_a a_a Phi_a, fixed amplitude-free Atom matrices Phi_a and independent
zero-mean amplitudes of variance sigma_a², define E_K=sum_a ||Phi_a||_F².
Then expected ||W||_F²=sigma_a² E_K. Effective fans are E_K/d_out and E_K/d_in;
using them in He/Xavier gives the same average forward/backward second-moment
targets as the corresponding dense initialization. For coordinate-basis Atoms
representing all dense entries these fans reduce exactly to d_in and d_out.

| Scheme | Target expected ||W||_F² |
|---|---:|
| He, fan_in, gain=sqrt(2) for ReLU | 2*d_out |
| He, fan_out, gain=sqrt(2) | 2*d_in |
| Xavier, gain=1 | 2*d_in*d_out/(d_in+d_out) |

More generally He uses gain²*d_in*d_out/fan_mode; Xavier uses
gain²*2*d_in*d_out/(d_in+d_out). Gain and mode are explicit in the utility.
For a square map and mean unit-amplitude Atom energy e, the fixed-shape He
formula is sigma_a=sqrt(2/e)/sqrt(A/N).

In the actual Polar kernel the profile depends on amplitude magnitude, so
solve the second-moment target with that dependency and clipping included.
Recover the saved original nominal standardized Gaussian draw z from the
public975 zero-q initializer. For each trial scale use
a=amplitude_max*clip(scale*z/amplitude_max, -1+1e-6, 1-1e-6), then encode s,t.
Evaluate these actual FP32 Atom values with public CPU FP64 factors.

For the realized magnitudes and centers, independently random amplitude signs
give the exact conditional expectation
E_sign ||W||_F²=sum_a a_a²||Phi_a(a_a,q)||_F². Geometry depends on squared
amplitude, so it is invariant under these sign flips. This remains valid with
overlapping Atoms. It does not say that one realized W has exactly that norm,
nor that all rows have equal energy or Dense-like correlations/singular values.
The factors yield each Atom energy as a product of squared vector norms, with
no dense W allocation and no TRAIN features or Dense reference model.

Search scales 0,.025,.05,.1,.2,.4,.8,1.6,3.2,6.4,12.8,32 for a sign bracket,
then at most32 bisections. Accept only measured target error <=0.5%.
No global monotonicity is assumed. If the declared search fails, retain its
history and stop before training; do not change q, amplitude bounds, Atom count
or add gain silently. This bounded search does not prove mathematical
unreachability outside the searched path. Symmetric clipping means amplitudes
are not exactly normally distributed; retain both nominal scale and physical
amplitude RMS and every clipping count. ReLU gain is not a universal choice for
other activations or residual architectures.

## Relation to completed initialization trials

All earlier N1024/H8/A4096 trials used q=0. Old scale .1/sqrt(A)=.0015625 and
width-aware .1/sqrt(A/N)=.05 both stalled at DEV AUC .5. The subsequent
TRAIN-feature Dense-RMS match used per-layer nominal scales .315625–.359375,
gain1 and no LayerNorm, reaching AUC .765233 versus Dense .771360 in one seed.
It matched empirical inputs and the legacy nn.Linear target, not He/ReLU.
The legacy dense internal weights have ensemble variance 1/(3*d_in), whereas
He/ReLU uses 2/d_in. This pilot changes the dense internal initializer too.

## Frozen initial learning comparison

Run two separate GPU cohorts, each with fresh paired Dense/CST arms:

- RTX4090: He, fan_in, ReLU gain sqrt(2), q=0.
- A100 MIG3g.40gb: Xavier, gain1, q=0.

Both use N1024/H8/seven internal maps/A4096/seed17, no normalization, gain1,
unchanged paired legacy Dense input/output layers, the same TRAIN100k/DEV50k
rows and saved seed117 order, true B4 FP32/TF32 off/eager, stockq1
AdamW/CSTAdamW LR.0003 eps1e-8 ordinary/amplitude decay.01 rawAtom decay0.
Only the seven internal square maps receive the selected initialization.
Dense uses the corresponding PyTorch normal initializer under explicit seeds.
For this square model Xavier fan_in/fan_out choices coincide.

One epoch /25,000 task updates per arm; fixed final, no HPO/best-DEV selection.
First pass fresh inherited full-network Y/dX/all-parameter FP64 VJP, independent
31-Atom quadrature, scalarAdam, native8 versus4+load4 and observedfirst+7
state parity. Forty synthetic network updates per GPU and scalar8 are separate
from50,000 task updates. Numerical gates and optimizer/load math are unchanged.
Retain all TRAIN/DEV predictions and fixed-first-B4 copy-only forward/gradient
probes at0/1000/5000/10000/15000/20000/25000, plus real first-step moments and
accepted displacements. Runtime remains gain1; q is an initial condition, not
a constraint fixing radial activity throughout training.

Record calibration cost, bare complete-step time and allocated/reserved peaks,
evaluation/probe overhead and total job time separately. Cross-GPU He/Xavier
times do not identify an initialization-method speedup. Keep exact source,
data, config and runtime pins, normal completion, full saved-evidence CPU audit
and clipping distributions before interpreting quality. No confirmation or
official test is scored. One seed/epoch is an exploratory screen; passing does
not establish noninferiority or authorize a universal default. Public adoption
requires its own reviewed API/contract change. Keep the previous raw RMS recipe
as a separate comparator, never relabel it as He.

Entrypoint:
`python -m cst_research.experiments.higgs_fan_initialization_run --help`.
The runner requires explicit `--scheme he|xavier` and `--q VALUE`, plus immutable
source/input/runtime paths. Do not sweep additional q or Atom budgets as an
automatic response to a failed target or learning result. This pilot is the
currently authorized task; the broad campaign remains paused and LM work waits.
