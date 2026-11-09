# Two-scale and two-window Fourier connection proposal

English (canonical) | [日本語](fourier-scale-connection-20261109.ja.md)

Owner-authorized formula derivation and CPU conformance only. This is a new
mathematical composition of the frozen public Fourier cell integrals, not a
production operator/Plan, model, optimizer, GPU or speed/learning change.
Keep all previous results on their exact definitions. This proposal changes
initial profiles and is not a checkpoint-compatible replacement.

## Fixed banks and geometry

Keep T_l=2^l*T0 and the existing K3 banks, P_l=3T_l. In unit-grid notation,
a fixed window n consists of unwrapped site IDs n*T through(n+3)*T-1.
Let b_j^(T,n)(mu,R) be the public periodic-density whole-cell integrals on this
immutable window, scattering repeated images modulo the global site count.
Each branch has nonnegative masses summing to1 and is smooth in mu,R for
0<R<P/2. This is an invocation of the public integrated_cells function,
not a copied profile evaluator. No L2 norm floor is introduced.

Use a quintic smoothstep S(t)=10t^3-15t^4+6t^5 on[0,1], extended by0/1.
S'=30t^2(1-t)^2; S'=S''=0 at both endpoints. This yields C2 glue when the
branches are C2. Numerical evaluation uses the symmetric endpoint form.

For fixed T, define q=(mu+1/2)/T-3/2, n=floor(q), t=q-n. The two fixed
window centers bracket mu. Around their nearest-window switch, use

\[
 u=\operatorname{clip}\!\left(\frac{t-(1/2-\delta)}{2\delta},0,1\right),
 \quad m_j^T=(1-S(u))b_j^{T,n}+S(u)b_j^{T,n+1}.
\]

Outside the transition band only one window is needed. Even when n changes,
the selected nonzero window and its derivatives agree on both sides.
The minimum distance to the nearer window edge among active windows is
(1-delta)*T. Therefore full core coverage requires R<=b*T with b<=1-delta.
This geometric condition is essential; smoothing two windows without padding
can clip the intended core.

## Radius connection

Choose0<delta<1/2 and0<a<b<=1-delta with b/2<a<b. For the smallest T satisfying
R<=b*T, use m^T alone below a*T; in[a*T,b*T], define

\[
 v=\frac{\log(R/(aT))}{\log(b/a)},\quad w=S(v),\qquad
 m_j=(1-w)m_j^T+w m_j^{2T}.
\]

At b*T, w=1,w'=w''=0; just above it, the next primary T is2T and its sole
branch is the identical m^(2T). At a*T, w=0 with the same zero derivatives.
Adjacent transition bands do not overlap because2a>b. Consequently the
composition is C2 across radius routing and position routing, periodic on the
global grid, convex and unit-mass. At most2 banks and2 windows per bank are
active per axis; coefficients can be shared across the two same-bank windows.
Input/output axis mixtures retain a single Atom's product factorization.

All active banks satisfy a/2<R/T<=b except the declared minimum/base-scale
region. The earlier strict interval[1/2,1] must therefore be relaxed for helper
banks; retaining it for every active bank leaves no nonzero overlap. The
illustrative choice delta=.1,b=.9,kappa=b/a=1.25 gives a=.72 and active ratios
(.36,.9]. This is a geometry-consistent example, not an adopted hyperparameter.
At the minimum tile width, reducing R further still faces the fixed-M
concentration limit. Primary tail claims here are restricted to R>=.5.

The full radius derivative includes the connection term:

\[
 \partial_R m=(1-w)\partial_R m^T+w\partial_Rm^{2T}
 +\frac{S'(v)}{R\log(b/a)}(m^{2T}-m^T).
\]

The center derivative analogously includes S'(u)/(2delta*T) times the
neighboring-window difference. Neither weight is stop-gradient. Making the
connection extremely narrow can produce large derivatives; C2 is not a bound
on conditioning. The new radius schedule changes the effective width and
initial mass energy; existing initialization requires separate qualification.

## Pre-execution CPU protocol

Use one-axis size128, spacing1, frozen operator0d277ae6dc0ae14fe515c7559f39535ed0238950.
FP64 M8 compares old public routing, radius-only glue, full glue, and two
intentional missing-weight-gradient controls. M16/32 evaluate full glue only.
Twenty fixtures cover a*T/b*T at T1/2/4 in both directions, old radius.5/1/2,
position-band endpoints/midpoint, mixed coordinate/radius and the periodic seam.
Eight displacement scales1e-2..1e-12 give140records/1120displacements.

A bounded tail screen uses kappa1.05/1.1/1.25/1.4 at fixed delta.1,b.9,
M8/16/32 and41 log-spaced radii.5..4:492 samples. Mix the public continuous
outside-core mass support_eps; the core-coverage proof makes this the same
unwrapped tail fraction for every active position window. Measure total tail,
not a certified maximum sidelobe or task tolerance. Narrower bands trade lower
helper-bank leakage against larger connection derivatives. Geometry probes49
points; independent tests also include size8 with repeated images.

Tests require convex weights, endpoint zero first/second derivatives, outer
chain-rule equality, mass error<=1e-12, core coverage and2-bank/4-window limits,
periodic parity, independent cell quadrature, quadratic Taylor decay at the
sampled boundaries and linear error when radius-weight differentiation is omitted.
Record absolute errors and error/parameter-step; tiny derivative norms and
roundoff make cosine/relative-Jv ratios insufficient. No throughput claim
follows from a constant branch count. Preserve source/result hashes, negative
controls and limitations before any GPU or learning decision.

Runner:scripts/inspect_fourier_connection.py.
Evidence:CST_RESEARCH_EVIDENCE_ROOT/fourier-scale-connection-cpu-20261109-01.
