# Radius, amplitude and coordinate gradients

English (canonical) | [日本語](radius-coordinate-gradient-discussion-20261108.ja.md)

These equations record a discussion during the independent-radius FFN pilot.
They are proposals/identities, not an implemented update law or empirical result.
The running pilot remains the frozen three-arm comparison in
[its protocol](ffn-independent-radius-20261108.md).

For one Atom, write `W_j=a_j K(r_j,mu_j)` and
`G=partial L/partial W`, with the Frobenius inner product. Then

```math
g_a=\langle G,K\rangle,\qquad
g_r=a\langle G,\partial_rK\rangle,\qquad
g_\mu=a\langle G,\partial_\mu K\rangle.
```

Coordinate and radius gradients already depend on each other through the kernel
shape. Both scale with amplitude; at exactly zero amplitude they vanish for a
finite differentiable factor, even though the amplitude gradient can be nonzero.
This identity does not establish the reason for a particular learning plateau.
It also does not imply a policy that moving coordinates must broaden radius.
For cell-mass kernels, derivatives are piecewise in routing/level membership;
check branch interiors and transitions separately rather than assuming global
smoothness from the within-branch expression.

## A fully differentiated Polar extension

For `a=a(s,t)` and `r=L(a)+(U(a)-L(a))p(s,t)`, define

```math
r_a=L'(a)+(U'(a)-L'(a))p,\qquad
\partial_s r=r_a\partial_s a+(U-L)\partial_s p,
```

with the analogous expression for `t`. A radius-live loss would give

```math
\partial_s\mathcal L=g_a\partial_s a+g_r\partial_s r,\qquad
\partial_t\mathcal L=g_a\partial_t a+g_r\partial_t r.
```

The current Polar task contract detaches its derived radius. The equations above
describe a **different** gradient contract, which would require its own declared
parameterization/validation rather than silently replacing the old backward.
It preserves the shared-coordinate representation and includes the L/U slope
terms; adding only a partial radius term would not be this chain rule.
The independent-radius pilot tests direct physical gradient learning first;
it does not test this Polar extension.

The linear envelope above is a possible construction. The current implementation
actually uses logarithmic interpolation, equivalent in exact arithmetic to
`r=L(a)^(1-q) U(a)^q`. Its interior derivative is

```math
\partial_s r=r\left[\left((1-q)\frac{L'}L+q\frac{U'}U\right)
\partial_s a+\log(U/L)\partial_s q\right].
```

For `rho=sqrt(s²+t²)`, the current map is `a=a_max s/rho`,
`q=clamp((rho²-1)/3,0,1)`. Away from singularities/clamp boundaries,
`a_s=a_max t²/rho³`, `a_t=-a_max st/rho³`, and in `1<rho<2`,
`q_s=2s/3`, `q_t=2t/3`. Outside the unclipped q interval its derivative is0.
Endpoint conventions need explicit tests, not an interior derivative assumption.

There is also an update-law issue: the current Polar update projects the raw
optimizer proposal onto the tangent of the `(s,t)` circle, then evolves radial
activity using its separate activity/regularization law. Merely adding the loss
radius derivative to backward would therefore **not** implement ordinary
gradient descent in q. A radius-live Polar experiment must declare both its
derivative and update contract and verify Parameters/moments/resume accordingly.

## Motion is a separate state/update proposal

Raw absolute position should not control radius on a translation-invariant
periodic grid. If using motion, use the actual periodic center displacement,
normalized by the relevant radius/spacing and recorded after the optimizer step,
not the raw gradient magnitude. For shared radius, one possible dimensionless
quantity and bounded response are

```math
D_t^2=\|\Delta\mu_t/r_t\|^2,\qquad
f(D_t)=\frac{D_t^2}{D_t^2+\tau^2}\quad(\tau>0).
```

This has `f(0)=0` and tends to1 for large motion. Smoothing it over time would
introduce optimizer state, time scales and checkpoint requirements. A formula
using the previous update is an update rule, not a static forward
reparameterization with ordinary chain-rule differentiation through that update.
Zero movement could mean convergence, weak amplitude, a flat narrow kernel or
small scheduled LR; it does not by itself show that narrowing is appropriate.

Using `p=f(D)` in the envelope gives `r→L(a)` when movement vanishes, **not**
necessarily global `r_min`. If `a→0` and both bounds approach `r_max`, broad
radius is preserved even at zero movement. This resolves the apparent conflict
between the two endpoint requests by giving the amplitude envelope priority.
Exact L/U functions, any history averaging, escape mechanism and combination
with task `g_r` remain undecided. No new motion-based experiment is authorized
or launched by this discussion; inspect the current pilot before selecting one.

## CPU chain-rule qualification and proposed next comparison

After the independent-radius pilot completed, the full s/t radius chain rule
was checked on six CPU FP64 Atoms with interior q values0.1/0.25/0.4/0.6/0.8/0.9.
The hand-derived expression and autograd agreed to4.44089e-16 maximum absolute
error; two central-difference spacings agreed within2.30e-9 over all four fields.
The live gradient had nonzero radial components of both signs, while the
radius-detached task gradient's radial component was numerical zero.
The radius map's q0/q1 inward one-sided slopes were also checked; a central
derivative at the clamp corners was not claimed.

The existing value helper selects U for q>=1: its autograd slope at q1 is0,
whereas the inward mathematical slope is nonzero in these cases. The current
detached-radius contract does not use that slope. A new radius-live contract
must explicitly implement its intended endpoint Jacobian; merely removing
detach from the existing helper would not supply it.

Run `python -m cst_research.experiments.polar_radius_chain_rule --output FRESH`
with the qualified source dependencies. The [program](../src/cst_research/experiments/polar_radius_chain_rule.py)
and [compact CPU verification](radius-coordinate-chain-rule-20261108.json) are
inspectable. This verifies the algebra on branch interiors, not a new CUDA
backward, optimizer, boundary-routing implementation or LM learning result.

The next proposed learning comparison keeps the current amplitude-dependent
L/U and logarithmic radius map, and compares radius-live versus radius-detached
task gradients under the **same** update law. One concrete candidate is ordinary
Adam followed by projection to `1<=sqrt(s²+t²)<=2`, with periodic centers and
identical LR/data/initialization; radial proposals are allowed. This needs an
explicit inward endpoint-gradient convention and full parameter/moment/resume
qualification. The existing tangent-only/activity-clock Polar can be retained
as context, but changing backward and optimizer together against it would not
isolate the radius gradient. No such LM comparison has been launched yet.
