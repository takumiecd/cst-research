# Polar: established relations and latent-coordinate design

English (canonical) | [日本語](polar-relations-20261108.ja.md)

This is a relationship inventory, not an adopted parameterization or a new LM
training result. It concerns the separable, L1-normalized **cell-mass** operator
used in the current FFN experiments. Other CST kernels can have different
normalization contracts. We retain the distinction between algebra, measured
behavior and proposed learning priors.

## 1. What amplitude actually measures

Write a single Atom, with amplitude-free input/output factors, as

\[
W_j=a_jK_j,\qquad K_j=u_j(r_j,\mu_{o,j})v_j(r_j,\mu_{i,j})^T,
\qquad \mathbf1^Tu_j=\mathbf1^Tv_j=1.
\]

Then the following identities hold:

\[
\sum_{o,i}(W_j)_{oi}=a_j,\qquad
\|W_j\|_F^2=a_j^2\|u_j\|_2^2\|v_j\|_2^2
=\frac{a_j^2}{n_{o,j}^{\mathrm{eff}}n_{i,j}^{\mathrm{eff}}},
\quad n^{\mathrm{eff}}(u)=\frac1{\|u\|_2^2}.
\]

For nonnegative unit-mass factors, effective cell counts lie between1 and the
number of sites. They count effective **cells, not Atoms**. Current amplitude is
signed total matrix mass, not a fixed matrix norm or task importance. Effective
counts depend on radius and center phase relative to the discrete grid. Radius
alone does not determine the Atom's norm. A broad factor typically distributes
mass over more cells; global monotonicity is not assumed across adaptive window
branches or profile families.

Total layer energy also contains overlap and cancellation:

\[
\|\textstyle\sum_jW_j\|_F^2=\sum_j\|W_j\|_F^2+
2\sum_{j<k}a_ja_k\langle u_j,u_k\rangle\langle v_j,v_k\rangle.
\]

Even a correctly measured single-Atom norm is not that Atom's usefulness to the
task. Data and other Atoms matter.

## 2. Exact gradient coupling and its limits

For matrix loss gradient \(G=\partial\mathcal L/\partial W\), at a differentiable
branch and finite shape derivatives,

\[
g_a=\langle G,K\rangle_F,\qquad
g_r=a\langle G,\partial_rK\rangle_F,\qquad
g_{\mu}=a\langle G,\partial_{\mu}K\rangle_F.
\]

At exactly zero amplitude, the loss gradients for radius and center vanish;
the amplitude gradient need not vanish. Small amplitude suppresses those
geometric gradients for fixed \(G\) and shape derivatives. The latter quantities
also change with shape, so amplitude alone cannot predict actual gradient size.
A narrow, grid-aligned Atom can additionally have very low center sensitivity.
Small center updates therefore do not prove that a useful optimum was reached.

With latent parameters \(z\), physical parameters \(\theta=\Phi(z)\) obey

\[
g_z=J_\Phi^Tg_\theta.
\]

Shared latent variables choose how physical updates are coupled. For a small,
unconstrained **vanilla SGD** step only,
\(\Delta\theta\simeq-\eta J_\Phi J_\Phi^Tg_\theta\).
Adam state, coordinatewise preconditioning and projection change this update.
A smooth reparameterization cannot recover a nonzero task gradient when all
physical task gradients vanish. It can move a center using an amplitude gradient
through shared variables; that is an induced update, not evidence for which
center direction the task prefers. A smooth chart with fewer than four scalars
cannot locally span all four independent directions \((a,r,\mu_o,\mu_i)\).

## 3. What existing Polar already expresses

For the current annular chart, \(\rho=\sqrt{s^2+t^2}\in[1,2]\),

\[
a=a_{\max}s/\rho,\qquad q=(\rho^2-1)/3,\qquad
r=L(a)^{1-q}U(a)^q.
\]

At differentiable interior points,

\[
\partial_qr=r\log(U/L),\qquad
g_s=g_a\partial_sa+g_r\partial_sr,
\quad g_t=g_a\partial_ta+g_r\partial_tr.
\]

The radius derivatives include both the amplitude-dependent bound derivatives
and the q derivative. If \(L=U\), the q direction cannot change radius.
At projection/clamp boundaries, a derivative convention must be explicit;
an interior formula alone does not define the whole implementation.
Writing s/rho=cos(theta) also gives a=a_max cos(theta): amplitude sensitivity
to this angular coordinate vanishes at a=±a_max. A clean forward formula can
therefore still contain a direction with a vanishing Jacobian. This algebraic
fact does not establish that trained Atoms are stuck at those endpoints.

The [matched full-chain FFN pair](ffn-polar-radius-gradient-20261108-results.md)
confirmed that the radius term can reach the s/t radial direction. It changed
final mean radius from2.058 to1.623 relative to the detached control; both grew
from0.975. DEV CE was3.42404 versus3.42239, with substantially overlapping curves.
This is not evidence for an optimal radius law or consistent accuracy gain.
The detached arm's q also moved under Cartesian Adam and projection: a zero raw
radial gradient is not the same as a frozen radial update.

## 4. Relations that remain hypotheses

| Proposed relation | Current status |
| --- | --- |
| \(a\to0\Rightarrow r\to r_{\max}\) | An exploration prior; not required by the loss or cell-mass normalization. |
| Large \(\lvert a\rvert\Rightarrow r\to r_{\min}\) | A concentration prior; a useful Atom can also be broad. |
| \(\|\Delta\mu\|\to0\Rightarrow r\to r_{\min}\) | Unverified; a small update can reflect small amplitude, a flat derivative, a small LR or optimizer state. |
| More center movement should widen radius | A possible controller, not an established property of the kernel. |
| Absolute center determines amplitude/radius | No universal rule established. Grid phase and task-dependent correlations must be distinguished. |

Observed movement \(\Delta\mu\) depends on an optimizer step and possibly its
history. A static map from current internal parameters cannot access it without
additional state or an explicitly defined update rule. Parameter independence
in storage also does not imply statistical independence after training.

## 5. A useful candidate for semantic internal parameters

First retain a four-dimensional reference chart with separate controls for
amplitude, radius choice and two center coordinates. Judge proposed sharing
against this reference; do not constrain center directions solely for elegance.

A second candidate follows directly from the norm identity. Define signed
**energy amplitude** \(b=a\|K(r,\mu)\|_F\). Then, for nonzero \(K\),

\[
\mu=G(\xi),\qquad r=R(b,q),\qquad
a=\frac{b}{\|K(r,\mu)\|_F},\qquad
W=b\frac{K(r,\mu)}{\|K(r,\mu)\|_F}.
\]

Now \(\|W\|_F=|b|\): b controls single-Atom size while q/center control shape.
No dense matrix is needed for the factor norm: \(\|K\|_F=\|u\|_2\|v\|_2\).
At fixed b, geometry derivatives are orthogonal to W, so their first-order
change does not change its Frobenius energy. This follows from differentiation
of \(K/\|K\|_F\), not from a successful training experiment.

This is a candidate, **not an adopted replacement**. It does not remove the
zero-amplitude geometric-gradient issue or inter-Atom overlap. With unconstrained
a it is a coordinate change; with the current amplitude cap/envelopes it may
change the feasible family. Keeping \(|a|\le a_{\max}\) requires
\(|b|\le a_{\max}\|K\|_F\). Keeping the original \(L(a),U(a)\) as well can create
an implicit equation; substituting b into those bounds is a new prior, not an
algebraic equivalence. Saturating latent maps and branch boundaries still need
explicit derivative contracts. Performance and learning benefits are untested.

The next analytical question is therefore whether the useful correlation is
between **mass amplitude and radius**, or instead between **energy amplitude,
shape and task gradient**. Start from matched physical values, examine both
amplitude definitions, center Jacobian norms, effective cell counts and Atom
overlap. A useful new chart should preserve needed directions and explain its
induced optimizer coupling before another broad sweep.

## 6. Reproducible CPU audit

[Program](../src/cst_research/experiments/polar_relations.py) and
[verified output](polar-relations-20261108-verification.json). Run with the
research source and torchcst commit
`9a57853f95fbe02e975fb082ad81bc4f3b9d7664` on PYTHONPATH; the numerical operator
source is unchanged from `88c2e8a7c4aa312f4f096c5169d712ace49ac979`:

```bash
python -m cst_research.experiments.polar_relations --output NEW_OUTPUT_DIRECTORY
```

CPU FP64, Torch2.13.0, N32 square, one physical cell-mass Atom, M8/base1/merge2.
The audit checks selected branch derivatives, mass/energy identities, the zero
amplitude limit, a norm-order reversal and normalized-shape orthogonality.
It is not an independent oracle for the Fourier profile or a GPU qualification.

| Example | a | r | grid phase | Atom Frobenius norm |
| --- | ---: | ---: | ---: | ---: |
| Narrow | 0.1 | 0.5005 | 0 | 0.0997043 |
| Broad | 0.2 | 4.7 | 0 | 0.0402947 |

Thus smaller mass amplitude can coexist with a larger Atom norm. At fixed
a1/r0.5005, moving the phase from0 to0.25 changed the center-Jacobian norm from
0.00008250 to1.02068. These are small constructed examples, not a diagnosis of
trained LM Atoms. Maximum mass and energy-identity errors were respectively
6.66e-16 and1.11e-16. No new GPU training was launched for this inventory.
