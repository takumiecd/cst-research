# Positive-gain function-class clarification

English (canonical) | [日本語](positive-gain-equivalence.ja.md)

This is an analytic clarification of the already executed frozen69 pilot, not a
new training result or a change to its sealed source/protocol.

The actual model has three **bias-free** square CST maps, ReLU after each,
positive scalar gains applied immediately before those ReLUs, and an ordinary
trainable final affine map with no hard weight bound. There is no intervening
normalization, dropout or additive hidden term. Let

\[
h_0=\operatorname{ReLU}(W_{\rm in}x+b_{\rm in}),\quad
h_\ell=\operatorname{ReLU}(W_\ell h_{\ell-1}),\quad
\widetilde h_\ell=\operatorname{ReLU}(g_\ell W_\ell\widetilde h_{\ell-1}),
\qquad g_\ell>0.
\]

Positive homogeneity implies, by induction,

\[
\widetilde h_\ell=\Bigl(\prod_{j=1}^{\ell}g_j\Bigr)h_\ell,
\quad
F_g(x)=W_{\rm out}\widetilde h_3+b_{\rm out}
=\bigl(GW_{\rm out}\bigr)h_3+b_{\rm out},
\quad G=g_1g_2g_3>0.
\]

Therefore the invertible parameter map is
\(W'_{\rm out}=GW_{\rm out}\), with all other parameters and the output bias
unchanged; its inverse divides that weight by G. **For fixed positive gains, the
global ideal-real-valued function class is the same as stock.** Per-site effective
amplitude bounds are rescaled, but an unrestricted output Dense absorbs the
product, so the global function class does not expand. For a fixed parameter
tuple, the ideal forward depends on the gains only through their product; this
does not establish a unique benefit from distributing RMS matching across sites.

This does not make the intervention initialization-only. With identical saved
trainable tensors, gain and stock generally start at different functions.
The parameter coordinates, gradient scales and adaptive-update/epsilon behavior
differ. Effective regularization expressed in the transformed output coordinates
also needs care; in particular, the multiplicative part of decoupled weight decay
itself commutes with this constant linear rescaling, which alone does not restore
the complete Adam trajectory. Finite FP32 arithmetic, overflow/underflow and
intermediate activation scales need not obey an exact bitwise commuting identity.

The actual gains are 521.2747192382812, 258.95751953125 and
945.0846557617188; G≈127575095.34510244. `analysis.json` binds both actual saved
initializations, passed raw audit03 and fully verified69/975 source checkouts.
A bounded public CPU-FP64 comparator on two fixed **synthetic** 4×28 inputs
(linspace and zeros) gave maximum absolute differences 8.67×10⁻¹⁹ and 0 against
stock with the transformed output weight. No dataset, labels, task evaluation,
GPU operation or new qualification gate was used.

The observed partial rescue at the same global ideal function class is evidence
that the stock AUC=.5 stall cannot be explained solely by an absolute lack of
representable functions or by its unchanged initial zero-column pattern. It
does not isolate conditioning from initialization, optimizer coordinates,
regularization or finite-precision effects, and it does not resolve the remaining
gap to Dense. Existing three-epoch/seed17 exploratory limits and all preserved
failed amplified componentwise diagnostics remain in force. No extra arm,
training, seed or density is implied.

Under the [external evidence convention](reproducibility.md), the analytic report is
`higgs-depth-gain-20261106/stats-positive-gain-equivalence-01/analysis.json`,
SHA256 `a90fa2323c706095b4975e8b81bc29c745322ae13a56fbb467dd8c8f8c05f5c7`.
Its reader SHA is `9d8d201eece3618feb1a89f15ea143a111637569d3f65dafe11df774fe5653b3`.
The measured consumer69/external975 and prior raw-audit identities are unchanged;
this later analytic note is separate from their training qualification.
