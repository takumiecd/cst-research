# Polar radius loss-gradient pair on RTX6000 Ada

English (canonical) | [日本語](ffn-polar-radius-gradient-20261108.ja.md)

Prospective bounded comparison: preserve the existing amplitude-dependent L/U,
q=clamp((s²+t²−1)/3,0,1) and logarithmic radius value. Compare `polar_detached`
against `polar_live`, under the same Cartesian AdamW and post-step projection to
1<=norm(s,t)<=2 (FP32 storage tolerance5e-7). Radial proposals are allowed;
there is no tangent removal, activity clock, radial regularizer or new LR search.
Centers remain periodic through the operator. A singular/nonfinite radial
proposal aborts. Adam moments remain Cartesian and are not transported by the
projection. Thus the detached arm's q can also move after elementwise Adam;
it is not a frozen-q control.

The live chain rule includes both amplitude-dependent L/U slopes and the q
term: grad_st L = g_a grad a + g_r grad r. q0/q1 use the inward interior slope.
An explicit straight-through value correction preserves the existing exact
forward envelope while defining that constrained endpoint Jacobian. This is
not a claim of an unconstrained two-sided derivative at a clamp corner.
The detached arm stops the entire radius path, including its amplitude terms.

Use the SAME realized initial s/t/centers and ordinary named Parameters:
width512,4pre-norm blocks,8heads,context128,FFN expansion4,8CST maps,
A=8 sqrt(512*2048)=8192 per map. Attention, embedding and tied vocabulary head
stay Dense. Minimum-based rational-zero-opening scale.001, birth.5, q0,
amplitude maximum1. Native operator88c2e8a7c4aa312f4f096c5169d712ace49ac979
is frozen unchanged. A research wrapper owns the original Polar atoms,
computes a physical Tensor, and calls public LinearBinding/Dispatcher on the
already qualified physical Plan. No runtime implementation is copied into the
research package and no public Polar default/backward changes.

AdamW baseLR.0003,Atom LR×16,betas(.9,.95),eps1e-8,decay0,globalclip1,
FP32/TF32off; same200-update warmup and20M-token LR/data horizon. Seed17,
batch2 x128 x accumulation4,976updates/999424tokens each, fixed order detached
then live on one actual RTX6000 Ada48GB. Token manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
Use only8192 DEV tokens; no confirmation/generation/HPO. Historical Polar and
independent-radius results are labeled context, not primary same-update controls.
This does not estimate convergence, noninferiority, best LR/radius or Atom order.

Before learning: CPU algebra/central differences at interior points, independent
Gauss quadrature of composed kernels, CPU/GPU map parity including both endpoint
slopes, all8192-Atom/all-four-field Y/dX/dP at256 noncontiguous rows in both FFN
directions, retained backward, exact paired initialization, all-site task updates
and serialized optimizer resume. Repeated32 full backwards without forced GC
must have allocated spread below128MiB to catch retained functional bindings.
Reuse the previous231 regression record/native build only after verifying full
operator source bytes and actual GPU/Torch/CUDA identities. Run fresh composition
and model qualifications; do not relabel an old admission.

Record55 fixed probes (step0,1..5,every20,976), outside update timers:
amplitude,radius,L/U,q and latent norm statistics/boundary occupancy, physical
partial dLoss/dr before the map, propagated radius contribution (0 in detached),
latent field gradients and their radial component. Physical partials need not
be0 in the detached arm; they are stopped upstream. These are unscaled/unclipped
fixed-probe gradients, not training-accumulation gradients or coordinate speed.
Overlay learning/radius/q/radial-gradient curves. Measure synchronized complete
update time and allocated/reserved training peaks, separating diagnostic/setup
wall time. Functional-binding/Python-map overhead is part of these observations;
this is not a new optimized kernel performance claim.

After976 updates, independently restore checkpoints, recompute DEV exactly,
check every trained wrapper's CPU-reference VJPs, final probe/checkpoint parity,
annulus feasibility and next-task parameter/Adam-state equality. Retrieve and
hash-verify compact evidence, verify owned workers inactive, then mark COMPLETED.
Preserve data/checkpoints/native builds and failed frozen attempts. The current
public Polar remains unchanged; adoption requires a separate decision.

Runner: `cst_research.experiments.ffn_polar_radius_qualification`, then existing
`cst_research.experiments.ffn_run --kind cst --attention-kind dense --width 512
--heads 8 --linear-backend levels --radius-mode {polar_detached,polar_live}
--track-radius --development-only` with explicit pinned data/admission hashes
and all settings above (CLI flags use spaces, e.g. `--width 512 --heads 8`).
