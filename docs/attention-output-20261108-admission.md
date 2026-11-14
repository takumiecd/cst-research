# Complete CST Attention and FFN: functional admission

**English** | [日本語](attention-output-20261108-admission.ja.md)

Functional admission passed at width512,4 pre-LayerNorm blocks,8 heads,FP32/TF32off,
with A=ceil(8√(d_in d_out)). This is not a convergence or quality-equivalence result.
Token embedding/tied vocabulary output and learned position embedding stay Dense.
Attention dot products, masking and softmax retain their usual operations.

| Model | CST projections | Full operator VJP | Causal isolation | Model update/resume |
| --- | ---: | --- | --- | --- |
| FFN | 8 | PASS | PASS | PASS |
| QKV | 12 | PASS | PASS | PASS |
| QKV+FFN | 20 | PASS | PASS | PASS |
| QKVO | 16 | PASS | PASS | PASS |
| QKVO+FFN | 24 | PASS | PASS | PASS |
| QKVO+FFN,stream64 | 24 | PASS | PASS | PASS |

The independent CPU FP64 full-Atom cases cover actual256-row square/up/down/O
operators. Maximum absolute errors across those cases: Y1.35e-6,dX2.27e-6,
all Atom gradients2.69e-5, passing unchanged componentwise atol3e-5/rtol3e-4.
Separate bounded Gauss quadrature checks profile mathematics. Manual masked
Attention matches Y/dX/QKVO gradients and future-token gradients are exactly zero.
All CST sites have finite nonzero gradients and change on a real2×128-token
model update. Serialized next task-update Parameter error is0; same-gradient
optimizer/moment/clock resume also matches exactly. These are bounded admission
updates, not the prospective976-update learning comparison.

Numeric torchcst `df24415afe442b9916e902d0f590fa5207016f66`, qualified consumer
`73751439a7643989adaab10889bdf6d547e67e3d`, RTX4090/Torch2.6.0+cu126.
The regression campaign passed223 scope/CUDA tests, zero skipped. Attempt03
archive SHA256 `e1ebb4914dd52adfb1c493a25c9c53b4ba453735812264a796641b51c0304103`:
27 compact originals verified and all10 owned prior-attempt PIDs ended. The
[compact admission JSON](attention-output-20261108-admission.json) records metrics.
Original sources, logs and negative attempts are preserved in the external
`attention-output-20261108-02`/`-03` evidence folders. The signed-center correction
and prior failure are described in the torchcst research record and PR155.

Attempt03's learner was refused before construction because its receipt omitted
the default closed-envelope powers. Consumer91ef5a6 emits all constructor settings
explicitly and adds a producer/admission regression. It leaves physical settings
and numerical gates unchanged. Attempt04 reruns consumer qualification with the
byte-identical, hash-verified operator source/build and regression evidence.
Five levels qualifications and their strict learner-admission preflights passed;
the last observed phase was stream64 qualification. Subsequent learning status is
unknown: execution ReadTimeout was followed by Hub HTTP503 while VPN remained
visibly connected. The durable controller/server was not stopped or restarted.
No learning curves, comparative CE/PPL, timing or memory result is claimed yet.
The study remains RUNNING with its supervision limit recorded.

Initial optimization PR154 and plan PR34 are merged. New implementation/evidence
remain in torchcst PR155 and cst-research PR35 pending complete execution/review.
