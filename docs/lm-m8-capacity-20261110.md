# M8-only Atom capacity follow-up

English (canonical) | [日本語](lm-m8-capacity-20261110.ja.md)

Status: prepared; no new learning result. The owner requests ordinary unconnected
M8 with doubled Atom count, to distinguish a capacity effect from M16 geometry.
Add only one fresh all-FFN arm: c16, A16384/map, versus the verified c8 M8
A8192/map control from `lm-geometry-epsilon-capacity-3090-20261110-02`.
Keep all eight FFN maps CST in the same four-block width512/expansion4 LM;
attention, embedding and tied vocabulary remain Dense. This is not a lower-bound
or depth-scaling claim. Doubled A retains A=c√(d_in d_out).

Fix ordinary LR.0003, all four raw Atom field LRs32×, AdamW(.9,.95), all epsilon
1e-8, decay0 and clip1. Unconnected M8, tau.003, fraction0, radius[.5,50], amplitude
cap32, named variance-scaling initialization, seed17, shuffle905 and FP32/TF32off
remain fixed. Larger-A initialization keeps the same variance target; it does
not duplicate old atoms. Verify all ordinary initial weights byte-match the old
M8, c8 raw initialization matches the old M8, and c16 raw initialization matches
the old same-c16 M16 family. The latter checks raw bytes, not function equality.

Use the same TinyStories GPT2 data manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`,
976 updates/999424 targets,20M schedule horizon, four two-context microbatches,
context128, DEV8192 targets, confirmation unused. Keep evaluations0/200/256/400/
600/800/976 and four fixed TRAIN probes at0/256/976. Retain radius/fraction,
per-layer output/cotangent/field-gradient and moment-scale records, checkpoints,
synchronized complete-step timing and allocated/reserved peaks (first10 omitted).

The prior same-GPU controls are explicit read-only inputs: Dense, DST-c8/c16,
legacy-M8-c8 and M16-c8/c16 at epsilon1e-8. Verify manifests/audits/source/data,
ordinary/raw initialization and active scalar budgets before reuse. DST has4A
active weights/map and the previously tuned/transferred sparse LR.003, no new
c16 tuning. DST uses dense storage. Label old controls and new run separately;
cross-job timing is exploratory, not a new paired performance claim. Require
explicit `--dst-run` inputs in plots. Do not rerun already completed controls.

Runtime source7a7dc11689d6905f905965ecdac3b41a839548e8 ([PR163](https://github.com/takumiecd/torchcst/pull/163))
adds only an explicit c16 metadata/envelope adapter and benchmark registration.
All preexisting runtime source files, including CUDA, remain byte-identical to
95f647f. The old c8 gate is unchanged. The research training loop admits explicit
model/optimizer factories; step/data/probe math remains unchanged. Freeze both
sources before execution and verify their hashes afterward.

Before learning: actual3090/24GB/CC8.6 identity,96 relevant native/regression
checks including c16 retained/multiple forwards, full-A initial up/down FP64
reference Y/dX/each raw field (maxabs≤3e-4 AND relativeL2≤1e-4 separately; retain
older componentwise violation counts; no relaxation), exact serialized next
update/observer equality, epsilon/LR/raw/ordinary/budget checks. Repeat full-A
oracles at the final state, require finite FP32 moments/clock976, audit256/976
checkpoints and reproduce final DEV. Leave failed evidence unchanged.

Entry: `cst_research.experiments.lm_m8_capacity_run` qualify/train/audit with exact
data SHA, `--previous-run`, qualification receipt and unused output directories.
A single-seed short improvement supports a capacity effect in this protocol;
it does not uniquely establish why stacking CST is worse or prove sufficient
Atoms. Lack of short improvement also does not establish a capacity ceiling.

## Stopped attempt and revised primary criterion

The initial attempt failed before learning at the old C++ c8 guard (8 failures,88 passes). Failed compact f99dd0bf295ee5533003b351d88add72f244ba984dbe62718d47a9a5396a1bfd is retained with5 verified payloads. The owner subsequently specified DST mask-one count=Atom count, keeping dense-mask storage and no COO/CSR conversion. Continue under [the active-count protocol](lm-m8-active-count-20261110.md); prior4A DST is auxiliary, not the primary comparison.
