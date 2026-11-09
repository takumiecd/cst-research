# Zero-start tau grid and a matched TinyStories DST reference

English (canonical) | [日本語](ffn-zero-start-tau-dst-20261109.ja.md)

The user requests lower tau exploration and DST on every subsequent comparison
plot. Previous DST evidence was HIGGS,not TinyStories; do not overlay different
tasks. Freeze tau={.0003,.001,.003,.01},initial fraction0,all CST field rates32×,
amplitude cap32,width512,four blocks,eight FFN maps8192Atoms each. Keep the
preceding zero-start protocol's seed17,token order,976steps/999424tokens,256step
prefix,data manifest,public moment initialization,no RMS calibration,AdamW
betas(.9,.95),eps1e-8,WD0,clip1,200warmup,20M-token LR horizon,baseLR3e-4,
FP32/TF32off and Dense attention/embedding/tied head. Fresh Dense and tau.003
controls accompany the new grid. Separate source-qualified jobs for CST and DST.

Port the prior HIGGS row-balanced RigL-style method(rectangular research consumer,
not the torchcst runtime). Match K=4A=32768 active FFN weight slots/map:16 per
up row and64 per down row. Biases/norms/outers are Dense. Start from the named
Dense LM initialization and independently seeded row masks; scale active initial
normal weights by1/sqrt(row density),preserving Dense target second moments.
This adapts the old HIGGS uniform initialization to the existing LM initializer.
Dense storage/matmul,masks and dense dormant gradients give no sparse-kernel
speed/memory claim. Slots include newly grown zeros; Tensor.numel is not the
active trainable budget. Active FFN scalar slots plus biases match CST272384.

Use global magnitude-threshold pruning with at least one old survivor per row,
row-balanced gradient regrowth with stable lower-column ties,zero grown weights,
and reset all pruned/grown Adam moments including immediate regrowth. Preserve
true survivor weights/moments. Cadence250,initial drop.3,cosine ending strictly
before floor(.75*976)=732:events250,500. As in the old reference,each topology
step replaces the all-parameter Adam update;976training steps/999424tokens but
974Adam updates. Growth uses the raw accumulated dense gradient before clipping;
ordinary updates mask inactive derivatives then globally clip. No Graph.

Tune only sparse-weight peak LR={.0003,.001,.003,.01},keeping ordinary peakLR3e-4.
All four candidates start with identical exact model/mask hashes,share seed/data
and full976-step budget. Fresh Dense control. Select the lowest final976 DEV CE,
ties lower LR; show all candidates separately and overlay the selected DST in
main Dense/CST curves. Label selection as development tuning,not a held-out
claim. No confirmation-set evaluation,extra seeds or automatic extended runs.
Future plots must retain the source-bound DST baseline and label its protocol.

Before training,CPU and actual GPU gates must cover independent rectangular
Y/dX/active-dW/raw growth-dW oracles,zero inactive weights/moments,exact row/global
budgets,stable ties,pruned/regrown reset and survivor preservation,skipped
all-parameter optimizer clock at events,serialized next regular/topology step,
full-model finite loss/gradients/shared ordinary initialization and source/data
identity. For CST reuse only byte/GPU/runtime-matched operator evidence and
fresh-qualify all four tau models including zero-boundary resume. Audit final
and256checkpoints and observer frames after training; preserve failures and
verified source/result/checkpoint hashes. Retrieve compact evidence and verify
owned processes inactive. No retuning failed arms or relaxed numeric gates.

Scoped complete-step timing includes transfers,accumulation,forward/backward,
clip,optimizer or topology work; probes/evaluation/observer/zero_grad/LR setup
are excluded. Keep topology timing separate as well as included in complete-step
samples. Record setup/wall and PyTorch allocated/reserved peaks. Sequential and
cross-job timings are exploratory; no sparse DST efficiency,convergence,
noninferiority,universal optimum or default adoption claim.
