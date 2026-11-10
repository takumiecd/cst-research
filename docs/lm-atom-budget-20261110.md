# M8 Atom-budget search toward Dense: protocol

English (canonical) | [日本語](lm-atom-budget-20261110.ja.md)

Prepared,not a result. The owner hypothesizes Atom count rather than M16 alone
explains the task gap and authorizes an M8 budget search on the newly reserved
RTX4090. The unmerged M16 implementation PRs159–162 were closed;branches/results
remain preserved. Equal-budget M8/M16 observations remain evidence,not proof of
one exclusive cause. No new M16 run is selected.

## Fixed task

4 blocks,width512,heads8,context128,expansion4;all8 FFN maps CST/DST,
Dense attention/embeddings/tied vocabulary. Seed17,shuffle905,976 updates/999424
TRAIN targets,20M schedule,8192-target DEV. Unconnected M8,envelope tau.003,
fraction0,radius[.5,50],amplitude cap32;ordinary LR.0003,all CST fields32×,
AdamW(.9,.95),epsilon1e-8,decay0,clip1,FP32/TF32off. Same initialization policy;
mean initial radii can change with A because the envelope depends on amplitude.

Run fresh Dense and M8 c16 anchor on the actual4090/24GB/CC8.9. Prior3090 data
is historical context,not a paired speed baseline. Data manifest
5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01 is frozen;
confirmation/test remains untouched. Existing common training and checkpoint
bodies are reused unchanged with explicit factories. Ordinary initialization
matches verified prior controls. A private AtomBudgetConfig validates all
ordinary settings through ModelConfig and admits only this pilot's budgets;
the ordinary config guard stays unchanged.

## Prespecified adaptive pilot

A=c√(d_in d_out). Coarse c16→32→64→128 corresponds to16384→32768→65536→131072
Atoms per Linear. At each visited count qualify,train and audit CST plus dense-mask
DST at sparse LR.003/.01/.03. Match DST mask-one count to A,not4A;keep dense
weights+Boolean mask,no COO/CSR. Same masks/values across LRs within count;
row-balanced RigL-style updates250/500/750,actual DST Adam clock973 versus976.
Select minimum final DEV CE for DST,tie smaller LR;retain every candidate and
label DEV selection,including whether the winner is at the grid boundary.

The descriptive near-Dense pilot criterion is final DEV CE≤fresh Dense CE+.01
AND next-token Top1≥fresh Dense Top1−.005 (0.5 percentage points). This corresponds
to at most about1% PPL increase by the CE rule;it is not a statistical equivalence
or convergence test. Stop coarse expansion at the first pass. If it passes above
c16,also visit the single midpoint between the prior failing coarse count and
first passing count (c24/48/96). Select the smallest tested passing CST count.
No further bisection or claim of an Atom lower bound. If none passes throughc128,
report no qualifying count within this prespecified cap. Selection is post-curve
DEV exploration by design,not independent confirmation. Longer/multiple-seed
confirmation is a separate stage after candidate selection,not claimed here.

## Gates and evidence

Library scope: private unconnected M8 budget route from torchcst PR164,not public
selection or central fixture certification. Keep math,optimizer and old c8/c16
admission unchanged. Source comparison strips only the RADIUS_LM_BUDGET host
guard and verifies all other existing runtime files;common research modules
and train/audit bodies must remain byte-identical to the previous verified run.
Native retained/multiple-forward/noncontiguous tests precede learning. At each
stage require full-A FP64 up/down Y,dX,each raw-field maxabs≤3e-4 AND relativeL2≤1e-4,
retain old component-failure counts,exact serialized-next-update/observer checks,
finite FP32 moments/clocks,checkpoint/DEV reload and explicit mask-count audits.
No tolerance relaxation. Fail closed and retain the source/results on failure.

Record complete synchronized step time including four microbatches,clip and
optimizer/topology;exclude first10,input prep,evaluation,probes,checkpoints and
oracles. Allocated/reserved peaks are separate;reserved retains observer history.
Retain layer output/cotangent/feature signals,geometry/moments,raw checkpoints
at256/976 and four fixed TRAIN probe contexts at0/256/976. Keep every executed
stage and DST LR candidate,source/data/result hashes,all adaptive decisions and
failures. Frozen snapshots are immutable. GPU reservation remains running after
owned commands finish. No merge/default adoption or unmeasured speed claim.

Pre-freeze checks:library CPU2288 passed/2677 optional skips,research CPU241 passed/61 skips;Ruff/registry/diff checks pass. The first full research invocation omitted scripts from PYTHONPATH and had14 import errors; the corrected invocation passes without a code change.
