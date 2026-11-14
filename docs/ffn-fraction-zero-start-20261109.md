# Initial fraction0 versus.5 at common CST field LR32×

English (canonical) | [日本語](ffn-fraction-zero-start-20261109.ja.md)

The user authorized starting fraction at0 with common32× CST field rates. Compare
fresh fraction0,fraction.5 and Dense in one RTX6000 Ada job. Do not change Atom
count,tau,amplitude cap,token order,training budget or non-Atom LR. Older curves
are context; repeat.5 and Dense as controls for this source/instrumentation.

TinyStories,width512,four pre-norm blocks,8heads,context128,expansion4. All8 FFN
maps use8192 Atoms(A=8√(d_in d_out)); attention,embedding and tied head are Dense.
Use tau=.003,amplitude cap32,radius[.5,50],M8,levels_wide_amplitude,public moment
initialization. Initial fraction is0 or.5. Same named seed and moment targets
(up variance.0004/down.00005); amplitudes are initialized for each fraction by
that public initializer. Initial Parameter digests therefore need not match
across fractions; record each and require qualification/training digest parity.
Center draws and data order are held fixed. No probe/output-RMS calibration.

CST amplitude,input/output centers and fraction all use32× the base LR,peak.0096.
Dense parameters,FFN bias,norms and all other non-Atom fields keep baseLR3e-4.
AdamW betas(.9,.95),eps1e-8,decay0,global clip1,200update warmup,common20M-token
schedule horizon. Seed17,batch2×128×accumulation4,976updates/999424tokens each;
observe/evaluate/checkpoint at256updates/262144tokens. Same data manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
DEV8192 targets only; no confirmation. FP32 Parameters,moments and physical rows,
FP64 map intermediates,TF32off,eager execution.

Reuse operator68-check evidence only after exact bytes/GPU/runtime/hash parity.
Fresh consumer qualification for both fractions includes all map VJPs,
outstanding forwards,actual fraction updates and exact serialized next step.
Extend the wide-amplitude resume gate to include initialfraction0 before freezing.
Do not interpret one-sided projection at0 as automatic vanishing gradients.

Record learning curves,initial/final activation RMS,fixed-probe field gradients,
amplitude-binned radius/fraction,per-layer distributions and actual per-Atom
fraction update/net/path statistics. Initial0 can only move inward; endpoint
occupancy alone does not distinguish legitimate lower-bound solutions from
blocked useful updates. Observer remains read-only and outside update timing.

Audit all6 final/prefix checkpoints: DEV difference<=1e-6,finite FP32 Adam moments,
256/976 counters,bounds and8 trained up/down encoded-row FP64 Y/dX/all-Atom
oracles (2CST×2checkpoints×2maps),unchanged atol1e-4/rtol4e-3. Independently
reproduce4 observer net-change frames. Preserve failures/source/checkpoint/result
hashes,retrieve verified compact evidence,retain checkpoints remotely and verify
owned PIDs inactive. No retuning/restarting failed arms or automatic extra sweep.

Timing includes accumulation/transfers/forward/backward/clip/optimizer,excluding
zero_grad/LR setup/probe/evaluation/observer. Report setup/wall and PyTorch
allocated/reserved peaks; sequential same-job timing remains exploratory without
order reversal. No process-wide/Graph memory or public-default claim. One-seed
initialization comparison does not establish convergence,noninferiority,optimal
fraction,capacity or a unique mechanism for the Dense gap.
