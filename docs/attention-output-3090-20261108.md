# Complete CST Attention/FFN pilot on RTX 3090

**English** | [日本語](attention-output-3090-20261108.ja.md)

The owner obtained an RTX 3090 after the Kyutech Hub recovered and a4090 was
unavailable. Read-only inspection confirmed the approved3090 profile, actual
NVIDIA GeForce RTX3090,24GB-class memory,compute capability8.6,Torch2.6.0+cu126
and CUDA12.6. The earlier4090 job directory was not visible on this server;
its completion remains unknown. Do not overwrite or relabel that experiment.

Use a new isolated directory and frozen operator
`b393ac653f3b3893c378acda1c1752919c1539d1`. Consumer changes only admit the
explicit3090 GPU identity and register this cohort; freeze its full source
before deployment. Do not change model mathematics, tolerances or default GPU
selection. Use a job-local dependency environment and native build directory,
CUDA architecture8.6,one experiment at a time, and the existing durable launcher,
qualification,learner and checkpoint-audit mechanisms.

Keep the same width512,4 pre-LN blocks,8 heads,context128,expansion4,batch2,
accumulation4,seed17,c8,minimum-001/q0,FP32/TF32off,AdamW baseLR3e-4 with whole
Atom group multiplier16,betas(.9,.95),decay0,clip1. Per-map Atom count is
ceil(8sqrt(d_in*d_out)):4096 for square maps and8192 for FFN maps. Public
variance scaling targets std0.02 for Q/K/V/up and0.02/sqrt(8) for O/down.
Use the frozen TinyStories/GPT2 manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`,
train/DEV only,no confirmation,generation,HPO or minimum-Atom search.

First run the223 regressions and all five levels model qualifications plus
the24-site levels_stream64 qualification. Require actual R256 independent CPU
FP64 Y/dX/all-Atom VJPs,Gauss quadrature,manual causal Attention with zero future
gradients,all-site updates and serialized next-update/optimizer-state checks.
Then run fresh Dense,FFN,QKV,QKV+FFN,QKVO,QKVO+FFN (0/8/12/20/16/24 CST sites),
each976updates/999424tokens with a20M-token data/LR horizon. Token/position
embeddings and tied vocabulary output remain Dense. Each final checkpoint needs
exact DEV recomputation,every trained CST map's bounded-row FP64 VJPs and two
independent restores with identical next task update and optimizer state.
Cache generations follow their explicit invalidation/load contract.

Retrieve and hash-verify compact logs,curves,geometry and audit evidence. Keep
large datasets,checkpoints,source and native builds in the named cluster
directory; report exactly which payloads were downloaded. Preserve partial or
failed results without automatic retries. Do not restart/stop unrelated work.
Overlay all learning curves and record update samples plus allocated/reserved
peaks. The existing learner timer excludes zero_grad/LR assignment,setup,
evaluation,probes and checkpoint IO; it includes transfer,four forward/backwards,
clipping,AdamW and synchronization. Wall time includes the excluded work.
Allocator peaks are not total process GPU memory. One seed/~1M tokens is a
development pilot,not convergence or noninferiority. Keep this3090 cohort and
its timing separate from the L4 isolated-Linear benchmark and all4090 results.
