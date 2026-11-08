# Complete Attention/FFN pilot on Colab L4

**English** | [日本語](attention-output-l4-20261108.ja.md)

The owner authorized Colab L4 after the Kyutech4090 server became unavailable.
Do not restart or duplicate that unknown remote job on its original server.
This is a fresh, separately registered L4 cohort, not a continuation/relabeling
of4090 observations. Use the host-wide shared pool, one worker, exact NVIDIA L4
identity/account verification, isolated source and verified download/VM cleanup.

Keep torchcst numeric source `df24415afe442b9916e902d0f590fa5207016f66`, including
the signed FP32 periodic-center fix. Consumer changes only add explicit L4
identity selection and record this protocol; pin its full commit/bytes before
submission. Use the installed Colab Torch/CUDA/Python runtime in a job-local
venv with inherited system packages, record actual versions and require training
receipts to match the L4 qualification runtime. Do not pool GPU/runtime medians.

The model/data/recipe/seed/budget match the fixed width512 protocol: four pre-LN
blocks,eight heads,context128,FFN expansion4,batch2,accumulation4,seed17,c8,
minimum-001/q0,FP32/TF32off,baseLR3e-4 with whole Atom group×16,Adam(.9,.95),
decay0,clip1. A=ceil(8√(d_in*d_out)); square maps4096Atoms and FFN maps8192.
Use the frozen TinyStories/GPT2 token archive and manifest
`5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01`.
Train/DEV only; no confirmation scoring, generation, HPO or lower-bound search.
Requested1M tokens means976updates/999424actual tokens, with20M-token data/LR
horizon. Compare fresh Dense,FFN,QKV,QKV+FFN,QKVO,QKVO+FFN (0/8/12/20/16/24 CST
sites). Token embedding/tied vocabulary and learned position embedding remain
Dense; masking/softmax/SDPA and LayerNorm retain their original operations.

Before any learner submission, the L4 qualification job runs the existing223
scope/CUDA regression cases, then all five levels model admissions and24-site
stream64 admission: independent CPU FP64 Y/dX/all-Atom VJPs on actual shapes,
Gauss quadrature, causal/manual Attention including O, all-site real update and
serialized next update, and same-gradient optimizer/moment/clock checks.
Require full model/backend/shape/kernel-default/source/runtime admission with
unchanged tolerances. A failed/OOM/nonfinite/timeout attempt is retained and
stopped; no automatic retry, weaker gate or hardware fallback.

After that job succeeds and is hash-verified, submit six separate whole-arm jobs
through the same one-worker queue. Each has fresh source, compilation, initialization,
its own exact qualification artifacts,976 training updates and independent trained
checkpoint audit (DEV exact, every CST map's bounded-row FP64 VJP, two independent
restores with identical next update/state). Record actual GPU UUID/runtime per job;
separate jobs/host load make timings exploratory even if the worker reuses one VM.
Keep all checkpoint payloads in the verified pool result archive. Materialize any
hardlinks byte-for-byte before transport so the archive contains regular files;
record original and final hashes. Do not discard the final or milestone checkpoint.
Large raw datasets and consumer source are transferred as hashed, necessary input
archives embedded in an ignored external driver, not committed to torchcst.

Overlay train/DEV CE/PPL versus actual tokens and wall time, and retain every
post-warmup update sample and allocated/reserved peak. Preserve the existing
learner clock: after zero_grad/LR assignment/initial sync, through transfer,
four forward/backwards, clipping,optimizer and final sync. Setup, evaluation,
probes and checkpoint IO are outside that timer and included in wall time.
Allocator peaks are not total process GPU memory. Pool duration is not throughput.
One seed/~1M tokens establish development behavior, not convergence, equivalence,
noninferiority, minimum Atom count, cross-GPU bit identity or universal speed.
