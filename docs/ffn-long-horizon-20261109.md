# Long-horizon FFN comparison

English (canonical) | [日本語](ffn-long-horizon-20261109.ja.md)

The short 976-step curves were still falling. Extend the fixed Dense, DST and CST
arms to 20M tokens to observe learning progress, without a new LR/tau search.
This is exploratory development evaluation at seed17, not an independent quality
confirmation. The RTX4090 gates must pass before training; preparation is not a result.

## Fixed design

Four blocks, width512, eight heads, context128, FFN expansion4. Replace all eight
FFN maps only; keep attention, embeddings and the tied vocabulary head Dense.
Use the existing hash-identified TinyStories token files, ordinary initialization,
seed17 and shuffled window order. Train fresh models for19,531 steps
(19,999,744 tokens), microbatch2 and accumulation4. Observe steps256,976,4882,9765
and19531; retain checkpoints and DEV8192-token CE every200 steps and at milestones.
Do not evaluate confirmation, calibrate RMS, add seeds, tune or stop by observed CE.

- Dense: ordinary peak LR0.0003.
- CST: public moment initialization, tau0.003, initial radius_fraction0,
  amplitude cap32, radius0.5..50, levels_wide_amplitude, A8192/map;
  all four atom fields32x LR. No output RMS matching.
- DST: row-balanced RigL-style dense-storage reference, K32768/map (4A),
  sparse peak LR0.003 fixed by the previous short DEV tuning. Ordinary LR0.0003.
  CST and DST each have272384 active FFN scalars including biases and residual scalars.

All use FP32, TF32 off, Adam betas(0.9,0.95), eps1e-8, weight decay0,
clip norm1, warmup200 and the same20M-token cosine horizon. DST has58 topology
steps250..14500, every250 until75% of19531. A topology step replaces the entire
Adam update; the final actual Adam clock is19473. The old short-run topology
horizon (events250,500) must not be resumed into this experiment.

## Execution and evidence

Run sequentially Dense, DST, CST on the same verified RTX4090. Freeze operator
and consumer archives before execution. Rerun the68 operator tests, CST model
qualification (including serialized optimizer continuation) and DST long-horizon
regular/topology/post-cutoff branch qualifications. Retain all raw failure evidence.
The data-free TrainingPlan checks freeze horizons, events, checkpoint counters
and prefix consistency. Record the shuffled-window hash for all three arms.

Report synchronized complete-step timing and allocated/reserved peak memory,
with preparation and wall time separately. Both runners retain returned logits
for the same lifetime between microbatches; the previous short DST runner used
an immediately released return value, so old memory/timing are not paired controls.
DST uses dense storage and matmul: it cannot establish sparse-kernel efficiency.
Evaluation/checkpoint/geometry overhead is outside the training-step timer;
CST diagnostics add wall time. Retain curves against tokens and elapsed time.

Audit all15 unique milestone checkpoints: optimizer counters and moments,
DEV reproduction, source/config/data identity, and independent FP64 output,
input-gradient and parameter-gradient oracles for trained CST/DST maps. Retain18
checkpoint filename hashes (final aliases included). Retrieve hash-verified compact
logs/receipts/source archives; leave bulky checkpoints on the GPU. Record actual
hardware and runtime, admission and limitations in a separate results note.
Evidence: CST_RESEARCH_EVIDENCE_ROOT/ffn-long-horizon-4090-20261109-01.

## Supplementary token accuracy

At the owner's request during Dense training, add teacher-forced next-token top1
accuracy on the same8192 DEV targets for all15 unique milestone checkpoints.
This evaluation is post hoc to source freeze and supplements the original CE and
perplexity; it does not change training, primary metrics, selection or confirmation.
Run its separately hash-identified evaluator after all training and trained audits,
so it cannot interfere with step timing. Verify its CE against the original logs.
Token top1 is not sentence exact match or free-generation quality. DEV is story-
disjoint from training, but8192 targets and one seed remain a small development pilot.
Report cumulative measured training time separately from elapsed wall time:
ordinary Dense/CST save checkpoints every200 steps, DST saves only milestones,
and CST also has geometry observers. These overheads make wall curves diagnostic,
not a controlled sparse arithmetic comparison.

## Execution and retrieval incident

The controller reported all3 training arms finished20M tokens and trained audits
passed; supplementary top1 evaluation had entered. Captured status outputs showed
rounded final DEV CE Dense2.053256, DST2.236804, CST2.301205. Before verified local
compact retrieval, the Hub stopped. Restarting the same4090 profile found no
original container-home job directory. Source archives/protocols remain local,
but detailed curves/checkpoints/accuracy evidence were not independently retrieved.
Status is STOPPED for incomplete evidence admission, not a failed-quality retry.
The known rounded values are status observations only. Do not infer top1 results
or claim completed verification. Fresh controls in the site-count study use
persistent NFS and periodic compact recovery; no old prefix is silently borrowed.
