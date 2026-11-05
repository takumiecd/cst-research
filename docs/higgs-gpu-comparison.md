# Bounded HIGGS A100 MIG environment comparison

English (canonical) | [日本語](higgs-gpu-comparison.ja.md)

This separate operational study compares a reserved **NVIDIA A100 80GB PCIe
MIG 3g.40gb** environment with earlier L4 budget pilots. It does not modify or
extend the running HIGGS alpha-confirmation family. The A100 partition exposes
42 SMs, 42,144,366,592 bytes and compute capability 8.0; it is not a full A100.
No DST experiments, density grid, hyperparameter tuning or held-out evaluation
are authorized by this protocol.

## Source and workload boundary

The adapter is `cst_research.experiments.higgs_gpu_comparison`. It consumes an
external, unchanged `git archive` of torchcst commit
`97580582ed28c4e494bf5c2656bbe94a85c544c3`, including its private HIGGS Graph
benchmark adapter. No torchcst runtime is copied into this repository or
represented as a supported public training API. Freeze this repository's
adapter commit and both source manifests before execution.

Both arms use the original `benchmarks.training.higgs_confirmation` runner:

- N2048, a 28→2048→2048→2 ReLU network, all layers trainable, bias-free middle.
- Dense middle versus CST middle with divisor 1000, alpha=0.001 and exactly
  floor(2048²/1000)=4194 atoms, each with four parameters.
- Seed17, LR0.001, FP32, TF32 off, CPU math threads1, true batch4 and the original
  AdamW/Polar update laws and initialization streams.
- One full epoch: 100,000 training examples, 25,000 actual optimizer updates.
  Development uses only validation indices [0,50000). Neither confirmation
  [50000,100000) nor the official test set is evaluated.
- Prepared archive SHA256:
  `62d32a1789ed737a5997ca6a0630548e6211af86c9cd6a0a605cd057d8bd32ac`.

The earlier reference is L4 qualification job
`l4job-0ac0ee0f5deb442eb09b8d63a18d1f4a`, specifically its
`budget-pilot-2048-dense` and `budget-pilot-2048-cst` cells. Compare their actual
source/data/order/runtime records, not an unrelated HPO curve.

## Runtime and resource policy

Use only the bundled **kyutech-a100** skill client for status, foreground
execution, upload and download. Use an isolated directory under the Jupyter
contents path `srv11/cst-lab/higgs-a100-20261105-*`. The execution kernel and
contents API may have different relative roots; verify the mapping before
extracting or executing. Never inspect/copy the token, change the server
profile, mutate global packages, touch unrelated jobs or substitute a different
GPU.

The initial server stack is Python3.12.9, Torch2.6.0+cu126, CUDA/NVCC12.6 and
driver570.86.15 on Intel Xeon Gold6348. An official Torch2.11.0+cu126 CPython3.12
wheel is available. Install dependencies only into the job's isolated virtual
environment. Do not install cu130 or change the driver to imitate the L4 stack.
Record the actual final Torch, Triton, Python, CUDA, NVCC, driver, CPU and device
properties. Even with Torch2.11 on both devices, CUDA/compiler/driver/Python/CPU
can differ. These are **environment comparisons**, not hardware-only causal
measurements. Preserve failures and never bypass stock-function, finite,
metadata, lifecycle or capture guards to obtain a timing.

## Admission before timing

The adapter verifies all source-file bytes and the prepared-data hash, then
requires the exact reserved MIG identity. Before either pilot starts:

1. Run the pinned HIGGS cohort, stock qualification and topology-guard tests
   unchanged. These include finite/bootstrap and capture-contract checks.
2. At N2048/A4194, compare Y, dX and every atom-parameter gradient against the
   independent public CPU FP64 local Fourier reference. Use the original error
   criteria, without relaxed tolerances.
3. Compare a retained 31-atom subset against independent 48-point quadrature,
   including Y, dX and all subset-parameter gradients.
4. For dense and CST, run eager and Graph trajectories for two epochs of
   16 minibatches, with dev32 available only in this qualification mode. Compare
   actual saved model, optimizer, gradient and clock tensors at both epoch
   boundaries, require byte equality, 32 real updates and five matching Adam
   clocks. This exercises capture and the evaluation/train boundary.

Save the raw oracle and checkpoint tensors, logs, comparisons and a passed
qualification record before starting timing. Any failure ends the experiment;
no automatic fallback is a qualifying result. Qualification costs are distinct
from the two pilots' timing.

## Execution and evidence

The reserved server must already be running. The resource skill, not this
adapter, owns connectivity and transfer. Example after deployment into isolated
source, input and result directories:

```bash
PYTHONPATH=RESEARCH_SOURCE/src JOB_ENV/bin/python \
  -m cst_research.experiments.higgs_gpu_comparison \
  --torchcst-source TORCHCST_SOURCE \
  --torchcst-manifest TORCHCST_SOURCE_MANIFEST.json \
  --adapter-manifest ADAPTER_SOURCE_MANIFEST.json \
  --adapter-commit FULL_RESEARCH_COMMIT \
  --data INPUTS/screen.npz --output FRESH_RESULTS
```

Each manifest contains `source_commit`, `archive_sha256` and a `files` mapping
from safe relative paths to SHA256 hashes. Archive the source from explicit Git
commits, independently verify archive and extracted-file hashes, and retain the
archives/manifests with the evidence. The adapter rejects unexpected source
files and rechecks both trees after execution. CUDA compilation uses a job-local
extension cache, MAX_JOBS2 and architecture8.0. Every process remains foreground
and bounded; no services or detached jobs are created.

`experiment.json` records incremental progress, runtime, numerical gates and both
pilot results. `manifest.json` binds all result files. Retain raw initial,
best-development and final checkpoints, predictions, source/result archives,
package/setup logs and manifests in ignored external evidence. Download and
verify hashes and actual tensors locally before reporting completion. Record
that compiled-extension bytes are independently verified only if retained.

Compare synchronized complete training time, preparation-inclusive time and
capture-inclusive allocated/reserved peaks. The original runner excludes
validation, checkpoint serialization, hash diagnostics and data/model placement
from training time; preserve those boundaries. Report the two same-environment
arms together, and retain the earlier L4 job's provenance. One epoch and one
seed are a bounded operational comparison, not a final accuracy or broad
hardware-scaling result.
