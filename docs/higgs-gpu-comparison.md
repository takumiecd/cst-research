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

## Completed observation

The adapter ran unchanged at cst-research commit
`19fb88d3e12b5a1be6403ccca3944214551794fe`, consuming the pinned torchcst commit
above. The foreground experiment completed, results were downloaded and the
local audit passed. The reservation/server profile was left unchanged; no GPU
work remains running from this experiment.

| Environment | Arm | Training seconds / 25,000 updates | Preparation-inclusive seconds | Peak allocated MiB | Peak reserved MiB |
|---|---|---:|---:|---:|---:|
| A100 MIG 3g.40gb | Dense | 12.252478 | 12.827648 | 116.188965 | 140 |
| Recorded L4 | Dense | 15.854979 | 16.600665 | 116.188965 | 140 |
| A100 MIG 3g.40gb | CST alpha0.001 | 17.282989 | 17.410841 | 38.834961 | 62 |
| Recorded L4 | CST alpha0.001 | 19.997565 | 20.180737 | 38.834961 | 62 |

Observed A100-environment training time was 22.72% lower for dense and 13.57%
lower for CST. Preparation-inclusive reductions were 22.73% and 13.73%.
CST/dense training-time ratios were 1.4106 on A100 MIG and 1.2613 on the recorded
L4. CST remains slower than dense in both environments. This is one ordered
pair per environment, with no independent repeat or hardware-only attribution.
The table retains the runner's timing boundaries; it is not full process wall
time. The successful A100 admission plus experiment took 283.26 seconds, excluding
dependency installation and transfer.

Actual A100 runtime: Python3.12.9, Torch2.11.0+cu126, CUDA12.6, Triton3.6.0,
NVCC12.6.85, driver570.86.15, Intel Xeon Gold6348 at2.60GHz. The recorded L4 used
Torch2.11.0+cu130/CUDA13.0 and Python3.13; its CPU model was not recorded. These
software/host differences remain part of the comparison.

Matching initial tensors and data order did not eliminate cross-environment
metric differences. Development CE/AUC after one epoch (A100 versus L4) were
0.580323/0.769598 versus0.578145/0.769254 for dense, and
0.569635/0.779743 versus0.567950/0.779926 for CST. These are development-only
observations, not a quality-equivalence or noninferiority result. Byte-parity
claims concern eager versus Graph within an environment, not final trajectories
across environments.

### Admission and independent local verification

- The unchanged GPU/stock/topology suites passed **136 tests**. Two legacy tests
  requiring frozen N1024/A52428/B4 MNIST inputs were skipped; they are not claimed
  as A100 validation.
- Full N2048/A4194 FP64 and 31-atom quadrature Y/dX/dP gates passed at the original
  `3e-5 + 3e-4*abs(reference)` bound. Worst absolute full-oracle errors were
  Y1.145e-9, dX1.173e-9 and dP9.693e-7.
- At both qualification epoch boundaries, dense's 25 and CST's 47 retained tensor
  fields were byte-identical between eager and Graph. Local verification also
  checked scalar training/development metrics, data orders and actual Adam
  clocks at16/32 updates.
- Both final pilots performed25,000 actual updates and24,992 Graph replays.
  Retained initial/best/final tensors, all gradients, Adam state/clocks, file and
  producer tensor hashes were inspected locally; all values were finite.
- Initial model tensors and the full epoch order were byte-identical to each
  arm's L4 reference. The prepared data and source identities also matched.
  Best-development logits retained the exact first50,000 validation labels;
  independent CPU recomputation of CE, accuracy and ROC-AUC agreed with the
  producer metrics. No confirmation or test evaluation occurred.
- A separate local Torch2.13 reader recomputed the independent FP64 and
  quadrature references from retained X/P/dY. Stored/reference differences were
  at most2.67e-15, and GPU errors still passed the unchanged bound. No reader
  clock/hash differences were observed.

The first attempt failed before the oracles or timing pilots: the server CUDA
include directory lacked `cusparse.h`. Its logs and failed result are retained.
A fresh second attempt used **command-local CPATH** entries pointing to the
matching job-local NVIDIA wheel headers, plus the virtual environment's bin
path for ninja. Both source trees, all tolerances and guards were unchanged;
no system headers or global packages were edited. The failed build is not
numerical evidence and has not been relabeled as a successful attempt.

### Retained provenance

External locator: `CST_RESEARCH_EVIDENCE_ROOT/higgs-gpu-comparison-20261105/`.
The evidence contains both attempts, setup/package/runtime records, both source
manifests, launch/header settings, build commands, raw oracles and checkpoints,
compiled primary radius extension, and the local audit/CSV. The source archive
and local audit program are preserved alongside the result archive.

| Artifact | SHA256 |
|---|---|
| Pinned torchcst source archive | `edbb585de56e304887b118069b7c7ecd9e45ae5fe2e08a5afca828ad8544eb5f` |
| Measured consumer adapter archive | `5266a6833c9c6132d82f2145de3fc4f5e8c199e40fb97d32cf43de0b6894d5a0` |
| Downloaded evidence archive,413,398,429 bytes | `59d8bfc25ab94871b1dd03e7e518291b15fd7b733cc523e51d657a2de41c88d5` |
| Outer transfer manifest | `32f0d3ddcbfcc4f7fcf50d031967d3dc8fd75086306ad09c9f7b79ab9334f057` |
| Retained/loaded primary radius extension | `1d124643bd6180409925b7eef39b3b35ad11fb5a8741e1991b6dec7db861f4a0` |

All83 outer-manifest files and24 raw PT files were retrieved and verified. The
L4 reference was independently checked against its receipt, all200 result and
1192 source-archive members, and the same pinned Git source. Other native guard
binaries were not retained, and allocator peaks do not measure total process GPU
memory. One epoch and seed do not establish final quality, an optimal alpha or
performance on a full A100 or other workloads.
