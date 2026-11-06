# HIGGS linear Atom-count deep pilot

English (canonical) | [日本語](higgs-linear-atom-scaling.ja.md)

**PREPARED, not executed.** The first delivery is Stage A: new native qualification
of all sixteen configurations, with no task training. The intended Stage B is a
separate reviewed one-epoch exploratory learning consumer. Neither registration
nor CPU preflight is GPU admission or a model-quality result.

## Architecture and question

Use trainable Dense input/output maps and `28 → N × H → 2`, with H=4/8 hidden
layers, H ReLUs and L=H−1=3/7 bias-free square maps. Compare Dense-stock with all
L square maps replaced by CST at **exact A=cN per map**, c=1/4/16. Initial pilot
widths are N=1024/2048, seed17; these are 4 Dense controls and 12 CST cases.
Widths4096 and later8192 are reserved for separate fixed-c validation.

\[
P_{\mathrm{Dense}}=LN^2+31N+2,\qquad
P_{\mathrm{CST}}=4LA+31N+2=(4cL+31)N+2.
\]

This counts learned scalars, linear in N only at fixed c **and** L. The L unit-gain
buffers are nontrainable. The algebra does not demonstrate linear runtime, peak
memory, or Dense-quality accuracy. Failure up to c16 would not refute existence
of a sufficient linear Atom budget with another constant or training recipe.

## Prespecified recipe and preparation history

Main recipe: **normalization NONE, gains1, no calibration**, B4, FP32, TF32 off,
eager ordinary complete steps, stock q1 CSTAdamW, LR0.0003. Ordinary weight decay
and amplitude weight decay are0.01; raw Atom weight decay0, betas(0.9,0.999),
Adam epsilon1e−8. Keep the unchanged public975 radius-cell-mass helper, Chebyshev
modes8, merge factor2, w_c0.3, spacing1 and
`GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C`. No coordinate-LR adapter,
Graph qualification/default changes, norm fallback, or hidden initialization change.

Input/output seeds are seed+1/+2, internal Dense seed+100+j and CST seed+1000+j.
Ordinary initialization is the legacy `torch.nn.Linear` default (internal uniform
±1/√N, not a He-ReLU adjustment). Public CST amplitude initialization has standard
deviation0.1/√A and balanced-output/uniform-input centers. Increasing c therefore
partly offsets initial variance through amplitude scaling. Deep signal/gradient
contraction can reflect initialization and conditioning; it is not a capacity proof.
H8 Dense adequacy remains unmeasured; a collapsed Dense control is not evidence
of CST success. Exact old H4/N1024 parity is checked at A104/1048 using rational
c13/128 and131/128, without changing the old factory.

The tentative LN20-case preparation was superseded **before any new GPU result**
by the user's normalization-NONE correction. LN is only a possible separate later
diagnostic with a matched Dense control. Existing nine-case learning evidence
`eefdef7e8b7598232c137a6f8589c74f1a511051d13bcbd8fe0b5a9a01dc8e84`
is recipe-selection context, not new fixed-c evidence. Old30 numerical modules
and10 study registries remain byte-identical. Historical preparation snapshots
remain in ignored evidence.

## Stage A: new native admission, not task training

Each actual N/H/c configuration must pass full-network Y, dX and every owned
parameter gradient against an independent CPU FP64 reference. It traverses
Dense/ReLU explicitly in reverse and calls unchanged975 chunked CST reference
with one256-Atom graph at a time. It uses every actual Atom and full width;
no full-network autograd graph or `[A,N,N]` tensor is materialized. Tiny public
models cross-check this manual reverse against independent full autograd.
Actual-largest CPU preflight RSS/time is retained externally; reference array
estimates alone are not RSS or a GPU efficiency claim.

Normal cotangent1 and amplified cotangent1e24 were chosen from the sixteen
**synthetic CPU** reports before GPU execution. The decision SHA is
`b86e9d69f211768239aaeda30b76863d16519f1bb13ee27ce0a25b30f79eeb31`.
Amplified gradients are **never applied to an optimizer**.

- Normal elementwise: |error|≤3e−5+3e−4|reference|, mandatory.
- Both scales: each gradient field and each packed Atom column must have relative
  L2≤3e−4 and max-error/reference-max≤3e−4. Norms are normalized in FP64;
  an exactly-zero reference requires exactly-zero actual values.
- Amplified dX and each packed column require some |reference| exceeding
  3e−5+3e−4|reference| and some nonzero actual value. Preserve amplified
  componentwise failures as diagnostics; report normal FP32 rounded-zero tails.
- Independently bridge each N/c with unchanged48-point polynomial quadrature on
  31 actual initialized Atoms, checking full Y/dX/all packed gradients. Sharing
  across depths requires identical first31 Atom bytes and retained raw SHA.
- Fresh synthetic32-input normal-CE trajectories must match eight uninterrupted
  updates against four plus native save/load plus four. Retain actual inputs,
  labels and eight explicit B4 orders, initial0, normal4/8 and resumed8 states,
  every gradient, m/v/clock, model-side optimizer store and scalar-Adam evidence.

There are H+3 owned parameter tensors (7/11), and exact3/7 actual CST owners at
indices2/4/6 and2/4/6/8/10/12/14. Native load alone may advance their generation
by `max(before+1,serialized)`; all other state bytes/metadata remain literal.
Subsequent checks use the actual post-load baseline, without rewriting counters.
The proof records synthetic update scope, not learning statistics. Stage A never
decodes the task NPZ or evaluates development/heldout data.

## Source, data, execution and evidence

External public runtime: `97580582ed28c4e494bf5c2656bbe94a85c544c3`;
manifest `17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`;
archive `edbb585de56e304887b118069b7c7ecd9e45ae5fe2e08a5afca828ad8544eb5f`.
Use the existing train100k/development50k preparation, artifact
`58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`
and manifest `530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`.
Development has already informed exploratory recipes; it is not a new formal
confirmation sample. No future confirmation, original confirmation or official
test payload is staged. Historical aggregate feature exposure remains disclosed
in the unchanged [data protocol](higgs-depth-position.md).

Stage A requires exact eight regular `research-inputs/` files:
`consumer-source-manifest.json`, `original-source-manifest.json`,
`original-source.tar.gz`, `train-development.npz`, `data-manifest.json`,
`runtime-pin.json`, `recipe-selection-audit.json`, `cotangent-decision.json`.
Every byte hash and exact input inventory is checked before the worker.
The exact RTX4090 runtime pin is
`1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b`:
Torch2.11.0+cu126/CUDA12.6/Triton3.6.0, Python3.12.9, CC8.9,128SM,
25262096384 GPU bytes. The root launcher separately verifies the approved Hub
profile. No installation, provisioning, retry or background process is performed.

```bash
python -B -m cst_research.experiments.higgs_linear_atom_driver \
  --consumer-commit FULL_REVIEWED_COMMIT \
  --cache-root /path/to/verified-linear-cache \
  --output /path/to/fresh-qualification-output --timeout 7200
```

Run from the exact clean consumer snapshot. The shared external source and
compilation path are keyed by public-source/runtime identity, independently of
consumer commit; every reuse rehashes source bytes and checks the binding.
Consumers are separately immutable per commit. Foreground timeout reaps the
owned process group; failed evidence is retained and nonzero exit propagated.
The recovery index is retained evidence, not reusable campaign state or a passed
qualification. Stage B must preserve all Stage A source bytes and bind actual
passed16 proofs/raw initial states, identical source/runtime/data, then start
fresh0. It cannot start training from qualification step8 or call qualification
completion the intended400k task updates.

Planned Stage B: one full100k epoch per case (25000 updates,400000 total), full50k
development plus same-selected-state train/development predictions, outside the
training clock. There is no HPO, three-seed claim, automatic advance or formal
minimum-alpha conclusion. Final Dense-quality claims need proper separate
Dense/CST LR tuning, common horizons, paired seeds and frozen margins/family.

Raw evidence planning cap is8GiB, not a promised final size. Preserve essential
oracle/resume/checkpoint arrays. Retain all SHA-bound compressed evidence locally
and immutable remote files; independently admit one case in an owned temporary
directory to avoid a permanent duplicate decompressed tree. Before execution,
actual local free space minus conservative compressed total, maximum scratch and
primary-study reserve must leave at least8GiB; the owner seals the concrete
storage/runtime limits. No existing evidence deletion is authorized. Evidence
locator: `$CST_RESEARCH_EVIDENCE_ROOT/higgs-linear-atom-scaling-20261106/`.
