# Contributing

English (canonical) | [日本語](CONTRIBUTING.ja.md)

Read this document and [AGENTS.md](AGENTS.md) before making changes. Distinguish
study design, implementation, observations and statistical claims, and leave
enough evidence for another researcher to trace the experimental conditions.

## Changes and review

- Use purpose-based branches such as `feat/<purpose>`, `fix/<purpose>`, `docs/<purpose>`, `refactor/<purpose>` or `test/<purpose>`.
- Integrate through a reviewed GitHub PR. Never push directly to `main` or substitute a local merge into main for PR integration.
- Make kernel, public operator and backend changes in torchcst, following that repository's workflow.
- Do not edit a frozen source snapshot and continue a running experiment with it. If a change is needed, use a separate version or study and distinguish its evidence from the existing run.
- Do not commit datasets, secrets, raw logs, checkpoints, internal VM/queue state or absolute user paths.
- Before cleaning up a worktree, check its commits, uncommitted changes, ignored evidence and recovery paths.

## Registering a study

Add JSON to `src/cst_research/studies/`, with a filename matching the study ID.
Record the source repository and full commit, runner entrypoint, public API/private
adapter boundary, separation of data selection and evaluation, comparison
conditions, random seeds, timing and memory measurement scopes, and evidence
locator. If the registry format changes, update its validator and meaningful
checks as well.

Fix the margins, comparison family and selection rules needed for a claim before
examining results. Distinguish changes made before execution from post hoc
exploration, and record their reasons, timing and effects. Do not turn the limited
power of three seeds or the scope of a conditional bootstrap into claims of a
universal optimum or a joint confidence guarantee. Change RUNNING to COMPLETED
only after the required execution, retrieval and verification have actually
finished.

Use a new study ID when changing the existing HIGGS design. Its training currently
runs in torchcst. Registration or CLI validation here does not constitute runner
migration, a rerun or adoption of its results.

## LM learning-curve controls

At the owner's request,include Dense and a source-qualified DST reference in
new LM learning-curve comparisons. Match data,model/ordinary initialization,
training-token budget and active FFN scalar budget. Record the DST algorithm,
mask/update rules and LR tuning; retain every candidate and label DEV selection.
Use the plotting --dst-run input and its compatibility checks. Do not substitute
HIGGS curves or incompatible task/source budgets. Radius/fraction diagnostics
remain CST-specific. A dense-storage DST accuracy reference does not establish
sparse-kernel speed or memory efficiency.

## Experiments and evidence

Freeze source, data and configuration identities before execution. Before using
the owner's GPUs, read the applicable resource skill, shared-pool rules and
allocation rules, and stay within the authorized scope. Write access to this
repository does not authorize new GPU allocations.

Retain raw evidence in ignored storage or managed external storage. Public notes
should record reproduction commands, source/data/result SHAs, job identifiers,
hardware/runtime, complete-step time, preparation time, allocated/reserved peaks,
failures and uncertainty. HPO timing across separate VMs is exploratory even when
the GPU type matches. Use paired comparisons within the same job for primary
timing comparisons. Keep earlier results tied to their original checkout,
workload and runtime; moving their history does not revalidate them.

## Checks

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m cst_research validate --json
```

The CLI checks require neither a GPU nor PyTorch. When adding a training runner,
separately validate independent oracles, complete steps, optimizer state, data
separation and a dense baseline under matching conditions.

PRs should briefly describe the purpose of the change, what actually ran, checks
that remain unperformed and where to find the evidence.
