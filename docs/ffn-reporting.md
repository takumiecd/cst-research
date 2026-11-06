# Aggregating FFN observations

English (canonical) | [日本語](ffn-reporting.ja.md)

`ffn_report` reads retrieved R8-style evidence without Torch or a GPU. It links
terminal command receipts, summary/curve hashes and passed trained-checkpoint
audits. It rejects partial/fixed-batch/instrumented runs, mixed sources/data/
recipes/horizons, missing or duplicate comparison arms, incomplete curve grids,
wrong Atom group rates, and incomplete trained projection/resume checks. This is
receipt aggregation, not independent re-execution of GPU math or authentication
of arbitrary files. Retain the actual audited source and raw evidence.

Copy the [request template](ffn-report-request.example.json), set paths relative
to the evidence root and replace both SHA placeholders using the actual terminal
receipt and passed audit. The template is not a completed result. Every shape
and seed needs Dense, CST and no-FFN records. Use `development` with seed17 or
`confirmation` with exactly29/43/71; keep one selected recipe and token budget
throughout a request. Do not use confirmation scores to choose that recipe.

```bash
python -m pip install -e '.[reporting]'
PYTHONPATH=src python -m cst_research.experiments.ffn_report \
  --evidence-root "$CST_RESEARCH_EVIDENCE_ROOT/ffn-feasibility-20261106" \
  --request evidence/report-request.json --output evidence/new-report --plots
```

Use a fresh output directory. JSON retains every run, both fixed-prompt samples,
source/data/request/evidence hashes, reporter-source hashes and plotting runtime.
Equivalent English/Japanese Markdown contains metrics and the first predefined
prompt for the first seed, without choosing attractive samples. PNG/SVG figures
overlay every training/development seed, means and sample-SD bands, and show
complete-update/allocated/reserved observations. The optional reporting extra
is needed only for `--plots`.

SD is across independent seed runs, not the thousands of updates within one
run, and is not a confidence interval. Paired CST-minus-control CE differences
retain each seed. A development run has no estimated seed SD. Parameter counts
are checked against the declared FFN budget; observed width points do not prove
asymptotic runtime/memory complexity or a formal noninferiority result. Review
the rendered figures and the generation failures before including them in a
research report. Raw data/checkpoints remain outside Git.
