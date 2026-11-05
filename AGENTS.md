# Agent instructions

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing this repository.

- This is the research repository. Public operators, runtime, backend and kernel implementations belong in torchcst; do not copy its runtime here.
- Use purpose-based branches (`feat/`, `fix/`, `docs/`, `refactor/`, `test/`) and integrate through a GitHub PR. Never push directly to main.
- Preserve the exact source/data/protocol boundary of a running experiment. Current HIGGS training remains at the pinned torchcst source and private experimental Graph adapter; registration is not migration or completed validation.
- Keep raw data, checkpoints, credentials and machine-specific absolute paths out of Git. Use the external evidence locator and ignored storage. Preserve required evidence and recovery information before cleanup.
- Before using owner GPUs, read the applicable resource skill and allocation/shared-pool rules. Repository instructions grant no new resource authorization.
- Validate meaningful metadata/protocol behavior with the stdlib test suite and `python -m cst_research validate`; distinguish these checks from GPU or model-quality evidence.
- Record incomplete, negative and inconclusive observations honestly. Keep formal fixed-family claims separate from exploratory learning curves and timing overlays.
