# Worktree cleanup before the radius-envelope implementation

English (canonical) | [日本語](worktree-cleanup-20261108.ja.md)

Eleven inactive worktrees were removed after an independent off-device restore
verification: six from torchcst and five from cst-research. Both repositories
retain their main checkout and one current development checkout. Unmerged
branches and PRs were preserved; no main push, merge, remote-branch deletion or
canonical experiment-output deletion was performed.

Before removal, all selected worktrees had clean tracked/untracked status. The
preservation includes all reachable Git refs/history in separate bundles and
complete retired directories, including ignored files and symlink targets,
excluding each root .git pointer. It contains11613 entries,284691846 logical
file bytes; the worktree archive is107362633 bytes. The separate bundles add
18116055 bytes (torchcst) and18417901 bytes (cst-research).

The Windows E-drive copy was independently extracted, every file SHA256/link
target checked, both bundles mirror-cloned, branch HEADs compared and git fsck
run successfully. Immediately before removal, local HEADs/status and every
preserved local file were checked again. No active process used a retired
worktree as its working directory. [Compact verification record](worktree-cleanup-20261108.json).

## Recovery

Archive ID: `20261108-cst-worktrees-01`. The owner's local recovery directory is
`torchcst/output/worktree-cleanup-20261108-01/`, with README.md, manifest.json,
verified.json, retirement.json, source bundles and worktrees.tar.gz. Its README
records the exact off-device location. The Windows copy also retains the
independently restored directories and mirror repositories. These are private
recovery records; machine-specific paths and raw payloads are not committed here.

Recover into a **new** location: mirror-clone the appropriate bundle, inspect
the manifest for the desired branch/HEAD, add a worktree for that branch and
overlay only the corresponding archived directory to recover ignored files.
Symlink targets are preserved as recorded and may need recreation on a different
host. Do not overlay the current development checkout.

The existing torchcst development checkout was moved from `lm-attention-levels`
to `radius-envelope` and placed on `feat/radius-envelope`; the previous
`kernel/lm-attention-levels` ref and draft PR155 remain intact. The research
checkout remains `lm-attention-output` on `feat/lm-attention-output`/draft PR35.
Historical source commits and their remote frozen experiment directories remain
unchanged. A recorded old local checkout path can be recreated from its ref.

## Selected follow-up design

The API name is `kernels.radius_envelope`, following the owner's request to
avoid `learnable_radius`. Preserve mass amplitude
a and learn an independent per-Atom q. Tau controls only the lower envelope:
U=r_max, L=r_min+(r_max-r_min)exp(-(a/tau)^2), r=L^(1-q)U^q. There is no center-motion
controller or energy-amplitude substitution in this implementation. Torchcst
owns the mathematical declaration, reference execution, gradients and optimizer
projection; this repository owns any future LM study. Implementation validation
is not a new LM learning result or an optimal-tau claim.
