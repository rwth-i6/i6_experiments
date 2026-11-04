# ASR combination research

This package is maintained on `berger_asr_research`, based on the cleaned
`berger_template_setups` examples. Reusable CTC, context-1/full-context RNN-T,
AED, Qwen-ASR 0.6B and ordinary LM examples stay in `example_setups`. The custom
OOV searches, delayed fusion, Qwen experiments, Open-ASR evaluations,
leaderboard normalization, reports and research plans live here.

## Layout and use

- `setup/data/`: custom lexica, OOV validation, pretrained inventories and
  benchmark data preparation.
- `setup/model_pipelines/`: research runtime, delayed fusion, rescoring,
  benchmark scoring, reports and analysis. Only changed/custom files are
  stored here; unchanged template modules are reused.
- `setup/experiments/`: custom drivers and historical experiment definitions.
  Finished tuning remains opt-in; it is preserved for reproducibility.
- `setup/experiments/open_asr/MODEL_COMBINATION_RESEARCH_PLAN.md` and
  `APPTEK_HANDOFF.md`: current research plan and company-side handoff.
- `tests/`: custom regression tests, separate from the generic template tests.
- `source_manifest.json`: pre-cleanup source paths and SHA256 checksums.

Existing research entrypoints remain under `users/berger/configs/baselines`
on this branch. Use the modular `controlled_comparison.py`,
`controlled_evaluation.py`, `error_complementarity.py`, `research_diagnostics.py`,
`qwen_tuning.py`, etc. The template branch's same entrypoint directory contains
only the smaller general examples. Use one manager for overlapping outputs.

## Compatibility and migration

Moving Python job classes changes Sisyphus job identities and breaks saved
pickles/configs. This branch therefore activates `bootstrap.py` from the root
package initializer in both managers and workers. It resolves the historical
`i6_experiments.example_setups.seq2seq_rasr_2025.*` names from the moved source
files first, then from unchanged templates. The classes keep their original
`__module__`; no hash is faked and no completed job files are rewritten.
Research source files intentionally retain those import names. Do not import
the same files as `users.berger.asr_research.setup.*`, which would create a
second set of classes and different identities.

The source compatibility layer is branch-local. It does not install anything
into the live checkout, modify site-packages or affect other checkouts.

`I6_ASR_RESEARCH_IMPORTS=0` disables compatibility before Python starts, for
inspecting the general templates on this branch. Do not change it between a
research manager and its workers. Generic-only consumers should use the
template branch instead.

Keep the current running checkout in place until its jobs finish. Then stop
the manager and switch the *same setup location* to this branch, or point a
separate company setup at its own checkout. Preserve Sisyphus settings, the
work/output/alias directories and external dependency checkouts when resuming
existing public jobs. **Do not start another manager against the same output
tree while the first manager is active.** A separate worktree is for review
and validation until the cutover.

The live checkout still has its original uncommitted/untracked source, so do
not use `git pull` or force a checkout over it. After the manager **and workers**
are stopped, preserve that state first (for example,
`git stash push --include-untracked -m pre-asr-research-cutover`), then switch to
`berger_asr_research`. Do not blindly pop the entire stash afterward: that
would restore research files to the template directories. The complete backup
and local archive branch provide an additional recovery point.

Representative before/after job hashes and native class pickle imports are
checked during migration. Source bytes are retained, including code-version
fingerprints; only the recipe-root resolver is made portable, retaining its
existing `RECIPE_ROOT` hash override. This does not guarantee every historical
graph is identical: review the manager's proposed jobs at cutover, especially
source-path-dependent CPU diagnostics, and investigate unexpected expensive
jobs before submitting them.

## Backup and archives

The complete live directory was backed up before changes, including ignored
and untracked files and `.git`. The local `berger_asr_precleanup_20261104`
branch also records the mixed source snapshot. Unrelated npm files, caches and
node_modules were preserved in the filesystem backup, not committed.
Research notes and finished experiment definitions are kept here rather than
silently discarded; use the current modular drivers for active work.
