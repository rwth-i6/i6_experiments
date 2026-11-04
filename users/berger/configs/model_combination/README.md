# Model-combination experiments

Maintained on `berger_dev`, based on the cleaned `berger_template_setups`.
General model/LM examples remain in `example_setups/seq2seq_rasr_2025`.

## Layout

| Directory | Contents |
| --- | --- |
| `users/berger/data/` | Custom datasets, lexica, OOV preparation and validation |
| `users/berger/model_pipelines/` | Model/scorer runtime, fusion, rescoring, normalization and analysis |
| `users/berger/configs/model_combination/` | Experiment drivers, launchers, tests and research notes |

Only custom/changed runtime modules are stored under the user directories;
unchanged generic modules are reused from the templates.
The research plan and AppTek handoff are under `experiments/open_asr/`.
`source_manifest.json` records pre-cleanup source paths and SHA256 hashes.

Point the Sisyphus setup's `config` symlink at this directory, then use
`./sis m config/research_diagnostics.py`, for example. The package
`py()`/`main()` runs the combined setup. Historical definitions remain
available; use modular launchers to keep the active graph manageable.
See [ENTRYPOINTS.md](ENTRYPOINTS.md) for groups and reports.

## Historical jobs and import compatibility

Moving Python job classes changes Sisyphus identities and breaks saved
pickles/configs. `bootstrap.py` is activated from the repository root package
on this branch, in both managers and workers. It resolves historical
`i6_experiments.example_setups.seq2seq_rasr_2025.*` imports from the split user
directories, falling back to unchanged templates. Classes retain their original
`__module__`; no hash is faked and no completed job files are rewritten.

Canonical `users.berger.data.*`, `users.berger.model_pipelines.*` and
`users.berger.configs.model_combination.experiments.*` imports alias those same
module objects, without creating duplicate job classes. Historical source
imports are retained for saved configurations. Old `configs/baselines`
filenames are compatibility launchers; historical fully qualified entrypoint
modules also resolve to the canonical files.

Compatibility is branch-local: no site-packages or other checkout is modified.
`I6_ASR_RESEARCH_IMPORTS=0` disables it before Python starts, for inspecting the
generic templates. Use the same value in a research manager and workers;
generic-only consumers should use `berger_template_setups`.

## Safe cutover

Keep the live checkout until its manager **and workers** are stopped. Preserve
its dirty state (for example,
`git stash push --include-untracked -m pre-berger-dev-cutover`) before switching
to `berger_dev`. Do not blindly pop the stash afterward: that would restore
research files to the template directories. Preserve Sisyphus settings,
work/output/alias paths and dependency checkouts when resuming jobs.
Do not run two managers against the same output tree.

The complete backup and local `berger_asr_precleanup_20261104` archive remain
available. `berger_asr_research` is retained as the first relocation snapshot.
Unrelated npm files and caches were backed up rather than committed.

The graph reuse probe covers 1,197 job IDs plus source manifests/pickle imports.
See [MIGRATION_VALIDATION.md](MIGRATION_VALIDATION.md). Review the proposed graph
at cutover rather than assuming every historical configuration is covered.
No recognition jobs are submitted by this refactor.
