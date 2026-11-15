# Template/research migration validation

Validated 2026-11-04 in isolated worktrees. The live checkout and its running
Sisyphus jobs were not changed, stopped or resubmitted.

The final `berger_dev` layout splits data and runtime into `users/berger/data`
and `users/berger/model_pipelines`; launchers and this document are under
`users/berger/configs/model_combination`. Canonical imports alias historical
modules. The source-fingerprint resolver accounts for the split while retaining
the fingerprint inputs and resulting job IDs.

Final split validation: 248 tests passed and three optional tests skipped.
Both historical and canonical split imports reproduce the same 1,197 job IDs.
The settings test locates external setup settings from the repository root
(or `ASR_SISYPHUS_SETTINGS`), rather than accidentally reading
`users/berger/settings.py` after the tests moved.

## Before/after reuse check

The mixed-source archive and migrated research branch constructed identical
sets of **1,197 job IDs**, including dependencies, for:

- Granite CTC/text-Qwen and Speech-Qwen full-suite graphs and two-pass rescoring.
- Parakeet RNN-T/text-Qwen and Speech-Qwen evaluation, 0.6B/1.7B Speech-Qwen,
  and focused development tuning.
- Own LibriSpeech CTC, context-1 transducer, full-context transducer and AED
  training/recognition on dev-other, including their default LM variants.

No tasks were executed by this graph comparison. Both sides used the same
external dependency checkouts and Sisyphus defaults. This validates relocation
for these graphs, not every historical setup or future dependency version.
`scripts/job_hash_probe.py OUTPUT.json` reproduces the graph inventory. The
pre/post JSON inventories are retained alongside the filesystem backup/worktrees.

The identical sorted JSON inventory has SHA256
`901a58ddfa97f9bd59d20fabdf1fce651a114f05f10eba4e270ba28b21599974`.

The compatibility tests also check the historical class module and pickle
identity, reuse of unchanged template modules, and source SHA256 manifests.

## Tests and smoke checks

- General templates: six focused tests passed, including audio-prompt/history
  scoring plumbing, partial-output formatting, lazy native-object pickling,
  template/research boundaries and last-hidden-state LM projection parity.
- Migrated research: 246 passed, three optional tests skipped because their
  local dependencies/builds were unavailable. No new model downloads or GPU
  recognition were run.
- The LibriSpeech template entrypoint constructed all four model families and
  92 recognition results (four evaluation splits), with 496 total jobs.
- A small real librasr gate decoded the Qwen template's Python scorer interface
  to the expected three-token path including EOS. This used synthetic scores,
  not Qwen model weights.
- Python compilation passed. Whitespace checks found only three pre-existing
  trailing blank lines in archived notes/normalizer files; those bytes are
  retained deliberately so source-based job fingerprints do not change.

The new standalone Qwen-ASR template still needs a cluster smoke run with the
pinned public weights. Its full GPU/model execution and WER are not claimed
validated by these CPU/interface tests. Existing completed research Qwen jobs
continue using their unchanged runtime definitions.

## External dependencies

The i6_core checkout has local changes (including Loquacious v2 preparation
and CMake support) and was not modified or committed in this refactor, which
covers i6_experiments. Record/transfer the required i6_core revision/patches
separately before deploying the research setup to another environment.
The template lazily imports optional v2 preparation so ordinary examples do
not require those local classes just to import. Preserve the current dependency
checkouts when resuming existing jobs.

## Completion and cutover readiness: 2026-11-14

All 234 selected evaluation cells now exist, with zero missing standalone
baselines. The submitted Granite/Speech-Qwen scorer-API check also matches
exactly on all 2,864 dev-other segments; see
[the final completion review](experiments/open_asr/COMPLETED_REPORT_REVIEW_20261114.md).

A fresh constructor-only comparison loaded the four actual active configs:
`controlled_comparison`, `controlled_evaluation`, `error_complementarity` and
`research_diagnostics`, using the same public settings, work directory and
completed exports in the live and isolated research checkouts:

- Both graphs contain 778 jobs, including 165 GPU jobs.
- All 165 GPU IDs are identical and have completion markers in the work tree.
- The 63 replaced IDs are exclusively CPU error-complementarity/research-analysis
  jobs, reflecting the corrected multi-fusion attribution code.
- A historical CPU PipelineJob cannot estimate its resource request because the
  old Loquacious lexicon source is unavailable on this host. Its identity is
  included in both inventories; no tasks were executed or recreated by the check.
- All tracked modifications/untracked files in the live checkout still match
  the complete November 4 backup byte-for-byte.

The two full graph inventories are retained in
`/work/asr4/berger/refactor/asr-20261104/cutover-20261114-{original,research}-graph.json`.
They supplement, rather than replace, the broader 1,197-ID migration probe.

The branch/config cutover has **not** happened at the time of this readiness
check: manager shutdown still needs confirmation. Keep the live source unchanged
until the manager and workers are stopped, then preserve its dirty state before
switching. Do not run two managers writing the same reports. A fresh manager
should only need the corrected CPU analyses for these four configs, not new
recognition. Other entrypoints and dependency changes require their own review.

## Live cutover completed: 2026-11-15

After the user confirmed Sisyphus was stopped, the live checkout was switched to
`berger_dev`. The setup's `config` symlink now points to
`recipe/i6_experiments/users/berger/configs/model_combination`. Settings, external
dependency checkouts and the work/output/alias locations were preserved.
The local `berger_template_setups` branch was fast-forwarded to its published
cleaned tip, `84efa1d62`, for separate generic-template use.

The pre-cutover source tree has a fresh 175 MB filesystem backup at
`/work/asr4/berger/backups/i6_experiments-20261115-pre-cutover`, plus Git stash
`4a4d643790d1a95b7749a288b873cbc2a6cf4005` (`pre-berger-dev-cutover-20261115`).
The original complete November 4 backup and archive worktree are also retained.
Do not apply the mixed-source stash over the cleaned branch; its original base
and configuration-link metadata are recorded in the cutover directory below.

Post-switch graph construction from the actual setup location loaded all four
active configs and matched the isolated readiness graph exactly: 778 jobs,
165 unchanged GPU IDs, all GPU completion markers present, and only the expected
63 CPU analysis replacements relative to the old live code. Historical and
canonical runtime imports refer to the same module objects. Filename-style
launchers expose `py()`; the optional parity launcher was corrected to use an
absolute experiment import when loaded as `config.ctc_api_parity`.

The cutover metadata, original tracked patch, saved settings, applied state and
post-switch graph/validation are under
`/work/asr4/berger/refactor/asr-20261104/cutover-20261115`.
No manager was started and no tasks were submitted during cutover. Restart the
same four configs to regenerate corrected CPU analysis; recognition is reused.
