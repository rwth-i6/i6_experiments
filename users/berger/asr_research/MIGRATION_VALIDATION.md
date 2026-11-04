# Template/research migration validation

Validated 2026-11-04 in isolated worktrees. The live checkout and its running
Sisyphus jobs were not changed, stopped or resubmitted.

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
`tools/job_hash_probe.py OUTPUT.json` reproduces the graph inventory. The
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
