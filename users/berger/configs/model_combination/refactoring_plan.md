# Template/research separation plan

Final amendment: branch `berger_dev`, with shared preparation under
`users/berger/data`, runtime under `users/berger/model_pipelines`, and the setup
under `users/berger/configs/model_combination`. The original plan below records
the first relocation; the split supersedes its broad package name.

The full pre-refactoring checkout (including .git, ignored and untracked files,
and symlinks without dereferencing them) is backed up at
`/work/asr4/berger/backups/i6_experiments-20261104-cleanup`.
A checksum-mode rsync dry run found no file-content differences (only the
source .git directory timestamp changed). The live checkout remains untouched
because running Sisyphus jobs import from it.

## Target branches

- `berger_template_setups`: general examples for CTC, context-1 monotonic
  transducer, full-context transducer, AED and Qwen-ASR 0.6B; ordinary 4-gram,
  LSTM and Transformer LM integration. Keep generic correctness fixes and
  explicit dataset/model configuration; remove research grids and local paths.
- `berger_asr_research`: based on the cleaned templates; custom code, research
  configs, tests and notes live in `users/berger/asr_research`. Preserve old
  Python job namespaces through an opt-in research import compatibility layer
  so relocating files does not itself reschedule expensive old jobs.
- A local pre-cleanup archive commit preserves the mixed source state, without
  committing node_modules, generated caches, or unrelated npm files.

## Execution

1. Inventory tracked changes and untracked files; create the archive commit in
   a separate worktree, without staging or changing the live checkout.
2. Build a template-only worktree from the branch base. Separate general
   correctness fixes from OOV, delayed-fusion, HF scoring and benchmark logic.
3. Simplify default experiment entrypoints; retain representative model/LM
   examples and a dedicated Qwen-ASR 0.6B RASR example. Remove local checkpoint
   examples, custom speech-LLM wrappers and inactive tuning grids from templates.
4. Add the research branch on top of the templates. Move custom files and
   mixed-file research overrides under users/berger; reuse unchanged template
   modules. Add explicit compatibility activation for historical job modules.
   Keep historical experiments opt-in instead of deleting their reproducibility.
5. Keep only useful usage documentation with the templates. Put research plans,
   audits, results, requirements and migration notes in the research package.
6. Validate compilation, import boundaries, representative graph construction,
   existing regression tests and representative pre/post-migration job hashes.
   Do not run GPU inference, rebuild models or submit any Sisyphus jobs.
7. Commit both final trees and push the template update and new research branch
   with ordinary fast-forward/new-branch pushes, without force pushing.
8. Leave the live checkout and active jobs intact. Record exact worktree paths,
   commits, compatibility limitations and restart instructions in the handoff.

## Limits

Module-path compatibility preserves Python class names and job identity where
inputs/code-version hashes are unchanged. Source-path-dependent CPU analyses
may need reruns. Existing generated configs and pickles must remain importable;
verify representative cases rather than promising all historical hashes are
identical. Full GPU/model equivalence requires later cluster execution.
