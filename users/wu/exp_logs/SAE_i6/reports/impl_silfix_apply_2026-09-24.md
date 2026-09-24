# Apply: SIL-run fix (`sil_run_collapse`, arm `ctrl_20_rc`) to the live tree (2026-09-24)

Status: DONE. I applied the reviewed patch and the config diff to the live tree. Nothing is committed.
The default path's job ids are unchanged. The CPU suite matches the reviewed worktree test by test.

Inputs: /work/asr4/hwu/tmp_dev/silfix.patch, /work/asr4/hwu/tmp_dev/silfix_config.diff,
reports/impl_silfix_2026-09-24.md, reports/review_silfix_2026-09-24.md (PASS_WITH_NOTES).
Scratch evidence is in SP = /var/tmp/claude-2764/-u-hwu-setups-librispeech-960-2026-09-24-unsupervised/e99790e1-d0a8-40e7-8845-f2cb4e1545a6/scratchpad.

## 1. Pre-apply check
- Branch: `haotian_cycle_consistency_unsupervised`. HEAD is dc52c8783, not the patch base 2ae4445f0.
  The three commits since the base (3ff973a7b, f35f5f1db, dc52c8783) touch only `exp_logs/SAE_i6/`.
- `git diff --stat 2ae4445f0 HEAD` on the six target paths is empty. `git status` showed no uncommitted
  edits to them. `git apply --check -v` passed for all six files.

## 2. Applied
- `git apply /work/asr4/hwu/tmp_dev/silfix.patch` in recipe/i6_experiments.
- `patch -p1 < silfix_config.diff` in the setup dir. There was no fuzz and no .orig or .rej file.
- I did not touch config/sae_i6_p0_screen.py (mtime still 15:47), settings.py or anything else.

Changed paths, to stage explicitly:
- recipe/i6_experiments (repo-relative):
  - users/wu/experiments/unsupervised_asr/model/blankfree_model.py (M)
  - users/wu/experiments/unsupervised_asr/tests/test_model_blankfree.py (M)
  - users/wu/experiments/unsupervised_asr/tests/test_model_lattice.py (M)
  - users/wu/experiments/unsupervised_asr/tests/test_model_sil_run_collapse.py (new, untracked)
  - users/wu/experiments/unsupervised_asr/training/arms.py (M)
  - users/wu/experiments/unsupervised_asr/training/config.py (M)
- setup dir: config/sae_i6_p0.py (M). The config dir is not inside recipe/i6_experiments, so this
  file is not part of that repo.

The other entries in `git status` were there before this apply and are not mine: SAE_i6_P0.md and the
untracked reports.

## 3. Verification
- Diff identity: the per-file `git diff` of the five modified files, plus `git diff --no-index /dev/null`
  for the new file, is byte-identical (`cmp`) to silfix.patch in patch order. `git apply -R --check`
  passes. All six files are byte-identical to the reviewed worktree /work/asr4/hwu/tmp_dev/silfix.
  The applied config/sae_i6_p0.py diff equals silfix_config.diff apart from the ---/+++ header lines.
  It is identical to the config the reviewer built (SP/sh_wt/config/sae_i6_p0_rc.py).
- Job ids: I used the shadow setup SP/sh_apply. Its recipe/ links to the live checkout, which is now
  patched, and to the live recipe/returnn, which is now pinned. Its config/ is a copy of the live
  configs. I loaded each config through sisyphus.loader (SP/dump_apply.py); outputs are in SP/sh_apply/probe/.
  - sae_i6_p0_screen: 75 ids. The ids and aliases are byte-identical to the pre-patch dump
    (SP/sh_live/probe). The ids sha1 is db40328d..., the same as in impl_returnn_pin.
  - sae_i6_p0: 164 ids. All 139 old id/alias pairs are present (old-id sha1 578dd2d8...).
    - There are 25 new ids, all `ctrl_20_rc`. They equal the implementer's new_jobs.tsv, and the full
      164-line tsv is byte-identical to the reviewer's worktree dump.
    - The training job is ReturnnTrainingJob.llSFybyKXkbL. ctrl_20 is still GiT88bxzoZbZ.
- CPU suite: I ran `pytest tests/` from the live package, with PYTHONPATH set to sh_apply/recipe and
  the pinned returnn. Result: 526 passed, 14 skipped, 12 xfailed, rc 0 (SP/pt_apply/suite.log).
  - The sets of PASSED, SKIPPED and XFAIL test names are identical to the reviewed worktree run
    (SP/pt_wt/suite.log).
  - Both T1.6 tests are still XFAIL with strict=True: test_t1_6_log_z_vs_run_collapse_oracle and
    test_t1_6_model_history_is_the_blankfree_one.
  - Pinned RETURNN against the old upstream clone made no difference in outcome.

## Notes
- The graph build wrote `__pycache__/*.pyc` under the package's training/ dir. These files are
  untracked and do not appear in `git status`.
- The live manager (pid 1554133) already imported the old modules. I did not restart or signal it.
