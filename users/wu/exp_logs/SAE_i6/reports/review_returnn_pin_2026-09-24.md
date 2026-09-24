# Review: pin `recipe/returnn` to the jobs' RETURNN_ROOT (2026-09-24)

Verdict: PASS_WITH_NOTES. Every RETURNN import I could trace for a P0 job resolves to the pinned root
`work/i6_core/tools/git/CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn`. That root is exactly
00171dfe252e plus `returnn_local_fixes.patch`. No job id moved. The seven forward configs already on
disk are byte-identical to what the pinned code writes now. No finished output needs a rerun. The two
notes at the end are not blocking: the live manager, and the link target being a job output.

Inputs: reports/debug_w2v2_forward_2026-09-24.md, reports/impl_returnn_pin_2026-09-24.md, and
reports/impl_silfix_apply_2026-09-24.md (the concurrent ctrl_20_rc edit to config/sae_i6_p0.py).
Scratch evidence: SP = /var/tmp/claude-2764/-u-hwu-setups-librispeech-960-2026-09-24-unsupervised/e99790e1-d0a8-40e7-8845-f2cb4e1545a6/scratchpad/rev
Nothing in the setup was modified. I did not touch the holds or manager 1554133.

## 1. Main process and DataLoader workers load the same RETURNN (checked by running it)

- Import path, followed by hand:
  - The job runs `python <pinned>/rnn.py returnn.config` (i6_core/returnn/forward.py:316-327).
    `sys.path[0]` is the realpath of the pinned root, so the main process loads
    `/work/asr4/.../CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn/returnn/*`.
  - Line 43 of `returnn.config` inserts `recipe` at sys.path[0]. RETURNN's DataLoader uses
    `spawn_non_daemonic` (returnn/torch/data/pipeline.py:652). The spawned child copies the parent's
    sys.path and re-runs rnn.py as `__mp_main__`, so it imports `returnn` from `recipe/returnn`.
  - `recipe/returnn` is now a symlink to the pinned root. The repo-root `__init__.py` redirects to the
    inner package, as it did before the swap and on JUPITER.
  - `recipe/returnn.upstream-master` cannot be imported under the name `returnn`, and no code or config
    refers to it (grep of config/, settings.py, the package and i6_core).
- A real run of the job, not a stand-in:
  - I ran `rnn.py` from the pinned root on CPU with the cleaned job environment (no PYTHONPATH).
  - The config was the byte-exact `returnn.config` of ReturnnForwardJobV2.5z8bUupJ1f8j, plus four
    appended overrides: device=cpu, 3 segments, use_cache_manager=False, expected_num_seqs=3.
  - `PYTHONVERBOSE=1` logged every import of the main process, of `TDL worker 0` and of the
    watch-memory child.
  - Result: rc 0. The worker delivered 3 seqs, and feats.hdf and the stats files were written (SP/rnn/).
  - Of the 211 `returnn` source files loaded, all resolve to the pinned root: 136 through the /work/asr4
    realpath (main process) and 75 through `recipe/returnn` (worker). The log has 0 hits for
    "upstream-master".
- Spawn repro on all 7 forward configs (debugger's script, SP/spawn/o_*.log): each child exits 0. Each
  loads `returnn.config` from `recipe/returnn`, which resolves to KQ3NuCaDE6QH, and restores
  `forward_callback` with the right `expected_num_seqs`.
  - Held jobs: 7134 (khVKNe5qaTxE), 7135 (NxXNQ8v87CCM, UzlPyMEw1eR4, vVoEfyB8FVzF) and 2849
    (AjTxcYsLi4E6).
  - Errored jobs: 2703 (5z8bUupJ1f8j) and 2864 (QO3M1P9dOc2o).
- Control: I put a directory at sys.path[0] whose `returnn` links to the upstream clone. The child then
  fails with the original `AttributeError: 'function' object has no attribute 'startswith'` and exit
  code 1 (SP/spawn/o_ctl.log). The test therefore tells the two layouts apart.

## 2. The pinned root is exactly 00171dfe plus the patch

- HEAD is 00171dfe252eaabc71f6c1fa5a5a910c11a14788. The reflog shows only the clone and a checkout.
- `git status --ignored` lists 3 modified files and `__pycache__` directories. There are no untracked
  files and no assume-unchanged or skip-worktree flags.
- `git archive 00171dfe` plus `patch -p1` with the job's `patches` parameter gives a tree that
  `diff -r -x .git -x __pycache__` finds identical to the pinned root.
- The package's `returnn_local_fixes.patch` is byte-identical (`cmp`) to the job's `patches` parameter.
- Submodules are uninitialised, the same as in the upstream clone.

## 3. Job ids and config text (my own dump, live tree, after the swap)

- `sae_i6_p0_screen`: 75 ids, sha1 db40328d..., the same list as the pre-swap dump.
- `sae_i6_p0`: 164 ids. All 139 old ids are present: their sha1 is 578dd2d8..., the value
  impl_gpu_route recorded before the swap. The other 25 are ctrl_20_rc.
- The graph build loads 115 `returnn.*` modules (56 for the screen config). Their realpath is the
  pinned root; none resolves anywhere else.
- Written config text:
  - The 7 existing `output/returnn.config` files were written at 17:13 and 17:48 under upstream. They
    are byte-identical (`cmp`) to what I regenerated now under pinned code, through
    `create_returnn_config`, the same call path as create_files.
  - The 31 graph-level configs from the implementer's upstream and pinned mirrors are identical after
    path normalisation. My 5 training configs match the upstream mirror.
  - `returnn/util/pprint.py`, the only RETURNN code that shapes the config text, is identical in the
    two trees.
- Every job with a `returnn_root` uses KQ3NuCaDE6QH: 32 ReturnnForwardJobV2, 6 ReturnnTrainingJob and
  3 BlissToOggZipJob. RETURNN_ROOT has a fixed hash_overwrite (default_tools.py:91-97), so
  `recipe/returnn` is not a hash input.

## 4. Other consumers

- Graph build: see section 3.
- Sisyphus workers:
  - `import returnn` goes through sisyphus' RecipeFinder, then `recipe/returnn`, then the pinned root.
    Worker environments drop PYTHONPATH: the pickled `_sis_environment` of the forward jobs keeps only
    HOME, USER, TMP and similar, and the manager's environment has no PYTHONPATH.
  - i6_core/lib/hdf.py appends `returnn_root`, which is also the pinned root.
  - lm/hlg.py sets the child's PYTHONPATH to `recipe`, which also leads to the pinned root.
- Finished jobs:
  - None of the 38 finished jobs imported RETURNN inside a sisyphus worker.
    - Importing their modules loads no `returnn.*`.
    - The only package job that writes an HDF with SimpleHDFWriter (PhoneTargetHdfJob,
      BlankfreeVadHdfJob) has not run. GoldPhonesJob writes `gold.json` only.
  - The 3 BlissToOggZipJobs ran `<pinned>/tools/bliss-to-ogg-zip.py`, whose `_setup_returnn_env`
    inserts the pinned root. They never used upstream, and the tool differs between the versions only
    by a renamed keyword argument.
  - `hdf.py` differs between the versions only in typing, `__repr__`, and a falsy return value
    (None vs False) on an early return. The HDF layout would be the same either way.
  - The 7 forward create_files outputs are identical (section 3).
  - Nothing needs a rerun. The two errored jobs have no partial output: output/ holds only
    `returnn.config`.
- Job pickles:
  - All 49 existing `job.save` files unpickle under the pinned tree. None references a `returnn.*`
    class.
  - Pickling all 164 plus 75 graph jobs references only the modules sisyphus, i6_core, i6_experiments
    and collections.
  - So nothing from the manager's in-memory RETURNN can reach a job.

## 5. Other import routes

- No `returnn` exists in the sae site-packages, in `.pth` files or in a user site for 3.11.
- `config/` and `sisyphus/` contain no `returnn`. settings.py sets no PYTHONPATH or RETURNN_ROOT.
- The work dir has only one CloneGitRepositoryJob.
- No route found.

## 6. Link target is a job output

- `JOB_AUTO_CLEANUP = False` (settings.py:433).
- Even when cleanup is enabled, sisyphus `_sis_cleanup` keeps `output/` (sisyphus/job.py:565). It
  removes only `work/` and archives the logs.
- Auto-cleanup cannot remove the target.

## Notes (not blocking)

1. Manager 1554133 (screen config, started 17:01) still holds upstream modules in memory.
   - I found no route from them to a job: no job pickle contains RETURNN objects, create_files runs in
     a LocalEngine subprocess (localengine.py:123), and the config is not async.
   - Loading `sae_i6_p0.py` for ctrl_20_rc needs a fresh manager anyway, and that restart clears this.
2. The target can still be lost by manual action: `rm`, a cleaner run with a graph that lacks the
   clone job, or `git checkout`/`pull` inside `recipe/returnn`. Any of these would now change the
   RETURNN of every job.
