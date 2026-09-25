# Review: P0 k2lat per-chunk live change, chunk-2 (cs2) and chunk-1 (cs1) variants, 2026-09-25

Reviewer: code-reviewer (read-only; nothing installed, cancelled, patched or committed).
Target: J = `S/work/i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl` (Slurm 4359832, gpu_32gb V100),
S = `/u/hwu/setups/librispeech-960/2026-09-24-unsupervised`, P = `S/recipe/i6_experiments/users/wu/experiments/unsupervised_asr`.
Inputs: `reports/impl_p0_k2lat_cs1_cs2_2026-09-25.md`, the reviewed chunk-16 change
(`reports/impl_p0_k2lat_perchunk_2026-09-25.md`, `reports/review_p0_k2lat_perchunk_2026-09-25.md`),
and the cs8 change (`reports/impl_p0_k2lat_perchunk_cs8_2026-09-25.md`).

## Verdict

PASS_WITH_FIXES. Both variants differ from the reviewed chunk-16 change only in the chunk size. The size reaches
J's model through the `lexlat_k2_chunk_seqs` kwarg, with no override. The job hash does not change, and
the install scripts refuse and accept exactly as intended. The chunk size moves no number: the CPU tests show
0 deviation against the held path. Nothing in the code needs fixing. One procedure fix is required for
case (b): a bare `scancel` does not hold J, because the running manager resubmits it within minutes.

## 1. Single delta

- `analysis_out/k2lat_perchunk_returnn_cs2.config` (sha 3a53ad6c...) and `..._cs1.config` (sha 5d9848cf...) are
  the chunk-16 config (sha 36e7032e...) plus exactly these 3 lines at the end of the epilog:

      # P0 k2lat: sequences per k2 intersect call, the model kwarg lexlat_k2_chunk_seqs (unhashed epilog)
      get_model = __import__("functools").partial(get_model, lexlat_k2_chunk_seqs=2)   # =1 in cs1

  (after one blank line). No other line differs.
- Diff against J's live `output/returnn.config` (sha dff329b67bb1..., byte-equal to the `_csN.config.orig`
  copies in analysis_out; J itself has no `.orig` yet): +8 lines. These are the chunk-16 epilog lines (import of
  `rt_chunked_backward.train_step`) plus the 3 lines above. Nothing is removed, and no hashed section changes.
- `analysis_out/k2lat_perchunk_cs{2,1}_sae_i6_p0.patch`: each is the cs8 patch with 3 lines changed:
  `K2LAT_CHUNK_SEQS = N`, the count in the docstring, and the evidence path. `patch --dry-run -p1 -d S` applies
  cleanly to the current `config/sae_i6_p0.py` (sha 0f609369..., the chunk-16 version).
- `analysis_out/k2lat_perchunk_returnn_cs{2,1}_install.sh` differ from the chunk-16 script only in NEW, NEW_SHA,
  the tmp/.orig names and the comments. The check logic (lines 11-15) and the install logic (lines 16-21) are
  unchanged.

## 2. Hash unchanged

A dry graph build ran through `sisyphus/sis --log_level 40 console --skip_config` on scratch copies of
`config/sae_i6_p0.py` (base, +cs2 patch, +cs1 patch). Each gives 164 job ids. The three sets are identical and
equal both the chunk-16 after-list and `log/sae_i6_p0.watch_jobs`; `jcKXbLMDk4hl` is in all of them.
`python_epilog_hash` is '' and the epilog length is 408. J's built config reproduces
`analysis_out/k2lat_perchunk_returnn_csN.config` byte for byte. The other 5 training configs are unchanged.

## 3. Runtime delivery, no override

Code path:
- `returnn/torch/engine.py:1242-1245` calls `get_model`, which is the epilog's `functools.partial` because
  `Config.load_file` execs the epilog after the prolog;
- `model/blankfree_model.py:438,543-547` receives the kwarg;
- `model/lexlat_k2_train.py:275-286` builds the spec (the only writer of `chunk_seqs`);
- `model/lexlat_k2.py:906-925` resolves the size in this order: kwarg, then `$LEXLAT_K2_CHUNK_SEQS`, then the
  default of 16. The kwarg wins over the env var, and the Sisyphus worker drops the env var anyway
  (CLEANUP_ENVIRONMENT, `settings.py:449-460`);
- `reverse_model/rt_chunked_backward.py:252` loops over the chunk bounds, and `:306-315` prints
  `installed (chunk_seqs = N)`.

Checks:
- A path check with `LEXLAT_K2_CHUNK_SEQS=16` set in the environment gives spec `chunk_seqs` 1 and
  `installed (chunk_seqs = 1)`, with 128 intersect calls for 128 sequences on cs1. On cs2 it gives spec 2 and
  64 calls. So the kwarg overrides the env var.
- `get_model` survives the pickling through `SubProcCopyGlobalConfigPreInitFunc` with 28 keywords, and the
  chunk value is kept.
- The live probes confirm delivery: probe cs2 (4363361) prints `installed (chunk_seqs = 2)`, and probe cs1
  (4363362) prints `'chunk_seqs': 1` in its spec.

## 4. Install scripts

The preconditions were run against the real J (lines 1-15 only, which do not write): both print OK, with the
current config at dff329b67bb1.

Mock J directories:
- Refuse when `epoch.008.pt` exists.
- Refuse when the live config is any other per-chunk config (chunk-16, cs8, or the other of cs1/cs2), because
  the script requires OLD_SHA dff329b6.
- Exit 0 as already installed when the config already has the same NEW sha.
- Install cleanly with an atomic cp and mv, check the sha, print the diff and INSTALLED.

Gap (note only): the guard checks only for `epoch.008.pt`. It would install if `epoch.009.pt` exists and
`epoch.008.pt` was already cleaned. That state cannot be reached before a switch. It would also be harmless,
because the per-chunk path is exact.

## 5. CPU exactness

Re-run in the sae env on CPU: 24 tests passed (cs2, cs1, and cs2/cs1 against a held path at chunk 16). The
max deviation of loss and gradients is 0 in every case, and the term is 0.0860062307299628 in all of them.

## 6. Restart mechanics (source facts)

- The manager (pid 1646677, `sis m -r config/sae_i6_p0.py`, non-interactive) never reloads the config file
  (`sisyphus/loader.py:46-114`).
- A cancelled J whose `usage.run.1` is more than 100 s old becomes INTERRUPTED_RESUMABLE (`task.py:385-418`).
  The manager resubmits it without asking (`manager.py:386-402` returns True when non-interactive,
  `:417-435`). The retry cap is progress-aware through `completed_fraction`.
- This already happened in this manager: the ctrl_20_s1/rc jobs were resubmitted at 12:55-12:56, loaded
  `epoch.003.pt` and started epoch 4 at global step 171. On resubmit the job is not marked failed.
- `J/error.run.1` gives the ERROR state, and the manager does not resubmit.
- A file `J/hold` (`STATE_HOLD = "hold"`, `job.py:1285-1287`) is checked before every other state in
  `_sis_state` (`job.py:697-705`). While it exists the manager does not submit J. The watcher counts it as
  MGRONLY and does not treat it as an error. `error.run.1.<suffix>` renames are ignored by the watcher.
- RETURNN resume: it starts from the newest epoch that has both `.pt` and `.opt.pt`, and the global step
  comes from the checkpoint. Every sub-epoch so far has been 57 steps, so the step after epoch M is 57*M.
  Check it against `':meta:global_train_step_end'` of epoch M in `J/work/learning_rates`. The seed is
  merge(epoch, global_step, random_seed), so the resumed sub-epoch has the same data order and seeds. The
  learning-rate schedule is file-based and constant 1e-4 for epochs 8-20.
- On resubmit, `J/log.run.1` becomes a new hardlink to `engine/...run.<newid>.1`. The old log stays at
  `engine/...4359832.1`.
- J state at 17:45: `epoch.001.pt`, `epoch.004.pt` and `epoch.004.opt.pt` exist, and J is in sub-epoch 5.
  The projection puts `epoch.007.pt` at about 19:10-19:30. The P1 ladder manager (pid 1726329) does not
  have J in its graph.

## Findings

MUST
1. Case (b) needs `touch J/hold` before the cancel. A bare `scancel` leaves J INTERRUPTED_RESUMABLE, and
   manager 1646677 resubmits it in about 2-3 min (`sisyphus/task.py:392-418`, `sisyphus/manager.py:417-435`).
   If nothing is installed yet, the resubmitted J reloads `epoch.007` on the held path at chunk 16, runs
   sub-epoch 8, and is expected to OOM (35-38 GiB against 32 GB). That wastes a slot and the roughly 1 h of
   startup, and leaves `error.run.1`. The chunk-16 procedure (impl report, section 3) installs before the
   cancel, so it never needed a hold. Case (b) installs later, so it does.

SHOULD
2. Replace the drift check of the prior review. `find P/{model,training,lm} -newermt '2026-09-25 13:00'` now
   lists 4 new files that nothing imports: `lm/w2vu2_text.py`, `model/w2vu2_generator.py`,
   `training/w2vu2_ctc.py` and `training/w2vu2_gan.py` (commits 68b39418a and 692d6e55a). Use these instead:
   - `git -C P diff --name-status 14a8042d7 HEAD -- model training lm reverse_model`: expect only `A` lines
     for w2vu2 files;
   - `git -C P status --short -- .`: expect empty;
   - `find -L S/recipe/returnn/returnn -name '*.py' -newermt '2026-09-25 13:00'`: expect empty.
3. Apply `k2lat_perchunk_csN_sae_i6_p0.patch` in the same step as `csN_install.sh`, with the same N.
   Applying it under the running manager is safe: the manager never reloads the file, and nothing else
   imports it. Applying it before a probe passes would record an N that has not been chosen.

Notes
- To switch to a different N after a csN install, a new reviewed script is needed. Each script accepts only
  the original config (dff329b6).
- The implementer's statement that "the manager restart and job.save caveat stands" means only a later manager
  restart. No restart is needed for the switch itself.

## Restart steps

N is the largest chunk size whose probe passed G0.K2M. Its files are
`analysis_out/k2lat_perchunk_returnn_csN_install.sh` and `analysis_out/k2lat_perchunk_csN_sae_i6_p0.patch`.

Case (a): a probe passes before sub-epoch 7 ends.
1. Confirm that G0.K2M PASS at N and that the probe's own `installed (chunk_seqs = N)` line is present.
2. Run the drift check (finding 2).
3. Run `bash S/analysis_out/k2lat_perchunk_returnn_csN_install.sh` and expect INSTALLED. Then run
   `patch -p1 -d S < S/analysis_out/k2lat_perchunk_csN_sae_i6_p0.patch`.
4. Wait for the next boundary: `J/output/models/epoch.00M.opt.pt` exists (M <= 7), ideally after
   "Epoch M evaluation: dev:" and before any epoch-8 step. Then `scancel 4359832`, and check that
   `squeue -j 4359832` is empty.
5. If `J/error.run.1` exists, rename it to `J/error.run.1.cancel_backup`.
6. Do not restart the manager. Wait for a new line in `J/submit_log.run` with `-p gpu_32gb` and a time of 72 h.
7. Check the new `J/log.run.1`:
   - `Load model .../epoch.00M.pt`;
   - `Starting training at epoch M+1, global train step X`, where X is `global_train_step_end` of epoch M
     (57*M);
   - `'chunk_seqs': N` in the spec and exactly one `installed (chunk_seqs = N)`;
   - at sub-epoch 8: the stability line and the memory columns, no CUDA OOM, and no
     `J/work/lexlat_k2_ABORT.json`.

Case (b): no probe passes by the end of sub-epoch 7.
1. Before about 19:10, run `touch J/hold`.
2. When `J/output/models/epoch.007.opt.pt` exists (after the epoch-7 dev eval if possible), run
   `scancel 4359832` and check that `squeue -j 4359832` is empty. If the cancel comes late and the held path
   crashes in sub-epoch 8, `error.run.1` appears, and `epoch.007` stays intact because it is the last
   checkpoint.
3. When a probe passes at N, run the drift check, then the install (expect INSTALLED), then the patch.
4. Rename any `J/error.run.1` to `J/error.run.1.cancel_backup`, then run `rm J/hold`. Only then check that
   manager 1646677 is still alive.
5. Wait for the automatic resubmit, then check the log as in (a) step 7 with M = 7 (step 399).
