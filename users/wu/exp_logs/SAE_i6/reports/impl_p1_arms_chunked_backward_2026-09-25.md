# impl: P1 Task B arms, per-chunk k2 backward and config/sae_i6_p1_ladder.py (2026-09-25)

Status: DONE_WITH_CONCERNS. Nothing was launched and nothing was committed. No frozen file was edited.

## Outcome

- **Per-chunk backward.** The new file `reverse_model/rt_chunked_backward.py` runs each HLG chunk's backward
  before the next chunk is intersected, then drops that chunk's lattice.
  - On the T1.19 fixture it reproduces the held path (`LexlatK2Runtime.step`) BIT FOR BIT at every chunk size
    tested: log Z_HLG, the term, the total loss, every non-timing monitor, and every gradient to log_q, theta
    and phi. Every maximum deviation is 0.0, against the budgets of 1e-9 (log Z) and 1e-7 (gradients).
  - It is selected for the rt arms only, through the arms' RETURNN config epilog. The P0 import closure is
    untouched.
- **Entry point.** The new file `config/sae_i6_p1_ladder.py` builds rt_r70, rt_r80 and rt_r90 from the ladder
  builders.
  - The arm configs differ from each other only as the dispatch allows, plus each job's own output path (see
    Config diff).
  - All P0 job ids are unchanged: 164 of 164.
  - The graph shares no unfinished job with P0.

## Hook design

The held step (`model/lexlat_k2_train.py:514-585`) returns `term = sum_i(-(z_hlg_i - z_h_i)/retained_i * w_i) /
n_keep`. The train step marks it with `mark_as_loss(scale=lam)`. The held `log_z_hlg` keeps every chunk's k2
graph alive until RETURNN's `total_loss().backward()`, and that retention is the OOM on a 46 GB L40S.

`ChunkedBackwardLexlatK2Runtime` (a subclass; only `step` is replaced) works as follows.

1. **Upstream gradient.** The upstream gradient of each z_hlg_i does not depend on any z value. It is
   computed before any intersection by `z_hlg_upstream`, which runs autograd on the held term expression over
   a zero leaf with `grad_outputs = lam`. On kept rows it equals `-(lam/n_keep)/retained_i`, and 0 elsewhere
   (test `test_upstream_is_the_held_autograd` checks this exactly).
2. **Detached leaf.** The HLG leg reads `leaf = dense_in.detach().requires_grad_()`.
3. **Per-chunk loop.** `log_z_hlg_chunked_backward` repeats the `chunked_tot_scores` loop exactly: the same
   `chunk_bounds`, supervision segments, beams, `max_active` and `get_tot_scores(log_semiring=True,
   use_double_scores=True)`. For each chunk it:
   - reads that chunk's monitors: the `lattice_sizes` counts and `_expected_words_one` (words and escapes);
   - calls `tot.backward(upstream rows)`, with a 0 upstream on rows whose lattice is empty (the held
     `where(scored, ...)` sends those rows no gradient);
   - drops the lattice.

   The gradient accumulates in `leaf.grad`.
4. **Held logic after the loop.** z_h (the unpruned H leg on the whole batch) is computed on the LIVE
   `dense_in`, as in the held path. Everything else then runs exactly as held:
   - the z_h finiteness assert;
   - the empty count and the per-batch `empty_raise_frac` abort (`_abort_on_empty`, the held method);
   - `scored`, `l_lex` and the term;
   - the same 8 monitors, with the same keys, order and expressions (`_monitors_from_reads`);
   - sec, peak and stability (`_stability` is the held method).
5. **Surrogate.** The step returns `term.to(log_q.dtype) + _InjectGrad(dense_in, leaf.grad, lam)`.
   - `_InjectGrad` has the value 0.0, so the loss value is unchanged, and no `-inf * 0` NaN can arise from
     impossible emissions.
   - Its backward returns `leaf.grad * (incoming / lam)`. RETURNN's upstream into the term is exactly lam
     (`loss * scale`, in the loss dtype, float64), so the factor is exactly 1.0.
   - The HLG gradient therefore reaches `dense_in` unchanged. `dense_in` also receives the z_h gradient
     through the live graph, and autograd carries both into log_q, theta and phi.
6. **Where phi gets a gradient.** The k2 input is `dp_log_q`, which comes from the recognizer only. So phi
   receives no k2 gradient in either path; the full-step test confirms phi's gradients are equal.
7. **No-grad passes.** Under no_grad (RETURNN's dev pass) no backward runs and the step returns the held
   value.
8. **Selection.** `EPILOG` rebinds the config's `train_step` to `rt_chunked_backward.train_step`, which
   swaps the model's runtime class (`install_chunked_backward`, which asserts the exact held class) and
   calls the unchanged `model.train_step.train_step`. The P0 closure is unaffected: the module list and the
   164 ids are identical.

Monitors whose MEANING changes are `lexlat_k2_sec` and `lexlat_k2_peak_reserved_gib`: they now include the
HLG backwards, which the held path ran outside that window. They are not comparable 1:1 with JUPITER's
values of the same monitors.

Remaining held memory:
- the z_h graph (unpruned H on the whole batch);
- one dense-sized `leaf.grad`;
- the rest of the step.

Chunking z_h as well was not done. It is an option if the probe still OOMs.

## Files

- NEW `recipe/.../unsupervised_asr/reverse_model/rt_chunked_backward.py` (284 lines): the runtime subclass,
  `z_hlg_upstream`, `_InjectGrad`, `install_chunked_backward`, `train_step` and `EPILOG`.
- NEW `recipe/.../unsupervised_asr/tests/test_rt_chunked_backward.py` (338 lines).
- NEW `config/sae_i6_p1_ladder.py` (setup dir).
- NEW analysis_out files:
  - `p1_ladder_rt_{r70,r80,r90,r100,r90_s1}_2026-09-25.returnn.config`
  - `p1_ladder_jobs_{probe,arms,all}_2026-09-25.tsv` (id, state, rqmt, aliases)
  - `p1_ladder_test_rt_chunked_backward_2026-09-25.log`
- No existing file changed. `ladder.py`, `genmarg.py`, `sae_i6_p1_fits.py`, `common.py`, `model/`,
  `training/` and `analysis/` were not edited.

## Tests (CPU, sae python, `PYTHONPATH=recipe:recipe/returnn:sisyphus`)

`tests/test_rt_chunked_backward.py`: 38 passed; 2 GPU tests deselected.

- **T1.21-style (runtime).**
  - Grid: chunk sizes 1/2/3/16 × epochs 1/2/3 (lam 1/3, 2/3, 1; on-set 1, ramp 3) × tau 1/2, on 5 sequences
    of lengths 2-5 with one keep=0 row.
  - Losses go through `rf` `mark_as_loss(scale=lam)` and `total_loss().backward()`. log_q is
    log_softmax(base + bias Parameter).
  - The held `log_z_hlg` is patched to raise inside the new runtime, so the new path is the one exercised.
  - Held gradients are asserted nonzero.
  - Result: max |dlogZ| = 0, max |dmonitor| = 0, max |dgrad log_q| = 0 and max |dgrad param| = 0 in all 24
    cases. Terms are equal (for example 0.304462374412771 at tau 1).
- **Empty lattice, kept** (strict graph, `_forced_b_batch`, chunks 1/2/16): monitors and gradients are
  equal, the empty row gets zero gradient, and no marker is written.
- **Empty lattice, abort** (chunks 1/16): the same RuntimeError, and markers are equal except for time and
  work_dir.
- **No-grad pass:** equal.
- **`_InjectGrad`:** value 0, and the gradient is exact with a -inf emission present.
- **Selection:** `train_step` installs the runtime and delegates, and refuses a model without a k2 leg; the
  EPILOG exec binds `train_step`.
- **Full blank-free `train_step`** (the fixture leg behind the model's recognizer; chunks 1/16; epochs 1/3):
  - the total is equal;
  - every loss is equal (NaN==NaN for `lexlat_k2_stability`), except `blankfree_frames_per_sec` and the two
    timing monitors;
  - every parameter gradient is equal, with max |dgrad| = 0 for theta, phi and the others.
- **Existing suites, rerun:** `test_model_lexlat_k2.py` + `test_model_blankfree.py` gave 117 passed, 4
  xfailed, 2 deselected.
- **NOT RUN:** the GPU variant `test_chunked_backward_equals_held_cuda`, marked `gpu` and `k2`. This node
  has no GPU driver, and launching was out of scope.
- **NOT SHOWN:** that the per-chunk path lowers peak memory at the arms' batch shape. Only the rt_r90 probe
  can show that.

## Config (`config/sae_i6_p1_ladder.py`)

- **Arms.** Each arm is `ladder.rt_train_config(data=get_inputs().data, graph=get_graph("inhouse_3gram"),
  phi_checkpoint=fit["checkpoint"], second_seed=False)`, then `train_arm(f"{arm}/training", cfg, 8,
  keep_epochs=(1,2,4,8), time_rqmt=11.5, alias_prefix="sae_i6/p1/ladder")`, as `ladder.build_rt` does.
- **Fits.** They come from `config.sae_i6_p1_fits.p1_fits(P1_RHOS, 0)`. The phi checkpoints resolve to the
  fits report's ids: r70 ruLnJFWyifwp, r80 6IkAAuzcBwBR, r90 uO8wkodbR2uh.
- **Per arm.**
  - `cfg.python_epilog = EPILOG`; `python_epilog_hash` is left as constructed, which is hash-neutral.
  - `rqmt["sbatch_args"] = ["-p","gpu_48gb"]`. Resulting rqmt: gpu 1, cpu 16, mem 64, time 11.5,
    gpu_mem 96.
  - rt_r90 additionally gets `post_config["torch_log_memory_usage"] = True`. The model monitor
    `lexlat_k2_peak_reserved_gib` stays on.
- **Reads.** `analysis.per.epoch_reads(train, (1,2,4,8), name=f"p1_ladder/{arm}", ...,
  split=common.READ_SPLIT, gold=inputs.gold)`. This is the same call `common.train_and_read` makes for P0.
  - Registered under `sae_i6/p1/ladder/<arm>/`: `learning_rates` and `ep<e>/dev-other/per.{json,txt}`.
  - The job aliases are `sae/4a/p1_ladder/...`, because `epoch_reads` fixes that prefix.
- **Switches** (read from the environment):
  - `P1_LADDER_STAGE` = probe (the default) or arms;
  - `P1_LADDER_R100=1`;
  - `P1_LADDER_SEED2=<tags>`, which has no default.
- **Checked model args (r70):**
  - tau [2.0]×8; lr [1e-05] + [1e-4]×7; num_epochs 8; keep [1,2,4,8];
  - k2: max_active 1000, onset 1, ramp 3, full_lam 1.0, chunk_seqs 4; beams 20/8; min_active 30;
  - HLG LexlatHLGBuildJob.avjHv1Xvjyqd with expected_build theta 0.0;
  - flat init FlatRecognizerInitJob.0J9d6wjrkRYH (seed 0);
  - no `sil_run_collapse` key; reverse and recognizer lr multipliers 30 and 1 (both trainable);
  - the epilog is present at the end of the file.

## Config diff (written from the graph's jobs)

- **r70 vs r80:** `reverse_checkpoint_path`, plus i6_core's own `model = .../ReturnnTrainingJob.<self>/output/models/epoch`.
- **r70 vs r90:** the same two lines, plus `torch_log_memory_usage = True`.
- **probe r90 vs arms r90:** identical.
- **For reference:**
  - r70 vs r100 (switch only): the same two lines.
  - r90 vs r90_s1: `recognizer_checkpoint_path` (flat seed 1, FlatRecognizerInitJob.DMSwTLXT9MWG),
    `reverse_checkpoint_path`, `model`, `random_seed = 1`, `random_seed_offset: 1000`, and no memory flag.

The `model =` line is the job's own output directory; it is not a config choice and cannot be equal.

## Dry graph build (`sis console -s`, sae python, setup dir; read-only)

| stage | jobs | shared with P0, all FINISHED | fit jobs (from `sae_i6_p1_fits`), NOT_FINISHED, no work dir | new, NOT_FINISHED |
|---|---|---|---|---|
| probe | 87 | 62 | 12 | 13: ReturnnTrainingJob rt_r90 EexT85vdfx25; 4 each of ExtractSubmoduleCheckpointJob, ReturnnForwardJobV2 and BlankfreeGreedyPerJob |
| arms | 113 | 62 | 12 | 39: rt_r70 9lD2HS2Mzgcl, rt_r80 5CL3pQyyOvQt, rt_r90 EexT85vdfx25, plus 12 of each read job |
| arms + R100 + SEED2=r90 | 148 | 63 (plus the finished FlatRecognizerInitJob.DMSwTLXT9MWG) | 12 | 73 (rt_r100 M1jS5YvLGi1O, rt_r90_s1 520N2IKdkPPn, and their fits and reads) |

- **Unfinished jobs shared with P0:** none, in every stage.
- **P0 check:** `sis console config/sae_i6_p0.py -s` gives 164 jobs. The diff against
  `analysis_out/g0g_p0_jobids_before_2026-09-25.txt` is empty, and the module list equals
  `g0g_p0_graph_modules_2026-09-25.txt`.

## Concerns

1. **Job merge by hash.** The epilog is hash-neutral, so a held-path build of the same config has the SAME
   id. For example, `rt_train_config` without the epilog gives EexT85vdfx25 for rt_r90, as checked in a
   console. Sisyphus merges jobs by hash, and the FIRST construction wins.
   - Risk: a manager that also loads a held-path builder of these arms (for example the package's
     `config/lexlat_v2.py` `ladder()`) could write the config without the epilog.
   - Checks before launch: confirm that `work/.../ReturnnTrainingJob.<id>/output/returnn.config` ends with
     the epilog. The job log prints "rt_chunked_backward: per-chunk k2 backward installed" once.
   - Alternative: set `python_epilog_hash = EPILOG`. This makes the arms distinct jobs but changes their
     hashes.
2. **Fit jobs in two graphs.** The probe stage registers the 12 unstarted fit jobs. A concurrent
   `sae_i6_p1_fits.py` manager and a ladder manager would both see them. Run one of the two managers.
3. **GPU parity not run, memory relief not shown.** The GPU parity test was not run, and the memory relief
   is not demonstrated. The batch shape is unchanged from the reference (features 88000, max_seqs 128), and
   the z_h leg is still held.
4. **Hashes depend on `ladder.py` under review.** The arm hashes depend on `ladder.py` (`rt_train_config`
   and `schedules`), which is under review. Any change there moves them.

## UNDETERMINED (not chosen; the builders exist behind switches)

- **Second seed.** The dispatch names corruption seed 1 and FlatRecognizerInitJob seed 1. The ladder's only
  second-seed switch (`rt_train_config(second_seed=True)`, `ladder.SECOND_SEED`) also sets RETURNN
  `random_seed` 1 and `random_seed_offset` 1000. The builder uses that switch. Whether those two belong to
  the second seed is open.
  - Which rung or rungs get a second seed is also open (`P1_LADDER_SEED2` has no default).
  - The seed-1 fit is built with `ladder._gold_fit` and registered under `sae_i6/p1/ladder/fits/<t>_s1/`.
- **rt_r100.** Whether rt_r100 carries the memory flag is open; it is not set. Its fit registers under
  `sae_i6/p1/fits/r100` (the `p1_fits` prefix).
