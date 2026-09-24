# P0 amendments: ctrl_20_s1, p0 PER read, report-only VAD counts, staged configs (2026-09-24)

Status: DONE_WITH_CONCERNS. All five changes are in place and both setup configs build. Every
existing job hash is unchanged, apart from the P0-only JS-rows job, which now also reads ctrl_20_s1.
The full CPU suite has one failure, `test_model_blankfree.py::test_t2_4_null_recognizer`. That
file is the other implementer's work in progress and none of my changes touches it (see Checks).
Nothing was launched, no manager was started and nothing was committed.

SETUP = `/u/hwu/setups/librispeech-960/2026-09-24-unsupervised`, PKG = `SETUP/recipe/i6_experiments/users/wu/experiments/unsupervised_asr`.

## Files

- `PKG/training/arms.py` (+29 lines): adds `CTRL_20_S1_SEEDS = {flat_seed 1, random_seed 1, random_seed_offset 1000}`
  (source: JUPITER `SAE_4A_prepro.md` l.13) and `ctrl_20_s1(*, data, num_sub_epochs=20)`. This preset
  is ctrl_20 with `data["flat_checkpoint"]` replaced by `FlatRecognizerInitJob(net_args=NET_ARGS, seed=1)`
  (alias `sae/4a/init/flat_s1`). It also passes `random_seed=1` and `random_seed_offset=1000` to
  `build_train_config`. It is added to `ARM_PRESETS` and `__all__`, and `ctrl_20` itself is untouched.
- `PKG/config/supervised_init.py` (+38 lines): adds `p0_per_read(rec, inputs=None)`, the arms' chain
  `posterior_dump` -> `BlankfreeGreedyPerJob` on `rec["checkpoint"]` (the `recognizer.` export of the
  `GetBestPtCheckpointJob` selection). It uses dev-other (`common.READ_SPLIT`), `NET_ARGS` and the
  default RETURNN, as `epoch_reads` does. `py()` now also registers
  `sae/4a/analysis_only/p0/selected_epoch` (`best.out_epoch`) and
  `sae/4a/analysis_only/p0/best/dev-other/per.{json,txt}`. The returned dict gains `"p0_per"`.
- `PKG/data/vad.py` (+50 lines): adds `write_counts_report(summary, expected, path)`, which writes the
  observed and expected totals per split and key, `rel_diff = (obs - exp) / exp`, `max_abs_rel_diff`
  and `all_equal`. `prepare_blankfree_data` gains `counts_report_path=None`: when set, a count mismatch
  writes that report instead of raising. `BlankfreeVadHdfJob` gains `counts_report_only=False`, which is
  in `__sis_hash_exclude__` and requires `expected_counts`. When True, the job has the output
  `out_counts_report` (`counts_vs_expected.json`); otherwise that attribute is `None`. The check that
  raises when a split keeps fewer than two frames still applies in both modes.
- `PKG/inputs.py` (+8/-3 lines): `counts_report_only=True` is passed to the VAD job only when
  `default_tools.get_ffmpeg_pin_accept() is not None`. Without the label the call is the same as
  before, and so is the hash.
- `SETUP/config/sae_i6_p0_screen.py` (new): `get_inputs`, `base.register_inputs` and
  `train_and_read(ctrl_20)` (PER at sub-epochs 1/4/10/20 plus the gaps at 20).
- `SETUP/config/sae_i6_p0.py` (rewritten):
  - ctrl_20, ctrl_20_s1 and k2lat_20_ma3000 (20 sub-epochs), each run through `train_and_read`;
  - `paired_delta(k2, ctrl)` at sub-epoch 20;
  - `paired_delta(ctrl, ctrl_s1, epoch=e)` for e in 1/4/10/20;
  - `js_rows("p0", {ctrl_20, ctrl_20_s1, k2lat_20_ma3000}, rows=[k2lat_20_ma3000_vs_ctrl_20, ctrl_20_vs_ctrl_20_s1])`;
  - `supervised_init.py()`.
- `PKG/tests/test_model_lexlat_k2.py` (appended): `test_t119_cpu_cuda_parity_log_z_hlg_and_log_z_h[tau 1, 2]`,
  marked `gpu` (and `k2` through the module). It runs T1.19's batch (lens 3/5/4, `random_log_q(3, 5, seed=3)`,
  the word_boundary graph, WIDE pruning) on the CPU and on CUDA and asserts `log_z_hlg` and `log_z_h`
  equal to rel 1e-5 (TOL). It checks the totals only, not the gradients.
- `PKG/tests/test_p0_amendments.py` (new, 7 tests):
  - the ctrl_20_s1 seeds reach the config: random_seed, the train feats' `random_seed_offset`, and the
    seed-1 flat-init job with NET_ARGS;
  - undoing those three deltas gives ctrl_20's config hash;
  - ctrl_20_s1's training job is distinct from ctrl_20's and has the same rqmt;
  - the VAD flag: False is hash-invisible and True gives a new hash and output;
  - `write_counts_report` produces the expected numbers;
  - `prepare_blankfree_data` with wrong counts raises in strict mode and writes the report in
    report-only mode;
  - `get_inputs()` gives the strict VAD without the label and the report-only VAD with it;
  - the p0 PER read reads the selected export, and its forward config hash, PER inputs and rqmt equal
    ctrl_20's ep20 read.

## Hash checks (sis console, sae env python, nothing run)

Before my edits, `config/sae_i6_p0.py` built 112 jobs, the same ids as `impl_settings_p0`. After the
edits it builds 139 jobs:
- All 111 earlier jobs except one keep their id. They include `ReturnnTrainingJob.zoGpJfrJw3As`
  (ctrl_20), `.ybGmGzaLx2Fn` (k2lat_20_ma3000), `.ep7ba66cqZkm` (gold phi), `.CqZ9Wr1y9sxl` (p0),
  `PairedPerDeltaJob.MvQYTb3AFdOW` and the VAD job `BlankfreeVadHdfJob.wW04G1dxqH75`.
- The one id that moved is `JsRowsReadJob.5g4N33JS8BlR` (P0-only), which is replaced by
  `.dKUOTuvFr9PM`, because the JS rows now include ctrl_20_s1.
- 28 jobs are new. ctrl_20_s1: training `ReturnnTrainingJob.bUyLkTQxRED3`, `FlatRecognizerInitJob.DMSwTLXT9MWG`,
  5 checkpoint extracts, 4 posterior dumps, 4 PER jobs and 2 gaps with 2 reverse-score forwards.
  p0 read: `ReturnnForwardJobV2.9eO4bvj0lEwv` and `BlankfreeGreedyPerJob.mMyU9tVhztYe`. Reads:
  4 seed-band `PairedPerDeltaJob`s and 1 JS job.

The package entry points were rebuilt and compared with the ids from `impl_settings_p0`:
- base.py: 108 of 108 equal;
- k2_word_lm.py: 212 of 212 equal;
- supervised_init.py: all 63 earlier ids present, plus the 2 new p0-read jobs (65).

The screen config builds 75 jobs, all of them in the full graph with the same ids.

## Job counts and GPU jobs

Screen (75 jobs): ReturnnForwardJobV2 13, L15ForwardSeqOrderJob 7, ExtractSubmoduleCheckpointJob 5,
BlankfreeGreedyPerJob 4, 3 each of the download, bliss and ogg jobs, 2 ReverseGapItemsJob,
1 ReturnnTrainingJob, and 1 each of the other input jobs. GPU jobs:
- ctrl_20 `zoGpJfrJw3As` (16 cpu, 64 GB, gpu_mem 96, 11.5 h) goes to gpu_48gb;
- 7 w2v2 L15 dumps (64 GB, 8 h) go to gpu_24gb;
- 4 ctrl_20 posterior dumps (24 GB, 2 h) go to gpu_24gb.

The VAD job (96 GB, CPU) uses the default partition. The screen has no HLG build and no k2 job.

Full P0 (139 jobs): ReturnnForwardJobV2 26, ExtractSubmoduleCheckpointJob 16, BlankfreeGreedyPerJob 13,
L15ForwardSeqOrderJob 7, ReverseGapItemsJob 6, PairedPerDeltaJob 5, ReturnnTrainingJob 5, 3 each of
the gap jobs, 2 FlatRecognizerInitJob and 1 JsRowsReadJob. GPU jobs:
- 5 trainings go to gpu_48gb (gpu_mem 96):
  - ctrl_20 `zoGpJfrJw3As`, 11.5 h;
  - ctrl_20_s1 `bUyLkTQxRED3`, 11.5 h;
  - k2lat_20_ma3000 `ybGmGzaLx2Fn`, 11.5 h;
  - gold phi `ep7ba66cqZkm`, 4 h;
  - p0 `CqZ9Wr1y9sxl`, 3 h.
- 7 w2v2 dumps and 13 posterior dumps (3 arms x 4 epochs, plus p0) go to gpu_24gb.

`LexlatHLGBuildJob.avjHv1Xvjyqd` (200 GB, CPU) goes to gpu_32gb. The 6 reverse-score forwards run on
the CPU. The rqmt of `PipelineJob.p4BOP5qZ6T1G` cannot be evaluated before its download exists; this
was already so in `impl_settings_p0`.

Probe scripts and outputs are in `/work/asr4/hwu/tmp_probe/p0_amend/` (`before_p0.tsv`, `after_*.tsv`,
`after_*.log`, `pytest_full.log`).

## Checks

- Targeted tests: 74 passed (73 in the combined run plus the fixed p0-read test, 7/7 in its file on rerun), 2 skipped (the CUDA parity test, no GPU here) and 3 xfailed, across
  `test_p0_amendments`, `test_model_lexlat_k2`, `test_config_graph`, `test_training_config`,
  `test_data_vad` and `test_data_librispeech`.
- Full CPU suite (`python -m pytest tests`, sae env, PYTHONPATH=SETUP/recipe:SETUP/sisyphus):
  428 passed, 7 skipped (4 ffmpeg-pin tests without a reference ffmpeg, 3 gpu tests) and 11 xfailed
  (the known strict xfails). There is 1 failure: `test_model_blankfree.py::test_t2_4_null_recognizer`,
  where `_rel(got, expected) <= 1e-10` fails with 2.59e-9 (13.798294937 vs 13.798294901). That file is
  on the other implementer's list and is being modified now (git shows it `M`). The test covers the
  null recognizer's train step, which none of my files touches.
- The CUDA parity test has NOT run: this desktop has no GPU. It must run on a gpu_48gb node, as G0.V
  requires.
- Nothing shows at run time that the report-only VAD mode works on the real data, that p0's PER chain
  works on the p0 checkpoint, or that ctrl_20_s1 trains. Only graph building and the tiny synthetic
  VAD run were exercised.

## Assumptions and open points

- Sign of the seed band: `paired_delta(ctrl_20, ctrl_20_s1)` (tag `ctrl_20_vs_ctrl_20_s1`), so the
  job's delta is ctrl_20 - ctrl_20_s1. This matches the gate's "band ctrl_20 - ctrl_20_s1" and the
  JUPITER table (`SAE_4A_prepro.md` l.232). The brief's wording "paired_delta of ctrl_20_s1 against
  ctrl_20" can also be read the other way round. A reversed call gives the same numbers with the
  opposite sign.
- JS rows: the P0 JS job now also includes ctrl_20_s1's ep20 decode, plus a row
  `ctrl_20_vs_ctrl_20_s1` next to the k2 row, following the convention "a JS row per registered PER
  row". The brief did not specify this. It moves only the P0-only JS job's hash.
- p0's read is at the SELECTED checkpoint only, as the brief asked, not at every kept checkpoint.
- The VAD report does not apply the 0.5 % threshold of G0.R0: it records the relative differences,
  and the gate reads them.
- `test_p0_amendments.py` imports `setup_inputs` and `outputs` from `test_data_vad.py`, which the
  other implementer is editing. If they rename those helpers, my VAD run test breaks.
