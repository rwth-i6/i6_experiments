# i6 settings.py and the P0 entry point (2026-09-24)

Status: DONE_WITH_CONCERNS. The settings, the P0 config, the librosa pin and the fixture copies are in
place. The P0 graph builds under the sae env python: 112 jobs. Every job it shares with the package
entry points has the same hash as there. Nothing was launched, no manager was started, and `work/`
is still empty. The concerns are the pre-launch items in the last section. They come from the
job requirements; none of them is a graph-build failure.

## Files

- `SETUP/settings.py` (the original is kept as `SETUP/settings.py.orig_2026-09-24`):
  - Added the package keys `SAE_PYTHON` and `G2P_PYTHON` (both `/work/asr4/hwu/conda/envs/sae/bin/python`),
    `FFMPEG_BINARY` (the env's `bin/ffmpeg`), `HF_HOME=/work/asr4/hwu/hf_home` (the dir was created),
    `KENLM_BINARY_PATH=/work/asr4/hwu/tools/kenlm_4cb443e6/build/bin` and `G2P_PATH` (the env's
    `bin/g2p.py`). `K2_PYTHON` and `FFMPEG_PIN_ACCEPT` are left unset.
  - `worker_wrapper` returns `call` unchanged, so no job runs in apptainer. The image selection and
    the blacklist are removed.
  - `DEFAULT_ENVIRONMENT_SET["PATH"]` puts `/work/asr4/hwu/conda/envs/sae/bin` first.
  - `check_engine_limits` keeps the 168 h time cap. A rqmt that already carries `-p` is left
    alone. A GPU job goes to `gpu_24gb` when `gpu_mem <= 24` (a missing gpu_mem counts as 0) and to
    `gpu_48gb` otherwise, so `gpu_11gb` is never used. A CPU job with `mem > 180` (the constant
    `CPU_MODERN_MAX_MEM`) goes to `gpu_32gb` without a GPU. Every other job keeps the default
    partition. The `cpu_slow` and `rescale` lists are removed.
  - The engine, `JOB_AUTO_CLEANUP=False`, `file_caching` and the other keys are unchanged.
- `SETUP/config/sae_i6_p0.py` (new): `py()` builds exactly what the brief lists, using only the
  package's functions: `get_inputs`, `base.register_inputs`, `train_and_read(ctrl_20)`,
  `train_and_read(k2lat_20_ma3000(graph=get_graph("inhouse_3gram")))`, `paired_delta(k2, ctrl)`,
  `js_rows("p0", {ctrl_20, k2lat_20_ma3000}, rows=[k2lat_20_ma3000_vs_ctrl_20])` and
  `supervised_init.py()`. The row tag uses the same format as in `k2_word_lm.py`.
- `PKG/env/environment.yml`: one added pip line,
  `librosa==0.11.0  # i6_core returnn/hdf.py imports it (missing from this hand-pinned list)`.
- Fixtures, copied unchanged (checked with cmp): `SETUP/config/skill_dummy.py` and
  `SETUP/recipe/sis_skill_test.py`.

## The P0 graph (sis console, sae env python, env bin first on PATH)

The graph has 112 jobs. Of these, 19 are ReturnnForwardJobV2, 11 ExtractSubmoduleCheckpointJob,
8 BlankfreeGreedyPerJob, 7 L15ForwardSeqOrderJob, 4 ReturnnTrainingJob and 4 ReverseGapItemsJob.
There are 3 each of BlissChangeEncodingJob, BlissToOggZipJob, DownloadHuggingFaceSnapshotJob,
DownloadLibriSpeechCorpusJob and LibriSpeechCreateBlissCorpusJob. There are 2 each of
BlankfreeDecodeGapJob, BlankfreeDerangementGapJob, CvHoldoutSplitJob and DownloadJob. Every other
class has 1 job: the pin check, the g2p jobs, the KenLM jobs, the HLG build, the lexicon trie, the
prior, the speaker eta, the VAD, the units, and the JS-rows and paired-delta reads. No x60,
off4 or k2_word_lm job is in the graph.

Jobs with a GPU, or with mem above 64 GB, and the partition `check_engine_limits` gives them:

| job | rqmt | partition |
|---|---|---|
| ReturnnTrainingJob `zoGpJfrJw3As` ctrl_20 | 16 cpu, 64 GB, gpu_mem 96, 11.5 h | gpu_48gb |
| ReturnnTrainingJob `ybGmGzaLx2Fn` k2lat_20_ma3000 | 16 cpu, 64 GB, gpu_mem 96, 11.5 h | gpu_48gb |
| ReturnnTrainingJob `ep7ba66cqZkm` gold phi | 16 cpu, 64 GB, gpu_mem 96, 4 h | gpu_48gb |
| ReturnnTrainingJob `CqZ9Wr1y9sxl` p0 supervised | 16 cpu, 64 GB, gpu_mem 96, 3 h | gpu_48gb |
| 7 ReturnnForwardJobV2, w2v2 L15 dumps (train shards 0-3, dev-clean, dev-other, seed_10h) | 4 cpu, 64 GB, gpu_mem 24, 8 h | gpu_24gb |
| 8 ReturnnForwardJobV2, posterior dumps (both arms, epochs 1, 4, 10, 20) | 4 cpu, 24 GB, gpu_mem 24, 2 h | gpu_24gb |
| LexlatHLGBuildJob `avjHv1Xvjyqd` | 16 cpu, 200 GB, CPU only, 6 h | gpu_32gb (no GPU) |
| BlankfreeVadHdfJob | 4 cpu, 96 GB, 8 h | default (cpu_modern) |

The other 4 ReturnnForwardJobV2 (the reverse-score forwards of the gap reads) run on CPU with
16 GB. The rqmt of PipelineJob `p4BOP5qZ6T1G` (the lexicon stress removal) depends on the size of
a download that does not exist yet, so it cannot be evaluated before its input exists. That is
normal for i6_core.

Probe scripts and their outputs are in `/work/asr4/hwu/tmp_probe/p0_probe/` (`probe.py` → `p0.out`,
`probe2.py` → `p0_2.out`).

## Hash equality

Sisyphus refuses a config path outside the setup ("Config name is invalid"). The throwaway configs
`/work/asr4/hwu/tmp_probe/p0_probe/cfg_{base,k2,sup}.py` were therefore exec'd and their `py()`
called inside `sis console --skip_config`. For comparison, P0 was built both ways; the two builds
gave the same 112 ids. Job counts: base.py 108, k2_word_lm.py 212, supervised_init.py 63.

- ctrl_20 is `ReturnnTrainingJob.zoGpJfrJw3As` in base.py, k2_word_lm.py and P0 alike.
- k2lat_20_ma3000 is `ReturnnTrainingJob.ybGmGzaLx2Fn` in k2_word_lm.py and P0.
- The paired delta k2lat_20_ma3000 vs ctrl_20 is `PairedPerDeltaJob.MvQYTb3AFdOW` in both.
- 111 of the 112 P0 jobs also occur in the union of the three entry points. The one exception is
  `JsRowsReadJob.5g4N33JS8BlR` (`sae/4a/js_rows/p0`). It is new by construction: `js_rows("p0", ...)`
  over two arms differs from the JS job of base.py (ctrl only) and from that of k2_word_lm.py (all
  arms).
- The id lists are in `ids_*.tsv` in the same dir.

## Checks

- A direct unit call of `check_engine_limits`: gpu_mem 11, gpu_mem 24 and a missing gpu_mem all
  give gpu_24gb, and gpu_mem 25 gives gpu_48gb. mem 181 gives gpu_32gb; mem 180 gives the default
  partition, with time capped from 500 to 168. An existing `-p` is kept. `worker_wrapper` returns
  its input unchanged.
- The graph build succeeded. It does NOT prove that the jobs run.
- No input path without a creator is missing. The kwargs of the jobs contain no JUPITER path
  (`/e/project`, `/p/`). In the package source, `/e/project` appears only in the default fallbacks of
  `default_tools.py`, which settings.py now overrides, in the unused top-level `vad_port.py`, and in
  docstrings and tests.

## Looks likely to fail or needs a decision before launch

1. **Trainings ask for gpu_mem 96 but get an L40S with 46 GB.** Whether the reference batch
   (88,000 frames, 128 seqs) fits has not been tested. P0.md allows a batch change only if it does
   not fit, and such a change voids the step-1 clause of G0.R1. The gold-phi and p0 trainings go to
   L40S too.
2. **The time rqmts are GH200 figures** (11.5 h for the arms). An L40S is slower, so the arms may
   hit the wall clock and depend on RETURNN's resume, which the patched updater fix is there for.
3. **The HLG build (200 GB) goes to gpu_32gb as a CPU-only job.** The brief reports that such jobs
   were accepted there; the time limit of that partition has not been checked.
4. mini_task jobs run on the LocalEngine of glukose (4 cpu, 8 GB), in the manager's env.
5. Not mine, but worth knowing: the `recipe/i6_experiments` checkout has uncommitted changes from
   another session (a modified `tests/test_lm_phone_prior.py`, new test files, and
   `reports/impl_tests_reverse_prior_agg_2026-09-24.md`). Nothing was committed here, because the
   dispatch named no commit.
