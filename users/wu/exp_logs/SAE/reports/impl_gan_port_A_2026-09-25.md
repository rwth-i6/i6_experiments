# Implementer A: the wav2vec-U 2.0 GAN (1c) inputs and training, i6 port (2026-09-25)

DONE_WITH_CONCERNS. Four checks reproduce production exactly: (a) the resolved config, (b) the text data, (d) the phone LM and (e) the seed selection. For (c), lengths and k-means ids match exactly, but the features differ from the banked dump at fp16-rounding level. That difference comes from the port's upstream VAD HDF, not from the converter. The pytest suite shows only one known failure.

Worktree `/e/project1/spell/wu24/worktrees/i6_experiments_cycle_consistency`, branch
`haotian_cycle_consistency_unsupervised`, package `users/wu/experiments/unsupervised_asr/`. Nothing
committed. `git status --porcelain` shows only `??` lines (A's six files plus B's and C's new files).

## Files (all new)

| file | content |
|---|---|
| `w2vu2_tools.py` | `get_w2vu_python()` (settings `W2VU_PYTHON`, required, fixed hash), `get_fairseq_root()` (settings `W2VU_FAIRSEQ_ROOT`, else sparse `CloneGitRepositoryJob` of v0.12.2 = 4a388e64: `fairseq_cli`, `examples`, `fairseq/config`; fixed hash), `get_fairseq_user_dir()` |
| `env/build_w2vu_env.sh` | port of the source env script; takes `<env_prefix>`; adds flashlight-text 0.0.7; writes the `<prefix>/bin/w2vu-python` wrapper; runs the gate through the wrapper |
| `data/w2vu2_features.py` | `MfccKmeansJob` (in-process port of `fit_mfcc`), `W2vu2FeatureDataJob` (VAD HDF stream -> fairseq `data/`), pure `convert_split`, `get_w2vu2_feature_data()` |
| `lm/w2vu2_text.py` | `FairseqTextDataJob` (in-process fairseq-preprocess), `TextToPhonemeJob` (port of posterior_hmm's), `get_w2vu2_text_data()`, `get_w2vu2_phone_lm()` (i6_core `KenLMplzJob` + `CreateBinaryLMJob`) |
| `training/w2vu2_gan.py` | vendored `w2vu2.yaml`, `w2vu2_gan_config`, `build_w2vu2_gan` (i6_core `FairseqHydraTrainingJob`), `W2vu2GanSelectJob`, `get_w2vu2_gan()` |
| `tests/test_w2vu2_gan_inputs.py` | 10 fast tests and 6 artefact tests (the artefact tests name banked paths) |

## Interface for B and C

- Software. `w2vu2_tools.get_w2vu_python() -> tk.Path` is the wrapper `<prefix>/bin/w2vu-python`. `w2vu2_tools.get_fairseq_root() -> tk.Path` is a dir holding `fairseq_cli/hydra_train.py`, `examples/`, `fairseq/config/`. `w2vu2_tools.get_fairseq_user_dir()` is `<root>/examples/wav2vec/unsupervised`. The getter names are stable.
- The wrapper sets:
  - `LD_LIBRARY_PATH=<prefix>/lib:<torch/lib>` (replaced);
  - `PYTHONPATH=<prefix>/fairseq_shim:$PYTHONPATH`;
  - `PYTHONNOUSERSITE=1`;
  - `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
  The first three are the source's `settings.py` `_w2vu_env_overrides`. The last is the source's `selftrain.py:295`, which set it for the 1d job only; the wrapper sets it for every w2vu job, the GAN included.
- `training.w2vu2_gan.get_w2vu2_gan(seeds=(0,1,2,3,4)) -> W2vu2Gan` has these attributes:
  - `.data`: `W2vu2FeatureDataJob`. `out_dir` is fairseq's `task.data` dir. `out_files[name][ext]` covers name in {train, valid} and ext in {npy, lengths, km, ids, stats.txt}. `valid` is dev-clean followed by dev-other; `valid.ids` gives the split boundary.
  - `.text_data`: `FairseqTextDataJob` with `out_dir`, `out_dict`, `out_bin`, `out_idx`.
  - `.kenlm_path`: the phone 4-gram binary, a `tk.Path`.
  - `.trainings[seed]`: `FairseqHydraTrainingJob` with `out_checkpoint_dir` (`checkpoints/`, holding `checkpoint_best.pt`, `checkpoint_last.pt` and `checkpoint_<ep>_<upd>.pt` every 1000 updates) and `out_fairseq_hydra_yaml`.
  - `.selection`: `W2vu2GanSelectJob`.
- What B needs:
  - `gan.selection.out_checkpoint` is a symlink to the selected seed's `checkpoint_best.pt`.
  - `gan.selection.out_hydra_yaml` is that seed's `fairseq_hydra_config.yaml`.
  - `gan.selection.out_selection` is `selection.json`.
  - `gan.selection.out_seed` is an output var.
  - Banked equivalent: `.../w2vu2/gan/FairseqW2vu2TrainJob.HOb2GgtYT7Bc/output/train/checkpoint_best.pt` (update 148000).

## Checks (CPU, login node; scratch `/e/project1/spell/wu24/worktrees/port_checks_gan/A/`, deleted)

- **(a) Resolved config.** Setup:
  - Built the ported s0 `FairseqHydraTrainingJob` with the banked data/text/LM paths and a sparse v0.12.2 checkout as the fairseq root.
  - Wrote its `fairseq_hydra_config.yaml` with i6_core's `FairseqHydraConfig.write`.
  - Ran the job's exact argv (`<w2vu python> <root>/fairseq_cli/hydra_train.py --config-dir ... --config-name fairseq_hydra_config.yaml checkpoint.save_dir=...`, with the root prepended to PYTHONPATH) under the wrapper's env.
  - A shim stopped fairseq at `tasks.setup_task`, right after `fairseq_cli.train.main` logs the resolved cfg.

  Result: 313 keys against the 313 keys of HOb2GgtYT7Bc's `train.log` line 2. Only two differ, both paths: `checkpoint.save_dir` and `common.user_dir`. There are no non-path differences.

  Import check:
  - `fairseq` resolved to the env's site-packages 0.12.2 (`__version__` 0.12.2).
  - `fairseq_cli` resolved to the checkout.
  - `examples` resolved to the wrapper's shim, which is the wheel's copy, as in production.
  - The tag's `fairseq_cli/` and `fairseq/config/` equal the wheel's, and its `examples/wav2vec/unsupervised` equals the wheel's (diff -r; the tag only adds the dangling `kaldi_self_train/st/{steps,utils}` symlinks).
  - The amsgrad workaround is `w2vu2_base_config` (assert False, then pop).

  Test: `test_resolved_config_equals_banked_s0_except_paths`.
- **(b) Text data.** `write_fairseq_text_data` on the banked p_sil 0.5 text (`PhonemizeWithSilJob.DbFgvZOGZQ8F/output/text.phn.gz`, threshold 1000) gave `dict.txt`, `train.bin` and `train.idx` byte-equal (cmp) to `FairseqPreprocessTextJob.bi2B89fES77z`: 39,630,169 lines, 3,272,250,173 tokens, 44 symbols, 0 unk. The banked text used:
  - lexicon `MergeLexiconJob.qKaOAPqURCkK/output/lexicon.xml.gz`;
  - g2p `ApplyG2PModelJob.myTIGtmrUIFq/output/g2p.lexicon` (model `TrainG2PModelJob.pD4nbqFLWtbi`, concurrent 16);
  - corpus `DownloadJob.g4jClO48cAvP` librispeech-lm-norm.
- **(c) Converter.** `convert_split` on the first 25 utterances of each SAjz8y1cT06g split, with audio from the banked HF dataset OYvh9012Pgkb and the banked centroids `MfccKmeansJob.yRS4DGsXvqKm`, compared with wxBxCIaJpqS2:

  | split | utterances | equal length | `.km` lines equal | npy equals HDF rows | max abs diff | exact fraction | rel. Frobenius |
  |---|---|---|---|---|---|---|---|
  | train | 25 | 25/25 | 25/25 (15,392 ids) | exactly | 3.0 | 0.874 | 2.2e-4 |
  | valid | 50 (25 dev-clean, 25 dev-other) | 50/50 | 50/50 (14,383 ids) | exactly | 4.0 | 0.737 | 3.3e-4 |

  The largest feature values are about 2500, where the fp16 spacing is 2, so a difference of 3 or 4 is one or two fp16 steps. The minimum frame cosine is 0.99997. The feature difference comes from the port's wav2vec2 L15 dump (`L15FeatureHdfJob`), which is a different forward from the source's. The test asserts only the exact parts.
- **(d) Phone LM.** The ported `TextToPhonemeJob.run` on the banked inputs gave a decompressed `phon.txt` with the same md5 as THKMON3k9LJQ (11fb0d99...). Counts: 40,418,261 lines in, 39,630,169 out, 335,120 unresolved. The unresolved word list is also equal. `lmplz -o 4 --interpolate_unigrams True -S 16G --discount_fallback 0.5 1.0 1.5` (KenLMplzJob's command with the sae/1a arguments) on it gave an ARPA with the same md5 as `KenLMplzJob.0aJeN88X6EdW/output/lm.gz` (17d92c83...). Note that the i6_core job itself did not run, only its command.
- **(e) Selection.** `W2vu2GanSelectJob.run` on the five banked seeds read these `weighted_lm_ppl` values: s0 15.8538, s1 16.2418, s2 17.4490, s3 17.7588, s4 63.5183. It selected seed 0; its `checkpoint_best.pt` is at update 148000. Test: `test_select_reproduces_banked_seed0`.
- **(f) Pytest.** The full suite: 488 passed, 23 skipped, 7 xfailed, 1 failed. The failure is `test_model_reverse.py::test_segment_scores_explicit_sum[3]`, on the baseline list. The other baseline failure, `test_t2_6_posterior_dump_to_per_chain`, passed this time. The suite includes B's and C's new tests.
  - The new file without artefacts: 10 passed, 6 skipped.
  - With `SAE_ARTEFACT_DIR`: all 16 passed. The 15-minute byte test ran as the script in (b) rather than inside pytest.
- **Graph build.** `get_w2vu2_gan()` builds with dummy `W2VU_PYTHON`/`FFMPEG_BINARY`: 5 trainings, the clone job and the aliases `sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s{0..4}` and `/select`.
- **Package hygiene.** Package code contains no cluster paths and imports no speech_llm or other users' dirs. `bash -n` passes on the env script, and its wrapper was rendered with a fake prefix to check that every variable is exported.

## Deviations from production (none changes an algorithm, hyperparameter or selection rule)

1. **Utterance order.** Within a split, the order is the VAD stream's: train follows the four shards, dev follows sorted ids. The source used HF row order. The set of utterances is the same (train 28,539 utterances and 15,427,853 kept frames, valid 1,612,502 frames, all equal to production).
2. **Centroids.** The MFCC centroids are not bit-reproducible, because the fit pool follows the ogg-zip audio order. The algorithm and all arguments are unchanged.
3. **Features.** They come from the port's VAD HDF, which differs from the source dump at fp16 rounding level (see (c)).
4. **Audio decode.** On i6, `.km` reads the port's ogg zips (a pinned-ffmpeg encode), not HF audio. It is unverified whether the ids stay identical.
5. **Missing files.** There is no `.tsv` and no `preprocess.log`; fairseq reads neither.
6. **Config form.**
   - The yaml is vendored (a test asserts it equals the installed copy) instead of read at run time.
   - `lr_scheduler` is written as `{_name: pass_through}`, because i6_core's `check_consistency` crashes on a string.
   - `max_update`, `max_epoch` and `save_interval` are passed as job arguments; `save_dir` comes from i6_core.
   - Keys are sorted.
   - The resolved config is identical.
7. **Environment variable.** `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` now also reaches the GAN, as C requested.

## Undetermined or outside my files

- torchaudio (MFCC) is missing from the port's `env/environment.yml`, and scikit-learn may be too. The source env had torchaudio 2.7.1 and sklearn 1.8.0. `environment.yml` is not my file.
- `settings.py` on i6 needs `W2VU_PYTHON = <prefix>/bin/w2vu-python`. On JUPITER the existing w2vu env has no wrapper; build a new env with `env/build_w2vu_env.sh`.
- i6_core's `plot` task `eval`s the `hydra_train.log` lines, which is not tested for the GAN's json log.
- Whether the i6 worker's PYTHONPATH carries a regular `fairseq` package, which would shadow the env's fairseq, is unverified.
- The port's `lm_corpus_lexicon_and_g2p` trains its own g2p model, so port-generated text and LM are not guaranteed byte-equal to the banked ones. (b) and (d) tested the code on the banked inputs.
