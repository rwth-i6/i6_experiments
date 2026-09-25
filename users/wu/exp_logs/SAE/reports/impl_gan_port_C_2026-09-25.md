# GAN port, role C: 1d CTC self-training student and its decodes (2026-09-25)

Status: DONE_WITH_CONCERNS. Worktree /e/project1/spell/wu24/worktrees/i6_experiments_cycle_consistency,
branch haotian_cycle_consistency_unsupervised, package users/wu/experiments/unsupervised_asr/.
Nothing committed; git status shows only '??'. The scratch dir port_checks_gan/C has been deleted.

## Decode route chosen: fairseq's own inference CLI, not RETURNN
The decode jobs run `examples/speech_recognition/new/infer.py` (production's command line) under the w2vu
python as an external tool. There is no worker script of ours. Reasons:
- it reproduces production's viterbi and KenLMDecoder exactly (20/20 hypotheses identical, below);
- a RETURNN forward would need a fairseq-to-HF conversion plus flashlight-text in the RETURNN env, each
  with its own equivalence proof, and it would buy nothing.
So model/w2vu2_ctc.py was NOT created, and no RETURNN training code was written.

## Files (all new)
- training/w2vu2_ctc.py
  - get_fairseq_w2v2_lv60_checkpoint(): DownloadJob, sha256 9b0748fb...736844
  - FairseqAudioManifestJob(*, ogg_zip, name) -> out_dir: FLAC per utterance + <name>.tsv + <name>.uid in zip order
  - FairseqCtcDataJob(*, manifest_dir, labels, gold, name="train") -> out_data_dir, out_dict_phn
    - drops utterances with an empty label
    - dict = sorted(label phones U gold phones), each line "X 1"
  - PRODUCTION_FINETUNE_CONFIG: vox_100h plus the 1d overrides; MAX_UPDATE 40000, MAX_EPOCH 0, SAVE_INTERVAL 1
  - build_ctc_training_job(*, data_dir, w2v_path, fairseq_python_exe, fairseq_root, rqmt=None)
    -> i6_core FairseqHydraTrainingJob with keep_epochs=[]
  - get_last_checkpoint(job) -> <out_checkpoint_dir>/checkpoint_last.pt
- analysis/w2vu2_ctc_decode.py
  - FlashlightLexiconJob(*, lexicon, dict_phn); convert_lexicon_lines is identical to production
  - OggZipWordRefsJob(*, ogg_zips: {split: zip}) -> refs from the bliss orth
  - CtcPhoneDecodeJob(*, manifests, checkpoint, dict_phn, gold, fairseq_python_exe, fairseq_root, max_tokens)
    -> per.json, hyps.json
  - CtcWordDecodeJob(*, manifests, checkpoint, dict_phn, lexicon, lm, fairseq_python_exe, fairseq_root,
    word_refs=None, beam=500, lm_weight=2.0, word_score=-1.0, max_tokens=1100000, train_shards=1)
    -> word_wer.json, word_hyps.json; train_shards is not hashed
- tests/test_w2vu2_ctc.py: fast tests plus artefact tests (SAE_ARTEFACT_DIR); decode equivalence is marked slow.

Wiring: pass fairseq_python_exe=w2vu2_tools.get_w2vu_python() and fairseq_root=w2vu2_tools.get_fairseq_root().
Checked: build_ctc_training_job builds with both getters (the fairseq root is the CloneGitRepositoryJob output).
The decode jobs' _examples_dir finds <root>/examples in the sparse checkout, and falls back to
<root>/fairseq/examples in site-packages.

## Inputs needed
- From B: a labels json {"labels": {utt_id: "P1 P2 ..."}} for train, CTC-collapsed and without sil
  (production's GanPseudoLabelJob.xjn6QnNqwEEH format).
- From the port: the w2vu2_tools getters; gold from GoldPhonesJob; ogg zips from get_ogg_zip
  (train-960 and dev); the official 4-gram.arpa.gz LM; the official lexicon.

## CONCERN (blocks a real training run, not the tests)
The training job needs TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1:
- torch 2.6 defaults torch.load to weights_only; loading the LV-60 wav2vec_vox_new.pt then fails with
  UnpicklingError (verified);
- production set it in the job env (live w2vu2/selftrain.py:288-295);
- i6_core FairseqHydraTrainingJob runs with os.environ and has no env hook;
- A's w2vu-python wrapper (env/build_w2vu_env.sh step 6) exports only LD_LIBRARY_PATH and PYTHONNOUSERSITE.
Proposal for A: add `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` to the wrapper. My decode jobs already set
it for their child process.
A second point for A: build_w2vu_env.sh does not pip-install flashlight-text (0.0.7 is in the reference env),
which fairseq's KenLM word decoder imports.

## Knob table: port vs production (BI1uYgPyTeQ0 / AQw3EcUo6rks)
- Resolved fairseq config: 358 vs 358 settings, 7 differ, all paths or runtime values:
  - paths: task.data, model.data, model.w2v_path, checkpoint.save_dir
  - runtime: distributed_init_method (tcp://localhost:17360 vs None, set at launch);
    distributed_num_procs and nprocs_per_node (4 vs 1, which are cuda.device_count() on the login node)
  - no non-path hyperparameter differs.
- Training rqmt: 4 GPU x 80 GB, 11.5 h.
- Audio route:
  - production: HF Ogg -> FLAC;
  - port: i6 FLAC -> pinned-ffmpeg Vorbis ogg zip -> FLAC PCM_16;
  - on 20 dev-other utterances: 0/20 bit-identical, max diff 1 LSB.
- Manifest order: the port uses zip order, production used HF order. Seeded batching therefore differs,
  so training will not be bit-reproducible (expected run-to-run spread).
- Invocation: the i6 job runs `<root>/fairseq_cli/hydra_train.py` with checkpoint.save_dir on the CLI;
  production used its own launch. The hyperparameters are identical.
- Decode data dir: a placeholder .phn per split (infer.py requires one; no effect on hypotheses).
- PER/WER: computed in-process (Levenshtein; WER on upper().split()), without fairseq's 4-digit rounding.
- LM: the port reads the .arpa.gz; it gunzips byte-identical to the banked 4-gram.arpa (sha bb250ea2...).
  flashlight KenLM reads .gz (checked).
- Lexicon sha d722bc29 = production's; the flashlight lexicon is byte-identical (206264 entries, 244 dups removed).
- requires_env="w2vu" (unhashed) is set on the decode jobs only; with A's wrapper it is redundant but harmless.

## Checks
- (a) Resolved config vs banked train.log line 28, via hydra_train.py under the w2vu python: PASS (above).
- (b) Not applicable (no RETURNN forward).
- (c) Decode equivalence on the first 20 dev-other utterances, CPU, 386 s:
  - phone: 20/20 hypotheses equal; PER 0.1597 (160/1002), same as production;
  - word: 20/20 hypotheses equal word for word; WER 0.2516 on these 20.
  - This does not replace a full-set WER (production: 0.1796 dev-clean / 0.2187 dev-other).
- (d) The yaml parses (fast test) and loads in hydra/fairseq (run (a)).
- dict.phn.txt reproduces production's (28539 train labels); refs equal production for 2703 dev-clean and
  2864 dev-other utterances.
- (e) Full suite:
  - after my change: 1 failed / 478 passed / 17 skipped / 7 xfailed;
  - baseline: 1 failed / 464 passed / 9 skipped / 7 xfailed;
  - the same pre-existing failure: test_model_reverse.py::test_segment_scores_explicit_sum[3]
    (1.46e-5 vs a 1e-5 tolerance; not my files).
- Rerun after A's w2vu2_tools.py landed: fast tests 6 passed / 3 skipped.
- No /e/project1 paths or fairseq imports in package code.

## Undetermined
- The train-set size and the pseudo-label source depend on B's job; the dict depends on them.
- Unverified end to end: a real training run and a full dev decode (needs GPU and a sis manager, out of scope).
