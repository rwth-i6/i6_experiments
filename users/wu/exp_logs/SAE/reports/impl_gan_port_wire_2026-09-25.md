# GAN port wiring (w2vu2 entry point) -- implementer report, 2026-09-25

Status: DONE_WITH_CONCERNS (graph builds and all checks pass; concerns are resource/hardware items for i6, listed at the end).

Worktree: /e/project1/spell/wu24/worktrees/i6_experiments_cycle_consistency (branch haotian_cycle_consistency_unsupervised), package `users/wu/experiments/unsupervised_asr/`. Not committed (as instructed).

## Files (new only; no tracked file modified, `git status --porcelain` shows only `??`)

- `config/w2vu2.py` (270 lines, new): `py()` entry point wiring A (inputs + fairseq GAN trainings + selection), B (checkpoint conversion, RETURNN forward, PER, pseudo-labels) and C (fairseq CTC student, phone PER, lexicon+KenLM word WER). Module docstring = the port's documentation for this line (banked numbers, env setup, deviations, resources, run command, "reproduction reference outside the GAN-free research line").
- `tests/test_w2vu2_config.py` (276 lines, new): 9 CPU graph tests (counts, seeds, per-seed dev evaluation, pseudo-label provenance, 1d wiring, rqmt, alias/output scoping, static and dynamic import allowlist).

Inputs from implementers: reports/impl_gan_port_A_2026-09-25.md, _B_, _C_.

## Graph (built on CPU, nothing run)

Job counts created by `py()` beyond the shared `get_inputs()` graph:
MfccKmeansJob 1, W2vu2FeatureDataJob 1, FairseqTextDataJob 1, TextToPhonemeJob 1, KenLMplzJob 1, CreateBinaryLMJob 1,
FairseqHydraTrainingJob 6 (5 GAN seeds + 1 CTC student), W2vu2GanSelectJob 1,
W2vu2GeneratorCheckpointJob 6 (5 seeds + selected), ReturnnForwardJobV2 11 (5 seeds x 2 dev splits + selected on train-clean-100),
W2vu2GanPerJob 10, W2vu2GanPseudoLabelJob 1,
FairseqAudioManifestJob 3, FairseqCtcDataJob 1, CtcPhoneDecodeJob 1, FlashlightLexiconJob 1, OggZipWordRefsJob 1, CtcWordDecodeJob 1.
Also created: CloneGitRepositoryJob x2 (fairseq sparse root when W2VU_FAIRSEQ_ROOT is unset; KenLM compile when KENLM_BINARY_PATH is unset) and an unused 3gram DownloadJob from `get_arpa_lm_dict()` (not registered, never run).

Aliases/outputs mirror production (`sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s{seed}`, `/select`, `sae/1d/...`) under the prefix `w2vu2/` via `gs.ALIAS_AND_OUTPUT_SUBDIR` (set in a context manager, restored after; the test asserts restoration). Shared port inputs (`get_inputs()`, lru_cached) are built before the scope and keep their own aliases. Registered: `w2vu2/sae/1c/.../s{0..4}/{dev-clean,dev-other}/per.{json,txt}`, `.../select/selection.json`, `w2vu2/sae/1d/{pseudo_labels.json, dict.phn.txt, per.json, hyps.json, lm/lexicon.phn.txt, word_refs.json, word_wer.json, word_hyps.json}`.

## Seed set: production had 5 seeds (0-4), passed explicitly as PRODUCTION_SEEDS

Evidence: `config/sae_1c_w2v2_pilot.py` uses `grid=seed_grid(5)`; banked FairseqW2vu2TrainJob train.log `seed` 0..4 in HOb2GgtYT7Bc (s0), KPeBBDiEJMxT (s1), MbO9o9hBZs2G (s2), zygHGPvCrQZn (s3), otFs6lCBF3wX (s4); `output/sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s0..s4/per.json` all exist; SAE_1c.md lines 58-64 list 5 seeds (s4 ppl 63.52, PER 0.851/0.862, collapsed). A's selection read on the banked logs: s0 15.8538 (min, selected), s1 16.2418, s2 17.4490, s3 17.7588, s4 63.5183. The dispatch's per-seed list (3 converged non-selected values) omitted the collapsed s4. A's file was not edited.

## rqmt (banked job.save unpickled with the production recipe) vs port

| job | production | port | change |
|---|---|---|---|
| GAN train x5 | gpu 1, gpu_mem 80, mem 60, time 11.5, cpu 8 | same, gpu_mem 32 | gpu_mem 80 -> 32 (i6 V100 32 GB) |
| CTC student (Wav2Vec2CtcFinetuneJob.BI1uYgPyTeQ0) | gpu 4, gpu_mem 80, mem 60, time 11.5, cpu 16 | same, gpu_mem 32 | gpu_mem 80 -> 32 (V100) |
| per-seed PER eval (W2vu2PerEvalJob x5) and pseudo-labels (GanPseudoLabelJob.xjn6QnNqwEEH) -> ReturnnForwardJobV2 | gpu 1, gpu_mem 40, mem 24, time 2, cpu 4 | same | B's default gpu_mem 24 overridden to production 40 |
| phone decode (Wav2Vec2CtcDecodeJob.qqKPLPBEt1K3, dev-only) | 1, 40, 24, time 2, 4 | same | C's default time 3 overridden to production 2 (3 was the decode_all job 1C2fmWJR1mcM) |
| word decode (Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks) | 1, 40, mem 64, time 11.5, cpu 8 | same | none |

gpu_mem 32 is justified by the banked logs: GAN min gb_free 88.6/95 GB (~6.4 GB used), CTC min gb_free 84.1/95 (~11 GB used per GPU). Forward/decode gpu_mem 40 fits L40S (46 GB) but not V100; how i6's settings map `gpu_mem` to a partition is unknown to me.

## Checks run

- New tests: `tests/test_w2vu2_config.py` 9 passed (6.2 s). Negative check: injecting `import i6_experiments.users.wu.util` into the dynamic allowlist script makes it fail with that module named (so the check can fail).
- Full port suite (speech_llm env, login-node CPU): 1 failed (only the pre-existing test_model_reverse.py::test_segment_scores_explicit_sum[3], same values as baseline), 497 passed (= 488 baseline + 9 new), 23 skipped, 7 xfailed, 280 s.
- Baseline before the new files: 1 failed (test_model_reverse.py::test_segment_scores_explicit_sum[3], pre-existing, 1.46e-5 vs tol 1e-5), 488 passed, 23 skipped, 7 xfailed.
- i6_core FairseqHydraTrainingJob plot parser (`eval` of `[train][INFO]`/`[valid][INFO]` lines) applied to the banked hydra logs: 0 failures over 984 GAN lines and 159 CTC lines (resolves A's open item, assuming the i6 fairseq writes the same log format).
- What was NOT checked: no job ran; graph build does not prove the jobs run on i6 or that the rqmt changes take effect on their cluster.

## torchaudio

Needed by `data/w2vu2_features.py` (MfccKmeansJob, W2vu2FeatureDataJob: MFCC via torchaudio) which run in the MAIN env (SAE_PYTHON), not the w2vu env; the w2vu env neither has nor needs it. So `env/build_w2vu_env.sh` was not changed. Proposal for the owner of `env/environment.yml`: add `torchaudio` (reference: 2.7.1 in the JUPITER speech_llm env; scikit-learn is already listed).

## Concerns / open items

1. Banked GAN job's sisyphus max-resource log shows RSS peak 307.67 GB against declared mem 60 (likely mmap'd features and dataloader workers double-counted, but unverified). The job succeeded on JUPITER with mem 60; i6 should watch for OOM kills. No change made.
2. Wall time: GAN took 10:27 h and the CTC student 15295 s on GH200; V100 will be slower, so GAN runs will likely exceed 11.5 h and rely on FairseqHydraTrainingJob's resumable run task. The CTC student requires a 4-GPU node.
3. Dispatch discrepancy: "10 dev-other flip one frame on port features, 4 from CPU/GPU near-ties" -- per report B it is 10 utterances across both dev splits (6 dev-clean, 4 dev-other; 8 feature drift, 2 near-ties) on port features, plus 4 near-tie flips (1 dev-clean, 3 dev-other) on the banked features. The docstring states B's numbers.
4. Assumptions: gpu_mem 32 derived from V100 32 GB; `W2VU_PYTHON` required in settings.py (no fallback), `W2VU_FAIRSEQ_ROOT` optional (else sparse clone), `KENLM_BINARY_PATH` optional (else compile).

## Scratch

`/e/project1/spell/wu24/worktrees/port_checks_gan/wire/` deleted after the checks; `port_checks_gan/A` was already absent.
