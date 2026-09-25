# A19 implementation: trigram-only lift ladder (2026-09-24)

Status: DONE_WITH_CONCERNS. The config builds, all CPU checks pass, and nothing was launched. There are two concerns, listed at the end: the launcher was kept, and the verdict job applies a reading of ambiguous rule text.

## Commit
- `bf5b2972` on `haotian_modality_matching_jupiter` in `recipe/2025-10-speech-llm`. Not pushed. The staged paths were:
  - `src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_lexlat_v2_triladder_v1.py` (new, 222 lines): `py()` builds the tri pack, its reads, the paired rows and the verdict reader.
  - `src/speech_llm/sae/emc/blankfree_a19_jobs.py` (new module, 279 lines): `A19TriLadderReadJob`. The A19 rule is quoted verbatim in the docstring together with the implementation convention, and the job prints both.
  - `src/speech_llm/sae/emc/test_lexlat_v2_triladder.py` (new, 428 lines): the CPU checks.
- No existing file was edited. Two paths in the checkout were already dirty and are not mine: the modified `config_sae_1g_v1.py` and the untracked `config_sae_3e1_d6_swap_cont_v1.py`. I left both unstaged.
- I wrote nothing in `config/`. The user creates the shim `config/sae_4a_lexlat_v2_triladder.py`, importing `py` from the config module.

## Construction
- Each arm `tri_rX` uses the same phi object that its twin `rt_rX` loads (asserted). The phis are built through `lad.build_fits()` and `lad.ladder_phis()`.
- The configs are built by `tri_train_config`, which makes the same `attrib_train_config` call as `lad.rt_train_config`. The only change is the `get_model` delta, which is `gp.model_args_delta("supphi_plain", ...)` = `{"reverse_checkpoint_path": phi}`. This is D10e's supphi_plain mechanism.
- The rt references are rebuilt through the ladder's own builder (`build_fits`, `ladder_phis`, `build_rt`). The pack ids are asserted: R1 `mZaZk7Ptt5Sg`, R2 `UdhhxiGIMBob`.

## Job ids (new, unfinished)
- pack = PackedBlankfreeTrainJob.DzrmcjOQ4I3r
- paired/tri_r100/ep1 = PairedPerDeltaJob.XERCbgiu1b5G
- paired/tri_r100/ep8 = PairedPerDeltaJob.FydRXKPifoFq
- paired/tri_r30/ep1 = PairedPerDeltaJob.oEVb5DQjAG72
- paired/tri_r30/ep8 = PairedPerDeltaJob.asSLPWQBTAWE
- paired/tri_r50/ep1 = PairedPerDeltaJob.a06Seov1AG2i
- paired/tri_r50/ep8 = PairedPerDeltaJob.wcPouikYogdU
- paired/tri_r70/ep1 = PairedPerDeltaJob.GCR5bLqhVqyy
- paired/tri_r70/ep8 = PairedPerDeltaJob.uu6CzwkAlixT
- read = A19TriLadderReadJob.5ny4LPLFrIXw
- tri_r100/ep1/per = BlankfreeGreedyPerJob.ER2oogMNoukU
- tri_r100/ep2/per = BlankfreeGreedyPerJob.qvlNilcm6sgM
- tri_r100/ep4/per = BlankfreeGreedyPerJob.inbrkSU0fHmK
- tri_r100/ep8/per = BlankfreeGreedyPerJob.EP2Fdf9LMfyt
- tri_r100/phi_ep8/report = GenDecodeReportJob.sY5kGZ6Mxnlh
- tri_r30/ep1/per = BlankfreeGreedyPerJob.Bnqmxo04s1rD
- tri_r30/ep2/per = BlankfreeGreedyPerJob.WqKoY3B0rwRe
- tri_r30/ep4/per = BlankfreeGreedyPerJob.hWrAWxpZ1zPZ
- tri_r30/ep8/per = BlankfreeGreedyPerJob.x2xeuyvu62dX
- tri_r30/phi_ep8/report = GenDecodeReportJob.uwhQkzuknrHV
- tri_r50/ep1/per = BlankfreeGreedyPerJob.iuqusnuTXQPo
- tri_r50/ep2/per = BlankfreeGreedyPerJob.k51TPG07XD4L
- tri_r50/ep4/per = BlankfreeGreedyPerJob.VkwkqZWxIYzR
- tri_r50/ep8/per = BlankfreeGreedyPerJob.qjL4ucgrsBVQ
- tri_r50/phi_ep8/report = GenDecodeReportJob.cEW4UDQW1g33
- tri_r70/ep1/per = BlankfreeGreedyPerJob.rDGiTWBt6hK0
- tri_r70/ep2/per = BlankfreeGreedyPerJob.NGStxlftNadj
- tri_r70/ep4/per = BlankfreeGreedyPerJob.GcyRR2bS35fR
- tri_r70/ep8/per = BlankfreeGreedyPerJob.U1DVS0X3Y2Ks
- tri_r70/phi_ep8/report = GenDecodeReportJob.RrA3nXaXl8ps

The graph holds 181 jobs, of which 66 are unfinished:
- 1 pack;
- 16 ExtractSubmoduleCheckpointJob, 16 ReturnnForwardJobV2 and 16 BlankfreeGreedyPerJob for the PER reads;
- 4 more ReturnnForwardJobV2 and 4 GenDecodeReportJob for the phi reads at ep8;
- 8 PairedPerDeltaJob;
- 1 A19TriLadderReadJob.

All 66 are downstream of the pack. The other 115 jobs are all finished L2-0 ladder jobs.

## Config diff (serialized returnn configs, each tri arm against its rt twin's banked config)
- The checked references are the banked files `PackedBlankfreeTrainJob.mZaZk7Ptt5Sg/output/rt_{r30,r50,r100}/returnn.config` and `UdhhxiGIMBob/output/rt_r70/returnn.config`.
- The rebuilt rt configs serialize byte-identically to these banked files.
- For all four arms, `get_model` differs in exactly 12 keys, and all of them are k2 keys: `lexlat_k2_hlg`, `_stats`, `_resources`, `_expected_build`, `_max_active`, `_onset`, `_ramp`, `_full_lam`, `_search_beam`, `_output_beam`, `_min_active_states` and `_chunk_seqs`. They are present in rt and absent in tri.
- At the top level, only `model` differs (the arm's own output directory).
- The text diff was also checked with an opcode-level assert: every change is either a deletion inside rt's k2 block or the `model` line.

tri_r30 diff (the other three arms are identical apart from the paths):
```
--- /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.mZaZk7Ptt5Sg/output/rt_r30/returnn.config
+++ tri_r30 (finalized pack config)
-        "lexlat_k2_hlg": "/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/lexlat_k2_jobs/LexlatHLGBuildJob.cdcxYJMjiYj5/output/HLG.pt",
-        "lexlat_k2_stats": "/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/lexlat_k2_jobs/LexlatHLGBuildJob.cdcxYJMjiYj5/output/build.json",
-        "lexlat_k2_resources": "/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/lexlat_jobs/LexiconTrieBuildJob.rlMsnTBSZXsB/output/lexlat_resources.npz",
-        "lexlat_k2_expected_build": {
-            "backoff_loops": "word_boundary",
-            "escape": True,
-            "sil_prob": 0.5,
-            "theta": 0.0,
-        },
-        "lexlat_k2_max_active": 1000,
-        "lexlat_k2_onset": 1,
-        "lexlat_k2_ramp": 3,
-        "lexlat_k2_full_lam": 1.0,
-        "lexlat_k2_search_beam": 20.0,
-        "lexlat_k2_output_beam": 8.0,
-        "lexlat_k2_min_active_states": 30,
-        "lexlat_k2_chunk_seqs": 4,
-model = "/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.mZaZk7Ptt5Sg/output/rt_r30/models/epoch"
+model = "/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.DzrmcjOQ4I3r/output/tri_r30/models/epoch"
```

## Evidence that k2 is off
- The model code gates k2 on `lexlat_k2_hlg`. When it is None, `model.lexlat_k2 = None` and `emc.lexlat_k2_train` is never imported. The train step marks the `lexlat_k2` loss only if `model.lexlat_k2` is not None.
- Test `stepfile`/`all` put a spy module in `sys.modules` for `speech_llm.sae.emc.lexlat_k2_train` and for `k2`.
- tri_r30's written `get_model` built the full-scale model on CPU with `lexlat_k2 = None`, and the spy was never touched.
- The written `train_step` then ran on 2 real dev utterances from the config's own dev dataset (`1963-142776-0014` and `5561-39621-0007`, cropped to 64 frames) at sub-epochs 1 and 8.
  - The rt on-set is 1, so k2 would be active at both sub-epochs.
  - The marked losses were agg, l_tau, rate and the blankfree_* monitors. There was no lexlat_k2 loss, and every loss was finite.
  - The check took 7 s.
- Positive control: rt_r30's written `get_model` reached the spy at `lexlat_k2_train.LexlatK2Runtime`.
- What this check does not show: it was a single cropped CPU step, so it does not measure any GPU time or memory saving.

## Resources
- The pack is built with `packed_blankfree_training`, as the R packs are. Its rqmt is `{'gpu': 4, 'cpu': 64, 'mem': 256.0, 'time': 11.5, 'gpu_mem': 96}`, identical to R1's.
- The time request is `lad.TIME_RQMT` = 11.5 h, equal to L2-0's.
- The projected time is 1.34 h per arm (8 x 601 s bed sub-epochs, with no k2 leg). This is a projection, not a measurement.
- The launcher is still `node1.K2_RETURNN_EXE` (concern 1).

## Hash checks
- No existing file was changed, and no existing module imports the new modules, so no file-sha `__sis_version__` can move.
- `config_sae_4a_lexlat_v2_ladder_v1.py()` builds 236 jobs, and all 236 are finished on disk, so no ladder id moved. Both R pack ids are asserted.
- Overlap with the live managers' graphs, each built in its own process. "Unfinished" means the job is unfinished in either graph:

| Config | Shared jobs | Shared and unfinished |
|---|---|---|
| em_v1 | 0 | 0 |
| a14_v1 | 115 | 0 |
| a17_v1 | 10 | 0 |
| keyinit_v1 | 10 | 0 |
| keysearch_s1_v1 | 15 | 0 |
| ladder_v1 | 115 | 0 |

## Checks run
Run as `python -m speech_llm.sae.emc.test_lexlat_v2_triladder all`, in the conda env with black on PATH, on the login node. It took 72 s and printed ALL PASS; I reran it on the committed state and it passed again (5 PASS). It covers:
- the reader edge cases;
- the reader's own `run` on L2-0's banked rt PER files as stand-ins, which gives SAME LADDER with rho*_tri = 0.7, and the paired self-rows, which give TRIGRAM ENOUGH;
- the graph checks;
- the config diff;
- the k2-off step.

## Left undetermined, and concerns
1. The launcher is kept as K2_RETURNN_EXE.
   - The k2 env is technically separable: no tri arm imports k2, and both envs have torch 2.7.1, CUDA 12.6 and numpy 2.4.6.
   - I kept it because A19 registers the k2 removal as the single delta, and swapping the interpreter would be a second one.
   - To drop the k2 executable, change the one constant `TRI_RETURNN_EXE` to `RETURNN_EXE`. That moves the pack hash, so the code review should decide.
2. The rule text is ambiguous, so the verdict job applies a reading, stated in the job's docstring before any result:
   - "LIFTs" and "every smaller rung lifting" are read as class LIFT.
   - "tri_r100 lifts" is read as LIFT or PARTIAL, which is A17's convention for the same phrase in this phase.
   - The job also prints the verdict under the other three combinations of these readings and flags READING-SENSITIVE when they disagree.
   - The rule has no tri_r0, so tri_r30 has no smaller rung.
   - Precedence: CANNOT_TELL if any ep8 PER is missing, then VOID, then CANNOT_TELL if non-monotone, then SAME LADDER, then K2 NEEDED ABOVE rho*_tri.
3. For K2 HELPS / TRIGRAM ENOUGH, the rule names no epoch, so the job prints both ep1 and ep8. M = 0.010 is a constant as registered; the job does not recompute it from rt_r0 / rt_r0_s2.
4. Not tested:
   - the GPU run itself;
   - the GenDecodeReportJob reads of the tri phis (the same call as A17 (i));
   - PairedPerDeltaJob on real tri decodes (its `run` was exercised on rt against rt).

Nothing was launched and no manager was started.
