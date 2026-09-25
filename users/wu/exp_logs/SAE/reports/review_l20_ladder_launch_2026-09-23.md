# Review: L2-0 ladder launch (node P, R1, R2), 2026-09-23

Verdict: APPROVE_WITH_NOTES. The three packs can be submitted now. Two findings: one wrong-set issue in
the competence reads that feed the G4a.L2.3 bar (does not touch the packs), and one memory risk not
covered by the pre-flight.

Code: recipe/2025-10-speech-llm @ haotian_modality_matching_jupiter. Runtime files (lexlat_k2*.py,
sae_blankfree x2, pack_jobs.py) unchanged since 6fd3d02e; clean in git status (only untracked A11
emtable modules). Shim config/sae_4a_lexlat_v2_ladder.py -> config_sae_4a_lexlat_v2_ladder_v1.py.

## 1. Graph and ids
- The shim loads 220 jobs. Packs: node P PackedBlankfreeTrainJob.WX41NC734WLo, R1 mZaZk7Ptt5Sg,
  R2 UdhhxiGIMBob (no job dir yet at 22:08). The pre-flight Pl51viCk4CVP lives in its own shim.
- All 16 fit jobs finished (4 CorruptSeedGold, PermuteSeedGold, 5 PhoneTargetHdf, 5 reverse fits,
  phi_c extract BPxyi2TzJcEH), plus the gold fit 16v7R6ztSq1u.
- lexlat_k2_chunk_seqs 4: set by rt_train_config (ladder_v1.py:258, :474-482) and present in all 8
  rendered R-arm configs. The stability read runs at STABILITY_CHUNK_SEQS = 1 (lexlat_k2_train.py:129).
  Node P has no chunk key, so it runs at the default 16 (lexlat_k2.py:909/920), as D10e did.
  Its int32 risk is low: a trained theta, and ZM3's rung-1000 call on 16 uniform sequences completed.

## 2. Resolved-config diffs (every pack arm rendered via ReturnnConfig.write)
- R1 rt_r0 against the pre-flight's on-disk Pl51viCk4CVP/output/rt_r0/returnn.config: only the `model` path differs.
- rt_r0 against D10e WEmvqF2yBFG8/supphi_k2lat: recognizer_checkpoint_path goes from p0 YtuVg6ZYuK0P to flat
  0J9d6wjrkRYH, "lexlat_k2_chunk_seqs": 4 is added, and the model path differs.
- Every other R arm against rt_r0 differs only in reverse_checkpoint_path: r30 HpmOCSCklRsB,
  r50 QGovsj3absf2, r100 Ud3O2Pa3Lotp, perm jjc51BwoW48b, r70 OLIUO3BGy0ug; cold_ctl has the line removed.
  rt_r0_s2 also differs in flat DMSwTLXT9MWG, random_seed = 1 and random_seed_offset 1000. The model path differs everywhere.
- Node P arms against D10e: reverse_checkpoint_path and the model path only.
- The registered deltas match. No stray difference. Constants shared by all arms: tau 2.0 x 8, k2 onset 1, max_active 1000,
  beams 20/8, HLG cdcxYJMjiYj5, lr [1e-5, 1e-4 x7], batch 88000, max_seqs 128, laplace:.1000.

## 3. Resources
- Each pack runs 4 arms, 1 GPU each (pack_jobs.py:517-524), with rqmt gpu 4 / cpu 64 / mem 256 / time 11.5.
  An arm that fails does not kill its siblings (pack_jobs.py:180).
- Time: the pre-flight read 16.1 s/step (median; max 23.6), which projects 2.23 h, or about 2.45 h
  with the pack factor. D10e's pack took 2.20 h wall, and node P mirrors D10e. The 11.5 h clamp leaves
  more than 4x margin. There is no hours assert in build_rt (ladder_v1.py:494-518); node P's assert
  (:527-529) projects 1.74 h.
- Host memory: pre-flight peak PSS 12 GB against 64 GB per arm.
- GPU memory, finding B below.

## 4. Label use
- Everything label-using stays inside L2-0. The docstring declares the config a disclosed diagnostic
  and states that no job applies the rules (ladder_v1.py:8-10, :68-69, :107).
- A10 ext reuses the gold and r100 CV marginals only in a10_diagnostics, marked report only
  (em_ext_v1.py:144-161; blankfree_phifirst_a10_read.py:17). Neither reaches a10_read, K* or any selection.
- Descriptive only: the corruption job's rate and PER, and PairedPerDeltaJob's own flags.

## Findings
A. Train/eval overlap in the competence reads that set the G4a.L2.3 bar.
   - Where: ladder_v1.py:364-383 -> blankfree_genmarg_jobs.py:1495, GenMargSampleJob(segments=gp.DEV_SEGMENTS),
     which is CvHoldoutSplitJob.PfpCPQRCfIAk cv.segments (285 utterances).
   - Every ladder phi (gold 16v7R6ztSq1u, r30, r50, r70, r100, permphi) is fitted on
     SEED_INPUTS train_segments = CvHoldoutSplitJob.sD7U6CYs8ACM train.segments (2821 utterances;
     s2g_supervised_phi_v1.py:20; ladder_v1.py:290). 25 of the 285 CV-holdout utterances (8.8%) are in
     that fit set, so each of these phis was trained on their labels.
   - Clean sets: phi_c and L2-1's phi train on the bed stream (PfpCPQRCfIAk train.segments, confirmed in
     rt_r0's config lines 97/170), which excludes the CV holdout. The dev-other set overlaps with none (0).
   - Effect: under A5, statistics (a) and (b) on "all 285 CV-holdout utterances" are read for the bar
     (its value at rho*) and compared with L2-1's phi. The ladder phis' CV values include memorized
     items; the reference phis' values do not. The bar is biased, and the direction is not known.
   - The banked gold CV reads (JhiT3D0MyfPg etc.) carry the same overlap. The spec does not disclose it.
   - Repair (not a launch blocker; the packs do not use these reads): read the ladder phis on the 260
     disjoint CV utterances, or take the bar from dev-other only.
B. The pre-flight's 76.7 GiB peak does not bound a diffuse-phi arm.
   - Where: lexlat_k2_train.py:554-583; pre-flight gpu samples.
   - The peak was set at steps 0-1, which are the two shortest batches of the ascending first laplace
     bin (maxlen 330 and 479, 128 sequences). Later batches reach maxlen 784-805.
   - rt_r0's lattice collapsed fast: 37210 -> 16360 -> 8512 arcs/frame over steps 0-2, 1007 by step 7,
     215 by step 12. The ep2 peak was about 31-33 GiB.
   - A uniform lattice on a full-length batch was never measured. Chunking does not bound the held
     memory (all chunks' graphs are kept until backward).
   - Which arms are exposed: cold_ctl (random phi), rt_r100 (content-free phi) and possibly rt_r70.
     If one of them stays near-uniform to steps 7-8, 1.5-2.2x the frames could exceed the 96 GB card.
     That would OOM that one arm; the pack cannot resume it.
   - Counter-evidence: D10a's random phi sharpened theta within sub-epoch 1 (4.86 -> 2.81 bits).
   - Mitigation: watch those arms' arcs/frame and peak through step 12, or run a 12-step cold_ctl
     pre-flight (under 0.1 GPU-h) before R2.

## 5. Jobs shared with live managers
- A10 ext (pid 3391279) shares 8 jobs, all FINISHED as of 22:08:
  - ReturnnForwardJobV2.JhiT3D0MyfPg and 7AqKpfoiElR8 (the gold and r100 CV marginals);
  - GenMargSampleJob.b7aFZd9Tse5X and zXJjnNqU7kTa;
  - CorruptSeedGoldJob.LCFfke9OsUqO;
  - BlankfreeSupervisedReverseInitJob.16v7R6ztSq1u and Ud3O2Pa3Lotp;
  - PhoneTargetHdfJob.q4EG2AG2Saex.
- D14 (config/sae_4a_supervised_decphi.py, pid 3640835), D15 (config/sae_4a_lexlat_k2_prior_ablation.py) and D17 (config/sae_4a_lexlat_durprior.py) share 0 jobs. The D15 and D17 managers are no longer running.
- Start rule: every shared job is finished, so the ladder manager can start now with the full output
  set. Two managers can no longer race on a shared job.

## 6. Compute
- Training: 3 exclusive 4-GPU nodes at about 2.2-2.5 h each, which is about 7.5 node-h or 26-30 GPU-h.
- Reads:
  - 48 per-arm dev-other forwards (exclusive nodes, about 0.02-0.05 h each), about 10 GPU-h charged at most;
  - 26 genmarg forwards at about 0.01 h each;
  - CPU readers.
- Total about 30-40 GPU-h. Three nodes run at once.
