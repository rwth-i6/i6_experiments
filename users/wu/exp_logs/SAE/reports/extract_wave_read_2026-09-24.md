DONE

Source: work/speech_llm/sae/emc/blankfree_genmarg_jobs/GenMargSelectionJob.dPjElzqvLUYt/output/{selection.json,report.txt}
(registered at /e/project1/spell/wu24/2026-07-13_unsupervised/output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_em/wave/selection/)

1. G4a.L2.2 verdict: SIGNAL
   selected restart: em_s13, S = 3.386297576482928 (S 3.386298 in report.txt) — no seed field beyond arm name em_s13
   best null: null_s02, S = 5.713271964867761 (best_null_value)
   null_range (max-min over 4 nulls, permuted holdout) = 0.0027184489421516744
   identity_band (max |dS| of seeds 1,2 vs exact reruns) = 2.7342171940336613e-05
     em_s01 rerun |dS| 0.000002 (report.txt), em_s02 rerun |dS| 0.000027 (report.txt)
   margin = max(null_range, identity_band, 0.01) = 0.01 nats/frame
   gap (best null S - selected S) = 2.3269743883848326

2. void: all 16 restarts False (not VOID) per selection.json "void" dict; eligible = 16 of 16.
   selected restart em_s13 emitted rate = 7.58090938261736 Hz (selected_emitted_rate_hz), within band [5.8, 14.49] Hz — ELIGIBLE.

3. Report-only clause: reference (best phi_c S - selected S) = -0.04215099668909694, best_phi_c = phic_s01 (S 3.344146579793831). reference_reading = "NOT BEYOND"

4. Reference points printed: phi_c restarts phic_s01 S 3.344147, phic_s02 S 3.347813 (report.txt). No "gold phi" or "L2-0 _r100" S value printed anywhere in report.txt or selection.json keys.

5. MISSING (searched: config_sae_4a_lexlat_v2_em_v1.py build_wave/selection_reads body, output/.../wave/reads/* tree, grep for "Hungarian"/"NMI"/"D4" in that config) — this wave's selection_reads produces only S (tau=1 NLL/frame, genmarg.json) and A8 emitted/expected rate (gendecode.json); no genmarg direct PER, Hungarian PER, NMI(symbol,phone), or E[d] job/field is created or read by this config for any restart or null.

6. num_subepochs (n_sub, WAVE_NUM_SUBEPOCHS) = 12; duration setting (WAVE_DURATION_SETTING) = "durinit" — matches expected durinit,12.

7. Job ids and paths:
   - GenMargSelectionJob.dPjElzqvLUYt — /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_genmarg_jobs/GenMargSelectionJob.dPjElzqvLUYt
   - selected restart em_s13 genmarg read: ReturnnForwardJobV2.i1PBNWq9yDsD — .../work/i6_core/returnn/forward/ReturnnForwardJobV2.i1PBNWq9yDsD
   - selected restart em_s13 decode read (rate): ReturnnForwardJobV2.bLyIfnZ6zIwM — .../work/i6_core/returnn/forward/ReturnnForwardJobV2.bLyIfnZ6zIwM
   - best null null_s02 permuted-holdout read: ReturnnForwardJobV2.uh4bSnceNS3m — .../work/i6_core/returnn/forward/ReturnnForwardJobV2.uh4bSnceNS3m
   - best phi_c phic_s01 read: ReturnnForwardJobV2.QjJjqW3gWPaV — .../work/i6_core/returnn/forward/ReturnnForwardJobV2.QjJjqW3gWPaV
   - rerun em_s01 read: ReturnnForwardJobV2.rzqvEajgMKfs — .../work/i6_core/returnn/forward/ReturnnForwardJobV2.rzqvEajgMKfs
   - (each of the other 13 restarts, 3 other nulls (+ their real-holdout twins), phic_s02, and rerun em_s02 has its own ReturnnForwardJobV2.<hash> under the same work/i6_core/returnn/forward/ tree, registered under output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_em/wave/reads/{restarts,restart_decodes,nulls,nulls_real,phi_c,reruns}/<arm>/ — not individually resolved here, listed at that path.)

Config read: /e/project1/spell/wu24/2026-07-13_unsupervised/recipe/2025-10-speech-llm/src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_lexlat_v2_em_v1.py (build_wave, selection_reads, lines ~427-563)
