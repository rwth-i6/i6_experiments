# Code review: P1 Task B corrupted-phi fits launch (2026-09-25)

PASS. No MUST-fix items. The launch is the gold-phi fit with only the label source changed. Three
non-blocking notes follow at the end: one deviation to record and two launch-order notes.

Reviewed: `reports/impl_p1_fits_2026-09-25.md` and `config/sae_i6_p1_fits.py`; `analysis/p1_nesting.py` and its
tests; `reverse_model/ladder.py` at HEAD 3dbdfb3bf (clean, last change 8280d6e1f); `supervised.py`,
`supervised_data.py`, `supervised_steps.py`, `model/reverse.py`, `data/gold.py`, `inputs.py`,
`config/supervised_init.py`; `settings.py` (`check_engine_limits`). The gold fit's work dir
(`ReturnnTrainingJob.Ac2eioZbRX7d`). JUPITER's reply (origin f9becb0f6,
`exp_logs/SAE/reports/reply_corruption_collapse_2026-09-25.md`). SAE_i6_P1.md Task B and the design review item 7.
Nothing was edited, launched or committed. The graph builds were read-only `sis console --skip_config` and
`sis console <cfg> -s`. The scratch scripts are in the session scratchpad.

## 1. Single delta against the gold fit Ac2eioZbRX7d

Method: in one console I built `p1_fits(register=False)` and `supervised_init.gold_phi()`; the latter rebuilt
Ac2eioZbRX7d. I then diffed the hashed kwargs, the job attributes and the RETURNN config dicts of both jobs,
and wrote each config file and diffed it against the gold fit's `output/returnn.config`. The rebuilt gold config
is byte-identical to the one on disk.
- `returnn.config`, r70/r80/r90 vs gold: 3 lines differ, all in the same way. They are `train` `files`
  (SupervisedReverseDataJob.{LO7eQJLYRc9M,pOT6A6WAq95V,3qfqdoscYbJ7}/train.hdf vs onjA2xZUQBdx), `dev` `files`
  (the same jobs' held.hdf) and `model` (each job's own dir). There is no init or preload key. `get_model`
  initialises seed 42 from scratch (`supervised_steps.py:get_model`). No gold job (Ac2eioZbRX7d,
  PhoneTargetHdfJob.foScplXsRIMc, SupervisedReverseDataJob.onjA2xZUQBdx) is in the fits graph.
- ReturnnTrainingJob kwargs outside returnn_config (epochs, keep_epochs, log_verbosity, time/mem/cpu, RETURNN
  exe/root): identical.
- SupervisedReverseDataJob: only `gold_json` differs (CorruptSeedGoldJob.<rung>/corrupted_phones.json) and
  `targets_hdf` (PhoneTargetHdfJob.{UgE8lffBU9f1,OtACGK7q2d1R,Otjvrdt4P6Ya}). `ids_json`, the split segments,
  the 4 unit HDFs and eta_npz are identical.
- PhoneTargetHdfJob: only `labels_json` differs (the same corrupted json). `ids_json` and `label_key=None` are
  the same as the gold targets'. So both the json and the targets are the corrupted strings (`ladder.py:431-437,458`).
  `supervised_data.py:94-100` also asserts target == json + 1 per utterance at run time.
- Batches: `_length_buckets` sorts by (len z, len y) and the corruption keeps length, so the batches and the
  per-epoch order match the gold fit's. The feasibility bound is unchanged: substitutes are never SIL, and
  every phone has D_max 25.
- Run-time code: RETURNN clone patches last modified 09-24 17:02; recipe runtime modules last modified
  09-24 13:03. Both predate the gold fit (09-25 01:37). The same env, torch 2.7.1 and RETURNN 00171dfe2.
- Every other difference: (a) rqmt `sbatch_args ['-p','gpu_24gb']`, where the gold fit ran on `gpu_48gb` (L40S,
  cn-509). This is unhashed and float-level only (see note N1). (b) Alias names. Nothing else.

## 2. Corruption jobs

The kwargs are rho 0.7/0.8/0.9, seed 0, and the P0 seed gold (SeedGoldPhonesJob.BUWKmEsK2zTk) and split
(CvHoldoutSplitJob.zYcc8EJsvdfV). The rng is `SeedSequence([seed, sha256(utt tag)[:8]])` (`ladder.py:124-126`),
so it does not depend on rho or on order. I ran the ported `corrupt_all` locally on the real i6 seed gold:
- Seed gold: 2849 utterances, 351,312 tokens, 1705 adjacent repeats in 1251 utterances. Exactly 39 symbols
  (the `phones.PHONES[:39]` set), no SIL.
- r70 realised 0.699905, PER 0.695288, corrupted repeats 14293. r80: 0.800081 / 14706. r90: 0.899946 / 14942.
  All three are within +-0.005. Positions differing = substituted, no collapse, lengths equal, 39 symbols.
- JUPITER's banked table is reproduced exactly: r30 0.300007/8799, r50 0.500031/12157, r70
  0.699905/0.695288/14293, r100 1.000000/14857. The i6 seed gold strings are therefore JUPITER's inputs, to the
  resolution of these statistics.
- Nesting on the real data: r70 in r80 and r80 in r90, 0 subset violations, 0 symbol mismatches. The result does
  not depend on key order.

## 3. analysis/p1_nesting.py

The logic is correct. Substituted = differing positions, justified because a token is never replaced by
itself (`ladder.py:169`) and checked through `positions_differing == substituted`. The subset and same-symbol
checks, the missing/extra tag and length checks, the rate from `parts.all` and the repeat delta are right. The
JUPITER r70 comparison rounds the rate to 6 decimals. The tests pass: `pytest analysis/test_p1_nesting.py` gives
8 passed, and `tests/test_reverse_ladder.py` gives 10 passed (sae python, PYTHONPATH recipe:recipe/returnn:sisyphus).
End to end on real-size data (locally generated corruption files in the job's format), the script gives
`nesting_holds` true, `rate_minus_rho` -9.5e-5 / +8.1e-5 / -5.4e-5 and `r70_vs_jupiter.all_equal` true. The
+-0.005 rate gate is applied by hand from `rate_minus_rho`.

## 4. Resources and launch safety

- Gold fit actuals (Ac2eioZbRX7d): 0.0788 h (4 min 43 s), host RSS max 3.05 GB, 34 s per epoch. Largest batch
  over all 2824 logged steps: 229 phones and 823 units. The impl report's "185 / 715" covered only some steps.
- My CPU measurement of an upper-bound batch (8 utterances, all at S 823 and U 229, forward and backward):
  0.60 GiB saved for backward, 1.73 GiB process peak RSS. Adding a CUDA context puts the plausible GPU peak at
  about 2-3 GiB, so A10 23 GB has a margin of at least 7x. rqmt time 4 h is raised to 72 h
  (`settings.py:389-395`); `-p` is kept (`:396`). gpu_24gb MaxTime is 7 days. mem 64 GB and 16 CPUs are the
  gold fit's own.
- gpu_24gb = cn-501..505 A10 (sm_86); cn-290 RTX 3090 is drained. It is neither gpu_11gb nor an A100, and the
  fit does not import k2. QoS cap gres/gpu=6. cn-504 and cn-505 are idle (8 A10, 32 CPUs each, so 2 fits per node).
- P0 isolation: I loaded both graphs myself. `config/sae_i6_p0.py` gives 164 job ids, the same set as
  `analysis_out/g0g_p0_jobids_before_2026-09-25.txt` (the diff of the sorted lists is empty). The fits graph has
  54 jobs: 42 are shared with P0 and all 42 are FINISHED; the 12 new ones are NOT_FINISHED and have no work dir.
  Its 21 outputs are the 12 `sae_i6/p1/fits/*` outputs plus 9 `sae/4a/{feats,units}` outputs identical to
  P0's. The P0 manager pid 1646677 is alive (`m -r config/sae_i6_p0.py`, PATH with the sae bin first).
- Launch (from /u/hwu/setups/librispeech-960/2026-09-24-unsupervised):
  `(PATH="/work/asr4/hwu/conda/envs/sae/bin:$PATH" setsid nohup /work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis --log_level 30 m -r config/sae_i6_p1_fits.py > log/sae_i6_p1_fits.manager.log 2>&1 < /dev/null &)`.
  Then write the single pid to `log/sae_i6_p1_fits.manager.pid` and check that its environ PATH starts with the
  sae bin. Expect 3 CorruptSeedGoldJob first, then 3 targets, 3 data jobs and 3 fits in squeue on gpu_24gb.

## 5. G1.F dev-NLL clause

This clause is readable from the registered `sae_i6/p1/fits/<tag>/learning_rates.dev_loss_nll_per_frame`, the
RETURNN `learning_rates` file. Its epoch-8 entry `dev_loss_nll_per_frame` is the value to read; the gold file
gives 3.2742494784987515. `phi_ep8.pt` covers the "ends at epoch 8" clause. Denominator: the unit frames of the
28 held seed utterances, which are the same for every rung. Targets: each rung's own corrupted held strings
(the design review, item 7, intends this). The clause is therefore a wiring check, not a like-for-like
competence comparison. The (B) statistic S on the 260 set is not produced by this config and needs a later reads
entry point.

## Notes (none blocks launch)

- N1. Device: the fits run on A10 and the gold fit ran on L40S. The effect is float-level and cannot move the
  dev-NLL order, but record it in SAE_i6_P1.md Deviations. Keep FIT_PARTITION for any later r100 fit, so that
  all rungs share one device.
- N2. Start the arms manager (the future `config/sae_i6_p1_ladder.py`) only after the three fits are FINISHED.
  Otherwise two managers would hold the same unfinished fit jobs. Its code review should also confirm that it
  builds the same fit ids (ruLnJFWyifwp / 6IkAAuzcBwBR / uO8wkodbR2uh). An edit to `ladder.py` could change them.
- N3. Correct the impl report's batch maxima to 229 / 823. The memory conclusion stands.
