# Review -- P-BT back-translation probe, commit d39cf6c (2026-09-15)

Status: PASS_WITH_CONCERNS. Read-only review; nothing edited, nothing launched.

## (1) Single delta / no hash movement -- VERIFIED

Hash census, run twice with the SAME working tree except the four committed files:
`/tmp/.../scratchpad/census.py` loads `config_sae_4a_s0b_inits_v1` + `config_sae_4a_s3_v1` and
prints every reachable `_sis_id()`. "before" = the working tree with `reverse.py` replaced by
`d39cf6c^:.../reverse.py` and `bt_probe.py` / `test_bt_probe.py` / `config_sae_4a_bt_probe_v1.py`
removed (stderr confirms `SegmentalReverseModel.sample` absent / present per run; the other
implementer's uncommitted edits are identical in both trees). Result: **98 jobs, byte-identical id
lists, 0 moved**. `git show d39cf6c --numstat`: `reverse.py` 91 insertions, 0 deletions, ONE hunk at
`@@ -362,6 +362,97 @@`; the other three files are new; only `config_sae_4a_bt_probe_v1.py` imports
`bt_probe`. `WARMUP_PHI` is a frozen `tk.Path` with `hash_overwrite`, so no training job is rebuilt.
Config artefacts: `units_dev` (dev-other `UnitsHdfJob`) is genuinely new and must finish first.

## (2) Labels -- no leakage found

`gold.json` reaches the job only through `GoldGate.load()` (bt_probe.py:433), which asserts
`phase == "read"`; `load_dev_bed` (:1180) is the only caller and the phases are set at :1051, :1059,
:1091, :1132, :1216, :1233, :1268, :1352. `gold_opened_in_phases` is banked in the summary. Train
subset = tc100 tags minus `pl_split.out_cv_segments` (:1058-1064). Sentences come from
`PhonemizeWithSilJob.DbFgvZOGZQ8F` = the phonemized LibriSpeech **LM** corpus (prior.py:1-16),
unpaired text, no dev transcripts; the read bed is dev-other, disjoint from tc100. PER reference and
the bed's eligible tag list are the S1a set-selection use of gold.

## (3) The loop -- matches the spec; no silent no-op found

decode (:1157, argmax, collapse-then-drop-blank, SIL kept, type = out-1, = the cold harness's
`build_fit_items`, analysis/emc_cold_fixed_point.py:374-400, edge SIL NOT added in either) ->
`reverse.fit` on feasible items only, S1a's realised schedule (epochs 10 / lr 3e-3 / batch 8 /
seed 0 == `S1aReverseLadderJob` defaults, s1a_job.py:268-271); `fit` does NOT reset parameters, so
the warm-phi arm really starts warm and refits are cumulative -> `sample()` draws d over the masked
support (`duration_log_probs` already zeroes outside [d_min, D_k], asserted at reverse.py:405) and
one unit per frame from `emission_log_probs(eta)[b,k,c,j]` with the drawn real eta (`eta_b` and the
collage speaker come from the SAME drawn subset row, :1291-1292/:1305) -> render, four modes, every
fallback counted (`new_synthesis_counts`, :735) and no mode silently substituting another ->
CTC with blank 0, target = id+1, `zero_infinity`, loss per OUTPUT frame as init_jobs' train_step,
fresh samples per batch/pass (:1296). Stride is 1 (init_jobs.py:97), so `out_lens = s_len >= 2U`
and the `u_lens <= out_lens` assert cannot fire from d_min alone. A FRESH Adam per round (:1273) is
a deliberate choice, reported by the implementer, not a reset of anything the spec pins.
`takes_off` is `None` (not silently False) when `rho_hz` is None, with a note in the json.

CONCERN (funding): `config_sae_4a_bt_probe_v1.py:73` sets `RHO_HZ = None`, and `rho_hz` is hashed.
As launched, `rate_band` / `rate_in_band` / `takes_off` are null, i.e. the pre-registered reading is
not evaluated in the artefact (the rate itself IS reported and the clause can be applied by hand).
Setting rho after launch re-runs all eight GPU arms. Decide before launch.

## (4) Read conventions -- match S3

PER: `greedy_collapse` (:258) is `GreedyPerJob.run`'s order verbatim (collapse repeats, drop blank,
drop SIL), corpus `(S+D+I)/N` with `eval_jobs.edit_counts`, `per_macro` beside it (eval_jobs.py:
607-640). Rate: `emitted_per_sec` excludes SIL, `_with_sil` beside it, seconds = frames/50 Hz
(`FRAME_RATE_HZ`). Bed: `select_bed_tags` (:850) is `analysis/emc_target_diag.select_tags`
line-for-line (`--n-utts 300 --select stride`, gold-eligible). Gap: S3's convention recomputed
in-process on the probe's bed with `with_edge_sil`, `build_derangement`, `_clusters_by_speaker`,
`cluster_bootstrap` (10,000 / seed 42 from `s1a_job.BOOT_*`). Minor: :1192 additionally drops bed
tags absent from the eta table -- `ETA_NPZ` is documented to cover all 5,567 dev utts
(config_sae_4a_s1b_v1.py:75-80), so it should be a no-op; check `read.n_utts == 300` on the first
arm before comparing PER rows to the banked cold reads.

## (5) rqmt 32 GB -- arithmetic, and it holds

Everything runs in `run()` in ONE process, so there is no forked-probe undercount: the quoted
2.6 GB (2000 x ~634 x 1024 fp16; the HDF really is fp16, feature_dump.py:199) + 0.4 GB dev + ~40 MB
run index is the parent's own load. Two terms are missing from the comment and neither breaks it:
`load_feature_unit_subset` holds the per-utterance dict AND the concatenated copy briefly (~5.2 GB
peak), and `build_synthesis_pools:714` materialises a float32 copy of the whole train matrix for
`global_mean` (~5.2 GB transient). Peak ~8 GB + torch/CUDA, well inside 32 GB. Size the rest from
the first arm's `usage.run.1`.

## (6) s3_jobs derangement-gap pairing -- CLAIM CONFIRMED

`reverse.evaluate` (reverse.py:576-606) iterates `_length_buckets` (:569), which sorts by
`(len(z), len(y))`; rows are appended in that BUCKET order, each stamped with its input `index`.
`S3DerangementGapJob._rows` (s3_jobs.py:236-241) does `{t: r for t, r in zip(tags, rows)}` -- it
pairs by POSITION, so `own_rows[t]` / `der_rows[t]` are generally NOT utterance t's rows. Both calls
share `z` (same primary key), so the two orders differ only via `len(y)`, and both lists are
complete and identical in tag set (both `_rows` calls run over the same `tags`, `evaluate` returns
one row per item).

Consequences for the banked read `S3DerangementGapJob.Ez8bGMB7LnPX`:
* STAND: `gap_per_frame = -0.3218` (a ratio of two sums, permutation-invariant), `own_/deranged_
  log_p_per_frame`, `n_selected 314/500` and every drop counter (the drops happen before
  `evaluate`), `identical_donor_string`, `distinct_decoded_strings`.
* ALSO STANDS, by an accident that must be re-checked if either sort key changes:
  `gap_macro_mean`. Both permutations sort primarily by `len(z)` = frames, so `frames[i]` is the
  correct denominator for BOTH `lp_own[i]` and `lp_der[i]`, and the mean splits into two
  complete-set macro means -> it equals the true mean of paired differences.
* FALL: `gap_macro_ci95 = [-0.6446, -0.0170]` (the bootstrap resamples mismatched deltas and
  assigns them to the speaker of `tags[i]`, the wrong utterance) and every row of
  `per_utterance.json` (own/deranged/frames/donor/tokens mislabelled by tag). Any G4a.3 clause of
  the form "gap > 0 with the CI excluding 0" is therefore undecided from this job; the sign
  conclusion from `gap_per_frame` is not.

The probe AVOIDS the defect: `bt_probe.py:1427-1437` asserts the index set is complete and sorts
`rows` by `row["index"]` before pairing.

## Checked and clean (no finding)

sample()'s position-bucket closed form and duration mask; sentence SIL thinning (keep 0.2/0.5,
edge SIL kept) and the reservoir draw's probability (n/(i+1)); `constructed_phi_0` reproducing
RETURNN's construction order; renderer held-out split is label-free and its centroid baseline is on
the same frames; collage pools/run index purity; per-round checkpoints and json fields.
