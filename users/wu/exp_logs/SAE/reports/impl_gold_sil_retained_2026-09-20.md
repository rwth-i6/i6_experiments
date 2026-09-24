# Gold SIL share on rVAD-retained frames, dev-other (implementer, 2026-09-20)

Status: DONE. One CPU analysis script written and run to completion (~1 min, no model forward, no GPU).

## What was built

`/e/project1/spell/wu24/2026-07-13_unsupervised/analysis/gold_sil_share_retained.py`

Reuses, does not reimplement:
- `speech_llm.sae.emc.private_code.load_gold_frames` (verbatim `analysis/emc_target_vs_gold.load_gold_frames`;
  gilkeyio MFA parquet, rasterised by `repr_audit.frame_phone_labels` on the 50 Hz centre grid,
  utterance dropped if its MFA intervals need > 1 frame beyond the feature length),
  `private_code.run_lengths`, `private_code._hdf_rows`.
- The bed's rVAD streams `BlankfreeVadHdfJob.SAjz8y1cT06g` (`raw_index` / `orig_length` / `units`,
  dev-other shard0), the ones `config_sae_4a_private_code_v1._vad("dev-other")` pins.
- Retained gold = `gold[raw_index]` per utterance, exactly `PrivateCodeAnalysisJob.run`'s `frame_gold_ret`.

Frame rate assumed: 50.0 Hz (1 frame = 20 ms). SIL spelling used: `SIL`, phone id 39 of 40
(`prior.PHONES[-1]` = `repr_audit.SIL`); the training text spells the same class `<SIL>`.

Run: `/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python analysis/gold_sil_share_retained.py`
from the setup dir.

## Cross-check (mandatory) -- MATCHED

- banked `PrivateCodeAnalysisJob.cQbcIJtOamLm/output/private_code.json`: `A_frame.frames` = 261295,
  `B_units.unit_frames` = 781130, utterances = 2864.
- this run: retained 50 Hz frames = 781130, output frames (sum of ceil(retained/3)) = 261295,
  utterances = 2864 of 2864 in the VAD stream (0 dropped, 0 missing). Both totals reproduce exactly.
- Clock note: 261295 is the recognizer OUTPUT clock (60 ms, 3 retained 50 Hz frames per output frame);
  the retained 50 Hz total is 781130. Earlier sections of `extract_sil_rate_2026-09-20.md` called
  261295 "retained frames"; that is the 60 ms clock, not a subset difference.

## Numbers (dev-other, 2864 MFA-covered utterances)

1. before VAD: all frames 919980, gold-SIL 186738, fraction 0.2030 (20.30 %).
2. after VAD: retained 781130 (84.91 % of raw), gold-SIL retained 61613, fraction 0.0789 (7.89 %).
3. runs on retained frames: 186083 gold runs, 10586 SIL runs, SIL run share 0.0569 (5.69 %),
   0.0603 SIL runs per non-SIL run (175497 non-SIL runs).
4. retained gold-SIL run lengths: median 5.0 frames (100 ms), mean 5.82 frames (116 ms),
   28.32 % of runs are 1-2 frames, 12.75 % are >= 10 frames.
5. utterances: 2864 read, 32 (1.12 %) with zero retained gold-SIL frames.

Against the references supplied with the task (prior text 13.8 % `<SIL>` tokens = 0.160 SIL per
non-SIL phone; ctrl_50 ep10 decode about 4.4 % SIL tokens): the post-VAD gold sits between the two
at the token level (5.69 % of runs, 0.0603 per non-SIL run) and above both at the frame level
(7.89 %). The surviving silence is mostly short.

## Files touched

- `analysis/gold_sil_share_retained.py` (new, 200 lines).
- `recipe/i6_experiments/users/wu/exp_logs/SAE/reports/extract_sil_rate_2026-09-20.md`: appended the
  section "Gold SIL share on retained frames (2026-09-20, implementer run)" (89 -> 156 lines, existing
  content untouched), containing the printed table, the command line and the label-using-diagnostic note.
- No commit (as instructed). Nothing else under `recipe/` was modified; the `git status` entries for
  `users/schmitt/...` and `users/zeyer/...` are other people's live edits in the shared checkout.

## Caveats / assumptions

- Items 1 and 2 are over the same utterance set (the MFA-covered subset of the VAD stream), so the
  before/after fractions are comparable; here that subset is all 2864 dev-other utterances.
- A run on retained frames can span a VAD-trimmed gap (adjacent after trimming). That is the intended
  reading: it is the silence the recognizer sees, in its own trimmed time. Stated in the script docstring.
- Nothing in the spec was left undetermined; the only judgment call was reporting the retained total on
  both clocks so the 261295 cross-check is unambiguous.
