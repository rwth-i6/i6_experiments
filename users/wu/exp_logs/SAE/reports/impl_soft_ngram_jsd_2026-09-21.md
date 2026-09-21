# n-gram JSD read of the soft pack's private code (implementer, 2026-09-21)

**DONE_WITH_CONCERNS.** One `NgramModeSeekingJob` registered, not launched.
Job: `NgramModeSeekingJob.KmbcDX5k6aaS`
(`work/i6_experiments/users/wu/experiments/unsupervised_asr/ngram_mode_seeking/NgramModeSeekingJob.KmbcDX5k6aaS`).

## Files

- `/e/project1/spell/wu24/2026-07-13_unsupervised/recipe/2025-10-speech-llm/src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_soft_ngram_v1.py`
  (new): the read. Committed as `c7efb59` on `haotian_modality_matching_jupiter`, explicit path, not
  pushed. No other file in that checkout was touched (`config_sae_1g_v1.py` and
  `config_sae_3e1_d6_swap_cont_v1.py` are other people's uncommitted work and were left alone).
- `/e/project1/spell/wu24/2026-07-13_unsupervised/config/sae_4a_soft_ngram.py` (new shim, setup dir,
  untracked like the other shims): `sis m config/sae_4a_soft_ngram.py`.
- `NgramModeSeekingJob` itself (`recipe/i6_experiments/.../unsupervised_asr/ngram_mode_seeking.py`)
  was NOT edited: any edit there moves its file sha and re-hashes the banked
  `NgramModeSeekingJob.vlhnotaFKKj5` priorshuf read.

## What it registers

The reference read verbatim (`config/sae_4a_attrib_ngram_priorshuf.py`, SAE_4A_attrib.md Results
2026-09-19): text side `SampleLinesJob.orN768ARKwlt/output/text.phn.gz` SIL stripped, its trigram
`PhoneNgramPriorJob.RtzbESkOedsT/output/prior.npz` for the SIL-inclusive secondary, corpus window
(1,000,000 counted / 10,000 held / stride 101), JSD orders 1-4, utterance-block bootstrap 1000
resamples seed 0, count matching, `jsd4_reference_line` 0.27, reference row = gold
`GoldPhonesJob.ZGSp0hxyd2YP/output/gold.json` split `dev-other`. No new statistic. (The dispatch
wrote the window hash as `orN768ARWklt`; the job on disk and in the reference config is
`orN768ARKwlt` and that is what is used -- it is also the soft pack's own prior window:
`output/.../sae_4a_soft_pack/prior/prior.npz` resolves to `RtzbESkOedsT`.)

19 rows: 6 arms x kept epochs 1, 4, 10, plus gold. Each row gets `greedy_phones.json` (primary) and
`greedy_raw.json` (secondary) of the dev-other decode, consumed as a frozen `tk.Path` into the
producing `BlankfreeGreedyPerJob` work dir under
`work/speech_llm/sae/emc/blankfree_eval_jobs/BlankfreeGreedyPerJob.<hash>/output/`:

| arm | ep1 | ep4 | ep10 |
|---|---|---|---|
| sf_20 | FWYN1sAPjwtN | hdEcYOx00EV8 | eQVw2XrcOWXg |
| soft_20 | Igtq66KuGgTG | Aru0VCh2WymN | ZzEhEJ2ghmA7 |
| soft_20_s1 | Yh4E5VBoTq9m | fCBwUXQqawI4 | 59MMKl1PFa6G |
| softshuf_20 | YDIjHjEL8AMm | afruyJiCbs2b | aP10M9b5T6bF |
| ctrl_20 | jpnkzOpo2Qsi | 6kcdsTuiciXr | vhscomRFFJIJ |
| ctrl_20_s1 | LvlxfSEbxv0a | 3D4UUlHfOLCu | mUXW2ACkKzDa |

Every hash was read off the pack's / prepro's registered `ep<k>/dev-other/` outputs and then
cross-checked against the `per_a` / `per_b` parameters in the `info` of the pack's own
`PairedPerDeltaJob`s (checked: soft_20_vs_ctrl_20 ep10, sf_20_vs_ctrl_20 ep4,
softshuf_20_vs_ctrl_20 ep1, soft_20_s1_vs_ctrl_20_s1 ep10, sf_20_vs_soft_20 ep10), so the rows are
the strings the banked paired PER deltas scored. All 18 decode jobs carry `finished.tar.gz`.

Registered outputs: `output/sae/4a/soft/ngram_mode_seeking.json` and
`output/sae/4a/soft/summary_ngram.md`; alias `sae/4a/soft/ngram_mode_seeking`.

## Epoch 20

Not registered. No ep20 decode exists for any of the four arms (the pack is still training) and
`rows` is a hashed constructor argument, so ep20 is a SECOND registration later. Because it will be
a separate job, adding it does not move this job's numbers. The two ep20 control decodes that do
exist are recorded in the module as `EPOCH20_DECODES` (ctrl_20 `9GnrDF23mUuG`, ctrl_20_s1
`MtzQhyVFh6xx`).

## Checks run

1. Config loads and census: `sis console -c ... config/sae_4a_soft_ngram.py` -> `NJOBS 1`, the one
   job being `NgramModeSeekingJob.KmbcDX5k6aaS`. No pack job, no decode job, no prior job in the
   graph -- every input is a frozen path, so nothing upstream re-hashes or re-runs.
2. Parameters printed off the constructed job: reference row `gold`; 19 rows as listed;
   `comparison_rows {'ep4': 'soft_20_ep10', 'gan': 'ctrl_20_ep10', 'gold': 'gold'}`; bootstrap
   1000 / seed 0 / orders (1,2,3,4); window 1000000 / 10000 / 101; corpus and prior paths as above.
3. Row sanity, off disk, with the job's own `load_phone_json` and `emc.prior`: each of the 18 decode
   rows covers exactly the 2864 gold dev-other ids (0 missing, 0 extra), every symbol is inside the
   39 ARPAbet monophones, no SIL token in the primary strings. Phone counts: ep1 48,676-68,041;
   ep4 168,429-175,688; ep10 166,597-169,862; gold 177,275.
4. Cost projection: the reference 6-row instance used 0.0157 h and 0.30 GiB peak (its
   `usage.run.1`); the class asks `cpu 2, mem 12, time 1`, ample for 19 rows.

Loading and these checks do not show the numbers are right; only the run does. Not launched.

## Concerns / assumptions for the planner

- **The rendered PASS/FAIL column does not apply here.** `NgramModeSeekingJob` requires
  `comparison_rows` under step 1's `ep4`/`gan`/`gold` slots and renders four differences with step
  1's pre-registered cold-cycle-vs-GAN margins. The dispatch did not say which rows fill the slots.
  I filled them with the pair the question is about at the latest kept epoch: `ep4` -> `soft_20_ep10`,
  `gan` -> `ctrl_20_ep10`, `gold` -> `gold`. The decisive number is then comparison (c),
  "soft_20_ep10 minus ctrl_20_ep10, 4-gram JSD", a paired count-matched difference bootstrap
  (positive = the scorer arm's decode is farther from the text than the frozen control's); (b) is
  soft_20_ep10 minus gold and (d) ctrl_20_ep10 minus gold. The verdict cells beside them are step
  1's margins and must not be quoted as a gate for this phase; the module docstring says so.
  If the planner wants a different pair in those slots, that is a one-line change and a new hash.
- **The count-matching budget is set by an ep1 row.** Pooling all epochs into one job (what the
  reference read did, and what makes the epochs comparable) matches every row down to 48,676 phones
  -- softshuf_20 ep1 -- i.e. ~29% of an ep10 row's tokens, so the rendered ep10 difference is read
  on ~48.7k phones per row and its CI is correspondingly wider. The unmatched full-count table is
  banked as the secondary. If the planner wants the ep10 contrast at near-full count, a second,
  ep10-only registration (budget ~166.6k) would give it; I did not register one (scope).
- Per-row JSD n = 1..4 for every arm and epoch -- the number the user actually asked for -- is the
  main table of `summary_ngram.md` and is unaffected by both points above except through the budget.

## Follow-up: one job per epoch beside the pooled one (2026-09-21, commit f0434bf)

DONE. Same config module, same conventions, same frozen decode paths; only the row partition
changes. Four registrations now:

| job | rows | budget row | outputs |
|---|---|---|---|
| `NgramModeSeekingJob.w4NlQzcoXIfd` | six ep1 arms + gold | softshuf_20 ep1, 48,676 phones | `sae/4a/soft/ep1/` |
| `NgramModeSeekingJob.E4gg11FZ1itE` | six ep4 arms + gold | ctrl_20_s1 ep4, 168,429 phones | `sae/4a/soft/ep4/` |
| `NgramModeSeekingJob.EKkpTHPdLKGD` | six ep10 arms + gold | soft_20 ep10, 166,597 phones | `sae/4a/soft/ep10/` |
| `NgramModeSeekingJob.KmbcDX5k6aaS` (pooled, unchanged hash) | all 18 + gold | softshuf_20 ep1, 48,676 phones | `sae/4a/soft/` |

`comparison_rows` per epoch job = that epoch's `soft_20`, `ctrl_20`, gold, so comparison (c) is
`soft_20_ep<N> minus ctrl_20_ep<N>`, 4-gram JSD; at ep10 it is now read at ~167k phones per row
instead of ~48.7k. The budget rows above are the phone counts measured in check 3; the job computes
its own. Across the three per-epoch jobs the matched numbers sit at different budgets and are NOT
comparable to each other -- the pooled job is the cross-epoch view, and its budget still matches the
2026-09-19 read's arrangement. The inherited PASS/FAIL cells still decide nothing (unchanged).

Checks: `sis console` census on `config/sae_4a_soft_ngram.py` -> `NJOBS 4`, exactly the four hashes
above, no pack job; the pooled job's rows, comparison_rows and hash are byte-for-byte what they were
at c7efb59; all eight registered outputs resolve to the right job dirs (`sae/4a/soft/ep{1,4,10}/`
and `sae/4a/soft/` x `{ngram_mode_seeking.json, summary_ngram.md}`). Not launched. Each job costs
about what the 6-row reference instance did (0.016 h, 0.30 GiB), well inside `cpu 2, mem 12, time 1`.
