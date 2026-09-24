# S3b-BT-aux, FULL TRIGRAM: implementer report, 2026-09-16

Checkout `recipe/2025-10-speech-llm`, branch `haotian_modality_matching_jupiter`.
Commits (not pushed): `b068d9b` (builder knobs + tests), `4d6f33c` (arms + pack v3).
Round ended on a turn limit; item 6 (pack v4 + its test + its workspace entry) is NOT STARTED.

## Status per dispatched item

| # | item | status |
|---|---|---|
| 1 | trigram table job | **NOT BUILT -- deliberately; see "Deviation" below** |
| 2 | builder knobs (`prior_history` / `lattice_reduction` / `lattice_checkpoint`) + tests | DONE |
| 2b | `random_seed` builder knob (scope addition) | DONE |
| 3 | four held arms `lam3_tri` / `bt_a_tri` / `bt_b_tri` / `bt_c_tri` | DONE |
| 4 | `config_sae_4a_s3b_pack_v3.py` + `test_pack_config3.py` + `config/sae_4a_s3b_pack3.py` | DONE |
| 5a | four held CR / seed arms `bt_b_tri_cr` / `lam3_tri_cr` / `bt_b_tri_s2` / `lam3_tri_s2` | DONE (built + hashed + config-diffed) |
| 5b | `config_sae_4a_s3b_pack_v4.py` + `test_pack_config4.py` + `config/sae_4a_s3b_pack4.py` | **NOT STARTED** |
| 6 | full emc test suite | DONE |
| 7 | hash census | DONE |
| 8 | commit by explicit path, no push, no launch | DONE |

## Deviation from item 1: no trigram table job was built

The dispatch asks for a sisyphus CPU job that counts trigrams over the banked prior text and writes
a table `PriorHistory("trigram")` consumes. **That table is already banked.** The banked fit
`work/speech_llm/sae/emc/prior/PhoneNgramPriorJob.TRPE0D5nF3bh/output/prior.npz` has keys
`['log_uni', 'log_bi', 'log_tri', 'phones', 'meta']`, with `log_tri` of shape (1681, 40) --
41 x 41 histories x 40 successors, rows indexed `h2 * 41 + h1`, which is the lattice's own
`outer * n_ctx + last`. The same job's `prior.stats.txt` records `held_ppl_order2 = 14.231361639391373`
and `held_ppl_order3 = 9.468940253825659`, i.e. the 9.4689 the dispatch asks to reproduce, from the
SAME corpus, the SAME 1,010,000-line window with every 101st line held out and the SAME interpolated
Witten-Bell as the bigram the launched arms already use. `definitions/sae_emc._prior_table` returns
`prior.log_tri` for `history == "trigram"` and its docstring already states "no new LM fit and no new
upstream job for any of them".

A second counting job would have produced a second, differently-provenanced table for the same
quantity and would have cost a job slot for nothing, so it was not built; the trigram arms read the
same `prior_npz` path the bigram arms read. What was added instead is a CHECK, not a fit:
`test_prior.test_the_banked_prior_carries_the_trigram_table_the_arms_consume` asserts the banked
table's shape, that its rows normalize to 1e-12, and that `prior.stats.txt` carries exactly those two
perplexities (it SKIPs if the npz is absent). The independent recomputation of both perplexities is
`analysis/prior_order_ppl.py` (max abs diff 0.0 on both tables, reports/impl_prior_order_ppl).

If the orchestrator wants the job anyway (e.g. to make the table's provenance a graph edge rather
than a banked path), it is a separate delta and I did not build it.

## Item 2: the builder knobs

`emc_train_jobs.py`:

* `EMC_LATTICE_REDUCTION = "auto"` / `EMC_LATTICE_CHECKPOINT = 0` -- literals, because the sisyphus
  manager has no torch and cannot import `emc.lattice`; `test_emc_train_jobs` asserts both equal
  `lattice.lattice_loss`'s own signature defaults.
* `EMC_RANDOM_SEED = 42` -- RETURNN's own default (`returnn/torch/engine.py:1238`, read immediately
  before `get_model`, so it IS the model seed: theta's and phi's init, the dropout draws, the
  shuffling).
* `build_emc_train_config(..., lattice_reduction=, lattice_checkpoint=, random_seed=)`. All three are
  HASH-NEUTRAL BY OMISSION: nothing is written at the default value, so every pre-existing config is
  bit-identical. The two DP keys go into `model_args`; `random_seed` goes into `config` and is
  asserted NOT to reach `model_args`.
* A history above the bigram is REFUSED without a stride (`lattice_checkpoint = S > 0`), because its
  un-checkpointed forward table is ~52 GiB at B 125 / T_pad 704.

Tests added: `test_the_dp_knobs_are_hash_neutral_by_omission`, `test_the_model_seed_is_hash_neutral_by_omission`
(both in `test_emc_train_jobs`), and `test_a_trigram_history_step_runs_under_d3_and_d4` in
`train_steps/test_sae_emc.py` -- a real CPU train step at `prior_history = "trigram"`,
`lattice_reduction = "matmul"`, `lattice_checkpoint = 3` on a tiny batch, which asserts the DP was
called with `history.n_hist == 1681` and a (1681, N_TYPES) prior table and that the returned
posteriors sum to 1 per frame (measured 4.8e-07).

## Items 3 / 5a: the eight held arms (single-arm `ReturnnTrainingJob` ids)

First node (`TRIGRAM_DP = prior_history "trigram", lattice_reduction "matmul", lattice_checkpoint 32`):

| arm | id | what it adds to its bigram twin |
|---|---|---|
| `lam3_tri` | `ReturnnTrainingJob.9peamnR211qH` | trigram, lam_bt 0 (control) |
| `bt_a_tri` | `ReturnnTrainingJob.PGxDAEONvJmq` | trigram, lam_bt 0.1 full |
| `bt_b_tri` | `ReturnnTrainingJob.WMUxmtdf0Wx1` | trigram, lam_bt 0.3 full |
| `bt_c_tri` | `ReturnnTrainingJob.HRwyxrczY6xM` | trigram, lam_bt 0.3 output_only |

Second node (USER ADDITION 2026-09-16, `TRIGRAM_VARIANT_ARMS`; not yet packed):

| arm | id | twin | its one key |
|---|---|---|---|
| `bt_b_tri_cr` | `ReturnnTrainingJob.WnrqZ9Ny6M7E` | `bt_b_tri` | `lam_cons 0.3`, `cons_views ["specaug"]` |
| `lam3_tri_cr` | `ReturnnTrainingJob.7X38hg3FYoJs` | `lam3_tri` | `lam_cons 0.3`, `cons_views ["specaug"]` |
| `bt_b_tri_s2` | `ReturnnTrainingJob.vWCDsWIJV1Su` | `bt_b_tri` | `random_seed 43` |
| `lam3_tri_s2` | `ReturnnTrainingJob.leZu891PYYm6` | `lam3_tri` | `random_seed 43` |

Verified by serialized-config diff, twin against twin: the CR arms add exactly
`'lam_cons': 0.3, 'cons_views': ['specaug']` to `model_args` and nothing else; the S2 arms add
exactly the config line `random_seed = 43` and nothing else. The speed view is deliberately absent,
so no second feature stream is declared and the per-key batch limit -- and therefore the batching --
is bit-identical to the twin's.

`ACTIVE_ARMS = ("lam3",)` is untouched, so none of the eight builds in the rate graph.

**lam_cons = 0.3**, per the coordinator's amendment of 2026-09-16 (the weight-1 S3b-C arms are the
ones that collapsed onto 3-4 symbols). Stated once as `LAM_CONS_TRI`.

### The one undetermined constant

`SECOND_MODEL_SEED = 43`. The plan asks for "a second model seed" and names no value; RETURNN's
default is 42, so 42 + 1 is stated in ONE place in the rate config, with a comment saying it is the
arbitrary-but-declared draw. Changing it moves `bt_b_tri_s2` and `lam3_tri_s2` and nothing else.
**If the orchestrator wants a specific seed, it is a one-line change and both ids move.**

## Item 4: pack v3

`config_sae_4a_s3b_pack_v3.py`, workspace entry `config/sae_4a_s3b_pack3.py`,
test `sae/emc/test_pack_config3.py`.

* `PackedEmcTrainJob.byYMQmBNEpLZ`, arms `{bt_a_tri, bt_b_tri, bt_c_tri, lam3_tri}`,
  rqmt **gpu 4 / cpu 64 / mem 256 / gpu_mem 96 / time 8.0 h**.
* Budget derivation (in the file, as a comment):
  `0.69 h` (pack v1's slowest arm, MEASURED, over 8 sub-epochs) `x 6.9` (the trigram step factor:
  10.31 s/step at D3 matmul + D4 S = 32 against the bigram's 1.48 s/step, both at B 125 / T_pad 704)
  `= 4.761 h`, plus `0.345 h` for the BT steps (one CTC forward/backward on 128 collage sentences,
  NO lattice DP, so NOT scaled by the prior order; sized as pack v2's own `MEASURED_ARM_HOURS x
  (BT_STEP_TIME_FACTOR - 1)`), `= 5.106 h`, `x 1.1` shared-node margin `= 5.617 h` derived need.
  The ask is 8.0 h, 1.4x that, because the 6.9 is a lattice-only bench and the 0.345 has never been
  timed on a GPU; well under the 11.5 h `settings.check_engine_limits` clamp.
* **Batch budget: PASSES UNCHANGED at max_seqs 128.** Nothing was changed. `lattice.check_batch_budget`
  caps B (`MAX_UTTS_PER_BATCH` 128) and B x T_max (`MAX_BATCH_FRAMES` 128,000), both independent of
  |h|; `max_seqs 128` / `batch_size 88,000` respect them exactly as they did at the bigram. Only
  `lattice.estimate_peak_gib` is a bigram fit and under-predicts the trigram peak (measured 9.90 GiB
  at S = 32, a tenth of a GH200) -- it is a diagnostic, not the guard.
* Reads: pack v2's full per-arm set, per sub-epoch, under
  `output/exp2025_11_06_speech_llms/librispeech/sae_4a_s3b_pack3/<arm>/ep<k>/<split>/`; 473 registered
  outputs. No arm carries a content head, so no MFCC chain is built and this graph borrows nothing
  from the source stage's prefix (asserted).
* Jobs: the pack graph has **280** jobs, **253** of which the rate graph does not already contain
  (the pack itself plus the four arms' per-sub-epoch reads, gaps and selections).

## Item 5b: pack v4 -- NOT STARTED

What remains, verbatim: `configs/config_sae_4a_s3b_pack_v4.py` over exactly
`{bt_b_tri_cr, lam3_tri_cr, bt_b_tri_s2, lam3_tri_s2}`, mirroring pack v3 (the arm-set assert should
read `set(PACK_ARMS) == set(rate.TRIGRAM_VARIANT_ARMS)`, the per-arm kwargs are
`dict(lam_rate=..., rate_rho_hz=..., **bt, **rate.TRIGRAM_DP, **rate.TRIGRAM_VARIANT_ARMS[tag])`, the
prefix `sae_4a_s3b_pack4`); `sae/emc/test_pack_config4.py` with `REFERENCE_IDS` = the four ids in the
table above; workspace `config/sae_4a_s3b_pack4.py`. The budget derivation is pack v3's plus whatever
the consistency term's extra augmented forward costs -- **that factor is not measured anywhere I could
find and I did not invent one**; pack v3's 8.0 h ask already has 1.4x headroom over its derived need,
which is the natural place to start, but the number needs an orchestrator ruling or a measurement.

The rate config is already complete for it: the four arms exist, are held, hash, and were
config-diffed against their twins. Pack v3's arm-set assert already subtracts
`rate.TRIGRAM_VARIANT_ARMS`, so a pack v4 cannot leak arms into pack v3.

## Item 6: tests

All run from the setup dir with `black` on PATH, conda python `env/conda/envs/speech_llm/bin/python`.

| module | result | checks |
|---|---|---|
| `sae.emc.test_prior` | PASS | 13 (1 new: the banked trigram table) |
| `sae.emc.test_lattice` | PASS | 39 (incl. the 20 banked bigram digests bit-for-bit) |
| `sae.emc.test_emc_train_jobs` | PASS | 13 (2 new: DP knobs, model seed) |
| `sae.emc.test_bt_aux` | PASS | 10 |
| `prefix_lm.model.train_steps.test_sae_emc` | PASS | 12 (1 new: trigram step under D3 + D4) |
| `sae.emc.test_pack_config` | PASS | 6 |
| `sae.emc.test_pack_config2` | PASS | 7 |
| `sae.emc.test_pack_config3` | PASS | 9 (new module) |

`test_pack_config4` does not exist yet.

## Item 7: census

`scripts/sae_4a_cons_census.py`, before/after the whole change:

* `s3` **98** jobs -- byte-identical
* `phase` **1021** jobs -- byte-identical
* `rate` **91** jobs -- identical apart from eight ADDED comment lines
  `# arm <tag>: not built (ACTIVE_ARMS = ('lam3',))`, one per new held tag. No job id moved.
* `ReturnnTrainingJob.DF6blPpto23t` (the launched lam3) present and unmoved.
* `PackedEmcTrainJob.SeZzGUScxq4x` (pack v1) and `PackedEmcTrainJob.wK3hCW0JdJ6G` (pack v2, the
  bigram BT node that is NOT launched) both rebuild at their banked ids.
* New job: `PackedEmcTrainJob.byYMQmBNEpLZ` (pack v3) plus the 252 read jobs of its four arms.

## Files touched

| file | delta |
|---|---|
| `sae/emc/emc_train_jobs.py` | +`EMC_LATTICE_REDUCTION` / `EMC_LATTICE_CHECKPOINT` / `EMC_RANDOM_SEED`, +3 builder kwargs, the trigram-needs-a-stride refusal |
| `sae/emc/test_emc_train_jobs.py` | +2 tests; the existing prior-history test now passes `lattice_checkpoint=32` on its non-bigram calls |
| `sae/emc/test_prior.py` | +the banked-trigram-table test |
| `prefix_lm/model/train_steps/test_sae_emc.py` | +the trigram D3+D4 train-step test |
| `configs/config_sae_4a_s3b_rate_v1.py` | +8 held arms, `TRIGRAM_DP` / `TRIGRAM_ARMS` / `TRIGRAM_VARIANT_ARMS` / `TRIGRAM_VARIANT_TWIN` / `LAM_CONS_TRI` / `SECOND_MODEL_SEED`, +build-time asserts; `ACTIVE_ARMS` unchanged |
| `configs/config_sae_4a_s3b_pack_v3.py` | NEW |
| `sae/emc/test_pack_config3.py` | NEW |
| `config/sae_4a_s3b_pack3.py` (workspace, untracked dir) | NEW |

Not touched and not staged: `config_sae_1g_v1.py` and `config_sae_3e1_d6_swap_cont_v1.py` are
another implementer's live working-tree changes in the same checkout; they were left alone.

Nothing was launched, nothing was pushed, `settings.py` was not touched.
