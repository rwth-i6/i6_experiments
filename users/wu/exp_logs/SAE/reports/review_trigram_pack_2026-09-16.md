# Review: trigram EMC wiring + pack v3, 2026-09-16 (pre-launch, read-only)

Verdict: **PASS**. Two non-blocking notes at the end. Commits b068d9b + 4d6f33c on
`haotian_modality_matching_jupiter`, checkout `recipe/2025-10-speech-llm`. Nothing launched, no
tracked file edited; everything below was rebuilt in this review, not taken from the implementer
report.

## 1. The trigram really runs in the DP

Traced end to end, `arm dict -> builder -> model_args -> model -> lattice call`:

* `config_sae_4a_s3b_rate_v1.py:290` `TRIGRAM_DP = dict(prior_history="trigram",
  lattice_reduction="matmul", lattice_checkpoint=32)` -> `**tri` at `:539` of the same file.
* `emc_train_jobs.py:820-847`: `prior_history != "bigram"` writes `prior_history` AND
  `prior_order = 3`; the two DP knobs are written only off their defaults; a non-bigram history
  without a stride is refused at graph build.
* `definitions/sae_emc.py:234-344`: the constructor NAMES all four keys, and `:266-271` raises on
  any `model_args` key it does not name, so an unplumbed knob is a job-start error, never a silent
  no-op. `_prior_table` returns `prior.log_tri` at `"trigram"`; `:342` asserts the table is
  `(n_hist, n_types)`.
* `train_steps/sae_emc.py:266-273`: `dp_kwargs` carries `history`, `reduction=model.lattice_reduction`,
  `checkpoint=model.lattice_checkpoint` into `lattice_loss`; names match
  `lattice.py:1228-1245`/`987-1003`. `rate_term._fd_passes` copies the SAME dict
  (`rate_term.py:625`), so the two (central-difference) tilted passes also run D3+D4 -- had it
  stripped `checkpoint`, the tilted pass alone would have been the ~52 GiB table.
* History size and layout: `lattice.trigram_history` -> `_prior_history("trigram", cfg,
  arange(41), 41, bos_id)` => |h| = 41 x 41 = 1681, row = `outer*41 + last` = `p_-2*41 + p_-1`,
  start = (BOS, BOS). The banked table is built with rows `h2*N_CTX + h1` (`prior.py:327`) and each
  line padded with two BOS (`prior.py:217-225`): same convention. SIL is an ordinary symbol in both
  (PriorHistory docstring "Blank / repeat / SIL"; `prior.py` inventory "39 ARPAbet + SIL").
* Banked table read directly off disk (not via the implementer's test):
  `PhoneNgramPriorJob.TRPE0D5nF3bh/output/prior.npz` has `log_tri` (1681, 40) float64, all finite,
  per-history row sums 0.9999999999999997 .. 1.0000000000000004. `prior.stats.txt` records
  `held_ppl_order3 = 9.468940253825659` (order2 14.231361639391373), 1,010,000 lines read /
  1,000,000 counted / 10,000 held at stride 101, interpolated Witten-Bell. Every written arm config
  points at that npz path.
* lam3_tri vs lam3: rebuilt both and diffed the written configs (`ACTIVE_ARMS` monkey-patched in
  memory only). Exactly 5 changed lines: `prior_order` 2 -> 3, plus `prior_history`,
  `lattice_reduction`, `lattice_checkpoint`. Same 5-line diff for bt_a/b/c -> *_tri.

## 2. Hash neutrality

Rebuilt and printed the ids myself: lam3 `DF6blPpto23t`, bt_a `6G8CpcswqlTR`, bt_b `NSxRSjyBWp5g`,
bt_c `UjJpp4iQ4kkT`, lam3_ct `DhsiFoHwBP5l`, pack v1 `PackedEmcTrainJob.SeZzGUScxq4x`, pack v2
`wK3hCW0JdJ6G` -- all at their banked values with the new builder kwargs in place, i.e. the
by-omission rule holds (`test_pack_config` and `test_pack_config2` both pass unchanged). I did not
re-run the 3-graph census; the ids above are the load-bearing part of it.

## 3. The eight arms

Built all twelve arms (4 bigram + 8 new) and diffed each twin pair:

| pair | changed lines |
|---|---|
| lam3/bt_a/bt_b/bt_c -> *_tri | 5: prior_order 3 + the three trigram keys |
| bt_b_tri -> bt_b_tri_cr, lam3_tri -> lam3_tri_cr | 2: `"lam_cons": 0.3`, `"cons_views": ["specaug"]` |
| bt_b_tri -> bt_b_tri_s2, lam3_tri -> lam3_tri_s2 | 1: `random_seed = 43` |

Ids: 9peamnR211qH / PGxDAEONvJmq / WMUxmtdf0Wx1 / HRwyxrczY6xM / WnrqZ9Ny6M7E / 7X38hg3FYoJs /
vWCDsWIJV1Su / leZu891PYYm6 -- the reported ones. BT terms in the written configs: bt_a_tri
`lam_bt 0.1 / bt_depth "full"`, bt_b_tri `0.3 / "full"`, bt_c_tri `0.3 / "output_only"`, all with
`bt_ramp_epochs 4`, `bt_batch_sents 128`, `torch_batching = bt_interleaved_batching`; the control
has no `lam_bt` key at all. `max_seqs 128` / `batch_size 88000` identical across all four.

Seed: RETURNN `returnn/torch/engine.py:1238-1245` reads `config.int("random_seed", 42)` and calls
`rf.set_random_seed` -> `torch.random.manual_seed` (torch backend `_backend.py:56`) immediately
before `get_model`, so 43 does reach module init, not only the data order. 42 is RETURNN's own
default, so omitting the key is exactly the old behaviour.

## 4. Pack v3

Rebuilt the graph (`speech_llm.sae.emc.test_pack_config3`, all 6 checks pass, plus my own rebuild):

* `PackedEmcTrainJob.byYMQmBNEpLZ`, rqmt `{gpu 4, cpu 64, mem 256.0, time 8.0, gpu_mem 96}`.
* Per-arm written config BYTE-IDENTICAL (up to each job's own output dir) to 9peamnR211qH /
  PGxDAEONvJmq / WMUxmtdf0Wx1 / HRwyxrczY6xM, and the single-arm side is built by the source config
  itself, not by a retyped kwargs copy.
* 473 registered outputs under `.../sae_4a_s3b_pack3/` = 4 arms x 118 (8 sub-epochs x 13 files +
  6 arm-level) + the units stats -- the same per-arm/per-sub-epoch set pack v1/v2 register; nothing
  under the rate, BT, cons, pack, pack2 or pack4 prefixes (asserted, and it holds because no arm
  here carries a content head).
* Budget: 0.687 h is real (pack v1 slowest arm `spec_speed`, 269-296 s x 8 sub-epochs,
  review_pack2_config_2026-09-15.md:58). 10.31 s/step and 9.90 GiB trace to
  impl_lattice_knobs_2026-09-15.full.md:109 (post-float64-fix column; the pre-fix 7.89 s is the
  earlier number), and the bigram control in the SAME post-fix run is 1.4968 s, so the factor 6.9 is
  the measured ratio. 0.69 x 6.9 + 0.345 = 5.106, x 1.1 = 5.617 h derived, 8.0 h asked, 11.5 h clamp.
  This is conservative twice over: the ratio is DP-only while the measured 0.69 h also contains the
  recognizer/phi/loader time that does NOT scale with |h| (a sub-epoch here is ~28 steps of ~10 s at
  the bigram, of which ~4.4 s is the three DP calls), and the BT term is priced as a whole extra half
  EMC step. 8.0 h is enough.
* Memory: the three DP calls per step are SEQUENTIAL at this batch (`fd_batch_fits` rejects the
  stacked 2 x 125 call against `MAX_UTTS_PER_BATCH = 128`), so the peak stays at the measured
  ~9.9 GiB + model on a 96 GB GH200, four arms on four separate GPUs, 64 GB host RAM per arm.
  `check_batch_budget` (B <= 128, B x T_max <= 128,000) is |h|-independent and unchanged at
  88,000 frames; `estimate_peak_gib` is a bigram fit and is only a message, never a guard.

## 5. Beyond the intended delta

Both commits touch only the listed files; the two dirty files in the checkout
(`config_sae_1g_v1.py`, `config_sae_3e1_d6_swap_cont_v1.py`) are another implementer's and are not
staged. `ACTIVE_ARMS` untouched (lam3 alone builds in the rate graph). Workspace entry
`config/sae_4a_s3b_pack3.py` is pack2's entry with the module and prefix swapped. No change to
`settings.py`.

### Non-blocking notes (no effect on this launch)

1. `sae/emc/lattice.py:308-313` (`trigram_history` docstring) still says the full trigram is
   "CORRECTNESS ONLY ... not an operating point on this code path" and quotes the superseded ~70 GiB
   / 14x figures. The next reader of that function is told the opposite of what the four funded arms
   do; a one-line docstring fix when something else touches the file.
2. `*_s2` (pack v4, not built yet): theta is loaded from the frozen flat init
   (`recognizer_checkpoint=flat_init.out_checkpoint`), so `random_seed = 43` varies phi's init,
   dropout, SpecAugment masks and shuffling, but NOT theta's init. The implementer report and
   `config_sae_4a_s3b_rate_v1.py:283-287` call it "theta's and phi's init". The w2v-U 2.0 seed
   spread it is compared against includes full-init variation, so this arm pair measures a narrower
   spread than that anchor; say so when the number is read. `SECOND_MODEL_SEED = 43` is still an
   undetermined constant (declared, not derived) and moves both *_s2 ids if the orchestrator picks
   another value.

Checked and found nothing wrong: prior table provenance and normalisation, history/BOS/SIL
conventions, the knob plumbing (including the tilted FD passes), all eight twin diffs, all banked
ids, the pack rqmt/budget/read set/alias layout, and the absence of gold data in training or in the
selection (reads use gold only for PER scoring and the derangement tag list, as in S3b-R).
