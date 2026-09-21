# Code review -- SAE 4A lexlat round 2 (2026-09-21)

Reviewed at speech-llm **HEAD = f0ab314** (+ dependency a05ab16) on
`haotian_modality_matching_jupiter`.  The "adaptive `_dest_block`" change was NOT a commit while I
read: it sits UNCOMMITTED in the worktree (`git diff HEAD -- src/speech_llm/sae/emc/lexlat.py`,
24+/14-, plus two new tests in `test_lexlat.py`); it is reviewed below as in-flight.

Verdict: **APPROVE_WITH_AMENDMENTS**.  Read-only; nothing edited, nothing launched.

## Blocking

**B1. `config/sae_4a_lexlat_e1.py` registers nothing -- E1 can never launch.**
The shim is `from ...config_sae_4a_lexlat_pack_v1 import py_e1` / `run = py_e1`; there is no name
`py` in the module.  `tools/sisyphus/sisyphus/loader.py:62-75` takes the entry point from the file
name, i.e. literally `py`, and on `AttributeError` with `function_name == "py"` it only logs
"No function named 'py' found in module ..." and calls nothing.  Importing the module registers no
job (everything in the config module is inside functions).  So `sis m -io config/sae_4a_lexlat_e1.py`
loads an EMPTY graph and the pack's funding gate silently never runs.  Every other alternate-entry
shim in `config/` uses the wrapper idiom (`config/sae_4a_prepro_devother.py`, `sae_4a_prepro_null.py`,
`sae_4a_prepro_data.py`: `def py(): return py_<x>()`).  Fix: `def py(): return py_e1()`.
The shim is an untracked workspace file and is in neither commit (implementer's item 6.4), so this
is invisible in `git show`.

**B2. At the pack's run shape the expansion block is not bounded by a memory budget, and E1 is
likely to die with CUDA OOM instead of returning a measurement.**
`lexlat_train.derive_max_candidates` (`lexlat_train.py:115-133`) multiplies the per-utterance arc
bound by `batch`, so the guard scales with B.  The in-flight `_dest_block`
(`lexlat.py:1028-1058`) returns `max(1, min(DEST_BLOCK=1024, n_new, max_candidates // (B*m_max)))`
-- and `max_candidates // (B*m_max)` is INDEPENDENT of B (both numerator and denominator carry B),
so at the pack's B = 128 the 1024 cap binds exactly as it does at B = 1.
One block then materialises three `[B, O, n_blk*m_max]` float64 tensors (`_slot_u` + the eager
`torch.where` in `_step`, `lexlat.py:1157-1162`).  Calibrating on the E0 OOM report (37.4 GiB at
B = 1, O = 51, n_new = 16384, m_max = 6014) gives 53.5 MB per unit of `m_max` at B = 128,
O = 51, n_blk = 1024: at the census's C = 1024 frame width `m_max ~ 376` that is **~20 GiB per
tensor, ~60 GiB for the three**, before the `[B, n_cand]` slot columns (`cols` in `_frame_slots`,
~9 columns x B x several 1e5 -> several GB) and before the bed's own tensors.  That is at or over
the 80 GiB bar, and an allocator failure raises rather than reporting `memory_pass = False`, so the
E1 GPU-h buys no number and the declared ladder (C = 512, word bigram) is re-measured only after a
code change.  The guard should be a BYTE budget independent of B (or `DEST_BLOCK` scaled by 1/B).
This is a spend/robustness finding, not a wrong-number finding: E1 is the right place to learn the
cost, but as written it may not survive to report it.

## Non-blocking

* **N1 (read of the E1 bar).** `lexlat_train_jobs.py:350,355`: `time_pass` is decided on
  `median x n_total`.  With `n_total ~52-56` batches and four 25-wide windows the plan covers the
  WHOLE sub-epoch (the job says so at :79-81), and then `mean x n_total` IS the measured wall clock
  while the median under-reports a right-skewed step-time distribution.  Read the bar on
  `seconds_per_subepoch_mean` when `n_measured == n_total`; both are in the json.
* **N2 (what the ratio compares).** The probe's second-per-step is the sum of four compute stages
  (`lexlat_train_jobs.py:298-313`); the bed's 601 s is an end-to-end sub-epoch wall clock including
  loader wait.  The 2.00x factor is therefore compute-vs-wall and is biased toward PASS; state the
  convention with the number.
* **N3 (Gate reads with no producer).** `config_sae_4a_lexlat_pack_v1.py` registers no
  `PriorGapAnalysisJob` rerun per arm decode and no wav2vec-U 2.0 selection statistic per kept
  epoch, both named in the Gate ("Disclosed label-using reads" / "Label-free reference read").
  Implementer discloses this (report item 6.1).  Both are cheap CPU reads downstream of the
  decodes and can be added without moving an arm's hash.
* **N4.** Design 5 check (a) -- step-1 loss of `lexlat_20` against `ctrl_20`'s logged step-1 loss to
  1e-4 -- is not automated; it is a two-log read after the pack starts (implementer item 6.2).
  It is a PRE-LAUNCH check in the Design; as built it can only be done post-launch.
* **N5.** `lexlat_neg_inf` is a per-batch COUNT marked with `ctx.mark_as_loss(..., as_error=True)`
  (`train_steps/sae_blankfree.py:181-191`), so the per-sub-epoch column is RETURNN's normalised
  average, not the count.  Amendment 3's "one NEG_INF utterance aborts the arm" must be read as
  "the column is nonzero", and that reading rule belongs in the phase file.
* **N6.** `monitors()` (`lexlat_train.py:455`) takes `lexlat_pruned_max` over ALL utterances
  including `z_zero` ones, while median/p95 are `keep`-masked.  Cosmetic; the max is a ceiling read.
* **N7.** Every lexical monitor (`expected_lex`, `escape_phones`, `word_phones`, `prior`) is
  accumulated only under `collect_stats` (`lexlat.py:1447`).  The bed leaves the model default
  `True` (`sae_emc.py:258`; `blankfree_train_jobs.py:175-177` does not set it), so the Gate's
  `lexlat_term_mean` / escape fraction / phones-per-word are live for these arms -- but if
  `collect_stats` is ever turned off they read 0 silently rather than raising.

## The eight dispatch items

1. **Single delta / default-off block / schedule / ADD mixing / constants / finality / guards --
   VERIFIED.**
   * Inert by default: `definitions/sae_blankfree.py:328-336` (`self.lexlat = None`, import INSIDE
     the guard) and `train_steps/sae_blankfree.py:91-100` (`getattr(model, "lexlat", None)`), so a
     banked arm re-importing this tree takes `lattice.lattice_loss` and `rate_term._fd_passes`
     unchanged.  `LexlatRuntime` is a plain object -> nothing in `state_dict`, strict load intact.
   * With a lexicon stated, `step_params` returns `(None, None)` while `lam_lex == 0`
     (`lexlat_train.py:288-314`), so before the on-set the arm runs the BANKED path, not the
     augmented DP at lam_lex = 0 (amendment 2).
   * Schedule: `lexlat_lambda` (`lexlat_train.py:98-112`) = 0 for 1-7, 1/3, 2/3, 1 at 8, 9, 10, 1
     for 11-20; on-set 5 -> 5, 6, 7; `assert_ramp` (:485-495) is called at model construction.
     `ARMS` (`config...lexlat_pack_v1.py:91-96`) = lexlat_20 (8), lexshuf_20 (8, shuffled),
     lexlat_20_e5 (5), lexlat_20_s1 (8, s1 seed) -- Design 5 exactly.
   * ADD mixing: `lexlat.py:1229` `lam = (1/tau) * lam_lex`, `prior = _prior_term(..., scale,
     prior_weight)`; `_frame_slots:834` `base = r + prior[h] + w_sym`, and every lexical addend is
     `lam * raw` (START :921, SIL :926, ESC-open :931-935, ESC-ext :940), CONT costs 0 (:906-912).
     That is `[log q + beta log P3 + lam_lex (word-LM + escape)] / tau`, ADDED, as ruling 2 says.
   * C = 1024 / C_esc = 64 flow from the config constants block (:77-88) through
     `_model_args_delta` (:157-174) to `LexlatParams` (`lexlat_train.py:303-307`); the escape
     reserve is `RESERVE_BOOST` on the top-C (`lexlat.py:1112-1122`).
   * Finality: `_frame_slots:872-876` and `_final_weight:1186-1213` -- root / open escape free,
     word end pays `lam * log p(w|sigma)` logsumexp'd over homophones, mid-word NEG_INF.
   * Design 4: rate TERM unchanged (same `lam_rate`, `rho`, `eps`, central difference) but the
     tilted passes go through the augmented DP (`train_steps/sae_blankfree.py:113-124`,
     `lexlat_train.py:348-393`) -- a disclosed reading of Design 4(a), stated in both docstrings.
     Normalisation stays per retained frame; no per-token mean anywhere.
   * Nothing beyond the lexicon differs from ctrl_20: `_arm_train_config` is prepro's call with the
     same `base["vad"]` streams (ctrl_20's `stream_kind` is "bed"), the same `attrib._priorshuf_prior`
     jobs, the same `rate.ETA_NPZ`, the same segments and the same `prepro._schedules()`, N = 20.
2. **Seeds and derangement -- VERIFIED.** `lexlat_20_s1` takes `prepro.CTRL_S1_SEED`
   (`flat_seed 1`, `random_seed 1`, `random_seed_offset 1000`) through `prepro._seed()` and
   `_set_seq_order_offset` (`config...lexlat_pack_v1.py:215-222, 227-240`), never re-typed; the
   other three write no `random_seed` and keep `FlatRecognizerInitJob(seed=0)`.
   `load_resources(..., shuffled=True)` swaps ONLY the `word_id` column (`lexlat.py:1770-1791`);
   `derange_pronunciations` is Sattolo single-cycle at seed 0 with a fixed-point assert (:208-237);
   the npz is the FINISHED `LexiconTrieBuildJob.rlMsnTBSZXsB`, pinned as a frozen path (:122-126),
   so trie `child` / `word_start` are bit-identical between arm and null.
3. **Gate producers -- mostly VERIFIED, two missing (N3).** Present: `lexlat_term_mean` (raw, not
   lam-scaled), `lexlat_pruned_mass/_p95/_max`, `lexlat_neg_inf`, `lexlat_escape_phone_frac`,
   `lexlat_expected_phones_per_word` (the POSTERIOR-expected quantity the phase file defines, band
   [2.505, 5.010] from the build job) -- all in `LexlatRuntime.monitors` (`lexlat_train.py:407-464`)
   and marked per step in the train step.  S/D/I per arm at every kept epoch: `per.json`
   (`blankfree_eval_jobs.py:99-100`) via `_register_epoch_reads`.  Rate: `decode_stats.json`.
   Derangement gap: 4 arms x 4 epochs x 2 splits.  Paired reads: 6 pairs x KEEP_EPOCHS (1, 4, 10,
   20) x 2 splits, with `ctrl_20` / `ctrl_20_s1` as frozen paths; band B stays in the prepro pack
   (`config_sae_4a_prepro_pack_v1.py:462-463`).  Frequency-stratified read at ep10/ep20 for
   lexlat_20 / lexshuf_20 / ctrl_20 with deciles fixed by `LexlatWordCountsJob` before any decode.
   MISSING: the Step 0 prior-gap rerun and the wav2vec-U 2.0 selection statistic.
   I verified all 16 frozen control decodes exist on disk and resolve to `BlankfreeGreedyPerJob`
   work dirs (ctrl_20 ep1/4/10/20 and ctrl_20_s1 ep1/4/10/20, dev-clean + dev-other).
4. **E1 -- VERIFIED except the two reading points (N1, N2) and the memory risk (B2).**
   Run shape asserted against the config itself (`lexlat_train_jobs.py:205-210`: batch_size
   {"features": 88000}, max_seqs 128); `lam_lex` asserted at full strength for the arm's own
   curriculum (:226-229) and the config asserts it again before building the job
   (`config...lexlat_pack_v1.py:399-403`); C asserted against the arm's runtime (:224-225);
   parameters pinned to the banked `ctrl_50` ep10 checkpoint (`probes.ARM = "ctrl_50"`,
   `probes.EPOCH = 10`) and RESTORED before every timed step, outside the timing (:283-284);
   laplace ordering through RETURNN's own loader with `random_seed` / `random_seed_offset` recorded
   (:243-244, :364); four windows of 25 plus the largest-padded-T batch (`_plan`, :161-172), with a
   two-walk shape assert so a non-deterministic loader errors (:277-279).  `set_epoch` +
   `_epoch_mp_shared` re-init the seq order on each new iterator (returnn/torch/engine.py:247-250,
   822-837), so the two walks are the same order.  Bar 1202 s / 80 GiB is in the class docstring
   (:38-39) and in class constants (:105-109), memory read on `max_memory_reserved` with
   `max_memory_allocated` beside it.  Extrapolation uses the TRUE batch count `n_total` of that
   sub-epoch (:350).  Fallbacks are expressible without a code change: C = 512 is the config
   constant, the word bigram is `lexlat_word_lm_order = 2` -> `truncate_to_bigram`
   (`lexlat_train.py:136-176`), whose docstring correctly discloses it is the ARPA's own 2-gram
   level and NOT the `lmplz -o 2` model the banked 0.03-nats price was measured on (exact for a
   cost measurement, would need re-pricing if taken for an ARM).
5. **Memory / no autograd -- VERIFIED for autograd, FINDING for the peak (B2).**
   `lexlat_forward_backward` is `@torch.no_grad()` (`lexlat.py:1358`) and the training path goes
   through it (`lexlat_train.py:329-346`), with the bed's own arc-posterior surrogate
   `-(1/tau)(sum post_q log q + sum seg_post G)` and a straight-through value; the fd passes run
   under `no_grad` on a detached table.  Nothing back-propagates through the lattice.
   The peak-memory reasoning at C = 1024, B = 128 is B2.
6. **Hash census -- NOT RERUN, but the mechanism is VERIFIED.** I did not load the graph in a
   console (turn budget).  The claim is mechanically sound: `Import._sis_hash` hashes the
   `code_object` PATH (`i6_experiments/common/setups/serialization.py:127-131`) and `PartialImport`
   adds only `hashed_arguments` (:201-203), so content edits to the two `prefix_lm/model` files
   cannot move a banked hash; neither edited file defines a Job class; the new `lexlat_*` keys enter
   only the four new arms' `hashed_arguments` (`attrib_train_config` asserts each key really
   changes something, `blankfree_attrib_jobs.py:226-231`).  `a05ab16` and the in-flight
   `_dest_block` touch no `__init__` kwarg and no `__sis_version__`, and `lexlat.py` is imported at
   run time, so `LexlatCensusJob.RDMTvt4ngAPB` keeps its hash.  The three round-1 probe ids and the
   prepro/budget counts (133 / 775 / 3) are the implementer's numbers, unverified here.
7. **Labels -- VERIFIED.** The training config carries no gold path; gold enters only
   `BlankfreeGreedyPerJob` (plain PER, `per = (s+d+i)/n`), `PairedPerDeltaJob`,
   `BlankfreeDerangementGapJob` and `LexlatFreqStratPerJob`, all read-side, none feeding training or
   checkpoint choice.  Kept epochs are fixed (1, 4, 10, 20) and the Gate reads sub-epoch 20, never a
   best-PER pick.  No rescored PER anywhere.
8. **Tests -- NOT CHECKED.** I did not run pytest and did not read `test_lexlat_train.py` or the
   bodies of the two new `test_lexlat.py` tests.  The deferred pruned-path manual-vs-autograd check
   exists by name (`test_lexlat.py:~660-695`,
   `test_manual_backward_matches_autograd_on_a_PRUNED_lattice`, with `pruned_max > 0` asserted and
   the degenerate `C - C_esc = 1` cell replaced by a finite-log-Z premise), and the in-flight commit
   adds `test_adaptive_destination_block_width` / `..._is_bit_identical_at_width_one`.  Whether the
   61 tests exercise the real call into the shared primitives at production dtype (Design 8(b)/(d))
   is UNVERIFIED by me this round.

## Also not checked
* E0 has not passed yet (its rerun was PENDING at review time).  Nothing here changes the standing
  order: E0 kill reads (a) and (b), then E1, then the pack.
* The adaptive `_dest_block` must be committed before launch; as of this review it is worktree-only,
  so a manager re-importing the tree would pick it up but `git show` would not show it.

## Amendments asked for before any GPU is spent
1. Fix `config/sae_4a_lexlat_e1.py` to expose `def py()` (B1) -- otherwise E1 never registers.
2. Make the expansion block width a byte budget independent of B, or record B2 as the expected E1
   failure mode and give the probe an OOM-tolerant path so a FAIL is reported rather than raised.
3. Read the E1 time bar on the mean extrapolation when the plan covers the whole sub-epoch (N1) and
   state the compute-vs-wall convention (N2).
4. Register the Step 0 prior-gap rerun and the selection statistic, or record in the phase file that
   the Gate's two disclosed reads move to a follow-up round (N3).
