# Review -- the score-function arm `sf_20` (speech-llm `ff4f005`), 2026-09-21

Read-only review of `ff4f005` against its parent `0474d7f`, the spec `SAE_4A_prior.md`
("Training arm, reopened by user ruling (2026-09-21)" + the user override at its end, amendments
A1/A3/A4/A5) and the round-1 reference `reports/review_soft_arm_r1_2026-09-21.md`.
Nothing was launched, nothing was edited, no test was run.

## Verdict

**DONE_WITH_CONCERNS.** The term is what the override specifies: the banked FFBS primitive at the
step's own posterior, A4's Fisher path score on un-expanded tables, A1's reward on the sampled
strings through the unmodified soft-arm primitives, centred per utterance, A5 per retained frame,
sign correct, default-off elsewhere. One finding, low severity, about a monitor column that is not
the same statistic as the soft arm's column of the near-same name. No blocking finding; the probe
`SfLamProbeJob.OHrcN9pEuXni` can be funded.

## Findings

1. `src/speech_llm/sae/emc/sf_scorer.py:479` (`_decode_monitor`) + `:219` (`collapse_greedy`) --
   `sf_reward_mean` is A1's reward on the GREEDY ARGMAX collapse, while the soft arms'
   `soft_reward_mean` (`soft_scorer.py:853`) is the reward of the MAX-PLUS (Viterbi) lattice
   string. The two columns differ by the string they score, not only by the arm. Concrete failure:
   a `sf_20` vs `soft_20` read that puts the two `*_reward_mean` columns side by side (the pairing
   `sf_20 - soft_20` invites exactly that) reports a difference that is partly argmax-vs-Viterbi.
   Disclosed in the implementer report (deviation 4) and in both docstrings, and the Gate is read
   on PER, not on these columns, so this is a reporting hazard, not a wrong gate number.

## Notes (not findings)

* The group baseline includes the draw itself (`centred_advantages`, `sf_scorer.py:245`), so the
  estimator is `(1 - 1/G)` times the true gradient. Harmless: `SfLamProbeJob` calibrates `lam_sf`
  on this same estimator, so the factor is absorbed by the weight.
* The term's denominator is the utterances with >= 2 scoreable draws (`sf_scorer.py:425`), while
  `l_tau`'s is all live utterances. Equal unless a group collapses; disclosed in the code.

## Checks 1-6

1. **Sampler -- done.** `sf_scorer.py:409` calls the banked `blankfree_sampler.sample_blankfree_paths`
   (`@torch.no_grad()`, `blankfree_sampler.py:133`, detaches its inputs) with the step's own `dp`
   values -- `temperature`, `anchor_weight`, `prior_weight`, `log_q_init`, `history`, `reduction`,
   `checkpoint` -- so the draws are from the objective's own posterior at the schedule's tau; no
   re-implementation, min-duration (`cfg.d_min = 2`) and run collapse are the primitive's.
   `num_samples = 8` from the config (`SF_NUM_SAMPLES`), seed
   `stream_seed(1234, epoch, step)` (`sf_scorer.py:120`), a pure function of the sub-epoch and the
   step, so a resumed sub-epoch redraws its own stream. `lattice_checkpoint = 32` is in the bed's
   model args, so `int(dp["checkpoint"])` cannot be `None`.
2. **Fisher path score -- done.** `sf_path_score` (`sf_scorer.py:129-173`) is
   `blankfree_sampler.path_score` term for term (scaled per-frame log-q gathers + emitted segment
   scores, prior term absent, `log Z` cancelling under `sum_g A_g = 0`). Both gathers are on the
   UN-EXPANDED tables: `torch.gather(acoustic, 2, sym)` at `:160` with `sym = [B, T, G]` (backward
   buffer `zeros_like(log_q)`, `[B, T, 40]`) and `torch.gather(flat, 1, idx)` at `:170` on
   `seg_table.reshape(b, -1)` (backward buffer one `seg_table`). No `expand()`/broadcast source
   anywhere in the file; at B ~ 114, U_max ~ 171 the backward allocates one copy of each table,
   not G. Reward is detached (`hard_reward` is `@torch.no_grad`, `adv.detach()` at `:452`);
   advantages centred per UTTERANCE over the scoreable draws (`.sum(-1)`, `sf_scorer.py:250`), not
   over the batch; masked draws leave the group and carry zero advantage. Per-retained-frame
   normalisation uses the same `retained` divisor `l_tau` uses (`train_steps/sae_blankfree.py:105`,
   `retained = lengths`). Sign: `term = -mean_b[mean_g(A*pathscore)/retained]` marked with
   `scale = lam_sf > 0`, so minimising raises `log w` of positively-advantaged paths.
3. **Reward -- done.** `soft_scorer.soft_log_prob` / `unigram_log_prob` are imported, not copied
   (`sf_scorer.py:76`); `soft_scorer.py` is not edited in this commit. BOS context, teacher
   forcing, no EOS, padded positions ignored. SIL dropped before the cap is measured
   (`kept_strings`, `:190`), `> scorer.max_positions` masked out of the group and counted in
   `sf_masked_long` (`:416`), never truncated. `r` is taken on the SAMPLED strings
   (`drawn.ids`, `:437-447`), not on any max-plus string. `sf_reward_mean_sample` is the draws'
   mean per token over the scoreable draws; `sf_reward_mean` is the greedy decode's (finding 1).
   `sf_reward_std_within` is the population std in nats per utterance, median over scored
   utterances -- A3's unit.
4. **Plumbing -- done, except the census re-run.** The guarded block marks the loss and the
   monitors (`train_steps/sae_blankfree.py:171-179`, `:247-250`); mutual exclusion is asserted at
   model build (`definitions/sae_blankfree.py:408`, `assert self.soft is None`) and the config
   returns early for `kind == "sf"` so the arm carries no `soft_*` key and the soft arms no `sf_*`
   key (`config_sae_4a_soft_pack_v1.py:282-300`). Hash neutrality: both edited model files reach
   the written config through `Import(code_object_path=..., unhashed_package_root=...)`
   (`blankfree_train_jobs.py:221-226`), i.e. hashed by the import path, not by file contents, so
   the edits structurally cannot move a banked hash. I did NOT re-run the six graph loads behind
   the census table (prepro 133 / budget 775 / lexlat 3 / falsifier 3 /
   `SoftLamProbeJob.wAJQ26T7iZzX`); that claim is the implementer's, and the mechanism above is the
   only reason I can give for it. `ARMS` puts `sf_20` in the slot `soft_20_r03` held, `seed: None`
   (so the ctrl_20 seed set and the bed's own `FlatRecognizerInitJob(seed=0)`), same
   `NUM_SUBEPOCHS` / `KEEP_EPOCHS` / schedules as the soft arms. `LAM_01 = 0.326639` and
   `LAM_03 = 0.979918` are `0.1 / 0.306148` and `0.3 / 0.306148` off the finished
   `SoftLamProbeJob.wAJQ26T7iZzX` (`SAE_4A_prior.md:664-666`); `LAM_SF_01` is `None` and `_lam`
   refuses it by name, so `py()` cannot build the pack yet. `softshuf_20` keeps
   `soft_derangement_seed = 0` and `SoftRuntime.step` permutes the same `y` into BOTH
   `soft_log_prob` and `unigram_log_prob` (`soft_scorer.py:840-845`). Pairings: the loop is
   `for cand, baseline in paired_arm_pairs()` with `per_a = baseline`, so `(SF_ARM, CTRL_ARM)` and
   `(SF_ARM, TREATMENT)` are `sf_20 - ctrl_20` and `sf_20 - soft_20` in the right orientation
   (negative = sf_20 better).
5. **Probe -- done.** `SfLamProbeJob` is a line-for-line mirror of `SoftLamProbeJob`: same frozen
   `ctrl_20` ep4 seed-0 checkpoint and written config, same 88,000 padded frames / `max_seqs` 128
   assert, `n_batches = 3`, median over the per-batch ratio, `lam_sf = target / ratio` for
   (0.1, 0.3), global L2 norm over `recognizer` + `reverse` together, `lam_tau` asserted 1.0, and
   the term run at the placeholder `PROBE_LAM = 1.0` with no total loss formed. It adds
   `torch.cuda.reset_peak_memory_stats()` per batch and banks `peak_allocated_gib` and
   `step_seconds` beside the ratio, and it runs TWO real backwards per batch at the run shape --
   which is the only place the added sampler pass's footprint has ever been measured.
6. **Disclosed read -- done.** `_register_sf_reads` instantiates falsifier (ii)'s own
   `blankfree_probe_jobs.SampledRewardProbeJob` at ep10 / ep20 (both asserted in `KEEP_EPOCHS`)
   with the pack's own scorer and the arm's own prior, the job's default 300-utterance seeded
   dev-other subset, `num_samples = 8`, `sample_seed = 1234`; `tau` is the arm's schedule value and
   the job asserts it against the model's own anneal (`blankfree_probe_jobs.py:452`). The statistic
   is `gold_gt_max_sample_fraction` on the `r_nn` column (neural minus unigram = A1),
   `blankfree_probe_jobs.py:172`, rendered in `probe.md` (`:264`). Forward-only, writes three files,
   nothing it writes is read by any training or selection path; the >= 0.95 UNINFORMATIVE rule
   stays pre-registered in the job's docstring and is not re-typed in the config.

## Not checked

* The census numbers were not re-derived (no graph load run); only the hashing mechanism was.
* `test_sf_scorer.py` was not executed; its 11 cases were read. Test 7 ("off = the banked path")
  exercises a local re-spelling of the train step's lines, not `train_step` itself -- the real
  call is exercised only by `SfLamProbeJob`, which runs the arm's own step function.
* Nothing has run on GPU: the memory and step-time behaviour of the extra sampler pass at the run
  shape is unmeasured, by design, until the probe reads out.
