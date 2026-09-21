# Code review: the soft (straight-through) scorer arm, round 1 (2026-09-21)

Verdict: **PASS_WITH_CONCERNS**. No blocking finding. Two non-blocking findings, both about
what a reader will do with a number, not about a wrong number.

Reviewed: `git show f99f9f6` on `haotian_modality_matching_jupiter` of `recipe/2025-10-speech-llm`
(full diff, all six files), the setup-dir shims, and the on-disk artifacts the config pins.
Spec: `SAE_4A_prior.md` lines 185-290 (A1-A5) and 545-600 (the reopened arm).

## 1. Scorer identity -- CONFIRMED

`PriorGapAnalysisJob.5wNIQs2lpC5P/input/` holds exactly one symlink:
`.../neural_phone_lm/NeuralPhoneLmTrainJob.xObXEwRpvmzd`. Its `output/train_log.json` gives
`selected_epoch 10`, `heldout_perplexity 5.0906`, `parameters 3312170` -- the spec's instance (a)
(4L / w256, ppl 5.091). `config_sae_4a_soft_pack_v1.py:149-150` pins that job's `output/model.pt`.
The implementer's refutation of the spec's class name `NeuralPhoneLmTrainJobV2` (line 568) stands:
that class's only instance is a 25.5 M / ppl 3.96 model. The path wins; the spec's class name is
wrong and should be corrected in `SAE_4A_prior.md`.

Its training text: 1,010,000-line window, 1,000,000 counted train lines, 10,000 held
(`train_log.json` `text`), i.e. the same window `SampleLinesJob.orN768ARKwlt` feeds
`PhoneNgramPriorJob.RtzbESkOedsT`, which is `ctrl_20`'s own `prior_npz_path`
(`.../PackedBlankfreeTrainJob.5EIGJJ1MkcO9/output/ctrl_20/returnn.config:54`). The reward's two
halves read one fitted window.

## 2. Reward conventions -- CONFIRMED

* `soft_scorer.py:844` `reward = strong - unigram_log_prob(...)`, i.e. A1's
  `log p_strong - log p_uni`, never the trigram.
* Unigram source: `config_sae_4a_soft_pack_v1.py:180-187` -> `private_code.prior_npz()` ->
  `PhoneNgramPriorJob.RtzbESkOedsT/output/prior.npz`; the ARMS get the same file through
  `attrib._priorshuf_prior()` (`config:486`, `:231`). Same file, two pins -- the probe and the
  arms calibrate and run on one baseline. (`_priorshuf_prior` is the UNBIASED uniform-window fit,
  not a shuffled prior; `config_sae_4a_attrib_v1.py:143-170`.)
* SIL dropped on both sides: `soft_scorer.py:806` `is_kept = valid & (ids != cfg.sil_id)`, applied
  before both halves. Matches `prior_gap`'s like-for-like `private`/`gold` convention
  (`prior_gap.py:83-88`).
* `>512` masked and counted, never truncated: `:808` `too_long = n_kept > scorer.max_positions`,
  `:809` excluded from `scored`, `:814` counted as `soft_masked_long`; `soft_log_prob` asserts
  `l <= max_positions` (`:641`).
* Index conventions: `soft_log_prob` (`:631-658`) is `neural_phone_lm.batch_tensors` +
  `batch_nll` in soft form -- BOS at position 0, teacher forcing, PAD id 41 (> the 40 phone
  columns, so no clobber of a phone channel), NO EOS term. `SoftScorer.logits` (`:614-628`)
  mirrors `CausalPhoneLm.forward` line for line with the embedding lookup written as the matmul
  it is; `tok` has `padding_idx=pad_id` so its PAD row is zero, exactly as the id path.
  `test_soft_scorer.py:198` pins all of this by CALLING `NeuralPhoneLmScorer.log_probs` and
  `witten_bell_utt_log_prob(..., order=1)` with real objects, agreement < 1e-5.
* Frozen / eval / no grad: `SoftScorer.__init__:602-604`. It is held in a plain dict on a plain
  object (`SoftRuntime._scorer`), never a submodule, so RETURNN's `model.train()` cannot flip it
  back to train mode and it cannot enter `state_dict` or the optimizer.

## 3. Straight-through -- CONFIRMED

* `viterbi_blankfree` (`:190-393`) is `lattice._forward_step`'s elementwise branch with
  `logsumexp -> amax`; it takes the step's own `dp` dict (`:775-779`), so same trigram history,
  same min-duration topology, same stride 3, same tau / anchor / prior weight as the loss.
* `segment_conditionals` (`:422-532`): the substitution moves exactly the token's recognizer
  gathers (emit + repeat frames -- in blank-free `w_blank` is NEG_INF, `lattice._arc_weights:515`,
  so every frame of a token carries `w_sym[k]`), its segment score, and three prior terms. The
  `frame_token` bookkeeping (`:387`) is right: the first arc cannot be a blank, so token u owns
  `[t_e(u), t_e(u+1)-1]`.
* Argmax-equals-Viterbi: asserted per batch, `max_gain > ARGMAX_TOL = 1e-6` raises
  `FloatingPointError` (`:800-804`). Safe at the arm's dtype: `ctrl_20`'s written config carries
  `lattice_float64: True`, which the soft arms inherit unchanged.
* `y_soft = onehot + (p - p.detach())` (`:538-540`, used at `:833`); `onehot` is built from long
  ids and carries no grad; the forward value is the hard string
  (`test...:238`, `rtol=0, atol=0` against the hard-string gradient).
* Gradient reaches theta and phi: `p` comes from `rec` (`_arc_weights(log_q)`) and `seg`
  (`seg_table`), i.e. `dp_log_q` and `dp_seg` of the live step. It cannot reach the scorer
  (requires_grad False). The Viterbi pass is `@torch.no_grad()` (`:190`) -- the segmentation
  carries no gradient, disclosed in the module docstring and in the train-step comment
  (`train_steps/sae_blankfree.py:153-155`).

## 4. Term, sign and plumbing -- CONFIRMED

* `soft_scorer.py:845-847`: `per_frame = reward / retained`, `term = -per_frame.sum()/n_scored`;
  `retained` is the divisor the step passes (`train_steps/sae_blankfree.py:106,158`), the same
  `lengths` `l_tau` divides by (`:107`). A5 satisfied.
* `ctx.mark_as_loss(soft_term, "soft", scale=model.soft.lam)` with `lam > 0` asserted
  (`soft_scorer.py:710`): the loss minimises the negative reward, i.e. maximises r. Sign correct.
* Default-off: `SaeBlankfreeModelV1.__init__` `soft_scorer=None` default
  (`definitions/sae_blankfree.py:271-274`), `self.soft = None` at `:365`, the module import
  inside the guard. The train step's branch is `getattr(model, "soft", None) is not None`
  (`:157`); the only lines added outside the guard are `soft_monitors = {}` and a loop over an
  empty dict, so a banked arm's loss and gradients are unchanged by construction
  (also asserted by `torch.equal` in `test...:313`).
* No banked hash can move: both edited files are serialized by `code_object_path`
  (`blankfree_train_jobs.py:221-226`), which does not hash source; `soft_scorer.py` holds no Job
  class and `soft_scorer_jobs.py` is a new file with a hand-bumped `__sis_version__ = 1`
  (the `sis-version-file-hash-moves-on-any-edit` rule is respected). Nothing outside the new
  config imports either module.
* Does `soft_*` reach the model? `_model_args_delta` (`config:217-235`) writes the keys into
  `attrib_train_config`'s `model_args_delta`, which lands in the `get_model` PartialImport's
  `hashed_arguments` (`blankfree_attrib_jobs.py:223-231`) and so into the written RETURNN config.
  I verified the mechanism by reading; I did NOT re-run the implementer's write-and-reload check
  (see NOT CHECKED).

## 5. Null arm `softshuf_20` -- correct, under-documented (FINDING B)

`derangement()` (`:543-569`) draws by rejection from a CPU-seeded `randperm`, fixes SIL, and
`assert_derangement` refuses any fixed point among the 39. `DERANGEMENT_SEED = 0` matches the
bed's null convention. The permutation is printed at model construction
(`definitions/sae_blankfree.py:382`) and banked in `describe()` -- "fixed recorded" satisfied.

`soft_scorer.py:841` applies the permutation to BOTH halves of the reward
(`y, p_s = y.index_select(-1, perm), ...`, and `y` then feeds `unigram_log_prob` at `:844`). This
is the right choice -- it makes `r_null(y) = r(sigma^-1 y)`, the same function on relabelled
strings, which is exactly the spec's "the reward keeps its statistics but not the phone
identities"; permuting only `p_strong` would leave a mixed function with a different mean and a
frequency exploit. But the spec's text says only "the scorer sees ...", and nothing in the code
says the baseline is permuted too: line 841 has no comment, the module docstring says only "the
optional fixed-point-free derangement", and `derangement`'s docstring speaks of "the scored
vector". The consequence: `soft_reward_mean` of `softshuf_20` is a DIFFERENT statistic from the
real arms' and must not be read in the same column, and a planner assuming the baseline was not
permuted would misread the null. Fix: one comment at `soft_scorer.py:841` and one clause in
`SAE_4A_prior.md` line 583.

## 6. Monitors -- CONFIRMED

`soft_reward_mean` (`:853`, `reward/tok`, tok = kept non-SIL tokens, forward value = hard
string), `soft_artefact_gap` (`:854`, `scorer(soft) - scorer(hard)` per token, with
`SOFT_ARTEFACT_CEILING = 0.3` declared at `:88` and reported, never enforced),
`soft_masked_long` (`:814`), `soft_argmax_mismatch` (`:815`), plus `soft_lam`, `soft_scored`,
`soft_term_mean`, `soft_tokens`. The expected rate and the rate FD check are the existing
`blankfree_expected_phone_rate_hz` / `blankfree_rate_fd_check`. All marked `as_error=True` under
their own `soft_*` keys and only when the term is on
(`train_steps/sae_blankfree.py:219-221`), so no banked arm's reported column moves. RETURNN
prints `as_error` losses per sub-epoch, as the lexlat monitors already are.

FINDING A (non-blocking, reading rule): `train_steps/sae_blankfree.py:216-217` and
`SAE_4A_prior.md:586` attach the target band "rise from about -0.67 toward +0.95" to
`soft_reward_mean`. Those two numbers are the Step 0b table's `private` / `gold` rows -- the
GREEDY collapsed decode on DEV-OTHER. `soft_reward_mean` is computed on the TRAIN split, on the
max-plus (Viterbi) string of the tempered lattice at that sub-epoch's tau, over the utterances
that were scored. It is the right quantity to monitor and the right direction to watch, but its
LEVEL is not like-for-like with -0.67 / +0.95, and at high tau early in the schedule the Viterbi
string is not the greedy decode. Read the band as a direction, not a threshold; the like-for-like
number is the disclosed prior-gap rerun at ep10 / ep20. Recommend amending the comment and the
plan line rather than the code.

## 7. Pack config -- CONFIRMED

* Four arms exactly as specified (`config:103-108`): `soft_20` LAM_01 / seed 0, `soft_20_s1`
  LAM_01 / the `ctrl_20_s1` seed set, `soft_20_r03` LAM_03 / seed 0, `softshuf_20` LAM_01 /
  seed 0 / derangement 0.
* N = 20, kept epochs 1/4/10/20, `TIME_RQMT`, `SEC_PER_SUBEPOCH` are taken from
  `config_sae_4a_prepro_pack_v1` by reference (`:114-119`), never re-typed. Schedules and the
  second seed come from `prepro._schedules()` / `prepro._seed()` (`:250`, `:277`). Streams are
  `attrib._control()["vad"]` for all four arms -- the bed stream `ctrl_20` and `ctrl_20_s1` both
  run (`prepro ARMS:120-124`). Flat init: the bed's seed-0 job, or the seed-1 job for the
  replicate, identical to prepro's construction. `_arm_train_config` (`:238-285`) is prepro's
  call argument for argument; the only delta is `model_args_delta`.
* Paired reads: the spec's five pairings (`:356-362`), at all four kept epochs and both dev
  splits, with `per_a` = baseline. Controls consumed as frozen decodes via `_ctrl_hyps`
  (`:190-214`), which asserts the realpath is a `BlankfreeGreedyPerJob` output and pins a sha of
  it -- a repointed prepro output moves the hash rather than silently changing the input. The
  seed band is read from the prepro pack, not re-registered.
* Derangement gap registered per arm / epoch / split (`:337-345`); prior-gap rerun at
  `PRIOR_GAP_EPOCHS = (10, 20)` only, four arms, dev-other, every other argument read from the
  banked Step 0 config (`:387-428`).
* `LAM_01 = LAM_03 = None` and `build()` calls `_lam(arm)` for all four arms before any job is
  constructed (`:466-467`), so the pack cannot be built before the probe is read. Correct
  realisation of "Recorded before the pack launches".
* Efficiency assert: `20 x 601 s x 2.00 = 6.68 h < TIME_RQMT` (`:479-483`).

## 8. `SoftLamProbeJob` -- CONFIRMED (code only)

`soft_scorer_jobs.py:338-367`: one forward of the ARM'S OWN `train_step` (read out of the pinned
`returnn.config`) with the runtime attached; `l_tau` and `soft` are read off `ctx.losses` AS
MARKED, both per retained frame (A5's one convention), and each differentiated over
`recognizer.*` + `reverse.*` with `torch.autograd.grad`. This is the shared function, not a
re-implementation. `lam_tau` is asserted 1.0 (`:301-304`), `losses[...].scale` is asserted
(`:358-359`), the arm is asserted to carry no soft scorer, no lexicon and no frame permutation
(`:293-300`), and the run shape is asserted against the config (`:279-283`,
`batch_size {"features": 88000}` / `max_seqs 128` -- both confirmed in `ctrl_20`'s written
config). Operating point: `ctrl_20` ep4, seed 0, 3 batches (`config:139-141`), the spec's.
Median ratio and `lam_soft = target / ratio` for 0.1 / 0.3 at `:469-472`, rendered per batch so
the spread behind the median is visible. `engine.set_epoch` after `_create_data_loader` is the
same order `lexlat_train_jobs.py:325-326` uses, i.e. the bed's convention, not a new risk.

## 9. Labels -- CONFIRMED

Nothing label-derived enters any training or selection path. The soft term reads `log_q`, the
segment table, the prior table, the lengths, the frozen LM and the window unigram. The scorer and
the unigram are fitted on LM text, the same text the in-lattice trigram already uses. The
label-using reads (greedy PER, paired deltas, derangement gap, prior-gap rerun) are all read-side
jobs on written decodes; `lam_soft` is fixed by a gradient-norm ratio, not by a PER.

## NOT CHECKED

* The 22 tests were not re-run in this review; I read them and they exercise what the report
  claims, but the PASS is the implementer's.
* The census (prepro 133, budget 775, lexlat probes 3 and the named job ids) was NOT
  independently rebuilt. The mechanism that makes it hold -- `code_object_path` hashing, the new
  `__sis_version__` file, no import of the new modules from anywhere else -- was verified by
  reading.
* The write-and-reload check that `model.soft` is a `SoftRuntime` in the written RETURNN config
  was not re-run; the serialization path was read instead.
* `SoftLamProbeJob.run()` has never executed anywhere (it needs a GPU). Its RETURNN calls are
  copied from a probe that has run.
* The pack's job hash does not exist yet and cannot until `LAM_01` / `LAM_03` are filled in.

## Recommendation

Launch `config/sae_4a_soft_probe.py`. Before the pack is built, fill `LAM_01` / `LAM_03` from
`summary.txt`, and record in `SAE_4A_prior.md` (a) the corrected scorer class name, (b) that the
null permutes the unigram baseline as well as the scorer, (c) that the `soft_reward_mean` band is
a direction and the like-for-like read is the prior-gap rerun.
