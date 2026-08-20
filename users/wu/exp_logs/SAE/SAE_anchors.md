# SAE — the reward's lam_1 (LM prior) and lam_2 (KL anchor)

## Approach

**1. Wire the two side-inputs the live train step never passed.** `compose_reward` has implemented
both terms since the reward was written, but each `_require`s a rollout-dependent side input —
`base_logprob_sum`/`n_text_tokens` for lam_1, `kl` from `ref_logprobs` for lam_2 — so setting either
raised by construction and every run to date had been recon-only. Neither can be precomputed (both are
functions of the rollout), so they enter the trainer as callables on `SampledText`, the same shape as
the existing `reward_fn` hook. lam_1's `p_base` is the AV's own decoder with LoRA-A switched off and
the audio prefix dropped, so "adapters off" *is* the frozen donor LM, teacher-forced identically to
`SpeechLmAvPolicy.logprobs`; lam_2's reference is a second fully frozen `SpeechLmV2` loaded from the
same checkpoint the live policy starts at, with every param frozen in *both* models shared, so the
snapshot costs one extra adapter + LoRA-A rather than a second 4 GB model. Non-trivial choices: the
snapshot is held in `eval()` mode whatever RETURNN does (LoRA dropout 0.1 — a reference that redraws
its own mask every step is not a reference); `ref_av` is built only when `lam_kl != 0` and each
side-input callable is invoked only when its lambda is nonzero, so at lambda=0 the module tree, the
state_dict and the resume path are untouched for the four live loops that re-import this source on
every resubmit; and each term gets a `reward/<term>_std_within_group` monitor, because advantages are
group-normalized and `lam * std(term) / std(recon)` is the mixing ratio a lambda actually sets.

**2. Smoke run on the validated 10 h bed, 1 GPU, 1 epoch, audio <= 8 s, two arms.** `recon` (`{}`)
against `anchors` (lam_1 = lam_2 = 0.02, placeholders chosen to be small and nonzero, not
recommendations). The control is not decoration — it makes "the anchors changed nothing they
shouldn't" checkable on the same bed.

| check | result |
|---|---|
| `reward_kl` ~ 0 at step 0 | 0.010, then 7.2e-4, -1.4e-4, 0.002, ... — sign-varying, 1e-3..1e-2 nats/token, exactly the LoRA-dropout residual predicted; a stub would give exactly 0.0 |
| `reward_lm_prior` in a plausible range | -6.3 .. -9.1 nats/token, well clear of -11.93 = log 1/151646; tracks length cleanly (text_len 8.5 -> -6.27, 6.58 -> -8.20) |
| mixing ratios, first 10 steps | std(recon) 0.071; lm_prior 0.462 -> **13 %** at 0.02; kl 0.017 -> **0.5 %** at 0.02 |
| full epoch (203 steps), bucket means | kl share 0.5 % -> 4 % as the policy moves; lam_1 share rises to ~22 % mid-epoch as std(recon) shrinks 0.071 -> 0.028 |
| control tracking + cost | `reward_recon` identical to 3 decimals at steps 0-1, separating at 1e-3 from step 2; GPU 10.7 vs 10.0 GB (the tying worked); +11.0 % s/step on cumulative `elapsed` |

**3. The full-length 4-epoch anchored pair on the validated 10 h bed** (user call: lam_1 = 0.01,
lam_2 = 0.02 before any §1e loop leans on the anchors). Frozen AR, per-epoch dev recogs. The control
is **not re-run**: `baseline()` gained `reward_kwargs`/`freeze_ar`/`eval_subset` whose defaults
reproduce the finished recon-only run byte-for-byte, proven against disk, so the anchored arm is the
same bed with exactly one variable changed.

| | anchored lam_1 .01 / lam_2 .02 | control (recon-only) |
|---|---|---|
| theta_0 | 16.9 / 20.7 | 16.9 / 20.7 |
| ep1 | 16.2 / 21.3 | 13.07 / 15.89 |
| ep2 | **12.91** / 16.49 | 13.87 / 16.51 |
| ep3 | 13.91 / 18.33 | **12.99** / 16.20 |
| ep4 | 13.99 / 17.90 | 14.47 / 17.09 |
| dev-clean mean (excl ep1) | **13.60** | 13.78 |
| dev-other mean (excl ep1) | 17.57 | **16.60** |
| ep3 -> ep4 delta | **+0.08 / -0.43** | +1.48 / +0.89 |

Epoch-1 in-run monitors on the real bed (steps >= 100, n = 797 rank-lines): recon -5.7635 with
within-group sd 0.0131; lm_prior -5.008 with sd 0.2236 = **17.1 %** at lam_1 = 0.01; kl 0.01417 with
sd 0.02181 = **3.3 %** at lam_2 = 0.02. Over the epoch in 200-step buckets, lam_1's share rises 11 %
-> 17 % and then sits at 16-17 % while lam_2's grows 1.2 % -> 13 % as KL rises 0.006 -> 0.09
nats/token and plateaus.

**4. Resume test (theta_0 across a restart), 2 epochs on the smoke bed.** Two failure modes, only one
of them loud: a hard key error if the checkpoint's 115 untied `ref_av.*` records cannot be mapped, and
a **silent** one where the restart re-snapshots theta_0 from the *current* policy, making lam_2 a
no-op from that point while every monitor keeps printing a plausible number. `reward_kl` discriminates
them, which is why this is worth a GPU rather than another audit.

## Conclusion

1. (2) All four smoke checks pass; the implementation is verified end to end, with the frozen snapshot
   demonstrably forwarding real weights rather than a stub.
2. (2) lam_1 carries a built-in **length coupling** — rollouts scored from BOS with no prompt context
   amortize the high-entropy first token over few tokens, so shorter rollouts get a worse per-token
   prior and `lam_lm * prior` rewards longer text. Within a group lengths differ, so it reaches the
   advantages. Plausibly a feature (it pushes against degenerate short codes) but it is a coupling, not
   a pure fluency term, and must be named as such wherever lam_1 is set for real. Measured at +0.26
   tokens (+1.4 %) mid-epoch against the control on identical batches.
3. (2 vs 3) **The smoke's +11 % step-time cost does not transfer**: on the full bed it is +0.6 %
   (2.320 vs 2.307 s/step over steps 50-200), because the encoder and policy forwards dominate. The
   +11 % was real but measured where the two extra forwards are a large slice of a small step. Only
   startup regresses, by +45 s, which is the `ref_av` load.
4. (3) **lam calibration numbers do not transfer across beds.** lam_1 = 0.01 was picked for a ~10 %
   mixing ratio and lands at 17 %, not because the prior changed (its spread only halved) but because
   the *recon* spread collapsed 5x — within a group of 12 the AR returns nearly the same recon to
   every member, so any side term of fixed size buys a larger share of the advantage than an easier
   bed suggests. Read the ratios on the bed that will be run.
5. (3) **dev-other is consistently worse with the anchors** — every epoch, mean 17.57 vs 16.60, never
   crossing, so unlike dev-clean this is not oscillation. dev-clean is a wash (best 12.91 vs 12.99).
6. (3) lam_2's late-regression claim is supported but not established: the control falls off at ep4
   (+1.48/+0.89) while the anchored arm does not (+0.08/-0.43) and ends better on dev-clean; the 1.4
   abs delta-of-deltas is the largest effect in the table, on n=1 transition per arm.
7. (3) **Verdict.** For §1e the trade is acceptable — lam_2 is mandatory there because nothing pins
   the text side, the anti-collusion guarantee is its purpose, and it buys late-training stability.
   For §2S, where the seed already pins the text, the anchors are not worth the dev-other cost: keep
   those loops recon-only.
8. (3) **The methodological result, and the more valuable one.** Four per-epoch readings were taken
   during this run and three were wrong or premature — ep1 read as a 5.4-abs catastrophe, ep2 as an
   overtake, ep3 as unresolvable anti-phase — and none survived the next data point. The epoch-level
   noise floor on this bed is ~1 abs, the same size as the effect being looked for. Consequence:
   **lam_1-only / lam_2-only arms at this budget would be unreadable.** Attribution needs multiple
   seeds per arm, a lambda deliberately large enough to clear 1 abs, or a bigger bed; do not spend GPU
   on a 1-seed 4-epoch single-anchor pair. Related discipline: per-step reward stats on 2 groups of 12
   are far too noisy to calibrate against — a single step-800 line read as a runaway 50 % lam_1 share
   where the bucket mean was 16 %.
9. (4) **Resume PASS, exactly.** Pre- and post-restart epoch-2 statistics are bit-identical
   (kl 0.0833, recon -5.5061, lm_prior -6.462 over n=46), so theta_0 was restored rather than
   re-snapshotted; the bit-identical reproduction additionally rules out optimizer-state and data-order
   drift across the restart. Operational note: the retry **overwrites `log.run.1`**, so the
   pre-restart baseline must be captured before the `scancel`.

## Catalog

`T/` = `work/i6_core/returnn/training/`.

| artifact | path |
|---|---|
| implementation | `sae/grpo/anchors.py` (new), `sae/grpo/trainer.py`, `definitions/sae_grpo.py`, `train_steps/sae_grpo.py` |
| tests | `sae/grpo/test_anchors.py` (10 new) + `test_trainer` (+4); 76 CPU tests pass |
| smoke arms | `T/ReturnnTrainingJob.QyeYIctvpywK` (anchors), `.Vj2m15i9zqxj` (recon control) |
| **full-length anchored arm** | `T/ReturnnTrainingJob.iNZtd4CRrSLR` |
| control (finished, byte-identical rebuild) | `T/ReturnnTrainingJob.qmkzvAX3gOVW` |
| resume test | `T/ReturnnTrainingJob.1hp40l6O9rga` |

Two tests carry most of the weight:
`test_base_lm_on_a_real_peft_wrapped_qwen3_is_the_pre_lora_donor` (a real `Qwen3DecoderV1` with a real
PEFT LoRA whose `lora_B` is randomized first, otherwise "adapters off" passes vacuously) and
`test_kl_vanishes_when_the_reference_is_the_sampling_policy`. `adapters_disabled` now **raises** if a
decoder carries LoRA params but has no `disable_adapter`, because otherwise `p_base` would quietly be
the *adapted* LM and lam_1 would degenerate into "reward the policy for being confident in its own
output" — an entropy-collapse driver no monitor would distinguish from a working prior.

Known bias, quantified and accepted rather than fixed: the k1 estimator differences two *different*
forward paths (`sampled.logprobs` from the KV-cached incremental sampler against `ref_logprobs` from
teacher-forced `decode_seq`), which do not agree exactly under bf16 autocast even at theta = theta_0,
so `reward_kl` carries a ~0.01 nats/token path-mismatch offset against real drift reaching 0.03-0.10
by step 150 — ~10x signal-to-offset, improving as drift grows. The estimator-consistent fix requires
reordering `grpo_step`, which four live loops import; not worth that risk for a 10 % bias on a term
that is itself 0.5-4 % of the reward spread.

## Verifier feedback

**2026-08-01 (planner).** Code read end to end (`anchors.py`, `reward.py`, `trainer.py::grpo_step`,
`definitions/sae_grpo.py`, `train_steps/sae_grpo.py`) — matches this log; all four gating decisions
are in the code as described. Tests independently rerun 55/55; smoke log steps 0-5 match verbatim.
Hash neutrality is observed rather than argued: at check time both 960 h arms were training *through*
the edited source tree.

Preflight audit of the anchored checkpoint, each item verified against the real file:
- The frozen-base tying survives `state_dict()` + `torch.save` — `epoch.001.pt` has 1862 keys over 947
  unique storages, `ref_av` contributing 115 untied records = **759 MB** (5.927 GB total vs the recon
  arm's 5.168 GB). This **corrects the log's earlier "~150 MB" snapshot estimate** (LoRA-A at r=128 is
  bigger than assumed) and is consistent with the observed +0.7 GB GPU.
- theta_0 cannot drift: AdamW skips params with `grad is None` and `epoch.001.opt.pt` is +11.8 KB on
  the anchored arm — names only, no moment tensors.
- `ExtractAvSubmodelJob` returns 719 keys, set-identical to the recon checkpoint's; no `ref_av.*` key
  matches, so the leak assertion does not fire.
- There is no DDP (`reduce_type="param"` makes `maybe_make_distributed_module` return None); the
  param-averaging path all-reduces frozen params including `ref_av`, but they are bit-identical across
  ranks, so it is a no-op to <= 1 ulp.
- `clear_autocast_cache()` still sits after *all* no-grad work including both new closures, so the
  RUN-1/2 zero-gradient bug cannot recur here.
