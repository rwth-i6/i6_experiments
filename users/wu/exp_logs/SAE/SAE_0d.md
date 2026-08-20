# SAE 0d — LM-prior domain adaptation to LibriSpeech text

## Approach

1. **Where the prior actually lives.** `grpo/anchors.py` computes the `lam_lm` term with the AV's own
   decoder, LoRA disabled and the audio prefix dropped — `p_base` is the AV's frozen Qwen donor, not a
   separate model. Read of `ReturnnTrainingJob.OLzy9Q2oC3mU/output/models/epoch.050.pt` (theta_0, 4.6 GB):
   423 decoder keys, 112 LoRA and **311 non-LoRA carrying 2.03 B donor weights**, and
   `definitions/sae_grpo.py:_load_submodel` loads them with `strict=False`. So the donor a config names
   through `av_args()` is overwritten by the checkpoint on the next line.

   | theta_0 checkpoint | params | dtype |
   |---|---|---|
   | decoder, non-LoRA (the donor) | 2.032 B | bf16 |
   | decoder, LoRA | 112 keys | — |
   | whole checkpoint | 2.381 B | — |

2. **The finetune.** Full finetune (no LoRA) of Qwen3-1.7B-Base on `librispeech-lm-norm`
   (40 418 261 lines / 803 288 729 words / 4.25 GB), exactly one pass: `partition_epoch` 10 and
   `num_epochs` 10, cosine over the same 10, so the run is one pass by construction. Normalization is
   `.lower()` on the corpus file, which is what `LowerCaseTextAndApplyVocab` does to the AV's own targets
   before `vocab.get_seq`; the corpus already carries LibriSpeech's uppercase, punctuation-free
   convention, so lowercasing is the whole of the match. fp32 parameters (`definitions/sae_text_lm.py`):
   the donor ships bf16, whose relative resolution is ~2^-8, and at max_lr 1e-5 an all-bf16 run would
   report a falling loss while most weights never moved. Gradient checkpointing on; batch 16 384 padded
   frames x 4 ranks, which is what bounds the [B, T, 151936] logits `train_lm_step` materializes.
   The 4 ranks shard the corpus (`_num_shards`/`_shard_index` from torchrun's `WORLD_SIZE`/`RANK`):
   `Dataset._get_partition_seq_order` bins by `partition_epoch * num_shards`, so 10 x 4 = 40 disjoint
   partitions are covered exactly once across the 10 sub-epochs — still one pass, at a quarter of the
   wall clock. RETURNN auto-shards by rank only inside `DistributeFilesDataset`, so a bare `LmDataset`
   does not get this for free. The donor is selected on dev perplexity (user's rule, 2026-08-08);
   dev is LibriSpeech dev-clean+dev-other transcript text, so this uses an annotation for selection —
   near-vacuously, since the curve is monotone and the rule can only ever return the last sub-epoch.

   | sub-epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | dev log_ppl | 3.7128 | 3.6757 | 3.6590 | 3.6423 | 3.6331 | 3.6254 | 3.6193 | 3.6144 | 3.6108 | **3.6098** |
   | dev ppl | 40.97 | 39.48 | 38.82 | 38.18 | 37.83 | 37.54 | 37.31 | 37.13 | 36.99 | **36.96** |

3. **Blocking pre-check (PLAN §0d(i)).** 8-gram and whole-sentence overlap between the raw LM corpus and
   the dev-clean/dev-other/test-clean/test-other references, with **train-clean-100 as a positive
   control** — its books are in the corpus by construction, so without it a near-zero dev/test rate is
   indistinguishable from a scan that matches nothing. The job refuses to certify disjointness if the
   control clears less than 20 % of its own 8-grams.

   | probe | utts | 8-grams found | whole sentences found |
   |---|---|---|---|
   | dev-clean | 2703 | **1.91 %** | 1.96 % |
   | dev-other | 2864 | **3.36 %** | 3.21 % |
   | test-clean | 2620 | **1.84 %** | 1.76 % |
   | test-other | 2939 | **1.59 %** | 2.01 % |
   | train-clean-100 (control) | 28 539 | **42.98 %** | 3.67 % |

4. **Re-SFT and the two loop arms.** theta_0 rebuilt by `config_sae_2s_av_sft_w2v2_v1.theta0()` on the
   adapted donor, identical in every other argument (`THETA0_KWARGS`), so the two 10 h SFTs differ in
   exactly one thing. Then the `shaped` arm of `config_sae_3a_psi_loop_100h_v1` and
   `config_sae_3a_psi_loop_960h_v1` on that init, alias suffix `_lbslm`. `recon` is not built: it cannot
   move under this phase, since the reconstruction term never sees the donor. The 960 h arm is 1 pass
   (10 sub-epochs, ~2 days) and the stock-donor 960 h arm is not built at all (user's call 2026-08-08):
   at 3 passes the pair cost ~12 days, and the stock arm had already turned by sub-ep3-4 under the
   scorer's own view (`SAE_3A.md` concl. 39). Consequence: the donor axis is read on the 100 h bed,
   where both arms exist; the 960 h bed axis is read `_lbslm` against `_lbslm`.

   | dev-clean / dev-other | stock donor | LBS-adapted donor |
   |---|---|---|
   | 10 h AV SFT, ep50, no loop (theta_0 / theta_0') | 16.91 / 20.64 | **11.43 / 15.54** |
   | — the same, test-clean / test-other | 15.28 / 20.78 | **11.99 / 14.34** |
   | 100 h `shaped`, sub-ep 1 | 10.89 / 16.49 | **7.14 / 11.35** |
   | 100 h `shaped`, sub-ep 2 | 7.42 / 12.51 | **5.95 / 10.16** |
   | 100 h `shaped`, sub-ep 3 | 6.79 / 10.61 | **5.47 / 10.55** |
   | 100 h `shaped`, sub-ep 4 | 6.47 / 10.77 | **5.23 / 9.06** |
   | 100 h `shaped`, sub-ep 5 | 6.13 / 10.79 | **5.53 / 9.76** |
   | 100 h `shaped`, sub-ep 6 | 6.18 / 10.37 | **5.36 / 9.87** |
   | 100 h `shaped`, sub-ep 7 | 6.06 / 10.37 | **5.27 / 9.96** |
   | 100 h `shaped`, sub-ep 8 (last) | 6.06 / 10.31 | **5.22 / 9.67** |
   | 960 h `shaped`, sub-ep 1 | 6.87 / 10.98 | **6.28 / 10.25** |
   | 960 h `shaped`, sub-ep 2 | 18.22 / 22.21 | **5.34 / 9.50** |
   | 960 h `shaped`, sub-ep 3 | 18.82 / 23.51 | 6.56 / 11.15 |
   | 960 h `shaped`, sub-ep 4 | 19.67 / 24.92 | 6.89 / 11.31 |
   | 960 h `shaped`, sub-ep 5 | — | 6.32 / 11.03 |
   | — the same, test-clean / test-other | — | 6.76 / 11.55 |
   | 960 h `shaped`, sub-ep 6 | — | 6.54 / 11.16 |
   | 960 h `shaped`, sub-ep 7 | — | 6.69 / 11.03 |
   | 960 h `shaped`, sub-ep 8 | — | 5.97 / 10.66 |
   | 960 h `shaped`, sub-ep 9 | — | 6.49 / 11.31 |
   | 960 h `shaped`, sub-ep 10 (last) | — | 6.46 / 11.41 |
   | — the same, test-clean / test-other | — | 6.94 / 12.16 |

   Every number in this log is the sclite WER the evaluation actually reports; no normalization,
   punctuation stripping or other scoring variant is applied to any of them (user's instruction,
   2026-08-08).

5. **PLAN §0d(ii): the offline gate read.** The shipped n=512 rollout dumps repriced with lam_lm under
   the stock vs the adapted donor (`ReplaceLmPriorJob`), same bed / n / G, then the full
   lam_lm x lam_len grid re-scored on both beds (`RewardShapeSweepJob`); the audio-free null is the
   `noar` family, whose rows differ only in lam_len. Every margin below is taken against the
   **strongest** null per statistic over that family (theta_0 bed: spearman 0.4668 -> 0.5258,
   sel_wer 0.1621 -> 0.1414, eta 0.1145 -> 0.3290; G-track bed: 0.5216 -> 0.5952, 0.1421 -> 0.1315,
   0.4156 -> 0.5921), which is the conservative reading the gate's "margin over the audio-free null"
   asks for. Rows are T=0.7, lam_len=0 — the loops' operating point — and margin is signed so that
   positive always means "the audio-conditioned scorer beats free English".

   | bed, lam_lm | donor | spearman | sel_wer | eta | margin: spearman / sel_wer / eta |
   |---|---|---|---|---|---|
   | theta_0, lam_lm=1 (loops) | stock | 0.6684 | 0.1578 | 0.1583 | +0.2016 / +0.0042 / +0.0438 |
   | theta_0, lam_lm=1 (loops) | adapted | **0.7132** | **0.1187** | **0.5644** | +0.1874 / **+0.0227** / **+0.2355** |
   | theta_0, lam_lm=0.3 (re-swept peak) | stock | 0.7520 | 0.0977 | 0.7823 | +0.2852 / +0.0643 / +0.6677 |
   | theta_0, lam_lm=0.3 (re-swept peak) | adapted | 0.7772 | 0.0919 | 0.8424 | +0.2514 / +0.0495 / +0.5134 |
   | G-track, lam_lm=1 | stock | 0.6274 | 0.1307 | 0.6062 | +0.1058 / +0.0114 / +0.1906 |
   | G-track, lam_lm=1 | adapted | 0.6753 | 0.1244 | 0.7118 | +0.0800 / +0.0072 / +0.1197 |
   | G-track, lam_lm=0.3 | stock | 0.6453 | 0.1266 | 0.6738 | +0.1238 / +0.0155 / +0.2582 |
   | G-track, lam_lm=0.3 | adapted | 0.6866 | 0.1216 | 0.7575 | +0.0914 / +0.0099 / +0.1654 |

6. **The donor swap on the label-free bed.** `config_sae_3d_gtrack_v1.theta0g_av_sft` refactored so
   theta_0^G and its donor arm share every argument but `qwen_hub_dir`, then re-run on the adapted
   donor as `seed10h_layer15_gtrack_pseudo_tc100_lbslm`, dev-only recogs at ep 2/4/6/8/10 exactly as
   the shipped init. This measures the channel the §0d(ii) gate cannot see — concl. 5 puts most of the
   theta_0-bed gain in the retraining while the gate's G-track rows price only the reward — and it
   stops at the init: no loop, unit refit or scorer is built from it, so the ~2.2 h x 4 GPU cost buys
   a dev WER against theta_0^G's 13.89 / 18.34 and nothing is committed to a 960 h loop. Still
   label-free: the transcripts stay the §1d student's, the donor saw only unpaired `librispeech-lm-norm`
   text, and nothing selects on annotation (the donor is the last sub-epoch of a fixed-length pass).

   | epoch | stock donor (theta_0^G) | adapted donor |
   |---|---|---|
   | 2 | 175.25 / 180.54 | 185.37 / 189.17 |
   | 4 | 28.27 / 33.04 | **14.62 / 18.91** |
   | 6 | 14.46 / 19.09 | **13.94 / 18.55** |
   | 8 | 13.91 / 18.74 | **13.78 / 18.16** |
   | 10 (the shipped init) | 13.89 / 18.34 | **13.66 / 18.03** |

## Conclusion

1. (1) **A donor swap is not a config change, it is a retraining**: the AV checkpoint carries the 2.03 B
   donor weights and the loop loads them over whatever `av_args()` named, so §0d only lands if the 10 h
   AV SFT is re-run on the adapted donor — which is what makes "use the finetuned ckpt for AV init"
   the whole of the phase rather than a one-line edit.
2. (2) The normalization this phase adopts is also the direct fix for `SAE_3A.md` conclusion 37: a prior
   trained on lowercase, punctuation-free text prices adding a comma or a capital NEGATIVE, where the
   stock multilingual-web donor priced it positive and `recon` was exactly blind to it.
3. (3) **The §0d(i) pre-check PASSES**: dev/test 8-gram overlap is 1.59–3.36 % against the
   train-clean-100 control's 42.98 %, a 13–27x separation, so the LM corpus does not contain the
   evaluation books and the prior may be trained on all of it. The whole-sentence statistic is the one
   to ignore — the control scores 3.67 % there, indistinguishable from dev/test, because the corpus is
   segmented differently from LibriSpeech utterances; only the 8-gram statistic has power, which is
   what the positive control was there to reveal.
4. (2) The full pass is the donor: dev perplexity falls monotonically 40.97 -> 36.96 with no turn, so
   "best by dev" and "most data seen" are the same checkpoint and the adaptation shows no sign of
   over-specializing on this corpus within one pass. Read dev from `work/learning_rates`, not the
   per-step `log_ppl` in the training log — under `sorted_reverse` eval ordering the running display is
   length-skewed and disagreed by ~0.8 nats.
   - **WRONG in part (2026-08-08, verifier):** "one pass by construction" is false as run — under
     `torch_distributed` each rank permutes with its own `random_seed_offset = rank*16127`
     (returnn `datasets/basic.py:283`), so the four ranks sliced four different orderings: one pass of
     VOLUME but only 68.4 % of distinct lines seen (= 1-(3/4)^4), ~31.6 % never seen, duplicates up
     to 4x; the dev-curve facts and the selected checkpoint stand.
5. (4) **The donor swap alone buys 5.5 / 5.1 dev and 3.3 / 6.4 test, before any loop runs**: theta_0'
   is 11.43 / 15.54 against theta_0's 16.91 / 20.64 on one changed argument, which is more than the
   whole 2S loop ever earned from theta_0 (its best, ep3, was 12.99 / 16.20) and better on test than
   psi 10 h `recon` ep4 (11.99 / 14.34 against 12.20 / 15.98). So the prior was costing the AV SFT
   itself, not only the GRPO reward — the phase's premise was that a donor swap is a retraining
   (concl. 1), and the retraining is where most of this gain appears.
6. (5) **The gate splits, and it splits on lam_lm, not on the donor**: absolute ranking improves
   everywhere with the adapted prior, but so does the audio-free null on both beds, so the gate's feared
   mechanism is real and the verdict depends entirely on where the loop sits — at the loops' own
   lam_lm=1 on the theta_0 bed the sel_wer and eta margins WIDEN (+0.004 -> +0.023, +0.044 -> +0.236;
   stock at lam_lm=1 was all but null-equivalent on eta) while spearman's shrinks by 0.014, whereas at
   the re-swept peak lam_lm=0.3 and on the G-track bed at every lam_lm all three margins SHRINK. So a
   better English prior buys ranking that the null can also buy, and the only cell where the adapted
   donor clearly buys *audio*-attributable ranking is the operating point the loops already run at —
   which is why re-tuning to the peak would trade the phase's gain for free English. Planner verdict
   (`PLAN.md` §0d Status, awaiting the user): pass for theta_0-bed loop use at lam_lm=1, do not chase
   lam_lm=0.3, no G-track use licensed.
7. (4) **The loop keeps the init's lead rather than closing it, and the saving is in sample budget**:
   the adapted arm is 7.14 / 11.35 at sub-ep 1 and 5.95 / 10.16 at sub-ep 2, against the stock arm's
   10.89 / 16.49 and 7.42 / 12.51 — so by its SECOND sub-epoch it has reached the stock arm's EIGHTH
   and last (6.06 / 10.31), at a quarter of the loop's sample budget. Read the 0.11 / 0.15 by which it
   passes that endpoint as a tie, not as superiority; the claim the numbers support at that point is
   same quality, 4x cheaper, which is the usability form of the win. By sub-ep 4 it is 5.23 / 9.06 and
   the tie reading no longer holds — that is 0.83 / 1.25 below the stock arm's endpoint and below
   every sub-epoch the stock arm ever produced, with four sub-epochs still to run. Sub-ep 2-5 oscillate
   inside 5.23-5.95 / 9.06-10.55 without a trend either way, which is the same behaviour the stock arm
   shows over its own sub-ep 3-8 (6.06-6.79 / 10.31-10.79) one band higher — so no sub-epoch here is
   yet a turn, and the arm's endpoint is the number to read.
8. (4) **On the 960 h bed the stock arm's WER collapsed after one sub-epoch and the adapted arm's has
   not**: 6.87 / 10.98 -> 18.22 / 22.21 -> 18.82 / 23.51 -> 19.67 / 24.92, against the adapted arm's
   6.28 / 10.25 at the matched sub-epoch. The collapse is where `SAE_3A.md` concl. 37 located the
   punctuation mispricing, and §0d concl. 2 predicts the adapted prior removes its cause, so sub-ep 2
   is the test: the stock arm went +11.4 / +11.2 there and the adapted arm's sub-ep 2 either does or
   does not.
9. (4) **The adapted donor removes the 960 h collapse**: at the pre-registered test point the adapted
   arm goes 6.28 / 10.25 -> **5.34 / 9.50** where the stock arm went 6.87 / 10.98 -> 18.22 / 22.21,
   which is better than every sub-epoch either 960 h arm has produced and better than the 100 h
   adapted arm at the same sub-epoch (5.95 / 10.16), so the 960 h bed's failure was the prior and not
   the bed — eight sub-epochs still to run.
10. (6) **The donor swap does NOT transfer to the label-free init: it buys speed there, not quality.**
    theta_0^G on the adapted donor ends at 13.66 / 18.03 against the stock init's 13.89 / 18.34 —
    0.23 / 0.31, about a twentieth of the 5.5 / 5.1 the identical swap bought on the theta_0 bed
    (concl. 3) — while the ep4 gap is enormous (14.62 / 18.91 vs 28.27 / 33.04), i.e. the better
    English reaches the same operating point in fewer epochs but does not raise it. Since the reward
    channel was already unlicensed on this bed (concl. 6), BOTH channels of the swap are now measured
    on the G-track and neither pays: what caps that init is the pseudo-label quality, not the donor's
    English, so a 960 h G-track loop cannot be justified by the donor swap and the lever stays on the
    §1d text (`PLAN_3E1` D4) rather than on the prior.

11. (4) **The 100 h arm is closed and the adapted donor wins it at the label-free pin**: all 8
    sub-epochs are in, and at `checkpoint_last` — the convention that selects on no annotation — the
    adapted arm ships **5.22 / 9.67** against the stock arm's 6.06 / 10.31, having passed that
    endpoint by sub-epoch 2 of 8; dev-other's own minimum sits earlier (9.06 at sub-ep 4) and is NOT
    what the arm ships, so read the 0.84 / 0.64 as the pinned comparison and the oscillation between
    sub-ep 4 and 8 as the noise band of this bed.

12. (4) **The 960 h arm's sub-ep 3 regression is ALL insertions, and they are not the suspect set**:
    dev-clean goes 5.34 -> 6.56 with substitutions still falling (2154 -> 2032) and deletions flat
    while insertions jump 488 -> 1204, of which `and` 351 + `but` 215 + `i` 208 are 64 % and "to" is
    5 tokens (4.7 % -> 0.4 %); dev-other repeats it (784 -> 1672 insertions, `i`/`and`/`but` = 53 %,
    "to" 0.6 %). This is the exploit class the D1 audit predicted from the lattice's per-inserted-state
    price (`SAE_3E1.md` c10) arriving on a live loop — every minimal-state word, not the excess-mass
    suspect set — so the pre-registered "to"-share monitor read IMPROVING through a 2.5x insertion
    blow-up and is, alone, blind to the failure it was built for.

13. (4) **The regression is permanent, and the loop is still learning underneath it**: five sub-epochs
    past the 5.34 / 9.50 peak the arm sits at 6.32-6.89 / 11.03-11.31 with no recovery, while
    dev-clean substitutions keep FALLING the whole way (2246 -> 2154 -> 2032 -> 1975 -> 1992 -> 1924
    -> 1979) and insertions hold at 2.4-2.9x the peak's (488 -> 1182-1415); dev-other repeats both
    (3837 -> 3552 substitutions, 784 -> 1592-1794 insertions). Every point of the regression is the
    insertion channel, so the lattice's insertion price (`SAE_3E1.md` c27, c28) is the operative
    defect at the WER level and not only in the probes, and sub-ep 2 stays this bed's best checkpoint.
    The arm has since run out to its last sub-epoch, and both halves of this need amending. Sub-ep 8
    recovers to 5.97 / 10.66, the best since the peak, purely by insertions falling to 1170 / 1640 --
    the floor of the sub-ep 3-7 band -- on new-low substitutions (1941 / 3510); sub-ep 9 and 10 give it
    straight back through insertions alone (1450 / 1954 -> 1416 / 1964) while substitutions go flat
    (1942 / 3535 -> 1950 / 3557). So the insertion channel moves WER in BOTH directions, and the "still
    learning underneath" half is now closed: both channels have stopped moving and the run ends at
    6.46 / 11.41 dev, 6.94 / 12.16 test, short of sub-ep 2 the whole way.

## Catalog

*(job dirs filled as they are created; cite `work/.../<Class>.<hash>`, never an `output/` alias)*

- LM finetune: `work/i6_core/returnn/training/ReturnnTrainingJob.SIhrrhRwkdh7`
- theta_0 re-SFT on the adapted donor: `work/i6_core/returnn/training/ReturnnTrainingJob.44uD7rH147GR`
- psi loop 100 h `shaped`, `_lbslm`: `work/i6_core/returnn/training/ReturnnTrainingJob.fFp8sXTA5Wug`
- psi loop 960 h `shaped`, `_lbslm`, 1 pass: `work/i6_core/returnn/training/ReturnnTrainingJob.vhyvv2waeU16`
  (finished 2026-08-10, all 10 sub-epochs); its last sub-epoch's WERs are
  `work/i6_core/recognition/scoring/ScliteJob.{Tvi0Xzt7XVdr,IISpNt7X9ONk,inWvxShQlJxX,yeLOxPObL00I}`
  (dev-clean, dev-other, test-clean, test-other)
- Leak pre-check: `work/speech_llm/sae/lm_prior_jobs/LmTextLeakCheckJob.zg9RUru8vQK4`
- Lowercased LM corpus: `work/speech_llm/sae/lm_prior_jobs/LowercaseTextCorpusJob.SqAFPqiRBD9k`
- Adapted-donor hub dir (what theta_0' consumed): `work/speech_llm/sae/lm_prior_jobs/ExportHfLmDirJob.460dedSQ4kAG`
- Gate read (5), repriced dumps: `work/speech_llm/sae/lm_prior_jobs/ReplaceLmPriorJob.{kR3sv14S4KNS,Wj2MbrD8gTgX}`
- Gate read (5), sweeps: `work/speech_llm/sae/reward_shape_sweep/RewardShapeSweepJob.{yCfwZSr3huv7,GUbnTUM2ggiv}`
  (stock: theta_0 bed, G-track bed) and `.{oaOqWCrd3ZPO,FPIKAU6TkEK4}` (adapted, same order)
- Approach 4 WERs: `work/i6_core/recognition/scoring/ScliteJob.{dH8VpeVgq6J5,K7YOtW6iNyjk,PZJ3AAQFeGhI,c7lErYqku073}`
  (stock theta_0: dev-clean, dev-other, test-clean, test-other) and
  `.{RycN2cVVrC4E,JAssxRJb2E6A,1r4Cx0mSDKP4,69ZJMUcxapi8}` (theta_0', same order)
- Stock-donor 960 h control: stopped at sub-ep4 and deleted 2026-08-08; its four sub-epochs are
  `SAE_3A.md` conclusion 39, whose scoring jobs are catalogued there
- Stock-donor theta_0: `work/i6_core/returnn/training/ReturnnTrainingJob.OLzy9Q2oC3mU`
- theta_0^G on the adapted donor (approach 6): `work/i6_core/returnn/training/ReturnnTrainingJob.XQoLoNsNHI6i`;
  the shipped stock-donor theta_0^G it is read against is `.2fb02hGUdHNj` (ep10, 13.89 / 18.34)

## Verifier feedback

- 2026-08-08 — Full audit (numbers, WER provenance, code): every table in approaches 1–4
  reproduces exactly. WER rows traced to `ScliteJob.{dH8VpeVgq6J5,K7YOtW6iNyjk,PZJ3AAQFeGhI,c7lErYqku073}`
  (stock) / `.{RycN2cVVrC4E,JAssxRJb2E6A,1r4Cx0mSDKP4,69ZJMUcxapi8}` (lbslm), ep50 =
  last-epoch-by-construction for both arms, eval pipelines byte-identical (returnn.config diff =
  `hf_hub_cache_dir` only); psi-view deltas verified, max 0.06 (theta_0 dev-other); leak table,
  dev-ppl table, corpus stats (independent wc), and the 2.03 B/423/112/311 checkpoint key counts
  all exact. Conclusion 5's comparisons quote their sources correctly (12.20/15.98 is genuinely
  test in `SAE_3A.md`).
- 2026-08-08 — One-pass claim in approach 2 flipped in part: see the conclusion-4 correction
  (rank-dependent shuffle seeds; 68.4 % distinct coverage at one-pass volume). No logged number
  invalidated. If the donor is ever re-trained, pin a common `random_seed_offset` (hash-changing).
- 2026-08-08 — The PLAN §0d(ii) gate read EXISTS, is FINISHED, and is missing from this log:
  `ReplaceLmPriorJob.{kR3sv14S4KNS,Wj2MbrD8gTgX}` reprice the shipped n=512 dumps;
  `RewardShapeSweepJob.{oaOqWCrd3ZPO,FPIKAU6TkEK4}` (adapted) vs `.{yCfwZSr3huv7,GUbnTUM2ggiv}`
  (stock, hash-reused). Log the stock-vs-adapted table (T=0.7 rows incl. the `noar` null) as its
  own approach + conclusion; the planner's verdict on it is in PLAN.md §0d Status (margin over the
  audio-free null is statistic-, lam-, and bed-dependent; null itself strengthens on both beds).
- 2026-08-08 — Loop arms started before the (ii) read ("loop use only after that read") — noted as
  procedural; at unchanged lam=1/T=0.7 they are the clean donor-axis A/B against the stock shaped
  arms and stand. Any move toward the re-swept peak (lm=0.3) is a planner/user decision, not an
  implementer default.
- 2026-08-08 — Catalog gaps: add the eight scoring jobs above, the gate-read jobs, and
  `ExportHfLmDirJob.460dedSQ4kAG` (the hub dir theta_0' actually consumed).
- 2026-08-18 — the gate read (5)'s donor comparison MIXES SCORING CONVENTIONS on
  generated rows, found via the HOM-0b terminator bug and planner-traced through the job
  graph: the stock arm's sweeps (`RewardShapeSweepJob.{yCfwZSr3huv7,GUbnTUM2ggiv}`) read
  the dumps' ORIGINAL `lm_prior` column, scored by the loop WITH the end-of-sequence
  token, while the adapted arm's sweeps (`.oaOqWCrd3ZPO,.FPIKAU6TkEK4`, consuming the
  `ReplaceLmPriorJob` outputs) re-tokenize the decoded text WITHOUT the terminator and
  divide by the dump's terminator-inclusive `n_tokens` — so adapted priors are biased
  upward by one token's log-probability, a per-row (not constant) shift of order
  |logprob(EOS)|/n_tokens. The job's own comparability check fired and went unread:
  `retok_exact_rate` 0.0916 / 0.0813 with `mean_retok_len` short of `mean_n_tokens` by
  ~0.7 and ~0.9 — the terminator signature — but the statistic was never banked in this
  log. NOT a conclusion flip yet: the stock arm and the audio-free null family are
  unaffected, and the largest quoted margin (eta widening +0.044 -> +0.236 at lam_lm=1,
  theta_0 bed) is far above the plausible bias scale; but the SMALL margins in
  conclusion 6 (sel_wer +0.004 -> +0.023; spearman -0.014) are within it. PRICED FIX
  (cheap, ~1 GPU-h per dump + CPU sweeps): re-run the two `ReplaceLmPriorJob`s with the
  0b terminator convention and re-sweep the adapted arm; until then the §0d planner
  verdict awaiting the user ("pass for theta_0-bed loop use at lam_lm=1") carries this
  caveat. Catalog note: the adapted-arm sweep hashes above are absent from the Catalog's
  gate-read row, which cites the stock pair only.
