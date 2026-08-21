# SAE §3d — the G-track: GAN-init fully-unsupervised autoencoder

## State
<!-- Overwritten in place, never appended; deleted at phase close. In-flight runs (job dir + the
question each answers), blockers, next action, proposals for the planner. -->

In flight 2026-08-21 17:45: the §3d.A scale arm, theta_0^G960 -- the from-scratch AV SFT on
pseudo-labels for all 281,241 utterances of the 960 h bed, one corpus pass partitioned into ten
sub-epochs with evaluation at sub-epoch 10, i.e. the registered exposure match against ten 100 h
passes for theta_0^G (`T/ReturnnTrainingJob.HuSkdbuVRg6d`, alias `sae_2s/config_sae_2s_av_sft_v1/
seed10h_layer15_gtrack_pseudo_960h_onepass/training`). It answers whether pseudo-labeling all 960 h
gives a better label-free AV starting point than the existing 100 h start. Started from scratch
17:42 (no earlier checkpoint exists); about 27 min per sub-epoch on four GPUs. Its sub-epoch-10
dev-clean/dev-other recognition, scoring and robust-WER chain (11 jobs) waits behind it.

This arm made no progress between 2026-08-20 19:33 and 2026-08-21 17:35 and the loss was purely
operational, not experimental: its two upstream dataset-transform jobs finished, but one finished
job's worker process never exited and held all four cpus of that manager's login-node engine, so the
training stayed runnable and unsubmitted for 22 h with the manager alive and no error anywhere.
Cleared by restarting the manager; the training was submitted within a minute. No job was cleared,
re-run or deleted and no hash moved, so nothing on record is affected. Recurrence and detection are
in the memory entry on workers of finished jobs; the watcher now heartbeats for it.

Blockers: none. Next action: let the training finish, then read the WER chain.

## Approach

**1. Build a label-free init and gate it before funding a loop.** Chain: §1d student word decode
(exists, 17.96 / 21.87) -> AV^G = AV SFT from scratch on audio -> pseudo-text -> units = k-means over
AV^G's **own** post-adapter states -> AR^G = AR SFT on pseudo-text -> those units, p10 -> §2.5(d)
reward-rank probe on AV^G's own rollouts. §1d output is admissible here as **initialization only**
(user carve-out): never an in-loop teacher, never the reward, never a selection signal, and the AR's
*target* stays the audio-derived unit stream, so the pseudo-text is conditioning input on both sides.
AV^G and AR^G are two separate initializations, not one model state — every dev WER in this log is an
AV^G decode and says nothing about the AR. Step 3 refits the codebook on the new policy's states
rather than reusing AV_10h's, because those units are a measurement made through a 10 h-supervised
adapter. Both SFTs are pinned at their last epoch, never at a dev optimum; the AV SFT is from scratch
with zero `preload_from_files`; the probe reads the **gold** tc100 dir, since a probe scored against
pseudo-transcripts would only report agreement with the §1d student. Bed = train-clean-100 rather than
PLAN's 960 h, because the flashlight beam decode at ~16 utts/min would cost ~260 GPU-hours *before*
the gate whose purpose is to decide whether the track is worth funding.

| system | dev-clean | dev-other | what it is |
|---|---|---|---|
| §1d CTC student (word decode) | 17.96 | 21.87 | the pseudo-labels AV^G trains on |
| AV_10h (theta_0), 10 h *paired* | 16.91 | 20.71 | the other track's init |
| self-training from AV_10h on the same pseudo-labels, ep4 | 13.05 | 17.74 | the operator this track must beat |
| **AV^G, ep10 (pinned)** | **13.89** | **18.34** | this track's init |

AV^G trajectory ep2/4/6/8/10 dev-clean: 175.25 / 28.27 / 14.46 / 13.91 / 13.89. Units refit on AV^G:
0/500 dead clusters, usage entropy 0.9702, fit-vs-assign frame agreement 1.000000.

**2. The gate, at 128 utterances x G=12.** Pre-registered bar: eta(T=0.7) >= 0.2 with gap_true > 0.

| T | spearman | frac rho>0 | sel_wer | mean_wer | oracle_wer | gap_true | **eta** |
|---|---|---|---|---|---|---|---|
| 0.3 | 0.053 | 0.487 | 0.1427 | 0.1435 | 0.1132 | -0.0003 | 0.0243 |
| 0.5 | 0.026 | 0.508 | 0.1575 | 0.1530 | 0.1082 | -0.0007 | -0.1000 |
| **0.7 (the gate)** | 0.100 | 0.591 | 0.1765 | 0.1736 | 0.1113 | +0.0011 | **-0.0462** |
| 1.0 | 0.593 | 0.953 | 0.3869 | 0.5118 | 0.2827 | +0.0811 | 0.5451 |

Sharper than eta: the gold transcript out-scores the mean rollout on only **75/128** utterances and
the best rollout on 24/128. Verified before being believed — the probe reads the gold dir
(`nUHRlXQVM0H3`, not the pseudo dir `XqPlB1nRGHyK`), `fix_true_case: True`, the dumped `true` rows are
lowercase at WER 0.000, and eta recomputed by hand from `rollouts.jsonl` reproduces -0.0462.

**3. Re-reading the gate under a fixed reward (n=512, one bed, honest split-half weights).** Both
etas above were measured through a composition that fails for both arms, so the arms were re-compared
on the **audio margin** = eta(shaped) - eta(best AR-free null), with both weights chosen per arm on
half the 512 utterances and scored on the other half.

| T | theta_0 | AR^G |
|---|---|---|
| 0.5 | **+0.201** | +0.003 |
| 0.7 | **+0.109** | +0.008 |
| 0.9 | **+0.192** | +0.031 |
| 1.0 | **+0.112** | +0.033 |

Same-bed, same-n, same-G reward columns at T=0.7, G=12: theta_0 gap_true +0.0124 / reward std 0.0324 /
spearman 0.170 against AR^G's +0.0020 / 0.0124 / 0.094.

**4. One-generation iterative self-training on train-clean-100.** Each start produced a fresh
beam-4 pseudo-transcript for all 28,539 training utterances, then continued for exactly four epochs
on either that own-label generation or the fixed §1d labels. Evaluation is corpus WER on all 2,703
dev-clean and 2,864 dev-other utterances. The table reports the fixed epoch-4 endpoint, with no
dev-set checkpoint selection.

| start | start WER | fixed-§1d labels, ep4 | own labels, ep4 | own minus fixed |
|---|---:|---:|---:|---:|
| theta_0 (10 h paired) | 16.91 / 20.64 | **13.05 / 17.74** | 15.87 / 20.26 | +2.82 / +2.52 |
| theta_0^G (GAN init) | **13.89 / 18.34** | 13.81 / 18.13 | 15.19 / 19.24 | +1.38 / +1.11 |

The own-label files have exact 28,539-ID coverage and zero empty hypotheses. Relative to the fixed
§1d pseudo-labels (not to gold transcripts), theta_0 changes 97.63% of utterances at 28.28% word
edit rate; theta_0^G changes 93.70% at 15.81% word edit rate. Thus the negative result is not caused
by accidentally copying the fixed labels.

## Conclusion

1. (1) **AV^G is the project's best label-free ASR**: it beats the §1d student that taught it by ~4
   points and the 10 h *paired* AV_10h by ~3 / ~2.4, using no transcripts at all, and sits 0.84 / 0.60
   behind the 10 h-supervised self-training control. That last row is the consistency check that makes
   the first two believable — two runs on the same pseudo-transcripts, one from scratch and one
   continuing from a supervised AV, land within 0.84 WER of each other, so ~4 points of
   student-over-teacher is what this AV architecture extracts from these pseudo-labels and
   initialization contributes under 1 point of it.
2. (2) The gate is missed: eta -0.0462 against a 0.2 bar, with real headroom present (within-group WER
   spread 0.044, mean 0.174 against oracle 0.111) that the reward does not see. T=1.0's healthy-looking
   eta is the known artefact — mean WER there is 0.512, so the reward separates garbage from
   non-garbage and nothing finer.
3. (2) Upstream cause, again: AR^G ends at dev CE 5.611 against a unit-stream usage entropy of 6.030,
   so the gold transcript buys 0.42 of 6.03 nats and a reward computed from an ~93 %-blind AR is
   almost constant by construction (within-group reward std 0.0123). That 0.42 is what text buys
   *unaided* — under p10 every unit position's input embedding is zeroed — which is more creditable
   than it reads, level with the 100 h gold-text ceiling arm's 0.409 and above the incumbent's 0.263.
   **AR^G is the best of the three scorers in information terms and the worst by eta.**
4. (2) This is the wall reproduced at a fresh init with a freshly-fitted codebook (different policy,
   states, k-means; 0/500 dead, 0.9702 entropy). Three initializations now fail the same gate the same
   way, which localizes it upstream of the init and the codebook.
5. (2) **The 0.2 bar is WITHDRAWN** (2026-08-05). It was justified by bracketing — eta 0.225 -> loop
   works, 0.16 -> loop collapses — and the n=512 re-run removed that justification: on gold tc100, the
   bed this gate used, the incumbent theta_0 reads -0.023 and its loop demonstrably runs and gains
   ~1.6 WER. A bar that rejects the working configuration cannot be used to reject anything. What
   survives is the unconfounded same-bed comparison: **AR^G prefers the truth 6.2x more weakly than
   the incumbent and spreads candidates 2.6x less.**
6. (3) Under the best composition available, with each arm free to choose its own weights, theta_0's
   margin is positive at all five temperatures (mean +0.148) and AR^G's is indistinguishable from
   zero. Its extra information about the unit stream is **redundant with what a language model already
   supplies**, so the decision survives the composition being fixed — no longer only "not worth
   funding" but "measured not to contribute under the best reward available".
7. **WRONG after the D6-PERIODIC/GAN launch: the original statement that neither refinement operator
   had run on this init is stale.** A reconstruction-loop family now has a six-leg prefix: from
   theta_0^G at 13.89/18.34, periodic refresh reaches 12.85/17.89 only at leg 2 and then degrades to
   18.38/24.01 by leg 6; the repaired frozen-scorer reference reaches 12.68/17.57 at its best matched
   point but slips to 13.54/18.56 one sub-epoch later. This establishes no durable GAN-init loop gain.
   The other half of §3d's operator question remains unanswered until the §3d.A same-start own-label
   and fixed-§1d-label continuations produce fixed-endpoint WER.
   **WRONG after §3d.A completed:** that final sentence is now stale; conclusion 10 gives the answer.
8. The comparison §3d asks for *has* been run at 10 h from a paired init, and the loop won: last-epoch,
   no dev-WER checkpoint picking, GRPO joint-AR ep4 13.15 / **16.13** on 2849 utterances against the
   self-training control's 13.05 / 17.74 on 28,539, with the shuffled control (202.54 / 207.59 at ep1)
   showing the reward is what does it. The control saturates at ~17.1-17.7, where a pseudo-label
   student must stall against a 21.87 teacher, while the loop passes through that floor — a
   reconstruction reward has no teacher ceiling. So what this gate blocks is narrower than "the
   autoencoder": both arms there start from the 10 h **paired** theta_0, and the gate's answer is that
   the *reward* does not survive replacing the paired seed with a label-free init.
9. The operator that produced conclusion 1 — the project's best label-free ASR — is plain pseudo-label
   self-training, no loop.
10. (4) **One-generation own-label self-training fails the same-start comparison for both starts.**
    At the fixed epoch-4 endpoint, theta_0's own labels are 2.82 / 2.52 WER worse than fixed §1d
    labels. For theta_0^G they are 1.38 / 1.11 worse and also 1.30 / 0.90 worse than the teacher
    start itself. Theta_0's own-label run improves on its start by 1.04 / 0.38, but most of the useful
    gain still comes from retaining the external §1d labels. The preregistered criterion required an
    own-label win over both anchors on both dev splits, so no second generation is warranted.

## Catalog

`T/` = `work/i6_core/returnn/training/`, `F/` = `work/i6_core/returnn/forward/`.

| artifact | path |
|---|---|
| graph (71 jobs, 15 finished upstream) | `config/sae_3d_gtrack.py` |
| **AV^G (theta_0^G), 10 epochs** | `T/ReturnnTrainingJob.2fb02hGUdHNj` |
| AR^G | `T/ReturnnTrainingJob.cGl2KHUclIlP` |
| the gate probe | `F/ReturnnForwardJobV2.faxctn9Uzcn6` |
| pseudo-transcript dir (shared with the self-training control) | `TransformAndMapHuggingFaceDatasetJob.XqPlB1nRGHyK` |
| gold tc100 dir the probe reads | `TransformAndMapHuggingFaceDatasetJob.nUHRlXQVM0H3` |
| 10 h loop comparison arms | `T/ReturnnTrainingJob.MquKQUTRgZj9` (joint), `.qmkzvAX3gOVW` (frozen) |
| 960 h packed-reader preflight, exact banked-hypothesis gate | `W/PackedWav2Vec2KenlmDecodeJob.mFKyL6x2Gc9o` -> `W/PackedDecodeAgreementJob.xEBbTHwTJScE` |
| 860 h packed §1d decode; four shards per four-GPU node, merged with banked tc100 | `W/PackedWav2Vec2KenlmDecodeJob.4CPtPQPBEczq` |
| theta_0^G960 one-pass AV SFT (waits for the full pseudo-text bed) | `T/ReturnnTrainingJob.1OALJ3Yaa9UL` |
| one-generation theta_0 / theta_0^G teacher hypotheses | `S/scorer_diag/SearchOutHypsJob.qwcf5P0za2SI` / `.VSbqKSm4Bmyo` |
| one-generation pseudo-label diagnostics | `S/selftrain/PseudoLabelDiagnosticsJob.Cixv9XXsMwQz` / `.MumIuNoveFvq` |
| theta_0 fixed-§1d comparator (banked) / own-label continuation | `T/ReturnnTrainingJob.xChfzEkd4CGE` / `.aTR981EDGPZe` |
| theta_0^G fixed-§1d comparator / own-label continuation | `T/ReturnnTrainingJob.sYvNhnEDQvli` / `.e4uDmyBdTlEG` |
| epoch-4 WER scorers, theta_0 fixed / own | `work/i6_core/recognition/scoring/ScliteJob.gN9U1SXNomnD`, `.vczyFFxliz4X` / `.OkZQ1bc87Jhi`, `.Hcb0d1OBuMXc` |
| epoch-4 WER scorers, theta_0^G fixed / own | `work/i6_core/recognition/scoring/ScliteJob.gJCk8XigFAux`, `.GnqbfR2a4QXw` / `.odINoSBb3YIS`, `.3i1nk9ppbhb1` |

`W/` = `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/word_decode/`,
`S/` = `work/speech_llm/sae/`.

## Verifier feedback

**2026-08-05.** Both qualifications on the gate are established elsewhere and are load-bearing
(`SAE_2S.md` approach 15): "worse than random" is not resolvable at n=128 (CI [-0.202, +0.096]), and
the incumbent's +0.2246 anchor was measured on the 10 h seed bed — theta_0's own training data, oracle
WER 0.0316 — so it is not this arm's comparator. Folded into conclusions 5-6.

**2026-08-20.** Audit confirms that no GAN-init AV SFT on 960 h pseudo-text had previously run: the
plain 960 h loop arms import the 28,539-utterance tc100 theta_0^G checkpoint, while GAN+HOM imports a
different tc100-derived pseudo-SFT. The isolated one-generation graph pins theta_0 ep50 and theta_0^G
ep10; its four same-start arms differ within each pair only by pseudo-text targets; the banked theta_0
comparator remains `ReturnnTrainingJob.xChfzEkd4CGE`; and no later generation is wired. That graph
has now completed with exact pseudo-label coverage and all fixed-endpoint WER artifacts; approach 4
and conclusion 10 record its failed fresh-label gate. Separately, the original packed-reader
equivalence attempt is invalid evidence: `PackedDecodeAgreementJob.xEBbTHwTJScE` matched only
289/298 hypotheses. Do not waive exact decoder equivalence for the distinct 960 h arm. Normative
specification and gate: `PLAN.md` §3d.A.
