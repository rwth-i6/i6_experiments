# SAE §3d — the G-track: GAN-init fully-unsupervised autoencoder

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
7. **§3d's question is UNANSWERED, not answered against the autoencoder.** The question compares two
   refinement operators and neither was run on this init: the loop was never launched and no
   self-training arm on AV^G's own pseudo-labels exists. What was measured is the gate — a property of
   the reward at the init point — and a decision not to fund the comparison. Declining to run a race is
   not losing it.
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

## Verifier feedback

**2026-08-05.** Both qualifications on the gate are established elsewhere and are load-bearing
(`SAE_2S.md` approach 15): "worse than random" is not resolvable at n=128 (CI [-0.202, +0.096]), and
the incumbent's +0.2246 anchor was measured on the 10 h seed bed — theta_0's own training data, oracle
WER 0.0316 — so it is not this arm's comparator. Folded into conclusions 5-6.

**2026-08-20.** An exhaustive config, alias, work-directory and history audit confirms that no
GAN-init AV SFT on 960 h pseudo-text has run. Theta_0^G (`ReturnnTrainingJob.2fb02hGUdHNj`) uses
the 28,539 train-clean-100 hypotheses in `TransformAndMapHuggingFaceDatasetJob.XqPlB1nRGHyK`;
the plain theta_0^G 960 h arms import that checkpoint, while GAN+HOM imports another
train-clean-100-derived pseudo-SFT. In every case 960 h is only the later unlabeled loop bed.
`TransformAndMapHuggingFaceDatasetJob.1c6JQRMlzCyy` provides reusable HF/Ogg Arrow audio for
281,241 utterances, but neither full pseudo-text nor the FLAC+manifest layout consumed by the word
decoder exists. The missing scale arm therefore requires scratch-resident or packed/sharded audio
staging plus §1d-student word decoding of the remaining 860 h before SFT; a per-utterance project
tree is unsafe at the reported 3,584,516 / 4,000,000 project inode usage. The user also funds one
own-label generation from theta_0 and theta_0^G with same-start §1d-label comparators; no result or
conclusion exists yet.
Normative specification and gate: `PLAN.md` §3d.A.
