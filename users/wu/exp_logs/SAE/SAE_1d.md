# SAE §1d — Rung-0 self-training on the §1c wav2vec2-L15 GAN

## Approach

**1. No-Kaldi self-training (paper stages 1 + 3).** The published recipe's stage 2 is an HMM-GMM
realign that needs Kaldi, which does not build on this aarch64 cluster; its role (LM-cleaning the
pseudo-labels plus a stronger acoustic model) is approximated by the CTC student's own capacity plus
SpecAugment. Chain: GAN teacher greedy phones for every LS-100 train utterance -> fairseq
`audio_finetuning` + `wav2vec_ctc` CTC fine-tune of wav2vec2-lv60 (reference `vox_100h.yaml`, budget
reduced 80k -> 40k updates to fit the 11.5 h window) -> pure-torch viterbi decode. Selection is
label-free throughout: `disable_validation=true` means fairseq only ever writes
`checkpoint_last.pt`, and that is what is decoded — a best-on-gold checkpoint would be an oracle
number and is not reported.

| system | dev-clean | dev-other | selection |
|---|---|---|---|
| §1c GAN-init (baseline) | 0.173 | 0.214 | ppl (s0) |
| **§1d self-trained student (last ckpt)** | **0.138** | **0.172** | last (label-free) |
| delta (abs / rel) | -0.035 / -20 % | -0.042 / -20 % | |

**2. Word-level decode of the student.** Lexicon + 4-gram KenLM flashlight beam over the CTC
student's frame posteriors, sharded 8 ways over the train split (single-threaded beam at ~16 utts/min
cannot finish 28.5 k utterances inside one walltime slot). This is the Rung-0 word number and the
source of the pseudo-transcripts the G-track initializes from; hyps complete for all splits
(2703 / 2864 / 28539, zero empty).

| system | dev-clean WER | dev-other WER |
|---|---|---|
| §1d student + lexicon 4-gram word decode | **17.96** | **21.87** |

**3. Why there is no "GAN + word LM, no self-training" number.** Asked and settled by reading
`eval_per.py:31-42`, the code the pseudo-labels come from: `dense_x_only=True, no_softmax=True` ->
per-frame argmax -> collapse repeats -> drop `<SIL>`. No lexicon, no LM, no beam anywhere upstream of
the student.

## Conclusion

1. (1) Self-training works without Kaldi: ~20 % relative on both dev sets, selection-honest
   (`checkpoint_last`, same greedy viterbi and SIL-free gold convention as the baseline). The student
   recovers the selected-vs-oracle GAN seed spread (0.172 ~ the oracle-best seed's 0.168) *without*
   peeking at labels.
2. (1) §1d's role is **initialization and comparison anchor (Rung 0), not absolute-PER matching**: our
   0.172 lands roughly where the paper *starts*, because the gap is two-part and the larger part is
   upstream — our GAN-only init is 0.214 ppl-selected / 0.168 oracle against the paper's 0.136, before
   any self-training, and the dropped Kaldi HMM + word WFST stages are the rest.
3. (2) The fully label-free stack (GAN phones -> CTC student -> lexicon 4-gram) lands within ~1.1
   points of the 10 h *supervised* AV, which makes it a serious self-training bar and the working
   fallback init for the G-track.
4. (3) A lexicon-constrained beam decode of the GAN is **ill-posed, not merely unimplemented**:
   `<SIL>` is a modelled unit trained to align with actual silence, not an epsilon, so the collapse
   rule cannot represent a genuinely repeated phone; generation runs unnormalized (`no_softmax=True`)
   while flashlight's decoder assumes per-frame log-probabilities; and the generator was never trained
   with an alignment-summing objective, so beam scores over its frames have no probabilistic reading
   to hand the LM. The word LM therefore attaches to the *student*, where blank exists and the decode
   is well-posed. If the number is ever wanted anyway, a bad result would be ambiguous between "the
   GAN is worse in words" and "the decode is mis-specified"; the honest GAN-vs-student comparison
   already exists at phone level (0.214 vs 0.172, same convention).
5. Next lever if the absolute number is ever wanted: flashlight + KenLM phone-LM label cleaning plus a
   second self-training round (both install on aarch64) — planner's call, not attempted.

## Catalog

`W/` = `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/`.

| artifact | path |
|---|---|
| pipeline + entry point | `w2vu2/selftrain.py`, `config/sae_1d_selftrain.py` |
| GAN pseudo-labels (28,539 utts, 0 empty, mean 124.5 phones) | `W/selftrain/GanPseudoLabelJob.xjn6QnNqwEEH` |
| **CTC student (the §1d system)** | `W/selftrain/Wav2Vec2CtcFinetuneJob.BI1uYgPyTeQ0` |
| phone decodes | `W/selftrain/Wav2Vec2CtcDecodeJob.{1C2fmWJR1mcM,qqKPLPBEt1K3,YwTdIJTgPaaF}` |
| **word decode (Rung 0 + G-track pseudo-text source)** | `W/word_decode/Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks` |
| registered summaries | `output/sae/1d/{pseudo_labels.json,word_wer.json}` |

## Verifier feedback

- 2026-08-20 — Rung-0 completion verified end to end. CTC training
  `Wav2Vec2CtcFinetuneJob.BI1uYgPyTeQ0` and word decode
  `Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks` have finished markers; the latter completed every decode
  shard and collection, covers all 2,703 dev-clean and 2,864 dev-other utterances with zero empty
  hypotheses, and records WER fractions 0.179607/0.218654. The plan's former "pipeline unfinished"
  status was stale; Rung 0 is 17.96/21.87 and Phase 2a is unblocked.
