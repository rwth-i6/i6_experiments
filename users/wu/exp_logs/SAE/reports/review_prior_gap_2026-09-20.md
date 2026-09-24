# Review: `PriorGapAnalysisJob` (SAE 4a Step 0), commit 1431dbf — 2026-09-20

Read-only review. Verdict: **DONE_WITH_CONCERNS**. Orders 1, 2, 3, 4, 6, 8 can be trusted as they
stand. The `lexicon` row's *gap* and its `discriminates_more` verdict cannot close the decision
table's lexicon clause without a version-bumped follow-up read. The job (slurm 1916086,
`work/speech_llm/sae/emc/prior_gap/PriorGapAnalysisJob.l0p0srBryKrs`) had not started at review time;
letting it run is still worthwhile — nothing below is wasted compute, the follow-up is an addition.

## Findings

### F1 (material). The lexicon gap and the trigram gap it is compared against are on different utterance sets
`prior_gap.py:593` builds every row's gap with `paired_gap(...)`, and `prior_gap.py:353` drops the
utterances that are not finite on BOTH sides. For orders 1-8 nothing is dropped. For `lexicon`,
measured on the job's own inputs (first 150 sorted dev-other tags, real bliss+g2p lexicon):

| set | segmentable |
|---|---|
| gold | 150/150 |
| private (SIL-free decode) | 105/150 |
| private_sil | 105/150 |
| gold_null | 48/150 |
| private_null | 28/150 |
| private_sil_null | 23/150 |

So the lexicon gap is a paired mean over roughly 70 % of the utterances while `verdict()`
(`prior_gap.py:399`) subtracts the trigram gap taken over 100 % of them. The ~30 % dropped are
exactly the utterances where the private code fails to decompose into English words, i.e. where a
lexicalised prior separates hardest. The margin is therefore biased toward "the lexicon does not
discriminate more", and the bias is not bounded by anything the job prints.

Not repairable from the output: `assemble_record` banks summaries only, no per-utterance scores, so
the trigram gap restricted to the lexicon's paired subset cannot be recovered from `prior_gap.json`.
Fix (needs `version` bump): compute every row's gap a second time on the lexicon's paired subset, or
dump per-utterance `(log p, tokens)` per prior per set.

### F2 (material, interpretive). A lexicon word the word trigram never saw costs one `<unk>`, not -inf
`prior_gap.py:196-199` scores a segmentation word with `model.BaseScore`. Verified against a real
lmplz model in this environment: an OOV word does not fail, it gets the `<unk>` unigram
probability. The lexicon built by `load_phonemization_lexicon` is bliss (200,000 words) + g2p, i.e.
**584,893 words / 472,131 distinct pronunciations, longest pronunciation 56 phones**, while the word
trigram is fitted on 1 M lines — a large share of the lexicon is `<unk>`-priced. One `<unk>` covers a
whole word whatever its phone length, and `best_segmentation` explicitly maximises over
segmentations, so a non-English phone string can buy a cheap per-phone score by taking a few long
unseen words. This is the `lm_prior` length exploit in a new place: the price is per word, the
denominator is per phone. The spec's "an out-of-lexicon segment impossible" is implemented; "an
out-of-word-LM segment is cheap" is a silent additional choice. Direction of the error is the one
that matters: it shrinks the lexicon gap. `words_oov_to_word_lm` is banked and rendered
(`prior_gap.py:722`), so the size of the effect is readable off the report — read it before reading
the lexicon gap.

### F3 (minor, but read the json not the md). Banked and never rendered
`set_summary` banks `log_prob_per_utterance` (`prior_gap.py:339`) — a column the spec asks for
("mean log-probability per token **and per utterance**") — and `paired_gap` banks `n_paired` /
`frac_paired` (`prior_gap.py:373-374`). `report_lines` prints neither. The render test
(`test_prior_gap.py:235-256`) only requires `log_prob_per_token`, the gap, the IS sd, the notes and
the decision, so it passes. Consequence: `prior_gap.md`, the registered artefact, does not say how
many utterances the lexicon gap is a mean over — the exact number F1 requires. `prior_gap.json` has
it.

### F4 (minor). `verdict`/`decision` do not distinguish "no evidence" from "no discrimination"
With too few paired utterances the gap is `nan` (`prior_gap.py:358`), `nan > x` is False, so
`discriminates_more` reads `no` and `decision()` can print
`prior_family_negative_phase_closes` (`prior_gap.py:419`). At ~70 % segmentability this will not
fire on these inputs, so it is recorded, not charged.

### F5 (minor). The pinning keys carry no job identity
`config_sae_4a_prior_gap_v1.py:61-74` pins the window, the word corpus and the two lexicons with
constant `hash_overwrite` keys (`sae4a_prior_gap_window_phn`, …); `pc.prior_npz()` likewise.
Repointing any of them at a different producing job would leave the hash unmoved and the finished
job untouched. Mitigations, checked: the decode's keys DO carry arm/epoch/split
(`sae4a_private_code_ctrl_50_ep10_devother_raw`), `version` IS in the hash (confirmed in the job's
`info`), and `run()` writes `os.path.realpath` of every input into both outputs, so a swap is
detectable after the fact.

### F6 (not a defect — a falsified prediction, note it in the phase file)
The spec predicts "the null strings have no segmentation". They do: 32 % of `gold_null` and 19 % of
`private_null` segment. Cause is the lexicon itself — **single-phone words cover 34 of the 39
phones** (all but AE, DH, NG, P, V). "An out-of-lexicon segment impossible" is a weak constraint
here, and `frac_segmentable` carries less signal than the pre-registration assumed.

## Checked and sound

* **Anchors reproduce.** Pooling the two banked `SymbolDeciphermentJob.m8EsFhL6ysqu` halves
  (fit/held, 1432 + 1432) gives trigram per token gold **-3.199171**, identity/SIL-kept
  **-3.986539**, identity/SIL-free **-4.628743** — the implementer's -3.1992 / -3.9865 / -4.6287, and
  the dispatch's -3.20 / -3.98 / -4.62. `witten_bell_utt_log_prob` (`prior_gap.py:159`) IS
  `PhoneNgramPrior.log_prob`, and `set_summary`'s pooled total/tokens with empty strings
  contributing 0/0 is exactly `private_code.prior_score_per_token`'s convention, so the job's
  `order3` row must land on those three numbers. If it does not, stop the read.
* **BOS / no EOS / denominator.** Verified against a real lmplz model built in this environment:
  `BeginSentenceWrite` + summed `BaseScore` equals `model.score(bos=True, eos=False)` to float
  precision, and `</s>` is not scored. Witten-Bell pads with `BOS_ID` and has no EOS
  (`prior.py:323-327`). The word LM uses the same `<s>` start, no `</s>`. Every prior, including the
  lexicon (`prior_gap.py:964`), divides by the string's own phone count. `</s>` is in the fitted
  KenLM models and unscored, which leaks a small per-token constant off the KenLM rows' absolute
  level; it cancels to first order in a gold-minus-private difference of per-token means.
* **Window parity.** `SampleLinesJob.orN768ARKwlt`: `n_out=1010000`, `seed=0`, corpus
  `PhonemizeWithSilJob.DbFgvZOGZQ8F`. `PhoneNgramPriorJob.RtzbESkOedsT`: `n_count_lines=1000000`,
  `n_held_lines=10000` on exactly that file. `counted_line_flags` (`prior_gap.py:457`) reproduces
  `count_ngrams`'s split line for line (non-empty AND `i % 101 != 0`), so the KenLM phone models and
  the word trigram see the trigram's 1,000,000 counted lines and none of its holdout. This is the
  uniform-sample window, not the alphabetical head.
* **Word text.** `count_kept_lines`' filter (`prior_gap.py:483`) is byte-identical to
  `phonemize_line`'s drop rule (`w2vu2/text.py:44`), and the lexicon map is built the same way
  (bliss first pronunciation, g2p `setdefault`). A misalignment cannot pass silently:
  `replay_window_texts:534` asserts per line that the window's phone line minus SIL equals the
  concatenation of the recovered word line's pronunciations, on all 1,010,000 lines.
* **lmplz.** `-o <order> --interpolate_unigrams True -S 8G -T <lms> --discount_fallback 0.5 1.0 1.5`,
  no pruning — the campaign canonical (`config_sae_1g_h4_matched_lm_v1.py:35`,
  `phoneme_lm.phoneme_ngram_lm`). `run()` asserts all 40 phones are in each phone LM's vocabulary
  (`prior_gap.py:948`), so no phone is ever scored as `<unk>`.
* **Viterbi exactness and cost.** `kenlm.State` is value-comparable and value-hashable (checked on a
  real model: two paths reaching the same context give equal states and equal hashes), so keying the
  DP table on it merges states correctly and the "best score per (position, state)" argument holds.
  Measured on real gold strings with the real lexicon: ~8.2 k extensions/utt with last-word state
  granularity, ~104 k with last-two-word granularity, and ~2.9x that summed over the six sets, i.e.
  1e8-1e9 kenlm calls for the whole read. 4 h / 16 GB is not a DP timeout risk. `beam=None` (exact)
  is in the hash. SIL is consumed free only at a boundary and no pronunciation contains SIL, so a
  SIL inside a word is correctly unsegmentable. Homophones: all words sharing a pronunciation are
  tried (`Lexicon.by_pron`).
* **Strings.** 2864 utterances; `gold['dev-other']`, `greedy_raw.json` and `greedy_phones.json` have
  identical key sets; `drop_sil(raw) == hyps` holds on disk for every tag; no token outside the
  39+SIL inventory; no SIL in gold. One empty gold utterance, correctly excluded from every paired
  number and contributing 0/0 to the pooled per-token means.
* **Null.** Per-utterance permutation, one `default_rng(0)` per set consumed in sorted-tag order,
  length and unigram histogram preserved, applied the same way to gold and to both private sets.
* **IS bracket.** sd (ddof 1) over utterances of the per-utterance TOTAL `log p_order - log p_3`
  (`prior_gap.py:385`), gold and private both printed, `decision()` reads the larger — the spec's
  conservative reading.
* **Train/eval separation.** All seven priors are fitted on the LibriSpeech LM text corpus; the
  scored strings are dev-other decodes and MFA gold. The trigram's own 10,000 held-out lines are
  excluded from the KenLM fits as well.
* **One delta.** `git show 1431dbf`: three files added, 1509 insertions, 0 deletions, nothing else
  touched; working tree clean for all three (no uncommitted edit that a running worker could import).

## Reading instructions for the numbers when they land

1. Check `priors.order3.sets.{gold,private,private_sil}.log_prob_per_token` against
   -3.1992 / -4.6287 / -3.9865. A mismatch invalidates everything below it.
2. Read the `like_for_like` pairing. `sil_kept` normalises over a different token inventory and is
   disclosed as not comparable.
3. For orders 4, 6, 8 the gap, spread, verdict and IS sd are sound as printed.
4. For `lexicon`, read `pairings.*.priors.lexicon.n_paired` / `frac_paired` from the JSON first
   (they are not in the .md), and `words_oov_to_word_lm` from the .md. Do not read
   `discriminates_more.lexicon` as closing the decision table's lexicon clause. The 6/8-gram proxy
   rows are on the full set and CAN close it.
