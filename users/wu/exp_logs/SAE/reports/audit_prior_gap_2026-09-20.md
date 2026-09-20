# Audit: Step 0 prior-gap diagnostic (PriorGapAnalysisJob.2RkbKYl0v1XK)

Read-only audit, fresh context, 2026-09-20. Nothing was edited, rerun or resubmitted. One
auxiliary LM was built in session scratch only
(`/tmp/claude-34349/.../scratchpad/q2/phones_o3_kn.bin`), never in the job dir.

Artifacts read: `work/speech_llm/sae/emc/prior_gap/PriorGapAnalysisJob.2RkbKYl0v1XK/output/
{prior_gap.json,prior_gap.md,per_utt.json}`, the job's `work/lms/` (real path
`/e/scratch/spell/wu24/2026-07-13_unsupervised/work/.../work/lms/`),
`recipe/2025-10-speech-llm/src/speech_llm/sae/emc/prior_gap.py`,
`.../emc/prior.py`, and section "Step 0" of `SAE_4A_prior.md` (lines 65-114) only.

## 0. Reproduction of the banked numbers

Every aggregate in `prior_gap.json` was recomputed from `per_utt.json` with an independent
script (paired per-utterance per-token deltas, even/odd halves on the sorted usable-tag list).
All match to the printed digits:

| row (like_for_like) | banked gap | my gap | banked spread | my spread | my paired SE |
|---|---|---|---|---|---|
| order1 | 0.1556 | 0.1556 | 0.0023 | 0.0023 | 0.0030 |
| order2 | 0.5205 | 0.5205 | 0.0048 | 0.0048 | 0.0072 |
| order3 (WB, live) | 1.3928 | 1.3928 | 0.0172 | 0.0172 | 0.0120 |
| order4_kenlm | 1.6662 | 1.6662 | 0.0078 | 0.0078 | 0.0090 |
| order6_kenlm | 1.6225 | 1.6225 | 0.0138 | 0.0138 | 0.0083 |
| lexicon_strict (n=609) | 2.5057 | 2.5057 | 0.0595 | 0.0595 | 0.0322 |
| lexicon_escape | 2.3213 | 2.3213 | 0.0159 | 0.0159 | 0.0105 |

Subset trigram on the strict subset: 1.0857 (mine 1.0857, spread 0.0288, SE 0.0303, n 609).
frac segmentable (lexicon_strict): gold 0.8760, private 0.2367, private_sil 0.2357,
gold_null 0.0244, private_null 0.0108 — the amendment-(iii) prediction (nulls fall well below
gold) holds. n_paired 2863/2864: the single dropped utterance is `1651-136854-0012`, whose GOLD
string is empty (0 tokens); exclusion by `paired_tags` is correct.

Independent check of the KenLM side: I re-scored `gold.txt`/`hyp.txt` with the job's own
`work/lms/phones_o4.bin` through `tools/kenlm/build/bin/query` (BOS context, per-token BaseScore,
`</s>` term dropped, x ln10). Max |my score - banked `order4_kenlm` log_prob| over 5728
string-scores = 3.7e-6 nats. The banked KenLM column is reproducible and the BOS/no-EOS
convention in the docstring is the one actually executed.

## 1. Does the printed decision follow from the spec's table? (CONFIRMED)

Printed outcome, both pairings: `no_row_of_the_table_fires`.

Clause-by-clause against the pre-registered table (SAE_4A_prior.md:93-101):
- clause 1 "4-gram discriminates more AND its per-utterance log-weight sd below 3 nats":
  discriminates yes (margin 1.6662-1.3928 = 0.2734 > max spread 0.0172; paired SE 0.0078,
  t = 35), but sd_read = max(12.794 gold, 17.921 private) = 17.921 nats >> 3. FAILS on the sd.
- clause 2 "the lexicon or its 6-gram proxy discriminates more BUT THE 4-GRAM DOES NOT":
  the 4-gram does discriminate more, so the clause's own precondition is false. FAILS.
- clause 3 "neither discriminates more": false (order4, order6, lexicon_escape all fire the
  read rule). FAILS.

So the table as written is SILENT on the state (discriminates, but sd too large), and the job's
`no_row_of_the_table_fires` is the honest report of that. The code implements the printed rule
literally (`prior_gap.py:559-610`): the fall-through branch is explicit, not an accident.

Is there a reading that fires? One: read clause 2's "the 4-gram does not" as "the 4-gram ARM is
not taken" (i.e. clause 1 did not fire) rather than "the 4-gram does not discriminate more".
Under that reading clause 2 fires -> `lexicon_score_function_term`, and the evidence points the
same way (lexicon_escape is 0.9286 +/- 0.0128 nats/token above the trigram, paired t = 73, while
the 4-gram importance weight has sd 12.8-17.9 nats and is unusable). But that is a restatement
after the numbers: the text as written conditions on discrimination, not on the sd. Verdict:
the printed outcome is correct; re-opening the table is a planner decision, not a re-reading.

Second ambiguity, disclosed: "per-utterance standard deviation ... of log p_order - log p_3" is
computed on the utterance-TOTAL log weight (`log_weight_sd`, ddof 1). Read per token instead,
the same weight has sd 0.180 (gold) / 0.247 (private) and clause 1 would fire. The job's reading
is both the literal one and the operationally correct one (an importance weight is per sampled
string), so I do not treat this as a defect - but the 3-nat threshold was never tied to a
length, and at ~62 tokens/utterance no 4-gram/trigram pair could have met it.

## 2. Estimator mismatch vs order (CONFIRMED, and the mismatch works AGAINST the 4-gram)

Buildable and built: `work/lms/window.phn.txt` (1,000,000 counted lines, the same lines the WB
trigram is fitted on - `replay_window_texts` writes only the `counted` flags) plus the job's own
`tools/kenlm/build/bin`. I built a KenLM modified-KN TRIGRAM with the job's exact lmplz command
(`-o 3 --interpolate_unigrams True --discount_fallback 0.5 1.0 1.5`) and scored the same strings
the same way. Same window, same lines, same tokenisation, same BOS/no-EOS convention.

| trigram/4-gram, like_for_like | gap /token | spread | paired SE |
|---|---|---|---|
| WB trigram (live, banked) | 1.3928 | 0.0172 | 0.0120 |
| KN trigram (mine, same text) | 1.1893 | 0.0141 | 0.0087 |
| KN 4-gram (job's own bin) | 1.6662 | 0.0078 | 0.0090 |

- Estimator effect at fixed order 3: KN3 - WB3 = -0.2034 (paired SE 0.0046, t = -44).
  Witten-Bell discriminates MORE than modified KN at the same order.
- Order effect at fixed estimator: KN4 - KN3 = +0.4768 (paired SE 0.0056, t = 85).
- Banked mixed contrast: KN4 - WB3 = +0.2734.

So the banked 4-gram advantage is not an estimator artefact - it is an UNDERSTATEMENT of the
order effect by 0.203 nats/token. (The banked contrast is still the right one for the decision:
the live scorer inside training is the WB trigram, so "does the 4-gram discriminate more than
the prior we actually use" is the question the arm would answer.)

Log-weight sd, decomposed (per utterance, nats):

| set | sd(p4KN - p3WB), banked | sd(p4KN - p3KN), order only | sd(p3KN - p3WB), estimator only |
|---|---|---|---|
| gold | 12.79 | 12.69 | 1.54 |
| private | 17.92 | 17.58 | 12.08 |

For gold the banked sd is essentially all order. For the decode the estimator-only weight alone
has sd 12.08 nats, but the order-only weight is still 17.58 - so clause 1's sd test fails by a
factor of ~6 under a strictly same-estimator 4-gram as well. The mismatch changes no verdict.

## 3. Is the lexicon_escape gap a fixed-price artefact? (OVERTURNED as a worry: it is not)

Recoverable from per_utt, because the escape price is an exactly identifiable component. I
verified that the per-phone escape prices ARE the order-1 row: for every string,
|sum_t escape_phone_log_prob(t) - order1 log_prob| <= 1.7e-12. Hence the all-escape floor
(one span, no lexicon) is `order1 + n*log 0.5 (+ one <unk> transition)`, and:

- the PURE fixed-price scorer's gap = 0.1556 /token — identical to the unigram gap, because the
  log 0.5 per phone is a per-token constant that cancels exactly in a per-token paired gap
  (only the single <unk> transition does not cancel, and it moves the gap by <= 0.001: a
  17-nat per-string constant shifts it by -0.0008, since mean(1/n_gold - 1/n_priv) = -4.8e-5).
- the remaining 2.1658 +/- 0.0105 nats/token is the margin the Viterbi lexical path earns OVER
  that floor: gold +1.9941 /token, decode -0.1717 /token (the decode's best path is essentially
  the all-escape path, paying its one <unk>). The gap is lexical routing, not pricing.
- price sensitivity, bounded rigorously: changing the per-phone price from log 0.5 to log c'
  changes each side's per-token score by between 0 and log(c'/0.5), so the gap moves by at most
  log(c'/0.5) in absolute value. Making the escape FREE (c'=1) leaves the gap in
  [1.628, 3.015] — still above trigram 1.3928 + spread 0.0172 for any price in that range, and
  a cheaper escape helps the decode (which escapes) more than gold, so the true value sits near
  the lower end. I did not re-run the Viterbi under another price (no python `kenlm` in this
  session's interpreter, and the word LM is a 274 MB binary); the bound above needs no rerun.
- the strict row says it independently, with no escape machinery at all: on the 609 utterances
  where BOTH sides segment with real words only, gap 2.5057 (spread 0.0595, SE 0.0322) against
  the trigram recomputed on those same 609, 1.0857 (spread 0.0288) - a margin of 1.42, LARGER
  than the escape row's 0.93 on the full set, even though that subset selects the most
  English-like decodes (segmentable fractions 0.876 gold vs 0.237 decode).

## 4. 6-gram below 4-gram: real (CONFIRMED as a real effect)

Paired per-utterance difference of the two gaps: -0.0437, paired SE 0.0053, t = -8.2 (half-split
spread of the difference 0.0060, n 2863). It is not within the spread; a 6-gram phone LM
discriminates gold from the decode LESS than a 4-gram does. Both fit both sides better in
absolute terms (gold -2.786 -> -2.574 /token, decode -4.504 -> -4.261 /token): the extra context
buys the DECODE as much as it buys gold. Against that, lexicon_escape - order4 = +0.6551
(SE 0.0096, t = 68). The spec's premise that the 6/8-gram is "the table-lookup proxy for a
lexicalised prior" is contradicted by its own numbers: the phone n-gram saturates by order 4 and
then regresses, while the actual lexicon is 0.66 nats/token above it. Whatever a lexical prior
contributes, extending phone context does not reach it.

## 5. Defects in pairing, denominators, nulls, BOS/no-EOS

Nothing that changes a verdict. Found and bounded:
- (a) `is_sd` is computed over all 2864 utterances while the gaps use 2863; the extra item is
  the empty gold string, whose weight is 0-0 = 0 against a mean of 25.6, so it slightly inflates
  the sd. Effect ~1/2864; clause 1 fails by a factor of 6 regardless.
- (b) `is_sd["lexicon_strict"]` is taken over each set's OWN finite utterances (gold n 2509,
  private n 678) - not a paired set, so those two sds are not comparable to each other. That row
  is disclosed only and never enters the table, so no verdict depends on it.
- (c) "spread" = |gap_even - gap_odd| is a single half-split realisation, not an SE; here it
  runs 0.5x-1.4x the paired SE (e.g. order4 0.0078 vs SE 0.0090), so the read rule's bar is of
  the right order but is not a stated confidence level. Every fired margin clears many SEs
  (order4 35 SE, escape 73 SE), so this is a convention note, not a defect.
- (d) The LM training text (`window.phn.txt`) carries SIL (every line begins and ends with SIL),
  while both scored strings in like_for_like are SIL-free. The BOS bigram context is therefore
  near-unseen and the first token backs off to unigram (-17 nats on the first gold token in my
  query run). This is a per-string constant on both sides; by the 1/n asymmetry above it moves
  the gap by <= 0.001 /token. The same mismatch also means all six rows score slightly
  out-of-distribution strings - symmetric between gold and decode.
- (e) The per-token denominator is each string's own token count (gold 61.90, decode 58.49 mean
  tokens), as documented in `units`. Paired per-utterance deltas, so no pooling artefact; the
  pooled column is banked separately and differs by ~0.05 (e.g. order4 1.7178 vs 1.6662).
- (f) Nulls: within-utterance permutation, seed 0, keeps length and unigram exactly (order1
  gold and gold_null rows are bit-identical). They are disclosed, never fed to the read rule.
  Informative side-read: under lexicon_escape the shuffled GOLD (-4.4791 /token) scores about
  the same as the real decode (-4.4613), i.e. the lexicon treats the decode as order-destroyed
  text.
- (g) Provenance: `name` = ctrl_50_ep10/dev-other, decode from
  `BlankfreeGreedyPerJob.dG4n46xTRSl0` (ep10 appears in its `info`); the job asserts
  greedy_phones == greedy_raw minus SIL. I did not verify the checkpoint identity beyond this.
- (h) sil_kept is correctly labelled NOT like-for-like and is not the row the table reads; its
  strict subset trigram base is -0.1839 with spread 0.2100 (noise-dominated), another reason it
  is disclosure only.

## Verdict

1 CONFIRMED (table silent, `no_row_of_the_table_fires` is correct as written; one alternative
reading of clause 2 would fire the lexicon arm, and the evidence supports that arm).
2 CONFIRMED (estimator mismatch understates the 4-gram by 0.203; same-estimator order effect
+0.477; order-only log-weight sd 12.7/17.6 nats, still >> 3).
3 CONFIRMED (fixed prices contribute 0.156 of 2.321 and cancel per token; lexical routing
contributes 2.166; any price between free and log 0.5 keeps the gap >= 1.63).
4 CONFIRMED real (-0.0437, SE 0.0053, t = -8.2); the 6-gram-as-lexicon-proxy premise fails.
5 No verdict-changing defect.
