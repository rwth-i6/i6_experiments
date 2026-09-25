# SAE 4A step 1 audit (fresh context, 2026-09-19)

Artifacts: `output/sae/4a/attrib/{ngram_mode_seeking.json,summary.md}` ->
`work/i6_experiments/users/wu/experiments/unsupervised_asr/ngram_mode_seeking/NgramModeSeekingJob.o2V5IRjMDy3K/output/`.
Criterion read: `SAE_4A_attrib.md` "Step 1" + the Step 1 bullets of "Design-review amendments"
(Results/State not read). Independent script: scratchpad `audit_step1.py` (own corpus pass, own
Witten-Bell fit, own JSD/bootstrap code), raw log `audit_out.txt`. Verdict: CONFIRMED_WITH_CAVEATS.

## (1) Do the four cells follow from the json under the stated margins? YES
(a) ep4-gan mean SIL-free trigram log-prob/phone = -0.3338, margin >= -0.10 -> FAIL.
(b) ep4-gold JSD4 = +0.4386, CI95 [0.4073,0.4169] excludes 0, margin >= +0.05 -> PASS.
(c) ep4-gan JSD4 = +0.4114, CI95 [0.3803,0.3897] excludes 0 -> PASS.
(d) gan-gold JSD4 = +0.0272, |.| <= 0.10 -> PASS.
The amendment's decisive read is the conjunction (i)&(ii)&(iii). (i) = (a) fails, so the
pre-registered mode-seeking signature is NOT confirmed by this read. summary.md renders cells only
and makes no composite claim; any downstream text must say "not supported, (a) failed".

## (2) Independent re-derivation (own code, same documented conventions)
Corpus pass reproduces the text side exactly: 1,000,000 lines / 78,778,657 SIL-free phones /
12,321,629 SIL removed / 413,706 distinct 4-grams.
Unmatched mean log P3 per phone (job -> mine): ep1 -6.202735 -> -6.202735; ep4 -3.337709 ->
-3.337709; gan -3.019196 -> -3.019196; gold -2.890948 -> -2.890948.
Unigram JSD (unmatched, job -> mine): ep1 0.216192 -> 0.216192; ep4 0.063554 -> 0.063554;
gan 0.002581 -> 0.002581; gold 0.003085 -> 0.003085.
Count-matched primaries, all four comparison points and all 1000-resample CIs and biases also
reproduce to 6 decimals (e.g. (b) 0.438581, CI [0.407273,0.416894]; (c) 0.411360; (d) 0.027221;
(a) -0.333792; matched counts 120170/2015, 120176/1909, 120120/1919). No arithmetic defect found.

## (3) GAN row provenance: verified
`GanPseudoLabelJob.P1lPMBEZUAiK` ran `eval_per.py --dump-labels` on split `valid` with
`FairseqW2vu2TrainJob.HOb2GgtYT7Bc/output/train/checkpoint_best.pt`. That train job carries
`common.seed: 0`. `cmp` shows checkpoint_best.pt byte-IDENTICAL to `checkpoint_823_148000.pt`
(same 21:09:18 mtime), so "seed 0, update 148000" holds. labels.json has 5567 ids = dev-clean 2703
+ dev-other 2864; the reader restricts to the 2864 sorted gold dev-other ids (ids_dropped 2703,
asserted equality) - I re-ran that restriction and got the same 178,180 phones. Decode =
argmax + consecutive-equal collapse + `<SIL>` dropped (`eval_per._decode_utt`), i.e. SIL-stripped
by construction, so the SIL-inclusive secondary is legitimately n/a.
Plausibility vs gold: 62.21 phones/utt vs gold 61.90 -> ratio 1.0051 (ep4 59.02, ratio 0.953).
UNVERIFIED: the "dev-other PER 0.214" label on this checkpoint and the claim that fairseq's
best_checkpoint_metric here is weighted_lm_ppl - the cited `W2vu2PerCurveJob.UDg7ulchmeE9` is not
on disk; the surviving `percurve_early_s0.json` is an earlier curve (best 97000, PER 0.223).
The identity of the decoded checkpoint is nevertheless established by the byte compare.
Blankfree rows trace to `BoundedBlankfreeTrainingJob.5lBwcDjv2ItL` epoch.001 / epoch.004,
dev-other, PER 0.8346 / 0.8649, N=177275 = the gold phone count.

## (4) Count-matching budget
120,187 phones, set by `blankfree_ep1` - a row that appears in NO decisive comparison. It truncates
ep4/gan/gold to 2015/1909/1919 of 2864 utterances (~66-70%). The rule is pre-declared in the
producing docstring (code review 2026-09-19), not in the gate text or the amendments. Consequence
is nil for the verdicts: the unmatched read gives (a) -0.3185 FAIL, (b) 0.4622 PASS, (c) 0.4350
PASS, (d) 0.0272 PASS.

## (5) CI conventions: documented behaviour, not a defect
Row JSD cells print `ci95_basic` = 2*theta - [q97.5,q2.5]. Where the bootstrap bias exceeds the
resample spread (gan JSD2 matched: bias +0.00119, spread 0.0020) the whole basic interval sits
below the point estimate - e.g. 0.0241 [0.0219,0.0239]. I reproduced the resample distributions
exactly, so this is the documented reverse-percentile interval behaving as expected under the
support-shrinkage bias, not a coding error. The comparison CIs are percentile intervals of the
per-resample paired differences exactly as documented (`_stat` on the difference samples).
No verdict flips under the alternative convention: (b) basic [0.4603,0.4699], (c) [0.4330,0.4425],
(d) [0.0245,0.0303]; (a) carries no CI clause. Presentation caveats: (i) the row cells read like
typos; (ii) these intervals are dominated by support-coverage bias, not by sampling noise, and the
text side is held fixed, so they are not general uncertainty intervals.

## (6) SIL and repeat handling
SIL: uniform in the primary read - every row had 0 SIL tokens at read time (blankfree
greedy_phones already SIL-free, GAN dumped SIL-free, gold carries no SIL); the text side is
SIL-stripped by the job. So (a) cannot be driven by SIL asymmetry.
Repeats: NOT uniform. Adjacent-equal phone pairs per token: ep4 0.00031, gan 0.00362, gold 0.00558,
text corpus 0.00585. Collapse (plus SIL removal) removes repeats the text side has. Quantified on
gold: collapsing gold's repeats moves its matched JSD4 0.274865 -> 0.278577 (+0.0037) and its
log-prob -2.890948 -> -2.888057 (+0.0029 nats). Both are ~2 orders below the (b) gap (0.4386) and
the (a) shortfall (0.234 beyond the margin), so the repeat asymmetry cannot explain either cell.

## Frame caveat (not a defect, but it bounds the reading)
The text side is the first 1,010,000 lines (every 101st held out) of a 39,630,169-line corpus that
is alphabetically sorted: the window contains only sentences starting with "a ...". It is exactly
the training prior's window (SIL-inclusive total 91,100,286 = PhoneNgramPriorJob.TRPE0D5nF3bh's
`tokens_counted`), which is the right reference for "prior-shaped", and all four rows face the same
text side, so the between-row differences are comparable. But the absolute JSDs (gold 0.2749
included) are not corpus-level values; the 0.27 reference line is already declared descriptive.
