# What the S3b recognizers actually emit -- sub-epoch 4, dev-other (2026-09-15)

Status: **DONE**. CPU-only read on the login node, no training, no launch, no manager touched
(the packed run under pid 2499469 was not polled). Nothing committed: `analysis/` is not a repo,
and this report is left uncommitted in the `i6_experiments` checkout as the brief asks.

Script: `/e/project1/spell/wu24/2026-07-13_unsupervised/analysis/emc_hyp_inspect.py`
Outputs: `/e/project1/spell/wu24/2026-07-13_unsupervised/analysis/out/emc_hyp_inspect.<arm>.ep4.dev-other.{txt,json}`
for `<arm>` in `lam1 lam10 spec spec_speed lam3` (txt 125-131 lines each, cap 200).
Run: `/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python analysis/emc_hyp_inspect.py`
from the setup dir, 49 s wall for all five arms, exit 0.

## Inputs and their verification

The **hypotheses were already stored** -- nothing had to be recomputed from the posterior dump.
Each `GreedyPerJob` writes `output/greedy_phones.json` beside the `per.json` the alias points at
(`eval_jobs.py:567`, `:658`): per utterance the collapsed greedy string (per-frame argmax over the
41 outputs, repeats collapsed, blank dropped, SIL dropped). The script resolves them through the
setup's `output/` aliases, never through a typed hash:

| arm | alias | GreedyPerJob | banked PER |
| --- | --- | --- | --- |
| lam1 | `sae_4a_s3b_pack/lam1/ep4/dev-other` | `hOV7DngnCkoS` | 0.829429 |
| lam10 | `sae_4a_s3b_pack/lam10/ep4/dev-other` | `IqibBGbnVs4L` | 0.913829 |
| spec | `sae_4a_s3b_pack/spec/ep4/dev-other` | `D2ZZOxZMAmZD` | 0.848405 |
| spec_speed | `sae_4a_s3b_pack/spec_speed/ep4/dev-other` | `rJPYYRNO9DKn` | 0.848552 |
| lam3 | `sae_4a_s3b_rate/lam3/ep4/dev-other` | `02WAVwvzkVTi` | 0.846651 |

Reference: `GoldPhonesJob.ZGSp0hxyd2YP/output/gold.json`, split `dev-other`, 2864 utterances,
177275 phones, 39 ARPAbet symbols, silence-free (1 utterance has empty gold and is excluded from
the ratio and the correlation). Every arm's sisyphus `info` is parsed and its `PARAMETER: gold`
asserted to be exactly that file, and `PARAMETER: split` to be `dev-other`.

Checks that ran on every arm (assertions, so a failure would have stopped the run):

* per utterance, the (S, D, I) of the local backtrace equals `eval_jobs.edit_counts(hyp, ref)`
  called with the real objects -- the shared primitive, not a fixture (2864 x 5 calls);
* per arm, the recomputed S / D / I / N integers equal the banked `per.json` exactly and the PER to
  1e-9. All five reproduce (e.g. lam1 S 82554 / D 62978 / I 1505 / N 177275).

The alignment DP is `analysis/per_error_pattern.py:align` (the campaign's convention: unit costs,
ties substitution before deletion before insertion), imported rather than re-typed.

## Results

| arm | PER | mapped PER (best bijection) | identity pair acc | best-bijection pair acc | fixed pts | H(hyp) bits | run>=3 share | max run | r(len) | corpus hyp/gold | emitted/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lam1 | 0.8294 | 0.8274 | 0.2777 | 0.2780 | 37/39 | 4.470 | 0.0026 | 4 | 0.9390 | 0.653 | 6.29 |
| lam3 | 0.8467 | 0.8458 | 0.2507 | 0.2509 | 34/39 | 4.330 | 0.0023 | 5 | 0.9132 | 0.722 | 6.95 |
| spec | 0.8484 | 0.8484 | 0.2248 | 0.2248 | 23/39 | 3.193 | 0.0019 | 4 | 0.9648 | 0.867 | 8.35 |
| spec_speed | 0.8486 | 0.8481 | 0.2240 | 0.2241 | 24/39 | 3.561 | 0.0021 | 4 | 0.9703 | 0.876 | 8.44 |
| lam10 | 0.9138 | 0.9104 | 0.2244 | 0.2249 | 31/39 | 4.517 | 0.0007 | 4 | 0.8912 | 0.942 | 9.08 |

Gold, the same for every arm: entropy 4.833 bits over 39/39 phones, mean run 1.0056, max run 2,
share of tokens in a run >= 3 exactly 0, mean length 61.90 phones, 9.635 phones/s.

**(5) A relabelling recovers nothing.** The best one-to-one map hypothesis phone -> gold phone,
fitted by Hungarian assignment on the 39x39 aligned-pair counts (the full confusion, correct pairs
on the diagonal plus substitutions off it -- excluding the diagonal would force the assignment off
identity and make the accuracy meaningless), raises the aligned-pair accuracy by at most 0.0005
absolute over the identity map and moves the PER by at most 0.0034 (lam10; 0.0000 for spec). The
arms are therefore **not a permuted private code**: the errors are not a consistent relabelling.
This is fitted ON the gold and is a diagnostic only -- it can never be a decode or a selection.
It is also a lower bound in one direction: the pair counts come from the identity alignment, so a
jointly refitted alignment-plus-permutation could score slightly higher.

**(2) Unigram mass.** Two regimes. `lam1 / lam3 / lam10` keep a broad inventory (entropy 4.33-4.52
bits vs gold 4.83, 37-39 of 39 phones used) with a plausible-looking top of the list (N, AH, D, S,
IH for lam1 against AH, N, T, IH, D for gold). `spec` and `spec_speed` are strongly collapsed:
`spec` puts 57 % of all emitted tokens on K / IH / T (0.2025 / 0.2001 / 0.1688) at entropy 3.193,
`spec_speed` 59 % on AH / DH / AE / N at 3.561. `lam10` concentrates differently: AH alone is
0.1568 of its tokens against 0.0953 in gold.

**(3) Confusion.** No arm shows a systematic one-to-one mapping. lam1's ten most frequent gold
phones keep 5.9-36.2 % correct with the top substitution taking only 8-14 % of that phone's
substitutions, i.e. the substitution mass is spread. Under `spec` the same table is the collapse
seen head-on: K / IH / T are the top-3 substitution target of *every* one of the ten most frequent
gold phones, and gold IY and EH are recovered 0.00 % of the time. The most-inserted phones follow
the same unigram peak (spec: K 1244, IH 1190, T 922).

**(4) No repetition pathology.** A hypothesis run of the same phone can only survive the greedy
collapse across a dropped blank or SIL frame, so it is a real repetition -- and there is almost
none: mean run 1.007-1.022, max run 4 (5 for lam3), and 0.07-0.26 % of tokens inside a run >= 3
(gold: mean 1.0056, max 2, 0 %). "A few phones repeated" is true at the level of unigram mass, not
at the level of adjacent repeats.

**(6) Length tracks well.** Pearson r of hypothesis length against gold length per utterance is
0.89 (lam10) to 0.97 (spec_speed); every arm is short of gold in total (corpus token ratio 0.65
lam1, 0.72 lam3, 0.87 spec, 0.88 spec_speed, 0.94 lam10), and the emitted rate ranks the arms the
same way (6.29 to 9.08 phones/s against 9.635). lam10, the closest in rate, is the worst in PER and
carries by far the most insertions (18137 vs 1505 for lam1).

**(1) What it looks like.** Two of lam1's printed alignments (`=` correct, S substitution,
D deletion of a gold phone, I inserted hypothesis phone):

```
  [shortest] 4515-11057-0053  gold 3 phones, hyp 4 phones, S 3 D 0 I 1, PER 1.333
    gold -   Y   EH  S
    hyp  R   AW  ER  TH
    op   I   S   S   S

  [median] 700-122866-0014  gold 50 phones, hyp 40 phones, S 33 D 10 I 0, PER 0.860
    gold IH  F   AY  HH  AE  D   AE  L   AH  S   B   EH  L   Z   K   R   UH  K   AH  D   N   OW
    hyp  S   OW  B   AH  T   S   AE  -   T   DH  N   DH  L   -   -   HH  UH  D   HH  AE  N   -
    op   S   S   S   S   S   S   =   D   S   S   S   S   =   D   D   S   =   S   S   S   =   D
```

The same 50-phone utterance under `spec` (47 emitted phones, S 36 D 5 I 2) shows the collapse:
`AH S K T IH K T S IH K S T D IH K R D IH NG K AH IH ...` -- length-tracking, phone-like, built out
of a handful of types.

Read across the five: the output is **phone-like noise that tracks utterance length**, with a
broad-but-skewed inventory in the lambda arms and a 3-4 symbol collapse in the SpecAugment arms;
it is not a repeated single phone, not a permuted code, and not fragments of correct phones
(correct positions do occur, but the arms' correct rate is not compared against any null here).

## Conventions the brief left open (stated in every txt and json)

* the eight utterances: tags with non-empty gold sorted by `(gold length, tag)`, the first four and
  the four at sorted positions `n // 2 - 2 .. n // 2 + 1`. Gold-only, so identical in every arm:
  shortest `1686-142278-0068`, `1686-142278-0041`, `4515-11057-0053`, `1255-138279-0008`;
  median `6841-88291-0033`, `700-122866-0014`, `700-122866-0027`, `700-122867-0022`;
* entropy is in bits (log base 2) of the unigram distribution over the 39 phones;
* the Hungarian objective is the full aligned-pair matrix (correct pairs included), as argued above;
* length ratio is reported twice -- mean of the per-utterance ratios and the corpus token ratio.

## Not measured (not requested)

No chance level for the identity pair accuracy (e.g. the accuracy a unigram-matched random
hypothesis would reach), so the 0.22-0.28 correct-pair share is a description, not a claim that the
arms are above or at chance. No dev-clean read, no other sub-epoch, no cross-arm paired test.

## Addendum (same day, orchestrator addition): a chance null for the PER

The "not measured" item above is now measured. `analysis/emc_hyp_inspect.py` gained section (7) and
all five arms were rerun (`--null-draws 5 --null-seed 0`, txt now 134-140 lines, still under the
200-line cap; every per-utterance and per-corpus assertion against the banked `per.json` passed
again). Per utterance a null string of the SAME length as that arm's hypothesis is drawn i.i.d.,
(a) uniformly over the 39 phones and (b) from that arm's own hypothesis unigram distribution, and
scored with the same `edit_counts` convention. Null b additionally gets the section (5) treatment
(its own 39x39 aligned-pair matrix, its own Hungarian bijection fitted on the gold, PER after
relabelling) and is drawn once more at the GOLD lengths. Five draws from seed 0, averaged; the
draw-to-draw sd is 0.0002-0.0006 PER everywhere, so the means below are stable to the fourth digit.
The null has no sequential structure whatever -- it is the floor a length-matched and
unigram-matched string reaches.

| arm | null a (uniform) | null b (arm unigrams) | arm PER | null b after its own bijection | null b at GOLD length |
| --- | --- | --- | --- | --- | --- |
| lam1 | 0.8876 | 0.8395 | 0.8294 | 0.8395 | 0.9000 |
| lam3 | 0.9012 | 0.8497 | 0.8467 | 0.8497 | 0.8972 |
| spec | 0.9176 | 0.8585 | 0.8484 | 0.8585 | 0.8884 |
| spec_speed | 0.9199 | 0.8673 | 0.8486 | 0.8671 | 0.8946 |
| lam10 | 0.9728 | 0.9216 | 0.9138 | 0.9204 | 0.8993 |

What it says:

* **The arms beat their own chance floor by 0.3 to 1.9 PER points.** lam1 +0.0101, lam3 +0.0031,
  spec +0.0101, spec_speed +0.0188, lam10 +0.0078 (arm PER below null b). Each margin is many
  draw-sd wide, so it is not draw noise -- but in absolute terms the sequential content of these
  recognizers is worth about one PER point over an i.i.d. string with the same length and the same
  unigram mass. The 0.83-0.91 PER is essentially the chance level of that length and that
  inventory.
* **The permutation gain seen in section (5) is a chance-level gain.** Hungarian relabelling lifts
  the null by at most 0.0012 (lam10; 0.0000-0.0003 elsewhere), the same scale as the 0.0000-0.0034
  it lifts the arms. That the arms gain nothing from relabelling is therefore not informative on its
  own -- but neither does a chance string, so the earlier conclusion stands: no permuted code.
* **Most of the unigram effect is the skew, not the content.** Going from uniform (null a) to the
  arm's own unigram distribution (null b) is worth 0.048 (lam1) to 0.059 (spec) PER, i.e. five to
  six times the arm's own margin over null b. The heavily collapsed arms gain the most from it.
* **lam10 is worse than chance at the gold length.** Its null b drawn at the GOLD lengths scores
  0.8993, better than lam10's own 0.9138; the arm's near-gold token count is bought with 18137
  insertions that cost more than its content returns. Every other arm is short of gold, and for
  them the gold-length null is worse than their own-length null (0.888-0.900), which is why the
  own-length null is the right comparison for them.

Nothing here changes an experimental decision by itself and no gate is defined on it: it is the
missing scale for the PER numbers already banked.
