# Audit: combined A15-F / A16 (a) / A16 (a2) reading (SAE_4A_lexlat_v2.md, lines 405-482), 2026-09-24

VERDICT: CONFIRMED_WITH_CORRECTIONS (DONE_WITH_CONCERNS).
- Every number I re-derived matches the artifacts.
- Claims 1 (A15-F, above manner level) and 2 (the r100 fit comparison is sharpness-confounded) hold.
  Within-class accuracy survives a selection-effect check that composition-matches each phi's class-correct frames.
- Claim 3's numbers hold, but its conclusion does not. "S's minimum is not at the right labelling (model
  error)" is not supported by these reads:
  - dS is by construction a prior-term-only statistic of a naming the phi co-adapted to.
  - The gold-vs-EM gap compares phis fitted on different data under different criteria, and it sits in the
    channel term, not the prior term.
  - The registered test of exactly this question, A14 (ii), is running and has not been read.
- Claim 4: only the lambda re-weighting of the phone trigram is ruled out as a fix, and a permutation-only
  label search is unsupported. The reads cannot yet tell search moves apart from an objective change.

Audit-side computations: scratch scripts in the session scratchpad (`wc.py`, `a2.py`), run on existing outputs
only. They are not banked. If any of the derived numbers below is to be cited in the phase file, it needs a
registered reader, per the campaign rule on derived statistics.

## Claim 1: A15-F within-class accuracy and ABOVE MANNER LEVEL: CONFIRMED

Source: `work/speech_llm/sae/emc/phi_content_controls/PhiContentReadJob.vySKYIh5RAaB/output/table.{txt,json}`.
I also recomputed from the per-phi `m_phi.npz`, the per-phi `content.json` emission maps, the MFA labels
(`MfaFrameLabelsJob.22i2qtnJUOWm`), the units HDF and the bed prior unigram.

- **Reproduction.** My recomputed within-class accuracy (R4 emis / class share) equals the table to 1e-9 for all
  15 phis and all 30 nulls (asserted).
- **Class share.** The six EM phis read 0.496-0.563; gold reads 0.728.
- **Within-class accuracy.** The EM phis read 0.538-0.623.
- **Amended threshold.** The rule takes the maximum of the 7-class oracle (0.3744), the random-partition maximum
  (0.4288) and the phi's own null maximum (0.346-0.394 per phi). The binding reference is the random maximum,
  0.4288.
  - The smallest margin is durfrz_s02: 0.538 against 0.429.
  - Against its own null maximum, the smallest margin is 0.16.
- **Sharp nulls.** Every column beats its sharp-null maximum for all 6 EM phis:
  - T3 primary: 1.99-2.20 against 0.21-0.26;
  - T3 relabel: 1.23-1.70 against -0.09 to +0.18;
  - R4 emis: 0.290-0.312 against 0.099-0.118;
  - relabelled fit: -6.0 to -6.7 against -10.0 to -10.6.
- **Selection-effect check.** Within-class accuracy conditions on class-correct frames, and chance differs by class
  (SIL 1 phone, glides and liquids 2, vowels 15). I computed each phi's own composition-matched
  "no within-class information" chance: per manner class, over that phi's class-correct frames, sum_k
  p(true = k) p(pred = k), weighted by that phi's class-correct composition.
  | | within-class | own chance | non-SIL within | non-SIL chance |
  |---|---|---|---|---|
  | EM x6 | 0.54-0.62 | 0.27-0.30 | 0.50-0.58 | 0.19-0.22 |
  | their 30 nulls | 0.29-0.39 | 0.16-0.22 | 0.27-0.37 | 0.14-0.19 |
  | gold | 0.81 | 0.28 | 0.79 | 0.22 |
  | r70 | 0.70 | 0.34 | 0.66 | 0.26 |
  - The EM phis' class-correct composition matches gold's: V 0.33-0.39 against 0.39, ST 0.11-0.17 against 0.15,
    FR 0.21-0.25 against 0.20, NA 0.08-0.11 against 0.10, SIL 0.07-0.11 against 0.08. So the result is not
    enriched in small classes.
  - The phis beat chance within every multi-phone class:
    - vowels 0.34-0.46 against 0.08-0.10;
    - stops 0.36-0.59 against 0.18-0.27;
    - fricatives 0.51-0.70 against 0.17-0.22;
    - nasals 0.66-0.76 against 0.39-0.50.
  - The best single phone per class, given each phi's own composition, reaches only 0.37-0.39 (non-SIL 0.30-0.33).
  - The label-informed emission map inflates even a destroyed phi by about 0.15 over its chance. The EM phis
    exceed their own chance by 0.26-0.34, so the net content beyond the map's selection is about 0.1-0.2.
  - Conclusion: the within-class result is not a selection artefact.
- **Class-assignment rule.**
  - The code (`PartitionBoundsJob.compute`) assigns each unit the class of its majority MFA phone. That is the
    amendment's PRIMARY rule. The literal "majority MFA class" gives within-class 0.3574, printed as VARIANT, and
    changes nothing, because the random maximum binds.
  - The references use the label-oracle Hungarian map. The phi's map comes from JS against the gold phi, not
    from the MFA labels. So the comparison is conservative for the EM phis.
- **Procedural note.** The amendment was made "before any A15-F job", but after the build check had measured one
  EM phi (durinit_s01, within-class 0.574) and its nulls (0.34-0.40) (`reports/impl_a15f_content_2026-09-24.md`).
  It made the bar stricter, so it does not favour the result.
- **Standing interpretation.** "Close to and somewhat below r70" is fair: R4 emis 0.29-0.31 against 0.342, and
  excess over own chance 0.26-0.34 against 0.36.

## Claim 2: the r100 comparison in the relabelled fit is sharpness-confounded: CONFIRMED

- Mean entropy of the m_phi rows, in nats (log 500 = 6.21):
  - EM 3.42-3.72; gold 3.38; phi_c 3.53;
  - r70 5.49; r100 5.67.
- So the EM phis are as sharp as gold, and r100 is nearly flat.
- A flat phi's penalty for a wrong string is capped near its entropy (r100 -5.26). A sharp phi without content
  pays about -10 (the sharpness-matched nulls, -10.0 to -10.6).
- The EM phis' -6.0 to -6.7 sit 3.6-4.0 nats per frame above their sharp nulls.
- So "worse than r100" did not measure missing content.
- The fit still cannot rank EM against r70 across sharpness. The phase file's "Still standing: below r70 on R4"
  rests correctly on R4, not on the fit.

## Claim 3: "S prefers own labels over the emission map, and EM phis over gold at lambda 1-3, so S's minimum is not at the right labelling (model error)": numbers CONFIRMED, inference NOT SUPPORTED

Numbers were re-derived from `phi_prior_scale_s/PriorScaleSReadJob.ttn9rEAZ1VQd/output/per_utterance.tsv`
(260 utterances, 159 speakers) and `phi_relabel_s/RelabelSReadJob.Y9RU2PSbVfuu/output/table.txt`.
- **Reproduction.** Every figure in the read reproduces:
  - S(gold) at lambda 1/2/3: 3.4735 / 4.0545 / 4.6015.
  - S(gold) - S(EM): +0.074 to +0.175 / +0.071 to +0.158 / +0.091 to +0.202. EM beats gold on 195-246 of 260
    utterances at lambda 1.
  - dS emis: +0.123 to +0.308 / +0.148 to +0.409 / +0.153 to +0.442.
  - permphi control: -0.598 / -0.794 / -0.847.
  - The ladder is strictly monotone at every lambda.
  - Frame-weighted pooling in place of the utterance mean keeps every sign: gap +0.06 to +0.16, dS +0.13 to
    +0.31 at lambda 1.
- **Consistency of the runs.**
  - The identity, emis and gold genmarg jobs have identical settings: null recognizer, trigram RtzbESkOedsT,
    blankfree band 25, d_min 2, d_max 25/50, float64.
  - The A15-E maps that A16 (a) used equal A15-F's maps for all six phis.
  - The relabelling permutes only `dur_logits` and `emb_type`, fixes SIL, and is asserted exact.
  - `prior_weight` multiplies log P_psi on every arc (`lattice._prior_term`), so this is P_LM^lambda as registered.

### The alternatives, checked

1. **dS is a prior-term-only statistic, and positive dS is what co-adaptation predicts.**
   - The relabelling carries each symbol's emission and duration rows with it, and every non-SIL type shares
     one topology. So p_phi'(x | y) = p_phi(x | g(y)), and S(emis) differs from S(identity) only through
     P_LM(g^-1(y)) against P_LM(y).
   - At lambda = 0, dS = 0 identically.
   - So dS measures one thing only: how well the trigram fits the phi's posterior symbol sequences under two
     namings.
   - EM shaped each phi's acoustic assignment to make its sequences trigram-typical under its own names. Any
     naming chosen by a criterion other than S (here, JS against the gold phi) is therefore expected to cost S.
   - "Label-blind" is contradicted by permphi (-0.60). "The label signal must come from the prior term" is
     vacuous, because the prior is the only label-sensitive term in S.
   - The growth of dS with lambda is also mechanical (it is 0 at lambda 0 and grows with lambda). It is not
     independent evidence.
2. **The one-to-one map over merged units is a biased stand-in for "the right labelling".**
   - The map sends only 20-28 of 40 symbols to their nearest gold phone. The other 12-20 are forced onto phones
     they do not resemble. For example, in uniform_s01, symbols Y and ZH are nearest to SIL and are mapped to IH
     and L.
   - Forced placements put symbols into sequence roles they cannot fill.
   - Contrast T3 relabel. The phi's channel explains the true transcript far better under the emission map
     (+1.2 to +1.7 nats per frame), yet S prefers the own labels.
   - At these phis the posterior sits far from the gold string: the gold string's joint cost is about
     0.6 + 6.0-6.7 nats per frame, against S of 3.3-3.6. So S's preference reflects the bulk of wrong strings.
     It says nothing about S at the right solution.
   - No negative control was run. r70 is right-labelled and merged, and its emission map moves 6 of 40 symbols
     (34 fixed points; T3 relabel -0.044). Under the same logic it would presumably read dS > 0.
3. **Data amount and training criterion (the gold phi is not "the gold model" of Yin's model-error test).**
   - Gold `16v7R6ztSq1u` was fitted by supervised conditional likelihood on 2821 utterances (epoch 8,
     step 2824).
   - The EM phis maximised the marginal S itself on the train-clean-100 stream: 48 sub-epochs = 12 passes, about
     10x the distinct utterances, step 2736.
   - In Yin et al. (`reports/lit_decipherment_relabel_2026-09-24.md` l.201-205), model error means that a model
     trained on gold labels of the same data under the same model scores below the EM solution.
   - The L2-0 ladder (same data, same criterion) is strictly monotone in label noise at every lambda
     (r30 - gold +0.246 at lambda 1). Holding data and criterion fixed, S therefore does reward correct labels.
   - Speaker coverage of the holdout (159 speakers) by the 10 h seed against 100 h may add to this. Not checked.
4. **Where the gap sits.** S_lambda is concave in lambda, with dS/dlambda equal to the posterior-expected LM NLL
   per frame. So S_(lambda+1) - S_lambda brackets the prior term's per-frame cost.
   | | lambda 1->2 | lambda 2->3 |
   |---|---|---|
   | gold | 0.581 | 0.547 |
   | EM | 0.575-0.615 | 0.495-0.530 |
   | EM emis | 0.609-0.676 | — |
   | permphi | 0.777 | — |
   | permphi true inverse | 0.581 | — |
   - The prior term is about 0.58 of about 3.4 nats per frame.
   - The gold-EM gap at lambda 1 does not come from the prior term. The trigram scores gold's posterior no
     better than the EM posteriors. The gap comes from the channel (emission, duration and entropy), which is
     where data amount and criterion act.
5. **Scale mismatch (Yin's mechanism).** The numbers argue against it as the lever. Between lambda 2 and 3, gold's
   posterior has the higher LM NLL, so up-weighting the trigram widens the gap. phi_c passes gold at lambda 3. It
   is untested beyond lambda 3, and by the registered rule lambda is not tuned from this read.
6. **Registered test pending.** A14 (ii) is the model-error test done properly: a gold-init phi under the A10
   recipe on the same stream, with S_g at 48 against 3.2990 plus the Hungarian PER.
   - It is live; 31 checkpoints are on disk in each of `PhiFirstProbeTrainingJob.{MU6Q3RanOQqF,T2V8nn5obzj9,fTBdXD0SwBaA,lR4CfDiHyAvH}`.
   - I did not read partial results.
   - PHONETIC BASIN LOWER would mean search error. NON-PHONETIC PREFERRED would mean model error.

So the verdicts are applied as registered: OBJECTIVE LABEL-BLIND OR WRONG (6/6, control valid) and NO LAMBDA <= 3.
But "model error" should not enter the State or the A16 (b) premise until A14 (ii) reads. The correct statement
is: at the EM phis' operating point, the prior term prefers their own naming (co-adaptation), and the phis'
advantage over gold is in the channel. Whether S's minimum is phonetic is untested.

## Claim 4: which fix class is licensed

- **Ruled out: lambda re-weighting of the phone trigram.** It fails at lambda <= 3, the gap is channel-located,
  and the trend at lambda 2-3 goes the wrong way.
- **Unsupported: a permutation-only label search on S.** The own naming is where the prior term is best, and
  relabelling cannot repair merges (literature).
- **Not yet distinguishable: unit-type search moves (split, merge, reassign) against an objective change** (a
  stronger or word-level prior, or channel capacity or sharpness constraints). A14 (ii) decides between them.
  - If A14 (ii) reads NON-PHONETIC PREFERRED, the objective must change. Two observations then point to where:
    - the trigram cannot tell gold posteriors from EM posteriors (equal LM NLL per frame), which favours a
      stronger or word-level prior;
    - the EM advantage sits in the channel, which favours a capacity or sharpness constraint.
  - If A14 (ii) reads PHONETIC BASIN LOWER, search moves and pipeline initialisation are licensed.
- **Analysis-only controls that would separate the confounds:**
  - (i) A14 (ii) itself: same data and criterion as the EM phis. It separates model error from
    "data plus criterion".
  - (ii) A supervised gold phi fitted with the 16v7R6ztSq1u recipe on the EM phis' train stream (label-using,
    analysis only). Its S against 3.4735 isolates data amount from criterion.
  - (iii) A16 (a) on r70 and r30 under their emission maps. These are right-labelled negative controls for what
    positive dS means.
  - (iv) Relabel-then-refit: each emission-relabelled EM phi continued for K A10 sub-epochs, against the identity
    phi continued for K. This tests whether the acoustic naming's basin is lower once co-adaptation is allowed.
  - (v) dS under random permutations. It calibrates whether the emission map is at least closer to S's
    preference than chance.

## Frame
- All three reads use the registered sets, phis and constants. A15-F uses the D4 dev-other 500 set, MFA labels
  22i2qtnJUOWm and the A10 sub-epoch-48 phis. A16 uses A13's 260 set, tau 1, trigram RtzbESkOedsT and lambda
  {1, 2, 3}.
- The selected epoch (48) is the registered one, not the wave's K* of 12.
- The gold-vs-EM comparison does not answer the model-error question by construction: different data and
  different criterion. That is the main finding.
