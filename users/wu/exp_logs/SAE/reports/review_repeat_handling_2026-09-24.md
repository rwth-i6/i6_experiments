# Repeat / loop handling in l_tau (phone trigram) and the k2 word-lexicon term -- read-only review, 2026-09-24

Verdict: OK. No term reads one run of identical frame labels as more than one phone token, and
no path count is inflated. l_tau and the k2 term use the same support: y = B(a), SIL counted as
an ordinary symbol, no two equal adjacent tokens. SIL-SIL is excluded too.
The price of that support is a restriction, not a counting bug. A genuine adjacent repeat (e.g.
"bus stop" -> S S) can only be produced as X SIL X. This affects 27 % of dev-other utterances but
only 0.56 % of reference phones. Numbers are in section 4.

Code: recipe/2025-10-speech-llm/src/speech_llm (HEAD 1e05689f). All paths below are relative to that.

## 1. l_tau (sae/emc/lattice.py at topology="blankfree", trigram history)

- The bed pins the configuration in prefix_lm/model/definitions/sae_blankfree.py:444-445
  (prior_history "trigram", order 3) and :479 (topology "blankfree", recognizer_stride 3).
- The blank arc is dead. lattice.py:511-516 sets w_blank to NEG_INF outside "ctc". The blank arc
  is the only arc into f = 0 (lattice.py:890-898, slot [..., :1]). So f = 0 holds mass only at
  the start state (BOS, f = 0) (lattice.py:1001-1002).
- The repeat arc (from f = 1 only) keeps h and s. It adds only the frame's log q(last(h)), with
  no prior term and no segment (lattice.py:842-844, 891-895). Only the emit arc moves h, and it
  moves it to next_h (lattice.py:885-889). So the trigram history advances exactly once per run.
- Re-emitting the same symbol as a new token is forbidden from f = 1.
  - The mask is same_nonsil = (last(h) == k) (lattice.py:347). The k != SIL exception applies
    only to "ctc" (lattice.py:348-349), so in the blank-free bed SIL is masked too.
  - The mask is applied to the source mass on both reduction paths: elementwise at
    lattice.py:571, matmul at lattice.py:700 and :730.
  - f = 0 exists only at the start, and BOS != k there. So no path ever has two equal adjacent
    tokens, SIL-SIL included. Each run is exactly one token, one prior term and one phi segment.
- Test evidence (run by me): test_blankfree_lattice.py passes 3/3. Its oracle (test_blankfree_lattice.py:13-40)
  - enumerates every frame string at stride 3 with the trigram history;
  - opens one token per run (path[t-1] != k), SIL included;
  - checks log Z, post_q and seg_post against the DP to 1e-10 on both reductions.
- Side note, cost only: the f = 0 plane of fwd[B, O, H, 2] is always NEG_INF after the start in
  this bed (the "CTC repeat flag" of SAE_4A_lexlat.md:53). That is dead weight, not an error.

### Trigram training text
- The text is the raw phonemised text with SIL at word boundaries with p = 0.5 (prior.py:1-15).
  The counts are taken with NO run collapse (prior.py:222-241). Example line from the
  window: "<SIL> AH AH <SIL> HH AO N T AH D ...".
- Measured on the bed's fit PhoneNgramPriorJob.RtzbESkOedsT (prior.npz, tri_counts):
  - 213,776 of 80,559,944 tokens with a non-BOS predecessor repeat that predecessor (0.265 %);
  - SIL-SIL count is 0;
  - most repeated: T 54,970, AH 34,530, D 26,438, S 20,475, DH 17,066.
- The lattice uses the RAW log_tri (definitions/sae_emc.py:556-559). The masked diagonal is not
  renormalised, so its mass is lost:
  - text-weighted mean P(k = last(h) | h) is 0.00265, i.e. 0.0027 nats per token;
  - the worst context loses 0.130;
  - by last phone: T 1.15 %, D 0.76 %, DH 0.75 %.
  - The (X, X) history rows exist in the table but are never visited.
- By contrast, L_agg projects its text bigram and trigram targets onto adjacent-distinct support
  and renormalises them (sae/emc/blankfree.py:33-37, 130-141; sae_blankfree.py:490-495). So the
  two terms treat the same text statistics under two conventions. The difference is 0.27 % of
  mass and changes no reading.

## 2. k2 term, L_lex = log Z_HLG(e/tau) - log Z_H(e/tau) (sae/emc/lexlat_k2.py, lexlat_k2_train.py)

- H (lexlat_k2.py:295-329), built at min_frames = 1:
  - min_frames = 1 is ceil(d_min 2 / stride 3), from lexlat_k2.py:283-292, and is asserted
    against the model in lexlat_k2_train.py:372-373.
  - State (p, 0) has a self-loop (p+1 : 0) that continues the run and emits no token (:323).
  - A new token is emitted only to q != p (:324-326), SIL included.
  - H is deterministic on the input label, hence unambiguous: each frame string has exactly one
    path. So a run of k frames parses as one token, and Z_H is not inflated.
- Test evidence (run by me in the scratch k2 env): test_lexlat_k2.py -k h_topology passes 12/12.
  - H . linear(y) equals the bed's transcript_logprob to 1e-4.
  - For every 4-frame string, the aux-label output equals collapse_adjacent(frames).
  - (test_l_round_trips_five_words errored on a missing module in that env; that test is not
    about H.)
- Composition (lexlat_k2.py:875-881): the disambiguation tokens are zeroed and removed with the
  epsilons. compose(H, LG) then matches H's epsilon-output self-loop to no L token.
- L (lexlat_k2.py:512-531):
  - pronunciations are concatenated directly;
  - after each word's last phone, one arc goes to loop_state (no SIL, log 0.5) and one to
    sil_state (log 0.5); sil_state -> loop_state consumes exactly one SIL (:515).
  - Consequences:
    - L cannot emit SIL SIL, consistent with H.
    - When word A ends in X and word B starts with X, the no-SIL join yields X X, which H cannot
      emit, so that half of the join is dead. The pair is reachable ONLY as A SIL B.
    - L has no SIL inside a word. So the 533 of 151,731 lexicon words whose own pronunciation has
      an adjacent repeat (e.g. ADULTERER ... ER ER, MIDDAY D D) are unreachable as lexical words.
      Only ESCAPE pieces joined by SIL can cover them. The escape loop (:554-561) goes through H
      too, so it cannot emit a repeat either.
- The pronunciation-trie DP route uses the same rule without a SIL exception
  (sae/emc/lexlat.py:998), so it has the same support.
- Consistency with l_tau: the two supports are the SAME. Both are the run collapse, SIL is an
  ordinary symbol, and there are no equal adjacent tokens. They differ only in pressure:
  - When a recognizer collapses "bus stop" to one S, l_tau prices the collapsed context with the
    trigram, P(T | AH S), which is a common context.
  - HLG must read B AH S T AA P as some other word sequence ("bus top") or as an escape. The k2
    term therefore pushes toward X SIL X, or toward phone strings that parse without the join.
  - The size of that push is UNMEASURED. No job scored these strings in either the collapsed or
    the SIL-inserted form.

### Over-count read (LexlatK2OvercountJob.f5Ljn6twbc4b)
- The job excludes any string with adjacent equal tokens (lexlat_k2_train.py:979-982). Its gold
  strings are SIL-free MFA references (GoldPhonesJob.ZGSp0hxyd2YP via
  LexlatEquivalenceProbeJob.lq61PSAg1DcC/output/step0_strings.json). Excluded: 772 of 2,864 gold
  strings and 22 private strings.
- The excluded gold strings are the long ones: median 74 tokens against 44 kept. They carry
  37.4 % of gold tokens, so the banked gold column describes a short-biased 62.6 % of tokens.
- The verdict is unaffected: the median is +0.287 against the 0.05 bar.
- The 22 private exclusions are recognizer decodes of the form X SIL X, which HLG would accept
  with the SIL kept. They were excluded only because SIL was dropped before the read.

## 3. phi (sae/emc/reverse.py)

- Standalone, phi imposes no adjacency rule. forward_logsum (reverse.py:299-350) places one
  segment per token of whatever y it is given, and the docstring says "SIL may repeat"
  (reverse.py:3). So two consecutive segments may carry the same symbol when y does.
- Inside l_tau that never happens. A segment is attached only to the emit arc (lattice.py:45-48,
  876-884); the repeat arc never touches s or seg_post (lattice.py:43-44, 1236); and an emit of
  last(h) is forbidden. So one run is exactly one phi segment with d in [2, D_k], and consecutive
  segments always differ. genmarg reads use the same lattice (blankfree_genmarg_jobs.py:49-56).
- Interaction with the collapse:
  - A genuine long X X, e.g. a cross-word S S, must be explained by ONE segment of up to 25
    unit frames (50 for SIL).
  - A run can last at most about (D_k + 2W)/3 recognizer frames: 25 for a phone, 33 for SIL.
  - With rVAD removing silence, neither bound is binding for S S (about 8-10 unit frames).
- Not one of the three terms, noted for completeness: BT synthesises through phi WITH the
  repeats and collapses only the target (bt_blankfree.py:33-41).

## 4. Size of the support restriction (dev-other gold, gilkeyio MFA parquet; reference keeps repeats, eval_per.py:98-103)

- 989 adjacent equal pairs in 177,275 reference phones (0.56 %), in 772 of 2,864 utterances (27.0 %).
  - 962 are cross-word pairs, 2.0 % of 47,849 word boundaries.
  - 27 are within-word (emperor's, sufferer, unnecessary, midday, suddenness, ...).
  - Top phones: T 321, D 152, S 119, N 76, AH 50.
- Only 6 of the 962 cross-word pairs have an MFA silence between the two phones. So the one
  reachable form, X SIL X, requires a SIL token that phi explains with at least 2 unit frames
  inside continuous speech.
- Evaluation collapses runs first and then drops SIL, with no second collapse
  (blankfree_probe_jobs.py:747-757, :130-132). So X SIL X scores as X X. Without it each pair is
  one deletion: a PER floor of 0.56 % absolute.

## What was not checked
- The runtime value of the k2 push at the 962 joins. Measuring it would need a k2 job.
- Whether any trained checkpoint actually emits X SIL X at these joins. The private set shows 22
  such decodes, but that says nothing about which joins they are.
