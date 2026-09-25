# Audit: SAE_4A_phiinit.md (phi-init plan), 2026-09-25

CONFIRMED_WITH_CORRECTIONS. Running an oracle split arm (O-SPLIT) before the label-free arm (A-CF) is a sound
way to test the idea, and most quoted numbers match their sources. But the plan should not go to the user or
to i6 until corrections 1-7 below are applied, for four reasons:
- One core step (class LMs "without collapsing") cannot run on the bed's lattice as written.
- The evidence section reverses an audited conclusion about S, and every label-free selection in the plan
  relies on S.
- The SIL-anchor precheck and A-SIL's anchor schedule are undefined.
- G2 and the decision table leave free parameters and let label-using reads set the design of the
  label-free arm. G1's bars are absolute JUPITER values, and their own reference arm fails them.

Plan audited (read-only): `recipe/i6_experiments/users/wu/exp_logs/SAE/SAE_4A_phiinit.md` (line numbers below
refer to it). Sources: `SAE_4A_rename.md` (ledger E1-E17, AN-0..AN-5, TP0), `SAE_4A_lexlat_v2.md` (A15, A17,
A19, keyinit, stage 1/2, repeat handling), `reports/lit_phi_init_scheme_2026-09-25.md`,
`reports/extract_tp0_read_2026-09-25.md`, `SAE_ref.md` (rVAD entry, phone-text defect), port code under
`/e/project1/spell/wu24/worktrees/i6_experiments_cycle_consistency/users/wu/experiments/unsupervised_asr/`
(`data/vad.py`, `reverse_model/phi_first.py`, `reverse_model/duration_prior.py`, `model/reverse.py`,
`model/lattice.py`, `model/blankfree_model.py`), and the memory rules named in the brief.

## 1. Numbers and claims ("Why this scheme", literature)

### Evidence item 1 (lines 31-35)
- AN-5: "key identity 0.05-0.17, flat over 48" matches the AN-5 audit correction (range over epochs 0.05-0.17;
  DELTA -0.040..-0.001). It omits "LOCKED is not frozen: keys change on 28-51 % of frames".
- TP0: DELTA +0.024 / +0.013 / -0.005 matches `extract_tp0_read` (+0.0239 / +0.0125 / -0.0051).
- AN-0: "restore 0.0053 or less" matches (rate-neutral lambda 4.4; all VALID cells <= 0.0053).
- TP0 control "1.00 to 0.78, at generative PER 0.34": 0.7798 and 0.3358 match, but 0.34 is DIRECT genPER.
  Hungarian is 0.3793. Every gate in the plan is on Hungarian, so the convention should be named.
- OVERSTATEMENT: "EM keeps the names phi has when its channel sharpens, from every start measured."
  - Every measured start (AN-0, AN-5, TP0) is a key-built phi, whose channel is sharp at sub-epoch 0. No
    read measures channel sharpness over time.
  - The only flat-start trajectories (E4, A15) show own-name agreement T1 falling 0.150 -> 0.077 and
    0.230 -> 0.115 between sub-epochs 4 and 12. So in flat EM, names still move (they get worse) after the
    first pass.
  - The TP0 audit also records that in 5pair the emission map puts AH, T, N, M and Y back on their own names.
    Whether their frames moved was not measured.
  - Licensed: "from a key-built (sharp) start, EM makes no net move toward the gold names in 12-48
    sub-epochs".

### Evidence item 2 (lines 36-39)
- A19 tri_r70 "0.33 PER": this matches (ep8 0.329).
- "as long as its per-unit argmax is right" states a condition that was never varied. There is one arm.
  - Its argmax is right on only 0.371 of frames (A15).
  - The phi was fit on paired 10 h strings with 70 % of tokens substituted (L2-0), so it is not a phi that
    has type-level names only.
  - Rephrase as an observation: "a phi fit to 70 %-substituted paired strings (per-unit argmax right by
    construction) lifted a random theta to 0.329".
- The gold-key basin (E1, E2) matches: S 3.207; genPER 0.346 / 0.391 at 48.

### Evidence item 3 (lines 40-44)
- "3.207-3.216 against 3.299" matches the sources: gold-key 3.207, r30-dur 3.2115, gold-EM 3.216, and A10
  S_min 3.29903.
  - The range leaves out the other basin members: G-dur 3.224, r70-dur 3.266, r70-EM 3.271.
  - The chance-PER KEY BASIN rank1 reached S 3.273 (genPER 0.861 Hungarian), only 0.007 from r70-dur. Near the
    top, S barely separates a basin phi from a chance phi.
- AN-2 "0.51-0.57" matches (+0.513 to +0.567).
- CONTRADICTED: "The objective is not the block; the search is."
  - E7 / A20: "at key level the objective, not the search, rejects right names".
  - AN-2: S PREFERS FOUND NAMES on 4 of 4 found partitions, by +0.25 to +0.35 over the oracle 1:1 names.
  - E11: S does not reward accuracy inside the basin.
  - The audited rename proposal says: "Whether S's optimum is right is shown only between basins (E1), not
    within them (E11)."
  - "gold ranks 136th of 188 (AN-1)" is a property of the objective J: AN-1 and V3 score fixed keys under J
    and run no search. Citing it as a search failure inverts its meaning.
  - This matters because every label-free selection in the plan is by S: the 8 class restarts, the 2 chains,
    the final, the A-SIL seeds and step 5. The plan should state that S ranks the wrong names first on
    found partitions, and that O-SPLIT and S1a are what test whether S sees names at class and split level.
- UNAUDITED: "Flat EM from random init commits names while the whole 39-phone permutation group is still
  open" is not a measurement anywhere in the ledger. It is the plan's hypothesis. The header "Evidence (all
  audited)" is therefore wrong for 3c, 3d and item 4.

### Literature (lines 47-66, 99-100)
- Knight 2006 C/V -> S/N/V, qualitative, "hijacking": matches lit 1a. The plan omits that Knight's classes
  were named by a fixed syllable theory and needed restarts.
- Petrov +1.1 F1 (88.4 vs 87.3): matches. The lit report marks it "prior report, not re-read", while the plan
  says "full texts".
- "No paper measures naming accuracy": matches.
- G&X 2008 "100 %" and Hulden "up to 100 %": match. Both are by type, and G&X is on 58,156 word types, judged
  by inspection.
- Hulden 25 % and Mayer /p b f h j l/: match. The plan omits Hulden's 60 % under the tier variant.
  - Mayer's top consonant split is three-way (/w r/ vs /p b f h j l/ vs rest), so step 1's "spectral
    bipartition (Mayer 2020)" adapts Mayer's procedure rather than following it.
- Line 60, "A flat channel names correctly only with an anchor. Knight 2006's cipher reaches 2.4 % errors
  with P(space|SPACE) = 1 fixed": INCOMPLETE CONDITIONS.
  - The 2.4 % (10 errors) needed the anchor plus a trigram on large source data, a cubed channel and source
    smoothing.
  - Plain EM with the anchor gave 68 errors (about 16 %).
  - The cubed channel is itself a reweighting of LM against channel, which this plan does not use.
- Klejch 2022, silence mapped to silence: matches. Omitted caveats: the front end is a supervised universal
  phone recogniser, and decipherment was worse than the knowledge-based mapping on 5 of 7 languages.
- Wang 2023 Assumption 2 and ESPUM 71.6 -> 39.2: match the lit report.
  - "Non-stationary" is the lit agent's own inference, marked "(mine)".
  - Theorem 1 assumes a deterministic unit-to-phone map, which units of purity 0.644 violate.
  - SIL at word boundaries (line 120) is a context cue, not a positional (index) marginal in Wang's sense.
- R&K 2011, 23.0 % on Zodiac-408: matches. But "conditions absent: word LM, word spaces" is partly wrong.
  - The same paper's Bayesian letter 3-gram, with no word LM, reaches 95.2 % on the homophonic cipher WITH
    spaces.
  - The phone text has SIL at word boundaries with p = 0.5.
  - So the rejection rests on determinism and noise, not on the word LM.
- Step 3 (line 99), "Knight's winning regime (40 or more restarts always solve 7x7)": the lit report makes
  this conditional on "a likelihood that ranks the right key first ... what the project has not shown for S
  at phone level". The plan drops that condition, and it runs 8 restarts, not 40.
  - With 2 free symbols, 8 restarts is enough only if S ranks the V/C naming right. That is S1a at level 1,
    which the decision table does not use (section 4).

## 2. Rule compliance of the label-free arms (N0, A-SIL, A-CF)

Compliant as written:
- No gold enters training.
- Trigram on the uniform-sample window with the prior's Witten-Bell estimator (step 2).
- d_min 2 (asserted in `model/lattice.py:206-214`; `model/reverse.py` masks d < 2).
- durinit: one maximum-entropy law shared by every non-SIL type, SIL uniform (`duration_prior.py:5-17`), so
  split children inherit identical durations.
- Generative PER direct / Hungarian / NMI reported (lines 75-76).
- Selection by held-out S on the 260 set.
- rVAD is label-free and is the bed's reference preprocessing (`SAE_ref.md`).

Leaks and tensions:
- (a) LABEL LEAK. G2 reads key identity and genPER (label-using), and table row 1 uses it to decide whether
  A-CF carries the anchor.
- (b) LABEL LEAK. Table row 2 sets the levels where step 5 runs from O-SPLIT's per-level split-naming rate
  and from S1a, both label-using.
  - Both (a) and (b) set a design parameter of a label-free arm from labels. The inherited constraint lets
    labels decide only "which proposal goes to the user".
  - Fix: bring each variant to the user as its own proposal (anchored or not; step 5 at level set L), or
    make the choice label-free.
- (c) The always-report-generative-per rule says genPER is report-only and "never a gate". Direct PER is the
  ASR number and Hungarian a diagnostic. The Objective, G1, G2 and G3 all gate on Hungarian.
  - Project precedent uses genPER as a decision read for proposals (TP-D), so gate use is defensible, but it
    needs the user's explicit OK.
  - The scheme targets names, and Hungarian re-maps symbols 1:1 on the decode confusion. The gates should
    therefore use direct PER, or require both.
- (d) The level-1 LM is a trigram over {SIL, V, C}. It is trigram-or-higher by the letter, but the rule's
  purpose (a phone history of order >= 2) is met only at the final 40-symbol stage. Disclose this.
- (e) Class-level genPER is undefined for class alphabets. Specify it (gold mapped to tree classes), or state
  that only 40-symbol checkpoints are read.
- (f) Tree F is correctly held out of the label-free arms until the user rules.

## 3. SIL anchor against `data/vad.py`

- Consistent with the code:
  - `vad.py:144-147` keeps rVAD speech frames only (`indices = flatnonzero(~silence)`).
  - It writes feats, units, `raw_index` (the kept positions) and `orig_length` (lines 205-212).
  - It outputs no mask and no unmasked units. The non-speech positions are the complement of `raw_index`
    within `orig_length`, and the unmasked units are the job's input `units_store`.
  - From BANKED_VAD_COUNTS, train has 18,088,388 original and 15,427,853 kept frames, so 2,660,535 frames
    (14.7 %) are non-speech. Tail frames beyond rVAD's output are padded as silence (lines 145-146).
- phi is not a table. Emission = MLP(emb_type(k) + emb_dur + emb_pos + eta_proj(eta)) -> 500 logits
  (`reverse.py:193-200`).
  - "SIL's emission row is set and held fixed" therefore needs a bypass of the shared MLP for k = SIL at
    every (d, r, eta). Otherwise gradients on the shared layers move SIL.
  - Specify this in Port on demand.
- A-SIL's hold period is undefined. "Held fixed through the class stage and then freed" (line 119), but A-SIL
  has no class stage. G2 cannot be run until the period is fixed (the whole run, or N sub-epochs).
- The precheck (lines 122-124) is not well defined:
  - (i) "a unit with at least half its mass in the non-speech histogram" is ambiguous. It could mean
    P(non-speech | u) >= 0.5 over the unmasked stream.
  - (ii) "almost no such frames" has no threshold.
  - (iii) The comparator, "the SIL frame share implied by the text and durinit", is about 0.49.
    - The unigram-times-mean-duration occupancy puts 0.487 on SIL against a gold frame share of 0.058
      (lexlat_v2 line 247). SIL's durinit is uniform on [2, 50] with mean 26, against 4.41 frames for phones.
    - durinit's own phone mean treats every retained frame as a phone frame (`duration_prior.py:14-17`).
    - So the comparator is about 8x any real SIL share and contradicts durinit's own convention.
  - The precheck is label-free. Fix: a numeric threshold on the fit-stream share alone, fixed now.
- The decision table has no row for the precheck dropping the anchor, or for a negative ruling 2.

## 4. Gates and screens

### G1 (lines 159-165)
- The reference arm fails both bars at 12. TP0 5pair ends at identity 0.5880 and genPER 0.5019 direct /
  0.5034 Hungarian (`extract_tp0_read`), so 0.588 < 0.59 and 0.503 > 0.50. State this, or anchor the bar
  explicitly at the start value 0.593.
- Lines 77-79 say all comparisons are within i6 and JUPITER numbers are reference values only, yet G1 and G3
  gate on absolute JUPITER bars.
  - O-CTRL runs on i6 in the same round but is only reported beside.
  - Pre-register one of two fixes now: a validity clause (G1 is read only if O-CTRL lands within a stated
    margin of TP0's control, 0.780 identity / 0.379 Hungarian), or a bar relative to O-CTRL.
- Both outcomes are reachable.
  - A right-named chain should end near the gold-key arm at 48 (identity 0.766, Hungarian 0.391).
  - Random naming from a V/C-correct start gives about 0.07 (SIL) plus 1/2^depth per phone, roughly
    0.10-0.15. This is my estimate, not a measurement, and it sits close to the 0.20 FAILS bar.
  - A same-start null (children's LM labels shuffled per level) would make FAILS interpretable.
- O-SPLIT runs 4D+12 sub-epochs, against 12 for O-CTRL and TP0, and its tau schedule per level is not stated.

### G2 (lines 166-170)
- Not a single delta. A-SIL has 4 seeds plus S-selection; N0 has 2 seeds. N0 "reused if i6 has run it" may
  also differ in code.
- It compares unpaired pooled max and min. The paired-data rule requires per-utterance deltas (genPER on the
  same D4 items) with a speaker-clustered CI.
- The 0.10 and 0.05 margins are not traced. The available spreads to cite are the A10 finals' Hungarian range
  0.838-0.862 (0.024) and A10/A11 argmax-key identity 0.10-0.12.
- HURTS has no floor. With a two-seed spread near 0, any drop reads HURTS, while HELPS needs +0.10. The rule
  is asymmetric, and HURTS removes the anchor from A-CF.
- No NEUTRAL outcome is named. Row 1 covers it only implicitly.

### G3 (lines 171-172)
- It is defined. Name the item set (D4, 500 utterances).
- The lift read "in A17 (i)'s form" includes the k2 lexicon and HLG. i6 has not ported them, and the
  phone-text defect changes them (151,731 against 182,215 words).
- The 0.36 bar is r70-EM's 0.362, which would itself fail it.

### S1a (lines 131-136)
- 0.01 nats per frame averaged over all frames is structurally out of reach for small splits.
  - A full derangement of every non-SIL name costs 0.51-0.57 per frame, about 0.6 per affected frame.
  - So a swap touching less than about 2 % of frames cannot reach 0.01, even if S sees it perfectly.
  - Normalise by affected frames or tokens.
- The bootstrap should be speaker-clustered, the project's pinned instrument.

### Decision table (lines 175-182): gaps
- In the PARTIAL rows, is A-CF anchored or not?
- "The levels that fail" has no per-level criterion: the split-naming rate has no threshold.
- VISIBLE at only some of the failing levels.
- Tree F passes while tree T fails.
- The anchor is dropped by the precheck or by ruling 2.
- O-CTRL fails on i6.
- S1a at level 1 reads NOT VISIBLE. The class stage would then pick the V/C naming blind, and the lit report
  says (A) then "fails at its first step".
- The class stage itself is not tested in round 1, because O-SPLIT starts from the gold class key. A G3 fail
  in round 2 could not be attributed to a stage. Register class-level key identity of the 8 class restarts
  as a report-only read.

## 5. Logic of the scheme

1. BLOCKING: class text with adjacent equal classes is outside l_tau's support.
   - In the blank-free topology the blank weight is NEG_INF, and `same_nonsil` masks any non-SIL token equal
     to its predecessor (`model/lattice.py:355-357, 519-524`).
   - lexlat_v2's audited repeat-handling section agrees: "A token equal to its predecessor ... is masked",
     "never two equal phones in a row", and the masked mass is not renormalised.
   - On the first 50,000 lines of the uniform sample (`SampleLinesJob.orN768ARKwlt/output/text.phn.gz`;
     4,455,296 tokens; ARPAbet vowel set as V):
     - 19.7 % of non-SIL tokens have a same-class predecessor (CC 696,113; VV 62,136 of 3,847,266);
     - at phone level, repeats are 0.26 %.
     - This is an auditor count on a slice, not a registered statistic.
   - Step 2's "without collapsing, so segments stay phone-sized and durinit is unchanged" therefore cannot
     hold. At level 1 about a fifth of the class trigram's non-SIL mass is masked. A consonant cluster must
     become one C segment, priced by the 4.41-frame phone law, or pass through SIL.
   - Every split level is affected (same-child neighbours), so O-SPLIT in round 1 is affected too.
   - The plan must choose one of two fixes:
     - lift the mask for class alphabets: a lattice-support change, which is an extra delta and is not in
       Port on demand;
     - collapse class runs: then the durations and the class LM change.
2. Identical child rows.
   - With identical embeddings and one shared durinit law, the between-children posterior is the LM's
     conditional given the parent sequence. That part is correct.
   - What breaks the symmetry in the update is only the correlation between a unit's identity and the
     parent-class context of its segment. If the child conditional barely varies with context, the children
     stay near-identical. The lit report warns of exactly this: "the within-class naming problem then
     reappears".
   - The plan states "The LM therefore names each pair" as a fact. It is what O-SPLIT tests.
   - Unspecified: whether child durations reset to durinit (discarding the parent's trained law), the Adam
     state of the new embeddings, and the tau schedule per level (A10 starts at tau 4).
3. Step 5.
   - Swapping the names of two children whose rows are identical is a no-op, so the swap must come after a
     refit. The wording should say so.
   - The S comparison between the two refits has no margin. A10's S seed spread (0.025-0.071 at 48) is above
     S1a's 0.01 bar, so step 5 can pick at chance even at VISIBLE levels.
   - Use a paired comparison with a margin taken from a same-config refit spread.
4. Sub-epoch total (line 109).
   - 12 + 4·6 + 12 = 48 is right.
   - A binary tree needs at least 5 split levels below the root for 24 consonants, and at least 4 for 15
     vowels. D = 6 therefore needs nearly balanced splits, and spectral splits are unlikely to be balanced
     (Mayer's top split isolates /w r/).
   - "Fixed by S0" and "down to single phones" must agree. Define D (levels below V/C) and state that the
     total matches A10's 48 only at D = 6.
5. Cost (line 200): of the 10 runs, only 6 run 48 sub-epochs (N0 x2, A-SIL x4). O-SPLIT x3 runs 4D+12 (36 at
   D = 6) and O-CTRL runs 12.
6. Tree F: its root and binary structure are undefined, since manner has more than two values. It must be
   specified before O-SPLIT runs it in round 1.

## Corrections ranked by severity (plan lines)

1. Lines 91-95, 101-109: the lattice repeat mask excludes about 20 % of level-1 class tokens. Choose a fix
   and add it to Port on demand. This blocks O-SPLIT and A-CF.
2. Lines 40-44: remove "the objective is not the block". Cite AN-2 PREFERS FOUND NAMES, E7, E11 and KEY
   BASIN at genPER 0.86. Mark 3c as a hypothesis.
3. Lines 117-124, 151: define the precheck threshold and a correct comparator, the A-SIL hold period, and
   the MLP bypass.
4. Lines 166-170, 179-180: make G2 paired, single-delta, with a floor on HURTS. Stop label reads from
   setting A-CF's anchor and step-5 levels, or bring the variants to the user as separate proposals.
5. Lines 160-161, 77-79: the reference arm fails the 0.59 / 0.50 bars. Add an O-CTRL validity or relative
   clause.
6. Lines 175-182: fill the decision table's gaps; use S1a level 1; add a report-only class-stage read.
7. Lines 134, 110-112: normalise the S1a bar per affected frame; give step 5 a spread margin.
8. Lines 31, 37, 60, 65, 99: state the conditions (sharp starts only; "as long as"; Knight's cubed channel
   and large trigram; R&K's 95.2 % with spaces; the dropped 40-restart caveat).
9. Lines 26, 160, 168, 172: gate on direct genPER, or both. Hungarian forgives names.
10. Lines 109, 200: define D (at least 5 levels below V/C); the cost line has 6 runs at 48, not 9.
