# SAE 4A -- phiinit: name the phones before phi's channel sharpens

## State

Created 2026-09-25 as a plan. The user asked on 2026-09-25: "Since the current conclusion is EM can hardly rename,
what is your idea? Maybe we need a new training scheme to initialize phi? Plan for it given current evidence and
literature review."
- Audited from a fresh context (`reports/audit_phiinit_plan_2026-09-25.md`, CONFIRMED_WITH_CORRECTIONS). All ten
  corrections are applied below.
- The work runs on i6, because the JUPITER project has ended. Nothing is funded.
- Every training step needs the user's OK. S0 is text only; S1a is a label-using analysis with no training.

No manager or watcher is live.

NEXT:
1. Wait for the user's OK on round 1 and the rulings (Rulings).
2. Once approved, i6 ports only what round 1 needs (Port on demand).
3. Run S0, which fixes D and the run lengths, then S1a and round 1.
4. The decision table decides what goes to the user for round 2.

## Objective

Find a label-free phi initialisation that gets per-unit names right before phi's channel sharpens.

Success is the basin, not gold:
- phi generative PER (direct, dev-other) of 0.50 or less, the top of the basin range 0.35-0.50;
- a recognizer lift to a dev-other PER of 0.362 or less, which is r70-EM's lift, the worst basin phi in A17 (i).

## Why this scheme: evidence, hypothesis, design

Evidence (audited; details in `SAE_4A_rename.md` Results and `SAE_4A_lexlat_v2.md`):
1. From key-built (sharp) starts, EM makes no net move toward the gold names within 12-48 sub-epochs:
   - the four found partitions: key identity 0.05-0.17 (AN-5), even though keys change on 28-51 % of frames;
   - the right partition with deranged names: DELTA +0.024 / +0.013 / -0.005 (TP0);
   - one LM-weighted E-step: restore of 0.0053 or less (AN-0).
   Right names are mostly kept: the TP0 control falls from 1.00 to 0.78, at direct generative PER 0.34.
   No read measures sharpness over time. In flat-start EM, own-name agreement is highest at sub-epoch 4 and then
   falls (0.150 to 0.077, and 0.230 to 0.115, by sub-epoch 12; E4).
2. Right-named phis are useful:
   - The gold-key phi (type-level names, no segmentation) reaches the basin: S 3.207, generative PER
     0.346 direct at 48 (E1, E2).
   - A phi fit to paired strings with 70 % of tokens substituted, with per-unit argmax right by construction,
     lifted a random theta to 0.329 (A19 tri_r70). That is one arm; the argmax condition was not varied.
3. S is a partial judge.
   - Between basins S ranks right: the basin sits at 3.207-3.271, against 3.299 for the A10 finals (E1).
   - On the right partition, S sees names: deranging them costs 0.51-0.57 nats per frame (AN-2).
   - On the found partitions S prefers the found names to the oracle 1:1 names, by 0.25-0.35 (AN-2), just as J
     does (E7).
   - S does not reward accuracy inside the basin (E11).
   - Near the bar it barely separates a chance-PER phi (KEY BASIN rank1, 3.273) from a basin phi (r70-dur, 3.266).
   Every label-free selection in this plan is by S. So S must be shown to see names at the class and split levels
   before it selects there; that is what S1a is for.

Hypothesis H-PI (not measured): flat EM commits to names within the first sub-epochs (E4), while the whole
39-phone permutation group is open and S cannot guide it. If every naming choice is made small (a 2-way split of a
right parent, from identical rows, so that the trigram's contexts decide), the names may be right before the
channel sharpens.
- O-SPLIT tests whether the LM names splits from right parents.
- S1a tests whether S sees split names.
- A-SIL tests the anchor on its own.

Literature (`reports/lit_phi_init_scheme_2026-09-25.md`):
- Coarse-to-fine alphabets:
  - Knight et al. 2006 bootstrap C/V to S/N/V. The result is qualitative, the classes were named by a fixed
    syllable theory, and it needed restarts; without it, EM "hijacks" extra symbols.
  - Petrov et al. 2006 split-merge: +1.1 F1 over direct training (prior report, not re-read).
  - No paper measures naming accuracy for coarse-to-fine against flat EM, and none does it for a named channel
    into noisy units. This scheme has no precedent, which is why the oracle arm comes first.
- Classes from text:
  - V/C is reliable by type: 2-state HMM EM is "100 percent" on 58,156 CMU-derived English word types, judged by
    inspection (Goldsmith & Xanthos 2008); Hulden 2017 reaches up to 100 %.
  - Below V/C the classes are phonotactic. In Hulden 2017, 25 % of splits follow one feature (60 % with the tier
    variant). Mayer 2020's top consonant split is three-way: /w r/, /p b f h j l/, and the rest.
- Anchors:
  - Knight 2006's cipher with P(space|SPACE) = 1 fixed: plain EM gives 68 errors (about 16 %). Reaching 10 errors
    (2.4 %) also needed a trigram on large source data, a cubed channel and source smoothing.
  - Without an anchor, the 7x7 mapping "locks up". 40 or more restarts selected by likelihood always solve it,
    but only because the likelihood ranks the right key first.
  - Klejch et al. 2022 map silence to silence, but their front end is a supervised universal phone recogniser, and
    decipherment lost to a knowledge-based mapping on 5 of 7 languages.
  - Wang et al. 2023 (Theorem 1) assume a deterministic unit-to-phone map; our units have purity 0.644. The
    reading that "non-stationary positional marginals" are needed is the literature agent's inference. ESPUM's
    positional unigram takes validation PER from 71.6 to 39.2. SIL at word boundaries is a context cue, not an
    index-positional marginal.
- Not adopted: the Ravi & Knight 2011 sampler.
  - With a letter 3-gram only, it reaches 23.0 % on Zodiac-408, but 95.2 % on a homophonic cipher with word
    spaces (our text has SIL at word boundaries with p = 0.5).
  - The rejection therefore rests on non-determinism (frame purity 0.644) and noise, not on the missing word LM.
  - Its sparse channel prior stays in `SAE_4A_rename.md`'s proposal.

## Constraints (inherited)

- Pure unsupervised and GAN-free. Gold-derived phis (O-CTRL, O-SPLIT, O-NULL) are disclosed analysis arms, never
  an init or fallback of a label-free arm.
- Labels never train, select or set a design parameter of a label-free arm; they decide only which proposal goes to
  the user. So the anchor in A-CF is set by ruling 2, not by G2, and the naming check (step 5) is its own proposal.
- Every LM term is a trigram or higher, fitted on the uniform-sample window. Disclosed: at class levels this is a
  class trigram, and the phone-history purpose of the rule is met only at the 40-symbol stage.
- d_min 2. Durations come from durinit (general knowledge).
- Selection inside an arm uses held-out S on the 260 set.
- Every 40-symbol phi reports generative PER on dev-other: direct, Hungarian and NMI.
  - Class-level checkpoints report class-level key identity only.
  - Gates use direct PER, because Hungarian re-maps names. Using generative PER as a gate needs the user's OK
    (ruling 5).
- All comparisons are within i6. The i6 phone text is the complete one (JUPITER defect: `SAE_ref.md`).
  - JUPITER numbers are references only.
  - The gates are relative to i6 arms, except the two absolute basin bars (0.50 and 0.362) and O-CTRL's validity
    clause.

## The scheme: a coarse-to-fine named alphabet with binary splits (A-CF)

1. **Tree (text only, label-free).**
   - Build a binary tree over the 39 non-SIL phones from the uniform-window phone text.
   - Root split: V/C by 2-state HMM EM (Goldsmith & Xanthos).
   - Every lower split: a spectral bipartition of each phone's two-sided context distribution. This is Mayer 2020's
     procedure adapted to two-way splits.
   - Continue down to single phones. SIL stays its own symbol.
   - D is the number of split levels below the root; D is at least 5 for 24 consonants. S0 fixes it before any
     training.
2. **Class LMs.**
   - At each level, map the text to that level's classes without collapsing, and fit the trigram with the
     prior's Witten-Bell estimator on the same window. SIL is kept, with the prior's BOS convention.
   - Lattice change (required): the blank-free lattice masks a non-SIL token equal to its predecessor
     (`model/lattice.py:355-357, 519-524`). At the V/C level about 20 % of non-SIL tokens have a same-class
     predecessor (the auditor's count on a 50k-line slice).
   - A-CF therefore needs a lattice option that allows equal adjacent class tokens at class levels. With it,
     segments stay phone-sized and durinit applies unchanged.
   - At 40 symbols the option is off and the lattice is the bed's own; a CPU test asserts identity there.
   - Collapsing class runs was rejected: it would change the durations and the split step.
3. **Class stage.**
   - phi over SIL, V and C, using the A10 durinit recipe with the level-1 trigram, plus the SIL anchor if
     ruling 2 allows it.
   - 8 restarts of 12 sub-epochs; select by class-level held-out S on the 260 set.
   - This relies on S ranking the right V/C naming first (S1a at level 1). Knight's restarts win only under that
     condition.
4. **Split chain.**
   - At each level, every class holding more than one phone splits into its two children.
   - Both children copy all of the parent's type-specific parameters: the type embedding and the duration
     parameters. The optimizer state is reset at each level, and the level's trigram replaces the previous one.
   - Refit 4 sub-epochs per level at tau 1.
   - With identical children, the posterior between them is the LM's conditional given the parent sequence. What
     separates the children is only how a unit's identity correlates with the parent-class context of its segment.
     Whether that names them is O-SPLIT's question.
   - Finish with 12 sub-epochs at 40 symbols, where the channel sharpens.
   - Total: 12 + 4D + 12 sub-epochs; 48 only at D = 6.
5. **Naming check (a separate proposal, A-CF+check).**
   - After each level's refit, refit the swapped naming of each split's two children for the same length. (A swap
     before the refit is a no-op, because the children are identical.)
   - Accept the swap only if the paired per-utterance S on the 260 set improves by more than the spread between
     two same-config refits (seeds 1 and 2) of the unswapped naming, measured at the first split level.

**SIL anchor (A-SIL, and A-CF if ruling 2 allows it).**
- The fit stream is rVAD-masked: `data/vad.py:144-147` keeps rVAD speech frames only. `raw_index` gives their
  positions, and the unmasked units are the VAD job's input. On train, 14.7 % of frames are non-speech.
- Silence units are the units u with P(non-speech | u) of at least 0.5 over the unmasked train stream
  (label-free).
- SIL's emission is fixed for the whole run to the unit histogram of rVAD's non-speech frames.
  - This needs a bypass of the shared emission MLP for k = SIL at every (d, r, eta) (`model/reverse.py:193-200`).
  - Every other type is unchanged.
- Precheck (label-free, CPU): the share of kept train frames whose unit is a silence unit. The anchor proceeds only
  if this share is 0.01 or more. Below that it would touch under 1 % of fit frames.
- Report only, label-using: the overlap of the histogram with the gold-key SIL units.

## Screens (no training)

- **S0 (text only, label-free).**
  - The tree, D and the class trigrams, with held-out perplexity per level.
  - The same-class-predecessor share per level.
  - Report only: the root split against the ARPAbet vowel set.
- **S1a (label-using).** On the gold class partition at each level, build a key-built phi with the gold class names
  and compute class-level S against each single-split swap. Scoring is paired per utterance on the 260 set.
  - VISIBLE at a split: the mean cost per affected frame is at least 0.1 nats, and its speaker-clustered bootstrap
    95 % CI excludes 0. Affected frames are those whose gold class is one of the two children. AN-2's derangement
    bar was 0.1 per frame over almost all frames.
  - Level 1 is the V/C naming.
- **S1b (label-using, after N0).** The level profile of N0's S-selected final: key identity at each tree level. It
  shows the level at which flat EM goes wrong.

## Arms

S0 fixes D, which sets every length below before any training. Let L = 12 + 4D + 12.

| Arm | Label use | What it is | Seeds | Sub-epochs |
|---|---|---|---|---|
| N0 | free | Flat A10 durinit, run in round 1 with the same code as A-SIL | 1-4 | L |
| A-SIL | free | N0 plus the SIL anchor, the single delta | 1-4 | L |
| O-CTRL | analysis | Gold-key phi, tau 1 | 1 | 4D + 12 |
| O-SPLIT | analysis | Gold class-key phi at level 1, then step 4 on tree T | 1-2 | 4D + 12 |
| O-NULL | analysis | O-SPLIT, but each level's children are assigned at random per token in the LM text, so the LM carries no child information | 1 | 4D + 12 |

Round 1 is these 12 runs. Tree F (phonetic features) is not in round 1: its binary structure is undefined, and it
would need ruling 3.

Round 2 proposals (each brought only as the decision table says):
- A-CF: steps 1-4; 8 class restarts, then the 2 S-best into chains, final selected by S. Also a report-only read of
  the class-level key identity of all 8 restarts, so that a fail can be attributed to a stage.
- A-CF+check: A-CF with step 5.
- The lift read on a G3 pass: joint training from the selected phi in the ported rt-ladder form, with dev-other
  greedy PER at ep8.

## Gates (fixed 2026-09-25, before any job)

- **G1 split naming (O-SPLIT against O-CTRL and O-NULL, direct PER).**
  - Valid only if O-CTRL ends with direct generative PER of 0.50 or less. Otherwise G1 reads CANNOT_TELL.
  - SPLIT NAMES: both O-SPLIT seeds end with key identity of at least O-CTRL minus 0.192, and direct generative PER
    of at most O-CTRL plus 0.166. These margins are TP0's 5-pair deficit against its control
    (0.780 - 0.588; 0.502 - 0.336), the level where 29 of 39 names are right.
  - SPLIT FAILS: the O-SPLIT seed mean of key identity is closer to O-NULL's than to O-CTRL's.
  - Otherwise PARTIAL.
  - Reported beside: identity per level, and the split-naming rate per level.
- **G2 anchor (A-SIL against N0, paired).**
  - Take the S-selected run of each arm, compare direct generative PER per utterance on D4 (500 dev-other
    utterances), and compute a speaker-clustered 95 % CI.
  - HELPS: A-SIL is lower, the CI excludes 0, and the mean delta exceeds N0's 4-seed spread (max minus min of direct
    generative PER).
  - HURTS: the mirror image.
  - Otherwise NEUTRAL.
  - Key identity is reported beside.
  - G2 decides only whether A-SIL goes to the user for the lift read (on G3). It does not set A-CF's anchor.
- **G3 basin.** The S-selected final's direct generative PER on D4 is 0.50 or less. A pass funds the lift read.
  - LIFT: dev-other greedy PER at ep8 of 0.362 or less.
  - This holds only if the i6 rt ladder reproduces JUPITER's ladder PERs within 0.01, as the port review found.
    Otherwise the bar is re-derived from the i6 ladder before the lift job, and the change is recorded as an
    amendment.
- All gates are funding decisions. A fail means "not funding it", not "it cannot work".

Decision table (round 1 to round 2):

| Reads | Brought to the user |
|---|---|
| Anchor precheck fails, or ruling 2 is no | A-SIL is not run; A-CF is unanchored; the rest of the table stands |
| G1 CANNOT_TELL | Report only; no A-CF; first find why the gold-key control drifts on i6 |
| S1a level 1 not VISIBLE | No A-CF: the class stage would pick the V/C naming blind. Report |
| G1 SPLIT NAMES, S1a level 1 VISIBLE | A-CF |
| G1 PARTIAL, S1a VISIBLE at level 1 and at every split level that has a VISIBLE read at all | A-CF+check, with A-CF beside |
| G1 PARTIAL otherwise, or G1 SPLIT FAILS | No A-CF. Report the per-level profile; the rename proposals stay the alternative |
| G2 HELPS and A-SIL passes G3 | The A-SIL lift read, beside any row above |
| G2 NEUTRAL or HURTS | Reported; A-SIL is not brought |

## Port on demand (i6; only what round 1 needs)

- New:
  - text-only tree and class-LM jobs, reusing the ported prior estimator;
  - a K-symbol alphabet with a class prior in the phi-first recipe;
  - the lattice option for equal adjacent class tokens, off at 40 symbols, with a CPU identity test;
  - a split job that copies parent parameters, swaps the LM and resets the optimizer;
  - O-NULL's child-randomised LM text;
  - the SIL anchor: the silence-unit histogram from the VAD job's input units and `raw_index`, and the emission
    bypass for SIL.
- From JUPITER: PhiFromKeyInitJob with PhiDurinitSwapJob (also for class keys); the gold unit key (analysis only);
  the key-identity read (AN-5's rule, extended to class level).
- Core fixes already requested: generative PER on dev-other with direct, Hungarian and NMI; S reads on the 260 set.
- Not needed: key search and J, table EM, the A15-F battery, the 48-sub-epoch K* reader.

## Cost

- A10's rate on JUPITER is 3.5 min per sub-epoch on one GH200. The i6 V100 rate is unknown. Measure the first 100
  steps and GPU use before packing.
- Round 1 at D = 6:
  - 8 runs of 48 sub-epochs (N0 and A-SIL);
  - 4 runs of 36 sub-epochs (O-CTRL, O-SPLIT, O-NULL);
  - 3 packs of 4 GPUs, about 2.8 h each at the JUPITER rate.
- Round 2: 8 class restarts of 12 sub-epochs, 2 chains of 4D + 12 sub-epochs, and one lift pack.

## Rulings needed from the user

1. OK for round 1: S0, S1a, N0, A-SIL, O-CTRL, O-SPLIT, O-NULL.
2. Is the SIL anchor from rVAD non-speech admissible as general acoustic knowledge? The bed already uses rVAD to
   mask, but this puts it into phi.
3. Is tree F (phonetic features) admissible in a label-free arm? This is the same question as the pending
   broad-class ruling in `SAE_4A_rename.md`.
4. Where the recipe uses rho, which value do new arms take: the i6 full-text value (about 9.679) or the hard-coded
   9.6619?
5. May direct generative PER (label-using) gate which proposal goes to the user here, as in TP-D?

## Not in this plan

- These stay with `SAE_4A_rename.md`'s proposal: the sparse channel prior, TP-A1, TP-B' and TP-D.
- An EOS term in the prior: positional anchoring at the utterance end touches about 2 of about 150 phones per
  utterance.
