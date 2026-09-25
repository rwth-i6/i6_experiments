# Audit: SAE_4A_rename.md "Proposal to the user (2026-09-25 ...)", 2026-09-25

Verdict: CONFIRMED_WITH_CORRECTIONS. What the proposal brings matches the registered decision table:
row 2 fires, row 6 does not (precedence amendment), TP-A2's condition fails on P1a. TP-B' is a
disclosed amendment of TP-B, and TP-D is the registered coarse-to-fine inventory. Every number I
re-derived matches its artifact, with one exception: the AN-0 restore denominator is misstated. Four
sentences state more than the evidence licenses (the headline, "even from the sharpest channel",
"nothing to act on", and "never by changing the objective"). One ruling question asks to let labels
choose an acceptance criterion. The TP-B' draft read cannot answer the user's question by
construction. Several untested regimes the user needs to know about are not mentioned.

Audited text: `recipe/i6_experiments/users/wu/exp_logs/SAE/SAE_4A_rename.md` lines 301-368 (commit
84437a45e). Nothing was edited or rerun.

## 1. Numbers re-derived from artifacts

| Claim in proposal | Artifact | Re-derived | Status |
|---|---|---|---|
| derangement costs 0.51-0.57 | audit_rename_an24 (from per_utterance.tsv) | +0.5132 / +0.5399 / +0.5673 | OK |
| found names beat oracle 1:1 by 0.25-0.35 | audit_rename_an24 table | +0.2508 to +0.3506 | OK |
| 7-11 duplicated, 8-12 unclaimed at 48; gold-key 1 and 1 | `An5ReadJob.2q262c8lhmdT/output/report.txt` A15-F block | ep48 DUP 8/11/7/11, UNCL 11/12/8/11; gold_key ep48 DUP 1, UNCL 1 | OK at 48. At ep0 (as built) the values were DUP 9/10/11/11 and UNCL 13/12/12/13 |
| restore at most 0.005 "of swapped frames" | audit_rename_an0 section 1; audit_rename_an0b section 3 | max 0.0053 of ALL train frames. Of the swapped frames (0.4068) it is 0.013 | WRONG DENOMINATOR |
| identity DELTA -0.040 to -0.001; keys change on 28-51 % | An5 report; audit_rename_an5 sections 1 and 5 | -0.0404 / -0.0047 / -0.0008 / -0.0131; 0.49/0.51/0.36/0.28 | OK |
| best arm at the S bar at PER 0.86 | audit_keyarms_read | rank1 S 3.27275 < 3.289; PER 0.858 / 0.861 | OK. Not carried: the 0.016 margin is below the A10 seed spread (0.025-0.071), one seed, and the four-arm mean is 3.297, above the bar |
| P1a False, P1b True | audit_rename_an24 | channel -3.049..-2.918 vs -2.877..-2.860; LM per phone -3.987..-3.809 vs -3.163..-3.124 | OK |
| A10 finals 3.299, bar 3.289 | lexlat_v2 A10 read; audit_keyarms | S_min 3.29903 (durinit_s01); bar 3.289 | OK |
| HuBERT PNMI 0.686 vs 0.575; Liu 2018 about 300; Zhao t > M^2 | lit_partition_levers Q1/Q2 | as quoted | OK. Liu 2018 is a GAN paper with oracle boundaries (background only), which is not disclosed |
| A10: 48 sub-epochs in about 3 h on one GPU | `PhiFirstProbeTrainingJob.tDzmBsvX73X5` usage used_time 2.80 h (1 GPU); pack `G0Vzzokj5PQC` finished.tar.gz usage 2.875 h (4 arms, 4 GPUs); epoch.001 23:58 to epoch.048 02:46 | about 3.5 min per sub-epoch | OK |

## 2. Sentence-level findings (exact text, then the replacement)

C1 (headline, overstated). "EM cannot correct wrongly named phones here because the fault lies in the
partition, not in the naming step. No naming-side lever reaches it."
- AN-0/AN-0b recorded H1, a naming-side lock-in, on the RIGHT partition, at one step. The AN-5 audit
  says "EM keeps whatever names it starts with, from either init". AN-5 also shows the found
  partitions admit 1:1 renamings at 0.43-0.49 while identity is 0.05-0.14, so the names are a large
  part of the fault. The only naming lever tested was one LM-weighted E-step (AN-0). A failed gate
  licenses "not funding it" only.
- Replace with: "In every regime measured, EM keeps the names it starts with: one LM-weighted E-step on
  the right partition with swapped names (AN-0, AN-0b), and 48 sub-epochs from the four found
  partitions (AN-5, one seed each). On those found partitions S itself prefers the found names to the
  best 1:1 naming (AN-2). The registered table (row 2) therefore does not fund the naming levers, and
  this proposal targets the partition, the only place where S has been shown to see names."

C2 (denominator and operating point). "One LM-weighted E-step restores at most 0.005 of swapped
frames, even from the sharpest channel (AN-0), and the cause is not an escape to empty rows (AN-0b)."
- restore divides by all train frames. The sharp channel is the regime LEAST favourable to an LM-led
  move (margin about 4.4 nats per frame). TP-A1's own regime, a flat channel iterated, was not
  exercised. Seed 1 only, 300 utterances. SIL was not masked.
- Replace with: "One LM-weighted E-step (lambda <= 4.4, either form) from the gold-key phi's sharp
  channel with 5 swapped pairs (seed 1, 300 utterances) restores at most 0.0053 of train frames (0.013
  of the 0.41 that were swapped) (AN-0). Masking the empty OY/ZH rows does not change this (AN-0b);
  SIL was not masked. Iterated EM, a flat-channel start and other seeds were not tested."

C3 (inverted logic; failed-gate rule). "The LM-weight lever (TP-A1) and the rate floor (TP-A2)
therefore have nothing to act on."
- "Not H3'" was a PRECONDITION for TP-A1 in row 1, not a reason against it. LAMBDA KEEPS THE BASIN
  also favoured it. P1c is True: the finals' non-SIL rate is 6.85-7.56 Hz against 8.42-8.55 Hz, which
  is exactly what a rate floor acts on.
- Replace with: "Under the registered table TP-A1 is not funded: AN-0 DEAD dropped AN-3, so row 1
  cannot fire, and row 2 drops the rename levers. TP-A2 is not brought, because its condition needs
  P1a, which is False. P1c is True (6.85-7.56 Hz against 8.42-8.55 Hz). Neither lever has been shown
  not to work."

C4 (literature overstated). "The literature treats duplicated and starved states as a search error of
EM when the global optimum is right. It fixes them with partition moves or restarts, never by changing
the objective"
- The project's own verified literature records objective-side remedies for EM's duplicated or
  balanced states:
  - Johnson 2007: a sparse Dirichlet prior (VB) improves 1-to-1 accuracy (lit_method_v2, [V]).
  - Ravi and Knight 2011: Bayesian decipherment, 95.2 % against EM's 32.6 % on a homophonic cipher
    (lit_phi_name_signal; the phase file's handoff list calls it "the strongest published fix").
  - Posterior regularisation did not fix frequent-class splitting (Ganchev 2010).
- lit_partition_levers' "never the objective" covers only the split-merge and mixture papers it read.
  Its own caveat, that S is right only BETWEEN basins (E1, E11), was dropped. The key-arm audit adds
  that the chance-PER rank1 (3.273) sits at r70's basin S level (3.266-3.271).
- Replace with: "The split-merge and mixture papers read for this phase treat duplicated and starved
  states as a search error when the global optimum is right, and fix them with partition moves or
  restarts (...). Objective-side fixes exist elsewhere with mixed results (sparse priors: Johnson 2007;
  Ravi and Knight 2011; posterior regularisation did not help, Ganchev et al. 2010). Whether S's
  optimum is right is shown only between basins (E1), not within them (E11)."

C5 (partition counts, "after stage-2 EM"). The numbers are correct at 48, but the sentence implies EM
produced them. As built (ep0) the arms had 9-11 duplicated and 12-13 unclaimed; EM slightly reduced
both. They are A15-F gold-type measures (label-using).
- Replace with: "These partitions duplicate 9-11 gold phone types and leave 12-13 unclaimed as built,
  and 7-11 and 8-12 after 48 sub-epochs of stage-2 EM; the gold-key arm has 1 and 1 at 48 (A15-F in
  AN-5, label-using, report only)."

C6 (minor scope). "as J does at key level (A20)": add "which largely re-expresses A20: the found names
were climbed on J, and the keys were selected on the same 260 set (AN-2 audit)". For KEY BASIN, add
"the margin 0.016 is below the A10 seed spread 0.025-0.071, one seed".

## 3. Logic under the decision table

- Brought and licensed: TP-B (row 2) and the coarse-to-fine inventory (row 2). Precedence applied
  correctly (AN-5 EM LOCKED, row 6 does not fire).
- Dropped and licensed: TP-A1 and TP0's lambda arms (row 2), TP-A2 (P1a False), TP-C (AN-1 NOT GOLD
  FIRST), TP0 (not PARTIAL). The proposal's stated reason for dropping TP-A1/TP-A2 is not the
  registered one (C3).
- TP-B' is disclosed as an amendment before any TP-B job. It stays inside TP-B's relocation-only lever:
  merge + split + name is a fixed-M relocation, and TP-B already relocated "to the region of a starved
  LM symbol". Not disclosed, but it must be: the name-choice step is a naming decision accepted under
  S, and on found partitions S ranks the wrong names first (AN-2). lit_partition_levers verdict 2 says
  this explicitly, and the proposal's "No naming-side lever reaches it" sits beside a naming step.
- TP-D amends the registered item ("about 100 classes") to a {100, 200, 300} sweep used for
  coarse-to-fine training. The change is literature-driven and fine, but it is not labelled as an
  amendment.

## 4. Rule conflicts

- Starts: the A10 finals and stage-2 key arms are label-free. The stage-1 keys were selected by
  held-out J; stage 0's label-using gate only funded the search. Minor: the six A10 finals include two
  uniform-duration restarts (A9: uniform "reported only, never chosen") and two durfrz restarts, while
  the phase constraint names durinit. State why they are admissible, or restrict the starts. The name
  choice (LM fit of decoded trigram contexts) and TP-D's second-level k-means on centroids are
  label-free.
- "Unclaimed name" is not defined label-free in TP-B'. The evidence bullet uses A15-F's gold-type
  "unclaimed", and lit_partition_levers' candidate set says "j plus the 8-13 already-starved names",
  which are gold counts. The registration must define it from the phi's own key or occupancy, never
  from A15-F.
- RULE CONFLICT, ruling question 1, second sentence: "If so, may that read be chosen by how well it
  separates the label-built basin set from the finals?" This lets labels set a hyperparameter (the
  acceptance criterion) of a label-free arm, which the phase Constraints declare "not revisitable
  here". Delete it, or flag it as a request to override an absolute rule. Also: LM per phone and rate
  are in the battery BECAUSE they separated label-built from label-free phis (AN-4). Promoting them to
  acceptance would be label-informed selection. Keep them report-only.
- Occupancy KL to the LM unigram, "reported beside, never used for acceptance": this is not an
  objective term, so it is consistent with the trigram-or-higher rule. Carry AN-1's lesson: exclude
  SIL, or the statistic rewards SIL share. Use the uniform-window prior's unigram.
- Generative PER is reported in both arms. Durations are durinit and learned, never MFA. The LM is
  trigram.

## 5. TP-B' design gaps (the draft read does not answer the gate's question)

- "Draft read: RELOCATION LOWER if the chain's final S is below its start by more than the seed spread
  on at least 2 of 3 starts."
  - The count is inconsistent: item 1 lists 10 starts, item 3 says "TP-D's S-best finals, or the A10
    finals".
  - The read is implied by acceptance: every accepted move already lowers S on the same 260 set beyond
    the bar. It is optimistic after 36-60 selections on that set.
  - There is no no-move control (the same EM continuation without moves).
  - It is S-only, and S reached the bar at PER 0.86. It cannot show that names are corrected.
- Needed at registration:
  - a no-move continuation control per start;
  - a read set disjoint from the acceptance set;
  - which "same-config spread" sets the bar: the A10 exact-rerun band is 3.4e-5, while the seed spread
    is 0.025-0.071, and this choice decides the outcome;
  - a label-using name read as the decision read (AN-5's identity rule), permitted because it decides
    what goes to the user and selects nothing inside the arm.
- "(Ueda eq. 8)" is a posterior inner product, so emission-row similarity is only an analogue. "SMEM's
  local-KL split is degenerate" is the literature agent's inference, marked "(mine)".
- TP-D has no draft read, and its S bar 3.289 was cleared by a chance-PER arm. "Tests the partition
  hypothesis directly" needs a registered name-bearing read. Petrov 2006 splits latent subsymbols under
  observed categories; TP-D coarsens the observations, so the citation is an analogy.
- TP-D counts: "The counts come from the literature: Liu et al. 2018 ... and HuBERT's PNMI ..." The
  PNMI drop argues against 100, not for it. Replace with: "The counts span the published range:
  Sicherman and Adi 2023 (second k-means on centroids; V-measure peak at 100-200) and Liu et al. 2018
  (GAN, oracle boundaries, background only; best near 300). Coarsening costs purity (HuBERT PNMI 0.686
  at 500 against 0.575 at 100)."

## 6. Cost against the measured A10 time

- A10 measured: 2.80 h for 48 sub-epochs on 1 GPU; the key-arm pack took 2.875 h. That is about 3.5 min
  per sub-epoch.
- TP-B' at 2 refit sub-epochs:
  - about 7 min per candidate plus the S forward;
  - 10-15 min per round holds for at most 4 candidates, and 5 candidates need a second wave (about
    20 min);
  - the estimate excludes job startup, the separate ReturnnForwardJobV2 S jobs and the queue wait
    between 12 sequential rounds;
  - the refit length is explicitly unfixed, and 4 sub-epochs doubles the cost.
- "2-3 node-hours per start" is right, but no total is given. Over the 10 listed starts it is 20-30
  node-hours (80-120 GPU-h).
- TP-D: the coarse fit is about 2.8 h, so "4-5 GPU-h" implies a 1.2-2.2 h continuation on 500 units.
  That length is not stated. "About two 4-GPU packs" holds only if both stages run in one pack job;
  if the expansion is a separate job, it is four packs.

## 7. Missing items the user needs

- TP0 was not run, so iterated EM from a clean partition with wrong names is still untested (E13). The
  premise is measured for one step on the clean partition, and for 48 sub-epochs on found partitions
  only.
- AN-3 was dropped (after AN-0 DEAD), so LAMBDA OPENS and an iterated or flat-start LM-led E-step were
  never measured.
- AN-1 read NOT GOLD FIRST under V1, V2 and V12. V3 (4- and 5-gram) was not built, so higher-order LM
  terms are untested. TP-C is not brought. The V12 near miss at the non-SIL divisor was +0.0076, below
  the 0.01 floor, with the divisor chosen after the result.
- The sparse channel prior (the handoff item beside TP-A1) is silently dropped and untested.
- One-seed margins: AN-0 used seed 1; AN-5 one seed per key; KEY BASIN rank1's 0.016 margin is below
  the seed spread. Also unmeasured: m without the duration weighting (AN-5), SIL escape (AN-0b), and
  N(s,u) at batched size.
- Constraints from the literature that were omitted:
  - SMEM's mean LL was below repeated EM on 6 of 7 sets (Zhao 2012).
  - The one held-out read found went the wrong way (SMEM 2000 toy).
  - No published work shows that coarsening makes naming recoverable (lit verdict 4).

## Single strongest reason

The decision (TP-B' and TP-D, no TP-A1, TP-A2 or TP-C) follows the registered table. The prose that
justifies it states unlicensed conclusions: "fault lies in the partition, not in the naming step"
against the H1 lock-in AN-0 recorded on the right partition; "nothing to act on" against P1c True and
row 1's own use of not-H3'; and "never by changing the objective" against the project's verified
Johnson 2007 and Ravi and Knight 2011 entries. It also misstates AN-0's denominator (0.005 of all
frames, 0.013 of swapped frames).
