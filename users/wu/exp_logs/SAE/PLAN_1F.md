# PLAN_1F — Statistics-matching initialization: design space (planner sub-plan)

Sub-plan of PLAN.md section 1f (registered 2026-08-12). Created 2026-08-16 from a five-agent
fan-out (grounding + two web sweeps + brainstorm, adversarial merge; 28 candidates screened).
Holds the 1f design detail so the PLAN.md section stays a page; collapses back to a verdict
when 1f's question closes. The two prerequisite kill conditions stand as registered; the arm
gate was REPLACED 2026-08-16 by the USER's better-than-unpaired criterion (see Verdicts and
rulings below).

## Verdicts and rulings (2026-08-16 — prerequisites ran, SAE_1f.md; USER re-set the gate)

**Kill (i) verdict: FIRED on the raw/deduped stream.** enc50_raw oracle-map PER 0.832
(dev-other) / 0.712 (dev-clean) against the 0.50 bar — worse than the 0.53-0.63 that closed
1a(i). The cap is measured over-segmentation, not confusability: insertions 0.591/0.692 at
2.79 deduped units per gold phone token, while substitutions (0.115/0.132) and frame-level
phone information (PNMI 0.682, H(phone|unit) 1.046 nats) are the best this program has
measured on any unit stream. The verdict licenses not funding unit-level token matching on
the raw/deduped stream; it does NOT say the encoder lacks content — the opposite is measured.
Fork staged, not fully exercised: the battery's pooled rows on the CURRENT codebook
(segment-pooled / Brown-K100 / unit-BPE) are the deciding measurement; feature-level ESPUM
(entry 5) is the fallback if they also cap; death only if everything caps. Prior-codebook
evidence on pooling: fixed-rate 12.5 Hz pooling removed the insertion mass (0.189 -> 0.013)
but at 0.287 deletions under the job's own grid-rasterization caveat
(AuditAvUnitsJob.zzZk9wq8vBfe) — fixed-rate is the wrong pooling; the data-driven rows are
the open question.

**Kill (ii) reading, and the bar it now gets (set before any matcher run).** The
boundary-crossing part of the co-occurrence graph carries the bigram (PMI spearman
0.515/0.517 vs a 0.03 floor), but a matcher sees the whole graph — 0.373/0.370 between the
seg_rand floor 0.214/0.216 and the seg_swap ceiling 0.413/0.398 — and the matcher's own
off-diagonal TV objective separates truth from no-correspondence by only 8.9-11 percent
relative, with the real stream 46-57 percent of the way from floor to ceiling, all AT the
oracle map no matcher reaches. Bar (pre-registered now, against the seg_swap/seg_rand
controls): transition-consuming matchers (ladder entries 1 and 4) may run on a representation
only if ChannelStructureJob on that representation shows tv_offdiag relative span
((seg_rand - seg_swap)/seg_rand) >= 25 percent AND the real stream >= 75 percent of the
floor-to-ceiling span on tv_offdiag. The raw/deduped stream fails both tv_offdiag terms as
measured (its observable PMI spearman fraction, 0.80/0.85, would pass — the objective, not
the correlation, is the binding constraint). Transition-FREE statistics (entries 3, 2, 6) are
structurally unaffected, and the measured 5.9x coarticulation inflation (a quarter of true
transitions invisible in the graph) is an argument FOR them. The committed ChannelStructureJob
read supersedes the uncommitted 0.146 scratchpad number for all registration purposes.

**USER ruling 1 (simplicity).** Initialization as simple as possible is a design constraint,
not a preference. Re-ranked queue: (0) complete the battery's pooled rows on the current
codebook — the only missing measurement, CPU-cheap, decides the representation; (1) entry 3
(unary fingerprint assignment — transition-free, one assignment solve); (2) entry 2 (ridge
positional-unigram, one solve) on the best pooled representation; (3) entry 6 (unit-BPE
lexicon heads); entries 1 and 4 only on a representation that passes the kill-(ii) bar above;
entry 5 (ESPUM training) last, as the least simple.

**USER ruling 2 (gate replacement).** Replaces the arm gate "dev-other PER <= 50 percent
under the 1.0 metric" (registered 2026-08-12; no matcher result existed, so replaceable)
because the USER re-set the criterion: the ONE requirement is that the init be better than
random/unpaired initialization. Operational form (planner): the init must dominate the
strongest CONTENT-FREE nulls — (n1) a marginal-matched random unit-to-phone map (same audio
marginals, no content) and (n2) the 1e pseudo-pair init (the incumbent, measured content-free
in the Z4 diagnosis) — on two pre-registered reads: plain PER as scored (labels eval-only)
and the audio-swap content-dependence control (the output must change when audio content
changes, beyond length/speaker). Margin: to pre-register before the first matcher read.
Coupling note: the 0.50 kill-(i) bar was derived from the old PER<=50 gate; under the new
gate a high-ceiling representation is no longer dead BY CONSTRUCTION — raw stays behind
pooled in the queue only because pooled dominates at equal cost. The fired kill verdict above
is not edited; it stands against its own registered bar.

**Battery verdict (2026-08-16, battery rows in `SAE_1f.md`; queue item 0 DONE).** Kill (i)
is CLEARED at the unit level by data-driven segment pooling: every pooled rung passes the
registered dev-other bar (`seg12.5` 0.414 / `seg16` 0.452 / `seg9` 0.481 vs 0.50; dev-clean
0.380 / 0.385 / 0.466), the best oracle ceilings this program has measured on any inventory —
the arm stays at the UNIT level and the feature-level ESPUM fallback is not exercised. The
cap was the token rate alone: coarsening the INVENTORY at fixed rate is catastrophic
(`brown100` 1.152 dev-other), so the 500-way codebook stays. Rung policy: no single rung is
label-selected into the method — the matcher screens run on ALL of `seg16`/`seg12.5`/`seg9`
(plus `ubpe12.5` for entry 6) and the gate table reports every rung against its own nulls;
`seg12.5` is named primary (leads both splits; matches the adopted coarse-granularity
target), `seg9` is the rung defensible without labels (textbook ~10 phones/s; the measured
gold rate 9.8/9.4 merely confirms). Rung choice for a FUNDED init is a planner/user call off
the gate table and is disclosed as eval-informed. Expectation pin: at matched fertility the
pooled stream still reads worse than a simulated channel with 35 percent random emissions —
the screens license the attempt, not optimism.

**Entry 2 CLOSED (2026-08-16).** Its own registered sigma_min gate fired, and structurally:
sigma_min(P_X) is exactly 0 on every pooled row (usable positions < units), 5e-33 raw,
2e-17 on the only full-column-rank-capable row (`brown100`), and a SIMULATED channel with
perfect recoverability also reads 0 — the failure is a property of utterance-position
supply, not of this channel. Licenses not funding the ridge estimator here; struck from the
queue (the first-real-speech-run novelty claim dies with it).

**Kill-(ii) bar: VOID AS MEASURED (2026-08-16; the bar text above stands unedited per the
gate rule — this verdict is appended, not an edit).** The battery falsified the bar's
ceiling assumption: on every pooled rung the real stream BEATS the `seg_swap` "ceiling" on
tv_offdiag (coarticulation pushes correlated errors onto the diagonal, so a channel that
erases a quarter of its transitions scores better on the matcher's own objective than one
that factorizes by construction), the position term exceeds 1, and the span term reads
3.8-11.4 percent everywhere against the 25 percent bar. The bar cannot rank representations
as written, and NO replacement bar is registered — the battery numbers are now known, so any
new threshold would be post-hoc. Consequence: entries 1 and 4 stay parked behind the
transition-free entries and can be funded, if ever, only against the arm gate directly
(their per-entry kill-tests still apply), after entries 3/6 report. The non-inverted reads
do say pooling bought real signal: boundary-crossing pairs 0.347 -> 0.700, observable PMI
spearman 0.373 -> 0.595, gap-to-floor on the TV objective tripled.

**Arm-gate margin (pre-registered 2026-08-16, before any matcher run; completes USER
ruling 2).** Nulls are built PER representation (magnitudes are per-arm): n1 = a
marginal-matched random unit-to-phone map on the candidate's own stream; n2 = the 1e
pseudo-pair protocol re-run on the same stream (the incumbent, content-free by the Z4
diagnosis). A candidate init passes only if BOTH reads hold on dev-other, plain PER as
scored, labels eval-only: (M1) candidate PER <= min(n1, n2) - 0.05 absolute; (M2) under the
registered audio-swap control (other audio, own text kept) the candidate's PER rises by
>= 0.05 absolute over its true-audio PER — a content-free map moves ~0 under swap, so M2 is
what n2-style inits cannot fake. The 0.05 is about an eighth of the oracle-to-content-free
span on the pooled rows — too large for seed/utterance noise across 5567 dev utterances,
far short of demanding near-oracle from a seed. Registered while candidate and both nulls
are unmeasured; only the oracle ceilings are known.

**USER ruling 3 (2026-08-16, lexicon-free text side).** The initialization screens run TWO
text-side arms, the PLAN_3A section-5c pattern transplanted: (a) phone-level REFERENCE arm —
statistics from the phonemized corpus T_phi, i.e. the form everything above assumes; (b)
LEXICON-FREE arm — the same solver against text statistics that involve no pronunciation
input: text-BPE-512 tokens (the scorer's adopted carry-forward text side, learned on the
unpaired corpus) and/or frequent whole words, with the ladder's unit-BPE word-matching
kill-test (high-frequency silence-adjacent unit-word must show a function-word signature)
as the arm's cheap precondition. The gap between the arms is reported as the MEASURED PRICE
of the lexicon, never argued. Composition note that motivates (b): a unit-to-phone map's
pseudo-labels need the lexicon + word decode a second time to become SFT-able text, while a
unit-to-BPE/word match outputs text directly — the phone arm carries an extra lexicon
touchpoint that must be disclosed in its supervision cost. Honest risks of (b), recorded
up front: text side 39 -> 512 types (thinner per-type statistics, larger assignment) and
English orthographic irregularity enters the channel, which a fixed-statistics solver
cannot learn away (the neural aligner could only because it trains on seed pairs). The
arm gate and margins above apply to each arm unchanged, nulls built per representation
and per text side; PER stays the scored read for both (phonemizing a text output for
evaluation is eval-only lexicon use, licensed at PLAN.md's prior-knowledge table). The
lexicon-free arm's oracle-ceiling read needs one new eval-only ingredient: gold word
alignments plus BPE tokenization of the gold transcript. If the loop's reward ever moves
to the BPE-graph scorer, that is a SEPARATE decision gated on the audio-margin control
(the orthographic side channel the phone graph deletes by construction reopens there).

**Entry 3 verdict: NOT FUNDED (2026-08-16, arm-gate read; `SAE_1f.md` approach 4,
FingerprintMatchJob.O4dpJTesB66u / .MHmUIV85g8Ry, planner-verified against both outputs).**
The gate fails on both pre-registered margins on every representation: best M1 margin
+0.015 (`seg12.5` dev-other) against 0.05, with outright losses to the pseudo-pair null n2
on `raw`/`seg16`/`ubpe12.5`; M2 audio-swap delta 0.005-0.022 against 0.05, and the anchored
control (M2 also run on the oracle, 0.37-0.60, and the random null, 0.004-0.020) places the
candidate at the null's own movement — the map is content-free by the gate's own criterion.
Both of the entry's registered kill-tests fail (manner 0.41-0.49 vs ~0.50; admitted-pair
precision 0.00-0.29 vs ~0.70, the bootstrap starting at chance from a PERFECT ten-pair
seed). Measured cause rather than solver blame: under the fingerprint cost itself the true
phone is in the top five for only 0.23-0.32 of unit mass (chance 0.128) — the matchable
transition-free statistics carry about twice chance information, far short of a 39-way
many-to-one assignment. Two ratifications: (a) the run's fingerprint is narrower than the
registered list — duration and mid-utterance silence adjacency were dropped because they
have NO text-side counterpart; the registered spec overpromised there and the narrowing is
correct, so the diffuseness reading covers everything actually matchable; (b) n2 as the 1e
protocol in map form is faithful, and it is the STRONGEST null on three of five
representations while content-free — the gate design working as intended. Solver discipline
verified: regularization fixed at 0.1 pre-run; the diagnostic sweep shows higher
regularization "improves" PER only by collapsing onto the phone-frequency prior (induced
marginal deviates by L1 1.2-1.4) — a PER-only gate would have been gamed; M2 is what
catches it. The verdict licenses not funding the fingerprint seed, not a claim that
transition-free matching could never work. Queue consequence: the remaining simple-family
step is entry 6's function-word kill-test plus ruling 3's lexicon-free text side — word
granularity is a different statistic class (recurrence of long token sequences) untouched
by either measured weakness (transition-free diffuseness; transition-graph coarticulation);
entries 1/4 stay parked; entry 5 (ESPUM training) is the only remaining entry with
published real-speech evidence and stays last per ruling 1 — funding it is the user's call
after entry 6 reads.

**USER ruling 4 (2026-08-16): no TIMIT bed.** The planner-proposed staged TIMIT
reproduction for entry 5 (reproduce the published setting, then swap in our
representation, then LibriSpeech) is DECLINED — no TIMIT wiring. If entry 5 is ever
funded it is judged directly on LibriSpeech against the registered arm gate; its last
place per ruling 1 and the evidence-scope note (TIMIT-only anchors, a research bet)
stand unchanged. (2026-08-17: the reproduction half of this ruling is LIFTED by
USER ruling 6 — see entry 7; the LibriSpeech-final requirement stands: any candidate
claimed as ours is still judged on LibriSpeech.)

**Entry 6 kill-test verdict: CLEARS — the lexicon-free arm keeps its precondition
(2026-08-16; `SAE_1f.md` approach 5, UnitWordProfileJob.vULzsMp1oise, planner-verified
against the job output).** At every granularity coarser than the bare unit at least one
top-20 unit-word carries the pre-fixed function-word signature (peak 3 hits on `seg12.5`,
2 on `ubpe12.5`, against the 2 the identical rule finds among the top-20 English words),
and the granularity curve turns over inside the swept range, so the count is not
budget-capped. Scope of the pass, from conclusions 20/22: the signature is positional
only (every hit a single unmerged unit; top unit-word 0.006 of token mass vs THE 0.061;
Zipf slope -0.90..-1.01 vs -1.39), and the hits are MORE onset-committed (8.5-14.9x base)
than real function words (4.3-5.7x), which no label-free read separates from an
utterance-onset acoustic effect. GREEN-LIT as an eval-only diagnostic: the implementer's
proposed control — read what the oracle map assigns the hitting units (403 on `seg12.5`,
397 on `ubpe12.5`) at utterance-initial position vs elsewhere — cheap, labels eval-only,
and it decides whether the surviving signature is linguistic or acoustic. Conclusion 23's
self-correction is acknowledged and matters: `seg12.5`'s recurrence is EXHAUSTED at 4.26
unit-words/s (no adjacent pair repeats after 38228 merges, both budgets slack), 1.5x the
English word rate — this qualifies the "recurrence of long token sequences is a different
statistic class" premise in the entry-3 verdict's queue consequence, and the screens are
correctly built to measure it per rung (stop reason reported, budgets 120000 merges /
1500000 fit tokens, far above every observed stall). Entry-3 leftovers closed: the
fingerprint map is NOT kept as a seed for entries 1/4 — the audio-swap control read it as
content-free, so it has nothing to transmit, and the bootstrap kill-test showed even a
perfect ten-pair seed stays at chance on these streams; conclusion 16 does not reorder
the ladder — whether word-granularity statistics escape the diffuseness is exactly what
the running screens measure, and entry 5 stays last per rulings 1/4.

**Ruling-3 screen frame: ratified with one overturn (2026-08-16, decided BEFORE opening
any LexFreeMatchJob output; two of the four jobs had already finished when ruled, their
reports unread).** RATIFIED: (a) rate matching from the unpaired corpus and the 2.8
words/s prior, each text side screened at the merge prefix nearest its own measured token
rate, unreachable rates printed not hidden; (b) the catch-all [OTHER] class withheld from
the candidate AND both nulls alike (verified in code: excluded from the transport, the
random null's mass vector, and the pseudo-pair null), so M1 compares maps, not decode
privileges; (c) the word side's 512 types — traceable to the adopted text-BPE-512 side's
type count, so the two lexicon-free arms differ in granularity, not capacity, and the
closed-vocabulary cost is priced by the coverage column; (d) keeping entry 3's full
fingerprint including the frequency column conclusion 17 says hurts — the registered
protocol kept for comparability beats a post-hoc "improvement". OVERTURNED: the oracle
ceiling must be RESTRICTED to the candidate's own map space ([OTHER] withheld from it
too). A ceiling is the best map in the class the candidate searches; an oracle allowed to
abstain onto [OTHER] holds a privilege no candidate has, so the candidate-to-ceiling gap
would conflate solve quality with an inaccessible option, and the closed-vocabulary cost
it was meant to show is already carried by the coverage column and the oracle's own
error decomposition. The implementer re-runs with the restricted constructor (new hash);
candidate, nulls, M1, M2 are unaffected by the oracle map, so the finished jobs' gate
reads stay valid and the re-run doubles as a determinism check; the unrestricted ceiling
may be kept as a diagnostic column but the registered arm-gap read uses the restricted
one.

**Onset control read: the entry-6 verdict is AMENDED IN SCOPE, and it now rests on direct
evidence (2026-08-16 later; `SAE_1f.md` approach 6, OnsetControlJob.wGkGvMmpWF5V,
planner-verified against the job output and the code).** The green-lit eval-only control
ran and SPLITS by representation. On `seg12.5` the surviving signature is LINGUISTIC:
unit 403 reads AH -> AH on phones and THE -> THE on words at initial vs other positions
(TV_ie 0.074-0.094 / 0.242-0.245 against other-type medians 0.291 / 0.915) and sits
FARTHER from the corpus's own utterance-onset mixture than a typical type — a genuine
THE-like unit. On `ubpe12.5` the headline hit (397, the 13.8-14.9x enrichment) is an
ARTEFACT: 100% of its frames are MFA silence, and since entry 6 DELETES proxy-silence
types when cutting segments (verified at lexfree_match.py segment_tokens), its presence
in the merge list proves the label-free edge-enrichment proxy missed it — ratified as
proven by construction, no mask dump needed. One genuine ubpe hit remains (608, Y -> Y /
YOU -> YOU, @0.75 only). Of `seg12.5`'s three hits only 403 is a word (423/432 are
phone-stable non-words: T and N with word reads that flip across positions). NET: the
lexicon-free arm's precondition STANDS — one genuine function-word-like unit per
representation, confirmed rather than counted — and the onset-acoustic confound recorded
in the entry-6 verdict above is RESOLVED in the linguistic direction for `seg12.5`; the
corrected hit counts are dated under conclusions 19/22. DEFECT recorded: the silence
proxy called 1716 of 8500 ubpe types silence yet missed a 100%-silence type (plausibly a
mid-utterance pause unit, invisible to EDGE enrichment by construction). Consequence for
the in-flight ruling-3 `ubpe12.5` rung and its restricted re-run: the unit-word stream
carries this contamination; NO mid-flight change (the design constants are registered,
both nulls see the same stream, and the coverage / sub-ins-del columns plus the oracle
make the damage visible); a proxy improvement (e.g. re-running enrichment at SEGMENT
edges after the first deletion pass) becomes a registered follow-up only if that rung
reads unreadably. Scale caveat noted, does not flip the call: the other-type median pool
thins to 1-3 types at the deciding prefixes (>= 20 initial occurrences filter); 403
clears even the 12-type prefix-0 pool by a wide margin.

**Citation provenance for entry 6 and the unit-BPE representation (2026-08-17; planner
six-angle web sweep, every recommended paper's abstract fetched and verified first-hand).**
In-program provenance: BPE-on-units entered as a surveyed technique in
`recipe/i6_experiments/users/wu/experiments/ssl/SPEECH_UNIT_BPE_REVIEW.md` (2026-06-20,
written for the pre-SAE BEST-RQ/CIF stack; its verdict — compression tool, not a
mappability tool — is calibrated on ~120 impure codes and does not transfer unexamined to
the 500-way codebook). Entry 6 itself — silence-delimited merges, the word-rate-targeted
prefix curve, and the fingerprint match of unit-words to text words — has NO origin
paper: authored in the 2026-08-16 planner fan-out. Construction citations: Wav2Seq
(arXiv:2205.01086; k-means -> dedup -> BPE "pseudo subwords" — the origin of the
mechanics), acoustic BPE (arXiv:2310.14580; the name and properties), DiscreTalk
(arXiv:2005.05525; earliest subwords on discrete speech symbols), Chang et al.
comparative study (arXiv:2309.15800; dedup+subword as standard supervised practice), Guo
et al. survey (arXiv:2502.06490). Rate-targeted granularity exists only via learned
syllable boundaries (SyllableLM arXiv:2410.04029; Sylber arXiv:2410.07168), and no
published acoustic BPE restricts merges at silence — nearest is concurrent
supervised-phoneme BPE that never merges across word boundaries (arXiv:2604.09332).
Matching-plan citations: structural antecedent Haghighi et al. 2008 (monolingual
fingerprint vectors + latent matching over frequent words); frequency-as-clue and
weak-clue combination Koehn & Knight 2002; founding non-parallel premise Rapp 1995 and
Fung 1995 (Fung names frequency/length/position as parallel-corpus-only clues — counter
with the shared LibriSpeech domain of our two sides); decipherment framing Ravi & Knight
2011, speech-side Klejch et al. 2022 (contrast: supervised universal phone recognizer);
mechanism precedent ESPUM arXiv:2310.02382 (positional statistics, phoneme level).
Cite-and-differentiate constraints (framing risks, binding on any writeup): Ni et al.
arXiv:2406.08380 already demonstrates word-level unsupervised ASR (joint masked
infilling, curated fixed vocabulary) and Chung et al. 2018 (arXiv:1805.07467) already
matches spoken words to text words unpaired (embedding geometry, pre-segmented words) —
we may claim neither word-level unsupervised ASR nor unpaired word matching as new, only
the mechanism (transition-free interpretable fingerprints + shared-substring consistency
filter + pre-registered function-word kill-test) on fully label-free unit-words on an
uncurated corpus. Kill-test grounding: Shi/Werker/Morgan 1999, Shi & Lepage 2008, Mintz
2003, Christophe et al. 2008, Goldwater et al. 2009 (utterance boundaries as the only
given delimiters). External confirmation of the measured shallow Zipf slope:
centre-based (k-means) clustering biases discovered lexicons toward uniform frequencies
(Slabbert/Malan/Kamper arXiv:2606.10781) — the -0.90..-1.01 slope reads partly as
clustering bias, not only as a segmentation failure. Adversarial novelty verdict: the
composition is unpublished as of this sweep (silence-constrained unit BPE, rate-targeted
BPE, fingerprint matching of unit-words to text words, and function-word-signature
validation all searched empty; SylCipher's citing papers checked).

**Ruling-3 screen frame: 2026-08-17 amendments (planner-verified against repr_pool.py, the
RelabelUnitsJob artifact, and the rewritten lexfree_match.py; gate and margins untouched).**
(1) Open-ceiling generation closed. The `ubpe12.5` open-ceiling job died at the 11 h wall
clock with no output and left the graph when `oracle_space` entered the hash; the four
restricted jobs are queued. RATIFIED without any open-ceiling re-run: one restricted pass
emits BOTH ceilings from the same counts on the same held-out rows — `oracle` fitted in
the candidate's map space, `oracle_open` unrestricted as the diagnostic (lexfree_match.py
622-623, reported per split). Pre-registered determinism check at batch close: on the
three seg rungs every oracle-INDEPENDENT column (cand, n1, n2, swap, M1, M2, M2_n1,
cover) must reproduce the finished open runs bit-for-bit, and the old `oracle` column
must equal the new `oracle_open` exactly; oracle-DERIVED columns (M2_oracle,
oracle_sub/ins/del) legitimately differ because the registered ceiling changed space.
The ubpe rung has no cross-generation copy (its open run wrote nothing); the seg rungs
cover the code path. Anyone comparing across generations reads `oracle_open`, never the
new `oracle`, against the old runs' `oracle`.
(2) Stream defect, verified: `ubpe12.5` was built with the merge budget at the function
default — learn_unit_bpe max_merges=8000 (repr_pool.py:129), not overridden at the build
call (repr_pool.py:437) — and RelabelUnitsJob.mkk17SxDKjG2's stats artifact confirms the
stop ON the budget: "8000 merges, vocab 500 -> 8500", measured 14.08 tok/s against the
12.5 target. Rulings: NO rebuild inside this batch — every logged ubpe read (battery
row, entry-3 row, entry-6 kill-test, onset control) is on this stream, both nulls see
the same stream, and magnitudes are per-arm; the rung keeps its registered name but
every reading of it names the measured operating point (14.08 tok/s, budget-stopped);
the "same rate, opposite mechanism" contrast with `seg12.5` is RETIRED — not claimable
on this build; a true-12.5 rebuild (budget raised until the rate target binds, stop
reason asserted at build time) is a registered follow-up, funded only if the gate table
makes the mechanism contrast decision-relevant. Standing rule extended: stream
CONSTRUCTION stages report their stop reason (budget vs target reached), exactly as the
screen's merge stage already must — a header that names the target while the budget
binds is the defect that hid this.
(3) Word-rate unreachability, pre-registered before any gate read: the unit-word rate
floor (prefix 1) sits above the 2.8 words/s target on every rung (implementer: 4.892 /
4.258 / 3.459 on seg16/seg12.5/seg9 — the seg12.5 floor is consistent with conclusion
23's 4.26; exact verification at batch close), so the words text side is screened
1.24-1.75x above word rate everywhere under the registered closest-attainable-rate rule
with the mismatch printed. The words row is read AT its printed operating point and any
verdict names the gap to 2.8; no post-hoc re-targeting. `bpe512` is genuinely
rate-matched; only `seg12.5` lands on the phone target (9.77 vs 9.86; `seg16` 12.30
over, `seg9` 7.04 under).
(4) Resume/checkpoint change RATIFIED as hash-neutral and rng-safe, verified in code:
Task(resume="run") plus a merge-list checkpoint keyed on K and max_merges; merge
learning is the only cached stage and draws nothing from the job rng — the single rng
consumer (the length-matched pairing) sits after the cache boundary — so a resumed run
scores bit-for-bit what a single-pass run scores. The 11 h timeout's cause is SETTLED
2026-08-17 (replaces the OPEN clause, because the implementer measured it): the
per-segment rebuild of the ~108k-entry merge table inside apply_unit_bpe — 32.6 ms per
50-unit segment at the real inventory vs 0.2 ms with the table prebuilt — burned the
wall in the prefix-curve and scoring phases; merge learning is exonerated (stopped at
108,169 merges, rate target binding under the 120000 budget, stop reason satisfied).
This amendment's in-place patch is superseded by (5).
(5) Audio/match job split (implementer 2026-08-17, user-directed; planner-verified in
code the same day). New UnitWordStreamJob owns the audio side — segmentation, merge
list, prefix curve — checkpointing merges.pkl and each prefix pickle as it completes, so
a wall kill resumes with progress; LexFreeMatchJob reads that artifact and loads only
the <=3 prefixes its text sides select, and the merge-table rebuild is memoized (built
once per prefix). All 8 ruling-3 hashes are new; the finished seg rungs re-run, and that
re-run IS the refactor's determinism gate, blocking commit: new seg9 must equal
LexFreeMatchJob.pyE0Gd5fqjS6 bit-for-bit in every column; check (1)'s cross-generation
columns then read unchanged on the new outputs. Planner code audit: score preservation
confirmed at source — the job rng is created at the same point and consumed at exactly
one site (the length-matched pairing, per text side, same loop order), nothing moved to
the audio job draws from it, the merge-table hoist is the identical dict construction,
and the cached audio artifact roundtrips exactly (JSON floats, pickled int64 streams).
Old dirs rug97MhrnhNb / Ckke5m2jdnX4 and the finished seg hashes stay on disk as the
determinism references and live log-Catalog citations. Approach 7 lands ~1 day late.
2026-08-17 verdict: PASSED — new seg9 (LexFreeMatchJob.PQGETAeQAVaZ) reproduces
pyE0Gd5fqjS6 bit-for-bit on both lexfree.json and lexfree.txt; planner re-ran the md5s
independently and they match. Committed as 7d5f368. Post-gate hardening ratified as
hash-neutral: the match job is deliberately non-resumable again (a wall kill now
surfaces loudly; it checkpoints nothing, so clearing is safe), the audio job keeps
resume with its real checkpoints, and every stage line prints elapsed-vs-requested wall.
(6) Corpus asymmetry across rungs, pre-registered BEFORE the ubpe rung lands
(2026-08-17). Fact, planner-verified from gold.json, the three audio metas, and the
merged units.pkl keys: the seg rungs' audio side is built on the 8,416-utt seed dump
(20.48 h — the only utterances with continuous states, which segment pooling needs)
while ubpe12.5's is built on the full 34,106-utt train+dev stream; both corpora contain
both dev splits in full and the seg set is a strict subset of the ubpe set, so eval_all,
the scored fifth, and the ceiling-fit rows are IDENTICAL across all four rungs — the
per-rung gate (each rung against its own nulls, which see the same stream) is unaffected
and PER is comparable at the eval-set level. What is NOT matched is train-side: the
screen's merge list and unigram fingerprints see ~4-5x more audio on ubpe (max_fit_tokens
caps only the merge FIT — kept/total to be reported when ubpe lands — the fingerprints
and the pseudo-pair null still consume the full stream). Reading rules: every cross-rung
reading names each rung's audio operating point; if a seg rung wins the selection, no
control is needed (it beat a data-favored competitor); if ubpe12.5 wins or its margin
decides funding, the win is not attributable before a corpus-matched control — one extra
screen of the ubpe stream restricted to the seed 8,416 utts (well-defined by the subset
property; seg-scale cost) — and if the restriction flips the order, the decision goes to
the user as data-budget-vs-representation with both numbers. Named residual: the
ubpe12.5 REPRESENTATION itself (the relabel-time merge list) was also learned on the
full stream; that layer is intrinsic to the artifact as dumped, is not matched by the
control, and is priced only by the already-registered true-12.5 rebuild follow-up.
(7) 2026-08-17 BATCH-CLOSE VERDICT: ruling 3 FAILS the arm gate in all twelve cells
(best dev-other M2 0.0252 = ubpe12.5 phones against the 0.05 bar; best M1 +0.0146 =
seg9 phones; M1 negative in 10 of 12). Planner re-verified every body cell, rate and
prefix against the four lexfree.json artifacts, and re-derived the closest-rate prefix
selection as the argmin over each rung's measured curve — all 12 selections reproduce;
the rule leaves no discretion. The lexicon-free text arm is NOT FUNDED, and the phone
reference side fails the same gate, so no 1f init is fundable from the screens run to
date; per the standing rule this licenses not funding, never "it would not have
worked". The rate mismatch does not carry the miss: the rate-MATCHED cells (seg12.5
phones 9.77 vs 9.86, seg16 bpe512 5.29 vs 5.39) fail by the same order of magnitude as
the mismatched words cells, so no rate-matched retry is proposed. Determinism:
amendment (1)'s cross-generation check PASSED exactly on all three seg rungs (every
oracle-independent column bit-for-bit vs the finished open runs; old oracle == new
oracle_open), and the job split reproduces the whole restricted generation bit-for-bit
(seg12.5 and seg16 md5-identical to their single-job runs; seg9 gated earlier).
Amendment (3)'s floors verified exact (4.892/4.258/3.459). Amendment (6): the
conditional corpus-matched ubpe control is NOT triggered (ubpe neither wins nor
decides a margin — it is last on every text side), and the true-12.5 rebuild is NOT
triggered (no mechanism contrast is decision-relevant under an all-fail table). Scope
guard on the log's conclusion 28: its valid basis is the per-rung gate structure plus
the all-fail table; the "worst despite 4-5x data" clause corroborates direction only —
K=8500-vs-500 and the budget-stopped stream (amendment (2)) confound any
representation attribution. Bookkeeping, sent to the implementer, no gate number
touched: the seg12.5/seg16 UnitWordStreamJob hashes are swapped in the log Catalog and
the approach-7 header metadata — mPnLApAbYnVG IS seg16 (43821 merges, kept 906940),
eIxgmMh99RSE IS seg12.5 (38228 merges, kept 720336, independently reproducing entry
6's 38228); the LexFreeMatchJob labels and their stream wiring are correct. OPEN FORK,
the user's: with ruling 3 failed, entry 3 not funded, entry 2 closed and entries 1/4
parked without a readable screen, the only unkilled ladder entry is 5 (ESPUM, judged
directly on LibriSpeech per USER ruling 4, funding bar raised by the evidence-scope
note) — fund entry 5, register a new screen for entries 1/4, or close 1f.
(7a) 2026-08-17 addendum (implementer probe, planner-verified; no gate number touched):
the ubpe12.5 WORDS cell's restricted ceiling is NON-FUNCTIONAL — PER 1.0667 vs the empty
hypothesis's 1.000 (word currency 1.4193), losing to the content-free pseudo-pair null
(0.8814) on the same held-out rows — so that one cell is additionally annotated
UNINFORMATIVE ABOUT MATCHING QUALITY: its failure is structural at the registered
operating point — scored-fifth emission capacity 7.114 symbols/s against a measured 2.786
reference words/s (2.55x; the corpus-wide 6.455/s understates the scored rows), with
within-segment repeat collapse map-dependent (the ceiling sheds 32% of emissions, the
candidate 8%) — not evidence about the solver. Amended 2026-08-17 later, replacing the
first version's claim, because the implementer refuted it: the "word-error floor ~1.32
for every always-emitting map" is RETRACTED — a constant map collapses each
silence-delimited chunk to one token and lands near 1.0, so no class-wide error floor
above 1.0 follows from the rate alone; the ceiling-loses-to-the-content-free-null read
below is the load-bearing uninformativeness statement. Cell-specific: the three seg words cells keep
functional restricted ceilings (0.671-0.770) and remain informative negatives, and ubpe's
open ceiling (0.699) shows the restricted-ceiling overturn is what exposed the pathology.
One number corrected before recording: the coverage column is frame-weighted (0.4344);
token-level coverage on the scored fifth is 0.7174, so the closed-vocabulary word-error
floor is 0.283 — the rate, not the coverage, binds. The gate row STANDS (M1/M2 are
candidate-vs-null reads, ceiling-independent; the cell's M1 is -0.520) and the batch-close
verdict is unchanged — the no-retry reasoning rests on the rate-MATCHED phones/bpe512
cells, untouched by this. The 512-type vocabulary (frame ratification (c)) and the 6.455/s
prefix-1 floor (amendment (3) rule) are traceable registered constants; no re-derivation
and no word-level re-run is registered — reopening a word-level cell would be a frame
redesign (rate gap, abstention) and a user decision. Record: SAE_1f.md verifier bullet
2026-08-17.

(7b) 2026-08-17 (text-sample coverage defect, disclosed post-close; verdicts not
reopened): the entry-3 and ruling-3 text statistics were computed on a sample that stops
at 60.6% of T_phi's lines (stride 80 x 300000 kept lines = source line 23,999,921 of
39,630,169), and librispeech-lm-norm is ALPHABETICALLY sorted by sentence, so sentences
beginning roughly P-Z never entered the fingerprints, marginals, or either null's
construction. Candidate and both nulls consumed the same truncated sample, so every
within-cell gate comparison stands and the all-fail verdict is unchanged; the limitation
is disclosed wherever those text statistics are quoted. Entry 6 (stride 400 x 100000 >=
the corpus) is unaffected. STANDING RULE, extending amendment (2)'s stop-reason rule:
every text-side sampling stage must satisfy stride*max_lines >= corpus lines and assert
it at run time; the entry-5 batch pins the proven full-coverage stride-400 sample.

**USER ruling 5 (2026-08-17): the 1f fork resolves — entry 5 is FUNDED, as simple as
possible.** The user greenlights "item 5: PUSM style matching", i.e. entry 5 as
registered — phone-level positional-unigram + skipgram matching on segments (ESPUM,
Wang/Hasegawa-Johnson/Yoo, ICASSP 2024, arXiv:2310.02382) — NOT the word-level PUSM
loss, which lives on curated fixed-vocabulary corpora in the discarded JSTTI class
(evidence-scope note). Second user instruction, same day: the whole process/algorithm
stays as simple as possible — ruling 1 extended to the funded batch's execution. Judged
directly on LibriSpeech (ruling 4) against the unchanged arm gate and margins; the
raised funding bar is operationalized as ONE contained batch (spec in the section
below): a failed gate closes entry 5, and with every other ladder entry killed, parked,
or closed, 1f then returns to the user. Registered before any entry-5 training run
exists.


## Entry 5 funded batch — ESPUM on the pooled seed stream (registered 2026-08-17, pre-run)

**Purpose.** Produce the 1f initialization by training-based statistics matching — the only
unkilled ladder entry — and thereby answer 1f's question: can statistics matching replace
the GAN as the bootstrap. A research bet, not a reproduction: no PUSM/ESPUM-family result
exists on unrestricted LibriSpeech, and our unpaired-text setting maps to the paper's
UNMATCHED-text TIMIT column — honest anchor PER 0.473 test (0.451 after its relabeling
iteration), phone level — not the matched one.

**Approach.** The reference mechanism verbatim wherever possible; the released code
(GraphUnsupASR, MIT license, verified first-hand 2026-08-17 along with the camera-ready) is
normative where the paper is silent. Reference core: one-hot DISCRETE frame input (the
paper's own finding: raw continuous features are unstable — so no feature dump is needed
anywhere), one-layer CNN kernel 4 stride 1 no-bias dropout 0.1 over frames, posteriors
mean-pooled within fixed segment spans, L1 matching of batch count statistics — positional
unigram at absolute segment index (batch statistics truncated to the common speech/text
length, as the code does); bi-skipgrams at skips 1-6; tri-skipgrams at skips {1,2}x{1,2}
with the code's 2-of-4 cycling; NO 4/5-gram terms (its ablation: they hurt) — plus
frame-level output smoothness (MSE, weight 16), Adam lr 0.004 betas (0.5, 0.98) eps 1e-6
clip 20, batch 640 utterances, 40000 updates, and no adversarial/code-penalty/gradient-
penalty terms. Seven registered deviations, each traceable: (i) segment boundaries FIXED
from the battery's Ward pooling — the learned segmenter and soft alignment are dropped
(simplicity ruling; the battery already measured these boundaries best); (ii) therefore no
relabeling iteration; (iii) input one-hot over OUR 500-way codebook at 50 Hz
(units_r12.5.pkl), not k-means-128; (iv) silence: the ruling-3 convention — proxy-silence
segments cut and deleted, no SIL output class, T_phi is silence-free — not TIMIT's
sil-class; (v) checkpoint/seed selection LABEL-FREE by the 1c-adopted unsupervised metric
(KenLM 4-gram phone-LM perplexity of decoded dev hypotheses weighted by vocabulary
coverage; the released code computes this family but its shipped config selects by phone
error rate against test references — quarantine-incompatible, so this deviation is
mandatory and disclosed); (vi) eval decode pinned to the screens' protocol — per-segment
argmax, collapse consecutive equal phones within a silence-delimited chunk, concatenate,
phone_error_counts against gold.json on the scored fifth — so the banked nulls price the
candidate legitimately; (vii) 3 seeds on the full loss (the paper ran one) — stability is
the method's claimed advantage and gets measured, not assumed. Audio side: the
8,416-utterance seed stream (20.48 h); rung seg12.5 primary (battery leader on both
splits; silence-stripped rate 9.77/s vs the 9.86/s phone text rate — the rate-matched
rung; disclosed eval-informed); boundaries, silence cuts, and positional indices read from
UnitWordStreamJob.eIxgmMh99RSE output/audio/prefix.0.pkl (per-segment ids + frame
starts/ends — the exact rows the banked nulls consumed; pin it as an explicit job input).
Text side: T_phi (TextToPhonemeJob.THKMON3k9LJQ; 39 phones, silence-free, line =
sentence), working sample stride=400 / max_lines=100000 — the entry-6 setting, full
alphabetic coverage per (7b), asserted at run time; text batches of 640 lines per update
drawn from that sample, seeded.

**Experiments.** (E1) one plumbing check: a supervised probe (cross-entropy on eval-only
MFA gold, fitted on the ceiling-fit rows, checkpoint discarded) through the same
input/pooling path must land near the battery's oracle-map ceiling for this stream (0.414
dev-other) — the retired continuous_gan.py scaffold reads a different encoder at a
different rate and cannot serve. (E2) the health pair on seg12.5: full loss x 3 seeds
(0,1,2) + bigram-only x 1 seed (the reference ablation's collapse arm: TIMIT 71.6 vs
39.2). Health checkpoint, no new constants: an arm has COLLAPSED if its
label-free-selected dev-other PER is not below n1 (0.8946, the banked marginal-matched
random-map null); expected signature: bigram-only collapses, full loss does not; if the
full loss also collapses, STOP and report — the boundary/rate source is suspect and the
fork is the user's. (E3) the gate read on the label-free-selected candidate (one
checkpoint across the three seeds); the other seeds' PERs are reported as stability
evidence, never as gate inputs. Job idiom: the PsiAlignTrainJob pattern (sisyphus GPU
job, torch in run(), resume checkpoint; ~4 single-GPU runs plus CPU selection/eval). No
sweeps — every constant above is pinned; any change is a new registration.

**Gate.** The registered arm gate UNCHANGED (ruling 2 + the 2026-08-16 margins), against
the banked seg12.5 phone-side nulls on the identical scored fifth
(LexFreeMatchJob.rk48Zk5U6jzW: n1 0.8946, n2 0.9239, oracle ceiling 0.4148 dev-other):
(M1) candidate dev-other PER <= min(n1, n2) - 0.05 = 0.8446; (M2) audio-swap PER rise
>= 0.05 under the screens' nearest-length-partner control applied to the generator
(donor audio forwarded, own reference kept). Plain PER as scored, labels eval-only, PER
never selects. A failed gate licenses not funding further ESPUM work — never "it would
not have worked" — and returns 1f to the user with no unkilled entry. One contingency,
planner call off the seg12.5 read only: a seg9 repeat (the label-free-defensible rung)
if seg12.5 passes the health checkpoint and the gate read makes rung choice
decision-relevant.

**Status.** FUNDED 2026-08-17 (USER ruling 5, with the simplicity instruction); spec
registered pre-run, before any entry-5 training exists. Awaiting implementer build +
launch. 2026-08-17 status update: built and launched — four
training arms running (full loss seeds 0/1/2 + bigram-only control), build audited
(addendum 2). E1 VERDICT 2026-08-17: PASSED — supervised probe 0.3565 dev-other PER
(EspumProbeJob.5KJjR2SsYBJT, planner-verified) vs the 0.4148 memoryless oracle-map
ceiling; registered reading: E1 is a ONE-SIDED floor test — the probe's kernel-4 context
is a strictly richer class than any per-unit map, so beating the ceiling is the pass
outcome, and the plumbing carries at least ceiling-level phone information; 0.3565 is a
supervised eval-only read, not an entry-5 performance claim, and no gate constant
changes. Watch item for the health checkpoint: deletions dominate even supervised
(0.2056 of 0.3565; insertions 0.0213), the expected sign of 9.771 seg/s against the
9.86/s phone rate after duplicate collapse — the collapse signature of the four arms is
to be read on the sub/ins/del split alongside total PER. GATE VERDICT 2026-08-17
(planner-verified from the four EspumEvalJob artifacts, the EspumPickJob artifact, and the
banked nulls; every M1/M2 re-derived): FAIL, both clauses, on the label-free-selected
candidate full_s1 (weighted phone-LM perplexity 31.41 vs 31.49/33.04; all three at
vocabulary coverage 1.0) — dev-other PER 0.8580 vs the 0.8446 bar (M1 +0.0365 vs 0.05)
and audio-swap rise M2 +0.0466 vs 0.05 (named CLOSE: 0.0034 short; M1 is not close).
Health checkpoint PASSED AS WRITTEN — no arm at or above n1 0.8946 — but the expected
signature is ABSENT: bigram-only (0.8748, best checkpoint at update 30000) beats two of
three full-loss seeds, the full-loss seed spread 0.0268 dwarfs its 0.0015 mean gap to the
ablation, and the reference's TIMIT 71.6-vs-39.2 separation did not transfer to this
stream; recorded as an observation about the stream/objective, not a verdict change.
Registered caution: the label-free selection metric separates the ablation cleanly
(53.86 vs 31-33) while PER does not — it is registered for picking checkpoints/seeds and
nothing here licenses using it to rank methods. Failure mode is IDENTITY, not rate and
not collapse: the arms emit 0.963-0.974 of the reference phone count, use all 39 phone
types, and carry ~80% of their error as substitutions — the E1 deletion watch item
resolved the opposite way. What the batch bought: best 1f arm on this rung (0.8580 vs
the 0.8809 unary fingerprint solve; both margins roughly tripled), still 0.44 above the
memoryless ceiling (0.4148) and 0.50 above the same path's supervised read (E1 0.3565) —
not a plumbing gap. Seg9 contingency NOT exercised (planner ruling: the registered
condition — the gate read makes rung choice decision-relevant — is unmet; the failure is
identity-dominated at near-correct emission rate, which a lower-rate rung does not
address). Per the registered gate: entry 5 CLOSES, further ESPUM work is NOT FUNDED —
which licenses "not funding it", never "it would not have worked" — the conditional
BPE-level follow-up's condition is unmet and it does NOT open, and 1f RETURNS TO THE
USER with no unkilled ladder entry. Numbers and the full table live in SAE_1f.md
approach 8; conclusions 31-34. 2026-08-17 (later) STATUS UPDATE: USER ruling 6
resolves the returned fork — 1f does NOT close; continued (skip)gram-matching work is
funded and registered as entry 7 (reproduce-then-bridge) below. This gate verdict and
both its clauses stand unchanged; entry 5 itself stays closed.

**Registration addendum (2026-08-17, pre-run — replaces three Approach sentences above,
because the implementer's pre-build audit caught them diverging from the normative code;
planner re-verified each in the downloaded source before ruling; no loss code existed yet).**
(a) MODULE PINNED: port wav2vecu_graph (wg_model.py), NOT the repo's unused espum module —
run.sh launches with common.user_dir=wav2vecu_graph and the shipped configs never invoke
espum.py (whose saved-buffer even/odd gradient splice is a different algorithm); the
smoothness 16.0 and code_penalty=gradient_penalty=0.0 constants are run.sh command-line
overrides (lines 128-130), not the YAML, confirming this is the anchor-producing path.
(b) TRI-SKIPGRAM AS-SHIPPED: sample_skipsizes indexes its 4-pair list modulo k=2
(wg_model.py:1039-1044), so the effective tri-skipgram set is the two pairs with first
skip 1 — (1,1) and (1,2), order swapped on odd updates — and (2,1)/(2,2) are dead code.
We run REFERENCE-VERBATIM (the two fixed pairs): the anchor PER 0.473 was produced by
this behavior, and a "fix" would silently change the method and un-anchor it. The
Approach's "{1,2}x{1,2} with 2-of-4 cycling" over-described the code and is replaced by
this sentence. (c) ORDER OF OPERATIONS AS-SHIPPED: frame LOGITS are mean-pooled within
fixed segment spans (FixedSegmenter.logit_segment, index_add_ / count divide) and softmax
is applied AFTER pooling, to the segment-level logits (wg_model.py:1320-1330); the
smoothness MSE (weight 16) penalizes ADJACENT SEGMENTS' logits, not adjacent frames
(wg_model.py:1703-1708, on the post-pooling tensor). Both replace the Approach's
"posteriors mean-pooled" / "frame-level smoothness" wording; the count statistics
themselves are computed from the post-softmax segment posteriors, unchanged. Noted, no
decision needed: under FIXED segmentation the trainable-segmenter fields
(segment_weight, binary pos_weight, soft_pool_join) are inert, and generator input
becomes 500-way one-hot in place of the config's float/127 — expected consequences of
registered deviations (i) and (iii), not drift.

**Registration addendum 2 (2026-08-17, post-launch pre-result — build audited, three
implementer decisions ratified before any number exists).** Planner re-verified the two
build-shaping source facts: length reconciliation is TRUNCATION of the longer side's tail
to the common speech/text length (wg_model.py:1352-1358; zero-padding would charge
unmatched tail positions full positional-unigram mass — the implementer caught and
reverted exactly that in their own first draft, and a unit test now asserts truncation),
with the smoothness term deliberately outside the cut (dense_logits captured pre-block,
untruncated mask); and position_skipgram is False in the shipped L1 config (line 73) and
by dataclass default, so the skipgram statistic is the non-positional matrix-product
form. Ratified implementer decisions, now registered: (d) SELECTION SET: the label-free
checkpoint/seed selection reads the 2,292 dev-other utterances OUTSIDE the scored fifth,
so the gate is never read on rows the checkpoint was chosen on — a strictly more
conservative refinement of E3's "decoded dev hypotheses" (the banked nulls are static
maps with no selection step, so no comparability cost). (e) STRUCTURAL QUARANTINE: the
training job receives only a list of utterance identifiers (EspumDevIdsJob) and no gold
path at all; EspumEvalJob is the only job that opens gold — "PER never selects" holds by
construction, not convention. (f) E1 TARGETS: per-segment majority non-silence phone
from the frame-level MFA alignment at 50 Hz, all-silence segments excluded via
ignore_index (0.96% of segments on the 300 fit utterances) — matches the frame-overlap
construction of the battery's oracle ceiling, keeping "lands near 0.414 dev-other"
readable; the rejected alternative (uniform time-warp of the gold phone sequence onto
the segment grid) would have depressed the probe for non-plumbing reasons. Coverage
constant confirmed at build time: T_phi = 39,630,169 lines; stride 400 x max_lines
100000 = 40,000,000 >= corpus, assertion passed with 99,076 sampled lines.

**Conditional follow-up — BPE-level ESPUM (USER 2026-08-17; direction registered, spec
deferred).** The user pins the order: phone level first; BPE/syllable-level ESPUM is IN
PLAN as the follow-up, opening only after the phone-level experiment succeeds (health
checkpoint passed and the arm gate read; a failed phone gate still returns 1f to the user
as registered — this follow-up does not soften it). Why it earns the slot: phone-level
entry 5 still consumes T_phi, i.e. a grapheme-to-phoneme dictionary on the text side —
the one lexical resource left in the training pipeline. A BPE-level ESPUM (text side =
BPE-512 learned on raw LM text) removes it: training becomes lexicon-free end to end, and
the pronunciation dictionary survives only inside eval scoring (BPE-to-phone expansion
for phone-currency PER, exactly the ruling-3 bpe512 protocol — an eval-only read, like
the labels themselves). The rate ground is already banked: BPE-512 text at 5.39
symbols/s vs the seg16 rung under its closest-rate merge prefix at 5.29/s (amendment
(7)) — a genuinely rate-matched pairing whose nulls and ceiling are already priced in the
ruling-3 artifacts; the segment spans for that operating point live in the seg16 stream
job (UnitWordStreamJob.mPnLApAbYnVG). Word-level ESPUM is NOT in the ladder: amendment
(3) stands — the 2.8 words/s target sits below every rung's attainable rate floor (best
3.459/s at seg9) and the words-cell autopsy (7a) showed that mismatch is structural.
Syllable level is the named alternative if BPE-512 disappoints, at the disclosed cost of
re-introducing the dictionary (syllabification runs through phonemicized text). Full
spec = a new registration at that time; nothing here changes the funded batch above, and
no BPE-level build starts until that registration exists. (2026-08-17 amendment:
with entry 7 registered under USER ruling 6, the opening condition transfers verbatim —
this follow-up opens on a phone-level gate PASS, now meaning entry 7 stage C, not before.)

## Entry 7 — ESPUM reproduce-then-bridge (USER ruling 6, 2026-08-17; registered pre-build)

**USER ruling 6 (2026-08-17):** "Try your best to make a PUSM-like approach work! I accept
that you even just reproduce, but some version of a (skip)ngram matching should work!"
Registered consequences: (a) the entry-5 fork consequence is OVERRIDDEN — 1f does not
close and continued (skip)gram-matching work IS funded; the entry-5 gate verdict itself
is unchanged (a failed gate licenses not funding; the user chose to fund — that is the
fork working as designed). (b) Ruling 4's TIMIT prohibition is lifted FOR REPRODUCTION:
the staged path declined on 2026-08-16 (reproduce the published setting, then swap toward
our setup) is now the funded structure; any candidate claimed as ours is still judged on
LibriSpeech. (c) The ruling-1/ruling-5 simplicity constraint YIELDS TO FIDELITY inside
entry 7 wherever the two conflict — the reference's own machinery (learned segmenter,
soft alignment, relabeling iteration) is in scope, because entry 5 measured exactly what
removing it (plus our units) does.

**Purpose.** The entry-5 autopsy localizes the failure to identity assignment, not
plumbing: the stream carries the information (supervised probe 0.3565 through the
identical path) yet the objective's optimum leaves phone identity near chance, and the
reference's own bigram-vs-full separation did not transfer to our stream. Entry 7 answers
one question: WHICH difference between our failed port and the working reference kills
the signal — by first reproducing the working point, then swapping one component at a
time toward our setup until the failure reappears.

**Approach.** Three stages, each gated before the next spends. STAGE A — REPRODUCE: run
the released GraphUnsupASR wav2vecu_graph path as shipped (env w2vu — real fairseq,
GH200-verified) on the reference bed: TIMIT, unmatched-text column (anchor PER 0.473;
0.451 after its relabeling iteration), their wav2vec2 features, k-means-128 units, their
learned segmenter and relabeling, tri-skipgram as-shipped (the two fixed pairs).
Selection: label-free (the entry-5 weighted phone-LM metric) for anything reported as
ours; ADDITIONALLY the best-PER-over-checkpoints read is taken as an oracle diagnostic —
eval-only, the same license as every oracle ceiling in this plan — because the published
anchor was selected supervised; the pair prices entry-5 deviation (v), the selection
cost, for free. STAGE B — BRIDGE (only after A passes): single-component swaps off the
reproduced point, one training run each, everything else at the reference setting:
(B1) FREEZE the segmentation — disable segmenter learning, soft alignment, and
relabeling, fixing boundaries at their initialization; this is exactly what our port did
[tests deviations (i)+(ii)]. (B2) their k-means-128 input replaced by OUR 500-way
codebook units, TIMIT audio forwarded through the frozen SAE encoder (label-free, cheap)
[tests deviation (iii)]. (B3) the BED swap: reference stack verbatim, audio = the
8,416-utterance LibriSpeech seed set (20.48 h) with their feature/cluster pipeline run on
it, text = the pinned T_phi sample [tests the extend-to-LibriSpeech bet itself; may
launch as soon as A trains stably — it does not wait on A's gate]. Read on every arm:
full-loss PER, bigram-only PER, their difference (the separation signature), and the
sub/ins/del split. Exact wiring of each swap is proposed by the implementer and ratified
pre-run as a registration addendum — the entry-5 pattern. STAGE C — TRANSPLANT (only
after B localizes): adopt the working configuration of the guilty component into the
LibriSpeech candidate and take the registered arm-gate read; if the unit stream changes,
nulls and ceiling are re-banked by the same LexFreeMatchJob protocol on the new stream
(new artifact, same construction — a stage-C registration constant, not a gate change).
Label quarantine unchanged throughout: labels never train and never select anything
reported as ours; TIMIT gold enters only eval reads.

**Experiments.** (A) TIMIT reproduction: full loss + bigram-only, one seed each to start
(the paper ran one); relabeling on the full arm only. Implementer step ZERO, before any
build: confirm TIMIT audio + phone transcripts are available/wired on this cluster — if
not, STOP and surface (licensing is a user matter, never a session workaround). (B)
B1/B2/B3, one seed each. (C) deferred to its own registration once B localizes. TIMIT
scoring uses the reference protocol (61-to-39 phone mapping) as scored — a control-bed
read, disclosed as such, never pooled with LibriSpeech numbers.

**Gate.** STAGE A (pre-registered measurement checkpoint): PASS if (a) the
oracle-diagnostic best-PER read of the full arm lands at or below 0.55 on TIMIT unmatched
— tolerance 0.077 over the 0.473 anchor, chosen to sit clearly in the full-loss regime
against the 0.716 collapse arm — AND (b) bigram-only exceeds the full arm by at least
0.10 (the published ablation separation is 0.324 on the matched column; no unmatched
ablation is published, so 0.10 is a planner-chosen conservative bar, registered here
pre-run). A failed stage A indicts our execution (env/code/data), not the method — debug
inside A; nothing proceeds to B until A passes; if A cannot be made to pass, report with
the discrepancy quantified. STAGE B carries no kill gate — pre-registered reading rule: a
swap is GUILTY if it alone drops the separation below 0.05 or raises full-loss PER by at
least 0.10 over the stage-A reproduced value; multiple guilty swaps are all reported,
and B3 failing while B1/B2 stay clean reads as "the bed, not the port". STAGE C carries
the UNCHANGED arm gate (M1/M2, 0.05 margins) against nulls banked on the stream actually
used. A stage-C fail returns 1f to the user again — with the localization in hand either
way, which is what this entry buys even on failure.

**Status.** REGISTERED 2026-08-17 pre-build, on USER ruling 6. Awaiting implementer:
TIMIT availability check (step zero), then stage A build. Datahall downtime 18-19 Aug:
build and queue now, expect launches to land after.

**Entry 7 amendment 1 (2026-08-17; replaces the stage structure above, because of the
step-zero result plus a USER clarification of ruling 6).** Step-zero result (implementer,
read-only, planner-accepted): TIMIT exists nowhere on this cluster (project/scratch trees,
HF caches, our source tree never wired it), and the w2vu env carries stock fairseq 0.12.2
with no GraphUnsupASR checkout. Surfaced to the user; USER clarification, same day:
"reproduction" means APPROACH-WISE, not corpus-wise — no TIMIT procurement, no TIMIT
wiring (ruling 4's reproduction lift above is therefore moot; ruling 4 stands whole).
Restructured entry 7: STAGE A (re-bedded) — the reference approach VERBATIM on our bed:
public GraphUnsupASR checkout added to the w2vu env (MIT license, repo verified first-hand
2026-08-17; ordinary build work), their full pipeline — wav2vec2 feature extraction,
k-means-128 units, learned segmenter, soft alignment, relabeling iteration, loss and
optimizer as shipped — run on the 8,416-utterance LibriSpeech seed audio (20.48 h); text
side = the pinned T_phi stride-400 sample, output classes the 39 phones, so eval lands in
the same currency, protocol, and scored fifth as entry 5. Arms: full loss + bigram-only,
one seed each; relabeling on the full arm only; selection label-free; PER never selects.
Pre-registered STAGE-A reading (signature-based — no published anchor exists off TIMIT):
the signature is PRESENT if bigram-only exceeds the full arm by at least 0.10 dev-other
PER (entry-5 seed noise was 0.027, so 0.10 is well clear of it); the full arm's PER is
reported against the entry-5 pick 0.8580, the banked nulls, and the 0.4148 ceiling as
CONTEXT — diagnostic reads, not the arm gate, which stays at stage C. If the signature is
PRESENT: the approach works on our bed, and localization proceeds on this bed — STAGE B
becomes (B1-prime) freeze segmentation at initialization, disabling segmenter learning,
soft alignment, and relabeling — exactly what our entry-5 port did [deviations (i)+(ii)];
(B2-prime) our 500-way codebook one-hot input in place of their feature/k-means front end
[deviation (iii)]; one seed each, guilty rule unchanged (a swap is guilty if it alone
drops the separation below 0.05 or raises full-loss PER by at least 0.10 over the stage-A
value). If the signature is ABSENT: ambiguous between our execution of their code and the
approach-on-this-bed — report and STOP; disambiguation options (a TIMIT license among
them) return to the user; no license is pursued otherwise. STAGE C unchanged (transplant
+ the registered arm gate on re-banked nulls). One disclosed line item: stage A's input
is wav2vec2 features — an external self-supervised audio model, label-free and
quarantine-compatible, but a supervision-cost line the current SAE stack does not carry;
whether a final 1f candidate may keep that front end (vs transplanting onto our units,
the default) is a stage-C question for the user if A succeeds. Wiring of every arm is
proposed by the implementer and ratified pre-run — the entry-5 addendum pattern.

**Entry 7 stage-A wiring addendum (2026-08-17, pre-launch; implementer proposal
ratified with three rulings — nothing queued when ruled).** Build ratified: the
GraphUnsupASR bundle is a fairseq user_dir overlay, not a fork; it runs under the w2vu
env's fairseq 0.12.2 via a one-symlink shim dir exposing `examples` top-level
(reversible, env untouched) plus dtw-python 1.7.5 installed --no-deps; import-level
registration of model and task verified first-hand by the implementer. Residual risk
disclosed: the README targets fairseq >= 1.0, so API drift may surface at runtime —
REQUIRED before either 40000-update launch: a short discarded smoke train (a few hundred
updates, losses finite and moving) and one w2vu_generate.py viterbi pass (the kaldi
import path); either failing returns to the planner, and the smoke run is never a read.
Front end ratified: wav2vec2 Large LV-60 (wav2vec_vox_new.pt — the checkpoint
FetchFairseqW2v2Job already pulls), transformer layer 14, 50 Hz (frame-compatible with
the battery stream by construction); faiss k-means, 128 centroids, fit on the 8,416 seed
utterances only at sample-pct 0.5 — scored-fifth rows are assigned, never fit. The
one-hot input width input_dim is COMPUTED on our data (count of centroids winning at
least one frame; the released 127 is their data, not a constant), registered sanity band
[100, 128] — outside the band, surface, never absorb. RULING 1 (initial segmentation):
option (b) — k-means cluster-change boundaries from the same CLUS128 front end, serving
as both the round-1 pooling and the BinarySegmenter target (one variable). Unsupervised,
self-contained, and the wav2vec-U lineage convention this bundle builds on. Option (a),
porting the two external TIMIT-pretrained segmenter repos, is REJECTED — it would import
a TIMIT-fitted component into a LibriSpeech arm, and the reference's own design treats
the initializer as a starting point that relabeling replaces. Option (c), our Ward
seg12.5 spans, is not stage A but stays in the bridge: with (b) as the init, B1-prime
becomes "freeze at the (b) boundaries; segmenter learning, soft alignment, relabeling
off", and a B1b arm — freeze at our Ward seg12.5 spans, the literal entry-5
configuration — is PRE-AUTHORIZED as the follow-on if B1-prime alone is not guilty;
stage-B wiring still ratifies pre-run after the stage-A read. Boundary rate per second
of the (b) init is reported as context (vs the 9.86/s phone rate), no gate on it.
RULING 2 (silence): silence-STRIPPED — the frame-mask variant on the battery's
kept-frame indices (SeedUtt.frame_index). This keeps PER in entry-5/null/ceiling
currency AND matches the reference's own without_silence TIMIT manifests, so it is
fidelity-consistent, not merely currency-driven; that the segmenter head never sees
silence is the disclosed residue. RULING 3 (signature clause, amended pre-result —
replaces amendment 1's clause because, as written there, it confounded loss with
relabeling): the registered signature read is MATCHED-ITERATION — bigram-only at
relabeling iteration 1 vs full loss at iteration 1, bar unchanged at 0.10 dev-other
PER; full loss at iteration 3 (the reference's operating point) is reported alongside,
and the full arm's iteration-3-minus-iteration-1 delta is the free read on deviation
(ii), what relabeling buys. Eval path ratified: w2vu_generate viterbi on the scored
fifth into the same EspumEvalJob scorer (sub/ins/del decompose identically); selection
label-free on the other four fifths, unchanged; PER never selects, at any stage.
Hyperparameters as shipped off config/l1 with the run.sh CLI overrides, per the
proposal's item 8.

**Entry 7 decode-path blocker ruling (2026-08-17, pre-launch; planner-diagnosed
first-hand, before the implementer's smoke check reached it).** Finding: the ratified
eval decode (w2vu_generate.py viterbi) cannot run in the w2vu env — the flashlight
python bindings are absent (ModuleNotFoundError verified), and fairseq 0.12.2's
W2lViterbiDecoder.decode hard-requires flashlight's CpuViterbiPath at decode time (the
guarded import at w2l_decoder.py:28-39 only warns, so every import-level check passes
and the failure surfaces as a NameError mid-decode). The kaldi-import risk flagged
earlier is RETIRED without action: the task module itself imports KaldiDecoderConfig at
top level (unpaired_audio_text.py:29), so the implementer's verified task registration
already crossed it. RULING — pure-torch equivalence shim, exact by construction: in
this fairseq version W2lDecoder sets asg_transitions = None unconditionally, so
W2lViterbiDecoder always calls CpuViterbiPath with an ALL-ZERO transition matrix, under
which the maximum-score path decouples across frames and equals per-frame argmax
(identical up to float ties, a measure-zero event). The already-ratified shim dir gains
a pure-python flashlight package: lib/sequence/criterion.py implementing
get_workspace_size (0), get_data_ptr_as_bytes (returns the tensor — the pair is
internally consistent and w2l_decoder passes both opaquely), and CpuViterbiPath.compute
as viterbi_path.copy_(emissions.argmax(-1)) with a REQUIRED guard asserting the
transitions tensor is all zero (if any caller ever passes nonzero transitions, raise —
never silently approximate); lib/text/{dictionary,decoder}.py as stubs that raise on
use (only the KenLM/FairseqLM decoders touch them; stage A never calls those). Required
unit test before launch: shim output equals brute-force zero-transition dynamic
programming on random emission tensors. Everything downstream of the solver —
get_tokens groupby-collapse, blank filtering, post-processing, w2vu_generate.py itself
— runs reference-verbatim and unmodified, which is why this beats the alternatives:
building flashlight from source on aarch64 into the known-fragile w2vu env is REJECTED
as risk without benefit (the replaced computation is an identity), and the repo's own
espum_greedy_generate_universal.py is REJECTED because it instantiates the espum module
excluded by registration addendum (a). Smoke check (ii) now must pass THROUGH the shim
end to end.

Status 2026-08-17 (late): ruling IMPLEMENTED and planner-verified first-hand. Shim logic
lives versioned in the repo (speech_llm/sae/flashlight_shim.py) with the ratified shim dir
holding only generated one-line re-exports from a committed builder — a deviation from the
ruling's letter, RATIFIED (identical import surface, better provenance). Two additions the
guarded import block forces, both source-verified: CriterionType.CTC is read in
W2lDecoder.__init__ for every decoder (opaque marker, never re-read on the viterbi path),
and LM must be a real class (FairseqLM subclasses it at module level); KenLM/Trie/lexicon
stubs raise on use. Required unit test passed and independently re-run by the planner under
the toolkit python (brute-force zero-transition DP match on four shapes incl. N=128;
non-zero transitions raise); guarded import completes under warnings-as-errors with all
three names resolving to the shim. Two pre-launch implementer fixes verified, hashes
unchanged: requires_env="w2vu" added to the five toolkit-spawning job classes (root cause
of the libtorch undefined-symbol crash — the py3.9 child inherited the main env's
LD_LIBRARY_PATH), and faiss k-means moved to a subprocess under the toolkit python (faiss
exists only there; the width-band assert raises inside the child). Errored prep jobs
recovered by marker rename, re-running past the old failure point. Awaiting the two smoke
checks; commit lands after they pass.

Status 2026-08-17 (later still): third environment incompatibility, fixed and verified —
torch 2.6 in the toolkit env defaults torch.load to weights-only unpickling, which rejects
the 2020 wav2vec2 Large LV-60 checkpoint's embedded argparse.Namespace (fairseq 0.12.2
passes no flag). Fix RATIFIED: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 in the module's toolkit
env helper — the identical idiom already registered in this repo for this same checkpoint
family (w2vu2/selftrain.py _torch_unsafe_load_env), trust scope = the official fairseq
download plus this graph's own checkpoints, never third-party artifacts; run-time env only,
hashes unchanged. Text-side prep FINISHED: 99,076 sampled lines, 39 plain-ARPAbet phone
types, fairseq preprocess clean. Risk note for the remaining passes: three distinct
bundle-vs-cluster incompatibilities have surfaced so far (interpreter/library inheritance,
single-env package, load default), all as hard failures; budget for more inside the not-yet-
run training and generation paths — which is what the two smoke checks are for.

Status 2026-08-18: data prep DONE, smoke (i) PASSED, smoke (ii) surfaced a labels
dependency in the released decode script — planner-verified and ruled. Prep reads:
silence masking keeps 0.735/0.715/0.713 of frames (train/valid/test); k-means one-hot
width = 128, INSIDE the registered [100, 128] band at its top edge (every centroid wins a
frame — the released 127 was data, as registered); k-means change-point boundary rate
27.9-28.6/s vs the 9.86/s phone rate, i.e. a ~2.9x over-segmented proposal set, which is
the reference's design (the learned segmenter prunes; reported, no gate). Smoke (i): 400
updates, all losses finite and falling (loss 1835->1132, segment 407->56), checkpoint
write+read exercised end to end; gradient clipping active at every logged step —
watch-item for the full arms, not a blocker. Smoke (ii) finding, source-verified: the
released generation script reads a gold phone segmentation UNCONDITIONALLY
(w2vu_generate.py:570 gt_bin_labels; the dataset fills it from <split>_gt.src, which the
reference's prep copies from TIMIT's gold alignments), so fairseq.task.labels=null does
NOT make the decode label-free — this amends the decode-path ruling's premise that
w2vu_generate.py runs unmodified, by replacement: it runs unmodified EXCEPT a three-hunk
conditional-gold patch applied to a run-time COPY (bundle checkout pristine; every anchor
must match exactly once or the job fails loudly; the patched copy is kept beside the
results as an auditable output). RATIFIED: hunks make the gold read None-safe using the
same [None]*len idiom the reference's own task module already uses for this exact field
(unpaired_audio_text.py:235); behavior with gold present is identical (planner-compared
line by line); predicted boundaries — what relabeling pools on — untouched and written
for every utterance. ALSO RATIFIED: pred_b_len counts only alongside gold, so without
gold both boundary-metric blocks are skipped entirely rather than logging a structurally
meaningless 0.0 precision — silence over a fabricated number, per the standing
impossible-scores principle. Net quarantine effect: the patch REMOVES a label read; the
decode on our bed is label-free by construction. Full arms remain gated on smoke (ii)
actually passing (re-queued through the patch), whose report carries the blank-index and
output-dim/dictionary-specials reads.

Status 2026-08-18 (later): BOTH SMOKES PASSED, planner-verified against artifacts; the two
stage-A arms are QUEUED under the standing clearance. Smoke (ii) through the patched decode:
572 hypothesis lines written, gold-side files all zero bytes (conditional fired as ratified,
no gold read or written), predicted-boundary file complete, no boundary metrics logged, WER
None. Wiring reads, off the loaded task not assumed: dictionary length 43 = 4 fairseq
specials (bos <s> 0, pad 1, eos </s> 2, unk 3) + our 39 plain-ARPAbet phones; no
<ctc_blank>, so the decoder's blank = the bos fallback (index 0) and silence = the eos
fallback; planner re-confirmed the generator's output projection is (43, 128, 4) — the only
parameter of width 43 — and that ZERO special symbols appear anywhere in the 572 decoded
lines. The logged LM_PPL inf is the script's inactive internal LM scorer (lm_model None),
not a failed read. Arms (labels off each job's own name field): GuaTrainJob.PZo12D74ij2M =
entry7_full_iter1 (loss_variant full) and GuaTrainJob.OfNoESzNJykY = entry7_bigram_only_iter1
(loss_variant bigram_only), 40000 updates each, one seed, relabeling full arm only; all six
prep/smoke hashes unchanged. Known operating facts: measured smoke pace puts a full arm at
~12.7 h vs the 11.5 h engine cap, so each arm resumes at least once (checkpoint_last.pt
every 2000 updates; at most ~2000 updates redone per interruption), and the 18-19 Aug
datahall downtime lands mid-run (outage error markers renamed, never cleared). Implementer
caught pre-launch that add_alias does not register a job in the sisyphus graph — both arms
were initially built unreachable; fixed by registering each arm's checkpoint output
(recurring trap, now in shared memory). Standing caution while arms are live: a resume
re-executes the job's run() under the CURRENT recipe source, so the training-command
construction in gua_jobs.py is frozen until both arms finish; new job classes may be added
alongside. Next read: stage-A signature at matched iteration per RULING 3.

Status 2026-08-18 (evening): both arms HALTED at update 12000 of 40000 (5h40m in) —
planner-verified cause is the shared-fileset file-count quota (OSError Errno 122 on the
checkpoint temp-file write, both arms, the registered EDQUOT failure mode), NOT code and
NOT the bundle. checkpoint_last.pt at update 12000 is intact and loadable for both arms
(planner re-loaded both); jobs resume from it, so at most the interval since the last save
is redone. The quota itself is a USER-owned resource decision (raise, or approve payload
reclaim); the signature read slips by the outage plus ~8h of remaining compute per arm.

Pre-result registration 2026-08-18 (before any stage-A PER exists) — clipping-regime
confound, implementer-surfaced, planner-ruled. Verified trace: with the released
clip_norm 20.0 and flat lr 0.004 on both arms, the full-loss arm is gradient-clipped on
100 percent of updates at every logged point through 13000 with gnorm RISING 116 -> 2187
(two orders of magnitude above threshold), while bigram-only settles at 93-98 percent
clipped with gnorm in the tens-to-hundreds (one single-step spike to 6062 at update
10000, an outlier either side of which is ~200). The full arm's effective step size is
therefore set by the clip, not the lr. RULING: (1) NO config change — clip_norm 20.0 is
the released constant applied identically to both arms; touching it would be an
untraceable design constant and break the reproduction frame. (2) The regime difference
is ENDOGENOUS to the loss contrast (the extra terms produce the larger gradients), so it
is part of the treatment being reproduced, not an external confound — but its effect on
the gate is ASYMMETRIC, and that asymmetry is now the registered reading: a PRESENT
signature (full beats bigram-only by >= 0.10) survives the confound a fortiori, because
always-clipping can plausibly only shrink the full arm's effective training, not
fabricate an advantage for it; an ABSENT signature gains a named execution-side mechanism
(full arm undertrained at clip-normalised steps, lr near inert) and remains exactly what
RULING 3 already declares it — ambiguous between execution and approach-on-this-bed,
report and STOP. Whether the reference's own runs also sat at 100 percent clipping on
their bed is unknowable from the release (no gnorm logs shipped), which is precisely why
ABSENT cannot be read as approach failure. (3) DIAGNOSTIC ADOPTED: the stage-A signature
table must carry the clip fraction and gnorm alongside each arm's PER at every read
point, so the regime is visible in the same table as the number it threatens. (4) If the
ABSENT fork is reached, one priced option for the user alongside the TIMIT license: a
single diagnostic re-run of the full arm at a raised clip ceiling, explicitly off-
reference, to separate regime from approach — registered as an option only, never run
inside stage A.

Clipping-regime ruling, MECHANISM CORRECTED 2026-08-18 (planner, still pre-result; replaces the
stated CAUSE in clause (2) and nothing else — clauses (1) NO CONFIG CHANGE, (3) DIAGNOSTIC ADOPTED
and (4) PRICED OPTION stand, and the ASYMMETRY CONCLUSION stands too). The original reading recorded
the regime difference as endogenous to the loss contrast, "the extra terms produce the larger
gradients", off a mid-run reading through update 13000. The whole-run logs falsify that cause:
bigram-only's mean gradient norm is LARGER, 8146 against full-loss 4210 whole-run and 25806 against
9054 in the last tenth. The correct mechanism is the clip FRACTION, not the gradient magnitude — the
full arm is fully clipped on 100.0 percent of epochs while bigram-only is fully clipped on 55.4
percent, and an epoch that is not fully clipped contributes unscaled updates, so at clip_norm 20.0
and a constant learning rate 0.004 the mean applied step is about 6.8e-05 for full-loss against
2.9e-04 for bigram-only whole-run (3.9e-05 against 7.4e-05 in the last tenth). THE FULL ARM TAKES
THE SMALLER EFFECTIVE STEP, by roughly 4x whole-run.
SUPERSEDED FORM, kept as dated history, planner error: an earlier amendment the same day read the
gradient-norm magnitudes alone, concluded "the two arms sit in the SAME clip-dominated regime", and
therefore withdrew the asymmetry — that inference used the wrong quantity (gradient norm rather than
applied step) and its conclusion is retracted.
Reading, unchanged in direction and now quantified: (a) a PRESENT signature survives a fortiori and
is STRENGTHENED — the arm that wins does so while applying roughly a quarter of the control's
parameter movement per update; (b) an ABSENT signature retains exactly the execution-side mechanism
clause (2) named — the full arm undertrained at clip-normalised steps — because it is the full arm,
the one expected to win, that took the smaller steps, which is the textbook shape of a real
difference being MASKED rather than fabricated; ABSENT therefore stays as ambiguous as RULING 3
declares, and clause (4)'s priced option (one diagnostic re-run of the full arm at a raised clip
ceiling, explicitly off-reference) becomes MORE relevant if ABSENT fires, not less; (c) the signature
table carries clip fraction, gradient norm AND applied step per arm at every read point — and the
free finding both arms share is an unintended DECAYING step size under a constant declared learning
rate (full-loss falls 1.8x from whole-run to last tenth, bigram-only 4.0x), a property of the
reference recipe as run rather than of either loss. Two measurement caveats owned by the implementer
and accepted: train_gnorm and train_clip are per-EPOCH aggregates, so the applied step is an
approximation of a per-update mean and the 4x RATIO is the less reliable of the two figures (its
DIRECTION is robust — bigram-only's roughly 4 percent unclipped updates alone put its mean scale
above the full arm's measured 0.0170); and the full-loss arm was still training at update 39312 of
40000, so every figure is recomputed at the read with the identical estimator on both arms.
Registered before any stage-A PER exists.

Clipping-regime ruling, WINDOW CORRECTION 2026-08-18 (implementer-found, planner-accepted; this
replaces the EVIDENCE cited in the mechanism correction above and confirms the correction it drew).
The whole-run gradient figures quoted there were computed over windows that DIFFER between arms — the
timeout resubmit overwrote the full arm's middle log segment, leaving 14..13,986 and 39,214..40,000 on
disk — so a mean taken across them averaged over a sign flip. Those whole-run figures are WITHDRAWN as
evidence. The citable read is the recomputation on the two windows both arms cover, one estimator,
999 and 58 epochs each:

| window | arm | gradient norm | epochs fully clipped | applied step |
|---|---|---|---|---|
| 14..13,986 | full loss | 1,617 | **100.0 %** | 1.07e-04 |
| 14..13,986 | bigram only | 1,256 | 50.7 % | **5.59e-04** |
| 39,214..40,000 | full loss | 8,981 | **100.0 %** | 3.64e-05 |
| 39,214..40,000 | bigram only | 20,395 | 60.3 % | **8.11e-05** |

Three consequences, and they separate cleanly. **(i) Clause (2)'s original premise is not falsified; it
is window-dependent.** The gradient-norm ordering FLIPS — the full arm's gradients are larger early and
smaller late — so "the extra terms produce the larger gradients" holds in the early window and reverses
in the late one. The mechanism correction above claimed the whole-run logs falsified that premise; that
claim rested on the mismatched windows and is withdrawn. **(ii) The correction it drew survives intact
and is now measured twice on matched windows.** The APPLIED-STEP ordering is the same in both windows —
bigram-only larger by 5.2x early and 2.2x late — and it is driven by clip FRACTION, not gradient
magnitude. The full arm takes the smaller effective step throughout, which is the reading the asymmetry
conclusion rests on, so that conclusion stands and its evidence is now stronger than what it replaced.
**(iii) A new fact that bears directly on clause (4).** The full arm is at 100.0 % clip fraction in BOTH
windows: it never received a single unclipped update anywhere in 40,000 updates, early or late. Clause
(4)'s priced diagnostic re-run at a raised clip ceiling therefore separates regime from approach more
sharply than when it was registered, and the decaying-step finding is sharpened with it — the declared
constant learning rate 0.004 in fact yields an applied step falling 2.9x (full) and 6.9x (bigram-only)
between the two windows, a property of the reference recipe as run rather than of either loss.

Status 2026-08-18 — STAGE-A SIGNATURE VERDICT (dev-other, scored fifth, 572 utterances, 34,135
reference phones; GraphUnsupASR as published on the seed bed, iteration 1, checkpoint pinned label-free
over the 18-point grid). **Against RULING 3's registered bar the signature is ABSENT and its sign is
REVERSED: bigram-only minus full-loss = -0.4394 against a bar of at least +0.10.**

| arm | picked | PER | sub | ins | del | phones/utt | swap M2 |
|---|---|---|---|---|---|---|---|
| full loss | 2,000 | 1.6843 | 0.6046 | 1.0766 | 0.0031 | 123.7 | +0.0013 |
| bigram only | 30,000 | 1.2449 | 0.6836 | 0.5547 | 0.0065 | 92.4 | +0.0079 |
| reference | — | — | — | — | — | 59.7 | — |

**Planner ruling: ABSENT is registered as the verdict against the bar, and it carries LESS than RULING
3 anticipated — not the execution-versus-approach ambiguity, but a third possibility RULING 3 did not
foresee, that NEITHER ARM PRODUCED A DECODE CAPABLE OF HOSTING THE COMPARISON.** Three independent
indicators, all in the table above. (1) Both PERs exceed 1.0: the edit distance is larger than the
reference itself, because both arms over-generate — hypotheses run 2.07x and 1.55x reference length.
The internal consistency check passes on both rows and is how that was verified rather than assumed:
insertions minus deletions equals the length ratio minus one, 1.0735 and 0.5482, matching 123.7/59.7 =
2.072 and 92.4/59.7 = 1.548. (2) Both arms fail BOTH registered arm-gate margins by wide margins: M1
needs at most 0.8446 and M2 needs a rise of at least 0.05 under the audio swap, measured at +0.0013 and
+0.0079. Neither arm shows measurable utterance-specific information. (3) The label-free selector spans
38.16 to 41.15 across all eighteen checkpoints — 7.8 % — and its minimum is the FIRST checkpoint, so
the two arms were pinned 28,000 updates apart by a metric flat across the whole run. A selector that
cannot distinguish any checkpoint is what a run whose quality does not vary produces. **The consequence
for scope: this verdict licenses "the stage-A signature question is not answerable at this operating
point", and it licenses neither "the extra loss terms do nothing" nor any statement about the approach
on this bed.**

**Correction to one reading offered with the submission, because it changes the inference.** The
substitution-only ordering (full 0.6046 against bigram-only 0.6836) does NOT restore the predicted
direction; it selects the length-favouring metric. With deletions near zero, recall against the
reference is 1 - sub - del = 0.3923 (full) against 0.3099 (bigram-only), favouring full; precision per
EMITTED token is 0.3923/2.0735 = 0.1892 against 0.3099/1.5482 = 0.2002, favouring bigram-only. Both
orderings are artifacts of the 2.07x-versus-1.55x length ratio, and both precision figures are within
reach of what matching the phone marginal alone would produce, which is consistent with the flat swap
control. Registered limit, because this program has just adopted the statistic elsewhere: the
insertion-forgiven error (substitutions plus deletions) is valid for ranking REPRESENTATIONS at a
matched output policy, where the length ratio is the property being priced; it is NOT valid for ranging
ARMS that over-generate by different factors, where it is precisely the statistic an over-generating
arm games. The headline stays the plain PER as scored.

**Next read, ruled rather than left open: FIXED-ENDPOINT DECODE at update 40,000 on both arms**, a grid
point declared in advance so no label enters selection, which removes the 2,000-versus-30,000 confound
for one decode per arm. Pre-registered so it cannot become a pick-the-better: the fixed endpoint is
adopted as this arm's reporting policy REGARDLESS of which way it comes out, and if it is worse it is
still what gets reported. Pre-registered interpretability condition, written before the number exists:
the signature question becomes answerable only if at least one arm clears M1 with a non-flat swap
control; if both arms again land above the nulls with a flat swap, stage A closes as NOT ANSWERABLE on
this bed and no further spend is licensed against it. The clip-ceiling re-run of clause (4) is NOT
recommended ahead of that number — it is off-reference and priced, it is the USER's call, and it needs
the fixed-endpoint read as its matched baseline — but the 100.0 %-in-both-windows finding above makes
its case stronger than when it was registered.

Status 2026-08-19 — STAGE A CLOSED, NOT ANSWERABLE, on the interpretability condition registered
2026-08-18 before the number existed. Fixed-endpoint decode at update 40,000, both arms, checkpoint
pinned by a declared-in-advance integer with no metric and no reference in the pinning job: full loss
1.6828 (sub 0.6153, ins 1.0650, del 0.0025, 70,404 hypothesis against 34,135 reference phones, swap
+0.0039); bigram-only 1.2409 (sub 0.6811, ins 0.5535, del 0.0062, 52,816 against 34,135, swap +0.0107).
**Two readings, and they separate cleanly.** (i) The signature is -0.4419 against the registered +0.10,
where the label-free-selected checkpoints gave -0.4394: the 28,000-update spread is NOT what produced
the sign, so the selector objection registered with the first read is RETIRED — it was a real defect in
the read and it was not the cause. (ii) Stage A nonetheless closes NOT ANSWERABLE, for the other reason
and exactly as pre-registered: both arms sit far above the 0.8446 margin and both audio-swap controls
are flat, so this is a contrast between two uninformative decodes rather than between two losses. The
condition was written before either number existed and is applied as written; no further spend is
licensed against stage A, and clause (4)'s clip-ceiling re-run remains a USER option rather than a
recommendation. Consequence carried elsewhere: `PLAN_1G.md` 1g.6's FREE READ 2 stays SUSPENDED, since
the relabeling iterations it would read were seeded from an iteration-1 checkpoint that fails both
margins, and `GuaTrainJob.EwdQgD4XqYPI` now has no registered consumer at all.

Status 2026-08-18 (recognition-chain wiring addendum — implementer proposal ratified pre-run
with one amendment; commit 4e7550b planner-verified; the jobs are deliberately NOT in the
running graph until the user restarts the entry-7 manager). Chain per arm: GuaDecodeSweepJob
(selection four-fifths, every grid checkpoint) -> GuaSelectJob (entry 5's weighted phone-LM
perplexity, argmin; the committed implementation reproduces the banked entry-5 pick to
<= 1.2e-4 relative, winner exact — SAE_1f.md 2026-08-18) -> GuaGenerateJob (scored fifth,
decoded once with the pinned checkpoint) -> GuaScoreJob (PER + sub/ins/del + the screens'
nearest-length-partner audio-swap re-pairing; the only class that opens reference phones,
strictly downstream of the pin). Full arm additionally: the reference's two relabeling
passes -> full_iter2/full_iter3, each with its own recognition — the RULING-3 matched-
iteration read and the iteration-3 operating point both exist by construction. RULINGS:
(a) selection grid = the checkpoints that exist. CORRECTED 2026-08-18 (implementer-found,
planner-verified from both arms' checkpoint listings; REPLACES the 19-point form below and
its stated cause, because that cause was the coincident symptom rather than the mechanism):
the grid is **18 points** — 2000..40000 minus 14000 AND 28000 — for the ITERATION-1 arms
only. The holes are DETERMINISTIC, not the quota: at ~14 updates per epoch every update
divisible by both the 2000-update save interval and the per-epoch count lands exactly on an
epoch end, where the trainer writes only `checkpoint_last` (no_epoch_checkpoints), so the
numbered snapshot is never created. That predicts holes at 14000 and 28000 and none other
below 40000; both arms' listings confirm exactly those two, and 28000 was lost with no quota
event anywhere near it. SUPERSEDED FORM, kept as dated history: "19 points, 2000..40000 minus
14000, because the EDQUOT outage permanently lost the update-14000 checkpoint" — the outage
did strike at 14000, which is precisely what made a one-off cause look sufficient; a rule
that recurs was read off a single symptom, and the 19-point grid would have crashed at 28000.
The hole is symmetric across arms so selection stays arm-fair; iteration-2/3 arms keep the
full 20-point grid unless their own epoch arithmetic says otherwise, which must be checked
rather than assumed. The absent update-0 read is a construction fact (no checkpoint exists),
disclosed. (b) Relabeling alignment margins 0.1 then 0.0 = the reference's own (run.sh
stages 2/4) — ratified. (c) Scored decodes at margin 0.0 in BOTH arms — ratified; proven at
source that margin reaches only segment outputs (w2vu_generate.py:330-333, :569), never the
phone hypotheses. (d) Relabeling aligns off the label-free pick — the one run.sh departure
in that chain, forced by the registered mandatory label-free selection (the released
checkpoint_best is error-rate-selected, quarantine-incompatible) — ratified and disclosed.
(e) The selection metric's magnitude is never compared to entry 5's (different decoder);
within-arm checkpoint comparison only — ratified per the standing magnitudes-are-per-arm
rule. Reporting duties carried forward: the stage-A signature table names the update-14000
hole wherever a selection curve is shown, and carries clip fraction and gnorm beside every
PER (live logs 2026-08-18 14:42: full arm gnorm ~2332 at 100 % clip, bigram-only ~593 at
95-100 % — the registered confound is live).

## Screen battery (prerequisites (i)+(ii) made operational; the first fundable step)

Run per REPRESENTATION of the same enc50 stream — raw, run-length-deduped, segment-pooled
(change-point), Brown-merged K=100 (already dumped), unit-BPE-pooled — so the representation
every matcher consumes is chosen by measurement, not argument:
- Prerequisite (i) per representation: oracle-map PER and H(phone|unit) via the 0a audit
  machinery (repr_audit.py / real_repr_probe.py, W2v2ReprAuditJob pattern). No enc50 number
  exists anywhere yet; this produces the first.
- Prerequisite (ii) per representation: correlation of the unit co-occurrence graph with the
  text phone-bigram graph. Known endpoints calibrate the bar: 0.97 simulated = recoverable,
  0.146 raw real = fatal. CAVEAT: the 0.146 came from an uncommitted scratchpad script
  (SAE_1a.md verifier note) — commit before any registration cites it.
- Extensions (same CPU pass, added 2026-08-16): sigma_min of the positional-unigram matrix
  P_X (gates ladder entry 2); Laplacian eigen-similarity of the unit-vs-phone co-occurrence
  kNN graphs (Sogaard et al. ACL 2018; screen-to-accuracy spearman 0.89 in the bilingual
  lexicon literature); normalized eigenvalue-spectrum overlay (does the 39-phone spectrum
  embed in the top of the unit spectrum). Calibrate every screen on the existing
  simulated-unit generator at controlled correlation levels before reading real numbers.

Kill readings: enc50 oracle-map PER >= ~0.53 kills unit-level mapping (arm moves to
feature/segment level, ladder entry 5, per the registered condition); best-representation
graph correlation below the simulation-calibrated turn-on kills the second-order matchers
(ladder entries 1 and 4); sigma_min(P_X) at double-precision noise kills ladder entry 2.

## Candidate ladder (each entry conditioned on the screen; kill-test before any spend)

1. **Fixed-core tri-factorization (method-of-moments; NOT EM).** Fit only the emission matrix
   M (39 phones x 500 units, column-stochastic) so that M-transpose Theta M reproduces the
   deduped unit co-occurrence matrix Omega, with the phone-bigram core Theta PINNED from the
   text corpus, never estimated from audio. Alternating NNLS / multiplicative updates on KL;
   optional anchor-unit identifiability route via separable NMF (greedy pivoted QR finds
   units emitted by a single phone, constrained regression fills the rest). CPU minutes,
   batch-free, one hyperparameter. Soft M escapes the deterministic-map ceiling; H(phone|unit)
   still binds. Free kill-test: numerical rank of Omega must clearly reach 39; then anchor
   phone-purity (labels eval-only) >= ~60 percent. Main risk: coarticulation violates the
   memoryless-emission assumption — same channel class as the closed 1a lane (different
   estimator, same misspecification risk); the screen is the pre-registered protection.
   [Huang/Fu/Sidiropoulos arXiv:1802.06894 verified first-hand, incl. second-order beating
   third-order at every sample size; the anchor-NMF transplant to speech is novel.]
2. **Ridge-regularized positional-unigram least squares.** One BLAS solve: O minimizing
   ||P_X O - P_Y||^2 plus ridge, rows projected to the simplex; positional unigrams of the
   deduped stream at silence-anchored bins vs phones from T_phi. This is the closed-form
   estimator of arXiv:2306.07926 — never run on real speech by anyone, so also the cheapest
   genuinely novel result available. Gated on sigma_min(P_X): the paper's own synthetic
   values (1e-31..1e-15) are numerically dead, and ridge breaks the uniqueness theorem, so
   pre-register the regularization path and validate the phase transition on simulated units
   first. Conditioning degrades exponentially in inventory size — favours small pooled
   inventories (Brown K=100 stream already dumped).
3. **Unary fingerprint assignment + count bootstrap (flicker-immune seed).** Per-unit
   transition-free fingerprints — frequency, utterance-initial/final rates, pre/post-silence
   adjacency, run-length mean/variance (duration proxy), position histogram — matched to the
   same phone statistics from text by one many-to-one assignment solve; optional growth by
   mutual-nearest-neighbour hard recounting with a CSLS-style margin (VecMap evidence: a
   0.52 percent seed bootstraps to 48 percent, and removing the margin collapses to zero).
   Uses NO transitions; consecutive repeats become signal (duration). Delivers manner-class
   grade at best — positioned as the seed for entries 1/2/4, never the solution. Kill-tests:
   fingerprints must separate manner classes (eval-only decision tree >= ~50 percent);
   bootstrap needs >= ~70 percent admitted-pair precision from a ten-pair diagnostic seed.
   Inherited warning (the 1c p_sil lesson): single marginals mislead — match the whole
   vector, never one marginal.
4. **Screen-gated entropic Gromov-Wasserstein on the two co-occurrence matrices.** Frequency
   marginals (uniform marginals are wrong under many-to-one homophony), fused variant adds
   the fingerprint cost, mandatory random-permutation null on the objective. Run ONLY if the
   screen shows dedup/pooling lifted the graph correlation into the simulation-calibrated
   recoverable band — published evidence is brutal at raw 0.146 (unseeded graph matching at
   or below the information-theoretic threshold; the bilingual analogue reads precision ~0).
   Unique asset: the GW objective tracks accuracy without labels — a label-free selection
   signal this project otherwise lacks (likelihood measured anti-aligned with PER).
5. **Segment/feature-level ESPUM (the registered 1f form; the ceiling escape; least simple).**
   One-layer CNN over pooled segments, positional-unigram + 2/3-skipgram L1 (nothing above
   N=3 — 4/5-gram terms measured catastrophic), single consumer GPU; its measured selling
   point over the excluded GAN is stability, and its honest accuracy anchor is ~2.4x worse
   PER than the GAN on TIMIT — a usability-not-superiority play by construction. Plumbing
   gate exists (continuous_gan.py --mode sup reproduces the 0a probe, PER 0.145). Health
   signature: a bigram-only arm collapses while uni+bi+tri does not; if uni+bi+tri also
   collapses on our segments, the boundary source is the poison — re-segment or kill.
6. **Unit-BPE lexicon-head matching (novel; word granularity; structural flicker defense).**
   BPE-merge the deduped silence-delimited stream into unit-words (vocab size calibrated on
   Zipf slope and type-token curve); match the top ~200 unit-words to frequent text words on
   rank-frequency, length-in-units vs length-in-phones, and positional/silence profiles;
   consistency-filter by shared-substring agreement; accepted pairs seed entries 1-4 as
   unit-to-phone-string alignments. Kill-test: some high-frequency silence-adjacent
   unit-word must show a function-word signature (very high utterance-initial rate,
   near-zero final rate, short); none means the segmentation is not word-like — stop.
   Supported by the verified coarse-granularity ablation of arXiv:2510.03639.

## Reference verdicts (2026-08-16, first-hand; resolves the verify-flags of 2026-08-12)

- **arXiv:2510.03639 (syllable-level, "SylCipher"):** REAL. Its granularity ablation stands
  (syllable/coarse units beat phone- and char-level rows everywhere; unsupervised Sylber
  boundaries near forced-alignment quality) — coarse granularity ADOPTED as 1f's default
  target unit. Its pipeline REJECTED: the statistics-matching stage fails from random init
  and works only as a refiner on a three-stage masked-LM with full-corpus batches; 21.8 CER
  is the matched setting (35.9 unmatched) and trails G2P toplines 2.6x. Do not cite it as
  evidence that pure statistics matching can bootstrap — at raw granularity its own text is
  evidence against.
- **arXiv:2306.07926 (Wang et al., ACL 2023):** REAL, theorems sound — the best available
  theory for why positional unigrams are load-bearing. The estimator was never run on real
  speech; its own Fig. 4 puts sigma_min(P_X) at double-precision noise; its limitations
  concede the theory assumes quantization lost nothing (measured false on the old
  inventory). Enters the ladder only as entry 2, ridge-regularized and sigma_min-gated.
- **Evidence-scope note for entry 5 (2026-08-16, USER prompted; verified first-hand):** no
  PUSM/ESPUM-family method has a published result on unrestricted LibriSpeech. ESPUM
  (2310.02382) is TIMIT-only at phone level (the 2.4x anchor); the word-level PUSM loss
  appears only on curated fixed-vocabulary corpora with low-frequency words removed; the
  20-23 percent WER in that line (2406.08380) is vocabulary-size-dependent, on curated
  data, and belongs to the already-discarded JSTTI system class. Funding entry 5 is
  therefore an extend-TIMIT-to-LibriSpeech research bet, not an anchor-backed
  reproduction — raises the funding bar on top of ruling 1's last place.
- **arXiv:2603.02285 (Yang et al., ICASSP 2026):** REAL. TAKE the full-column-rank
  identifiability condition — their sigma_min(P_C) ~ 3e-4 on LibriSpeech transcriptions is
  the first real-corpus confirmation that the TEXT side is identifiable — and the proof that
  the memoryless per-position channel assumption is NECESSARY: raw-frame flicker violates
  that necessary condition, the strongest theoretical justification for computing every
  matched statistic on deduped/pooled streams. REJECT the proposed training loss: it is the
  1a fertility-HMM decipherment likelihood in gradient form — adopting it would reopen the
  permanently closed lane through the back door.

## Discard record (why the rest of the 28 died)

- Homophonic-cipher search (hill-climbing/annealing, Nuhn beam): scoring assumes a
  length-preserving image of the plaintext; flicker insertions/repeats break exactly that,
  and the measured beam fragility on CLEAN ciphers seals it.
- Third-moment tensor methods (tensor CPD, spectral HMM): consume the statistic flicker
  corrupts worst (one spurious repeat shifts a whole triple); lose the published
  head-to-head to second-order at every sample size; spectral HMM never recovers the
  emission matrix (improper learning).
- Embedding-isometry family (VecMap sorted-similarity seed, Wasserstein-Procrustes,
  cycle-ICP): the controlled non-isomorphism ladder in the bilingual literature kills them
  below our operating point, and many-to-one homophony breaks the permutation
  parameterization; only the CSLS margin idea survives, inside ladder entry 3.
- Empirical-ODM (both published forms): the 50k-token batch schedule is the anti-simplicity
  datapoint; the SPDG small-batch fix is min-max SGD never shown on speech. Transferable
  lessons absorbed into entry 5.
- JSTTI: masked-LM transformer training, not fixed-statistic matching; headline corpus was
  curated with forced alignments.
- Sinkhorn multi-statistic matching, Umeyama spectral matching as standalones: dominated —
  their live parts are already ladder entries 1/2/4 and the screen battery.
