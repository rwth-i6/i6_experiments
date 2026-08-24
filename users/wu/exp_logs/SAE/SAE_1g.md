# SAE_1g — Evidence for a simple weak SAE initialization

## State
<!-- Overwritten in place, never appended; deleted at phase close. In-flight runs (job dir + the
question each answers), blockers, next action, proposals for the planner. -->

State as of 2026-08-24 -- 1g.2 is READ and CLOSED on its gate; 1g.9 is CLOSED by its own
off-ramp; the whole 1g.10 family is CLOSED by the planner; 1g.11 is COMPLETE through all four
experiments and its gate is read (clause 3 fails on the control). 1g.2a (H4-LM) items 1-4 are
complete. The other live implementer work is PLAN_1F entry 8 (SAE_1f.md); PLAN_3E1 D9 is banked
and waits on the user (SAE_3E1.md).

DONE (1g.12 experiment 1): `G12ResourceGateJob.3h2iIpk6lpaB` -- verdict PASS for one count-4
curve (4 h) and RESOURCE_INFEASIBLE for all five starts in one process (17 h) against the 11.5 h
clamp, 4 GiB either way; verified by the verifier 2026-08-24, approach 20 and verdicts 52-54.

DONE (1g.12 experiments 2 and 3): ALL TEN CELLS. The five accepted-bigram cells reproduce their
banked 1g.11 cell (verdicts 55-56) and the means and variances 1g.11 never persisted now exist as
`parameters.npz`. The five matched-4-gram cells were killed mid-run by the 2026-08-24 cluster
filesystem event, re-run from the start at 11:39 and finished by 14:15; their rows are banked in
approach 21 and read in verdict 58. That verdict is NOT a clause verdict: the criterion ranks the
content-free random-map control above the gold-informed reference at BOTH fitting orders, which
says the criterion is not a content statistic, and says nothing yet about decoded output.

DONE (1g.13 experiment 1): `G13StreamBuildJob.Ob8Rh8y51x9M` -- 28.01 segments per second on the
bed against the ~28 published anchor, all 128 clusters used, approach 23 and verdict 57. The
stream's update role carries 1,436,262 segments, 2.46x the tokens 1g.12 fits on, which projects to
about 8 h per order-4 curve at the standing 1.5 multiplier against the 11.5 h clamp -- inside it
with little room, so 1g.13 experiment 4's resource read decides, not that projection.

DONE (1g.12 experiment 4): ALL TWENTY exact order-4 readouts finished 2026-08-24 14:14, zero
exactness violations everywhere, approach 22 and verdicts 59-60. The manager exited on completion;
its watcher reported STALLED because the one-shot graph read called finished cells "waiting", which
is the known console misreport -- the on-disk census is 4,752 jobs and zero unfinished. The verifier
reconciled verdicts 55-60 and closed the readout launch verification (2026-08-24).

DONE (1g.13 experiment 2): BOTH JOBS. `G13VadFirewallJob.Usfy2NF0LiSQ` -- the recomputed trim mask
agrees with the banked dump on all 5,567 utterances with ZERO kept-count disagreements, and its
0.1459 trimmed share reproduces the dump's own recorded `vad_dropped_frac` exactly; roles carried at
the registered 3,565 / 890 / 1,112 in three separate files. `G13RoutesJob.hStPuE1UqLK6` -- the
partition and text carried verbatim and proved, the silence mask empty by decision, and the
substantive read: p = 0.68898090 at mean duration 3.2152, one-state ADMISSIBLE and two-state
ADMISSIBLE, against seg12.5's p = 0.23560298 and one-state REFUTED. Approach 24 and verdict 61. Its
measured memory peak was 141.45 GiB against H1's 141.52, confirming the mid-run correction from the
24 GiB originally declared; it finished inside its original 2 h request.

DONE (1g.13 experiment 3): ALL SIXTEEN JOBS, finished 2026-08-24 15:21. The five registered start
protocols transport to the v1-equivalent stream and produce five distinct, valid starts; only the
controlled reference needed new code and its frame-to-segment map is proved on all 3,565 labelled
utterances. The espum arm picked full seed 0 at update 24,000 (weighted phone-LM perplexity 33.4666,
all 39 phones covered) against its bigram-only control's 64.1514 at 36 of 39. Approach 25 and
verdicts 62-64. Two things worth carrying forward: the cross-stream perplexity pair (33.4666 here
against the accepted 32.5352) is NOT comparable, because this generator emits 2.44x the tokens for
the same 890 utterances; and the espum arm's wall clock barely moved (52 min per seed against 47),
because its schedule is a fixed 40,000 updates over text -- so nothing here relaxes the order-4
resource question, which lives in the repair curve. Its watcher reported STALLED with one runnable
job left; the on-disk census is 16 of 16 finished, the same console misreport as before.

DONE 2026-08-24, the planner's ruling on the topology guard implemented and pushed (speech-llm
`6c68303`). The guard now asserts each ROUTE'S OWN registered expectation from one table read by
all three job modules: `seg12.5/phones` keeps exactly the pair it was verified at, so no 1g.12
behaviour changes and a test pins that it has not moved; the v1-equivalent route asserts two-state
ADMISSIBLE and REPORTS one-state. The reporting duty is implemented rather than described -- the
measured one-state verdict, its ratio and the class ceiling travel into every cell artifact and into
the resource gate's honesty report. An unregistered route is refused rather than defaulted. The
same commit fixes the recorded `_route_mask(h1)` default-route defect at those three call sites, so
that item is closed. The guard had NO test before this commit, which is why every existing suite
passed unchanged; `scripts/g12_route_topology_test.py` (28/28) now covers it, including that a
two-state admission failure or an INDETERMINATE on the v1 route still stops the cell.

BLOCKING BUG FOUND AND FIXED (1g.13 experiment 4, 2026-08-24, speech-llm `41127e8`). The first
gate run, `G12ResourceGateJob.4iWPXMh9yoJN`, sized PASS at 9 h against the 11.5 h clamp but reported
zero reached histories for four of its five starts. That column is not a reporting quirk: the
backward recursion in `h4_context_engine.py` was rescaled by ALPHA's per-frame normalizer, which
keeps `alpha * beta` inside float64 only while the forward and backward masses stay near each other.
With sharply peaked emissions -- which is what a concentrated start over 512-dimensional
observations gives -- `raw / scale` overflows to `+inf` and `alpha * beta` then evaluates `inf * 0`
to NAN.

Why nothing stopped: the log-likelihood is read off the alpha recursion alone, so it stayed finite;
`mstep_from_statistics` guards `weight <= 0.0` and a NAN is not `<= 0`, so it would have passed; and
the gate's own `reached = occupancy > 0.0` counts NAN as unreached and prints a plausible zero.
`gaussian_context_pass` calls `context_forward_backward` with exactly the arguments the occupancy
probe uses, so THE SAME gamma feeds the E-step's sufficient statistics -- experiment 5 would have
fitted NAN means and variances for four of five starts, and the gate's own health indicators would
not have caught it. This is broader than the diagnostic column the verifier flagged.

BANKED NUMBERS ARE UNAFFECTED, checked rather than assumed. The banked 1g.12 gate's own probe cell
(utterance 2902-9006-0015, 353 tokens, all five banked starts, the matched 4-gram automaton) was
re-run under the old and new normalizers in one process: identical log-likelihoods, gamma agreeing
to 7.8e-16, and history occupancy reproducing the banked 60,879 exactly. All ten banked 1g.12 cells
carry finite fitted parameters. Beta is now rescaled by its own per-frame maximum, which cancels
exactly because `joint` is renormalized over its own frame before anything reads it -- a change of
normalizer, not of the quantity. Two guards added so the class cannot recur silently:
`gaussian_context_pass` raises on a non-finite sufficient statistic, and the gate raises on a
non-finite occupancy. `scripts/h4_context_engine_test.py` gains a peaked-posterior case that fails
9 checks on the old normalizer and passes on the new.

DONE (1g.13 experiment 4): `G12ResourceGateJob.cQ3wfqsTamPP`, re-measured under the fixed engine.
PASS for one order-4 curve at a 9 h request against the 11.5 h clamp and 30 GiB against 256;
RESOURCE_INFEASIBLE for all five starts in one process at 43 h, so the build shape is one job per
start, the same shape 1g.12 runs. Approach 26 and verdicts 65-67. Experiment 5 is FUNDED at that
shape. The margin is thin in the direction that costs the whole fold -- 2.5 h of headroom where
1g.12 had 7.5, the fitting job caps its own request at the clamp, and these jobs do not resume --
so the first real cell's wall clock gets read against the 1.1389 h per E-step projection before the
remaining cells are launched (verifier caution (c), accepted).

DONE (the registered anchor for the backward-recursion fix): `G12EngineEquivalenceJob.sWWDLbPKglfP`
-- all five banked 1g.12 probe cells reproduce their log-likelihood at a difference of exactly 0.0
and their 60,879 histories exactly; the two normalizers agree to 8.882e-16 against a 1e-12
tolerance; the counter-case separates them (the superseded expression non-finite, the current one
finite). Verdict 68. This discharges the verifier's persist-the-harness request: the claim now
reproduces from the graph and re-runs whenever the engine changes. The superseded engine is DERIVED
from the live source by one substitution asserted to match exactly once, never kept as a copy -- a
copy would stop being the same function the moment the engine moved, and the comparison would
quietly become the copy against itself.

Its first attempt errored on a `NameError` AFTER every one of the five cells had already passed:
the job died writing its artifact because a helper (`_file_sha256`) was dropped when the module was
rewritten. The tests did not catch it because they never execute `run()`, which is a real gap in
how this file is tested and not a fluke; the suite now carries an AST lint that every
module-private helper called anywhere in the module is defined in it, verified to fail when that
helper is removed again. The job is stateless, so recovery was the error marker renamed to
`.backup` rather than a clear. `scripts/g12_engine_equivalence_test.py` 25/25.

The whole 1g.13 experiment-4 graph is finished on disk with zero unfinished jobs; the watcher
reported STALLED with one runnable, the same console misreport seen throughout this subphase.

FOUND 2026-08-24 while scoping experiment 5, before any of it was built. The plan's experiment 5
says in as many words that on this stream ALL FOUR CORNERS NEED FITTING, none is decode-only -- in
1g.12 the two table corners were decode-only only because 1g.2a had already banked their emission
tables. Two consequences the registered experiment 4 does not cover:

1. THE TABLE ARM'S COST ON THIS STREAM IS UNMEASURED. Experiment 4 was registered in 1g.12
   experiment-1 form, and that form measures the GAUSSIAN arm -- its own docstring says so, the
   difference being the per-token density evaluation a table lookup does not have. So the 9 h
   PASS licenses the ten Gaussian cells and says nothing about the ten table cells. The table arm
   is very likely cheaper, but `g12_resource.py` opens with the rule that nothing may be requested
   from the scheduler on the strength of an estimate, and "very likely cheaper" is an estimate.
   The accepted table-arm gate class (`h4_context_resource.H4ContextResourceGateJob`) exists and
   measured this at order 4 on seg12.5 (verdict 22, wide margin); running it on the new stream is
   about ten minutes. TAKEN AS COMPLETING EXPERIMENT 4 RATHER THAN AMENDING THE PLAN: the plan
   funds four corners and its experiment 4 says "before any cell is funded", so measuring the arm
   it did not name is what that sentence asks for. If it reads RESOURCE_INFEASIBLE that is a stop
   for the planner and not a fallback I pick.
2. THE TABLE PATH IS NOT PORTED. `h4_context_resource.py:247` and `h4_context_diagnostic.py:117`
   both hardcode `h1["routes"]["seg12.5/phones"]` and call `_route_mask(h1)` at its default, which
   is the same defect already fixed in the three g12 modules; `H4ContextRepairJob` additionally
   constrains `row_name` to `DIAGNOSTIC_ROWS` and `lm_identity` to identities that do not include
   `accepted-2g`. So the table half of experiment 5 needs the route widening those two modules
   never got, on the pattern already ruled for the topology guard.

BUILD ORDER THIS IMPLIES, and the reason for it: port and test the table path FIRST, then measure
the table arm, then launch cells -- never widen shared source while cells are running, because a
running job re-imports the recipe tree on every resubmit.

DONE 2026-08-24 (1g.13 experiment 5, step (a)): the table arm is PORTED, hash-neutrally, in three
commits (speech-llm `a8279ef`, `f651ffd`, `1d8f669`). Verified against the live graph after each:
142 jobs of the ported and consuming modules, zero moved and zero unfinished, so the twenty banked
1g.2a repair cells, the sixty banked local decodes and the accepted table-arm gate all keep their
hashes. Four things were widened, each hash-excluded at its default so the accepted stream is
untouched and a second stream hashes differently:

  - `route`, replacing `h1["routes"]["seg12.5/phones"]` and the default-route `_route_mask(h1)` in
    both modules -- the same defect already fixed in the three g12 modules;
  - the topology assertion, now the shared `route_topology_read` rather than a hardcoded pair, so
    both modules read the ONE table the planner ruled;
  - the START POPULATION, now a route-keyed table beside the topology one. This was NOT foreseen in
    the finding above and is the substantive part: 1g.13's five registered start names are
    `controlled_reference`, `espum`, `fingerprint`, `pseudo_pair_seed0`, `random_map_seed1000`,
    NOT the seg12.5 names. Mapping one onto the other in a config would have attributed each 1g.13
    cell to a name that means something else (`real/espum_seed0_update30000` names a seg12.5 update
    step; the 1g.13 espum pick is at update 24,000). Keyed by route, a config cannot invent a
    population and a cross-stream name is refused where it is handed over;
  - the FITTING LM scope, now including `accepted-2g`, built from the calibration artifact by
    `bigram_lm_from_matrix` exactly as the Gaussian arm's own accepted-bigram cell builds it, with
    a cell carrying both an automaton and the calibration artifact refused.

FOUND while porting, from the artifacts rather than assumed: the three registered 1g.13 start
protocols write three different manifest schemas and NO single field is present in all of them.
`H3InitializerJob` and the espum projection record the update-role binding as `fit_ids_hash`;
`G13ReferenceStartJob` records it as `h1_hashes["update"]`; only the first two carry the accepted-H1
content digest. So on a non-accepted route the binding is asserted through whichever field the
manifest uses (neither is refused), together with the route key -- which is what protects stream
identity, per the plan's own note that the route artifact refuses the accepted key outright. Which
checks ran travels into the cell's manifest, because a reader of a 1g.13 cell has to be able to SEE
that the content digest was unavailable rather than infer that it was fine. The accepted route keeps
exactly its three original checks. `scripts/h4_context_port_test.py` 50/50, including that a
seg12.5 start name on the v1 route and a v1 name on the accepted route are both refused, and a
negative control on the two-state lift.

DONE (1g.13 experiment 5, step (b)): `H4ContextResourceGateJob.8M4rSjaBlikH`, PASS. The whole 1g.13
experiment-4 graph is finished on disk with zero unfinished; its watcher reported STALLED with
three runnable, the same console misreport seen throughout this subphase. Verdict 69 and approach
28 carry the numbers. The headline is the opposite of the prior that made this gate look like a
formality: the table arm asks for 10 h against the Gaussian arm's 9 on the same stream -- cheaper
per E-step but carrying one more of them -- so the TABLE corners have 1.5 h of clamp headroom where
the continuous corners have 2.5. Both arms are funded; the wall-clock read on the first launched
cell matters more for the table corners than for the Gaussian ones.

NEXT ACTION, in order:

1. 1g.13 experiment 5 step (c), now unblocked: build the four-corner factorial at repair counts
   (0, 4) over the five starts, one job per start as the gates sized it, every cell decoded by the
   exact order-4 readout with the LM-blind local decode as its no-LM leg. Each arm's request is
   READ from ITS OWN gate artifact and never written into the config -- the two gates disagree by
   an E-step and by an order of magnitude in memory, so a single copied number would be wrong for
   one arm. Launch ONE cell of EACH arm first and read its wall clock against that arm's own per
   E-step projection before committing the rest: 1.5 h of headroom on the table corners, 2.5 on the
   continuous ones, jobs that cap their own request at the clamp, and no resume, so an optimistic
   projection costs the whole fold rather than a resubmit.
2. When experiment 5's four jobs finish: switch the 1g.12 manager from `sae_1g_12_exp5` to
   `sae_1g_12_exp6` (exp6's graph contains exp5's, so the two must never run at once) and read the
   gate. Until that job runs there is still NO phone error rate anywhere in 1g.12.
3. The verifier's experiment-3 hygiene list is NOT done and is now scoped rather than deferred
   again. Both items change a FINISHED job's output: `num_units` as a named manifest field is a
   `run()` change, so the banked manifests would still lack it and only a re-run would help; and
   renaming the espum projection's `espum_final_start.npz` to `start.npz` would leave the finished
   job dir holding the old name while consumers resolve through its finished marker and ask for the
   new one -- which breaks the arm rather than tidying it. Both belong to a rebuild of those starts,
   not to a hash-neutral edit. In the meantime the port removes the reason the naming mattered: the
   start population is read from the route's registered table, not from a glob.

IN FLIGHT (1g.12 experiment 5), launched 2026-08-24 17:00 under manager `sae_1g_12_exp5`, four
jobs. The observation-null seam is built as the planner ruled: `G12ObservationNullJob` now persists
its redrawn SELECTION-fold vectors in the segment twin's own shape (retained positions carry their
redrawn vector, dropped positions NaN so a reader that ever took one gets a NaN and not a plausible
number), records the draw seed, that file's content hash and a hash of the whole update+selection
draw, and the readout cell is handed that file as `segments_pkl`. `g12_readout_jobs.py` is
byte-identical, so the twenty banked cells are not re-certified. Speech-llm `1c25f58`.

The null is fitted at BOTH fitting orders rather than only the registered matched 4-gram. Clause 3
asks all three contrasts to be read with the observation null "the same way", and contrast (b) is a
FITTING-ORDER contrast that a single-order null cannot enter; the bigram cell is minutes. The
whole-draw hash is what makes that pair a fitting-order contrast rather than two beds: both cells
record it and experiment 6 refuses to run if they disagree. FOR THE PLANNER: this is one cheap job
beyond the registration's letter, not a change to any clause, and the registration's own cell (the
matched 4-gram) is unaffected either way.

  - `G12ObservationNullJob.tDiHo9tPpn5Z` (accepted-2g) FINISHED 17:04; 645,028 redrawn tokens over
    both folds, draw sha 98a1cc7e, selection artifact sha 38d68786.
  - `G12ObservationNullJob.QfLZEyTjxE6o` (matched-4g) RUNNING -- the ~4 h cell the gate sized.
  - `G12ExactReadoutJob.ij9vB58klqDW` / `.axh5u2jyP9Va` -- the two null readouts, each decoding its
    own cell's redrawn selection vectors.

Both null artifacts written today carry a prose `role` line naming the matched 4-gram because that
phrase was hardcoded; the authoritative `fitting_lm` field is correct in both, and the string is now
derived from the cell (speech-llm `c4d3f13`), so any later run reads right.

BUILT AND READY (1g.12 experiment 6): `G12EvaluateJob.oStN2ghRhR7l`, one job, the first phone error
rate anywhere in 1g.12. Twenty-two cells -- four corners by five starts plus the null at both
orders -- each scored under BOTH decoders, because contrast (a) is a cell against ITSELF under the
other decoder. Conventions are 1g.11's, imported; the plan's clause-1 amendment is implemented as
two independent columns (`admitted`, `shows_content`), so an inadmissible content-bearing cell
cannot read as an empty one. The babble null is re-banked at this decode's own decoded lengths,
inside this job, for the reason 1g.11 gave. Contrast (c) is the one contrast the observation null
structurally cannot enter -- it is an EMISSION-MODEL contrast and the table arm observes the frozen
unit IDs, which the null preserves by construction -- so the artifact records that rather than
leaving a hole; its content-free control is the random-map start. Waiting on experiment 5 to
finish. Suites `scripts/g12_evaluate_test.py` 37/37 and `scripts/g12_nulls_test.py` 32/32, the
latter including a negative control that a wrong scatter of the redrawn vectors is caught by
reading the artifact back through `retained_token_view`.

Proposals for the planner:

1. RATIFIED 2026-08-22 by the planner as exactly the registered local-winner exemption, and now
   asserted in the graph by `H4ProvisionalWinnerAuditJob` with empty audit mappings (approach 12):
   the frozen-versus-next-beam winner audit is satisfied without running it because all 85
   provisional winners are `decoder.kind = "local"`. The at-most-320 budgeted shard cells were not
   spent.
2. The label-free half of the baseline pre-evaluation-ready condition already reads positive -- the
   selector assigns a nonzero repair count to two of the four real starts
   (`espum_seed0_update30000` and `pseudo_pair_seed0`, both count 4) -- so the only outstanding
   input to that condition is the controlled method-level safety read, which is a label read.

## Approach

This log contains experimental evidence only; the current method, gates, and future work are defined
in `PLAN_1G.md`. `T_phi` below means the unpaired text converted to 39 stress-free ARPAbet phones. A
“rung” is one fixed audio-unit stream: adjacent-deduplicated raw codes or one of the pooled streams
`seg16`, `seg12.5`, and `seg9`.

1. **Channel-shape screen (1g.0).** On 2,703 dev-clean and 2,864 dev-other utterances, measure how
   much adjacent audio units depend on one another and compare it with three increasingly flexible
   channel shapes:

   - one audio segment per text symbol;
   - variable duration with conditionally independent emissions;
   - two ordered emission states per text symbol.

   The decision statistic is measured audio-pair mutual information divided by the maximum allowed
   by the tested shape. A ratio at or below 2 is admissible. The plug-in and Miller–Madow-corrected
   estimates must agree or the cell is indeterminate. The one-segment column needs no gold duration.
   The independent-duration and two-state columns were read at within-symbol rates measured from
   gold boundaries; they are historical diagnostics and do not set a prospective candidate's
   duration.

   Dev-other inputs to the decision:

   | audio stream | lag-1 mutual information | gold within-symbol pair rate | cross-utterance floor | gold cross-boundary share |
   |---|---:|---:|---:|---:|
   | adjacent-deduplicated raw codes | 3.2964 | 0.7093 | 0.2423 | 0.291 |
   | `seg16` | 2.6171 | 0.4226 | 0.4005 | 0.577 |
   | `seg12.5` | 2.3730 | 0.3164 | 0.4995 | 0.684 |
   | `seg9` | 1.9596 | 0.2037 | 0.6444 | 0.796 |

   Resulting ratios on dev-other, with the duration-bearing columns at that gold-derived operating
   point; parenthetical values are the only materially different dev-clean reads:

   | stream / text symbols | one segment | independent duration | two ordered states |
   |---|---:|---:|---:|
   | raw / phones | 5.76, rejected | 1.69, admissible | 1.30, admissible |
   | raw / characters | 5.62, rejected | 1.96, admissible (**2.14, rejected**) | 1.47, admissible |
   | `seg16` / phones | 4.58, rejected | 2.42, rejected | 1.50, admissible |
   | `seg16` / characters | 4.47, rejected | 2.73, rejected | 1.66, admissible |
   | `seg12.5` / phones | 4.15, rejected | 2.81, rejected | 1.64, admissible |
   | `seg12.5` / characters | 4.05, rejected | 3.08, rejected | 1.79, admissible (**1.99, indeterminate**) |
   | `seg9` / phones | 3.43, rejected | 3.01, rejected | 1.72, admissible (**1.91, indeterminate**) |
   | `seg9` / characters | 3.34, rejected | 3.14, rejected | 1.84, admissible (**2.03, rejected**) |

   Subtracting the complete cross-utterance floor still leaves every pooled independent-duration
   cell above 2 (2.02–2.79). The raw-character independent-duration result is split-dependent, not a
   clean pass.

2. **Spectral two-class anchor (1g.4).** Split text symbols and audio units into a
   syllabic/non-syllabic pair using the largest-eigenvalue eigenvector of the symmetric normalized
   Laplacian. The text side is a positive control. “Mass accuracy” weights each unit type by its
   number of evaluated occurrences; the majority is the score from always choosing the dominant
   class. “Containment” checks whether the proposed unit–phone mask retains the class of each unit's
   majority phone. The registered hard gate requires mass accuracy at least 0.85, at least 0.20 above
   the measured majority, and containment at least 0.85.

   The initial audio read used segment duration to orient the two classes because the descriptor dump
   did not yet exist. The registered energy/periodicity orientation was subsequently run; it flips
   `seg16` and `seg12.5` and leaves the verdict unchanged. The canonical dev-other read is the fixed
   572-utterance evaluation fifth; the 540-utterance dev-clean fifth gives the same all-fail verdict.

   | side / stream | top-eigengap check | canonical mass accuracy | measured majority | verdict |
   |---|---|---:|---:|---|
   | text phones (`T_phi`) | pass | 1.0000 | 0.6095 | positive control passes |
   | text characters | pass | 0.9764 | 0.6130 | positive control passes |
   | audio raw | fail | 0.5488 | 0.5711 | fail |
   | audio `seg16` | fail | 0.4452 | 0.5449 | fail |
   | audio `seg12.5` | pass | 0.4968 | 0.5312 | fail |
   | audio `seg9` | unstable | 0.7867 | 0.5154 | fail |

   The later permutation and bootstrap uncertainty reads were not persisted in a catalogued
   artifact, so this log does not treat their reported numbers as evidence. The saved point-estimate
   jobs do preserve every accuracy above, and every stream fails the 0.85 accuracy bar independently
   of either pre-check.

3. **Deterministic hard two-class descriptor screen (1g.4).** Compute seven waveform descriptors on
   the frozen wav2vec2 50 Hz grid, average each descriptor per unit over the 8,416-utterance seed bed
   (3,685,941 frames), and set its binary mass cut from the syllabic proportion of unpaired text. No
   label enters the descriptor dump or cut. The canonical reads use the same fixed 540/572 evaluation
   fifths and silence-unit convention as Approach 2.

   | stream | dev-clean best accuracy | dev-other best accuracy | dev-other majority | dev-other margin | gate |
   |---|---:|---:|---:|---:|---|
   | raw | 0.7894 | 0.7929 | 0.5711 | +0.2218 | fail |
   | `seg16` | 0.7849 | **0.8130** | 0.5449 | +0.2681 | fail |
   | `seg12.5` | 0.7629 | 0.7859 | 0.5312 | +0.2548 | fail |
   | `seg9` | 0.7588 | 0.7824 | 0.5154 | +0.2671 | fail |

   Energy is the best descriptor on every stream. These seven descriptors are seven alternative
   estimators of the **same** syllabic/non-syllabic target. They are not the six independent
   articulatory memberships required by the registered soft-product specification, so this screen
   does not answer that soft-product prerequisite.

4. **Exploratory phone repair rehearsal (E5).** The completed job exercises the soft
   two-sub-state Baum–Welch implementation on `seg12.5`, but its configuration does not implement the
   corrected experiment in `PLAN_1G.md`:

   | item | completed exploratory implementation |
   |---|---|
   | population | all 2,864 dev-other utterances used both to fit the oracle map and to score it |
   | preprocessing | differs from the frozen 1f fixture |
   | x-axis | code field `fraction_correct` is misnamed: it is the probability of retaining the reference label; `1` retains all and `0` redraws all |
   | initial channel | hard unit-to-phone map with 0.9 mass on assigned units and a uniform remainder |
   | state split | independently randomized ±10% perturbations |
   | duration | fixed mean 1.463, derived from the old gold-boundary operating point |
   | text sample | every 80th line, capped at 300,000 lines |
   | update | 30 soft emission-only Baum–Welch steps under a pinned phone bigram |
   | decoder / stopping | posterior argmax; stopping iteration selected by weighted phone-LM perplexity |
   | real candidates | neither ESPUM nor fingerprint nor the two treated controls is run |

   The old prose called retention 0 the reference endpoint and retention 1 content-free; the code
   does the reverse. The endpoint results were:

   | configured retention | realized units matching reference | start PER | LM-selected PER | step-30 PER |
   |---:|---:|---:|---:|---:|
   | 0 | 0.016 | 1.0109 | 0.8409 | 0.8409 |
   | 1 | 1.000 | 0.4865 | 0.4589 | 0.6699 |

   These show that the implementation runs and that the fixed 30 repair steps can drift even from
   its own fitted reference. Because the same utterances built and scored that reference, and because
   no real seed or treated control was run, the numbers are engineering evidence only and cannot
   fire a gate.

5. **Banked phone-seed artifact audit.** Neither 1f seed persisted a per-unit map, and both original
   seeds were fitted on all 8,416 utterances, including the fixed evaluation audio. The deterministic
   fingerprint map can be recomputed with its original arguments to verify its complete recorded
   error decomposition, but that full-bed reconstruction is provenance only. ESPUM did persist the
   selected neural checkpoint, a context-dependent convolution with `conv.weight` shape
   `(39, 500, 4)` rather than a per-unit table; that checkpoint is likewise transductive provenance,
   not a held-out input. A decisive held-out row requires construction-only fingerprint recomputation
   or ESPUM retraining and a newly measured operating point.

   The frozen encoder normalization, PCA, and K-means were fitted only on the 2,849 dedicated train
   utterances, and segment pooling is per utterance; those common transforms did not fit evaluation
   audio. The historical `UnitWordStreamJob.eIxgmMh99RSE` did learn its proxy-silence mask from all
   8,416 utterances, however. That stream is valid for fixture reproduction but not as the unchanged
   prospective held-out stream.

6. **Construction-only topology read (H1).** On the frozen 6,414-utterance update partition from the
   8,416-utterance seed bed, fit each route's duration from unpaired complete text and update audio,
   then read both channel shapes on those same masked update sequences. The execution snapshot archived
   the four imported source modules before computation; its SHA-256 is
   `b939c19d669b1b5c585915cb7a634196d31b64f38113698cabab35a1503832d9`. Both plug-in and
   Miller--Madow ratios must be at most 2 for a shape to be admissible.

   | route | retained units | fitted mean duration | one-state ratios (plug-in, MM) | two-state ratios (plug-in, MM) |
   |---|---:|---:|---:|---:|
   | `seg12.5` / phones | 397 | 1.308221 | 3.2424, 3.1845 | 1.8525, 1.8194 |
   | raw / characters | 395 | 2.601966 | 2.6394, 2.6322 | 1.8086, 1.8037 |

7. **Corrected H2/H3 phone calibration path.** H2 keeps a zero-probability duration self-loop
   impossible and treats the normalized, once-floored emission table `B(unit | phone)` as its
   canonical scoring, decoding, perturbation, and repair input. Its validator requires finite,
   positive rows whose sums agree with one at zero relative tolerance. The production decoder now
   rejects coerced grid/shard/count values and non-finite scalar evidence and proves deterministic
   shard coverage. The actual wired start has the intended 39-phone by 500-unit inventory and exact
   H1/H3 provenance. It retains eight alternatives as an output-only cap; one-best and confidence use
   the complete surviving beam. Deleted silence is one shared duration boundary for fixed scoring,
   decoding, and repair forward--backward, while phone-LM history continues across the gap. H3
   reconstructs `seg12.5` tokens as maximal runs on the
   original frame raster, removes frozen-mask silence runs as chunk boundaries, and pools ESPUM logits
   over each run's complete frame span. On the real 8,416-utterance seed bed this construction contains
   exactly 715,099 retained runs in 72,842 chunks, matching the verifier's reference counts. The
   calibration graph fits fingerprint, random-map, pseudo-pair, and ESPUM rows on the H1 update role;
   ESPUM reads the disjoint selection role label-free. The strict calibration projection averages the
   frozen selected checkpoint posterior on exactly the update role to persist `Q(phone | unit)` and
   `B(unit | phone)`. A separate graph wires blind construction-role fingerprint, random-map,
   pseudo-pair, and ESPUM refits plus the ESPUM projection. Resume state includes NumPy, CPU/CUDA
   Torch, exact CUDA device count and one nonempty RNG tensor per device, permutation, and batch-offset
   state; a serialized GH200 interrupted-versus-uninterrupted trajectory check exercises it. Projection
   manifests hash all eight modules imported by the runtime path. Its final graph contains all four
   initializer families; the trainer verifies runtime source hashes before work begins.

8. **H4 controlled calibration and repair production stage.** The first production H4 graph freezes
   the calibration inputs before any selection or evaluation labels are read. Its preparation job
   uses the accepted H1 roles and silence mask, the exact `seg12.5` frame raster, every line of
   `T_phi`, and pinned dev-clean/dev-other MFA parquet snapshots. It fits the positive-reference
   channel only on the 3,565 labelled dev utterances inside the 6,414-utterance update role and
   persists content hashes for those external inputs and every local runtime source. The fixed
   library contains the reference, 50 retain/redraw maps (ten `q` levels by five initial draws), ten
   soft-damage rows, and 20 marginal-random maps. The same update-only two-state repair is then wired
   at counts 0, 1, 2, and 4 for all 81 controlled starts and the four accepted H3 calibration starts.
   This stage does not yet decode, score the 890-utterance selection role, freeze the selector, or
   construct the final 7,304-utterance refits; those remain downstream H4 work.

   The graph completed on 2026-08-20. `H4CalibrationPreparationJob.DPv4aIqwPEzM` produced the
   start bundle, phone LM, and one donor table covering all 8,416 construction/evaluation-bed
   utterances. All 85 repair trajectories now contain finite normalized two-state tables at exactly
   counts 0, 1, 2, and 4: 81 controlled trajectories plus fingerprint, random-map seed 1000,
   pseudo-pair seed 0, and ESPUM seed 0/update 30,000. This is completed calibration infrastructure,
   not an H4 gate result.

9. **Corrected H4 recovery and decoder-resource preflight.** A role firewall first materializes only
   the 3,565 labelled dev utterances inside the 6,414-utterance update role. From those sealed inputs,
   the recovery stage regenerates `Q(phone | unit)` for the 71 non-soft controlled starts, imports the
   four persisted H3 `Q`/`B` pairs, and authorizes reuse only when canonical `Q`-to-`B` conversion and
   the old count-0 repair state match exactly. It replaces exactly the ten old B-space soft starts by
   the registered Q-space mixtures and reruns only those trajectories. Every one of the 85 direct-Q
   adapters and 340 lossless count adapters depends on the recovery authorization.

   Before selector freeze, the donor graph contains only the exact 890-utterance selection role. Its
   same-speaker support law requires donor retained length at least the source length and donor chunk
   count at most the source count before duration/rate ranking; sources without compatible donors are
   retained as `no_swap`. The resource graph then measures the exact 32-way update and selection
   workloads over three hash-deduplicated representative channel tables and all 48 decoder settings,
   followed by the globally slowest/highest-RSS setting on the heaviest real shard. It contains no
   construction/evaluation donor, decode, scoring, selector, or final-refit job.

10. **H4 label-free global-beam boundary.** The baseline `legacy-2g` continuation reuses the frozen
    three-table update inventory and passing 10-hour/2-GiB update contract. On its canonical heaviest
    shard it runs exactly 144 cells: three representatives by the fixed 12
    `(language-model scale, insertion penalty)` settings by beams 64/128/256/512. For each setting,
    the first adjacent beam pair passing at least 99.9% exact one-best agreement and strictly less
    than `1e-4` absolute decoder-score change per retained unit on every representative freezes the
    smaller beam; a setting with no passing pair is ineligible. This stage reads update audio only and
    stops before selection-surface decoding, controlled labels, construction refit, or evaluation.

    The boundary completed on 2026-08-21, and none of the 12 sequence-decoder settings has a stable
    beam under the registered criterion. Even the adjacent pair with the best worst-representative
    one-best agreement reaches only 0.7313 (required 0.999), while the pair with the smallest
    worst-representative score change reaches 0.005448 nats per retained unit (required strictly
    below 0.0001). Thus the baseline H4 surface retains the local decoder only; no sequence setting
    may enter selection, and the graph stopped without opening selection labels or evaluation.

11. **H4 pre-label selection surfaces (1g.2).** With the sequence family ruled out by approach 10,
    the baseline surface is local-only: 340 local decodes (85 starts by repair counts 0/1/2/4),
    3,400 fixed-text donor scores (10 frozen assignments per tuple), one selection surface and one
    provisional-maxima read, over the 890 selection utterances and reading no label. The statistic
    itself contributes from the 513 donor-eligible sources of those 890 -- 235 dev-clean and 278
    dev-other, the same set in every tuple and every assignment -- because the 377 `no_swap` sources
    have no eligible donor and are absent by construction. Each tuple's statistic is the
    own-minus-donor fixed-text channel rate
    (`own_logp/own_retained_units - donor_logp/donor_retained_units`), summed with `math.fsum` in
    sorted-id order over the eligible rows of each split, then weighted by the registered fixed
    shares 432/890 and 458/890 -- the no-renormalization rule, so the weights are the split sizes of
    the full 890 and not of the 513 that contribute. The maximum over the
    10 frozen assignments is that tuple's provisional maximum; ties break on the registered order
    `legacy-2g;repair_count;local;lambda_outer_beta_inner;initializer;seed;update`. Both artifacts
    carry `contains_labels: false` and `frozen_pre_label: true`, and their recorded `code_identity`
    sha256 of `h4_selector_jobs.py` (`517401b9...`) matches the committed file at `84808a8` byte for
    byte.

    The graph completed 2026-08-21. All 85 starts (81 controlled, 4 real) produced a finite
    provisional maximum and every one of the 85 winners is `decoder.kind = "local"` and
    `eligible = true`. The five registered cross-start rows:

    | cross-start row | provisional maximum | winning repair count | winning assignment |
    |---|---:|---:|---:|
    | `real/random_map_seed1000` | 10.7753 | 0 | 0 |
    | `real/fingerprint` | 10.1520 | 0 | 0 |
    | `controlled/reference` | 5.8265 | 4 | 6 |
    | `real/espum_seed0_update30000` | 4.2613 | 4 | 0 |
    | `real/pseudo_pair_seed0` | 0.1437 | 4 | 5 |

    Two internal consistency reads, neither a gate: `controlled/random_map_seed1000` returns
    10.7753, identical to its `real/` twin at every printed digit, and all 324 controlled
    within-sequence choices (81 rows by 4 counts) plus all 4,080 `global_beam_ineligible` entries
    (340 tuples by 12 settings) are ineligible, so the surface carries no sequence-decoder score
    anywhere -- which is what approach 10 requires.

    The two channel degeneracies among the 81 controlled starts are CONSTRUCTION, not a copy error,
    and both are the same fact: `Q_LEVELS` ends at 1.0 (`h4_jobs.py:34`), so level index 09 is the
    undamaged endpoint of each damage ladder. In the map ladder `keep_count` is then the whole live
    set, so `assignment[keep] = reference_map[keep]` overwrites every drawn entry and the draw seed
    cannot survive -- `controlled/map_q09_draw00..04` are necessarily one channel, and they return
    10.3872214431 at all five draws. In the soft ladder `canonical_soft_q` early-returns
    `reference.copy()` at `q_level == 1.0` (`h4_production.py:223-224`), so `controlled/soft_q09` IS
    the reference channel and returns the reference's 5.8264784397 exactly. The two q09 endpoints do
    NOT coincide with each other: the map ladder builds a hard one-hot `q` from the reference map
    while the soft ladder keeps the reference's own soft rows, which is why one reads 10.3872 and
    the other 5.8265. Effective independent controls are therefore 76 of 81, a property of the
    registered ladder design that any clustered interval or null spread over the controls must
    respect.

12. **H4 controlled validation read (1g.2 label boundary).** The planner opened the controlled
    reference labels on 2026-08-22 (`PLAN_1G.md` Status; `PLAN.md` queue item 1, user priority).
    Two jobs sit on the frozen graph and nothing else: `H4ProvisionalWinnerAuditJob.kBCapQOpk1Hj`
    emits the audited maxima -- with EMPTY audit mappings, which is how the local-winner exemption
    is ASSERTED rather than assumed, because that job errors on a sequence winner lacking an audit
    -- and `H4ControlledValidationJob.Otv6GBVY8ZUj` is the only label reader in Phase 1g
    (`speech_llm/sae/h4_validation_jobs.py`, config `sae_1g_h4_controlled_validation`). No decode,
    score, refit or maximum is recomputed: every input was already on disk, so the read is one CPU
    job over frozen artifacts plus one gold file.

    The four real H3 rows are not passed to the reader at all; it refuses any non-`controlled/`
    key at construction, so their own errors cannot be read even by mistake. Registered statistics
    (transcribed into the module docstring, so the reporting rule lives with the code that
    produces the numbers): reference vs the strongest null under a simultaneous 95 % interval,
    formed by bootstrapping `Sel(reference) - max_over_controls Sel` so bounding the maximum
    bounds every control at once; Spearman(`Sel`, -error) globally and inside the predeclared
    starting-PER band 0.80-0.93; the reference-start paired count safety read PER(r)-PER(0) for
    r in 1/2/4 against the 0.05 margin; and the within-trajectory rank/regret/count-0 bounds.
    Bounds are the ONE-SIDED 5th/95th percentiles with the two-sided pair reported beside them,
    10,000 resamples at seed 20260822, resampled channel cluster then donor assignment then
    utterance within split.

    Duplicate channels are collapsed by the artifacts' own `channel_array` sha256 rather than by
    parsing arm names; that rule independently reproduces approach 11's degeneracy finding from
    the surface alone, which is why it is the rule rather than a hand-written exception list.
    85 starts carry 79 distinct channels through THREE duplicate groups: the five `map_q09` draws
    (one channel), `soft_q09` with `controlled/reference`, and -- cross-namespace, so the reader's
    controlled-only view never sees it -- `controlled/random_map_seed1000` with
    `real/random_map_seed1000`. Effective independent CONTROLS are 76 of 81, unaffected by the
    third group since it pairs a controlled arm with a real one. (Corrected 2026-08-22: the first
    version named two groups and so could not reach 79 from 85; the verifier found the third in
    the audited maxima.) Before any interval is taken the reader rebuilds every controlled
    tuple's frozen `Sel` from that tuple's own stored per-utterance deltas and refuses to continue
    unless it matches to 1e-9 -- otherwise the bootstrap would be resampling a different statistic
    than the one the surface froze, which nothing downstream would catch.

    `scripts/h4_validation_test.py` carries 53 synthetic-only checks: the interval readings in all
    three directions, a planted quality-tracking selector that must PASS and its inversion that
    must read NEGATIVE, a harmless and a damaging repair count, the verdict table, and the label
    firewall. It builds every fixture in memory and reads no real artifact, so it runs before the
    boundary opens and cannot launder a real number into a passing check.

    `H4SelectorFreezeJob` is deliberately NOT in this graph. It raises unless the selector verdict
    is PASS -- correct, since a failed selector must not freeze a maximum -- so adding it now
    would place a job in the graph whose outcome is not yet known. It is built in a follow-up
    config only if the validation artifact reads PASS.

13. **User-funded descriptive PER read over the four real H3 seeds (1g.2, gate already closed).**
    Registered by the planner on 2026-08-22 from the user's request to compute PER on real
    dev-other data, as a measurement over the closed gate rather than a revision of it
    (`PLAN_1G.md` 1g.2 Status). One CPU job, `H4RealSeedPerJob.vu6Dp6HkJ2pH`, in its own module
    and its own config: plain per-split PER on the 890 selection-role utterances (432 dev-clean,
    458 dev-other) for the four real rows at counts 0/1/2/4, from the frozen surface's EXISTING
    decode artifacts against the same `GoldPhonesJob.ZGSp0hxyd2YP` gold. No decode was run, no
    score recomputed, no maximum reranked.

    `H4ControlledValidationJob` and its label firewall were not touched, as the registration
    requires. The two jobs are mirror images and refuse each other's inputs -- the validation
    reader rejects any non-`controlled/` key, this one requires exactly the four registered real
    rows at exactly the four registered counts -- so neither can quietly become the other. The
    reporting rule lives in the job's docstring AND in its output payload: descriptive and
    evaluation-only, selects nothing, funds nothing, and any later decision that uses these
    numbers to pick a seed or count must be re-registered with the label circularity disclosed as
    a supervision cost. The 1,112-ID held-out evaluation stays sealed.

14. **1g.2a: matched higher-order fitting LMs and the exact context-state repair engine.**
    User-mandated trigram/4-gram work, registered by the planner on 2026-08-22 as out-of-trigger
    (`h4_lm_trigger` is False from verdict 18). Funded scope is Experiments items 1-4; the F arm
    and every selector-shaped consequence stay closed under the 1g.2 verdict.

    The question item 1 answers is separation, not size. The accepted baseline `legacy-2g` is an
    add-one bigram, so raising the order alone would move SMOOTHING and ORDER together. The
    matched family is unpruned modified Kneser-Ney at orders 2/3/4 built from the SAME pinned
    complete `T_phi`, the same 39-phone inventory and the same BOS/EOS convention, with the
    canonical `phoneme_ngram_lm` settings (`interpolate_unigrams=True`, `pruning=None`, discount
    fallback 0.5/1.0/1.5 -- Kneser-Ney needs the fallback because a ~40-symbol vocabulary has no
    singleton unigrams). Matched 2 vs 3 vs 4 is then the order contrast; `legacy-2g` vs matched 2
    is the separate smoothing bridge.

    An ARPA is a backoff SCORING function, not a generator: `<unk>` carries mass, `<s>` is a
    token, and a history's continuations need not sum to one over the symbols the repair law
    permits. The compiler therefore evaluates the full backoff recursion at every history, drops
    `<unk>`/`<s>` as successors, explicitly renormalizes over the 39 phones plus EOS, and RECORDS
    the removed mass per history instead of absorbing it -- that recorded number is the audit
    trail for a leaking ARPA. The engine's BOS-padded history collapses to a single `<s>` before
    the lookup, since repeating it would query n-grams the training text never contains.

    `H4LegacyLmJob` exists because the baseline was never an artifact: `Phase1gH1Job` builds the
    add-one bigram inside its own run and keeps only what it derived. The job rebuilds it through
    the same code path and refuses to continue unless the rebuilt phone-sequence hash and line
    count match what the accepted H1 recorded. That check, not a file hash, is what binds the
    bigram to the `T_phi` the accepted surface was fitted on.

    Item 2 is the engine those automata feed. The accepted engine is bigram-specific and its dense
    transition matrix at order 4 would be about 118 GB, so it cannot represent the arm at all. The
    ruled form keeps the state as (duration sub-state, BOS-padded history), makes duration moves
    elementwise-diagonal, and leaves exactly one contraction over phone-exit arcs against the
    normalized per-history table; emissions stay tied by (phone, sub-state) and broadcast rather
    than materializing per history, and the backward pass is shared across sub-states, which the
    topology proves. Reachable histories come out at exactly 1+39+39^2+39^3 = 60,880 and an
    order-4 E-step over the update fold costs about 5.6e12 operations, both derived here and both
    matching the plan's independent projection.

    | artifact | deviating parameters | reachable histories / arcs; max renormalized mass | job hash |
    | --- | --- | --- | --- |
    | `legacy-2g` | add-one, order 2 (the accepted baseline) | 40 / 1,600; 0 (add-one needs none) | `H4LegacyLmJob.lZI6TrYdVpev` |
    | `matched-2g` | MKN, order 2 | 40 / 1,600; 7.05e-08 | `H4MatchedLmJob.T8ImJUXHaB0l` (`KenLMplzJob.ef5FXMvv8af5`) |
    | `matched-3g` | MKN, order 3 | 1,561 / 62,440; 1.274e-04 | `H4MatchedLmJob.Jb2m4aM2fUTy` (`KenLMplzJob.tis71OtNidgL`) |
    | `matched-4g` | MKN, order 4 | 60,880 / 2,435,200; 1.274e-04 | `H4MatchedLmJob.VpVkGMMy7xKW` (`KenLMplzJob.bg0iYRzBQynx`) |

    All four normalize to machine precision (max absolute error 5.6e-16). The renormalized mass is
    what each ARPA was spending on symbols this automaton cannot emit, and it is small: the
    matched family loses at most 1.3e-04 at any history, so the order contrast is not an artifact
    of throwing mass away. The legacy rebuild reproduced the accepted H1's recorded phone-sequence
    hash exactly over 39,630,169 phone lines, which is the binding that says this bigram is the
    one the accepted surface was fitted on -- a file hash would not have shown that.

    Two reruns happened in this phase and they were decided differently, which is worth stating in
    one place because the record otherwise reads as contradictory. The three MATCHED compiles were
    rerun, because their first outputs were provably wrong: one had read a superseded ARPA. The
    LEGACY artifact was rerun for a cosmetic reason -- to write the matched family's manifest field
    names so the four automata could be compared without translating between two vocabularies. That
    second rerun was a MISTAKE OF PROCESS, and the verifier was right to hand it back
    (2026-08-22): unlike the 1g.2 validation rerun, this artifact had consumers, and rewriting a
    manifest that finished cells had already bound could have stranded them. It happens to be safe
    -- the rerun completed before any diagnostic cell started, and every legacy cell's
    `input_content_sha256.automaton` equals the manifest's current `automaton_sha256` (`0aa488aa`),
    checked directly -- but that is timing, not design. The rule stands as the 1g.2 rerun set it: a
    finished artifact may be rerun only when nothing has consumed it, and a cosmetic gain never
    clears that bar.

    A residue of that decision remains and is left alone deliberately: `code_identity` hashes the
    whole module, and `h4_lm_artifacts.py` holds both job classes, so the legacy artifact now
    records a different module hash from the three matched ones. The difference is the legacy job's
    own manifest fields and memory request, neither of which the matched compile path executes.
    Re-running three correct artifacts to erase a cosmetic hash difference would repeat the mistake
    above, so the coarseness is recorded here instead.

    Orders are read from each job's own `identity`/`order` fields in the loaded graph, not from a
    hand-written hash-to-order list. Time and memory for the KenLM jobs trace to the order-4 run
    of the same job class over this same corpus (`KenLMplzJob.0aJeN88X6EdW`: 0.05 h elapsed,
    3.3 GiB max RSS at `mem=16`, `time=2`).

    Verification is synthetic-only and reads no real artifact, so it runs before anything is
    measured and cannot launder a real number into a passing check: `scripts/h4_context_engine_test.py`
    24/24 -- orders 2/3/4 reproduce exhaustive path enumeration in likelihood, posteriors AND
    counts, with and without the deleted-silence boundary rule, and the `legacy-2g` instantiation
    reproduces the accepted dense engine's likelihood, posteriors and M-step; `scripts/h4_lm_artifacts_test.py`
    24/24 -- backoff evaluation, BOS/EOS handling, renormalization audit, and the plan's separate
    requirement that the matched order-2 automaton reproduce exhaustive enumeration.

    The engine is the E-step only, so the trainer itself is a third piece:
    `h4_context_em.py` is the counterpart of the accepted `soft_reestimation_curve`, preserving
    everything that routine owns -- the two-state symmetry break applied once immediately before
    repair step 1, the unperturbed count-0 snapshot, the single common floor, the pinned fitting LM
    the M-step never touches, and the meaning of a snapshot at count `n` (the table after `n`
    M-steps with the likelihood evaluated at that table). It is written in shard form even for one
    shard, because the plan requires shards to aggregate likelihood and expected counts before ONE
    common M-step: a per-shard M-step is a different estimator, not a parallel implementation of
    the same one, and making the single-process path the one-shard case is what keeps that true
    rather than merely intended. Shards return UNNORMALIZED counts; only the driver floors and
    normalizes. `scripts/h4_context_em_test.py` 36/36, the decisive check being that the order-2
    instantiation reproduces the accepted trainer snapshot for snapshot across counts 0/1/2/4 --
    without that, a higher-order likelihood would not be comparable to the banked baseline at all
    -- plus 2/3/5-way sharding changing no number and the boundary contract refusing every way of
    getting it wrong.

    A real corruption was caught while item 1 first ran, and it is the reason `parse_arpa` now
    checks the ARPA's own declared per-order counts against what it read. The login-node
    LocalEngine ran each `KenLMplzJob` more than once (order 2 twice, order 3 three times), and a
    compile read `lm.gz` while lmplz was rewriting it. The order-2 compile failed loudly. The
    order-3 compile did NOT: it finished, its `ngram_counts` match the final file exactly, and
    only its banked `arpa_sha256` gives it away -- no file on disk carries that hash any more.
    Structurally-plausible truncation is invisible to every other check, so all matched compiles
    are rerun under the hardened reader and the family shares one compiler identity.

    Item 3, the measured resource gate, is COMPLETE and PASSES (verdict 22 and its correction):
    exact order 4 on the heaviest of the 32 update chunks costs about 50 s and 0.67 GiB, giving a
    1 h per-shard and 5 h whole-fold request at 2 GiB against 11.5 h and 256 GiB. Nothing higher
    than the baseline bigram was requested before that measurement existed.

    Item 4, the fixed-duration diagnostic, is COMPLETE in its label-free likelihood half: all 20
    cells (five starts x four fitting LMs) finished, each running the accepted 0/1/2/4 repair
    trajectory over the whole 6,414-utterance update fold (584,424 retained audio units) with only
    the fitting LM changed. Per-audio-unit log likelihood at repair count 4, read from each cell's
    own `row_name` and `fitting_lm.identity` fields:

    | start | `legacy-2g` | `matched-2g` | `matched-3g` | `matched-4g` |
    | --- | --- | --- | --- | --- |
    | `controlled/reference` | -5.2736 | -5.2736 | -5.2418 | -5.2201 |
    | `real/espum_seed0_update30000` | -5.3568 | -5.3568 | -5.3653 | -5.3821 |
    | `real/fingerprint` | -5.6262 | -5.6262 | -5.6082 | -5.6079 |
    | `real/pseudo_pair_seed0` | -5.8930 | -5.8930 | -5.8839 | -5.8822 |
    | `real/random_map_seed1000` | -5.6547 | -5.6547 | -5.6255 | -5.6410 |

    These numbers are NOT an order choice and cannot become one. Each column is the likelihood of
    the same audio under a DIFFERENT fitting LM, so the columns are not readings of one model on a
    common scale; a higher-order LM is a larger model, and the plan's own reporting rule -- carried
    in every cell's payload -- says D selects nothing, chooses no order, funds no final refit and
    cannot close the unrun coherent matched-4 arm. The own-minus-donor and descriptive-error halves
    of item 4 are not yet built, and the ordering question is not answerable from this table alone.

    The DECODE and descriptive-error half of item 4 is COMPLETE 2026-08-22
    (`config/sae_1g_h4_context_diagnostic_per.py`, `h4_context_decode.py`): 60 channel adapters,
    60 local decodes and one error read, all finished with no errors. The frozen decoder for all
    five of these starts is the LOCAL decoder -- `H4ProvisionalMaximaJob.ejmy4sdTOcS3` records
    `decoder.kind == "local"` with no lambda, no insertion penalty and no beam on every baseline row
    -- so no beam search, no sequence decode and no decoding 4-gram enter this half. Counts 1/2/4
    decode the repaired tables the 20 repair cells already banked; count 0 is read from the frozen
    1g.2 direct-Q decode, and the read job re-hashes that column in all four fitting-LM positions
    and refuses the grid if they ever disagree. Bed and gold are the 890 selection-role utterances
    of the 1g.2 descriptive read, so the count-0 and `legacy-2g` columns are comparable with
    verdict 21.

    Pooled corpus phone error rate (total edits over total reference phones), 890 selection
    utterances:

    | start | count | `legacy-2g` | `matched-2g` | `matched-3g` | `matched-4g` |
    | --- | --- | --- | --- | --- | --- |
    | `controlled/reference` | 0 | 0.3934 | 0.3934 | 0.3934 | 0.3934 |
    | `controlled/reference` | 1 | 0.3913 | 0.3913 | 0.3886 | 0.3882 |
    | `controlled/reference` | 2 | 0.4042 | 0.4042 | 0.3941 | 0.3930 |
    | `controlled/reference` | 4 | 0.4168 | 0.4168 | 0.4089 | 0.3985 |
    | `real/espum_seed0_update30000` | 0 | 0.8573 | 0.8573 | 0.8573 | 0.8573 |
    | `real/espum_seed0_update30000` | 1 | 0.8579 | 0.8579 | 0.8603 | 0.8600 |
    | `real/espum_seed0_update30000` | 2 | 0.8603 | 0.8603 | 0.8576 | 0.8555 |
    | `real/espum_seed0_update30000` | 4 | 0.8528 | 0.8528 | 0.8466 | 0.8492 |
    | `real/fingerprint` | 0 | 0.8673 | 0.8673 | 0.8673 | 0.8673 |
    | `real/fingerprint` | 1 | 0.8673 | 0.8673 | 0.8673 | 0.8673 |
    | `real/fingerprint` | 2 | 0.8656 | 0.8656 | 0.8649 | 0.8651 |
    | `real/fingerprint` | 4 | 0.8586 | 0.8586 | 0.8557 | 0.8564 |
    | `real/pseudo_pair_seed0` | 0 | 0.9136 | 0.9136 | 0.9136 | 0.9136 |
    | `real/pseudo_pair_seed0` | 1 | 0.8757 | 0.8757 | 0.8774 | 0.8774 |
    | `real/pseudo_pair_seed0` | 2 | 0.8563 | 0.8563 | 0.8585 | 0.8562 |
    | `real/pseudo_pair_seed0` | 4 | 0.8096 | 0.8096 | 0.8114 | 0.8103 |
    | `real/random_map_seed1000` | 0 | 0.9015 | 0.9015 | 0.9015 | 0.9015 |
    | `real/random_map_seed1000` | 1 | 0.9015 | 0.9015 | 0.9015 | 0.9015 |
    | `real/random_map_seed1000` | 2 | 0.9001 | 0.9001 | 0.8985 | 0.8990 |
    | `real/random_map_seed1000` | 4 | 0.8921 | 0.8921 | 0.8868 | 0.8874 |

    The `legacy-2g` and `matched-2g` columns are not a copying error and were checked as such: the
    two decodes are separate artifacts with different content hashes, produced from repaired tables
    with different array hashes, and their decoded phone sequences are nevertheless BYTE-IDENTICAL
    on all 890 utterances in all 15 repaired cells. For contrast, `matched-4g` differs from
    `legacy-2g` on 664, 796 and 849 of 890 utterances at counts 1, 2 and 4 on the reference.

    One design point, recorded because it is the only place the two halves of item 4 could have
    diverged: H4-LM-D freezes the decoder and changes only the fitting LM DURING REPAIR, so the
    local decoder's phone prior stays the accepted `phone_lm.npz` in all twenty columns. That is
    why these cells cannot reuse `H4LocalDecodeJob`, which pins its prior FILE to the channel's
    fitting LM -- the right binding in 1g.2 where the two are one object, the wrong one here. The
    binding is re-aimed, not dropped: the hypotheses still carry the channel's `fitting_lm_sha256`,
    so the frozen scorer still refuses a channel and a decode from different fitting LMs.

    The OWN-MINUS-DONOR half is COMPLETE 2026-08-22 as well, so item 4 is now finished on both
    trajectories the plan asks for. It reuses the same 60 channel adapters plus the 60 decodes above
    fanned into `H4FixedTextScoreJob` at the ten frozen donor assignments (600 jobs), and the
    count-0 column reuses the frozen 1g.2 adapter and its ten already-finished score jobs. The
    statistic is NOT re-derived: `H4ContextOwnMinusDonorJob` calls `compute_selection_aggregate`
    from the 1g.2 selector module, the same function the frozen selection surface calls, so a cell
    here and a cell there are the same statistic by construction. All 80 cells came back eligible;
    no cell was dropped. The reader lives in its own module `h4_context_scores.py` rather than in
    `h4_context_decode.py`, because both modules hash their own source into every job identity and
    adding a reader to the decode module would have moved the hash of all 121 finished decode cells,
    re-run them and orphaned the artifacts this Catalog cites.

    Own-minus-donor (own log probability per own audio unit minus donor log probability per donor
    audio unit, averaged over eligible non-`no_swap` utterances within each split, weighted 432/890
    and 458/890, then averaged over the ten frozen donor assignments):

    | start | count | `legacy-2g` | `matched-2g` | `matched-3g` | `matched-4g` |
    | --- | --- | --- | --- | --- | --- |
    | `controlled/reference` | 0 | 4.2212 | 4.2212 | 4.2212 | 4.2212 |
    | `controlled/reference` | 1 | 4.5828 | 4.5828 | 4.6133 | 4.5886 |
    | `controlled/reference` | 2 | 5.0198 | 5.0198 | 5.0152 | 4.9173 |
    | `controlled/reference` | 4 | 5.8265 | 5.8265 | 5.6251 | 5.3723 |
    | `real/espum_seed0_update30000` | 0 | 3.2048 | 3.2048 | 3.2048 | 3.2048 |
    | `real/espum_seed0_update30000` | 1 | 3.3912 | 3.3912 | 3.3297 | 3.2423 |
    | `real/espum_seed0_update30000` | 2 | 3.6519 | 3.6519 | 3.5188 | 3.3815 |
    | `real/espum_seed0_update30000` | 4 | 4.2613 | 4.2613 | 3.9725 | 3.7627 |
    | `real/fingerprint` | 0 | 10.1520 | 10.1520 | 10.1520 | 10.1520 |
    | `real/fingerprint` | 1 | 6.0444 | 6.0441 | 4.6789 | 4.5984 |
    | `real/fingerprint` | 2 | 5.1106 | 5.1104 | 3.9494 | 3.7163 |
    | `real/fingerprint` | 4 | 4.5317 | 4.5316 | 3.5672 | 3.2584 |
    | `real/pseudo_pair_seed0` | 0 | -0.0272 | -0.0272 | -0.0272 | -0.0272 |
    | `real/pseudo_pair_seed0` | 1 | 0.0266 | 0.0266 | 0.0245 | 0.0244 |
    | `real/pseudo_pair_seed0` | 2 | 0.0506 | 0.0506 | 0.0480 | 0.0510 |
    | `real/pseudo_pair_seed0` | 4 | 0.1437 | 0.1437 | 0.1390 | 0.1404 |
    | `real/random_map_seed1000` | 0 | 10.7753 | 10.7753 | 10.7753 | 10.7753 |
    | `real/random_map_seed1000` | 1 | 7.1353 | 7.1351 | 6.3593 | 6.7122 |
    | `real/random_map_seed1000` | 2 | 5.6890 | 5.6888 | 4.5740 | 4.3967 |
    | `real/random_map_seed1000` | 4 | 4.8425 | 4.8424 | 3.7748 | 3.4494 |

    Not funded and not run: the 81-row controlled library, any selector refit, any order choice, any
    final refit.

15. **1g.9 experiment 1: locate the phone-repair collapse.** For the five 1g.2a starts at repair
    counts 0 and 4, under the accepted two-state topology at the frozen H1 duration p=0.235603 and
    the `legacy-2g` fitting LM, compute the posterior expected symbol-ENTRY distribution `q_bar`
    and the posterior expected rate by forward-backward, the same two statistics from the banked
    frozen local one-bests (no new decode), and each proposed constraint term's gradient. `p_text`
    is the accepted calibration `phone_lm`'s `phone_prior` over the complete 39,630,169-line
    unpaired phone corpus; `r_target` = 0.7644 symbols per retained unit, derived from the frozen
    H1 length-law fit (the memoryless reading `1-p` agrees to four decimals, and 53,498 update /
    5,110 selection forced boundaries at deleted-silence gaps are the one term a healthy posterior
    may legitimately exceed it by). Gradients are reported as `lambda_equal = ||grad L_HMM|| /
    ||grad L_term||` in the `B = softmax(theta)` parameterization. Label-free; selects nothing.
    One job, `H4CollapseLocateJob.gZ9d6e3E7ZGu`, 31 minutes.

    Posterior on the 890 selection utterances, matched to the decode. Residuals are relative to
    `r_target`; `cl0` is the registered clause-0 pattern (posterior within total variation 0.15 and
    rate within 20 percent, while the decode meets fewer).

    | start | count | post TV | post rate res | dec TV | dec rate res | distinct | cl0 |
    |---|---|---|---|---|---|---|---|
    | `controlled/reference` | 0 | 0.0498 | -0.014 | 0.0854 | +0.148 | 37 | no |
    | `controlled/reference` | 4 | 0.0436 | -0.054 | 0.0658 | +0.121 | 37 | no |
    | `real/espum_seed0_update30000` | 0 | 0.0334 | +0.153 | 0.1058 | +0.266 | 35 | yes |
    | `real/espum_seed0_update30000` | 4 | 0.0690 | +0.000 | 0.0847 | +0.222 | 36 | yes |
    | `real/fingerprint` | 0 | 0.1165 | +0.208 | 0.1545 | +0.233 | 39 | no |
    | `real/fingerprint` | 4 | 0.0317 | -0.042 | 0.1357 | +0.209 | 38 | yes |
    | `real/pseudo_pair_seed0` | 0 | 0.0023 | +0.027 | 0.8345 | -0.856 | 3 | yes |
    | `real/pseudo_pair_seed0` | 4 | 0.0120 | -0.001 | 0.6871 | -0.506 | 9 | yes |
    | `real/random_map_seed1000` | 0 | 0.0851 | +0.118 | 0.0291 | +0.263 | 37 | yes |
    | `real/random_map_seed1000` | 4 | 0.0664 | -0.055 | 0.0402 | +0.240 | 37 | yes |

    Update-role posterior (the fold the constrained objective would run on), the two constraint
    terms' `lambda_equal`, and the per-retained-unit log likelihood on each fold.

    | start | count | TV | KL coverage | rate | lam coverage | lam rate | sel LL/unit | upd LL/unit |
    |---|---|---|---|---|---|---|---|---|
    | `controlled/reference` | 0 | 0.0473 | 0.0068 | 0.7563 | 6.068e+06 | 9.440e+07 | -5.6506 | -5.6452 |
    | `controlled/reference` | 4 | 0.0439 | 0.0064 | 0.7208 | 1.223e+06 | 5.283e+06 | -5.2971 | -5.2736 |
    | `real/espum_seed0_update30000` | 0 | 0.0365 | 0.0039 | 0.8825 | 3.296e+06 | 5.820e+06 | -5.7715 | -5.7663 |
    | `real/espum_seed0_update30000` | 4 | 0.0736 | 0.0160 | 0.7624 | 1.034e+06 | 4.040e+07 | -5.3792 | -5.3568 |
    | `real/fingerprint` | 0 | 0.1205 | 0.0449 | 0.9242 | 1.422e+06 | 1.927e+07 | -7.2593 | -7.2930 |
    | `real/fingerprint` | 4 | 0.0352 | 0.0052 | 0.7259 | 1.987e+06 | 1.077e+07 | -5.6502 | -5.6262 |
    | `real/pseudo_pair_seed0` | 0 | 0.0006 | 0.0000 | 0.7861 | 1.454e+08 | 1.532e+07 | -5.9457 | -5.9438 |
    | `real/pseudo_pair_seed0` | 4 | 0.0108 | 0.0008 | 0.7620 | 3.509e+06 | 2.812e+07 | -5.9037 | -5.8930 |
    | `real/random_map_seed1000` | 0 | 0.0809 | 0.0712 | 0.8519 | 2.166e+06 | 7.495e+06 | -6.9734 | -6.9771 |
    | `real/random_map_seed1000` | 4 | 0.0630 | 0.0175 | 0.7178 | 8.085e+05 | 8.302e+06 | -5.6775 | -5.6547 |

    Not funded and not run: 1g.9 experiment 2's constrained refits, experiment 3's unigram-matched
    babble null, and every constrained-arm lambda. Their graph does not exist.

16. **1g.10: the full-model (LM-aware) sequence decode of the audited count-4 channels.** The
    registered prefix-mass beam decoder under the frozen H1 two-state law with the banked KenLM
    phone 4-gram REPLACING the fitting bigram, run on the three audited count-4 channels over the
    12-point grid (lm_scale in {0.5, 1, 2, 4} x insertion penalty in {-2, -1, 0}). Beam 512 is the
    decision beam and runs the full 890-utterance selection role over 32 shards per cell (1,152
    chunk jobs plus 36 merges); beam 256 exists only to supply the adjacent-pair agreement and
    drift columns and runs ONE shard per cell on the planner's budget ruling -- the heaviest
    selection-role shard taken from the measured resource contract's own `shard` block, 27
    utterances, the same index for every cell. The global-beam eligibility flag is read for
    provenance and deliberately NOT applied: measuring the surface that bar was hiding is the
    point. Nothing is refit and nothing is selected; the held-out evaluation stays sealed.

    | row | readable cells of 12 | cells clearing the babble null | best correct-phone fraction (cell) |
    |---|---|---|---|
    | `controlled/reference` (positive control) | 7 | 12 | 0.6010 (lam 0.5, beta 0) |
    | `real/espum_seed0_update30000` | 7 | 12 | 0.1907 (lam 2, beta 0) |
    | `real/pseudo_pair_seed0` (collapsed) | 1 | 12 | 0.1756 (lam 1, beta 0) |

    The LM-blind local decoder on the same channels, for reference (this is what 1g.9 measured):
    `controlled/reference` correct-phone 0.5832 at TV 0.0658 and length ratio 1.1205;
    `real/espum_seed0_update30000` 0.1472 at TV 0.0847 / ratio 1.2224; `real/pseudo_pair_seed0`
    0.1904 at TV 0.6871 / ratio 0.4941 on only 9 distinct phones.

    Beam-stability explanation duty, over all 36 cells (agreement and drift on the probe's 27
    utterances; margins on all 890):

    | quantity | min | median | max |
    |---|---|---|---|
    | one-best agreement, beam 256 against beam 512 | 0.2222 | 0.6111 | 0.8889 |
    | median score margin, nats per retained unit | 1.210e-03 | 4.345e-03 | 1.540e-02 |
    | fraction of utterances in a cell at or below the flat threshold | 0.061 | -- | 0.466 |

    ZERO of 36 cells reaches the registered 0.999 agreement level and ZERO of 36 has a median
    margin at or below the registered flat threshold of 1e-3 nats per retained unit.

    1g.10b re-ran the same 36 cells on the same 27-utterance contract shard at beam 1024, through
    a dedicated probe class that imports the production decoder and first reproduces one banked
    production chunk exactly at beam 512 (parity cell). Adjacent-pair columns, both read on those
    27 utterances:

    | quantity | min | median | max |
    |---|---|---|---|
    | one-best agreement, beam 256 against beam 512 | 0.2222 | 0.6111 | 0.8889 |
    | one-best agreement, beam 512 against beam 1024 | 0.3704 | 0.7037 | 0.8889 |
    | score drift per retained unit, beam 256 against beam 512 | 1.567e-03 | 8.173e-03 | 1.821e-02 |
    | score drift per retained unit, beam 512 against beam 1024 | 1.061e-03 | 5.143e-03 | 2.755e-02 |

    1g.10c re-decoded two of the three rows -- the positive control and the real ESPUM arm, the
    collapsed row refused in code -- at four extension points (lm_scale in {1, 2} x insertion bonus
    beta in {+1, +2}), each paired against ITS OWN beta-0 production cell at the same lm_scale on
    the same 890 utterances. The parity cell reproduced the banked production chunk exactly.
    Deltas are correct-phone fraction, extension minus baseline, so positive means the bonus
    recovers phones:

    | cell | paired delta | 95 pct CI (stratified) | CI (unstratified) | frac of utterances improved |
    |---|---|---|---|---|
    | `controlled/reference` lambda 1, beta +1 | +0.0222 | [+0.0188, +0.0256] | [+0.0188, +0.0256] | 0.565 |
    | `controlled/reference` lambda 1, beta +2 | +0.0358 | [+0.0315, +0.0403] | [+0.0314, +0.0403] | 0.664 |
    | `controlled/reference` lambda 2, beta +1 | +0.0312 | [+0.0278, +0.0347] | [+0.0277, +0.0347] | 0.570 |
    | `controlled/reference` lambda 2, beta +2 | +0.0555 | [+0.0506, +0.0605] | [+0.0505, +0.0604] | 0.740 |
    | `real/espum_seed0_update30000` lambda 1, beta +1 | -0.0122 | [-0.0144, -0.0101] | [-0.0144, -0.0101] | 0.189 |
    | `real/espum_seed0_update30000` lambda 1, beta +2 | -0.0294 | [-0.0330, -0.0261] | [-0.0329, -0.0260] | 0.166 |
    | `real/espum_seed0_update30000` lambda 2, beta +1 | -0.0021 | [-0.0046, +0.0003] | [-0.0046, +0.0002] | 0.310 |
    | `real/espum_seed0_update30000` lambda 2, beta +2 | -0.0090 | [-0.0124, -0.0058] | [-0.0123, -0.0058] | 0.322 |

    Pooled description beside the paired read, never instead of it: the control's best extension
    cell reaches PER 0.3943 (lambda 1, beta +2) against its own beta-0 cell, and the real arm stays
    between 0.8102 and 0.8417 across all four. Decoded length rises with beta in every cell (the
    control 51.0 -> 57.2 units at lambda 1; the real arm 50.8 -> 60.2), which is the bonus doing
    what it is for; all eight cells clear their own babble null, and all eight use the full
    39-phone inventory.

    Per row at beam 512 against 1024 (median agreement over that row's 12 cells):
    `controlled/reference` 0.7222, `real/espum_seed0_update30000` 0.7407,
    `real/pseudo_pair_seed0` 0.6296. Cell by cell against the same cell's 256-vs-512 column,
    agreement rises in 24 of 36 cells, falls in 10 and ties in 2; drift falls in 25 and RISES in
    11. Best cell 24 of 27; ZERO of 36 cells reaches the pre-registered 26-of-27 cross-channel
    quoting bar.


17. **1g.11 experiment 1: the continuous twin of the `seg12.5` observation stream.** One vector per
    TOKEN of the frozen discrete stream: the frozen state dump standardized and projected on the
    frozen PCA basis (`QuantizeStatesJob.FWpGhC941JMi`), averaged over exactly the frames of each
    run-length run of the frozen `seg12.5` stream, then standardized per component by a scale
    fitted on the dedicated train half alone. Two checks, answering different questions. THE
    PIPELINE CHECK, asserted: at the WARD segmentation the re-assigned segment means must reproduce
    the frozen stream bit-for-bit, run through `repr_pool.pool_utterance` itself. THE TWIN CHECK,
    reported: the share of TOKENS whose re-assigned mean still returns the frozen code, which is
    the question of whether the token-level twin is the discrete stream's twin at all.

    | bed | utterances | tokens | ward segments | frozen PCA dim | kept | pipeline check | twin check | job |
    |---|---|---|---|---|---|---|---|---|
    | seed bed, `seg12.5` | 8,416 | 919,248 | 921,432 | 96 | 96 | exact, 8,416 of 8,416 | 919,248 of 919,248 (100.0000%) | `G11ContinuousSegmentsJob.hImWJG0X4eZh` |

    Scale fitted on 2,849 dedicated train utterances, 448,204 segment vectors. 2,184 ward segments
    (0.237%) are absorbed into tokens across 1,334 of the 8,416 utterances.

18. **1g.11 experiment 2: the Gaussian repair cells.** The five 1g.2a starts at repair counts 0 and
    4 under the constrained update rule (shared diagonal covariance, M2 floor), EM on the accepted
    H1 update role and decode on its selection role, with the single disclosed per-row-covariance
    relaxation on `real/espum_seed0_update30000`. The table arm is NOT re-decoded: the audited
    1g.2a one-bests are the comparator, so this produces the Gaussian side only. Retained after the
    frozen silence mask: 584,424 of 751,195 tokens (0.7780) over 6,414 update utterances and 60,604
    of 77,566 (0.7813) over 890 selection utterances. Duration p 0.2356; 103 of 500 unit IDs masked.
    `sym/tok` is decoded symbols per RETAINED TOKEN and is NOT the table arm's decoded-length-versus-
    gold ratio; the two have different denominators and may not be compared.

    | cell | log-likelihood | floor share | clipped | sym/tok | distinct |
    |---|---|---|---|---|---|
    | controlled/reference tied 0 | -77,517,283.6 | 0.0000 | 0 | 0.8003 | 39 of 39 |
    | controlled/reference tied 4 | -73,987,478.3 | 0.0000 | 0 | 0.7789 | 39 of 39 |
    | espum_seed0_update30000 per_row 0 | -79,791,528.0 | 0.0000 | 0 | 0.9220 | 38 of 39 |
    | espum_seed0_update30000 per_row 4 | -72,994,684.4 | 0.0000 | 68 | 0.8741 | 39 of 39 |
    | espum_seed0_update30000 tied 0 | -79,791,528.0 | 0.0000 | 0 | 0.9220 | 38 of 39 |
    | espum_seed0_update30000 tied 4 | -74,726,774.7 | 0.0000 | 0 | 0.8694 | 39 of 39 |
    | fingerprint tied 0 | -79,836,863.3 | 0.0000 | 0 | 0.8791 | 39 of 39 |
    | fingerprint tied 4 | -74,841,463.6 | 0.0000 | 0 | 0.8080 | 38 of 39 |
    | pseudo_pair_seed0 tied 0 | -80,515,490.0 | 0.0000 | 0 | 0.0152 | 3 of 39 |
    | pseudo_pair_seed0 tied 4 | -77,940,277.7 | 0.0000 | 0 | 0.7411 | 39 of 39 |
    | random_map_seed1000 tied 0 | -79,781,460.4 | 0.0000 | 0 | 0.8948 | 37 of 39 |
    | random_map_seed1000 tied 4 | -75,065,550.4 | 0.0000 | 0 | 0.7887 | 39 of 39 |

    `clipped` counts (token, state) emissions held at the float64 dynamic range out of about 45.6
    million per cell; it is a numerical floor, not a model term.

19. **1g.11 experiments 3 and 4: the observation null and the evaluation against gold.** The
    observation null is the same class with one method overridden: every retained token's vector
    redrawn i.i.d. with replacement from the pooled corpus segment-vector marginal over both folds,
    token counts, unit IDs, retained index and duration boundaries asserted to survive, fitted and
    decoded through the arm's own code (`G11ObservationNullJob.orOc9h6K3cuR`; 645,028 vectors
    redrawn; log-likelihood -79,970,371.2 at count 0 to -75,850,789.7 at count 4). The evaluation
    scores BOTH arms against gold on the 890 selection utterances -- the table arm from its banked
    one-bests, never re-decoded -- on the unit-cost Levenshtein of `h4_validation_jobs`, pooled
    PER, with the edit decomposition beside every PER and the babble null computed in the same job.
    The 1,112-utterance evaluation fifth is untouched. `len/gold` is decoded symbols over GOLD
    phones (clause 1's own wording) and is NOT the 1g.10 family's ratio to `r_target`.

    | cell | PER | corr | len/gold | TV | S | I | D | null p99 | c1 | c2 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | gaussian reference tied 0 | 0.3498 | 0.6502 | 0.7947 | 0.1018 | 6,687 | 1,066 | 13,594 | 0.1550 | . | . |
    | gaussian reference tied 4 | 0.4429 | 0.5571 | 0.7735 | 0.1186 | 11,212 | 996 | 14,821 | 0.1581 | . | . |
    | gaussian espum per_row 0 | 0.8440 | 0.1560 | 0.9155 | 0.1625 | 41,658 | 2,349 | 7,504 | 0.1224 | Y | . |
    | gaussian espum per_row 4 | 0.8523 | 0.1477 | 0.8680 | 0.2134 | 41,252 | 1,355 | 9,412 | 0.1370 | Y | . |
    | gaussian espum tied 0 | 0.8440 | 0.1560 | 0.9155 | 0.1625 | 41,658 | 2,349 | 7,504 | 0.1224 | Y | . |
    | gaussian espum tied 4 | 0.8491 | 0.1509 | 0.8634 | 0.2161 | 40,901 | 1,292 | 9,632 | 0.1385 | Y | . |
    | gaussian fingerprint tied 0 | 0.8404 | 0.1596 | 0.8730 | 0.1173 | 40,330 | 1,605 | 9,359 | 0.1351 | Y | . |
    | gaussian fingerprint tied 4 | 0.8392 | 0.1608 | 0.8024 | 0.1695 | 37,892 | 632 | 12,694 | 0.1518 | Y | . |
    | gaussian pseudo_pair tied 0 | 0.9855 | 0.0145 | 0.0151 | 0.8806 | 34 | 0 | 60,111 | 0.0123 | . | . |
    | gaussian pseudo_pair tied 4 | 0.8711 | 0.1289 | 0.7359 | 0.2860 | 36,139 | 456 | 16,572 | 0.1578 | . | . |
    | gaussian random_map tied 0 | 0.8846 | 0.1154 | 0.8885 | 0.1010 | 43,247 | 1,967 | 8,773 | 0.1296 | Y | . |
    | gaussian random_map tied 4 | 0.8752 | 0.1248 | 0.7832 | 0.1956 | 39,110 | 538 | 13,769 | 0.1560 | . | . |
    | obsnull espum tied 0 | 0.8946 | 0.1054 | 0.9487 | 0.1701 | 44,778 | 3,344 | 6,475 | 0.1077 | Y | . |
    | obsnull espum tied 4 | 0.9252 | 0.0748 | 0.9630 | 0.2123 | 47,601 | 3,304 | 5,563 | 0.1021 | Y | . |
    | table reference 0 | 0.3934 | 0.6066 | 0.8711 | 0.0854 | 10,909 | 2,617 | 10,486 | 0.1366 | Y | Y |
    | table reference 4 | 0.4168 | 0.5832 | 0.8505 | 0.0658 | 11,844 | 2,235 | 11,357 | 0.1417 | Y | Y |
    | table espum 0 | 0.8573 | 0.1427 | 0.9612 | 0.1058 | 42,550 | 3,702 | 6,068 | 0.1043 | Y | . |
    | table espum 4 | 0.8528 | 0.1472 | 0.9279 | 0.0847 | 42,144 | 2,752 | 7,155 | 0.1176 | Y | . |
    | table fingerprint 0 | 0.8673 | 0.1327 | 0.9359 | 0.1545 | 43,150 | 2,935 | 6,848 | 0.1128 | Y | . |
    | table fingerprint 4 | 0.8586 | 0.1414 | 0.9174 | 0.1357 | 42,232 | 2,566 | 7,606 | 0.1195 | Y | . |
    | table pseudo_pair 0 | 0.9136 | 0.0864 | 0.1095 | 0.8345 | 1,409 | 0 | 54,350 | 0.0701 | . | . |
    | table pseudo_pair 4 | 0.8096 | 0.1904 | 0.3751 | 0.6871 | 11,236 | 18 | 38,159 | 0.1485 | . | . |
    | table random_map 0 | 0.9015 | 0.0985 | 0.9584 | 0.0291 | 45,103 | 3,690 | 6,227 | 0.1037 | Y | . |
    | table random_map 4 | 0.8921 | 0.1079 | 0.9416 | 0.0402 | 44,378 | 3,252 | 6,818 | 0.1106 | Y | . |

    Clause 3, the decision read -- paired per-utterance correct-phone delta at count 4, candidate
    minus its own start's banked table cell, stratified within evaluation split, 10,000 resamples
    at seed 42. Every interval below excludes zero.

    | pair (candidate minus its own table cell) | delta | 95 pct interval | role |
    |---|---|---|---|
    | gaussian reference tied 4 | -0.0208 | [-0.0260, -0.0154] | arm |
    | gaussian espum per_row 4 | +0.0055 | [+0.0013, +0.0096] | arm |
    | gaussian espum tied 4 | +0.0098 | [+0.0058, +0.0137] | arm (the gate's selected real start) |
    | gaussian fingerprint tied 4 | +0.0273 | [+0.0226, +0.0321] | arm |
    | gaussian pseudo_pair tied 4 | -0.0746 | [-0.0799, -0.0698] | arm |
    | gaussian random_map tied 4 | +0.0251 | [+0.0202, +0.0302] | CONTENT-FREE CONTROL |
    | obsnull espum tied 4 | -0.0772 | [-0.0817, -0.0728] | CONTENT-FREE CONTROL |

    Clause 4: the floor share is 0.0000 in every Gaussian and null cell. The registered babble null
    (100 draws from `p_text`) reproduces the same null at 1,000 draws to within 6.1e-04 in the
    worst of the 24 cells (`gaussian random_map tied 4`), so the noisy-bar concern raised before
    the run does not bite at this margin -- the smallest clause-2 gap in the table is 0.0090 in
    absolute terms, an order of magnitude above the bar's own wobble. The unigram-matched null of
    the 1g.10 family is printed beside both and decides nothing.

20. **1g.12 experiment 1: the measured resource read for the Gaussian arm at order 4.** The
    accepted order-4 gate measured the TABLE arm; this is its counterpart for the continuous
    emission model, in the same form -- the same deterministic 32-way `ids[j::32]` sharding of the
    accepted H1 update role, the same probe rule (the longest update utterance, ties to the higher
    ID), the same forked-child measurement so each cell reports its own peak, the same 1.5
    multiplier and the same 11.5 h / 256 GiB limits. It runs no M-step, decodes nothing and reads no
    label. The probe population is all five funded starts rather than an entropy bracket: at five
    starts the bracket IS the population. Job `G12ResourceGateJob.3h2iIpk6lpaB`; the fitting LM is
    the matched Kneser-Ney order-4 automaton of 1g.2a item 1.

    The engine underneath is the accepted context recursion with ONE addition the plan directs:
    `context_forward_backward` now takes an optional per-token `(batch, time, 2, 39)` emission
    array, the order-k twin of the argument `channel_h.marginal_forward_backward` already carried at
    order 2. Exactly one of the categorical table and the per-token array may be given. Established
    before the job ran (`g12_gaussian_context_test` 57/57): the dense path reproduces the
    categorical path at orders 2, 3 and 4 in likelihood, posteriors and aggregated counts; and run
    at order 2 the new `gaussian_context_repair_curve` reproduces `g11_gaussian.gaussian_repair_curve`
    -- the banked 1g.11 code path -- parameter for parameter and criterion for criterion, tied and
    per-row, with and without deleted-silence boundaries.

    | cell | start | sec | RSS GiB | reached histories | reached arcs |
    |---|---|---|---|---|---|
    | probe utterance (353 retained tokens) | controlled/reference | 0.92 | 1.32 | 60,879 | 2,435,160 |
    | probe utterance | real/espum_seed0_update30000 | 0.91 | 1.32 | 60,879 | 2,435,160 |
    | probe utterance | real/fingerprint | 0.90 | 1.32 | 60,879 | 2,435,160 |
    | probe utterance | real/pseudo_pair_seed0 | 0.92 | 1.32 | 60,879 | 2,435,160 |
    | probe utterance | real/random_map_seed1000 | 0.90 | 1.32 | 60,879 | 2,435,160 |
    | heaviest chunk (19,515 retained tokens, 201 utterances) | controlled/reference | 48.79 | 1.29 | . | . |
    | heaviest chunk | real/random_map_seed1000 | 48.88 | 1.29 | . | . |

    Projected from the heaviest chunk standing in for all 32 (an upper bound):
    0.4345 h per whole-fold E-step over the 584,424 retained
    update tokens, five E-steps per count-4 curve, so
    4 h for one curve and
    17 h for all five starts in one process, both at 1.5x;
    4 GiB either way.

21. **1g.12 experiments 2 and 3: the Gaussian cells at order 4 and the order-2 re-fit.** Ten jobs,
    one per (start, fitting order), the shape experiment 1 licensed. Five fit the matched 4-gram;
    five re-run 1g.11's own operating point -- the accepted add-one bigram -- because 1g.11
    persisted its decoded output and its statistics but not the fitted means and variances, which
    is why its corner has to be produced again. Every bigram cell asserts itself against the banked
    1g.11 cell BEFORE writing any artifact, at tolerances declared in the producing job
    (`log_likelihood_rtol` 1e-9, decoded-symbol disagreements exactly 0); a corner that does not
    reproduce is a broken re-run and not a result about fitting order, so it writes nothing. The
    decode here is the LM-BLIND local decoder only -- 1g.11's own, unchanged -- which is the no-LM
    leg of clause 3's readout contrast; the exact order-4 one-best readout is experiment 4 and does
    not exist yet. Requests are read from the gate artifact, never written in the config. All ten
    cells are complete: the five matched-4g cells were killed mid-run by the 2026-08-24 cluster
    filesystem event, re-run from the start the same day, and finished in 2.2 h each.

    `reproduces` is the bigram corner's acceptance check against the banked 1g.11 cell, as
    `criterion relative difference / decoded-utterance disagreements`; the declared bar is 1e-9 and
    exactly zero. `sym/tok` is per RETAINED TOKEN and licenses no cross-arm comparison. The
    criterion is a log likelihood UNDER THE CELL'S OWN FITTING LM, so a matched-4g row and an
    accepted-2g row are in different currencies and their magnitudes must never be subtracted;
    within one fitting LM all five starts score the same 584,424 retained update tokens under the
    same model class, so a count-0-to-count-4 gain may be compared across starts and is what verdict
    58 reads.

    | start | fitting LM | count | criterion | floor | sym/tok | distinct | reproduces | job |
    |---|---|---|---|---|---|---|---|---|
    | real/pseudo_pair_seed0 | accepted-2g | 0 | -80,515,490.0 | 0.0000 | 0.0152 | 3 | 9.3e-16 / 0 of 890 | `0nngx4f5pX69` |
    | real/pseudo_pair_seed0 | accepted-2g | 4 | -77,940,277.7 | 0.0000 | 0.7411 | 39 | 1.3e-15 / 0 of 890 | `0nngx4f5pX69` |
    | controlled/reference | accepted-2g | 0 | -77,517,283.6 | 0.0000 | 0.8003 | 39 | 1.7e-15 / 0 of 890 | `OBwHBeOmwYU5` |
    | controlled/reference | accepted-2g | 4 | -73,987,478.3 | 0.0000 | 0.7789 | 39 | 2.6e-15 / 0 of 890 | `OBwHBeOmwYU5` |
    | real/random_map_seed1000 | accepted-2g | 0 | -79,781,460.4 | 0.0000 | 0.8948 | 37 | 1.9e-16 / 0 of 890 | `OyooGnuVi7EK` |
    | real/random_map_seed1000 | accepted-2g | 4 | -75,065,550.4 | 0.0000 | 0.7887 | 39 | 1.2e-15 / 0 of 890 | `OyooGnuVi7EK` |
    | real/fingerprint | accepted-2g | 0 | -79,836,863.3 | 0.0000 | 0.8791 | 39 | 3.7e-16 / 0 of 890 | `iZaUwq3DQVjj` |
    | real/fingerprint | accepted-2g | 4 | -74,841,463.6 | 0.0000 | 0.8080 | 38 | 4.0e-16 / 0 of 890 | `iZaUwq3DQVjj` |
    | real/espum_seed0_update30000 | accepted-2g | 0 | -79,791,528.0 | 0.0000 | 0.9220 | 38 | 9.3e-16 / 0 of 890 | `uczGmykabX6i` |
    | real/espum_seed0_update30000 | accepted-2g | 4 | -74,726,774.7 | 0.0000 | 0.8694 | 39 | 1.0e-15 / 0 of 890 | `uczGmykabX6i` |
    | real/pseudo_pair_seed0 | matched-4g | 0 | -80,515,325.7 | 0.0000 | 0.0152 | 3 | n/a, no banked cell at this order | `DgOI3SI1cwph` |
    | real/pseudo_pair_seed0 | matched-4g | 4 | -78,025,235.8 | 0.0000 | 0.7449 | 39 | n/a, no banked cell at this order | `DgOI3SI1cwph` |
    | controlled/reference | matched-4g | 0 | -77,529,644.2 | 0.0000 | 0.8003 | 39 | n/a, no banked cell at this order | `8OzLoDv4PPlt` |
    | controlled/reference | matched-4g | 4 | -74,260,984.1 | 0.0000 | 0.7832 | 39 | n/a, no banked cell at this order | `8OzLoDv4PPlt` |
    | real/random_map_seed1000 | matched-4g | 0 | -79,843,988.2 | 0.0000 | 0.8948 | 37 | n/a, no banked cell at this order | `dDKq6J6AQEIP` |
    | real/random_map_seed1000 | matched-4g | 4 | -75,404,850.9 | 0.0000 | 0.7733 | 39 | n/a, no banked cell at this order | `dDKq6J6AQEIP` |
    | real/fingerprint | matched-4g | 0 | -79,890,010.4 | 0.0000 | 0.8791 | 39 | n/a, no banked cell at this order | `BrQtRIAKaWwU` |
    | real/fingerprint | matched-4g | 4 | -75,354,549.9 | 0.0000 | 0.7964 | 38 | n/a, no banked cell at this order | `BrQtRIAKaWwU` |
    | real/espum_seed0_update30000 | matched-4g | 0 | -79,805,418.7 | 0.0000 | 0.9220 | 38 | n/a, no banked cell at this order | `kHwPYElOcCPr` |
    | real/espum_seed0_update30000 | matched-4g | 4 | -75,188,448.6 | 0.0000 | 0.8626 | 39 | n/a, no banked cell at this order | `kHwPYElOcCPr` |


22. **1g.12 experiment 4: the exact beam-free order-k one-best readout
    (`g12_exact_decode.py`).** Built and accepted against its pre-registered fixtures; no cell has
    been decoded yet. It is the maximizing twin of the accepted context recursion -- same
    `(duration sub-state, BOS-padded history)` state space, same sub-stochastic path law, same
    BOS start and end-of-sequence terminal -- with the path sum replaced by a path maximum and
    backpointers kept, and it reads the recursion's history algebra from the engine module rather
    than restating it. The banked prefix-beam decoder was not reusable here because 1g.10b found no
    affordable beam meeting the stability duty (0 of 36 cells) and 1g.10c closed decode-parameter
    probing, so a beam cell's number cannot carry a cross-arm comparison; an exact decode has no
    adjacent-beam disagreement by construction. DISCLOSED ESTIMATOR CHANGE, carried in the module
    docstring and in every artifact: this maximizes over PATHS while the banked one-best maximizes
    over LABEL SEQUENCES, whose scores sum over the duration paths sharing a label sequence, so
    1g.12's decoded numbers are a new currency and are never compared to banked 1g.10 numbers.

    Acceptance, run before any cell exists (`scripts/g12_exact_decode_test.py`, 33/33): the decoded
    score AND the decoded phone sequence reproduce exhaustive enumeration of every legal
    (phone sequence, duration path) at orders 2, 3 and 4, with and without deleted-silence
    boundaries; on every utterance of a multi-utterance batch the best-path score is at most that
    utterance's own exact forward log-likelihood; a single-path fixture decodes with zero slack;
    the categorical and per-token emission paths decode identically on inputs that mean the same
    thing; and packing utterances into a batch does not change any decode. The certificate is
    one-sided by nature -- it catches a decoder that OVERSTATES a score -- so a decoder that
    understates is caught by the enumeration half instead, and the reporting path itself is
    exercised by handing the certificate an inflated score and requiring the count to fire.

    Measured cost at the real array sizes (39 phones, 96 units, order 4, one utterance per call):
    6.7 to 7.3 ms per retained token, so the whole 890-utterance selection fold (60,604 retained
    tokens) is about 7 minutes per cell, and peak resident memory is 0.18 GiB. That is well inside
    the 0.49 GiB the plan sized, because the float64 trellis is never stored -- backtracking needs
    only one base-40 digit and one sub-state bit per state per frame.

    The twenty cells that apply it are wired (`g12_readout_jobs.py`,
    `config/sae_1g_12_exp4.py`, `scripts/g12_readout_jobs_test.py` 22/22): five starts by four
    corners, all decoding under the SAME matched 4-gram automaton, because the fitting-order
    contrast is only a fitting-order contrast if the decoder is held fixed across it. Nothing is
    refitted -- the Gaussian corners read the parameters experiment 3 persisted and the table
    corners read banked emission tables -- and no label is read here. The order-4 table cells are
    pinned by path because attaching their own config would also attach the CLOSED 1g.10b/c
    prefix-beam probe graph; to stop that pinned hash-to-start mapping from being wrong silently,
    each table is checked inside the job against the manifest it wrote (the start it names, the
    fitting LM it names, the digest of the bytes loaded, and, when it was fitted under the identity
    the decoder uses, that the two automaton digests agree), and which of those checks a manifest
    supports is recorded in the artifact rather than assumed. On-disk census of the experiment-4
    graph: 4,752 jobs, 25 unfinished, and those 25 are exactly the twenty readouts and the five
    re-running cells -- the closed harness is not in front of a manager. Clause 4 is enforced in
    the producing job: on a nonzero violation count the artifacts are written and the job then
    fails, so the number is on disk for a human while every downstream reader stays blocked.

    All twenty cells finished 2026-08-24 with ZERO exactness violations and an identical
    renormalized mass of 1.274e-04 nats per emitted phone. `sym/tok` is per RETAINED TOKEN over the
    same 60,604 selection-fold tokens approach 21's column uses, so the two columns are comparable
    within a start; `worst slack` is the largest nats by which a cell's best-path score fell below
    its own exact forward log likelihood, and is a bound on the decode, not a quality statistic. No
    label is read in any of these cells and no cell is compared to a banked 1g.10 number.

    | start | emissions | fitting LM | count | symbols | sym/tok | distinct | worst slack | job |
    |---|---|---|---|---|---|---|---|---|
    | controlled/reference | gaussian | accepted-2g | 0 | 48,000 | 0.7920 | 39 | 7.50e-03 | `oBiqeae8wWC2` |
    | controlled/reference | gaussian | accepted-2g | 4 | 45,477 | 0.7504 | 39 | 7.86e-07 | `oBiqeae8wWC2` |
    | controlled/reference | gaussian | matched-4g | 0 | 48,000 | 0.7920 | 39 | 7.50e-03 | `9Um6DfyjYOTo` |
    | controlled/reference | gaussian | matched-4g | 4 | 45,876 | 0.7570 | 39 | 2.84e-06 | `9Um6DfyjYOTo` |
    | controlled/reference | table | accepted-2g | 0 | 47,684 | 0.7868 | 39 | 1.40e-02 | `I611BeAHLe5p` |
    | controlled/reference | table | accepted-2g | 4 | 45,505 | 0.7509 | 39 | 6.36e-02 | `I611BeAHLe5p` |
    | controlled/reference | table | matched-4g | 0 | 47,684 | 0.7868 | 39 | 1.40e-02 | `dD4BNLaWpw7H` |
    | controlled/reference | table | matched-4g | 4 | 46,498 | 0.7672 | 39 | 4.49e-02 | `dD4BNLaWpw7H` |
    | real/espum_seed0_update30000 | gaussian | accepted-2g | 0 | 49,599 | 0.8184 | 39 | 1.57e+00 | `CYMKVwReFIsN` |
    | real/espum_seed0_update30000 | gaussian | accepted-2g | 4 | 47,695 | 0.7870 | 39 | 2.08e-02 | `CYMKVwReFIsN` |
    | real/espum_seed0_update30000 | gaussian | matched-4g | 0 | 49,599 | 0.8184 | 39 | 1.57e+00 | `h7dasAET4GnW` |
    | real/espum_seed0_update30000 | gaussian | matched-4g | 4 | 46,775 | 0.7718 | 39 | 9.84e-02 | `h7dasAET4GnW` |
    | real/espum_seed0_update30000 | table | accepted-2g | 0 | 53,275 | 0.8791 | 39 | 1.26e+00 | `hWFBzIf5Yv8G` |
    | real/espum_seed0_update30000 | table | accepted-2g | 4 | 45,348 | 0.7483 | 39 | 1.38e+00 | `hWFBzIf5Yv8G` |
    | real/espum_seed0_update30000 | table | matched-4g | 0 | 53,275 | 0.8791 | 39 | 1.26e+00 | `65oYgLdH07Cx` |
    | real/espum_seed0_update30000 | table | matched-4g | 4 | 44,859 | 0.7402 | 39 | 1.53e+00 | `65oYgLdH07Cx` |
    | real/fingerprint | gaussian | accepted-2g | 0 | 42,449 | 0.7004 | 39 | 5.50e-01 | `lG2mByGPTl9k` |
    | real/fingerprint | gaussian | accepted-2g | 4 | 43,597 | 0.7194 | 38 | 1.13e-01 | `lG2mByGPTl9k` |
    | real/fingerprint | gaussian | matched-4g | 0 | 42,449 | 0.7004 | 39 | 5.50e-01 | `VsyhuzNPAhgr` |
    | real/fingerprint | gaussian | matched-4g | 4 | 42,540 | 0.7019 | 38 | 1.73e-02 | `VsyhuzNPAhgr` |
    | real/fingerprint | table | accepted-2g | 0 | 55,135 | 0.9098 | 39 | 2.49e-01 | `WxFKOAXtOebE` |
    | real/fingerprint | table | accepted-2g | 4 | 42,462 | 0.7006 | 39 | 5.19e-01 | `WxFKOAXtOebE` |
    | real/fingerprint | table | matched-4g | 0 | 55,135 | 0.9098 | 39 | 2.49e-01 | `CbW0CfvXWzkJ` |
    | real/fingerprint | table | matched-4g | 4 | 40,713 | 0.6718 | 39 | 1.66e+00 | `CbW0CfvXWzkJ` |
    | real/pseudo_pair_seed0 | gaussian | accepted-2g | 0 | 59,307 | 0.9786 | 36 | 6.72e+00 | `6D1qhe7DtE9m` |
    | real/pseudo_pair_seed0 | gaussian | accepted-2g | 4 | 31,503 | 0.5198 | 39 | 1.16e-02 | `6D1qhe7DtE9m` |
    | real/pseudo_pair_seed0 | gaussian | matched-4g | 0 | 59,307 | 0.9786 | 36 | 6.72e+00 | `t5PYGScA2mI5` |
    | real/pseudo_pair_seed0 | gaussian | matched-4g | 4 | 31,135 | 0.5137 | 39 | 1.84e-04 | `t5PYGScA2mI5` |
    | real/pseudo_pair_seed0 | table | accepted-2g | 0 | 55,462 | 0.9152 | 38 | 6.87e+00 | `8V9ycwOYQirF` |
    | real/pseudo_pair_seed0 | table | accepted-2g | 4 | 36,319 | 0.5993 | 39 | 4.22e+00 | `8V9ycwOYQirF` |
    | real/pseudo_pair_seed0 | table | matched-4g | 0 | 55,462 | 0.9152 | 38 | 6.87e+00 | `PKYuPuJRsVHv` |
    | real/pseudo_pair_seed0 | table | matched-4g | 4 | 36,211 | 0.5975 | 38 | 4.31e+00 | `PKYuPuJRsVHv` |
    | real/random_map_seed1000 | gaussian | accepted-2g | 0 | 43,635 | 0.7200 | 39 | 1.19e+00 | `BG3TJzElV7ui` |
    | real/random_map_seed1000 | gaussian | accepted-2g | 4 | 41,978 | 0.6927 | 39 | 4.15e-02 | `BG3TJzElV7ui` |
    | real/random_map_seed1000 | gaussian | matched-4g | 0 | 43,635 | 0.7200 | 39 | 1.19e+00 | `gpgV68WMinJF` |
    | real/random_map_seed1000 | gaussian | matched-4g | 4 | 41,418 | 0.6834 | 39 | 4.33e-02 | `gpgV68WMinJF` |
    | real/random_map_seed1000 | table | accepted-2g | 0 | 31,693 | 0.5230 | 39 | 9.59e-02 | `DaJLRyZ5N1Dn` |
    | real/random_map_seed1000 | table | accepted-2g | 4 | 41,992 | 0.6929 | 39 | 1.12e+00 | `DaJLRyZ5N1Dn` |
    | real/random_map_seed1000 | table | matched-4g | 0 | 31,693 | 0.5230 | 39 | 9.59e-02 | `IoSypXGTLch4` |
    | real/random_map_seed1000 | table | matched-4g | 4 | 41,731 | 0.6886 | 39 | 7.08e-01 | `IoSypXGTLch4` |


23. **1g.13 experiment 1: the wav2vec-U v1-equivalent stream (`g13_stream.py`, `g13_jobs.py`,
    `config/sae_1g_13_exp1.py`).** Built and launched 2026-08-24; one CPU job, no cell yet. The
    subphase exists because every 1g arm to date runs on this project's own `seg12.5` construction,
    so no 1g result separates "the training paradigm is the binding constraint" from "the
    segmentation is". This job fits the v1 clustering and the v1 PCA on the dedicated-train role
    alone, segments every bed utterance at its cluster-ID change points, and writes both twins --
    the discrete label sequence (the run's own cluster ID, a 128-symbol alphabet) and the continuous
    one (the run's mean PCA-512 vector). The two twins are the same segmentation BY CONSTRUCTION,
    because the discrete label is the run's own cluster ID rather than a second quantization.

    THE BED PARTITION, established before any code was written and asserted inside the job. The two
    banked rVAD-trimmed layer-15 dumps partition the 8,416-utterance seed bed exactly: the 2,849
    dedicated-train utterances are the train dump's own intersection with the bed
    (`W2vu2FeatureDumpJob.HyHAk3OCbruI`, 28,539 utterances, 15,427,853 retained frames, 14.71% of
    frames dropped by the trim, zero utterances dropped), and the other 5,567 -- 3,565 update, 890
    selection, 1,112 evaluation -- are the ENTIRE valid dump
    (`W2vu2FeatureDumpJob.WbaqNnxXpbRK`, 5,567 utterances, 1,612,502 frames, 14.59%). No utterance
    is in both and none is missing. That matters twice: it is the "full bed coverage" the subphase
    assumes, and it puts the only role a transform may be fitted on in the train dump, which the job
    also asserts.

    FAITHFULNESS IS TESTED AGAINST THE REFERENCE IMPLEMENTATION, not against a reading of it.
    `scripts/g13_faiss_reference_test.py` (10/10) runs under the `w2vu` environment, which holds the
    fairseq 0.12.2 the equivalence claim is about: the PCA matches `faiss.PCAMatrix(d, dim,
    eigen_power=0)` component for component up to eigenvector sign, with identical induced geometry
    and identical component variances; the assignment matches `faiss.IndexFlatL2` exactly on every
    frame; the segmentation matches `torch.unique_consecutive` and `merge_clusters.py`'s own mean
    pooling; and the substituted full-batch Lloyd k-means reaches 0.9979 of `faiss.Kmeans`'s
    within-cluster sum of squares at the same 50 iterations and 3 restarts -- reported, not
    asserted. `scripts/g13_stream_test.py` (35/35) pins the arithmetic independently, including that
    the PCA is demonstrably NOT a whitening. `scripts/g13_jobs_test.py` (34/34) builds two real
    dumps on disk and runs the job end to end.

    MEASURED, `G13StreamBuildJob.Ob8Rh8y51x9M`, complete 2026-08-24 13:05. `seg/s` is segments per
    second of RETAINED audio at the dump's 50 Hz encoder frame rate.

    | role | utterances | retained frames | segments | seg/s |
    |---|---|---|---|---|
    | update | 6,414 | 2,569,600 | 1,436,262 | 27.95 |
    | selection | 890 | 266,488 | 150,079 | 28.16 |
    | evaluation (sealed) | 1,112 | 308,759 | 175,269 | 28.38 |
    | dedicated_train (fit population, inside update) | 2,849 | 1,532,345 | 848,038 | 27.67 |
    | whole bed | 8,416 | 3,144,847 | 1,761,610 | 28.01 |

    Fit: k-means inertia 1.228e+11, best of three restarts (the third), all 128 clusters used on the
    fit population and all 128 used on the bed, rarest carrying 88 segments, zero reseeded. The run
    used its full 50-iteration budget without the labels going stable, which is also exactly what
    v1's fixed `niter=50` does -- the artifact's word "converged" for that case was misleading and
    the wording is corrected in the producing code. PCA-512 at eigen_power 0 keeps 0.9114 of the
    variance; zero degenerate components in the segment scale.

    Constants traced to the real v1 scripts and cited at the line rather than restated:
    `prepare_audio.sh:60-61` (spec CLUS128 at sample-pct 1.0), `wav2vec_cluster_faiss.py:50-69`
    (that spec means no PCA, no L2 norm, not spherical) and `:192-200` (niter 50, nredo 3),
    `prepare_audio.sh:68` with `pca.py`'s `--eigen-power` default 0, `apply_pca.py:71` (applied as
    `x @ A + b`), `prepare_audio.sh:73-74` with `merge_clusters.py`'s `unique_consecutive` and mean
    pooling. The four deviations from v1 are declared in the module and copied into the artifact:
    feature-level rather than audio-level trim, no adjacent-pair pooling (`prepare_audio.sh:76-77`
    not applied; that leg is not funded), a substituted k-means implementation, and the shared
    encoder tap. Silence is EMPTY BY DECISION and the emitted boundary artifact is all-False.

24. **1g.13 experiment 2: H1's route read re-run on the v1-equivalent stream, and the VAD-mask
    firewall (`g13_jobs.G13RoutesJob`, `g13_firewall.py`, `configs/config_sae_1g_13_exp2_v1.py`).**
    Two CPU jobs. Neither produces a result; together they are the precondition for experiment 3,
    because four of the five starts need a duration parameter and a silence mask fitted on the new
    stream and the fifth -- the controlled reference -- needs gold on a raster that does not exist
    until the firewall makes it.

    The route job emits an H1-SHAPED artifact, so every accepted downstream consumer, all of which
    reach into one JSON for the partition, the duration parameter and the mask, consumes 1g.13
    unchanged. The partition and the text-side digests are carried VERBATIM from the accepted H1
    and the copy is proved rather than asserted: every role hash is re-derived by the accepted
    reader, and T_phi is re-read and must reproduce the accepted line count and sequence digest --
    which is what makes "the matched 4-gram automaton transports untouched" a checked claim. Only
    the route is new. Its key is deliberately NOT `seg12.5/phones`, so a consumer that reaches for
    the accepted key on a 1g.13 artifact gets a KeyError instead of the wrong stream's duration;
    the constructor refuses that key outright. The silence mask is empty BY DECISION and the
    artifact carries the reason: the stream is already rVAD-trimmed, so H1's edge-enrichment split
    would select real speech units, and it is already run-collapsed, so the width statistic that
    split consumes is one for every token. The topology verdict is REPORTED, not asserted --
    whether this stream refutes one state and admits two is an open question of the subphase, so
    the repair cell that requires it is the thing that stops.

    The firewall exists because the banked dumps store no VAD mask, only a kept frame COUNT per
    utterance, and 1g.13's stream lives on the trimmed raster while the gold alignments are
    rasterized on the untrimmed one. It recomputes the trim mask deterministically and proves it
    the only way available: the recomputed kept count must equal the declared length for every one
    of the dump's 5,567 utterances. The raster convention is the dump worker's, re-derived rather
    than assumed -- `min(encoder feature-extractor length, subsampled kaldi MFCC length)`, then the
    rVADfast 0.4 two-subframe majority mask truncated to it and tail-padded as silence
    (`unsupervised_asr/w2vu2/dump_w2vu2_data.py:175-200`). The encoder length comes from the
    checkpoint's own convolution kernels and strides; the MFCC and the VAD call the real
    primitives. Gold leaves ROLE-SEPARATED -- update, selection and evaluation in three different
    files -- so a fitting job that names the update file cannot reach the other two.

    Checked against ground truth before either job was written: twelve real dev utterances put
    through this exact derivation reproduce their banked dump lengths exactly, with the encoder and
    MFCC lengths equal in all twelve.

    `ratio` is the measured lag-one mutual information over what the class allows; the registered
    admission band is ratio <= 2 under BOTH estimators (`structure_screen.ADMISSIBLE_RATIO`), so
    ADMISSIBLE means "this class can produce a dependency this strong", not "this class is right".
    The accepted seg12.5 row is quoted from the accepted H1 for contrast and is not a run of this
    job.

    | run | p | mean duration | lag-1 MI (MM) | one-state ratio | two-state ratio | verdicts | job |
    |---|---|---|---|---|---|---|---|
    | v1-equivalent stream, empty silence mask | 0.68898090 | 3.2152 | 2.2498 | 1.199 | 0.906 | one ADMISSIBLE, two ADMISSIBLE | `hStPuE1UqLK6` |
    | seg12.5, accepted H1 (for contrast, not a run here) | 0.23560298 | 1.3082 | 2.2315 | 3.185 | 1.819 | one REFUTED, two ADMISSIBLE | `Phase1gH1Job.HbxKiuBTJ8aN` |

    | run | headline | job |
    |---|---|---|
    | VAD-mask firewall | 5,567 utterances, 0 kept-count disagreements; 1,888,037 untrimmed frames to 1,612,502 kept (0.1459 trimmed, reproducing the dump's own recorded `vad_dropped_frac`); roles 3,565 / 890 / 1,112 | `Usfy2NF0LiSQ` |

25. **1g.13 experiment 3: the five registered start protocols re-derived on the v1-equivalent
    stream (`configs/config_sae_1g_13_exp3_v1.py`).** Sixteen jobs, thirteen of them new. Four of
    the five starts are the ACCEPTED job classes handed the new unit stream and the new H1-shaped
    route artifact and nothing else -- the seeds, the Sinkhorn regularization, the pseudo-pair
    length window, the espum schedule and its label-free pick rule and the fitting text all
    transport verbatim. Only the controlled reference needed new code, because it is produced
    inside `H4CalibrationPreparationJob`, whose recovery path cannot be re-run, and because it
    counts gold on the untrimmed frame raster while this stream is a segmentation of the TRIMMED
    one. `num_units=128` is passed explicitly to every espum run: the module default is a
    hard-coded 500, and on this alphabet the default would silently build a generator over 372
    units that are never observed.

    Nothing here is a result about the channel, and no cell is funded by it. The question is only
    whether the protocols transport -- five distinct, valid, usable starts on a stream carrying
    2.46x the tokens over a 128-symbol rather than 500-symbol alphabet.

    Every start is a row-stochastic 39 x 128 emission table over P(unit | phone): all entries
    strictly positive, and the largest deviation of any row sum from 1 across all five starts is
    8.9e-15. All five differ, and by wide margins -- the smallest mean total variation between any
    two of them is 0.43 (espum against pseudo-pair) and the largest 0.97.

    `mean emission entropy` is descriptive shape, in nats over each start's OWN alphabet;
    `normalised` divides by the log of that alphabet size purely so the two streams' columns can
    be set side by side. It is not a comparison currency: the readable thing is the ordering
    inside a column, and the seg12.5 column is quoted from the accepted starts for contrast, not
    re-run here.

    | start | protocol, transported verbatim | peak RSS / declared (GiB) | wall clock | mean emission entropy, nats (normalised) | accepted seg12.5 counterpart, normalised | job |
    |---|---|---|---|---|---|---|
    | fingerprint | fixed-reg deterministic, reg 0.1, 6 position bins, hard argmax | 131.9 / 192 | 18 min | 1.0393 (0.2142) | 0.3434 | `lR5Q4q1xRtqV` |
    | random-map seed 1000 | canonical marginal-random | 131.7 / 192 | 13 min | 1.4849 (0.3060) | 0.3413 | `m4sNBqlCwK2Z` |
    | pseudo-pair seed 0 | length-matched proportional, window 16, text reuse | 132.2 / 192 | 14 min | 4.6476 (0.9579) | 0.9356 | `fGmIiECLQ2XW` |
    | controlled reference | gold counts on the trimmed raster, H3's emission floor | 2.2 / 32 | 13 min | 2.9038 (0.5985) | 0.5674 | `kG9pmxczOVgF` |
    | espum, projection of the picked checkpoint | full loss, label-free pick | 120.1 / 192 (per training) | 52 min per seed | 3.9151 (0.8069) | 0.6822 | `2EB1uTDlskOy` |

    The espum arm's own fan-out. The pick rule is label-free and unchanged: weighted phone-language-model
    perplexity on the 890-utterance selection role -- ordinary perplexity divided by squared
    emitted-inventory coverage, lower better -- evaluated every 2,000 updates over 40,000.

    | run | loss | seed | selected update | weighted phone-LM perplexity | phone inventory covered | emitted tokens on the 890 | job |
    |---|---|---|---|---|---|---|---|
    | full seed 0, PICKED | full | 0 | 24,000 | 33.4666 | 39 of 39 | 146,029 | `oAOLIZZHVaVz` |
    | full seed 1 | full | 1 | 40,000 | 34.2041 | 39 of 39 | 146,650 | `18iF7DTcCNyF` |
    | full seed 2 | full | 2 | 32,000 | 33.8412 | 39 of 39 | 146,610 | `E9fojuqhcBDZ` |
    | bigram-only control | bigram_only | 0 | 14,000 | 64.1514 | 36 of 39 | 81,724 | `q59UQC0AW5Oc` |
    | accepted seg12.5 full seed 0 (contrast, not a run here) | full | 0 | 30,000 | 32.5352 | 39 of 39 | 59,751 | `97FwGhhItdpO` |
    | accepted seg12.5 bigram-only control (contrast, not a run here) | bigram_only | 0 | 40,000 | 55.4678 | 38 of 39 | 58,836 | `h4LngSZ4YvKL` |

    The controlled reference is the only new code, so its checks are listed rather than summarised.
    It reads gold from the VAD-mask firewall's UPDATE file only; the selection and evaluation files
    are never opened by it.

    | check | result |
    |---|---|
    | every labelled utterance lies inside the update role | 3,565 of 3,565 |
    | the re-derived frame assignment collapses to the banked segment sequence, position for position | 3,565 of 3,565 |
    | the route declares no silence unit | 0 |
    | T_phi reproduces the route artifact's line count | pass |
    | labelled trimmed frames, of which emitting | 1,037,255, of which 966,669 |
    | units with at least one labelled frame | 127 of 128; the remaining unit backs off to the T_phi phone prior |

26. **1g.13 experiment 4: the measured order-4 resource read on the v1-equivalent stream
    (`configs/config_sae_1g_13_exp4_v1.py`).** It runs 1g.12 experiment 1's OWN job class rather
    than a copy, so the two resource contracts are comparable by construction and not by argument:
    the same 32-way sharding, the same probe rule (the longest update utterance, ties to the higher
    ID), the same forked-child measurement so each cell reports its own peak, the same 1.5
    multiplier over the measured maxima, the same 11.5 h / 256 GiB limits, and the same
    `size_request` arithmetic decides both. Two inputs are adapted inside that class where this
    stream genuinely differs -- the codebook, fitted on RAW pre-PCA features following wav2vec-U
    v1, is carried into the observation space by the stream's own PCA (exact, because that map is
    affine); and the plain [phone, unit] starts are lifted into the two duration sub-states by
    duplicating each phone's row, which is what the topology rather than the start separates.

    THE FIRST RUN FOUND A BUG INSTEAD, and it is the reason there are two rows below.
    `G12ResourceGateJob.4iWPXMh9yoJN` sized PASS but reported ZERO reached histories for four of
    its five starts. The backward recursion in `h4_context_engine.py` was rescaled by ALPHA's
    per-frame normalizer, which bounds `alpha * beta` inside float64 only while the forward and
    backward masses stay near each other; a concentrated start over 512-dimensional observations
    breaks that, `raw / scale` overflows to `+inf`, and `alpha * beta` then evaluates `inf * 0` to
    NAN. Three independent guards failed to stop it: the log-likelihood is read off the alpha
    recursion alone so it stayed finite and plausible; `mstep_from_statistics` guards
    `weight <= 0.0` and a NAN is not `<= 0`; and the gate's own `reached = occupancy > 0.0` counts
    NAN as unreached and prints a believable zero. `gaussian_context_pass` calls
    `context_forward_backward` with exactly the arguments the occupancy probe uses, so the same
    gamma feeds the E-step's sufficient statistics -- experiment 5 would have fitted NAN means and
    variances for four of five starts with every health indicator reading clean.

    The first run's numbers are SUPERSEDED, not merged: its occupancy column is unquotable, and its
    timing measured code that no longer exists. It is kept below only to show what moved. The
    re-measurement needed no clear -- naming the subphase in the artifact moved the hash, because
    `subphase` is hash-excluded only at 1g.12's value. `4iWPXMh9yoJN` is now an orphan by hash and
    is superseded evidence, not unneeded debris.

    | run | engine | chunk (s) | h per E-step | one curve | all 5 in one process | request (GiB) | reached histories | verdict |
    |---|---|---|---|---|---|---|---|---|
    | 1g.13, re-measured `cQ3wfqsTamPP` | fixed | 128.13 | 1.1389 | 9 h | 43 h | 30 | 59,204 - 60,879 | PASS one curve; one job per start |
    | 1g.13, first run `4iWPXMh9yoJN` (SUPERSEDED) | pre-fix | 124.81 | 1.1094 | 9 h | 42 h | 30 | 0 for four of five, NAN | occupancy unquotable |
    | 1g.12, accepted `3h2iIpk6lpaB` (for contrast, not a run here) | pre-fix, unaffected | 48.88 | 0.4345 | 4 h | 17 h | 4 | 60,879 for all five | PASS one curve; one job per start |

    The cost read, against the accepted stream. This fold carries 1,436,262 retained tokens against
    584,424 (2.46x) at observation dimension 512 against 96, and the probe utterance is 893 tokens
    against 353. Measured chunk time rises 2.62x, which tracks the token count and NOT the
    dimension -- the order-4 recursion dominates and the Gaussian density evaluation adds about 7%
    on top of the token scaling. Memory rises 7.5x (30 GiB against 4), which is where the
    512-dimensional twin is paid for; it is nowhere near the 256 GiB limit. Time is where it binds:
    9 h against the 11.5 h clamp is 2.5 h of headroom where 1g.12 had 7.5.

    | measurement | 1g.13 (re-measured) | 1g.12 (accepted) |
    |---|---|---|
    | update fold, retained tokens | 1,436,262 | 584,424 |
    | observation dimension | 512 | 96 |
    | probe utterance | `422-122949-0013`, 893 tokens | `2902-9006-0015`, 353 tokens |
    | heaviest chunk (index 2 of 32) | 48,417 tokens, 201 utterances | 19,515 tokens |
    | engine peak, forked and isolated | 10.27 GiB | 1.32 GiB |
    | host peak, loading the twin and building the view | 9.14 GiB | 0.77 GiB |
    | time headroom on one curve | 2.5 h | 7.5 h |

27. **1g.12 experiment 5: the continuous observation null, fitted and DECODED as the arm is
    (`g12_nulls.py`, `configs/config_sae_1g_12_exp5_v1.py`).** The null is the Gaussian repair job
    with ONE method overridden, the observation seam, so the fitting automaton, the retained-token
    view, the census, the start means, the constrained update, the variance floor, the local
    decoder and the artifact schema are all the arm's, reached through the same calls. What
    experiment 5 adds is the DECODE half of "null": the job now persists its redrawn SELECTION-fold
    vectors in the segment twin's own shape -- retained positions carry their redrawn vector,
    dropped positions NaN -- and the exact readout cell is handed that file as `segments_pkl` in
    place of the arm's twin, so the null decodes its own acoustics. `g12_readout_jobs.py` is
    byte-identical, which is what keeps the twenty banked cells and 1g.13's shared readout out of
    it. The seam is asserted on both sides: the graph edge in the build, the persisted file's
    content hash in the artifact, and the whole-draw hash so the two orders are one bed.

    The null is fitted at BOTH fitting orders, one cheap job beyond the registration's letter, so
    contrast (b) can carry the null too; see State for the note to the planner.

    | cell | job | redrawn tokens | draw sha | selection artifact sha | exact readout | violations |
    |---|---|---|---|---|---|---|
    | accepted-2g | `G12ObservationNullJob.tDiHo9tPpn5Z` | 645,028 | 98a1cc7e | 38d68786 | `G12ExactReadoutJob.ij9vB58klqDW` | 0 |
    | matched-4g | `G12ObservationNullJob.QfLZEyTjxE6o` | in flight | - | - | `G12ExactReadoutJob.axh5u2jyP9Va` | - |

    The seam is confirmed end to end on the finished pair rather than argued: the null's exact
    order-4 decode differs from the ARM's exact order-4 decode of the same start and fitting order
    (`G12ExactReadoutJob.CYMKVwReFIsN`) on ALL 890 utterances, 54,883 decoded symbols against
    47,695. A readout handed the arm's twin by mistake would have produced the arm's decode.

28. **1g.13 experiment 5 step (a): the table arm ported to a second stream (`h4_context_diagnostic.py`,
    `h4_context_resource.py`).** Four widenings, each hash-excluded at its default -- route,
    the shared topology read, a route-keyed START POPULATION, and the accepted bigram as a funded
    fitting LM -- plus the table-arm resource gate taught to take a start population (no manifests:
    identity is the registered name, dedup is each array's own digest) and to lift a plain
    [phone, unit] start through the same `two_state_start_table` the Gaussian gate uses.

    | check | result |
    |---|---|
    | banked jobs of the ported and consuming modules, re-verified after each commit | 142, zero moved, zero unfinished |
    | v1-equivalent repair cells colliding with a banked seg12.5 cell | 0 of 10 |
    | suite `scripts/h4_context_port_test.py` | 50/50 |

    THE TABLE ARM'S OWN GATE, `H4ContextResourceGateJob.8M4rSjaBlikH`, PASS. Read beside the
    Gaussian gate on the SAME stream, same probe utterance, same heaviest chunk, same 32-way
    sharding and the same 1.5 multiplier. The two hour figures are NOT directly comparable without
    the E-step column, and that column is where the surprise is: the drivers genuinely differ, so
    both constants are right for their own arm -- the table curve records the criterion at count 0,
    records it AGAIN immediately after the symmetry-break perturbation before step 1, then at steps
    1 to 4, which is six E-steps; the Gaussian curve has no symmetry-break pass and evaluates at
    counts 0 to 4, which is five.

    | measurement | table arm | Gaussian arm |
    |---|---|---|
    | gate | `H4ContextResourceGateJob.8M4rSjaBlikH` | `G12ResourceGateJob.cQ3wfqsTamPP` |
    | heaviest chunk, order 4 (s) | 123.85 | 128.13 |
    | E-steps per count-4 curve | 6 | 5 |
    | whole-fold request at 1.5x (h) | 10 | 9 |
    | headroom against the 11.5 h clamp (h) | 1.5 | 2.5 |
    | measured peak (GiB) | 1.35 | 10.27 engine + 9.14 host |
    | request (GiB) | 3 | 30 |
    | reached histories, probe utterance | 59,319 - 60,879 | 59,204 - 60,879 |

    The v1 route's topology read fired as the planner's ruling requires: two-state ADMISSIBLE
    ASSERTED (ratio 0.9057 against an allowance of 2.4841) and one-state ADMISSIBLE REPORTED
    (1.1992 against 1.8761), the reported class travelling into the artifact rather than deciding
    anything.


## Verdicts

1. **Approach 1: one segment per text symbol is rejected.** It exceeds the registered ratio on all
   eight dev-other cells and on both estimators. This is a result about that channel shape, not about
   every Phase-1 initializer.

2. **WRONG AS AN UNSCOPED CLAIM (old Approach-1 conclusion): independent duration is rejected.** At
   the historical gold-derived duration point it is rejected on every pooled stream, passes for raw
   phones, and is split-dependent for raw characters. Subtracting the measured cross-utterance floor
   does not rescue a pooled cell. This does not fix or reject a duration fitted prospectively without
   labels.

3. **WRONG AS STATED (old Approach-1 two-state conclusion): seven dev-clean cells pass and one is
   indeterminate.** The exception rows were attached incorrectly. All eight dev-other cells pass. On
   dev-clean, five pass, `seg12.5`/characters and `seg9`/phones are indeterminate because the two
   estimators straddle the threshold, and `seg9`/characters is rejected. These are gold-duration
   diagnostics; a prospective duration must be fitted and checked without labels.

4. **Approach 2: the spectral anchor fails its registered gate.** The phone and character text
   controls pass near ceiling, while every audio stream misses the 0.85 accuracy requirement. The
   failure remains valid after the polarity correction.

5. **WRONG / UNVERIFIED (old Conclusions 2, 23, and 24): the silence pre-check demonstrated that the
   eigenvector tracked silence, and later uncertainty tests repaired the pre-check.** No catalogued
   artifact preserves the later permutation or bootstrap outputs, so neither claim can support a
   verdict. This is not load-bearing: every audio stream independently fails the saved accuracy gate.

6. **WRONG AS STATED (old Conclusion 4): the audio partition carries no information about the text
   partition.** The experiment supports only the narrower claim that this fixed binary partition and
   its registered containment/accuracy metrics did not recover a useful correspondence.

7. **Approach 3: the deterministic hard descriptor route also fails.** Its best held-out result is
   energy on `seg16`, 0.8130 mass accuracy versus the required 0.85. The +0.20-over-majority half
   passes on dev-other, but the gate is a conjunction. Both evaluated splits fail.

8. **WRONG AND SUPERSEDED (old descriptor population read):** the first report used all 2,703/2,864
   labelled utterances. That operating point is not comparable with the registered fixed fifth and
   must not support a verdict. The corrected 540/572-utterance reports give the 0.7588–0.8130 range
   in Approach 3. The all-utterance artifacts remain provenance only.

9. **WRONG (old Conclusion 5): the descriptor route replaces the failed spectral route.** That was a
   temporary next-step statement. The hard descriptor experiment subsequently ran; both exercised
   1g.4 routes failed their registered gates. Current funding status belongs in `PLAN_1G.md`.

10. **WRONG / NOT ANSWERABLE (old Conclusion 19): the six-factor soft product failed its registered
    prerequisite.** The implementation counted seven alternative descriptors for one binary target,
    rather than testing six independent memberships. No six-factor channel or prerequisite screen was
    run, so no experimental verdict exists; current funding status belongs in `PLAN_1G.md`.

11. **WRONG AND SUPERSEDED (old E5 endpoint and hard-stop interpretation).** The completed code uses
    retention 1 for the reference and 0 for a random redraw, fits and evaluates on the same
    utterances, and does not run the actual seeds or controls. Its reference endpoint moved from
    0.4865 PER to an LM-selected 0.4589 and then drifted to 0.6699 at step 30; its random endpoint
    moved from 1.0109 to 0.8409. These observations are non-decisive and fire no gate.

12. **Approach 5 establishes the seed-provenance constraint.** The original fingerprint and ESPUM
    artifacts saw the evaluation audio and are transductive rows only. Neither qualifies for the
    held-out gate or can silently inherit its original headline; `PLAN_1G.md` specifies the required
    construction-only operating point.

13. **Approach 5 localizes the preprocessing correction.** The frozen encoder, PCA/K-means, and
    per-utterance pooling need no refit. The proxy-silence mask does: its historical construction saw
    evaluation audio, so the prospective route must learn that mask on update audio and freeze it.

14. **Approach 6 selects the two-state phone channel.** On the construction-only `seg12.5`/phone
    read, the one-state channel is decisively rejected while the two-state channel is admissible under
    both estimators. Freeze `p=0.23560298` (mean duration 1.308221) and the two-state topology for the
    H3/H4 phone path. The raw-character row makes the same topology choice, but does not unblock H6's
    separately gated handoff.

15. **Approach 7 freezes ESPUM seed 0 at update 30,000.** All three full-loss ESPUM generators were
    fitted on the exact 6,414-utterance H1 update population and selected without labels on the
    disjoint 890-utterance H1 selection population. Weighted phone-language-model perplexity (lower
    is better; ordinary perplexity divided by squared emitted-inventory coverage) was 32.5352 for
    seed 0 at update 30,000, 32.5912 for seed 1 at update 38,000, and 33.1554 for seed 2 at update
    34,000. This freezes the seed/update for projection and the later 7,304-utterance construction
    refit; it is not an evaluation phone-error-rate result.

16. **Approach 10 rejects every baseline sequence-decoder setting at the label-free beam boundary.**
    Across all 12 language-model-scale/insertion-penalty settings, no adjacent beam pair from
    64/128/256/512 passes both stability clauses on all three frozen update representatives. H4 is
    not failed by this result: its mechanically admissible baseline surface is reduced to the local
    decoder, which is the only decoder family that may proceed to the selection stage.

17. **Approach 11 persists the five pre-label provisional maxima, and every one of them is a
    local winner.** All 85 starts carry a finite own-minus-donor maximum computed with no label
    read (`contains_labels: false`, `frozen_pre_label: true`), and all 85 winning tuples are
    `decoder.kind = "local"`. `PLAN_1G.md` requires a frozen-versus-next-beam winner audit only for
    a sequence winner and states that a local winner needs none, so the audit precondition standing
    in front of the controlled labels is discharged by construction rather than by running the
    audit. The label-free half of the baseline pre-evaluation-ready condition also reads positive:
    of the four real starts the selector assigns a nonzero repair count to two
    (`espum_seed0_update30000` and `pseudo_pair_seed0`, both count 4). The controlled method-level
    safety read is a label read and has not run, so pre-evaluation readiness itself is not decided
    here.

18. **The frozen own-minus-donor selector FAILS 1g.2: it is not uninformative but systematically
    INVERTED, so H4 is unresolved and no baseline maximum may freeze.** The controlled reference
    labels opened once (`H4ControlledValidationJob.Otv6GBVY8ZUj`, approach 12; 81 controlled arms,
    76 effective channels, 10,000 resamples at seed 20260822). Under `Sel` the reference channel
    scores 5.826478 while the strongest null `controlled/random_map_seed1007` scores 10.807694 and
    the strongest control of either damage family `controlled/map_q05_draw00` scores 10.848025:
    reference minus strongest control is -5.0215 with a one-sided 95 % interval of
    [-5.071922, -4.978244], so the reference loses to a content-free random map by about five
    units and the interval is nowhere near zero. Rank agreement runs the wrong way at every scale
    -- global Spearman(`Sel`, -error) -0.7493 [-0.826461, -0.622575], inside the predeclared
    starting-PER band 0.80-0.93 (48 channels) -0.5125 [-0.702978, -0.252155], and within
    trajectory [-0.928046, -0.789989]. All five NEGATIVE clauses -- the three correlation clauses
    just listed plus the two reference-versus-control comparison clauses -- take the registered
    reading for an upper bound at or below zero; none is merely unresolved. (Corrections
    2026-08-22, verifier hand-back: the point estimate reads -5.0215 not -5.03, and the earlier
    "all four correlation clauses" miscounted -- the artifact carries three correlation clauses
    and two comparison clauses. No interval, verdict or consequence moves.)

    This is a property of the frozen score, not of the reader. It reproduces the pre-label
    cross-start ordering banked before any label existed (the random-map null above the reference).
    The error instrument reads plausibly against the independently banked anchors, with the
    comparison stated at its own operating point: on the 458 dev-other utterances OF THE SELECTION
    ROLE the reference channel's count-0 PER is 0.4149 and `random_map_seed1007` reads 0.9094
    (`controlled_evidence.json`, `per_arm[*].per_by_count_and_split`), against SAE_1f's
    memoryless-oracle-map 0.4148 and random-map-control 0.8946 computed on the DISJOINT
    572-utterance dev-other evaluation fifth. Different utterance sets and, for the null, a
    different draw, so this is corroboration that the instrument measures ordinary phone error in
    the expected regime -- it is NOT an identity check, and the closeness of 0.4149 to 0.4148
    carries no more weight than that. (Correction 2026-08-22, verifier hand-back: the first
    version of this paragraph cited both numbers as an "essentially exact" match without naming
    either utterance set, and they existed in no artifact field because I had computed them in an
    unregistered console command. The split-resolved PER is now produced by the job itself and the
    read was rerun to bank it; every other number in this verdict was reproduced identically under
    the fixed seed.) So the selector is measuring something real and ranking it upside down.

    The two margin clauses pass and do not rescue it: mean selection regret is
    [0.015849, 0.020977] and selected-minus-count-0 is [-0.006181, 0.004722], both inside the 0.05
    margin. A small regret is what an inverted score looks like when the counts within one channel
    differ little -- it constrains the damage per pick, not the validity of the ranking.

    Consequence, taken from the pre-registered gate and not softened after the fact: `Sel` has
    failed, so H4 has no selector, likelihood cannot rescue it, and no contrastive update may be
    invented after labels are read. The 85 provisional maxima stay frozen and unreranked, no
    `H4SelectorFreezeJob` is built, and the 7,304-ID and 4,455-ID final refits and the 1,112-ID
    evaluation stay closed. The failure closes the tested score/channel-shape/decoder/
    representation combination only -- it does not close all Phase-1 initializers, and it is a
    decision not to fund this selector rather than a finding that repair cannot work.

19. **The count method-level safety read PASSES: every nonzero repair count is safe at the
    controlled operating point.** On the reference start's local decoder the paired differences are
    PER(1)-PER(0) [-0.003457, -0.000787], PER(2)-PER(0) [0.008906, 0.012588] and PER(4)-PER(0)
    [0.021156, 0.025474], all with an upper bound below the registered 0.05 margin, so all three
    are SAFE and the read is PASS (reference PER by count 0.3934 / 0.3913 / 0.4042 / 0.4168; count
    1 is a slight improvement). The H4-LM trigger therefore does NOT fire: it is fired by a count
    read that finds no safe count, and this one found three. Read with verdict 18 this is the
    informative split -- repair itself is not what failed at this operating point; the score that
    was supposed to choose among repairs is. Baseline H4 is still not pre-evaluation-ready,
    because that condition needs the safe nonzero count AND a valid frozen selector, and the
    selector is negative.

20. **The sequence family is UNRESOLVED, and untested rather than failed.** No mechanically
    eligible sequence tuple exists to test, because the baseline global beam table ruled all 12
    grid points ineligible (verdict 16). This blocks nothing at this boundary: every provisional
    maximum is local, so the registered local-winner exemption applies, and the sequence verdict
    would only have bound a sequence winner.

21. **DESCRIPTIVE (selects nothing): on plain PER over the selection role, no real seed beats a
    content-free control, and the best number in the whole table belongs to a control.** Read
    under approach 13's registered rule -- these values may not pick a seed, a count or a setting,
    and they do not touch the closed 1g.2 gate. Pooled PER over the 890 selection utterances, best
    count per row, with dev-other beside it (458 utterances of the selection role): ESPUM
    seed-0/update-30,000 0.8528 pooled / 0.8624 dev-other at count 4; fingerprint 0.8586 / 0.8691
    at count 4; pseudo-pair 0.8096 / 0.8105 at count 4; random-map 0.8921 / 0.9022 at count 4. At
    count 0 the same rows read 0.8573 / 0.8673 / 0.9136 / 0.9015 pooled.

    Two of these four rows are content-free CONTROLS, not candidate seeds -- pseudo-pair and
    random-map -- so the comparison that matters is candidates against controls, and it does not
    favour the candidates. The pseudo-pair control moves furthest under repair (0.9136 to 0.8096
    pooled, -0.104 from count 0, the largest movement of any row) and ends BELOW both candidate
    seeds; ESPUM and fingerprint move by -0.0045 and -0.0087. So at this operating point a
    content-free control's repaired output has lower phone error than either real seed's best,
    which is the opposite of what a content-bearing seed should show. The candidate-versus-
    random-map margin at best count is 0.0393 for ESPUM and 0.0335 for fingerprint, in the
    neighbourhood of the historical 0.0365 recorded for selected ESPUM over the stronger control.

    Provenance and limits, so this is not over-read: these are selection-role numbers on the full
    890-utterance selection bed (432 dev-clean + 458 dev-other; the artifact's own
    `selection_ids_count`), while the banked SAE_1f anchors (ESPUM 0.8580,
    fingerprint 0.8809, pseudo-pair 0.9239, random-map 0.8946) were computed on the DISJOINT
    572-utterance dev-other evaluation fifth at their own historical transductive operating
    points; the rows are in the same regime but are not the same measurement, and no held-out
    number exists here because the 1,112-ID evaluation stays sealed. Nothing in this verdict
    changes verdict 18: the gate closed on the selector, and this read neither rescues nor
    re-argues it.



22. **A14: exact order-4 repair CLEARS the measured resource gate, with a wide margin.** On the
    heaviest of the 32 update chunks (chunk 2, 19,515 retained units) one exact order-4 E-step
    takes 49.12 s and peaks at 0.67 GiB, so a full repair curve out to count 4 is requested at
    1 h and 2 GiB against limits of 11.5 h and 256 GiB. Order 3 costs 0.05 s and 0.17 GiB per
    probe utterance against order 4's 0.90 s and 0.67 GiB. Measured reachability is 60,879
    histories and 2,435,160 arcs at order 4 and 1,560 histories and 62,400 arcs at order 3 --
    one below the `1+39+39^2+39^3` and `1+39+39^2` bounds in each case, the all-BOS history
    being unreachable once the first phone is emitted. This licenses item 4 to run each
    (start, fitting LM) curve in ONE process over the whole update fold rather than as 32
    sharded jobs: 32 chunks x 49 s x 6 E-steps is about 2.6 h, inside the 11.5 h clamp.
    Artifact `H4ContextResourceGateJob.HA1vzRL7MEAz`. This is a statement about affordability on
    this machine, not about repair.

    CORRECTION 2026-08-22, the request clause only: the "1 h" first reported was the PER-SHARD
    request, and item 4 runs the whole fold in one process, so it would have asked for a
    thirty-second of what it needs and died at the wall with nothing saved -- these jobs do not
    resume. The gate now emits both figures and a separate `whole_fold_verdict`; on re-measurement
    the heaviest chunk read 50.82 s, giving 1 h per shard and 5 h whole-fold at 2 GiB, both PASS,
    with 6.5 h of headroom. Memory does not scale with the fold, because shards are processed one
    sub-batch at a time. The PASS verdict and every other number above stand.

    Second correction, same date, the probe timing only: the order-4 probe cells read 0.9366-0.9499 s
    in the artifact as it now stands, not the 0.90 s above (verifier). A few percent of node-to-node
    variation on a sub-second cell changes nothing, because the request is sized from the chunk
    rerun and not from the probe.

23. **A14: the smoothing bridge is empirically NULL, so an order-3 or order-4 difference is
    attributable to order rather than to smoothing.** On the label-free update-fold likelihood,
    add-one `legacy-2g` and matched modified-Kneser-Ney `matched-2g` agree to within 4e-5 per audio
    unit on all five D starts at all four repair counts, so rounded displays can still differ in the
    fourth decimal; the largest disagreement anywhere is `real/fingerprint` count 0, -7.293028
    against -7.293065, a gap of 3.7e-5. (Corrected 2026-08-22: the first wording said "agree to four
    decimals", which its own example contradicts.) This is the expected result rather
    than a surprise: smoothing only matters where counts are sparse, and a 39-phone inventory over
    39,630,169 lines of text leaves no sparse bigram, so both estimators sit on essentially the
    maximum-likelihood bigram. It matters anyway, because the plan's whole reason for building a
    matched family was that raising the order alone would confound order with smoothing -- and the
    confound turns out to be negligible at this corpus size. SCOPE: this is the label-free
    likelihood half of item 4 only, on the update role. It says nothing about which order decodes
    better, and the own-minus-donor and descriptive-error halves are not yet run. Artifacts are the
    20 `H4ContextRepairJob` cells under `work/speech_llm/sae/h4_context_diagnostic/`.

24. **A14: on descriptive phone error rate the matched order-3 and order-4 fitting models are not
    a usable gain over the baseline bigram at this operating point, and the smoothing bridge is
    null here too.** Bed: the 890 selection-role utterances (432 dev-clean, 458 dev-other), plain
    pooled corpus PER against the same gold as verdict 21, at the frozen local decoder with the
    accepted phone prior held fixed across every column. (a) `legacy-2g` and `matched-2g` decode to
    byte-identical phone sequences in all 15 repaired cells, which is the decode-side counterpart
    of verdict 23. (b) On the prospective reference, higher fitting order reduces the DAMAGE repair
    does rather than producing a gain: every fitting LM is worse at counts 2 and 4 than the
    unrepaired 0.3934, and order 4 shrinks that loss from +0.0234 to +0.0051 at count 4. (c) On the
    four real H3 starts, count-4 repair helps under every fitting LM (largest,
    `real/pseudo_pair_seed0`, 0.9136 to 0.8096), but changing order on top of that moves PER by at
    most 0.0062 in either direction and does not move it consistently: at count 4, order 4 beats
    the baseline bigram on espum (-0.0037), fingerprint (-0.0022) and random_map (-0.0047) and
    loses on pseudo_pair (+0.0006), while order 3 beats order 4 on three of the four. Every real
    start stays in the 0.81-0.91 PER band under every fitting LM, so the start dominates the
    fitting order by an order of magnitude. SCOPE AND STATUS: this is descriptive and reads labels.
    PLAN_1G 1g.2a Gate says in as many words that perplexity and PER cannot select order, so this
    verdict does not choose a fitting LM, does not authorize the coherent matched-4 arm (item 5) and
    does not close it; the own-minus-donor half of item 4 is still unrun. Artifacts: the 60
    `H4ContextLocalDecodeJob` cells and `H4ContextDiagnosticPerJob.IYHS4cX3j3XV` under
    `work/speech_llm/sae/h4_context_decode/`.

25. **A14: item 4 is COMPLETE on both trajectories, and its two halves DISAGREE about fitting
    order, so the arm answers no order question.** Same bed and operating point as verdict 24, plus
    the label-free own-minus-donor statistic on all 80 cells, every one eligible. (a) The smoothing
    bridge is null on this statistic too: add-one and matched Kneser-Ney at order 2 differ by at
    most 2.7e-4 anywhere (`real/fingerprint` count 1), against cell-to-cell spreads of whole units.
    It is small-but-nonzero here where the decoded sequences were byte-identical, which is the
    expected asymmetry -- the score reads the channel, the decode reads only its argmax. (b) Raising
    the fitting order LOWERS own-minus-donor at count 4 at every one of the five starts, by -0.0033
    (`real/pseudo_pair_seed0`), -0.4542 (reference), -0.4986 (espum), -1.2733 (fingerprint) and
    -1.3931 (random map). Higher is what the frozen selector maximizes, so on the label-free
    statistic the higher-order fitting model looks WORSE, and by a margin far larger than the
    smoothing bridge. (c) That is the opposite direction from verdict 24, where order 3 or 4 gave a
    small phone-error improvement on three of the four real starts. The two halves are not merely
    differently sized, they point opposite ways: within a start, the rank correlation between the
    two over the twelve repaired cells is +0.917, -0.629, +0.822, -0.993 and +0.907, and +0.112
    pooled over all 60 -- no consistent sign. (Corrected 2026-08-22 after a verifier hand-back: the
    first printed values +0.902, -0.636, +0.734, -0.979, +0.853 and +0.110 rested on no artifact and
    used positional instead of average ranks for ties. The values above are Spearman with
    average-rank ties on the unrounded banked numbers, computed and banked by
    `H4ContextAgreementJob.zd6RBdYcvzti`, which carries the convention in its own output; the
    qualitative picture and this verdict's conclusion are unchanged.) This reproduces the 1g.2 finding that the
    own-minus-donor selector is inverted rather than uninformative, now with fitting order as the
    moving coordinate instead of the arm, and it is why the label-free half cannot be read as an
    order preference in either direction. CONSEQUENCE AND SCOPE: item 4 is complete and its answer
    is that a fixed-duration diagnostic at this operating point does not identify a better fitting
    order -- the label-free statistic and the error rate disagree, and PLAN_1G 1g.2a already forbids
    the error rate from selecting order. This closes no method: per the gate, "a negative
    fixed-duration result cannot close the coherent higher-order method", so the unrun coherent
    matched-4 arm (item 5) is untouched, neither authorized nor refused, and remains the planner's
    call. Artifacts: `H4ContextOwnMinusDonorJob.SygqXhY8F2Qt` under `work/speech_llm/sae/h4_context_scores/` and
    the 600 `H4FixedTextScoreJob` cells it reads.

26. **A15: the phone-repair collapse is NOT in the posterior the 1g.9 constraints would act on --
    at count 4 every one of the five starts already satisfies both proposed targets.** On the 890
    selection utterances at repair count 4 (the planner's pre-stated clause-0 decision read),
    posterior total variation to `p_text` runs 0.0108 to 0.0736 against the 0.15 criterion and the
    posterior rate residual runs -5.5 to 0.0 percent against the 20 percent band -- all five starts
    pass both, including the one whose decode collapses. The constraint gradients say the same in
    their own currency: `lambda_equal` is 8.1e+05 to 1.5e+08, so either term would need a weight
    six to eight orders of magnitude above the likelihood's scale before the optimizer could feel
    it, which is what a penalty on an already-satisfied quantity looks like. This is the registered
    clause-0 off-ramp condition, and the ruling on it is the planner's.

27. **A15: the collapse is DECODE-RESIDENT and specific to ONE start.** At count 4, decoded total
    variation to `p_text` is 0.0402 to 0.1357 for four of the five starts, which emit 36 to 38 of
    the 39 phones; `real/pseudo_pair_seed0` emits 9 and sits at 0.6871, with a decoded rate 50.6
    percent BELOW `r_target` while the other four run 12 to 24 percent above it. At count 0 it
    emits 3 phones at total variation 0.8345 and 85.6 percent below target. Its AH overproduction
    against `p_text` is +0.835 at count 0 and +0.415 at count 4, beside the 1g.2 audit's
    +0.417 for the same cell.
    CORRECTION 2026-08-22: "independently reproducing the audit's figure" overstated it. The +0.415
    here is an excess over `p_text`, the audit's +0.417 an excess over the gold 890 unigram -- two
    different references agreeing to 0.002, which is corroboration, not identity. Every other
    number in the verdict stands. So the "babble" the 1g.2
    audit characterized is a property of one start under the frozen local decoder -- a per-unit
    argmax over `Q * prior` with run collapse, which consults neither the fitting language model
    nor the duration law -- and not a property the repair objective produces: the very same
    emission table yields a healthy posterior when the language model and duration law are summed
    over.

28. **A15: a near-zero posterior total variation is NECESSARY but NOT SUFFICIENT for a healthy
    posterior, and this table contains the counterexample.** `real/pseudo_pair_seed0` has the
    LOWEST posterior total variation of all ten cells (0.0006 at count 0, 0.0108 at count 4) and
    the worst decode. Its per-retained-unit log likelihood is also the worst of the count-4 column
    (-5.8930 update, against -5.2736 to -5.6547 for the other four) and it barely moves from count
    0 (-5.9438), while `real/fingerprint` moves -7.2930 to -5.6262 and `real/random_map_seed1000`
    -6.9771 to -5.6547. An emission table that carries little audio information leaves the
    posterior to fall back on the fitting language model's own marginal, which IS approximately
    `p_text` -- so the clause-0 statistic is satisfied most easily by the least informative
    channel. Any reading of clause 0 must carry the likelihood column beside the divergence.

29. **A15: decoded unigram distance to `p_text` does not discriminate, because the registered
    random-map control passes it best.** `real/random_map_seed1000` has the SMALLEST decoded total
    variation of all five starts at both counts (0.0291 and 0.0402), better than the reference
    (0.0854, 0.0658). Bearing on the pre-registered 1g.9 gate: clause 1's readability criterion
    (decoded total variation <= 0.30) is met by a control that carries no phonetic content, so
    clause 1 admits cells rather than evidencing them, and clause 2's comparison against the babble
    null is doing all the discriminating work. Reported for the planner; clause 1 is
    pre-registered and I am not proposing to change it.

30. **A16: 1g.10's TABLE IS BLOCKED BY ITS OWN PRE-REGISTERED EXPLANATION DUTY -- the beams
    disagree while the score margins are wide, which is the decoder-defect branch, not verdict
    28's flat-score branch.** The duty was written into the producing module before any statistic
    of this job existed: tiny margins where instability is measured confirm verdict 28's mechanism
    (an uninformative channel rides the language-model marginal, so hypotheses tie), whereas wide
    margins with persisting instability indicate a decoder defect and BLOCK any reading of the
    cells. The numbers land squarely on the second branch. Zero of 36 cells reaches the registered
    0.999 adjacent-beam agreement level (min 0.2222, median 0.6111, max 0.8889), and zero of 36
    has a median score margin at or below the registered flat threshold of 1e-3 nats per retained
    unit (min 1.210e-03, median 4.345e-03, max 1.540e-02). The job therefore prints "DECODER
    DEFECT SUSPECTED -- no cell of this table may be read until that is explained", and the
    per-cell columns below it are NOT read here. WHAT THIS LICENSES AND WHAT IT DOES NOT: it
    licenses "the registered sequence decoder does not currently produce a readable surface on
    these channels", and it does NOT license any comparison among the three rows, any statement
    about the phone-repair route's viability, or any reading of the correct-phone, total-variation
    or babble-null columns -- which is exactly what the duty exists to prevent. Two facts the
    planner will want beside this, offered as observations and not as rescues: the population is
    heterogeneous rather than uniformly wide (between 6.1 and 46.6 percent of utterances WITHIN a
    cell do sit at or below the flat threshold, so the cell medians are summarizing a mixture),
    and the agreement column is read on the probe's 27 utterances, where a single disagreeing
    utterance is 3.7 percentage points and the 0.999 test cannot be passed by anything short of
    perfect agreement -- though neither observation softens a measured agreement of 0.2222. The
    threshold and the duty are pre-registered and are not adjusted here; whether the suspected
    defect is investigated, and how, is the planner's call.

31. **A16: the positive control's own reading is internally consistent with the blocked verdict,
    and is reported only as a decoder-health observation.** `controlled/reference` produces 7 of 12
    readable cells and its best cell reaches a correct-phone fraction of 0.6010 (lam 0.5, beta 0),
    against 0.5832 for the LM-blind local decoder on the same channel -- so the full model is not
    grossly broken on a channel known to carry content. That is what makes the instability finding
    pointed rather than trivial: the decoder produces sensible content on the control while its
    adjacent beams still disagree on 1 utterance in 3. This entry records the observation; under
    verdict 30 it is NOT a licence to read the control's cells as a result, and no row is compared
    against another.

32. **A16: 1g.10a returns DISCHARGED -- the adjacent-beam instability is pruning reshuffle in a
    correct scorer, not a decoder defect, so the duty's block is satisfied.**
    `H4CrossBeamDefectJob.2pV5rHuWJW3d`. TEST D (determinism): 81 utterances across three
    disclosed cells, decoded twice each through the same entry point, ZERO violations at 1e-12
    nats. TEST U (upper bound): all 1,944 banked winners -- both beams, every probe
    utterance-cell -- rescored against their exact unpruned all-alignments totals, ZERO
    violations at 1e-6 nats, so no pruned sum exceeds the total it is a subset of. The exact
    score is an identity rather than a second implementation: its channel part is the
    `marginal_path_log_score` the chunk jobs already bank, and the language-model and insertion
    terms are alignment-independent and factor out of the path sum. WHAT THIS LICENSES: that the
    beam-512 table may be read AS DESCRIPTIVE with each quoted cell carrying its own 256-vs-512
    agreement beside it. It does NOT license cross-channel comparison, which the registration
    holds pending 1g.10b, and it does not revisit verdict 30, which stands as written for its
    date -- the block was correct until it was explained.

33. **A16: beam 512 is NOT converged, and the exact-currency column says so in the only currency
    where the question is well posed.** Rescoring both beams' winners unpruned and differencing:
    of 384 differing-winner cases exact(w512) beats exact(w256) in 352 and LOSES in 32 (8.3
    percent), median gain +0.0109 nats per retained unit, range -0.0567 to +0.1657. A wider beam
    that were merely searching a superset could never lose, so this is the non-nested kept-set
    effect the scouting round saw, now measured in a beam-independent currency: `prune` ranks
    whole prefixes by their surviving mass, and because those masses grow with the beam the
    ranking reshuffles, so a prefix kept at 256 can fall outside 512's top set. This is the
    quantitative reason 1g.10b's beam-1024 probe exists, and it is context rather than a gate.

34. **A16: 1g.10b's parity cell PASSES and ZERO of 36 cells clears the cross-channel quoting bar,
    so cross-channel comparison on the 1g.10 table stays closed and beam escalation is not
    funded.** `H4Beam1024ReadJob.tKbQ0MHLdX03`, 36 probe cells plus the parity cell, all on the
    27-utterance contract shard. PARITY: the probe class at beam 512 reproduced the banked
    production chunk's one-best sequences AND scores exactly on
    `controlled/reference|lambda=0.5|beta=-1`, zero mismatches -- so every beam-1024 column below
    is a measurement of the beam and not of a second decoder implementation. BAR: the best cell
    reaches 24 of 27 (0.8889) against the registered 26 of 27, and the median cell reaches 19 of
    27 (0.7037); no cell is quotable across channels. WHAT MOVED between doublings: median
    agreement 0.6111 (256 against 512) to 0.7037 (512 against 1024), about +0.09 per doubling,
    from which reaching the bar's 0.963 would take several further doublings at a cost that
    doubles each time -- so the escalation is declined on the measurement, not on taste. The
    within-channel paired read stays the standing currency for this table.

35. **A16: the convergence is a TENDENCY, not a per-cell fact, and that limits what the trend
    above may be used for.** Comparing each cell against its own 256-vs-512 column: one-best
    agreement rises in 24 of 36 cells, FALLS in 10 (down to -0.1852) and ties in 2; the score
    drift per retained unit falls in 25 and RISES in 11, by up to +9.333e-03 nats per unit (2.62x
    its own 256-vs-512 value). This corrects one intermediate number in the planner's 2026-08-23
    reading of the same artifact, which recorded drift as "down in every cell"; the medians it
    quoted (0.704 against 0.611) and its ruling are unaffected, since a bar cleared by no cell is
    cleared by no cell under either reading. It matters for what comes next: an extrapolation
    that treats each further doubling as monotone improvement per cell is not supported by this
    artifact, which is a second reason the escalation was correctly declined.

36. **A16: the positive insertion bonus recovers phones on the positive control and does NOT on the
    real arm -- the two rows split by SIGN, within channel and paired.**
    `H4InsertionBonusReadJob.da3bGeQIkS0R`, parity cell PASS. All four `controlled/reference`
    cells are positive with 95 percent intervals excluding zero (+0.0222 to +0.0555, best at
    lambda 2 / beta +2, improving 74.0 percent of utterances). Three of four
    `real/espum_seed0_update30000` cells are NEGATIVE with intervals excluding zero (-0.0090 to
    -0.0294) and the fourth straddles zero (-0.0021, [-0.0046, +0.0003]). On a channel known to
    carry content the bonus buys back real phones; on the real arm the extra length it pays for is
    not content. This is a WITHIN-channel paired read against each row's own beta-0 cell -- it
    quotes no comparison between the two rows, which 1g.10b's bar keeps closed (agreement on the
    contract shard runs 0.593 to 0.778 here, no cell near 26 of 27). It answers the USER-directed
    question "does an insertion bonus help" with: on this evidence, only where there was something
    to recover.

37. **A16: the stratified resampling convention made no material difference to this result, and
    that is reported rather than quietly dropped.** The planner's 2026-08-23 ruling made the
    PRIMARY interval resample within each evaluation split at the bed's fixed 432/458 counts, with
    my literal unstratified reading printed beside it as a named sensitivity. Across all eight
    cells the two intervals agree to within 1e-4 in every bound. The ruling was still the right
    call -- it fixes the convention in advance rather than after seeing which one is narrower --
    but no conclusion here rests on it, and a reader should not infer that it was load-bearing.

38. **A17: the continuous twin is the frozen pipeline's own, asserted rather than argued.** At the
    Ward segmentation the re-assigned segment means reproduce the frozen `seg12.5` stream
    bit-for-bit on all 8,416 utterances, run through `repr_pool.pool_utterance` itself rather than
    a re-derivation of it. The emission swap 1g.11 tests is therefore one change and not two: the
    features, the PCA basis and the segmentation are the same objects the discrete arm reads.

39. **A17: the token-level twin costs nothing, and the reason is a theorem rather than luck.**
    All 919,248 tokens re-assign to their frozen code, including every one of the 2,184 whose mean
    spans two Ward segments the merge had absorbed. This was predicted to fall BELOW 1 and does
    not: an absorbed pair is two adjacent segments assigned the SAME centroid, a nearest-centroid
    cell is a convex polytope, and the token mean is a frame-count-weighted convex combination of
    two points inside that cell -- so it cannot leave it. The measurement is what establishes that
    no tie on a cell boundary bites in practice. Consequence for the registration's count-identity
    clause: the token reading satisfies it by construction AND agrees with the frozen assignment
    exactly, so choosing it costs no fidelity and there is no residual to disclose.

40. **A17: the registered 128-component truncation is vacuous on this stream.** The frozen basis
    carries 96 components, so keeping "the leading 128" keeps all of them; the primary cell and the
    registration's optional full-dimension sensitivity cell are the same cell, and no truncation
    axis exists to sweep.

41. **A18: the constrained update improves the criterion on every registered cell.** All six
    start-covariance cells rise from count 0 to count 4. Largest rise `espum_seed0_update30000`
    per_row +6,796,844 (-79,791,528 to -72,994,684); smallest `pseudo_pair_seed0` tied +2,575,212
    (-80,515,490 to -77,940,278). This says the EM implementation ascends on the real fold as the
    enumerated fixtures said it would; it says nothing about whether the decoded output carries
    content, which needs experiment 3's nulls.
    [2026-08-23 correction, on the verifier's hand-back: the original parenthetical named
    `pseudo_pair_seed0` the largest rise and `controlled/reference` the smallest. Both were wrong --
    I ranked the cells by their count-0 log-likelihood MAGNITUDE and then labelled that ranking as
    one of rises. The verdict's claim, that every cell ascends, is unaffected. The six rises are
    +2,575,212 (`pseudo_pair_seed0`), +3,529,805 (`controlled/reference`), +4,715,910
    (`random_map_seed1000`), +4,995,400 (`fingerprint`), +5,064,753 (`espum` tied), +6,796,844
    (`espum` per_row).]

42. **A18: clause 4's honesty line reads clean -- no cell sits on the variance floor.** `floor
    share` is 0.0000 in all twelve cells, so no emission row won by variance collapse and the
    registered floor is doing no work at this operating point. The one numerical event anywhere is
    68 clipped emissions of about 45.6 million on `espum_seed0_update30000 per_row 4`, which is the
    per-row relaxation widening the density spread past float64's range.

43. **A18: the count-0 decode of `real/pseudo_pair_seed0` is degenerate and the repair recovers
    it.** It emits 0.0152 symbols per retained token over 3 of 39 phones at count 0, against 0.80
    to 0.92 over 37 to 39 phones for every other start, then reaches 0.7411 over 39 of 39 by count
    4. DESCRIPTIVE ONLY: no gold is read in this job, and a full inventory at a plausible rate is
    exactly what a content-free control also produces (verdict 29), so none of this is evidence of
    content. That reading waits on the babble null and the paired gold read.

44. **A18: no cross-arm comparison is licensed from this table.** `sym/tok` is per RETAINED TOKEN;
    the table arm's audited collapse is quoted as decoded length against GOLD. Different
    denominators, so the two numbers may not be set beside each other, and the paired
    per-utterance contrast clause 3 registers is experiment 4's.

45. **A19: clause 3, the decision read, FAILS -- and it fails on the CONTROL, not on the arm.**
    The selected real start's Gaussian gain over its own banked table cell is +0.0098
    [+0.0058, +0.0137], which excludes zero in the Gaussian arm's favour, so clause 3's first
    condition passes. Its second condition does not: the content-free random-map control gains
    +0.0251 [+0.0202, +0.0302] against the same comparator -- 2.6 times larger, with an interval
    lying ENTIRELY ABOVE the arm's. The registration's word "comparable" carries no number and
    none is needed here, because the control does not merely match the arm's gain, it exceeds it
    with non-overlapping intervals. Under the registration this licenses "continuous emissions are
    not funded at this operating point; evidence toward the training paradigm as the binding
    constraint, jointly with the banked oracle gap (0.4148 achievable on this stream, 0.85+
    found)" -- never "the paradigm cannot work". The attribution is conditional on the shared
    `seg12.5` segmentation both arms inherit.
    [2026-08-24 completion, not a correction: the deciding control is itself clause-1
    INADMISSIBLE -- `gaussian random_map tied 4` decodes 0.7832 of gold length, under the 0.80
    floor. I did not surface that when I wrote this verdict and should have, since a reader of
    verdict 45 alone would not know it. The planner ruled it counts as registered
    (`PLAN_1G.md` 1g.11 Status 2026-08-24, ruling ii): the registration scopes clause 1 to
    clause-2 readability and names clause 3's controls with no admission precondition, so
    filtering it out after seeing that doing so flips the verdict would be an unregistered gate
    edit -- and would delete the very length pathology the control exists to expose. The verdict's
    claim is unaffected either way, because verdict 46's positive control fails independently.]

46. **A19: on the one channel known to carry content the Gaussian swap LOSES phones.**
    `controlled/reference` at count 4 is -0.0208 [-0.0260, -0.0154] paired against its own table
    cell; pooled PER 0.4168 for the table arm against 0.4429 for the Gaussian one. The gate's
    positive control therefore runs in the OPPOSITE direction to the small positive deltas on the
    real starts, which is what identifies those deltas as a length or rate effect rather than the
    geometric inductive bias the subphase was testing for.

47. **A19: no Gaussian cell shows content by clause 2, and the reference cell fails it for a
    reason a reader must not misread.** Only `table|controlled/reference` shows content, at both
    counts (margins over the babble p99 of +0.4700 and +0.4415). `gaussian|controlled/reference`
    has margins of +0.4952 and +0.3990 -- far above the registered 0.05 bar -- but is recorded
    `c2 = .` because clause 2 reads READABLE cells only and clause 1 excludes it: its decoded
    length is 0.7947 and 0.7735 of gold, under the [0.80, 1.25] band. The Gaussian reference
    channel CARRIES content; it is inadmissible on length, not empty, and quoting its clause-2
    column without this sentence would invert the fact.

48. **A19: clause 1 excluded the content and admitted the babble -- verdict 29's warning running
    in reverse.** Seven of twenty-four cells fail admission, and two of them are the Gaussian
    reference cells, which hold the two highest correct-phone fractions in the whole table (0.6502
    and 0.5571). Meanwhile both observation-null cells and every real-start cell within 0.04 of
    the babble bar pass. Verdict 29 recorded that a content-free control passes clause 1 best;
    this adds the other half, that a content-bearing cell can fail it, and confirms clause 1 as
    admission only in BOTH directions.

49. **A19: the criterion ascends where the PER worsens, including on observations with no
    structure left in them.** The observation null -- every retained token's vector redrawn from
    the corpus marginal, nothing else changed -- rises +4,119,582 in log-likelihood from count 0
    to count 4 (-79,970,371.2 to -75,850,789.7), against +5,064,753 for the real arm on the same
    start, while its PER goes the wrong way, 0.8946 to 0.9252. On `controlled/reference` the same
    repair rises +3,529,805 and worsens PER from 0.3498 to 0.4429. Verdict 41 established that the
    constrained update improves the criterion on every registered cell; this establishes that
    roughly four fifths of that improvement survives destroying the observations, so criterion
    ascent is not evidence of content and verdict 41 must never be quoted as if it were.

50. **A19: the Gaussian arm DOES read the acoustics -- which is what makes clause 3's failure
    informative rather than vacuous.** Redrawing the observations costs the arm 0.0761 of pooled
    correct-phone fraction on the selected real start (0.1509 with the real continuous twin,
    0.0748 with the marginal-redrawn one), and the null's paired delta against the banked table
    cell is -0.0772 [-0.0817, -0.0728]. So the continuous features carry signal the model uses;
    the finding is that the categorical table already extracts as much of it, not that the
    emission swap was inert.

51. **A19: clause 4 reads clean across every cell of both experiments.** The share of variance
    components at the M2 floor is 0.0000 in all twelve experiment-2 cells and both observation-null
    cells, so no emission row in this subphase won by variance collapse and the registered floor
    did no work at this operating point. The registered babble null is also stable at its own
    operating point: the 100-draw bar reproduces the 1,000-draw bar to within 6.1e-04 in the worst
    cell, an order of magnitude below the smallest clause-2 gap in the table.

52. **A20: the Gaussian order-4 repair curve is affordable one start at a time and NOT as a
    population.** Measured, not estimated: 48.88 s on the heaviest of 32 update chunks, so 0.4345 h
    per whole-fold E-step and 4 h for a count-4 curve at the standing 1.5 multiplier, against 17 h
    for all five starts in one process and an 11.5 h clamp that cannot be raised. Verdict PASS for
    one curve, RESOURCE_INFEASIBLE for the single-process shape. This is a statement about this
    machine and about nothing else; the build shape follows it rather than the reverse.

53. **A20: the continuous emission model does not narrow the context the recursion visits.** On the
    longest update utterance every start reaches 60,879 histories and 2,435,160 arcs -- identical
    across all five starts and identical to what the accepted order-4 TABLE gate measured on its own
    probe utterance. The reachable order-4 identity count is 1+39+39^2+39^3 = 60,880, so the arm
    visits all but the all-BOS start history. Cost at order 4 is therefore a property of the state
    space, not of how peaked the emission model is, which is why one 4 h request covers every cell.

54. **A20: the Gaussian arm's order-4 cost is within 4% of the table arm's at the same order.**
    48.88 s here against the accepted gate's 50.82 s on the same 32-way sharding of the same update
    role. The per-token diagonal-Gaussian density over 78 rows and 96 components is negligible
    beside the history contraction, so the emission swap is nearly free at this order -- which
    matters for the attribution the subphase is built on: neither corner of the order-4 column is
    handicapped by its own cost.

55. **A21: the order-2 instantiation of the new context path IS the banked 1g.11 code path, on the
    real fold and not only on fixtures.** All five bigram corners reproduce their banked 1g.11 cell
    at both counts: criterion relative differences between 1.9e-16 and 2.6e-15, six orders of
    magnitude inside the declared 1e-9 bar, and ZERO decoded-symbol disagreements across all 890
    selection utterances in all ten cell-count pairs. The two paths differ in their dynamic
    programming (dense 78-by-78 transition against the history contraction) and in their batching
    (256 utterances per call against one), so agreement at round-off is the strongest statement
    available -- bit-identity is not, and was never claimed. What this buys is the attribution the
    subphase rests on: the order-4 column is produced by the same code as the order-2 column, so a
    difference between them is the fitting order.

56. **A21: the fitted Gaussian parameters 1g.11 never persisted now exist for all five starts.**
    That is the whole reason experiment 3 was funded, and it is discharged: `parameters.npz` per
    cell holds `mu` and `var` at counts 0 and 4, so no later experiment in this subphase refits
    anything to read a bigram cell. The criterion rises at every start (e.g. `controlled/reference`
    -77,517,283.6 to -73,987,478.3) and the variance floor share is 0.0000 in every cell, matching
    verdict 51 -- these are 1g.11's own numbers recovered, not new evidence, and they are quotable
    only within their own arm.

57. **A23: the v1-equivalent stream reproduces the published wav2vec-U v1 token rate on this bed.**
    28.01 segments per second over the whole 8,416-utterance seed bed, against the ~28 segments per
    second the v2 paper's Table 1 measures for the v1 pipeline on LibriSpeech dev-other. That anchor
    was written into the plan BEFORE this job ran and is a different corpus split measured by
    different people, so hitting it to three significant figures is an independent check on the
    construction rather than a fit to it -- and it is the one number that could have shown the
    "equivalent segmentation" claim to be wrong at a glance. The rate is stable across roles (27.67
    to 28.38), so no role is segmented differently from the fold the transform was fitted on. Two
    consequences follow immediately. First, the stream is 2.24x seg12.5's rate and its update role
    carries 1,436,262 segments against the 584,424 retained update tokens 1g.12 fits on, i.e. 2.46x
    the tokens per E-step; at 1g.12's measured 0.4345 h per whole-fold order-4 E-step that projects
    to about 1.07 h per E-step and, at five E-steps and the standing 1.5 multiplier, about 8 h per
    curve against the 11.5 h clamp -- inside it, but with little room, which is exactly why 1g.13
    experiment 4's resource read is mandatory and is not discharged by this projection. Second, the
    alphabet does not collapse: all 128 clusters are used on the bed and the rarest still carries 88
    segments, so the discrete twin is a real 128-symbol stream rather than a nominal one.

58. **A21: at the matched 4-gram, as at the accepted bigram, the criterion ranks the content-free
    control ABOVE the gold-informed reference, so criterion magnitude carries no content read.**
    All ten matched-4g rows exist and every cell rises from count 0 to count 4, which on its own
    says only that four repair steps of a hill-climbing update climb the hill they are climbing. The
    ordering is the informative part. Gains at the matched 4-gram, all on the same 584,424 retained
    update tokens under the same LM and therefore comparable across starts:
    `real/espum_seed0_update30000` +4,616,970, `real/fingerprint` +4,535,461,
    `real/random_map_seed1000` +4,439,137, `controlled/reference` +3,268,660,
    `real/pseudo_pair_seed0` +2,490,090. The random map is the registered content-free control and
    the controlled reference is the only start built from labels, and the control gains 1.36x what
    the reference gains. The accepted-bigram corner ranks them the same way (+4,715,910 against
    +3,529,805, 1.34x), so raising the fitting order from 2 to 4 does not change the ordering, does
    not shrink the control's lead, and does not make the criterion a content statistic. This is the
    same pattern that failed 1g.11's clause 3 on its control, now shown to survive the strongest LM
    operating point this campaign owns. NOT A CLAUSE VERDICT and not a statement about the decoded
    output: clauses 1, 2 and 3 need the exact order-4 readout and the re-banked nulls, which are
    experiments 4 to 6. It also does not license "the repair objective is worthless" -- it licenses
    only "this criterion, at this operating point, does not rank starts by content."

59. **A22: the exact order-4 readout is certified on every one of the twenty cells, and the
    fitting LM provably does not reach the decode.** Zero exactness violations in all twenty, and
    an identical renormalized mass of 1.274e-04 nats per emitted phone, so no cell's number is a
    beam artifact and none is a different automaton read twice -- which is the thing 1g.10's
    prefix-beam family could not have, and the reason its whole line was closed. A second check
    falls out of the design and reads clean: at repair count 0 nothing has been fitted, so a cell's
    accepted-2g and matched-4g rows must be identical because both decode under the same order-4
    automaton. They are, to the symbol, for all five starts in both emission models. That is the
    positive control for "the fitting-order contrast is a fitting-order contrast" -- had the
    fitting LM leaked into the decoder, these ten pairs would differ.

60. **A22: the near-total collapse of the pseudo-pair channel is a READOUT property, not a channel
    property -- but this says nothing yet about whether either readout is correct.** The
    unrepaired pseudo-pair Gaussian channel decoded LM-blind emits 0.0152 symbols per retained
    token using 3 of 39 phone types (approach 21). The SAME channel, same fold, same 60,604
    retained tokens, decoded by the exact order-4 LM-aware readout emits 0.9786 symbols per token
    using 36 of 39 (approach 22). No parameter differs between those two rows; only the readout
    does. So the "collapse" that 1g.9 localized and 1g.10 tried and failed to characterize with a
    beam is, in this cell, produced by reading the channel without its language model. Everywhere
    else the two readouts move modestly and mostly in one direction -- the LM-aware decode emits
    FEWER symbols than the LM-blind one at every other Gaussian cell (controlled reference 0.7832
    to 0.7570 at count 4, fingerprint 0.7964 to 0.7019, random map 0.7733 to 0.6834, espum 0.8626
    to 0.7718 at the matched 4-gram) -- so the rescue is specific to the collapsed cell rather
    than a uniform effect of adding a language model. WHAT THIS IS NOT: a claim that the LM-aware
    output is better. Symbol rate and inventory usage are not correctness, no gold is read in any
    of these cells, and a decoder that emits a plausible number of plausible symbols can still be
    wrong about every one of them. The correctness read is experiment 6, on the 890 with gold,
    against the re-banked nulls; clause 3(a)'s readout contrast is decided there and not here.

61. **A24: on the v1-equivalent stream a one-state channel is NOT refuted, where on seg12.5 it is
    -- the segmentation, not the phone inventory, is what made within-symbol duration structure
    necessary.** The two streams carry almost identical lag-one mutual information (2.2498 nats
    here against the accepted H1's 2.2315), so the raw dependency did not change. What changed is
    what a one-state channel is ALLOWED to produce: 1.876 nats here against 0.701 on seg12.5, so
    the ratio falls from 3.185 to 1.199 and lands inside the registered admission band of 2. The
    mechanism is visible in the fitted duration and needs no interpretation: the within-symbol pair
    rate rises from 0.2356 to 0.6890 and the mean duration from 1.31 to 3.22 audio tokens per phone,
    which is what a 28-segments-per-second stream must give against a 12.5-per-second one. When most
    adjacent token pairs lie inside one phone, a model with no within-symbol substructure already
    accounts for most of the adjacent-token dependency. The two-state class remains ADMISSIBLE on
    both streams (ratio 0.906 here, 1.819 there), so nothing is refuted that was previously
    admitted. CONSEQUENCE FOR THE TOPOLOGY: none by itself. The minimum-duration-2 scorer topology
    is standing by the USER's 2026-08-15 ruling and is not a function of this measurement; what this
    verdict changes is only the EVIDENCE for it, which on this stream is absent rather than
    contrary. CONSEQUENCE FOR THE BUILD: the 1g.12 repair cell refuses to fit unless the route reads
    one-state REFUTED and two-state ADMISSIBLE, which is seg12.5's signature, so a 1g.13 cell stops
    at that guard as designed -- see the planner proposal in State.

    UPDATE 2026-08-24, the pointer above: the planner ruled on that proposal and the guard is
    amended, so a 1g.13 cell no longer stops at it -- the guard now asserts each route's own
    registered expectation (v1-equivalent: two-state ADMISSIBLE asserted, one-state REPORTED) and
    the ruling is in PLAN_1G.md 1g.13 Status, not in State.

62. **A25: all five registered start protocols transport to the v1-equivalent stream unchanged and
    produce five distinct, valid starts -- the transport question of experiment 3 is answered
    yes.** Four of the five ran as the accepted job classes with no code change at all; only the
    controlled reference needed new code, and for a reason that is about where gold lives rather
    than about the protocol. Every start is a strictly positive row-stochastic 39 x 128 emission
    table (largest row-sum deviation 8.9e-15) and no two are close: minimum pairwise mean total
    variation 0.43. The concentration ordering of the accepted stream survives intact --
    fingerprint most concentrated, then random-map, then the controlled reference, then espum,
    with pseudo-pair very near uniform -- at normalised entropies 0.2142 / 0.3060 / 0.5985 /
    0.8069 / 0.9579 against seg12.5's 0.3434 / 0.3413 / 0.5674 / 0.6822 / 0.9356. The one
    substantive shift is the espum start, which is markedly more diffuse here (0.8069 against
    0.6822). This verdict is about usability only: it says the factorial CAN be run on this
    stream, and says nothing about whether any of these starts carries content.

63. **A25: the espum selection perplexity is NOT comparable across the two streams, and inside
    this stream the full loss beats its registered control decisively.** The cross-stream numbers
    look adjacent -- 33.4666 here against the accepted 32.5352 -- and reading them as "slightly
    worse" would be a currency error: weighted phone-LM perplexity is per EMITTED token, and for
    the same 890 selection utterances this generator emits 146,029 tokens against the accepted
    stream's 59,751, a factor of 2.44 that tracks the stream's own token rate. Two different
    length regimes, so no cross-stream ranking is licensed from this pair and none is taken. The
    comparison that IS internal reads clean: against its own bigram-only control at the same
    seed and schedule, the full loss gives 33.4666 against 64.1514 and covers all 39 phones where
    the control reaches 36, reproducing the qualitative behaviour the accepted stream showed
    (32.5352 against 55.4678, 39 against 38). The three full seeds agree closely (33.4666,
    33.8412, 34.2041), so the label-free pick is not choosing between materially different
    generators.

64. **A25: the espum arm's cost does NOT scale with the stream's token rate, so experiment 4's
    resource question is untouched by this experiment's timings.** Each full-loss training took 52
    minutes against the accepted stream's 47, despite the 2.46x token count, because the espum
    schedule is a fixed 40,000 updates at fixed batch sizes over TEXT -- the stream enters only
    through the periodic selection decodes. The construction-only starts likewise ran in 13-18
    minutes at a 132 GiB peak against the 192 declared, sized from H1's measured 141.52 GiB
    same-scale reference. None of this speaks to the repair curve, which is the leg that visits
    every segment and where the 2.46x actually bites; the mandatory order-4 resource read stands
    exactly as registered.


65. **A26: one order-4 repair curve on the v1-equivalent stream FITS, and the binding resource is
    time with 2.5 hours of margin.** The measured heaviest chunk is 128.13 s over 48,417 retained
    tokens, which projects to 1.1389 h per whole-fold E-step and, at five E-steps and the standing
    1.5 multiplier, to a 9 h request against the 11.5 h clamp. Memory is not the constraint
    anywhere: 10.27 GiB in the isolated engine plus 9.14 GiB to hold the twin gives a 30 GiB
    request against a 256 GiB limit. All five starts in one process is 43 h and stays
    RESOURCE_INFEASIBLE exactly as on the accepted stream, so the build shape is one job per start.
    This is a statement about this machine and this fold, not about the arm; it funds experiment 5
    at that shape and nothing else. The margin is thin enough to matter, and it is thin in the
    direction that costs the whole fold: the fitting job caps its own request at the clamp, and
    these jobs do not resume, so the first real cell's wall clock is read against this projection
    before the rest are launched.

66. **A26: the order-4 cost tracks the TOKEN count and not the observation dimension, which is why
    a 5.3x wider observation cost 2.62x rather than 13x.** The fold grew 2.46x in retained tokens
    and 5.33x in dimension against 1g.12, and the measured chunk time grew 2.62x -- about 7% above
    the token scaling alone. The recursion over the order-4 context, not the Gaussian density
    evaluation, is what the E-step spends its time on, which is the same reading verdict 53 gave
    from the other direction on the accepted stream. Memory behaves oppositely and rises 7.5x,
    because that is where the 512-dimensional twin is actually paid for. The practical consequence
    for any future stream on this route: a rate change is a time risk and a dimension change is a
    memory risk, and only the first one is near a limit here.

67. **A26: the reachable order-4 context is fully visited from every start, so no start prunes the
    automaton the fitting model gets.** With the engine fixed, the five starts reach 59,204 to
    60,879 histories of the 60,879 the accepted stream reached from all five -- at most 2.8% below
    it (1,675 of 60,879), with the near-uniform pseudo-pair start reaching exactly the same
    60,879. The ordering is the expected one, the most concentrated start (fingerprint) visiting
    fewest, and three of the five report IDENTICAL counts (59,319 / 2,372,760) -- which is itself
    the caution below, since a column that ties across unrelated starts is reading structural
    support and not start quality. Two things this
    does NOT say: it is a statement about which histories carry any posterior mass at all, not
    about how much, so it is not a measure of how well any start uses the context; and it is not
    comparable across streams as a quality read, only as the observation that neither stream's
    starts collapse the context. One incidental reading worth recording because it will recur: on
    the single probe utterance the fingerprint start leaves one of the 78 emission rows with
    exactly zero posterior mass. On one 893-token utterance that is ordinary -- no utterance visits
    every phone -- but at fold level `mstep_from_statistics` refuses a zero-weight row by design,
    and that refusal is a collapse to report rather than a guard to soften.


68. **A26: the backward-recursion fix changed the NORMALIZER and not the QUANTITY, measured
    against a banked artifact rather than argued from the algebra -- so no 1g.11 or 1g.12 number
    moves.** `G12EngineEquivalenceJob.sWWDLbPKglfP` recomputes the accepted
    `G12ResourceGateJob.3h2iIpk6lpaB`'s own five probe cells under the current engine: all five
    reproduce that artifact's log-likelihood to a difference of exactly 0.0 and its history
    occupancy at 60,879 exactly. Recomputed again under the superseded normalizer, which does not
    overflow at 96 dimensions and 353 tokens, the two posteriors agree to at worst 8.882e-16
    against a 1e-12 tolerance set at the rounding such a recursion actually reaches. The separation
    is asserted in the same job and is what keeps the rest from being vacuous: on a peaked case in
    1g.13's shape (893 tokens, log-density gap 700) the superseded normalizer is NOT finite and the
    current one is, so this job could not pass against an engine that never carried the defect.
    Independently, all ten banked 1g.12 repair cells were checked to carry finite fitted
    parameters. NO VERDICT IS MARKED WRONG, because none rested on a number that moved -- what the
    bug destroyed was work not yet done, which is the only reason this is a footnote to 1g.13
    experiment 4 rather than a correction to 1g.12.

69. **A28: the TABLE arm on the v1-equivalent stream is not cheaper in wall clock than the Gaussian
    arm -- it asks for MORE hours and has LESS headroom -- and is cheaper only in memory.** The
    measured order-4 gate `H4ContextResourceGateJob.8M4rSjaBlikH` reads PASS at 10 h for the whole
    fold in one process against the 11.5 h clamp and 3 GiB against 256, beside the Gaussian arm's
    9 h and 30 GiB on the same stream, the same probe utterance and the same heaviest chunk. Per
    E-step the table arm IS slightly cheaper -- 123.85 s against 128.13 s on the heaviest chunk,
    3.3% -- but its count-4 curve carries six E-steps against the Gaussian's five, because the
    table driver records the criterion again immediately after the symmetry-break perturbation and
    the Gaussian driver has no symmetry-break pass. Both constants are right for their own driver;
    what is wrong is reading one arm's hour figure against the other's without that column. The
    consequence for experiment 5 is the opposite of the prior that made this gate look like a
    formality: the TABLE cells have 1.5 h of headroom against the clamp where the Gaussian cells
    have 2.5, so the wall-clock read on the first launched cell matters more for the table corners
    than for the continuous ones. This is a measurement of this machine and this fold, never a
    statement about the arms.


## Catalog

1g.12 experiment 5, the continuous observation null at both fitting orders (approach 27): the null
cells `work/speech_llm/sae/g12_nulls/G12ObservationNullJob.tDiHo9tPpn5Z` (accepted-2g) and
`.QfLZEyTjxE6o` (matched-4g), each carrying `observation_null.json`, `null_segments.pkl`,
`parameters.npz` and its own LM-blind `hypotheses.json`; their exact readouts
`work/speech_llm/sae/g12_readout_jobs/G12ExactReadoutJob.ij9vB58klqDW` and `.axh5u2jyP9Va`. Code
`sae/g12_nulls.py`, `configs/config_sae_1g_12_exp5_v1.py`, `scripts/g12_nulls_test.py` (32/32) at
speech-llm `1c25f58` and `c4d3f13`.

1g.12 experiment 6, the gate reader (approach 27, unrun until experiment 5 closes):
`work/speech_llm/sae/g12_evaluate/G12EvaluateJob.oStN2ghRhR7l` (`evaluate.json`, `evaluate.txt`);
code `sae/g12_evaluate.py`, `configs/config_sae_1g_12_exp6_v1.py`,
`scripts/g12_evaluate_test.py` (37/37) at speech-llm `c4d3f13`.

1g.13 experiment 5 step (b), the TABLE arm's own measured order-4 read on the v1-equivalent stream,
PASS at 10 h and 3 GiB (approach 28, verdict 69):
`work/speech_llm/sae/h4_context_resource/H4ContextResourceGateJob.8M4rSjaBlikH`
(`resource_gate.json`, `resource_gate.txt`); the port itself in `sae/h4_context_diagnostic.py`,
`sae/h4_context_resource.py`, `sae/g12_repair_jobs.py` with
`scripts/h4_context_port_test.py` (50/50) at speech-llm `a8279ef`, `f651ffd`, `1d8f669`.

1g.10c positive insertion-bonus cells, parity PASS, sign split between rows (verdicts 36-37): `work/speech_llm/sae/h4_insertion_bonus/H4InsertionBonusReadJob.da3bGeQIkS0R` (`insertion_bonus.json`, `insertion_bonus.txt`); 256 beam-512 chunks + 8 probes + 1 parity chunk under `work/speech_llm/sae/h4_insertion_bonus/`, merged by the production `H4SequenceDecodeMergeJob`; code `sae/h4_insertion_bonus.py`, `scripts/h4_insertion_bonus_test.py` (14/14) at speech-llm `3d395de`.

1g.10b beam-1024 convergence probe, parity PASS and 0 of 36 cells quotable (verdicts 34-35): `work/speech_llm/sae/h4_beam1024_probe/H4Beam1024ReadJob.tKbQ0MHLdX03` (`beam1024.json`, `beam1024.txt`); 36 probe chunks + 1 parity chunk under `work/speech_llm/sae/h4_beam1024_probe/H4Beam1024ProbeChunkJob.*`; code `sae/h4_beam1024_probe.py`, `scripts/h4_beam1024_probe_test.py` (8/8) at speech-llm `8e2c841`.

1g.10a cross-beam defect diagnostic, verdict DISCHARGED (verdicts 32-33): `work/speech_llm/sae/h4_cross_beam_defect/H4CrossBeamDefectJob.2pV5rHuWJW3d` (`cross_beam_defect.json`, `cross_beam_defect.txt`); code `sae/h4_cross_beam_defect.py`, `scripts/h4_cross_beam_defect_test.py` (9/9) at speech-llm `294c8fc`.

1g.10 full-model decode read, BLOCKED by its explanation duty (verdicts 30-31): `work/speech_llm/sae/h4_full_model_decode/H4FullModelDecodeReadJob.MXhi20TtG1I0` (`full_model_decode.json`, `full_model_decode.txt`); 1,152 beam-512 chunks + 36 merges + 36 single-shard beam-256 probes under `work/speech_llm/sae/h4_decode_jobs/`; code `sae/h4_full_model_decode.py`, `configs/config_sae_1g_h4_full_model_decode_v1.py`, `config/sae_1g_h4_full_model_decode.py`, `scripts/h4_full_model_decode_test.py` (9/9) at speech-llm `359dbeb`.

| evidence | concrete artifact or source |
|---|---|
| 1g.12 experiments 2 and 3, the ten Gaussian repair cells at fitting orders 2 and 4 (approach 21, verdicts 55-56, 58) | accepted-2g `work/speech_llm/sae/g12_repair_jobs/G12GaussianContextRepairJob.` `0nngx4f5pX69`, `iZaUwq3DQVjj`, `OBwHBeOmwYU5`, `OyooGnuVi7EK`, `uczGmykabX6i`; matched-4g `.8OzLoDv4PPlt`, `.BrQtRIAKaWwU`, `.dDKq6J6AQEIP`, `.DgOI3SI1cwph`, `.kHwPYElOcCPr` (each `parameters.npz`, `repair.json`, `repair.txt`, `hypotheses.json`); code `sae/g12_gaussian_context.py`, `sae/g12_repair_jobs.py`, `configs/config_sae_1g_12_exp23_v1.py`, `config/sae_1g_12_exp23.py`, `scripts/g12_repair_jobs_test.py` (31/31) |
| 1g.12 experiment 4, the twenty exact order-4 one-best readouts (approach 22, verdicts 59-60) | `work/speech_llm/sae/g12_readout_jobs/G12ExactReadoutJob.*` (20 dirs; each `readout.json`, `readout.txt`, `hypotheses.json`); code `sae/g12_exact_decode.py`, `sae/g12_readout_jobs.py`, `configs/config_sae_1g_12_exp4_v1.py`, `config/sae_1g_12_exp4.py`, `scripts/g12_exact_decode_test.py` (33/33), `scripts/g12_readout_jobs_test.py` (22/22) at speech-llm `5e245b1` |
| 1g.13 experiment 3, the five phone starts on the v1-equivalent stream (approach 25, verdicts 62-64) | `work/speech_llm/sae/h3_jobs/H3InitializerJob.lR5Q4q1xRtqV` (fingerprint), `.m4sNBqlCwK2Z` (random-map seed 1000), `.fGmIiECLQ2XW` (pseudo-pair seed 0); `work/speech_llm/sae/g13_firewall/G13ReferenceStartJob.kG9pmxczOVgF` (controlled reference); `work/speech_llm/sae/h3_projection/H3CalibrationEspumProjectionJob.2EB1uTDlskOy` (espum) -- each `start.npz` and `start.json` (the espum projection emits `espum_calibration_start.npz`/`.json`); code `sae/g13_firewall.py` at speech-llm `6bfa29d`, config wiring `configs/config_sae_1g_13_exp3_v1.py` at `a0d2808`; `config/sae_1g_13_exp3.py` and `scripts/g13_reference_start_test.py` (20/20) are workspace files under no version control |
| 1g.13 experiment 3, the espum arm's registered fan-out on the v1-equivalent stream (approach 25, verdicts 63-64) | `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.oAOLIZZHVaVz` (full seed 0, picked), `.18iF7DTcCNyF` (full seed 1), `.E9fojuqhcBDZ` (full seed 2), `.q59UQC0AW5Oc` (bigram-only control); `work/speech_llm/sae/h3_jobs/H3EspumPickJob.ud5adF5qEliC` (`frozen_selection.json`), `work/speech_llm/sae/h3_jobs/H3MaskedEspumStreamJob.6OiRRPPXl1w8` (`manifest.json`); accepted-stream contrast rows quoted from `EspumMatchTrainJob.97FwGhhItdpO` and `.h4LngSZ4YvKL` |
| 1g.13 experiment 4, the measured order-4 resource read on the v1-equivalent stream (approach 26, verdicts 65-67) | `work/speech_llm/sae/g12_resource/G12ResourceGateJob.cQ3wfqsTamPP` (`resource_gate.json`, `.txt`); SUPERSEDED first run, kept as the record of the pre-fix measurement, `.4iWPXMh9yoJN` -- orphaned by hash, superseded evidence rather than debris; code `sae/g12_resource.py`, `sae/g11_gaussian.py`, `configs/config_sae_1g_13_exp4_v1.py`, `config/sae_1g_13_exp4.py`, `scripts/g13_resource_gate_test.py` (43/43) at speech-llm `41127e8` |
| The 2026-08-24 context-engine backward-recursion fix and its registered anchor (approach 26, verdict 68) | `work/speech_llm/sae/g12_engine_equivalence/G12EngineEquivalenceJob.sWWDLbPKglfP` (`engine_equivalence.json`, `.txt`) -- reproduces the banked `G12ResourceGateJob.3h2iIpk6lpaB` probe cells under the current engine and pins the two normalizers against each other, worst posterior difference 8.882e-16; code `sae/h4_context_engine.py`, `sae/g12_gaussian_context.py`, `sae/g12_engine_equivalence.py`, `scripts/h4_context_engine_test.py` (48/48), `scripts/g12_engine_equivalence_test.py` (25/25) at speech-llm `98ddf9f` |
| 1g.13 experiment 2, the H1 route read on the v1-equivalent stream (approach 24, verdict 61) | `work/speech_llm/sae/g13_jobs/G13RoutesJob.hStPuE1UqLK6` (`phase1g_h1_v1_equivalent.json`, `.txt`); code `sae/g13_jobs.py`, `scripts/g13_routes_test.py` (25/25) at speech-llm `4d0fad6` |
| 1g.13 experiment 2, the VAD-mask firewall (approach 24) | `work/speech_llm/sae/g13_firewall/G13VadFirewallJob.Usfy2NF0LiSQ` (`gold_update.pkl`, `gold_selection.pkl`, `gold_evaluation.pkl`, `trim_masks.pkl`, `firewall.json`, `firewall.txt`); code `sae/g13_firewall.py`, `configs/config_sae_1g_13_exp2_v1.py`, `config/sae_1g_13_exp2.py`, `scripts/g13_firewall_test.py` (30/30) at speech-llm `4d0fad6` |
| 1g.13 experiment 1, the wav2vec-U v1-equivalent stream (approach 23, verdict 57) | `work/speech_llm/sae/g13_jobs/G13StreamBuildJob.Ob8Rh8y51x9M` (`units.pkl`, `segments.pkl`, `boundaries.pkl`, `component_scale.npy`, `transform.npz`, `stream.json`, `stream.txt`); code `sae/g13_stream.py`, `sae/g13_jobs.py`, `configs/config_sae_1g_13_exp1_v1.py`, `config/sae_1g_13_exp1.py`, `scripts/g13_stream_test.py` (35/35), `scripts/g13_faiss_reference_test.py` (10/10, run under the `w2vu` env), `scripts/g13_jobs_test.py` (34/34) at speech-llm `7f3f312` |
| 1g.9 experiment 1, locate the collapse (approach 15, verdicts 26-29) | `work/speech_llm/sae/h4_collapse_locate/H4CollapseLocateJob.gZ9d6e3E7ZGu`; code `sae/h4_collapse_locate.py`, `configs/config_sae_1g_h4_collapse_locate_v1.py`, `config/sae_1g_h4_collapse_locate.py`, `scripts/h4_collapse_locate_test.py` (6/6) at speech-llm `d08cd88` |
| 1g.2a item 4 rank agreement between the two halves | `work/speech_llm/sae/h4_context_agreement/H4ContextAgreementJob.zd6RBdYcvzti` |
| 1g.2a item 4 label-free own-minus-donor, five starts x four fitting LMs | `work/speech_llm/sae/h4_context_scores/H4ContextOwnMinusDonorJob.SygqXhY8F2Qt` |
| 1g.2a item 4 descriptive PER, five starts x four fitting LMs | `work/speech_llm/sae/h4_context_decode/H4ContextDiagnosticPerJob.IYHS4cX3j3XV` |
| 1g.2a item 4 repaired-table decodes, 60 cells at counts 1/2/4 | `work/speech_llm/sae/h4_context_decode/H4ContextLocalDecodeJob.*` (60 dirs) |
| 1g.2a fitting LM `legacy-2g` (add-one bigram) | `work/speech_llm/sae/h4_lm_artifacts/H4LegacyLmJob.lZI6TrYdVpev` |
| 1g.2a fitting LM matched-2g / 3g / 4g (unpruned MKN) | `work/speech_llm/sae/h4_lm_artifacts/H4MatchedLmJob.T8ImJUXHaB0l`; `.Jb2m4aM2fUTy`; `.VpVkGMMy7xKW` |
| 1g.2a matched MKN ARPAs, orders 2/3/4 | `work/i6_core/lm/kenlm/KenLMplzJob.ef5FXMvv8af5`; `.tis71OtNidgL`; `.bg0iYRzBQynx` |
| 1g.2a fixed-duration diagnostic, 5 starts x 4 fitting LMs (item 4) | `work/speech_llm/sae/h4_context_diagnostic/H4ContextRepairJob.*` (20 cells) |
| 1g.2a measured resource gate (item 3) | `work/speech_llm/sae/h4_context_resource/H4ContextResourceGateJob.HA1vzRL7MEAz` |
| 1g.0 structure screen, dev-clean | `work/speech_llm/sae/structure_screen/StructureScreenJob.Xyy7r1zTK9hU` |
| 1g.0 structure screen, dev-other | `work/speech_llm/sae/structure_screen/StructureScreenJob.U3QYclOJHgq2` |
| spectral duration-polarity reads, clean/other | `work/speech_llm/sae/spectral_split/SpectralVCJob.AK0OUD2QcPXz`; `work/speech_llm/sae/spectral_split/SpectralVCJob.ZA7uvQ2s7Zta` |
| spectral registered-polarity reads, clean/other | `work/speech_llm/sae/spectral_split/SpectralVCJob.dP9A1geKgd45`; `work/speech_llm/sae/spectral_split/SpectralVCJob.koxlC99UA0t6` |
| raw and pooled unit streams | `work/speech_llm/sae/quantize_states/MergeUnitsPklJob.ncxcd3vouD5E`; `work/speech_llm/sae/repr_pool/SegmentPoolUnitsJob.IHRNqQfnxrQ3` |
| frozen encoder states and train-fit quantizer | `work/speech_llm/sae/av_states/AvStatesJob.c4Ak1rACchRC`; `work/speech_llm/sae/quantize_states/QuantizeStatesJob.FWpGhC941JMi` |
| historical full-bed silence-delimited stream | `work/speech_llm/sae/lexfree_match/UnitWordStreamJob.eIxgmMh99RSE` |
| phone text (`T_phi`) | `work/i6_experiments/users/wu/experiments/posterior_hmm/data/phon_lm/TextToPhonemeJob.THKMON3k9LJQ` |
| normalized character text | `work/i6_core/tools/download/DownloadJob.g4jClO48cAvP` |
| descriptor dump | `work/speech_llm/sae/descriptors/UnitDescriptorsJob.cSmt6LY5WVOu` |
| canonical descriptor read, dev-clean 540 | `work/speech_llm/sae/descriptors/UnitClassReadJob.m0usvL4Oxlv2` |
| canonical descriptor read, dev-other 572 | `work/speech_llm/sae/descriptors/UnitClassReadJob.kvZb0zdRznOY` |
| superseded all-utterance descriptor reads | `work/speech_llm/sae/descriptors/UnitClassReadJob.yeB6P7J3rdwz`; `work/speech_llm/sae/descriptors/UnitClassReadJob.FIwiUeQ5bgGv` |
| descriptor audio manifest | `work/speech_llm/sae/gua_jobs/GuaAudioManifestJob.rdVx8r37h78h` |
| completed exploratory E5 rehearsal | `work/speech_llm/sae/seed_basin/SeedBasinJob.Zm3EuTveSGBL` |
| original fingerprint reads, clean/other | `work/speech_llm/sae/fingerprint_match/FingerprintMatchJob.O4dpJTesB66u`; `work/speech_llm/sae/fingerprint_match/FingerprintMatchJob.MHmUIV85g8Ry` |
| selected ESPUM checkpoint | `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.lALR9ldNG8f1` |
| completed but invalid H1 read | `work/speech_llm/sae/channel_h/Phase1gH1Job.Bz5bcz5grt8B` (runtime source was not frozen; see verifier feedback) |
| accepted construction-only H1 read | `work/speech_llm/sae/channel_h/Phase1gH1Job.HbxKiuBTJ8aN` (`phase1g_h1.json`, source snapshot, and progress log) |
| H2 common channel and bounded raw-character fitting | commits `0556513`, `80408cf`; `src/speech_llm/sae/channel_h.py` |
| H3 role manifest and valid frozen fingerprint fixture | `work/speech_llm/sae/h3_jobs/H3RoleManifestJob.3hl4qCJKKlUN`; `work/speech_llm/sae/lexfree_match/LexFreeMatchJob.lQqgQGjbVe6A`; `work/speech_llm/sae/h3_jobs/H3FrozenFingerprintFixtureAssertJob.Vv2w4KN5173K` |
| invalid H3 prospective stream (quarantined) | `work/speech_llm/sae/h3_jobs/H3MaskedEspumStreamJob.423jpYfDsDkM` |
| invalid H3 prospective initializer batch (quarantined) | `work/speech_llm/sae/h3_jobs/H3InitializerJob.mV1ulU6v75Zr`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.wuJrQHnNPq9m`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.lQQcUgcUry7z` |
| invalid H3 prospective ESPUM batch (quarantined) | `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.nJBsngMMc59s`; `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.CCohNXNZgPsX`; `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.jU2ODb1ahIlK`; `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.F0AcQFAYmv0E` |
| corrected H2/H3 core implementation | commits `8c8eec5`, `5be4263`, `3ba4917`, `04a0b3a`, `925c0be`, `4ba0f32`, `88762f2`, `bda896a`; `src/speech_llm/sae/channel_h.py`; `src/speech_llm/sae/channel_decode_jobs.py`; `src/speech_llm/sae/h3_projection.py`; `src/speech_llm/sae/h3_resume_equivalence.py`; `src/speech_llm/sae/espum_jobs.py` |
| corrected H3 pooled-run stream and calibration starts | `work/speech_llm/sae/h3_jobs/H3MaskedEspumStreamJob.GqAphDUVZJ7f`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.6ifXwi6C9o4b`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.wP5OnAoxzDow`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.gNAARAXeogOt` |
| corrected H3 ESPUM calibration fan-out and frozen pick | `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.97FwGhhItdpO`; `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.eQyuM6m4rPX2`; `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.lk3V9mM67j0m`; `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.h4LngSZ4YvKL`; `work/speech_llm/sae/h3_jobs/H3EspumPickJob.ezmw64E1JwzI` |
| selected calibration projection and GPU resume-equivalence evidence | `work/speech_llm/sae/h3_projection/H3CalibrationEspumProjectionJob.s4GWy36bdWxZ`; `work/speech_llm/sae/h3_resume_equivalence/H3EspumResumeEquivalenceJob.yL2E4UjTDxQ6` |
| live H2 count-0 channel snapshot | `work/speech_llm/sae/channel_decode_jobs/Phase1gChannelSnapshotJob.TJWNeqBXGjfy` |
| completed H2 decoder timing grid | `work/speech_llm/sae/channel_decode_jobs/Phase1gDecoderPreflightJob.egplrTqzH7Ys` (fastest cell); `work/speech_llm/sae/channel_decode_jobs/Phase1gDecoderPreflightJob.5xQSQqaShtXI` (largest elapsed-time cell) |
| H3 construction-population final initializers | `work/speech_llm/sae/h3_jobs/H3InitializerJob.uKw59MBJC4Hj`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.ABTGA9vIwwI8`; `work/speech_llm/sae/h3_jobs/H3InitializerJob.BS1nPUwf1fel` |
| H3 selected construction-population ESPUM refit and strict projection | `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.t1l7N4lQ9dtY`; `work/speech_llm/sae/h3_projection/H3FinalEspumProjectionJob.PJMwUGUXUb7s` |
| H4 artifact, donor, bootstrap, and gate harnesses | commits `93f6261`, `4e67695`; `src/speech_llm/sae/h4_harness.py` |
| H4 calibration preparation and update-only repair graph | commit `c2e930b`; `work/speech_llm/sae/h4_jobs/H4CalibrationPreparationJob.DPv4aIqwPEzM`; reference `H4RepairJob.x1TyHJMfEVpb`; fingerprint `.iUFh7IwniCMl`; random-map seed 1000 `.Ds0zM1NTY2C1`; pseudo-pair seed 0 `.aeetC3NfgPxB`; ESPUM seed 0/update 30,000 `.ViPSmq4Am8vX` |
| H4 corrected recovery and decoder-resource preflight | commit `436ea50`; update-only reference `work/speech_llm/sae/h4_production_jobs/H4UpdateReferenceArtifactJob.DZa7gIj8rZNj`; Q recovery `work/speech_llm/sae/h4_production_jobs/H4QRecoveryJob.ar34r8ltGTGW`; selection donor `work/speech_llm/sae/h4_production_jobs/H4RoleDonorTableJob.w2RMXcCJyGoy`; update contract `work/speech_llm/sae/h4_decode_jobs/H4ResourceContractJob.kFA99bygctlt`; selection contract `work/speech_llm/sae/h4_decode_jobs/H4ResourceContractJob.kyMk7fwm027C` |
| H4 baseline global-beam boundary | code commit `3de988a`; reducer `work/speech_llm/sae/h4_beam_jobs/H4GlobalBeamTableJob.ro6L8QCnqYpx` |
| H4 pre-label selection surface and provisional maxima | commit `84808a8`; `src/speech_llm/sae/h4_selector_jobs.py`; surface `work/speech_llm/sae/h4_selector_jobs/H4SelectionSurfaceJob.MKHfnUO9XwkU`; maxima `work/speech_llm/sae/h4_selector_jobs/H4ProvisionalMaximaJob.ejmy4sdTOcS3` |
| H5 handoff and H6 character-route interfaces | commit `ce265ce`; `src/speech_llm/sae/handoff.py`; `src/speech_llm/sae/character_route.py` |

| 1g.2 audited provisional maxima (local-winner exemption asserted) | `work/speech_llm/sae/h4_selector_jobs/H4ProvisionalWinnerAuditJob.kBCapQOpk1Hj` |
| 1g.2 controlled validation read (verdicts 18-20; the only label reader) | `work/speech_llm/sae/h4_validation_jobs/H4ControlledValidationJob.Otv6GBVY8ZUj` |
| controlled reference gold phones | `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/eval/GoldPhonesJob.ZGSp0hxyd2YP` |
| 1g.2 descriptive real-seed PER (verdict 21; selects nothing) | `work/speech_llm/sae/h4_real_seed_per/H4RealSeedPerJob.vu6Dp6HkJ2pH` |
| 1g.11 experiment 1 continuous observation twin of `seg12.5` (segments.pkl, boundaries.pkl, component_scale.npy) | `work/speech_llm/sae/g11_continuous/G11ContinuousSegmentsJob.hImWJG0X4eZh`; code `sae/g11_continuous.py`, `configs/config_sae_1g_11_v1.py`, `scripts/g11_continuous_test.py` (19/19) at speech-llm `16b1063` |
| 1g.11 experiment 4 evaluation against gold, the gate table (evaluate.json, evaluate.txt) | `work/speech_llm/sae/g11_evaluate/G11EvaluateJob.sWoS1bP4Nd12`; code `sae/g11_evaluate.py`, `configs/config_sae_1g_11_exp34_v1.py`, `config/sae_1g_11_exp34.py`, `scripts/g11_exp34_test.py` (26/26) at speech-llm `5a344f2` |
| 1g.11 experiment 3 observation null (observation_null.json, hypotheses.json, repair.json, repair.txt) | `work/speech_llm/sae/g11_nulls/G11ObservationNullJob.orOc9h6K3cuR`; code `sae/g11_nulls.py` at speech-llm `5a344f2` |
| 1g.11 experiment 2 Gaussian repair cells (hypotheses.json, repair.json, repair.txt) | `work/speech_llm/sae/g11_repair_jobs/G11GaussianRepairJob.NogH62uMEI7T`; code `sae/g11_gaussian.py`, `sae/g11_repair_jobs.py`, `configs/config_sae_1g_11_exp2_v1.py`, `scripts/g11_gaussian_test.py` (40/40), `scripts/g11_repair_jobs_test.py` (20/20) at speech-llm `92e5d24` |
| 1g.12 experiment 1 measured resource read, Gaussian arm at fitting order 4 (resource_gate.json, resource_gate.txt; PASS for one curve, RESOURCE_INFEASIBLE for all five starts in one process, 4 h and 4 GiB per cell) | `work/speech_llm/sae/g12_resource/G12ResourceGateJob.3h2iIpk6lpaB`; code `sae/g12_resource.py`, `sae/g12_gaussian_context.py`, the `emissions_by_time` seam in `sae/h4_context_engine.py`, `configs/config_sae_1g_12_exp1_v1.py`, `scripts/g12_gaussian_context_test.py` (57/57), `scripts/g12_resource_test.py` (50/50) |

## Verifier feedback

- 2026-08-20 — The pre-rewrite theory battery has been audited by provenance and relevance. The
  completed, decision-bearing evidence is now summarized in `PLAN_1G.md`: the raw-versus-pooled
  rate/identity tradeoff, the need for two-state within-symbol structure, the weakness of aggregate
  matching as content evidence, the positional design's rank limitation, the damage from hard unit
  coarsening, and the scoped likelihood/error warning. The generic finite-HMM identifiability theorem
  is literature context only and does not establish identifiability of the live tied duration model.
  The archived corpus-size, pair/triple nullity, moment-sample, repair-basin, decoder-gain, and anchor
  predictions are uncommitted synthetic motivation, not results or gate evidence.

- 2026-08-20 — H1 remains accepted at the exact 6,414/890/7,304/1,112 partition and two-state phone
  choice `p=0.23560298`; do not rerun it. H2 is now closed at the common-engine level: commits
  `88762f2` and `bda896a` require and propagate the same explicit deleted-silence boundary vector
  through repair, scoring, and decoding. The current channel suite passes 23/23, including exact
  boundary-aware repair enumeration. This resolves the previously material mismatch at 53,498 gaps
  affecting 97.71% of the update utterances. The accepted 39-phone by 500-unit snapshot, strict
  evidence/merge checks, 48 timing cells, and eight-alternative output-only cap remain unchanged.

  H3 calibration remains valid: the exact 6,414/890 roles selected full-loss ESPUM seed 0/update
  30,000 at weighted phone-LM perplexity 32.5352; strict projection and GH200 resume equivalence pass.
  Final-refit fingerprint `H3InitializerJob.uKw59MBJC4Hj`, random map `.ABTGA9vIwwI8`, and
  pseudo-pair `.BS1nPUwf1fel` finished on all 7,304 construction IDs. The selected ESPUM construction
  refit `EspumMatchTrainJob.t1l7N4lQ9dtY` and strict final projection
  `H3FinalEspumProjectionJob.PJMwUGUXUb7s` also finished. The projected 500-by-39
  `Q(phone | unit)` and 39-by-500 `B(unit | phone)` are finite and normalized; their manifest binds
  all 7,304 construction IDs, the frozen seed-0/update-30,000 choice, every input, and all eight
  runtime sources. The refit does not hash selection IDs. Never relaunch calibration or this final
  graph.

  H4's first calibration graph exposed two material interface defects before any selection read: its
  donor law did not guarantee the exact support condition `C_d <= N <= T_d`, and its controlled bundle
  omitted the original count-0 `Q(phone | unit)`. Exact `-inf` donor scores must not be clipped, and
  `Q` cannot be recovered provenance-safely from normalized `B`. The corrective directives from this
  read are superseded by the verified 2026-08-21 recovery entry below; no selector, final refit, or
  evaluation result existed to reinterpret.

- 2026-08-21 — The corrected H4 prerequisite graph is independently verified and complete. All 71
  regenerated non-soft controls reproduce their retained count-0 `B` exactly, and all four imported
  H3 pairs reproduce canonical `B` from persisted `Q`; therefore exactly 75 trajectories are reused
  and the ten soft Q-space trajectories are rerun. The graph exposes 85 direct-Q starts and the full
  85-by-4 lossless channel inventory. The production selection donor law yields 513/890 eligible
  sources (235/432 clean and 278/458 other), with 377 explicit `no_swap`; these are construction-time
  facts about the frozen table, not a content result. Resource preflight is exactly 288 probes: three
  representatives by 48 cells on each of update and selection, plus one global worst-cell shard rerun
  and contract per role. Both contracts pass the registered 1.5-times resource rule. The update role
  measured 23,768.19 seconds and 1.043 GiB on its heaviest 19,515-unit shard, yielding a 10-hour,
  2-GiB production request; selection measured 3,069.04 seconds and 0.908 GiB on its heaviest
  2,466-unit shard, yielding 2 hours and 2 GiB. The graph has 821/821 jobs finished and no scheduler
  or problem state. Evaluation audio, labels, donors, decodes, and scoring remain absent.

  The reported selector blocker is **VERIFIED and resolved prospectively**. Independent graph
  reconstruction finds no full-role local or sequence decode, fixed-text score, selector, final refit,
  or evaluation job among the 821 completed prerequisites. The raw scorer interface deliberately
  retains `ell_own`, `ell_donor`, `T_source`, `T_donor`, and phone length without choosing a
  normalization. Unequal lengths are material in the frozen donor table: among 5,130 assignment rows,
  4,803 (93.6%) have a longer donor; the median `T_donor/T_source` is 1.159 and the maximum is 5.25.
  The stored update likelihood has no decoder-family, `lambda`, `beta`, or beam coordinate and rises
  at every retained 0/1/2/4 step in all 85 trajectories; using it would choose count 4 everywhere
  while leaving the rest of the deployment tuple undefined.

  `PLAN_1G.md` now freezes own-minus-donor as the **sole deployment selector**. For tuple `c`, fixed
  source decode `z_ic`, and donor assignment `s`, it is

      Delta_ics = log P_Bc(U_i | z_ic)/T_i
                  - log P_Bc(U_d(i,s) | z_ic)/T_d(i,s).

  Each denominator is that input's own positive retained-unit count after the frozen silence mask.
  `P_Bc` marginalizes the duration/state paths, including their exits, and excludes `G_fit`, `G_dec`,
  insertion penalty, beam score, and posterior terms. A common source denominator or phone-length
  denominator is forbidden: for a content-independent per-unit rate and `T_d > T_i`, either creates a
  positive own-audio advantage from donor length alone, whereas separate rate normalization gives
  zero.

  For each of ten frozen assignments, take equal-utterance clean and other means over the common
  eligible rows, combine them with fixed weights 432/890 and 458/890, then average the ten assignment
  values equally; higher wins. Do not renormalize after `no_swap`, clip non-finite values, or drop
  tuple-specific rows. Beam is not a score-selected coordinate: one global beam per
  `(G_fit,lambda,beta)` is derived on the three frozen representative tables and heaviest update
  shard, adding at most 144 stability cells **per fitting-LM identity**. The audit targets are exactly
  the prospective label-fitted reference plus the ESPUM, fingerprint, random-map seed-1000, and
  pseudo-pair H3 rows: any provisional sequence winner must pass a two-beam audit on all 6,414 update
  IDs. Each changed H3 final-refit table must pass again on its 7,304 construction IDs, and the
  reference on its distinct 4,455 dev construction IDs, before evaluation. Each five-row audit adds at
  most 320 shard cells. A triggered coherent matched-4 arm gets its own representatives/global-beam
  table; baseline evidence transfers only by exact tuple and hash identity. The controlled PER reads issue method-
  level repair and sequence-family verdicts; they never prune or rank an individual real-start count
  or grid point.
  H4 maximizes this point statistic over admissible settings within each fixed H3 start; deterministic
  ties prefer `legacy-2g` when applicable, lower repair count, local decoding, the registered
  `(lambda,beta)` enumeration, and canonical start/seed order. Construction likelihood remains the EM
  objective and a finite-value/manifest health diagnostic only; it is never a selector, fallback, or
  tiebreaker. A failed selector or selected-winner beam audit leaves H4 unresolved without reranking.

  Preserve all 821 completed jobs. Add the bounded representative/global-beam extension and new
  deterministic consumer boundary. Before any controlled label read, generate every controlled score
  plus the reference/four-H3 selection surfaces, persist their provisional maxima, and finish the
  update-role winner audits. Only then open controlled labels to validate the fixed selector and
  method families; the four H3 error rows remain sealed, and a pass freezes the unchanged maxima. A
  triggered H4-LM repeats this label-reader-free boundary before its new controlled verdict. Final-
  refit the four H3 rows on 7,304 construction IDs and the reference on 4,455 dev IDs, build the
  evaluation-release artifact after all required final-table audits, and only then open the 1,112-ID
  evaluation once. Do not edit the source-hash-bound `h4_production.py` or `h4_decode_jobs.py`; add new
  consumer jobs/module/config around their artifacts. No prerequisite repair or resource-preflight
  rerun is required.

- 2026-08-20 — Higher-order `G_fit` is registered as conditional H4-LM work, not H6; H6 remains the
  character route after a valid H5 phone handoff. First complete the corrected baseline bigram H4
  assay prerequisites: mechanics, positive controls, donor-score calibration/correlation, and selector
  validity. The method-specific nonzero-count/update-health outcome is not a prerequisite. A failed
  prerequisite is fixed first and does not trigger an LM arm. Trigger H4-LM before evaluation only if
  those prerequisites pass and either the controlled method-level read finds no safe nonzero count or
  the label-free selector assigns count 0 to both real phone seeds. Controlled labels never attach
  safety to or prune a selected real count. Passing both independent conditions is merely pre-
  evaluation-ready, not a held-out content result. The motivating Ney
  order sweep changes recognition
  order for its learned- and correct-channel rows together, so it does not isolate fitting order or
  predict the project effect.

  H4-LM separates the legacy add-one bigram from a matched unpruned modified-Kneser--Ney 2/3/4
  fitting family while keeping the banked 4-gram decoder fixed. H2-LM must implement exact context-
  state repair, reproduce dense `legacy-2g` and exhaustive tiny matched order-2/3/4 reads, and obtain
  measured trigram/4-gram timing and RSS; the 48 H2 beam-decoder cells cannot size repair. The fixed-
  `p` diagnostic is bounded to the reference and four accepted H3 starts: matched order 2 reads the
  smoothing bridge, order 3 is directional, and order 4 is the proposed context probe. It cannot feed
  H5. Only matched order 4 receives a full controlled-library arm, with a separate H1-LM EOS-
  conditioned duration/topology read on the 6,414 update IDs and its own representative/global-beam
  contract. Label-reader-free jobs freeze the combined provisional maxima before matched-4 controlled
  errors are read; F then repeats the expanded selector and sequence-family gates rather than
  inheriting the latter from baseline. It advances only if matched-4 has a safe nonzero controlled
  point and, independently, ESPUM or fingerprint has an unchanged nonzero combined maximum; otherwise
  evaluation remains closed without label-based count substitution. Only unchanged passing maxima may
  refit/release. Held-out LM perplexity is descriptive and never selects order. This is planned work,
  not a result.

- 2026-08-21 — H4 global-beam boundary VERIFIED; approach 10 and conclusion 16 CONFIRMED. Full
  independent recomputation from the 144 raw `H4SequenceDecodeChunkJob` cells reproduces every
  claim: no `(lambda,beta)` setting has an adjacent beam pair passing both clauses on all three
  representatives; the best worst-representative one-best agreement is 0.7313432835820896
  (lambda=2, beta=-2, pair 256->512) and the smallest worst-representative score change is
  0.005448072638504445 nats per retained unit (lambda=0.5, beta=-2, 256->512), matching the
  logged 0.7313/0.005448; the reducer's own per-setting verdicts agree cell-for-cell with the
  recomputation, no anomalies. Frame verified at source and artifact: `beam_pair_statistics`
  implements exact one-best symbol-sequence equality and the fsum-over-retained-units
  denominator with inclusive 0.999 and strict 1e-4 bounds, and `derive_global_beams` walks
  ascending adjacent pairs requiring EVERY representative, freezing the smaller beam of the
  first passing pair; the three representatives are the frozen resource-contract triple (340
  tables deduplicated to 316 hashes, entropy-sorted, indices 0/157/315:
  `controlled/map_q04_draw03` r0, `controlled/map_q06_draw01` r2, `controlled/soft_q00` r0);
  the shard is the canonical heaviest update chunk `update[2::32]` (201 IDs, 19,515 retained
  units, hash-bound to contract `H4ResourceContractJob.kFA99bygctlt`); zero selection or
  evaluation IDs appear in any cell; every `code_identity` sha256 matches the current checkout,
  and commits `3de988a`/`84808a8` touched neither `h4_production.py` nor `h4_decode_jobs.py`
  (last touched by `436ea50`). Consequence ratified as plan-consistent: all 12 sequence
  settings are label-free ineligible, the baseline surface retains the LOCAL decoder only, H4
  is NOT failed, no provisional sequence winner can exist so the sequence-family gate cannot
  bind at this boundary, and local winners need no beam audit. Reading note: with 201 shard
  utterances the 99.9% clause is effectively 201/201 unchanged one-bests — a property of the
  frozen shard size, not an implementation choice. The `sae_1g_h4_prelabel_surfaces` launch
  condition ("12 grid verdicts verified") is MET.

- 2026-08-22 — Pre-label selection-surfaces launch VERIFIED (speech-llm `84808a8`; manager pid
  3796121 with watcher attached). The graph's new work is exactly 340 `H4LocalDecodeJob` (85
  starts x counts 0/1/2/4), 3,400 `H4FixedTextScoreJob` (ten frozen assignments per tuple), one
  `H4SelectionSurfaceJob` and one `H4ProvisionalMaximaJob`; no sequence decode or merge exists,
  and the config enforces local-only by construction (`_load_completed_global_table` raises
  unless the beam table classifies all 12 points, and an ineligible row contributes no tuple).
  All 966 pre-existing dirs (821 prerequisites + 144 beam cells + the reducer) show zero files
  modified after launch except two startup auto-cleanup re-tars with outputs untouched; nothing
  rerun, no error markers. Label firewall confirmed at source: `h4_selector_jobs.py` contains no
  transcript, edit-count, or evaluation reader; the selector implements Section 4's `Sel`
  exactly (own-minus-donor per-unit contrast, 432/890 and 458/890 weights, ten assignments
  averaged with `math.fsum` in sorted order; beam and likelihood excluded from the tie order);
  the later-boundary freeze/audit classes exist in the module but are not instantiated, and both
  launched selector jobs re-assert their own source identity at run time. `h4_production.py` and
  `h4_decode_jobs.py` remain untouched since `436ea50`. Both log migrations verified: SAE_1g
  Conclusion -> Verdicts byte-identical (16 items, all 8 WRONG markers verbatim), SAE_3E1
  append-only (59-61 added after untouched 1-58) — no plan reference can dangle. Progress at
  ~22:20: all 340 local decodes finished, 1,420+/3,400 scores done, zero errors.
  State corrections handed to the implementer (State is implementer-owned): (i) "the best
  adjacent-beam pair keeps only 62%" is unsupported by the artifact — the binding best
  worst-representative agreement is 0.7313 (best mean/pooled 0.7678); 0.6202 is the best pair of
  the WORST setting (lm_scale 0.5, beta 0), the likely misread; (ii) the launch split "635
  finished / 343 runnable / 3,730 waiting" is the known one-shot console status misreport —
  ground truth 966 finished / 3,742 unfinished (the 4,708 total and "nothing in a problem
  state" are correct); (iii) the cleanup note's "input/ goes" is wrong — `JOB_CLEANUP_KEEP_INPUT
  = True` is effective and cleaned dirs on disk retain `input/`; cleanup keeps `output/`,
  `info`, and `input/`, removes internal `work/`, and tars logs; (iv) trivial: manager start
  21:48 not 21:47, and "two orders above the 1e-4 bound" overstates (smallest
  worst-representative change is 54x). The JOB_AUTO_CLEANUP console fact itself is verified
  correct (`sisyphus/__main__.py:217-218` forces False for every non-manager subcommand;
  `settings.py:238` True; no effective override). One harmless exception: startup cleanup
  failed on `GoldPhonesJob.ZGSp0hxyd2YP` (its `work/` never existed), which keeps a plain
  `finished` marker.

- 2026-08-22 (H4 pre-label surfaces VERIFIED; controlled labels ruled open; two observations
  banked for the validation read). Approach 11 and verdict 17 confirmed against the raw
  artifacts: 85 starts (81 controlled + 4 real, counted from the maxima JSON), all finite, every
  winner `decoder.kind="local"` and eligible, and every winner the true argmax over its four
  tuples (0/85 violations); the five registered cross-start rows match at all printed digits and
  the controlled/real random-map twins are bit-identical at full float64 (shared channel content
  hash, only name-bearing hashes differ). Surface counts confirmed by construction and by input
  symlinks: 340 tuples x 10 assignments = 3,400 `H4FixedTextScoreJob` inputs, 4,080
  global_beam_ineligible entries, 324 controlled within-sequence choices all ineligible — no
  sequence-decoder score exists anywhere in either artifact. `code_identity` sha256 equals the
  committed `h4_selector_jobs.py` at `84808a8` exactly; the maxima file's recorded
  `selection_surface_sha256` equals the surface file's actual digest; `choices_sha256` and
  `expected_tuple_keys_sha256` recompute exactly. Label firewall re-confirmed from the job
  `info` inputs (no transcript/WER/corpus reader among all 4,167 surface inputs) and both
  artifacts carry `contains_labels: false` / `frozen_pre_label: true`. The Section-4 aggregation
  is implemented as registered: fixed 432/890, 458/890 weights over per-split eligible-row
  means, NO renormalization after `no_swap` removal — all 3,400 weighted means and all 340 `Sel`
  values reproduce exactly. Verdict 17's local-winner reading is frame-correct against
  `PLAN_1G.md` ("A local winner needs no beam audit"); the ruling opening the controlled
  reference labels is in `PLAN_1G.md` Status. Both hand-backs CLOSED same day (commit
  `1c0ab7a88`, verifier-checked): the approach-11 coverage wording now states the 513
  donor-eligible contributing sources and the no-renormalization weights correctly, and the q09
  degeneracies are confirmed as construction at source — `Q_LEVELS` ends at 1.0
  (`h4_jobs.py:34`) so level 09 is each ladder's undamaged endpoint; at q=1.0 the map ladder's
  `keep_count` is the whole live set and `assignment[keep] = reference_map[keep]` discards the
  draw (`h4_jobs.py:307-316`), and `canonical_soft_q` early-returns `reference.copy()`
  (`h4_production.py:223-224`) — all three citations verified against the committed source.
  Two observations for the validation stage, no action now: (i) only 79 distinct channel
  content hashes among the 85 starts — `controlled/map_q09_draw00..04` are all one channel and
  `controlled/soft_q09` is bit-identical to `controlled/reference`, both by the ladder
  construction above, and (correction 2026-08-22, third group found in the audited maxima)
  `controlled/random_map_seed1000` is bit-identical to `real/random_map_seed1000`, which is
  what makes 85 go to 79 rather than 80 — so effective independent controls are 76 of 81
  (unaffected by the third, cross-namespace group), which any clustered CI
  or null spread over the controls must
  respect; (ii) the cross-start surface ranks the random-map null 9th of 85 (10.7753)
  and the reference 73rd (5.8265), with the registered five rows at ranks 9/72/73/76/84 — the
  controlled label validation must be read knowing the pre-label `Sel` ordering places nulls
  above the reference, and winning repair counts across all 85 starts are {0: 72, 4: 13} (counts
  1 and 2 never win).

- 2026-08-22 (1g.2 controlled validation read VERIFIED; verdicts 18-20 ACCEPTED on the gate;
  one material hand-back). Code audit (`h4_validation_jobs.py` at `31ea348`, tree matches the
  commit): the label firewall is CONSTRUCTION-time (`h4_validation_jobs.py:691-698`, non-
  `controlled/` keys raise before outputs exist) with a runtime backstop cross-checking each
  artifact's own `arm_name`/`repair_count`/`role` fields; the job recomputes no score and
  reranks no maximum (only writes are its own three outputs); the `Sel`-rebuild guard sits
  before the RNG and hard-fails at 1e-9 (`:456-465`); duplicate collapse is by the artifacts'
  own `channel_array` sha256 (`:329`, provenance to `h4_production_jobs.py:884`); verdict logic
  implements the registered readings exactly (correlation NEGATIVE iff upper95 <= 0, count SAFE
  iff upper95 <= 0.05 matching the gate's "no greater than"; `h4_lm_trigger` fires iff no safe
  count). Test suite re-run by the verifier: 50/50, synthetic-only, no real artifact read.
  Artifact cross-check (`H4ControlledValidationJob.Otv6GBVY8ZUj`, finished 5m34s; audit job
  `kBCapQOpk1Hj`): every clause interval, point estimate, and verdict word in verdicts 18-20
  traces to `controlled_evidence.json` fields and brackets its point estimate; bootstrap
  metadata as registered (10,000 resamples, seed 20260822, three levels); `n_finite` = 10,000
  on EVERY interval block (discharging the audit's no-minimum-finite-gate concern for this
  run); no NaN anywhere; the audited maxima are value-identical to the provisional maxima
  except the schema string plus two provenance keys, zero remappings, all 85 winners
  `decoder.kind = "local"` counted in the artifact itself, and the validation artifact records
  the audited file's sha256. Instrument validity is artifact-traceable WITHOUT the anchor
  sentence: the reference channel reads 0.3934-0.4168 PER by count under the error measure
  while the 48 in-band controls were constructed into the 0.80-0.93 starting-PER band.
  HAND-BACK CLOSED (a501d2fd0 + speech-llm 7146bc6, rerun verified 2026-08-22): verdict 18's
  instrument numbers (count-0 dev-other PER 0.4149; random-map 0.9094) had come from an
  unregistered console command and existed in no artifact field. The job now emits
  `per_by_count_and_split` and `split_sizes` and the read was RERUN at the same hash. My direct
  read of the rerun artifact: ALL previously banked values reproduce identically (17
  interval/point endpoints, the three count intervals, per_by_count, all verdicts, trigger
  False, n_finite 10,000 everywhere), and the anchors are now artifact-traceable — reference
  count-0 dev-other 0.4149 and `random_map_seed1007` 0.9094 on the 458 selection-role dev-other
  utterances (`split_sizes` {432, 458} matching the registered 432/890, 458/890 weights) —
  with the verdict text correctly restating the SAE_1f comparison as corroboration across
  DISJOINT utterance sets, not an identity check. The rerun is accepted ONLY on this pairing:
  nothing had consumed the old artifact (no freeze job exists; everything downstream is closed
  by the gate) and reproduction is bit-identical under the fixed seed — an artifact with any
  consumer would have required a new-hash job instead. Suite re-run by the verifier: 53/53;
  dead code (shadowed `spearman`, `resample_sel`, `has_sequence`) removed in the same commit;
  working tree matches 7146bc6. Wording corrections all absorbed (five NEGATIVE clauses =
  three correlation + two comparison; -5.0215; the third cross-namespace duplicate group).
  Shared-tree item stands, correctly untouched by the implementer (neither of us wrote it): the
  UNCOMMITTED `config_sae_1g_v1.py` edit wiring a second `Phase1gH1Job` with `gold_json` does
  not enter the 1g.2 graph and the accepted H1 stays pinned, but its label-boundary status
  needs its author to explain it — raised to the USER (plus untracked
  `config_sae_3e1_d6_swap_cont_v1.py`).

- 2026-08-22 (user-funded descriptive PER read VERIFIED; verdict 21 accepted; one wording
  hand-back). `H4RealSeedPerJob.vu6Dp6HkJ2pH` finished; the code (`h4_real_seed_per.py` at
  speech-llm `97662ff`, tree matches) implements the registration exactly: construction-time
  exact-key firewall (the four `real/` rows at counts 0/1/2/4, nothing else accepted), runtime
  cross-check of each decode artifact's own `arm_name`/`repair_count`/`role` fields, the
  890-utterance selection-role assertion, the shared verified `pooled_per`/`utterance_errors`
  measure, the reporting rule in docstring AND payload, provenance sha256s, and
  `held_out_evaluation: SEALED` in the artifact; `H4ControlledValidationJob` untouched. All
  sixteen pooled and all quoted dev-other values in verdict 21, every best count (4 for all four
  rows), and both candidate-versus-random-map margins (0.0393/0.0335) reproduce from the
  artifact's own fields exactly; split sizes {432, 458}. Anchor attributions verified in
  SAE_1f.md: n2 is the pseudo-pair null in map form (:91, :459) so 0.9239 is correctly
  attributed; the 0.8809 row is the unary fingerprint candidate (:84, :222, :407); the
  historical "+0.0365" (:218, :397) reproduces as ESPUM's distance to the stronger null n1
  (0.8946 - 0.8580), the same construction as today's margins — SAE_1f's own M1-bar phrasing
  differs, so the margins are read as null-distances, which is what verdict 21 does. Suite
  re-run by the verifier: 58/58. WORDING HAND-BACK (one number): verdict 21's
  "892-utterance-bed subset" — the bed is 890 (432 + 458; artifact `selection_ids_count` 890).
  Verdict 21's framing is correct where it matters: descriptive, selects nothing, the gate
  untouched, and the headline honestly stated — no real seed beats a content-free control on
  plain PER at this operating point, and the table's best number belongs to the pseudo-pair
  control.
  Notes, no action: the bootstrap draws its three levels independently per iteration and shares
  them across arms rather than literally nesting (defensible — utterances are shared across
  channels, and every observed interval is far from its threshold); the winner-audit
  local-exemption loop asserts the five baseline rows, not all 85 maxima (immaterial here — the
  artifact itself counts 85 local winners); dead code in the label-reading module (`spearman`
  defined twice, unused `resample_sel`, dead `has_sequence` parameter) was a review hazard —
  removed hash-neutrally in speech-llm `7146bc6` (this note absorbed 2026-08-22); the suite never asserts an end-to-end overall PASS (moot for this
  NEGATIVE read). Gate consequences are ruled in `PLAN_1G.md` 1g.2 Status (2026-08-22): H4
  unresolved, selector closed on this combination, maxima frozen, refits and the 1,112-ID
  evaluation stay closed, H4-LM not triggered; the direction fork goes to the user.

- 2026-08-22 (1g.2a items 1, 3, and item 4's likelihood half VERIFIED; one process hand-back,
  three wording nits; no banked number moves). Verified READ-ONLY under the harness temp-full
  blocker (no Bash, so no test re-execution; see deferred items). Engine
  (`h4_context_engine.py`): line-level review against the accepted dense law — BOS start into
  sub-state 0, duration moves p into sub-state 1 from BOTH sub-states, exits q*P(j|h) into
  sub-state 0 of the shifted history from both, boundary rule kills duration moves and keeps
  history through the exit contraction, terminal q*P(EOS|h) after the last real frame, per-frame
  scaling — every arc reproduces `repair_hmm`/`marginal_forward_backward`, and the shared
  backward variable is proven by the topology (both sub-states have identical outgoing arcs);
  the base-40 padded history axis with the reachability mask is exact by construction. Cells:
  eight of 20 payloads covering ALL five rows and ALL four fitting LMs reproduce the approach-14
  count-4 table digit for digit (also fingerprint count-0 -7.293028 / -7.293065 behind verdict
  23); every cell banks the same accepted H1 (`5a3ef9de`), unit stream (`59192a3d`), and an
  `input_content_sha256.automaton` equal to the CURRENT manifest's `automaton_sha256` (legacy
  `0aa488aa`, matched-3g `bc176309`, matched-4g `f38eedfc`), so the cells provably consumed the
  post-hardening automata and the caught mid-rewrite ARPA corruption reached no banked number.
  Gate (`resource_gate.json`): verdict 22 and its correction reproduce exactly — 340 candidate
  tables deduplicated to 316, selected indices 0/157/315, probe utterance 2902-9006-0015 (353
  units), heaviest chunk 2 (19,515 units, 201 utterances) re-measured 50.82 s / 0.6734 GiB,
  multiplier 1.5 giving 1 h per-shard and 5 h whole-fold at 2 GiB, `m_steps_run` 0, pruning
  none, `whole_fold_verdict` PASS, reached 60,879/2,435,160 (order 4) and 1,560/62,400 (order
  3), one below the bounds as read. Matched manifests reproduce the approach-14 automaton table
  (histories/arcs/renormalized-mass/normalization error) and the diagnostic config enforces the
  whole-fold gate, exact scope (5x4), alias plus registered output, and gate-read (never
  hand-copied) resource requests.
  HAND-BACK (process, the one material item): the in-flight `legacy-2g` cosmetic rerun
  contradicts approach 14's own recorded principle ("the artifacts were NOT rerun ... to chase a
  cosmetic hash") and, unlike the 1g.2 validation rerun, has CONSUMERS — the five finished
  legacy cells bank `input_content_sha256.automaton_manifest = a01ee7ce...` and `.automaton =
  0aa488aa...`, so a rewrite that adds manifest fields changes the manifest digest (and the
  automaton digest too unless `np.savez_compressed` is byte-stable here), stranding finished
  cells citing content no longer on disk. Park the rerun if it has not finished; if it has,
  verify both output files byte-identical to the banked digests, and on ANY difference restore
  the banked content and move field-name alignment into a new job for future consumers only.
  Either way, reconcile the State bullet ("is rerunning to write the same manifest field names")
  with the approach-14 paragraph and record the final decision in exactly one place.
  Wording nits: (i) verdict 23 "agree to four decimals" is contradicted by its own example
  (-7.2930 vs -7.2931 differ in the fourth displayed decimal; the actual gap is 3.7e-5 — say
  "within 4e-5 per audio unit, so rounded displays can differ in the fourth decimal"); (ii)
  verdict 22's "order 4's 0.90 s" per probe utterance is measured 0.9366-0.9499 s (say 0.95 s);
  (iii) State's item-3 bullet still carries the pre-correction single "1 h and 2 GiB" figure —
  align with the corrected per-shard/whole-fold pair.
  Deferred items DISCHARGED 2026-08-22 evening, shell restored (replaces the deferred list, same
  date, because all four items ran): (a) three suites re-executed green — engine 36/36, LM
  artifacts 36/36, EM driver 45/45; the counts exceed the banked 24/24/24/24/36/36 because the
  scripts grew during the hardening round (mtimes 11:12-11:49, before any diagnostic cell ran at
  12:29+), so the banked counts describe an earlier revision and every current check passes.
  (b) Entropy cross-check confirmed and stronger than State's claim: the accepted decoder
  contract `H4ResourceContractJob.kFA99bygctlt` and the gate agree to all 16 digits on ALL THREE
  probe indices (1.9224904277792318 / 3.06327078252676 / 5.860070935311313), same array sha
  `c6d85886` for the lowest. (c) Legacy-2g rerun timeline verified on disk: rerun finished
  12:25:39, earliest diagnostic cell started 12:29:31 — the "completed before any cell started"
  narrative holds, with under four minutes of margin. (d) Feedback commits pushed. The State
  fan-in inode proposal is ACCEPTED and the PLAN.md storage-placement paragraph amended
  (planner re-measured: 342 input symlinks in the live gate dir and in each of the three cleared
  copies, 1,026 debris); cleared-dir deletion stays the user's call.
- 2026-08-22 (1g.2a item 4 descriptive PER half VERIFIED; verdict 24 ACCEPTED, no hand-backs).
  Everything checked reproduces from the artifacts, not the log: all 80 table cells match
  `H4ContextDiagnosticPerJob.IYHS4cX3j3XV` `per_pooled` at four decimals; the bed is the
  registered one by content hash (`gold_sha256` and `accepted_h1_sha256` both equal the
  verdict-21 artifact `H4RealSeedPerJob.vu6Dp6HkJ2pH`; 890 = 432+458); the count-0 column
  equals the frozen 1g.2 direct-Q PERs for all four real starts and is hash-identical across
  the four fitting-LM positions in the payload. The frozen-decoder claim was verified at its
  source: all 425 decoder blocks in `H4ProvisionalMaximaJob.ejmy4sdTOcS3` are kind local with
  null lm_scale/insertion/beam, and all 60 decode cells carry ONE decoder block and ONE
  `phone_prior_sha256` (`9b4a00f4`), so the prior is fixed across every column as the design
  point requires. Verdict 24(a) was reproduced from the raw hypotheses, not the file hashes:
  legacy-2g and matched-2g decoded sequences are byte-identical in 15/15 repaired cells (their
  differing `hypotheses_sha256` is the re-aimed `fitting_lm_sha256` binding, present as
  described), and the reference-row divergence counts against matched-4g are exactly
  664/796/849 of 890 at counts 1/2/4. Verdict arithmetic checked unrounded: count-4
  order-4-minus-legacy deltas are -0.00365 (espum; the verdict's -0.0037 is the correct round,
  the displayed-table -0.0036 is not), -0.00221, -0.00469, +0.00064, and the largest order
  effect is espum matched-3g -0.00624. Scope discipline is right: the no-selection reporting
  rule is pre-registered inside the artifact with the label-circularity disclosure, held-out is
  SEALED, and the verdict claims no order choice with the own-minus-donor half still open.
  Observation, no action needed: `h4_context_decode.py` has no synthetic suite of its own, but
  it reuses the suite-covered decode kernel (`repaired_b_local_decode`) and PER arithmetic
  (`pooled_per`/`utterance_errors`), and the count-0 byte-equality to the frozen 1g.2 decode is
  a stronger end-to-end anchor for the wiring than a synthetic suite would be.
- 2026-08-22 (1g.2a item 4 own-minus-donor half: verdicts 25(a)/(b) and the frame VERIFIED;
  ONE HAND-BACK on the 25(c) correlation numbers, conclusion unaffected). Verified from the
  artifacts: all 80 table cells reproduce from `H4ContextOwnMinusDonorJob.SygqXhY8F2Qt` as the
  mean over ten `weighted_mean` entries at four decimals; every cell carries exactly ten
  assignments with constant eligible counts 235/278 and `ineligible_cells` is empty; the
  smoothing-bridge maximum is exactly 2.702e-4 at `real/fingerprint` count 1 as claimed; the
  five count-4 order-4-minus-legacy deltas reproduce to the digit; `donor_table_sha256` equals
  the frozen 1g.2 donor table (present in the maxima, winner-audit and selection-surface
  artifacts); `accepted_h1_sha256` equals the PER half's; the module imports
  `compute_selection_aggregate` from the 1g.2 selector module as stated; and the decisive
  count-0 reuse claim is proven at bit level — all five count-0 aggregates occur bit-equal
  inside `H4SelectionSurfaceJob.MKHfnUO9XwkU`'s frozen surface. The one hand-back of this
  round — verdict 25(c)'s six correlation numbers rested on no artifact and reproduced under
  none of nine conventions tried — is ABSORBED and CLOSED same day: the corrected verdict now
  prints +0.917, -0.629, +0.822, -0.993, +0.907 and +0.112 pooled, banked by
  `H4ContextAgreementJob.zd6RBdYcvzti` with the convention stated in its own output
  (Spearman, average-rank ties, unrounded pooled PER against the ten-assignment mean, twelve
  repaired cells per start, count 0 excluded as one artifact in four columns). Verifier closure
  checks, all passing: the banked values equal the verifier's independent computation to full
  float precision; the artifact's `per_sha256` and `scores_sha256` are exactly the sha256 of
  the two half payloads, so the agreement read is content-bound to what it correlates; and the
  correction's stated mechanism is proven rather than asserted — positional (first-occurrence)
  tie-breaking reproduces the six original numbers EXACTLY (+0.902, -0.636, +0.734, -0.979,
  +0.853, +0.110), so the discrepancy is fully explained and both number sets are understood.
  The verdict's conclusion was never in question; verdict 25 is now ACCEPTED in full and item 4
  is closed on both trajectories.
- 2026-08-22 (1g.9 experiment-1 launch VERIFIED; three implementer pins RATIFIED; one clause-0
  reading pre-stated before results). The launch is real: `H4CollapseLocateJob.gZ9d6e3E7ZGu` on
  disk and running under a live engine at check, the experiment-2 constrained arms correctly
  unbuilt until the planner rules on this output. Suite re-run by the verifier: 6/6 (torch
  forward-backward against `channel_h.marginal_forward_backward`, gradients against central
  finite differences, `E[N|T]` against a direct sum). Docstring review (speech-llm `d08cd88`) —
  every pinned constant traces: `p_text` is the `phone_prior` array of the accepted calibration
  over the complete 39,630,169-line unpaired phone corpus, the exact corpus and preprocessing the
  fitting LM is estimated from, unsmoothed; `r_target` is DERIVED from the frozen H1 length-law
  fit exactly as the subphase registers, with the memoryless reading `1-p` = 0.7644 beside it and
  the forced deleted-silence-boundary count reported as the one term a healthy posterior may
  legitimately exceed the target by — RATIFIED, the caveat is real and correctly separated;
  `lambda_equal = ||grad L_HMM|| / ||grad L_term||` in the `B = softmax(theta)` parameterization
  — RATIFIED, it answers the dead-band check and gives experiment 2's lambda magnitude a
  traceable origin; the fold pin (gradients on the 6,414 update role, clause-0 read on the
  matched 890 with the update-role posterior beside) — RATIFIED, it removes a stage/fold
  confound the registered clause 0 had left open. Pre-stated by the planner NOW, before any
  artifact exists (also in `PLAN_1G.md` 1g.9 Status): the clause-0 decision read is the COUNT-4
  row, where posterior and decode read the same repaired emission table; count 0 is context
  only, because there the decode reads the start's direct `Q` while the posterior reads its `B`
  — the asymmetry State itself discloses. Also noted for the clause-0 reading: the frozen local
  decoder is a per-unit argmax with run collapse and consults neither the language model nor the
  duration law, so a decoder-resident collapse is a live mechanism this diagnostic can genuinely
  separate, not a straw man.
- 2026-08-22 (1g.9 experiment-1 result VERIFIED in full; verdicts 26-29 ACCEPTED; clause-0
  ruling issued in `PLAN_1G.md` 1g.9 Status). Every number in both approach-15 tables
  reproduces from `H4CollapseLocateJob.gZ9d6e3E7ZGu/output/collapse_locate.json` under the
  verifier's independent recomputation: all ten (posterior TV, posterior rate residual, decoded
  TV, decoded rate residual, distinct symbols, clause-0 flag) tuples to the printed precision,
  the `lambda_equal` range 8.085e+05 to 1.454e+08, the pseudo-pair decoded AH excesses +0.8345
  and +0.4151, and the per-retained-unit log likelihoods to all four decimals including the
  divisors (60,604 selection / 584,424 update retained units). The pinned constants are stated
  inside the artifact as required: `r_target` 0.7643970 with its derivation text, `p` 0.23560298,
  the memoryless reading, the forced deleted-silence boundaries (53,498 update / 5,110
  selection), and `p_text`'s source with the 39,630,169-line count. The clause-0
  operationalization (posterior passes both targets while the decode meets fewer) is a faithful
  reading of the registered pattern, and the decision does not rest on it: the all-five-pass
  fact plus `lambda_equal` decides. One precision note, non-blocking: verdict 27's "+0.415
  reproduces the 1g.2 audit's +0.417" compares an excess over `p_text` with the audit's excess
  over the gold 890 unigram — two different references agreeing to 0.002, which is
  corroboration, not identity. Verdict 28's counterexample is verified and is the round's
  deepest finding: the lowest posterior total variation of all ten cells belongs to the start
  with the worst likelihood and the worst decode, so the proposed constraint statistic is
  satisfied most easily by the channel carrying the least audio information. Clause 0 FIRED;
  experiments 2-3 stay unbuilt (verified: no constrained-arm graph exists); the direction fork
  is with the user.
- 2026-08-23 (1g.10 launch round; VERIFIED AND ACCEPTED; beam-256 cut ruled in `PLAN_1G.md`
  1g.10 Status). A sampled chunk (`H4SequenceDecodeChunkJob.S25eY8DyW2cx`: pseudo-pair, count
  4, beam 256, lm_scale 1.0, insertion penalty -1, chunk 26) traces every constant to the
  registration: KenLM `CreateBinaryLMJob.hvZoC014xnIe`, H1 `Phase1gH1Job.HbxKiuBTJ8aN`, count
  adapter, contract `H4ResourceContractJob.kyMk7fwm027C`, 2 GiB / 2 h; 1,773 of the 2,304 new
  chunk dirs existed on disk mid-submission at verification time. The docstring pins are
  ratified as registered (imported 1g.9 targets; 1g.2 `edit_distance`; per-cell matched babble
  null, 1,000 draws, seed 42; the two-way explanation duty). The implementer's budget flag was
  correct and material: the planner takes the offered cut -- beam 256 restricted to one fixed
  shard per cell (canonical heaviest selection-role shard, same index everywhere), dropping
  1,152 chunks; surplus beam-256 jobs are deleted per the standing delete rule; beam 512 is
  uncut. The reader's key set re-registration is with the implementer.
- 2026-08-23 later (beam-256 cut VERIFIED EXECUTED; hand-back ABSORBED). On-disk chunk count
  is exactly the predicted post-cut census (1,332 = 144 historical beam-table cells + 1,152
  beam-512 + 36 single-shard beam-256 probes); the probe shard traces to the measured
  contract's own shard block (`H4ResourceContractJob.kyMk7fwm027C`: chunk_index 28, 2,466
  retained units), not a typed number. The implementer's arithmetic correction is accepted:
  the surplus is 1,116 (31 of 32 shards on 36 cells), not the ruling's 1,152, which would
  have deleted the kept probes. Agreement/drift columns carry their 28-utterance support in
  payload, header and `beam_probe_note` as ruled.
- 2026-08-23 (1g.10 result round VERIFIED; the duty's block is endorsed; 1g.10a registered in
  `PLAN_1G.md` 1g.10 Status 2026-08-23 result). Verified on disk: 1,332 chunk dirs with ZERO
  `error.run*` markers (completion proven by the reader having consumed every merge;
  per-chunk `finished` markers live inside `finished.tar.gz` after auto-cleanup);
  `H4FullModelDecodeReadJob.MXhi20TtG1I0`'s payload carries the duty verdict verbatim
  ("DECODER DEFECT SUSPECTED -- ... no cell of this table may be read until that is
  explained"), `beam_probe_utterances` = 27, and the printed grid matches approach 16's table
  (agreement min/median/max 0.2222/0.6111/0.8889; margins 1.210e-03/4.345e-03/1.540e-02;
  zero cells passing either branch bar). The implementer's refusal to read any cell, the
  fenced decoder-health observation (verdict 31), and the facts-not-rescues framing of the
  mixture and 27-utterance observations are all exactly right. The 27-vs-28 correction is
  accepted from the artifact -- the prior round's "28-utterance support" line above is
  OBJECTIVELY WRONG on that count (the ruling's shard index 28 was conflated with the
  utterance count; the probe shard holds 27 utterances). NEXT: the 1g.10a cross-beam defect
  diagnostic (banked data only; step-zero selection-rule question; scoring-determinism and
  pruning-monotonicity tests at 1e-9 nats per retained unit; pre-registered consequences
  including the conditional beam-1024 probe) is registered and ready for the implementer.
- 2026-08-23 (step-zero STOP VERIFIED AND ENDORSED; tests re-registered). The stop was exactly
  the registered behavior. Planner verified both code claims in `channel_h.py`: selection is
  the registered argmax form, but `prune` ranks prefixes by the logsumexp of their surviving
  states and discards whole prefixes -- the banked score is a pruned path-sum and is
  legitimately beam-dependent, so the original TEST A would convict a decoder behaving as
  designed (117 of 588 same-winner scores at 1e-9 is description, not a defect count). The
  scouting anomalies (13 same-sequence scores lower at beam 512; 8 lower total retained
  masses) are consistent with non-nested kept sets under this pruning rule and carry no
  verdict. REPLACEMENT (in `PLAN_1G.md` 1g.10 Status 2026-08-23 re-rule): TEST D
  (bit-determinism, double-decode of three disclosed cells at beam 256, 1e-12 nats) and
  TEST U (banked pruned score <= exact all-alignments forced score of the same sequence
  + 1e-6 nats, for all 972 utterance-cells and both beams' winners -- the impossibility
  bound that also polices the scouting anomalies), plus the exact-currency winner-gap
  distribution as context. Consequences unchanged: any violation blocks; both passing
  discharges the suspicion as a designed-in approximation and unlocks branches (ii)/(iii)
  as registered.
- 2026-08-23 (1g.10a launch AND result round VERIFIED; DISCHARGED; consequences applied in
  `PLAN_1G.md` 1g.10 Status 2026-08-23 discharge). Launch verified: the module docstring
  carries the re-ruled rule verbatim; the exact score is the pre-existing
  `marginal_path_log_score` (the chunk jobs already bank it as `channel_log_probability`),
  so the identity test is not circular -- and the test suite's own load-bearing check (real
  decoder at a prune-nothing beam must reproduce the exact score to 1e-9, all alternatives,
  silence policy on, plus a pruning-bites check against vacuity) is exactly the right
  validation; tests re-run by the verifier, 9/9; inputs trace to the registered KenLM
  `CreateBinaryLMJob.hvZoC014xnIe`, frozen H1 `Phase1gH1Job.HbxKiuBTJ8aN`, and the banked
  chunk artifacts. Result verified from the artifact: TEST D 81 utterances re-decoded, 0
  violations at 1e-12; TEST U 1,944 checks, 0 violations at 1e-6; exact-currency 352
  positive / 32 negative of 384, median +0.0109 nats per retained unit. The report's fences
  (reads no cell, quotes no comparison, authorizes nothing itself) are exactly right. The
  planner applies the pre-registered consequences: beam-512 table readable as descriptive
  with per-cell agreement disclosed; 1g.10b (beam-1024 probe, 36 cells x contract shard 28,
  reader extension, 26-of-27 cross-channel bar) is registered and ready to build.
- 2026-08-23 (1g.10b blocker VERIFIED; ruled OPTION (b) in `PLAN_1G.md` 1g.10 Status
  2026-08-23 ruling). The stop was right and all three load-bearing claims verify in the
  code: `DECODER_BEAMS = (64, 128, 256, 512)` (`channel_h.py`), `cells` is a hashed
  constructor argument of `H4GlobalBeamTableJob` (`h4_beam_jobs.py:399`), and the 1g.10 read
  consumes `out_global_beams` (`config_sae_1g_h4_full_model_decode_v1.py:222`) -- option (a)
  would orphan the banked global-beam table and the discharged read exactly as reported. The
  self-correction on the source-identity guard (runtime drift guard, not a hash input) is
  the right kind of checking. Ruled: option (b) with identity guards -- imports not copies,
  beam hard-pinned to 1024 plus a 512 parity mode, and a mandatory parity cell (probe class
  at beam 512 on the median observed-agreement cell must byte-reproduce the banked chunk's
  one-bests and scores; the 1g.10b reader refuses beam-1024 columns without it). Option (c)
  stays rejected as analyzed.
- 2026-08-23 (1g.10b and 1g.10c launch rounds VERIFIED; the 1g.10c resample flag RULED). The
  1g.10b build satisfies every option-(b) guard, including the source-grep enforcement of
  imports-not-copies and the parity cell read from the 1g.10a artifact's own disclosed
  selection; tests re-run by the verifier, 8/8; 37 probe dirs on disk; both discharged
  readers hash-unchanged as claimed. The 1g.10c build likewise: tests re-run, 9/9 (the
  load-bearing ones -- genuinely paired, non-finite deltas refused -- are the right ones);
  265 job dirs on disk mid-submission; the sizing mechanism (KenLM prefix cost linear in
  hypothesis length under a lengthening bonus) is real and accepted; the production merge
  reuse and the in-code refusal of the excluded pseudo-pair row are both endorsed. The
  flagged convention is ruled in `PLAN_1G.md` 1g.10 Status 2026-08-23 launch ruling:
  STRATIFIED resample (within fixed 432/458 splits) is the primary interval, matching the
  fixed-composition estimand and the family convention the implementer correctly pointed to
  (`h4_harness._bootstrap_content_values`); the unstratified interval prints beside as
  sensitivity. Decided before any statistic exists; only the reader re-hashes.
- 2026-08-23 (stratified ruling VERIFIED EXECUTED; absorbed). The read revision is now a
  separate constant from the decode revision -- the right call, and load-bearing:
  `implementation_revision` is a hashed chunk argument, so a shared bump would have re-hashed
  all 256 running decode jobs. Census verified: 265 chunk dirs, zero orphans, only the reader
  re-keyed (`vJSHAkECj8hH` -> `H4InsertionBonusReadJob.da3bGeQIkS0R`). Tests re-run by the
  verifier, 12/12 -- the new fixture proving the stratified draw holds composition fixed and
  tightens the interval exactly where the two splits disagree in sign is the right proof of
  the ruling's mechanism. Nothing further; awaiting the decode results.
- 2026-08-23 (1g.10c fold-bug round hand-back: RESOLVED, verified on disk). The rerunning
  reader was the orphaned unstratified v1 `vJSHAkECj8hH`, submitted by a manager still
  holding the pre-re-key graph. Resolution checked by the verifier: the orphan is inert (no
  `finished` marker, error marker renamed to backup, its SLURM rerun gone from the queue),
  no stale manager is alive (the only live manager is entry 8's), the v2 `int('HH')` defect
  -- phone strings and an unpacked dict passed into `_decoded_statistics`, whose contract
  is symbol ids, ordered lengths, dict out -- is fixed at speech-llm `3d395de` with two
  direct-call regression tests, and no number was ever produced by any broken run. One
  deviation from my instruction, ACCEPTED: the v1 dir is left in place rather than deleted
  -- the standing delete rule targets wrong trainings burning compute, an inert errored
  dir is debris, and it joins the cleared-dir debris list awaiting the user's call. The
  why-the-12-tests-missed-both analysis (fixture-fed statistics, never a direct call into
  `_fold`/`_decoded_statistics`) matches the standing test-the-call principle; endorsed.
- 2026-08-23 (1g.10c result VERIFIED; planner reading and closure in `PLAN_1G.md` 1g.10
  Status 2026-08-23 1g.10c result). `H4InsertionBonusReadJob.da3bGeQIkS0R` finished with
  zero outstanding error markers; parity PASS printed by the producer. Every number in the
  Approach 16 table -- all eight paired deltas, both interval columns, frac-improved, the
  pooled PER/length columns -- matches `insertion_bonus.txt` verbatim, and the payload
  prints its own resample conventions (primary stratified at fixed 432/458, unstratified
  named sensitivity), satisfying the derived-statistics-need-a-job rule. Tests re-run by
  the verifier, 14/14. Verdict 36 is faithful to the table: four control cells positive
  excluding zero, three of four real-arm cells negative excluding zero, the fourth
  straddling; within-channel, paired, cross-channel kept closed with contract-shard
  agreement disclosed (0.593-0.778). Verdict 37's stratification-made-no-difference report
  (every bound within 1e-4) is verified and is the right way to retire a convention ruling.
  Ruled in the plan: 1g.10c CLOSES -- mechanism confirmed causal, truncated-grid concern
  discharged, no further decode-parameter probes on this harness.
- 2026-08-23 (1g.10b result read by the planner from the artifact; ruling in `PLAN_1G.md`
  1g.10 Status 2026-08-23 result). `H4Beam1024ReadJob.tKbQ0MHLdX03`: parity cell PASS (the
  option-(b) identity guard held end to end), 0 of 36 cells clear the 26-of-27 bar (best
  24/27; median 512-vs-1024 agreement 0.704 vs 0.611 at 256-vs-512; drift per unit down in
  25 of 36 cells and up in 11). The bar fired as designed: cross-channel quoting stays
  closed, beam escalation is not funded (trend ~+0.09 median agreement per doubling),
  within-channel paired reads remain the standing currency. The implementer's banking of
  this result in Approach/Verdicts is expected in their next round; this entry records the
  planner-side reading so the two can be cross-checked. CORRECTION (2026-08-23, after
  verdict 35): this entry originally claimed drift fell "in every cell" -- WRONG; the
  implementer's 25-down/11-up count is re-verified by the planner from both artifacts and
  is the record. The ruling is unaffected (11 cells with rising drift under a doubling
  argue harder against escalation, not softer). The same universal claim in the PLAN_1G
  Status entry and in commit 66e4f5bf7's message is corrected by replacement in the plan;
  the commit message stands as history.
- 2026-08-23 (1g.11 experiment 1 VERIFIED COMPLETE; both flags RULED as Approach amendments,
  `PLAN_1G.md` 1g.11). Independent verifier reads: the frozen quantizer's `pca_components`
  is (96, 1024) and centroids (500, 96) -- flag 1's fact confirmed from the artifact, the
  128 pin discharged as a ceiling-plus-no-refit instruction, sensitivity cell dropped as
  vacuous. Flag 2's token reading RATIFIED (count identity and "that token's frames" both
  select it; it is what the paired read needs), with the reading rule now pinned in the
  plan: after a passing pipeline check, a token mismatch outside the absorbed set is a
  STOP. The job's own report matches the State entry field for field, the twin check landed
  at 100.0000 pct (all 2,184 absorbed tokens included), and `g11_continuous_test.py` re-run
  by the verifier is 19/19. The segment-fitted (not frame-fitted) component scale is
  ratified with the implementer's stated reason. The nested-gold fixture story (a flat read
  would have fitted the scale on evaluation utterances and looked right) is exactly the
  leakage class the fit/assign split exists for -- good catch, nothing further needed.
- 2026-08-23 (constrained update rule round VERIFIED and RATIFIED; ruling in `PLAN_1G.md`
  1g.11 Status). The clamp-as-constrained-maximizer argument is mathematically sound (per
  component the expected complete-data log likelihood is unimodal in the variance with
  maximum at the weighted second moment, so projection onto `{var >= v_min}` is exact) and
  is fixture-checked where the floor binds, which is the right insurance against a silently
  non-monotone EM. The shared-recursion seam is the load-bearing attribution decision and
  is verified safe three ways: `source_identity` is a runtime guard, not hashed, so banked
  jobs stay put; no live manager holds a channel_h consumer (only sae_3e1_d9_1 runs); the
  five channel_h-family dirs without finished markers are known debris. Verdicts 38-40 are
  faithful to the artifact (the verifier had independently read the same numbers before the
  banking); verdict 39's convexity theorem is checked and correct -- the token mean is a
  convex combination of two segment means inside one convex nearest-centroid cell.
  Verifier re-runs: `g11_gaussian_test` 21/21, `h4_context_engine_test` 36/36,
  `h4_collapse_locate_test` 6/6.
- 2026-08-23 (primitives shakeout round VERIFIED; the decode fix RATIFIED; no plan change --
  the experiment-2 build clearance stands). The `validate` flag fix (speech-llm 50ee93f) is
  correctly scoped: the verifier read the diff -- validation is not dropped but REPLACED
  with a density-appropriate check (shape, finiteness, strict positivity), and the
  duration-occupancy reduction stays the one shared implementation, so the attribution
  property survives the fix. The named cause (every decode fixture was one-state, so the
  two-state reduction branch was never exercised) plus the regression test on the real
  topology is the right closure for a fourth wrong-call instance. The eight-utterance
  login-node shakeout is within the standalone-probe allowance and is correctly disclaimed
  as non-evidence; its retained share ~0.79 independently corroborates the corpus reading
  (720,315 retained of 919,248 pre-mask tokens = 0.7836), so the plan's phone-stream count
  is confirmed as the post-silence-mask count. Verifier re-runs: `g11_gaussian_test` 24/24,
  `test_channel_h` 23/23.
- 2026-08-23 (retained-token keying round VERIFIED; experiment-2 inputs accepted as resolved;
  no plan change). The keying design is the right discipline for the wrong-call failure
  class: `retained_token_view` derives its index by CALLING `h4_jobs._retained_runs` and
  then asserts against that primitive's own output, so a misalignment surfaces as an error
  instead of a corrupted attribution verdict. Verifier spot-checks: `g11_gaussian_test`
  re-run 28/28; `Phase1gH1Job.HbxKiuBTJ8aN` finished on disk; start labels cross-checked
  against the artifacts' own fields, not the typed list -- `H4RepairJob.ViPSmq4Am8vX`'s
  inputs name `espum_calibration_start` and `.Ds0zM1NTY2C1`'s start manifest carries
  `name = random-map` in its own JSON. Role counts (6,414/890/7,304/1,112) match the
  registered splits, 0 twin-coverage misses, and the 0.7737 update-role retained share is
  consistent with the 0.7836 corpus ratio. The 4,709-job zero-unfinished reuse claim for
  the prelabel-surfaces graph is accepted on the implementer's check (a no-spend claim).
  Experiment 2 is wiring from here; the clearance stands.
- 2026-08-23 (fold-scale rewrite round VERIFIED; CPU sizing ratified in the plan Status).
  The 230 GB tensor was caught by PROJECTING to the registered fold before any job existed
  -- the same-scale discipline working as intended; a fixture-passing implementation that
  dies at scale is precisely the trap the banked constants rule names. The rewrite's two
  safeguards are the right ones: matmul forms ASSERTED against the direct forms for both
  covariance variants (algebra checked, not assumed), and a chunking-invariance test so the
  fold's answer cannot depend on batch boundaries. `floor_share` per count satisfies the
  gate's clause-4 honesty line by construction. Verifier re-run: `g11_gaussian_test` 35/35.
  The measured cost (5.9 s / 1.8 GB per 1,024 utterances, four updates) makes the cells a
  small sisyphus CPU job -- ruled compliant with the registration's GPU clause, which
  covers model-forward computation only (plan Status clarification).
- 2026-08-23 (experiment 2 result round VERIFIED; approach 18 and verdicts 42-44 faithful;
  one precision note on verdict 41). `G11GaussianRepairJob.NogH62uMEI7T`'s report matches
  the banked table cell for cell (all twelve rows, header constants, the 68-of-45.6M clip
  count); the report itself refuses clause verdicts before the controls exist, which is the
  pre-registration discipline in the artifact. The clip-count reporting fix is endorsed
  with its reasoning: a safeguard that fires and prints 0.00000 reads as evidence nothing
  happened. The underflow event is confined to the disclosed per-row relaxation cell -- the
  PRIMARY tied arm is numerically clean in every cell. Internal consistency checked: the
  espum per-row and tied count-0 rows are identical, as they must be (the relaxation only
  diverges through updates). Verifier re-run: `g11_gaussian_test` 40/40. Both hand-backs
  ABSORBED same day: verdict 41 now carries the corrected largest/smallest examples plus a
  dated correction naming the original error (a count-0 magnitude ranking mislabelled as
  rises) and listing all six rises -- number-identical to the verifier's recomputation;
  the stale eight-utterance sentence in State is cleared. Nothing further on experiment 2.

- 2026-08-24 (experiments 3-4 result round VERIFIED; gate RULED in `PLAN_1G.md` 1g.11 Status:
  clause 3 FAILS, continuous emissions not funded at this operating point, the
  wav2vec-U-faithful follow-up not funded, route direction to the USER). Verification
  performed: `G11EvaluateJob.sWoS1bP4Nd12/output/evaluate.txt` matches approach 19 cell for
  cell -- all 24 admission/content rows, all seven clause-3 paired deltas, and the
  three-column babble bar (worst 100-vs-1000-draw disagreement 6e-04 at
  `gaussian|random_map|tied|4`, matching verdict 51). The observation-null artifact matches
  (log-likelihood -79,970,371.2 to -75,850,789.7; 645,028 redrawn vectors = 584,424 update +
  60,604 selection retained tokens, an identity the reports satisfy across jobs; floor
  0.0000, clipped 0). Clause assignments recomputed from the printed columns: 7 of 24 cells
  fail clause 1 exactly as marked, only `table|controlled/reference` reaches clause 2.
  Comparator provenance verified: the table arm's ten one-best files are `H4LocalDecodeJob`
  outputs of the audited prelabel-surfaces graph, each carrying its arm name, role and
  repair count in its own info file, with mtimes of 2026-08-21 -- pre-dating the 1g.11
  registration -- so "banked, never re-decoded" holds by construction. Verifier re-run:
  `g11_exp34_test` 26/26. Verdicts 45-51 are faithful to the artifact; the evaluate job's
  refusal to invent a "comparable" threshold after seeing the intervals is the registered
  discipline, and the two rulings that refusal required (no-number ruling; the control's
  clause-1 status does not remove it from clause 3) are recorded in the plan, not here.
  Nothing further on 1g.11; no new work is licensed by this round.

- 2026-08-24 (1g.12 experiment 1 round VERIFIED; dated verdict line appended to `PLAN_1G.md`
  1g.12 Status). `G12ResourceGateJob.3h2iIpk6lpaB/output/resource_gate.txt` matches approach 20
  cell for cell (all seven table rows, 0.4345 h per E-step, 4 h / 17 h / 4 GiB, PASS one curve
  and RESOURCE_INFEASIBLE single-process). Verdict 53's arithmetic checks (1+39+39^2+39^3 =
  60,880; 60,879 reached = all but the all-BOS history). Verdict 54's comparator verified from
  `H4ContextResourceGateJob.HA1vzRL7MEAz` itself: 50.82 s max_time on the same probe utterance
  and the same heaviest chunk 2 of 32 (19,515 tokens), 48.88/50.82 = 3.8% under. All ten
  experiment 2/3 cell jobs exist on disk (launches verified by job dirs, not aliases): five
  `lm_identity=accepted-2g` with the banked `G11GaussianRepairJob.NogH62uMEI7T` reproduction
  inputs and per-start reproduce keys, five `lm_identity=matched-4g` with the
  `H4MatchedLmJob.VpVkGMMy7xKW` automaton and reproduction disabled, all five starts in each
  column, none finished. Verifier re-run: `g12_gaussian_context_test` 57/57,
  `g12_resource_test` 50/50 -- the Catalog's counts are right and State's "45/45" is stale.
  Two hygiene items for the implementer, no send-back: (i) State still carries the gate as IN
  FLIGHT and the old test count -- overwrite on next touch; (ii) the gate job has a cleared
  FINISHED first attempt (`.cleared.0001`, 49.06 s, "2 GiB either way") re-run after the memory
  request accounting was made conservative (now 1.5x the sum of engine and parent RSS, 4 GiB) --
  timing within run noise, verdict unchanged either way, but a cleared-and-re-run measurement
  should be one scratch line so the banked table's provenance is readable. Verdicts 52-54 are
  faithful to the artifacts.

- 2026-08-24 (1g.12 experiment 4 build + relaunch and 1g.13 experiment 1 launch round VERIFIED;
  approaches 22-23; no verdicts handed, none due). Cell census matches disk exactly: all five
  bigram cells FINISHED with `parameters.npz` persisted (the 1g.11 non-persistence gap is
  closed) and their 1g.11 reproduction certificates asserted before writing; the five
  matched-4-gram cells were cleared by the filesystem kill and are re-running
  (`.cleared.0001` siblings); 20 readouts + 5 re-running = the 25 unfinished approach 22
  counts. The twenty `G12ExactReadoutJob` cells have NO on-disk dirs yet -- consistent with
  approach 22's own "no cell has been decoded yet" (graph-enumerated census, unstarted jobs
  have no dir); their launch verification stays OPEN until dirs appear. Verifier re-runs, all
  at the logged counts: `g12_exact_decode_test` 33/33, `g12_readout_jobs_test` 22/22,
  `g13_stream_test` 35/35, `g13_jobs_test` 34/34 (speech_llm env), and
  `g13_faiss_reference_test` 10/10 under the `w2vu` env against real faiss 0.12.2-era code --
  the equivalence claim is tested against the reference implementation as registered.
  `G13StreamBuildJob.Ob8Rh8y51x9M` exists on disk and is running with parameters matching the
  registration field for field (K=128, iters 50, redos 3, seed 0, PCA 512, fit scope 2,849,
  bed 8,416, both banked dumps as inputs). Finished-cell outputs correctly refuse clause
  verdicts before their controls exist. Nothing to send back.

- 2026-08-24 (results round VERIFIED: approaches 21/22/24, verdicts 55-60; dated lines appended
  to `PLAN_1G.md` 1g.12 and 1g.13 Status). Recomputed from the banked tables: verdict 58's five
  matched-4g gains, both control/reference ratios (1.358, 1.336), verdict 55's reproduction
  range (1.9e-16..2.6e-15, ten of ten at 0 of 890 disagreements), verdict 59's ten count-0
  identity pairs (identical to the symbol in both emission models), verdict 60's collapsed-cell
  contrast (0.0152/3 LM-blind vs 0.9786/36 LM-aware, same parameters) and its
  "fewer-symbols-everywhere-else" claim (holds at every non-collapsed Gaussian cell), and
  verdict 57's derived chain (28.01/12.47 = 2.25x rate, 1,436,262/584,424 = 2.46x tokens,
  0.4345 h x 2.46 x 5 x 1.5 = 8.0 h). Artifact spot-checks match: `kHwPYElOcCPr` repair.txt
  criterion pair (-79,805,418.7 / -75,188,448.6), readout `6D1qhe7DtE9m` rows and certificate
  (violations 0, renormalized 1.274e-04 = the registered fit-side bound), `G13StreamBuildJob`
  stream.txt (28.01/s, 128/128, rarest 88, inertia 1.228e+11, PCA 0.9114, four deviations
  verbatim, labels read: 0). All twenty `G12ExactReadoutJob` dirs exist -- the launch
  verification left open last round CLOSES. `G13RoutesJob.hStPuE1UqLK6` and
  `G13VadFirewallJob.Usfy2NF0LiSQ` exist on disk (approach 24's table is empty pending their
  runs). State is current; earlier hygiene items absorbed. One wording nit, no correction
  needed: verdict 57's "to three significant figures" overstates the anchor's own precision
  (~28, two figures); the match itself is real. Verdict scoping is disciplined throughout --
  58 and 60 both refuse to be clause verdicts. Nothing to send back.

- 2026-08-24 (approach 24 results + verdict 61 + proposal 1 round VERIFIED and RULED; ruling in
  `PLAN_1G.md` 1g.13 Status). The route artifact matches verdict 61 line for line
  (`G13RoutesJob.hStPuE1UqLK6`: p 0.68898090, mean 3.215237, one-state ADMISSIBLE, two-state
  ADMISSIBLE); the ratio arithmetic recomputes (2.2315/0.701 = 3.18 refuted on seg12.5,
  2.2498/1.876 = 1.199 admissible here, band 2; mean durations equal 1/(1-p) both ways). The
  firewall finished with role-separated gold (`gold_update/selection/evaluation.pkl`) and
  `trim_masks.pkl` as registered. The guard code reads as proposed
  (`g12_repair_jobs.py:180-184`, hard-coded seg12.5 signature). RULING: guard amended to assert
  each route's own registered expectation -- seg12.5 unchanged, v1-equivalent route requires
  two-state ADMISSIBLE only, one-state verdict reported in every cell artifact and the gate's
  honesty report; the standing minimum-duration-2 topology is untouched and holding it fixed
  across subphases is what keeps contrast (d) a segmentation contrast. Full rationale in the
  plan. The implementer may edit the guard accordingly and proceed to experiment 3.
  Implementation VERIFIED (speech-llm `6c68303`): per-route registry with seg12.5 unchanged,
  v1 route two-state-only with one-state reported, unregistered routes refused; all four g12
  suites re-run clean after the edit (57/57, 50/50, 33/33, 22/22). Proposal 1 discharged.

- 2026-08-24 (1g.13 experiment 3 round VERIFIED; dated line appended to `PLAN_1G.md` 1g.13
  Status). Every number in approach 25 and verdicts 62-64 reproduces by independent
  recomputation: emission-table positivity and row sums, the entropy column, the full ten-pair
  total-variation matrix, the espum selections re-derived as the argmin of each curve with the
  weighted metric (ordinary perplexity divided by squared coverage) recomputed at all 126 curve
  points, both contrast rows read from their own banked artifacts (`97FwGhhItdpO`,
  `h4LngSZ4YvKL`, at their num_units=500 on their own stream), and the controlled reference's
  q / marginal / emissions re-derived outside the job to max difference 0.0 with the collapse
  proof repeated on all 3,565 labelled utterances. Suites `g13_reference_start_test` 20/20 and
  `g12_route_topology_test` 28/28 re-run. Hygiene, no numeric impact: (1) the espum projection
  emits `espum_calibration_start.npz`/`.json`, not `start.npz`/`start.json`, and its json has
  no top-level `name` field (only the upstream checkpoint sidecar's) -- anything globbing
  `start.npz` across the five starts silently misses that arm; (2) no start.json records the
  alphabet size as a named field (128 is implied only by array shapes) -- record `num_units`
  at the next touch; (3) the Catalog's code line was corrected in place by the verifier
  (objectively wrong references): `6bfa29d` carries `g13_firewall.py` only, the config wiring
  landed in `a0d2808`, and the two workspace files (`config/sae_1g_13_exp3.py`,
  `scripts/g13_reference_start_test.py`) are under no version control.

- 2026-08-24 (1g.13 experiment 4 first run, and the engine defect behind it; supersedes the
  first flag now that the cause is known). My occupancy flag (0/0 reached histories for four of
  five starts, NaN counted as zero by `occupancy > 0.0`) understated the scope: the implementer
  traced it to the backward recursion's normalizer in `h4_context_engine.py` -- beta was
  rescaled by ALPHA's per-frame normalizer, overflowing to `inf * 0 = NAN` under peaked
  emissions -- and the SAME gamma feeds `gaussian_context_pass`'s sufficient statistics, so
  experiment-5 fitting on this stream would have produced NaN parameters behind a finite
  likelihood. Fix `41127e8` VERIFIED independently: (i) code read -- beta is now rescaled by
  its own per-frame maximum (`h4_context_engine.py:359-360`), which cancels exactly because
  `joint` renormalizes over its own frame (`:368-369`) and the likelihood never reads beta;
  (ii) the re-measuring gate's v1 probes reproduce the first run's log-likelihoods to the last
  printed digit while occupancy becomes finite (fingerprint 59,204 reached where the broken run
  printed 0) -- a two-run identity on real data; (iii) banked 1g.12 unaffected by MY OWN scan,
  not the implementer's: zero RuntimeWarnings in all ten fitting-cell logs and every
  `parameters.npz` finite with minimum variance 3.96e-02, far off any floor; (iv) suites
  re-run by the verifier (h4_context_engine 48/48 including the new peaked-posterior case,
  h4_context_em 45/45, g12_gaussian_context 57/57, g12_resource 50/50, g13_resource_gate
  43/43); (v) hash claims verified -- `subphase` sits in `__sis_hash_exclude__` at 1g.12's
  value, so `3h2iIpk6lpaB` keeps its hash and its outputs are untouched on disk (mtime still
  10:13:35). Both guards are in place (E-step raises on a non-finite sufficient statistic, the
  gate raises on non-finite occupancy), discharging my items (a) and (b) from the first flag.
  Still standing: (c) headroom was thin at 9 h against the 11.5 h clamp on the superseded run
  -- read the re-measured request and the first real cell's wall clock against it; (d) never
  clear `3h2iIpk6lpaB` (its recorded code identity predates two edits at an unchanged hash).
  The persist-the-harness request is DISCHARGED: the equivalence check is now the registered
  `G12EngineEquivalenceJob.sWWDLbPKglfP` (verified in the next entry).

- 2026-08-24 (1g.13 experiment 4 re-measured round VERIFIED; dated line appended to
  `PLAN_1G.md` 1g.13 Status). `G12ResourceGateJob.cQ3wfqsTamPP` matches approach 26 and
  verdicts 65-66 in every field, and the sizing arithmetic recomputes: 32 chunks x 128.125 s =
  1.1389 h per E-step, x5 E-steps x1.5 = 8.54 -> 9 h one curve, x5 starts = 42.71 -> 43 h;
  chunk 2 is the heaviest at 48,417 tokens; the artifact names its subphase and carries the
  per-route topology block exactly as ruled (two-state asserted, one-state reported with ratio
  and ceiling). `G12EngineEquivalenceJob.sWWDLbPKglfP` verified: all five banked probe cells
  reproduce their log-likelihood and 60,879 histories exactly, worst posterior difference
  8.882e-16 against the 1e-12 tolerance, and the counter-case separation is present (the
  superseded normalizer non-finite on the 893-token peaked case) -- without it the equivalence
  would be vacuous. The superseded engine is derived from the live source by one substitution
  asserted to match exactly once, which I read and consider sound (a kept copy would drift).
  Suite `g12_engine_equivalence_test` 25/25 re-run by the verifier; the disclosed NameError
  recovery (marker renamed, stateless job) and the AST-lint follow-up (`dea3097`) are
  reasonable. Verdict 68's no-number-moves conclusion now rests on a registered artifact; my
  independent scan from the previous entry corroborates it. Two notes, no action needed:
  (1) verdict 67 says "at most 2.7% below" -- 1,675/60,879 is 2.75%, so 2.8% is the correct
  rounding; (2) three starts (controlled reference, espum, random-map) report IDENTICAL
  reached counts and arcs (59,319 / 2,372,760), which supports verdict 67's own caution that
  the column reads structural support rather than start quality. Launch-one-cell-first before
  the remaining experiment-5 cells (State next action 1) is the right embodiment of caution
  (c); keep it.
