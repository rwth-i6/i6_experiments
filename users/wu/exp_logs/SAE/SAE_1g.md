# SAE_1g — Evidence for a simple weak SAE initialization

## State

State as of 2026-08-25. Every subphase carried by this log is CLOSED: 1g.2 read and closed on its
gate (verdict 18); 1g.9 closed by its own clause-0 off-ramp (verdict 26); the 1g.10 family closed by
the planner (verdicts 34, 36); 1g.11 complete, clause 3 failing on its control (verdict 45); 1g.2a
items 1-4 complete (verdict 25); 1g.12 complete through all six experiments (verdicts 70-72); 1g.13
complete through all seven experiments, its gate read and ruled by the planner 2026-08-25
(`archive/SAE_1g_spec_legacy.md` 1g.13 Status).

Terminal run pointers: the 1g.12 gate table is `G12EvaluateJob.yJgxKex9peLp` (approach 30); the
1g.13 gate table and its segmentation contrast are `G12EvaluateJob.a3419LhkI7JT` (approach 31).
4,803 of 4,803 1g.13 jobs carry a finished marker, no manager of this setup runs and nothing is
queued; `sae_1g_13_exp5` and `sae_1g_13_exp6` are blocked behind `sae_1g_13_exp7` in
`sis_managers.sh` (shared graph).

Next experimental action: none is funded from this log. Closing the subphase, and any funding of a
wav2vec-U-faithful completion despite the negative (d) read, is the USER's word. Two items sit on
the USER's desk: the direction fork left by verdict 18, and whether the minimum-duration-2 topology
is enforced or relabelled (unresolved feedback, 2026-08-25).

Nothing is in flight.

## Gates and standing rules

Method, funding status and future work live in `archive/SAE_1g_spec_legacy.md`; this log holds
evidence. Thresholds as registered before their results, with amendments marked.

- **Structure screen (1g.0, H1, 1g.13 experiment 2).** Decision statistic = measured audio-pair
  mutual information divided by the maximum the tested shape allows. "A ratio at or below 2 is
  admissible" (`structure_screen.ADMISSIBLE_RATIO`), required under BOTH the plug-in and
  Miller-Madow estimators; if the two straddle, the cell is INDETERMINATE. ADMISSIBLE means "this
  class can produce a dependency this strong", not "this class is right".
- **Spectral two-class anchor (1g.4).** Hard gate, a conjunction: mass accuracy at least 0.85, at
  least 0.20 above the measured majority, and containment at least 0.85.
- **H4 global-beam boundary (approach 10).** For each `(lm scale, insertion penalty)` setting, the
  first adjacent beam pair passing at least 99.9% exact one-best agreement AND strictly less than
  `1e-4` absolute decoder-score change per retained unit on every representative freezes the
  smaller beam; a setting with no passing pair is ineligible.
- **1g.10 explanation duty** (written into the producing module before any statistic existed): tiny
  margins where instability is measured confirm the flat-score mechanism; wide margins with
  persisting instability indicate a decoder defect and BLOCK any reading of the cells. Registered
  levels: 0.999 adjacent-beam one-best agreement; flat threshold 1e-3 nats per retained unit.
- **1g.10a replacement tests** (re-ruled 2026-08-23, replacing the original TEST A, which would have
  convicted a decoder behaving as designed): TEST D bit-determinism at 1e-12 nats; TEST U banked
  pruned score <= exact all-alignments forced score + 1e-6 nats. Any violation blocks; both passing
  discharges the suspicion.
- **1g.10b cross-channel quoting bar.** 26 of 27 one-best agreement on the contract shard; a parity
  cell (probe class at beam 512 reproducing a banked production chunk) is mandatory or beam-1024
  columns are refused.
- **1g.9 clause 0.** Posterior within total variation 0.15 of `p_text` and rate within 20 percent of
  `r_target`, while the decode meets fewer. Pre-stated before results: the COUNT-4 row is the
  decision read (count 0 is context only, since there the decode reads the start's direct `Q` while
  the posterior reads its `B`). Clause 1 readability: decoded total variation <= 0.30.
- **1g.11 / 1g.12 / 1g.13 evaluation clauses.** Clause 1 (admission): decoded length in [0.80, 1.25]
  of gold. Clause 2 (content): margin of at least 0.05 over the babble null's 99th percentile, read
  on clause-1-READABLE cells only. Clause 3 (decision): paired per-utterance correct-phone delta
  with a 95 percent interval excluding zero, and the content-free controls' gain no more than
  "comparable" — the registration carries NO number, so the artifact prints the intervals side by
  side and the verdict is the planner's. CARRIED 1g.11 RULING (standing, fired in 1g.12 and 1g.13):
  a control gain that EXCEEDS the arm's with non-overlapping intervals is beyond any reading of
  "comparable". Clause 4 (honesty): variance-floor share zero, decoder exactness violations none.
  Interval convention (planner ruling 2026-08-23): STRATIFIED resample within the fixed 432/458
  splits is primary, unstratified printed beside as sensitivity; two-sided 95 percent percentile,
  10,000 resamples, seed 42.
- **Clause-1 rulings that were tested and NOT amended.** (i) A control that is itself clause-1
  INADMISSIBLE still counts in clause 3 (planner 2026-08-24, ruling ii): the registration scopes
  clause 1 to clause-2 readability and names clause 3's controls with no admission precondition, so
  removing it after seeing that removal flips the verdict would be an unregistered gate edit.
  (ii) Clause 1 is admission only, in both directions (verdict 48).
- **1g.2 controlled validation readings.** Correlation NEGATIVE iff upper 95 <= 0; a repair count
  SAFE iff upper 95 <= 0.05 (the gate's "no greater than"); `h4_lm_trigger` fires iff NO safe count
  exists. Bounds are one-sided 5th/95th percentiles with the two-sided pair beside them, 10,000
  resamples at seed 20260822, resampling channel cluster then donor assignment then utterance within
  split. Duplicate channels collapse by the artifacts' own `channel_array` sha256.
- **Deployment selector `Sel` (frozen before labels).** `Delta_ics = log P_Bc(U_i | z_ic)/T_i -
  log P_Bc(U_d(i,s) | z_ic)/T_d(i,s)`, each denominator that input's own positive retained-unit count
  after the frozen silence mask; a common source denominator or a phone-length denominator is
  forbidden. Per frozen assignment take per-split eligible-row means, combine at the fixed weights
  432/890 and 458/890 with NO renormalization after `no_swap`, then average the ten assignments
  equally; higher wins. Ties break on
  `legacy-2g;repair_count;local;lambda_outer_beta_inner;initializer;seed;update`. Construction
  likelihood is never a selector, fallback or tiebreaker. A failed selector or failed winner audit
  leaves H4 unresolved without reranking. A local winner needs no beam audit.
- **Resource rule.** Nothing may be requested from the scheduler on the strength of an estimate;
  requests are measured, sized at the standing 1.5 multiplier, and read from the gate artifact
  rather than written in a config. Limits: 11.5 h queue clamp (cannot be raised; these jobs do not
  resume and `min(11.5, requested)` applies) and 256 GiB.
- **Re-fit reproduction bar (1g.12 experiments 2-3).** A bigram corner must reproduce its banked
  1g.11 cell before writing any artifact: `log_likelihood_rtol` 1e-9 and decoded-utterance
  disagreements exactly 0.
- **Topology guard — AMENDED, current form.** ORIGINAL: the 1g.12 repair cell refuses to fit unless
  the route reads one-state REFUTED and two-state ADMISSIBLE (seg12.5's signature). AMENDMENT
  (planner 2026-08-24, current): one shared per-route registry — `seg12.5/phones` keeps exactly the
  pair it was verified at; the v1-equivalent route ASSERTS two-state ADMISSIBLE and REPORTS one-state
  into every cell artifact and the gate's honesty report; an unregistered route is refused.
- **Minimum-duration-2 scorer topology** is standing by the USER's 2026-08-15 ruling and is not a
  function of any measurement here (see verdict 61 and the unresolved 2026-08-25 label finding).
- **Rerun rule (set by the 1g.2 validation rerun, re-affirmed 2026-08-22).** A finished artifact may
  be rerun only when nothing has consumed it; a cosmetic gain never clears that bar. The `legacy-2g`
  rerun broke this rule and was safe only by timing (rerun 12:25:39, earliest consuming cell started
  12:29:31), not by design.
- **PILOT_ONLY rule — DEPARTED FROM by USER direction 2026-08-24 20:55.** The build rule registered
  only the two most expensive 1g.13 cells until their wall clocks were read; the USER asked for the
  subphase sooner, so the flag was flipped before either pilot finished and the remaining eighteen
  fitting cells were launched. What was given up: protection against a projection wrong by more than
  the margin already inside the request (2.02x Gaussian, 1.74x table). Both gates had already PASSED
  for exactly this population, and neither arm prints its own progress, so a pilot was all-or-nothing.
- **Descriptive-read rule (approach 13, verdict 21).** Plain-PER reads over a closed gate select
  nothing, fund nothing; any later decision using them must be re-registered with the label
  circularity disclosed as a supervision cost. Held-out LM perplexity never selects order.
- **Sealed data.** The 1,112-utterance evaluation role is never opened anywhere in this log.

## Approach and results

`T_phi` is the unpaired text converted to 39 stress-free ARPAbet phones. A "rung" is one fixed
audio-unit stream: adjacent-deduplicated raw codes, or one of the pooled streams `seg16`, `seg12.5`,
`seg9`. Job dirs for every cited run are in the Evidence index.

1. **Channel-shape screen (1g.0).** 2,703 dev-clean and 2,864 dev-other utterances; three shapes
   (one audio segment per text symbol; variable duration with conditionally independent emissions;
   two ordered emission states per text symbol). The one-segment column needs no gold duration; the
   duration-bearing columns were read at within-symbol rates measured from GOLD boundaries and are
   historical diagnostics that do not set a prospective candidate's duration.

   | audio stream | lag-1 mutual information | gold within-symbol pair rate | cross-utterance floor | gold cross-boundary share |
   |---|---:|---:|---:|---:|
   | adjacent-deduplicated raw codes | 3.2964 | 0.7093 | 0.2423 | 0.291 |
   | `seg16` | 2.6171 | 0.4226 | 0.4005 | 0.577 |
   | `seg12.5` | 2.3730 | 0.3164 | 0.4995 | 0.684 |
   | `seg9` | 1.9596 | 0.2037 | 0.6444 | 0.796 |

   Ratios on dev-other; parenthetical values are the only materially different dev-clean reads:

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
   cell above 2 (2.02 to 2.79). The raw-character independent-duration result is split-dependent.

2. **Spectral two-class anchor (1g.4).** Syllabic/non-syllabic split from the largest-eigenvalue
   eigenvector of the symmetric normalized Laplacian; text side is a positive control. The first
   audio read oriented the classes by segment duration; the registered energy/periodicity
   orientation was then run, flips `seg16` and `seg12.5` and leaves the verdict unchanged. Canonical
   read is the fixed 572-utterance dev-other evaluation fifth (the 540-utterance dev-clean fifth
   gives the same all-fail verdict).

   | side / stream | top-eigengap check | canonical mass accuracy | measured majority | verdict |
   |---|---|---:|---:|---|
   | text phones (`T_phi`) | pass | 1.0000 | 0.6095 | positive control passes |
   | text characters | pass | 0.9764 | 0.6130 | positive control passes |
   | audio raw | fail | 0.5488 | 0.5711 | fail |
   | audio `seg16` | fail | 0.4452 | 0.5449 | fail |
   | audio `seg12.5` | pass | 0.4968 | 0.5312 | fail |
   | audio `seg9` | unstable | 0.7867 | 0.5154 | fail |

   The later permutation and bootstrap uncertainty reads were NOT persisted in a catalogued
   artifact, so their reported numbers are not treated as evidence here.

3. **Deterministic hard two-class descriptor screen (1g.4).** Seven waveform descriptors on the
   frozen wav2vec2 50 Hz grid, averaged per unit over the 8,416-utterance seed bed (3,685,941
   frames), binary mass cut set from the syllabic proportion of unpaired text. No label enters the
   dump or the cut. Same fixed 540/572 evaluation fifths and silence-unit convention as approach 2.

   | stream | dev-clean best accuracy | dev-other best accuracy | dev-other majority | dev-other margin | gate |
   |---|---:|---:|---:|---:|---|
   | raw | 0.7894 | 0.7929 | 0.5711 | +0.2218 | fail |
   | `seg16` | 0.7849 | **0.8130** | 0.5449 | +0.2681 | fail |
   | `seg12.5` | 0.7629 | 0.7859 | 0.5312 | +0.2548 | fail |
   | `seg9` | 0.7588 | 0.7824 | 0.5154 | +0.2671 | fail |

   Energy is the best descriptor on every stream. These seven descriptors are seven alternative
   estimators of the SAME syllabic/non-syllabic target, not the six independent articulatory
   memberships the registered soft-product specification requires, so this screen does not answer
   that prerequisite.

4. **Exploratory phone repair rehearsal (E5).** The completed job exercises the soft two-sub-state
   Baum-Welch implementation on `seg12.5` but does not implement the corrected experiment: all 2,864
   dev-other utterances both fit and score the oracle map; preprocessing differs from the frozen 1f
   fixture; the code field `fraction_correct` is MISNAMED (it is the probability of RETAINING the
   reference label, so 1 retains all and 0 redraws all — the old prose had it backwards); hard
   unit-to-phone start with 0.9 mass on assigned units; fixed mean duration 1.463 from the old
   gold-boundary point; every 80th text line capped at 300,000; 30 emission-only steps under a
   pinned phone bigram; posterior argmax with the stopping iteration chosen by weighted phone-LM
   perplexity; no real seed and no treated control.

   | configured retention | realized units matching reference | start PER | LM-selected PER | step-30 PER |
   |---:|---:|---:|---:|---:|
   | 0 | 0.016 | 1.0109 | 0.8409 | 0.8409 |
   | 1 | 1.000 | 0.4865 | 0.4589 | 0.6699 |

   Engineering evidence only: the same utterances built and scored the reference and no real seed or
   control was run, so these numbers fire no gate.

5. **Banked phone-seed artifact audit.** Neither 1f seed persisted a per-unit map and both were
   fitted on all 8,416 utterances, including the fixed evaluation audio, so both are transductive
   provenance. The ESPUM checkpoint is a context-dependent convolution (`conv.weight` shape
   `(39, 500, 4)`), likewise provenance. A decisive held-out row requires construction-only
   fingerprint recomputation or ESPUM retraining at a newly measured operating point. The frozen
   encoder normalization, PCA and K-means were fitted only on the 2,849 dedicated train utterances
   and pooling is per utterance, so those transforms did not see evaluation audio; the historical
   `UnitWordStreamJob.eIxgmMh99RSE` DID learn its proxy-silence mask from all 8,416 and so is valid
   for fixture reproduction only.

6. **Construction-only topology read (H1).** Frozen 6,414-utterance update partition of the
   8,416-utterance seed bed; each route's duration fitted from unpaired complete text and update
   audio, both shapes read on those masked update sequences. The execution snapshot archived the
   four imported source modules before computation, SHA-256
   `b939c19d669b1b5c585915cb7a634196d31b64f38113698cabab35a1503832d9`.

   | route | retained units | fitted mean duration | one-state ratios (plug-in, MM) | two-state ratios (plug-in, MM) |
   |---|---:|---:|---:|---:|
   | `seg12.5` / phones | 397 | 1.308221 | 3.2424, 3.1845 | 1.8525, 1.8194 |
   | raw / characters | 395 | 2.601966 | 2.6394, 2.6322 | 1.8086, 1.8037 |

7. **Corrected H2/H3 phone calibration path.** H2 keeps a zero-probability duration self-loop
   impossible and treats the normalized, once-floored `B(unit | phone)` as the canonical scoring,
   decoding, perturbation and repair input (finite positive rows summing to one at zero relative
   tolerance). Deleted silence is ONE shared duration boundary for scoring, decoding and repair
   forward-backward, while phone-LM history continues across the gap; that boundary vector is
   required and propagated by the common engine (it resolved a material mismatch at 53,498 gaps
   affecting 97.71% of update utterances). The wired start is 39 phones by 500 units with exact
   H1/H3 provenance and an eight-alternative output-only cap. H3 reconstructs `seg12.5` tokens as
   maximal runs on the original frame raster, removes frozen-mask silence runs as chunk boundaries
   and pools ESPUM logits over each run's complete frame span: on the real 8,416-utterance seed bed,
   exactly 715,099 retained runs in 72,842 chunks. Calibration fits fingerprint, random-map,
   pseudo-pair and ESPUM on the H1 update role; ESPUM reads the disjoint selection role label-free.
   Roles are the exact 6,414 / 890 / 7,304 / 1,112 partition; the final refit does not hash
   selection IDs. H1 is ACCEPTED at that partition and at `p=0.23560298`; do not rerun it.

8. **H4 controlled calibration and repair production stage.** Inputs frozen before any selection or
   evaluation label is read: accepted H1 roles and silence mask, the exact `seg12.5` frame raster,
   every line of `T_phi`, pinned dev-clean/dev-other MFA parquet snapshots. The positive-reference
   channel is fitted only on the 3,565 labelled dev utterances INSIDE the 6,414-utterance update
   role. The fixed library is the reference, 50 retain/redraw maps (ten `q` levels by five draws),
   ten soft-damage rows and 20 marginal-random maps. Update-only two-state repair is wired at counts
   0, 1, 2 and 4 for all 81 controlled starts plus the four accepted H3 calibration starts. Completed
   2026-08-20; `H4CalibrationPreparationJob.DPv4aIqwPEzM` produced the start bundle, phone LM and
   donor table, and all 85 trajectories carry finite normalized two-state tables at exactly counts
   0/1/2/4. Completed infrastructure, not an H4 gate result.

9. **Corrected H4 recovery and decoder-resource preflight.** A role firewall materializes only the
   3,565 labelled dev utterances inside the update role. 71 non-soft controlled starts regenerate
   their `Q(phone | unit)` and all reproduce their retained count-0 `B` exactly; the four persisted
   H3 `Q`/`B` pairs reproduce canonical `B` from persisted `Q`; so exactly 75 trajectories are reused
   and the ten B-space soft starts are replaced by the registered Q-space mixtures and rerun. Two
   interface defects found before any selection read and corrected: the donor law did not guarantee
   `C_d <= N <= T_d`, and the controlled bundle omitted the original count-0 `Q` (which cannot be
   recovered provenance-safely from normalized `B`); exact `-inf` donor scores must not be clipped.
   The production donor law yields 513/890 eligible sources (235/432 clean, 278/458 other) with 377
   explicit `no_swap` — construction-time facts about the frozen table, not content. Measured
   contracts (both pass the 1.5x rule): update 23,768.19 s and 1.043 GiB on its heaviest 19,515-unit
   shard -> 10 h / 2 GiB; selection 3,069.04 s and 0.908 GiB on its heaviest 2,466-unit shard ->
   2 h / 2 GiB. Unequal donor lengths are material: of 5,130 assignment rows, 4,803 (93.6%) have a
   longer donor, median `T_donor/T_source` 1.159, maximum 5.25.

10. **H4 label-free global-beam boundary.** `legacy-2g` continuation on the frozen three-table update
    inventory; canonical heaviest shard `update[2::32]` (201 IDs, 19,515 retained units), 144 cells =
    three representatives x 12 `(lm scale, insertion penalty)` settings x beams 64/128/256/512.
    Representatives are the frozen resource-contract triple (340 tables deduplicated to 316 hashes,
    entropy-sorted, indices 0/157/315: `controlled/map_q04_draw03` r0, `controlled/map_q06_draw01`
    r2, `controlled/soft_q00` r0). Completed 2026-08-21: NO setting has a stable beam. The best
    worst-representative one-best agreement is 0.7313432835820896 (lambda 2, beta -2, 256->512)
    against the required 0.999, and the smallest worst-representative score change is
    0.005448072638504445 nats per retained unit against the required strict 1e-4. Reading note: with
    201 shard utterances the 99.9% clause is effectively 201/201 unchanged one-bests. Consequence:
    the baseline H4 surface retains the LOCAL decoder only.

11. **H4 pre-label selection surfaces (1g.2).** Local-only surface: 340 local decodes (85 starts x
    counts 0/1/2/4), 3,400 fixed-text donor scores (10 frozen assignments per tuple), one selection
    surface and one provisional-maxima read over the 890 selection utterances, reading no label. The
    statistic contributes from the 513 donor-eligible sources (235 dev-clean, 278 dev-other, the same
    set in every tuple and assignment); the 377 `no_swap` sources are absent by construction and the
    fixed weights are the split sizes of the full 890, not of the 513. Both artifacts carry
    `contains_labels: false` and `frozen_pre_label: true`; recorded `code_identity` sha256 of
    `h4_selector_jobs.py` (`517401b9...`) matches the committed file byte for byte. Completed
    2026-08-21: all 85 starts produced a finite provisional maximum and all 85 winners are
    `decoder.kind = "local"` and `eligible = true`.

    | cross-start row | provisional maximum | winning repair count | winning assignment |
    |---|---:|---:|---:|
    | `real/random_map_seed1000` | 10.7753 | 0 | 0 |
    | `real/fingerprint` | 10.1520 | 0 | 0 |
    | `controlled/reference` | 5.8265 | 4 | 6 |
    | `real/espum_seed0_update30000` | 4.2613 | 4 | 0 |
    | `real/pseudo_pair_seed0` | 0.1437 | 4 | 5 |

    Two internal consistency reads, neither a gate: `controlled/random_map_seed1000` returns 10.7753,
    identical to its `real/` twin at every printed digit; and all 324 controlled within-sequence
    choices plus all 4,080 `global_beam_ineligible` entries are ineligible, so no sequence-decoder
    score exists anywhere. Cross-start ranking: the random-map null is 9th of 85 and the reference
    73rd, the five registered rows at ranks 9/72/73/76/84; winning repair counts over all 85 starts
    are {0: 72, 4: 13} (counts 1 and 2 never win).

    CHANNEL DEGENERACIES, construction rather than copy error: `Q_LEVELS` ends at 1.0
    (`h4_jobs.py:34`), so level 09 is each ladder's undamaged endpoint. In the map ladder `keep_count`
    is then the whole live set and `assignment[keep] = reference_map[keep]` discards the draw, so
    `controlled/map_q09_draw00..04` are ONE channel returning 10.3872214431 at all five draws; in the
    soft ladder `canonical_soft_q` early-returns `reference.copy()` at `q_level == 1.0`
    (`h4_production.py:223-224`), so `controlled/soft_q09` IS the reference and returns
    5.8264784397. The two q09 endpoints do not coincide with each other. 85 starts carry 79 distinct
    channels through THREE duplicate groups (the five `map_q09` draws; `soft_q09` with
    `controlled/reference`; and, cross-namespace, `controlled/random_map_seed1000` with
    `real/random_map_seed1000`). EFFECTIVE INDEPENDENT CONTROLS ARE 76 OF 81, which any clustered
    interval or null spread over the controls must respect. (Corrected 2026-08-22: the first version
    named two groups and so could not reach 79 from 85.)

12. **H4 controlled validation read (1g.2 label boundary).** Labels opened by the planner 2026-08-22.
    Two jobs only: `H4ProvisionalWinnerAuditJob.kBCapQOpk1Hj` emits the audited maxima with EMPTY
    audit mappings (the local-winner exemption ASSERTED — the job errors on a sequence winner lacking
    an audit), and `H4ControlledValidationJob.Otv6GBVY8ZUj` is the only label reader in Phase 1g.
    Nothing is decoded, scored, refit or reranked. The four real H3 rows are refused at construction,
    so their errors cannot be read even by mistake. Registered statistics (transcribed into the
    module docstring, so the reporting rule lives with the producing code): reference vs the
    strongest null under a simultaneous 95 percent interval formed by bootstrapping
    `Sel(reference) - max_over_controls Sel`; Spearman(`Sel`, -error) globally and inside the
    predeclared starting-PER band 0.80-0.93; the reference-start paired count safety read
    `PER(r)-PER(0)` for r in 1/2/4 against the 0.05 margin; and the within-trajectory
    rank/regret/count-0 bounds. Before any interval is taken the reader rebuilds every controlled
    tuple's frozen `Sel` from that tuple's own stored per-utterance deltas and refuses to continue
    unless it matches to 1e-9. `H4SelectorFreezeJob` is deliberately absent: it raises unless the
    selector verdict is PASS, so building it before the outcome was known would place a job in the
    graph whose result is unknown.

13. **User-funded descriptive PER read over the four real H3 seeds (1g.2, gate already closed).**
    One CPU job, `H4RealSeedPerJob.vu6Dp6HkJ2pH`: plain per-split PER on the 890 selection-role
    utterances (432 dev-clean, 458 dev-other) for the four real rows at counts 0/1/2/4, from the
    frozen surface's EXISTING decode artifacts against `GoldPhonesJob.ZGSp0hxyd2YP`. No decode, no
    rescore, no rerank; the label firewall of approach 12 was not touched, and the two jobs refuse
    each other's inputs. The 1,112-ID held-out evaluation stays sealed.

14. **1g.2a: matched higher-order fitting LMs and the exact context-state repair engine.**
    User-mandated, registered out-of-trigger (`h4_lm_trigger` is False from verdict 18); funded scope
    is items 1-4 only. The matched family is unpruned modified Kneser-Ney at orders 2/3/4 from the
    SAME pinned complete `T_phi`, same 39-phone inventory and BOS/EOS convention, canonical
    `phoneme_ngram_lm` settings (`interpolate_unigrams=True`, `pruning=None`, discount fallback
    0.5/1.0/1.5). Matched 2 vs 3 vs 4 is the ORDER contrast; `legacy-2g` vs matched-2 is the separate
    SMOOTHING bridge. The compiler evaluates the full backoff recursion at every history, drops
    `<unk>`/`<s>` as successors, renormalizes over 39 phones plus EOS and RECORDS the removed mass
    per history. `H4LegacyLmJob` rebuilds the never-persisted baseline bigram through the same code
    path and refuses unless the rebuilt phone-sequence hash and line count match what the accepted H1
    recorded — reproduced exactly over 39,630,169 phone lines.

    | artifact | deviating parameters | reachable histories / arcs; max renormalized mass | job hash |
    | --- | --- | --- | --- |
    | `legacy-2g` | add-one, order 2 (accepted baseline) | 40 / 1,600; 0 (add-one needs none) | `H4LegacyLmJob.lZI6TrYdVpev` |
    | `matched-2g` | MKN, order 2 | 40 / 1,600; 7.05e-08 | `H4MatchedLmJob.T8ImJUXHaB0l` (`KenLMplzJob.ef5FXMvv8af5`) |
    | `matched-3g` | MKN, order 3 | 1,561 / 62,440; 1.274e-04 | `H4MatchedLmJob.Jb2m4aM2fUTy` (`KenLMplzJob.tis71OtNidgL`) |
    | `matched-4g` | MKN, order 4 | 60,880 / 2,435,200; 1.274e-04 | `H4MatchedLmJob.VpVkGMMy7xKW` (`KenLMplzJob.bg0iYRzBQynx`) |

    All four normalize to machine precision (max absolute error 5.6e-16); the matched family loses at
    most 1.3e-04 at any history, so the order contrast is not an artifact of discarded mass.

    ENGINE (item 2). The accepted engine is bigram-specific and its dense transition matrix at order 4
    would be about 118 GB. The ruled form keeps the state as (duration sub-state, BOS-padded
    history), makes duration moves elementwise-diagonal and leaves one contraction over phone-exit
    arcs against the normalized per-history table; emissions stay tied by (phone, sub-state) and
    broadcast; the backward pass is shared across sub-states, which the topology proves. Reachable
    histories are exactly 1+39+39^2+39^3 = 60,880 and an order-4 E-step over the update fold costs
    about 5.6e12 operations. The trainer `h4_context_em.py` preserves the accepted routine's
    semantics (one symmetry break immediately before repair step 1, unperturbed count-0 snapshot,
    single common floor, pinned fitting LM the M-step never touches) and is written in shard form so
    shards aggregate likelihood and expected counts before ONE common M-step; shards return
    unnormalized counts and only the driver floors and normalizes.

    DATA CORRUPTION CAUGHT AND FENCED. The login-node LocalEngine ran each `KenLMplzJob` more than
    once (order 2 twice, order 3 three times) and a compile read `lm.gz` while lmplz was rewriting it.
    The order-2 compile failed loudly; the ORDER-3 compile did not — it finished with `ngram_counts`
    matching the final file and is given away only by its banked `arpa_sha256`, which no file on disk
    carries any more. All matched compiles were rerun under a hardened reader that checks the ARPA's
    own declared per-order counts, and the family shares one compiler identity. Verified afterwards:
    every banked cell's `input_content_sha256.automaton` equals the current manifest digest (legacy
    `0aa488aa`, matched-3g `bc176309`, matched-4g `f38eedfc`), so the corruption reached no banked
    number. A residue is left deliberately: `code_identity` hashes the whole module and
    `h4_lm_artifacts.py` holds both job classes, so the legacy artifact records a different module
    hash from the three matched ones; rerunning three correct artifacts to erase it would repeat the
    rerun mistake.

    ITEM 3, the measured resource gate (`H4ContextResourceGateJob.HA1vzRL7MEAz`): PASS — see
    verdict 22 and its correction for the per-shard/whole-fold pair.

    ITEM 4a, label-free likelihood half: all 20 cells (five starts x four fitting LMs) ran the
    accepted 0/1/2/4 trajectory over the whole 6,414-utterance update fold (584,424 retained audio
    units) with only the fitting LM changed. Per-audio-unit log likelihood at repair count 4:

    | start | `legacy-2g` | `matched-2g` | `matched-3g` | `matched-4g` |
    | --- | --- | --- | --- | --- |
    | `controlled/reference` | -5.2736 | -5.2736 | -5.2418 | -5.2201 |
    | `real/espum_seed0_update30000` | -5.3568 | -5.3568 | -5.3653 | -5.3821 |
    | `real/fingerprint` | -5.6262 | -5.6262 | -5.6082 | -5.6079 |
    | `real/pseudo_pair_seed0` | -5.8930 | -5.8930 | -5.8839 | -5.8822 |
    | `real/random_map_seed1000` | -5.6547 | -5.6547 | -5.6255 | -5.6410 |

    Each column is the likelihood of the same audio under a DIFFERENT fitting LM, so the columns are
    not readings on a common scale and cannot choose an order.

    ITEM 4b, descriptive PER half (60 channel adapters, 60 local decodes, one error read). The frozen
    decoder for all five starts is the LOCAL decoder — `H4ProvisionalMaximaJob.ejmy4sdTOcS3` records
    `decoder.kind == "local"` with no lambda, insertion penalty or beam on every baseline row — so no
    beam search and no decoding 4-gram enter this half. Counts 1/2/4 decode the repaired tables;
    count 0 is read from the frozen 1g.2 direct-Q decode and the reader re-hashes that column in all
    four fitting-LM positions and refuses the grid if they disagree. Bed and gold are the 890
    selection-role utterances of the 1g.2 descriptive read. Only the fitting LM changes DURING
    REPAIR: the local decoder's phone prior stays the accepted `phone_lm.npz` in all twenty columns
    (one `phone_prior_sha256`, `9b4a00f4`, across all 60 cells), which is why these cells cannot
    reuse `H4LocalDecodeJob`, whose prior file is pinned to the channel's fitting LM.

    Pooled corpus PER (total edits over total reference phones), 890 selection utterances:

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

    The `legacy-2g` and `matched-2g` columns are separate artifacts with different content hashes,
    produced from repaired tables with different array hashes, whose decoded phone sequences are
    nevertheless BYTE-IDENTICAL on all 890 utterances in all 15 repaired cells; `matched-4g` differs
    from `legacy-2g` on 664, 796 and 849 of 890 utterances at counts 1, 2 and 4 on the reference.

    ITEM 4c, own-minus-donor half: the same 60 adapters and 60 decodes fanned into
    `H4FixedTextScoreJob` at the ten frozen donor assignments (600 jobs), count 0 reusing the frozen
    1g.2 adapter and its ten finished score jobs. The statistic is NOT re-derived —
    `H4ContextOwnMinusDonorJob` calls `compute_selection_aggregate` from the 1g.2 selector module, so
    a cell here and a cell there are the same statistic by construction; all five count-0 aggregates
    occur bit-equal inside the frozen surface. All 80 cells eligible, ten assignments each at constant
    eligible counts 235/278.

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

15. **1g.9 experiment 1: locate the phone-repair collapse.** Five 1g.2a starts at counts 0 and 4
    under the accepted two-state topology at `p=0.235603` and `legacy-2g`; posterior expected
    symbol-ENTRY distribution `q_bar` and posterior expected rate by forward-backward, the same two
    statistics from the banked frozen local one-bests (no new decode), and each constraint term's
    gradient. `p_text` is the accepted calibration `phone_lm`'s `phone_prior` over the complete
    39,630,169-line unpaired phone corpus, unsmoothed; `r_target` = 0.7644 symbols per retained unit
    from the frozen H1 length-law fit (the memoryless reading `1-p` agrees to four decimals; 53,498
    update and 5,110 selection forced boundaries at deleted-silence gaps are the one term a healthy
    posterior may legitimately exceed it by). Gradients are
    `lambda_equal = ||grad L_HMM|| / ||grad L_term||` in the `B = softmax(theta)` parameterization.
    Label-free, selects nothing. `H4CollapseLocateJob.gZ9d6e3E7ZGu`, 31 minutes.

    Posterior on the 890 selection utterances, matched to the decode; residuals relative to
    `r_target`; `cl0` is the registered clause-0 pattern:

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

    Update-role posterior (the fold a constrained objective would run on), both terms' `lambda_equal`
    and the per-retained-unit log likelihood on each fold:

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

16. **1g.10: the full-model (LM-aware) sequence decode of the audited count-4 channels.** Registered
    prefix-mass beam decoder under the frozen H1 two-state law with the banked KenLM phone 4-gram
    REPLACING the fitting bigram, on the three audited count-4 channels over the 12-point grid
    (lm_scale in {0.5, 1, 2, 4} x insertion penalty in {-2, -1, 0}). Beam 512 is the decision beam
    over the full 890-utterance selection role (32 shards per cell); beam 256 exists only for the
    adjacent-pair columns and runs ONE fixed shard per cell on the planner's budget ruling (the
    heaviest selection-role shard from the measured contract's own `shard` block,
    `H4ResourceContractJob.kyMk7fwm027C` chunk_index 28, 2,466 retained units, 27 utterances, the
    same index for every cell). The global-beam eligibility flag is read for provenance and
    deliberately NOT applied. Nothing is refit or selected.

    | row | readable cells of 12 | cells clearing the babble null | best correct-phone fraction (cell) |
    |---|---|---|---|
    | `controlled/reference` (positive control) | 7 | 12 | 0.6010 (lam 0.5, beta 0) |
    | `real/espum_seed0_update30000` | 7 | 12 | 0.1907 (lam 2, beta 0) |
    | `real/pseudo_pair_seed0` (collapsed) | 1 | 12 | 0.1756 (lam 1, beta 0) |

    LM-blind local decoder on the same channels, for reference: `controlled/reference` correct-phone
    0.5832 at TV 0.0658 and length ratio 1.1205; `real/espum_seed0_update30000` 0.1472 at TV 0.0847 /
    ratio 1.2224; `real/pseudo_pair_seed0` 0.1904 at TV 0.6871 / ratio 0.4941 on 9 distinct phones.

    Beam-stability duty over all 36 cells (agreement and drift on the probe's 27 utterances; margins
    on all 890):

    | quantity | min | median | max |
    |---|---|---|---|
    | one-best agreement, beam 256 against beam 512 | 0.2222 | 0.6111 | 0.8889 |
    | median score margin, nats per retained unit | 1.210e-03 | 4.345e-03 | 1.540e-02 |
    | fraction of utterances in a cell at or below the flat threshold | 0.061 | -- | 0.466 |

    ZERO of 36 cells reaches 0.999 agreement and ZERO of 36 has a median margin at or below 1e-3.

    1g.10a (`H4CrossBeamDefectJob.2pV5rHuWJW3d`): TEST D 81 utterances across three disclosed cells
    decoded twice, ZERO violations at 1e-12 nats; TEST U all 1,944 banked winners rescored against
    their exact unpruned all-alignments totals, ZERO violations at 1e-6 nats. The exact score is an
    identity, not a second implementation (its channel part is the `marginal_path_log_score` the
    chunk jobs already bank; LM and insertion terms are alignment-independent). Exact-currency winner
    gaps: of 384 differing-winner cases exact(w512) beats exact(w256) in 352 and LOSES in 32 (8.3
    percent), median gain +0.0109 nats per retained unit, range -0.0567 to +0.1657.

    1g.10b re-ran the same 36 cells at beam 1024 through a probe class that imports the production
    decoder and first reproduces one banked production chunk exactly at beam 512 (parity cell PASS):

    | quantity | min | median | max |
    |---|---|---|---|
    | one-best agreement, beam 256 against beam 512 | 0.2222 | 0.6111 | 0.8889 |
    | one-best agreement, beam 512 against beam 1024 | 0.3704 | 0.7037 | 0.8889 |
    | score drift per retained unit, beam 256 against beam 512 | 1.567e-03 | 8.173e-03 | 1.821e-02 |
    | score drift per retained unit, beam 512 against beam 1024 | 1.061e-03 | 5.143e-03 | 2.755e-02 |

    Per row at 512 against 1024 (median over that row's 12 cells): `controlled/reference` 0.7222,
    `real/espum_seed0_update30000` 0.7407, `real/pseudo_pair_seed0` 0.6296. Cell by cell against the
    same cell's 256-vs-512 column, agreement rises in 24 of 36, falls in 10, ties in 2; drift falls
    in 25 and RISES in 11. Best cell 24 of 27; ZERO of 36 reaches the 26-of-27 bar.

    1g.10c re-decoded the positive control and the real ESPUM arm (the collapsed row refused in code)
    at four extension points (lm_scale in {1, 2} x insertion bonus beta in {+1, +2}), each paired
    against ITS OWN beta-0 production cell at the same lm_scale on the same 890 utterances; parity
    cell reproduced the banked chunk exactly. Deltas are correct-phone fraction, extension minus
    baseline:

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

    Pooled description beside the paired read, never instead of it: the control's best extension cell
    reaches PER 0.3943 (lambda 1, beta +2) against its own beta-0 cell, and the real arm stays
    between 0.8102 and 0.8417 across all four. Decoded length rises with beta in every cell (control
    51.0 -> 57.2 units at lambda 1; real arm 50.8 -> 60.2); all eight cells clear their own babble
    null and use the full 39-phone inventory. Contract-shard agreement here runs 0.593 to 0.778, so
    the 1g.10b bar keeps cross-channel comparison closed.

17. **1g.11 experiment 1: the continuous twin of the `seg12.5` observation stream.** One vector per
    TOKEN of the frozen discrete stream: the frozen state dump standardized and projected on the
    frozen PCA basis (`QuantizeStatesJob.FWpGhC941JMi`), averaged over exactly the frames of each
    run-length run of `seg12.5`, then standardized per component by a scale fitted on the dedicated
    train half alone (2,849 utterances, 448,204 segment vectors). PIPELINE CHECK (asserted): at the
    Ward segmentation the re-assigned segment means must reproduce the frozen stream bit-for-bit,
    run through `repr_pool.pool_utterance` itself. TWIN CHECK (reported): the share of TOKENS whose
    re-assigned mean still returns the frozen code.

    | bed | utterances | tokens | ward segments | frozen PCA dim | kept | pipeline check | twin check | job |
    |---|---|---|---|---|---|---|---|---|
    | seed bed, `seg12.5` | 8,416 | 919,248 | 921,432 | 96 | 96 | exact, 8,416 of 8,416 | 919,248 of 919,248 (100.0000%) | `G11ContinuousSegmentsJob.hImWJG0X4eZh` |

    2,184 ward segments (0.237%) are absorbed into tokens across 1,334 of the 8,416 utterances.
    Registered reading rule (pinned by the planner): after a passing pipeline check, a token mismatch
    outside the absorbed set is a STOP.

18. **1g.11 experiment 2: the Gaussian repair cells.** The five 1g.2a starts at counts 0 and 4 under
    the constrained update rule (shared diagonal covariance, M2 floor), EM on the accepted H1 update
    role and decode on its selection role, with the single disclosed per-row-covariance relaxation on
    `real/espum_seed0_update30000`. The table arm is NOT re-decoded: the audited 1g.2a one-bests are
    the comparator. Retained after the frozen silence mask: 584,424 of 751,195 tokens (0.7780) over
    6,414 update utterances, 60,604 of 77,566 (0.7813) over 890 selection utterances. Duration
    p 0.2356; 103 of 500 unit IDs masked. `sym/tok` is decoded symbols per RETAINED TOKEN and is NOT
    the table arm's decoded-length-versus-gold ratio.

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
    million per cell: a numerical floor, not a model term, confined to the disclosed per-row cell.

19. **1g.11 experiments 3 and 4: the observation null and the evaluation against gold.** The null is
    the same class with ONE method overridden: every retained token's vector redrawn i.i.d. with
    replacement from the pooled corpus segment-vector marginal over both folds, with token counts,
    unit IDs, retained index and duration boundaries asserted to survive, then fitted and decoded
    through the arm's own code (`G11ObservationNullJob.orOc9h6K3cuR`; 645,028 vectors redrawn;
    log-likelihood -79,970,371.2 at count 0 to -75,850,789.7 at count 4). The evaluation scores BOTH
    arms against gold on the 890 selection utterances (the table arm from its banked one-bests, never
    re-decoded) on the unit-cost Levenshtein of `h4_validation_jobs`, pooled PER, with the edit
    decomposition beside every PER and the babble null computed in the same job. `len/gold` is
    decoded symbols over GOLD phones and is NOT the 1g.10 family's ratio to `r_target`.

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

    Clause 3, the decision read — paired per-utterance correct-phone delta at count 4, candidate minus
    its own start's banked table cell, stratified within evaluation split, 10,000 resamples at seed
    42. Every interval excludes zero.

    | pair (candidate minus its own table cell) | delta | 95 pct interval | role |
    |---|---|---|---|
    | gaussian reference tied 4 | -0.0208 | [-0.0260, -0.0154] | arm |
    | gaussian espum per_row 4 | +0.0055 | [+0.0013, +0.0096] | arm |
    | gaussian espum tied 4 | +0.0098 | [+0.0058, +0.0137] | arm (the gate's selected real start) |
    | gaussian fingerprint tied 4 | +0.0273 | [+0.0226, +0.0321] | arm |
    | gaussian pseudo_pair tied 4 | -0.0746 | [-0.0799, -0.0698] | arm |
    | gaussian random_map tied 4 | +0.0251 | [+0.0202, +0.0302] | CONTENT-FREE CONTROL |
    | obsnull espum tied 4 | -0.0772 | [-0.0817, -0.0728] | CONTENT-FREE CONTROL |

    Clause 4: floor share 0.0000 in every Gaussian and null cell. The registered babble null (100
    draws from `p_text`) reproduces itself at 1,000 draws to within 6.1e-04 in the worst of the 24
    cells (`gaussian random_map tied 4`), an order of magnitude below the smallest clause-2 gap
    (0.0090), so the noisy-bar concern raised before the run does not bite. The unigram-matched null
    of the 1g.10 family is printed beside both and decides nothing.

20. **1g.12 experiment 1: measured resource read for the Gaussian arm at order 4.** The counterpart
    of the accepted TABLE-arm order-4 gate, in the same form: the same deterministic 32-way
    `ids[j::32]` sharding of the accepted H1 update role, the same probe rule (longest update
    utterance, ties to the higher ID), the same forked-child measurement, the same 1.5 multiplier and
    the same 11.5 h / 256 GiB limits. No M-step, no decode, no label. The probe population is all five
    funded starts. `G12ResourceGateJob.3h2iIpk6lpaB`; fitting LM is the matched order-4 automaton.
    Engine addition: `context_forward_backward` takes an optional per-token `(batch, time, 2, 39)`
    emission array, the order-k twin of the argument `channel_h.marginal_forward_backward` already
    carried at order 2, with exactly one of the categorical table and the per-token array allowed.
    Established before the job ran: the dense path reproduces the categorical path at orders 2, 3 and
    4 in likelihood, posteriors and aggregated counts, and at order 2 `gaussian_context_repair_curve`
    reproduces the banked 1g.11 `g11_gaussian.gaussian_repair_curve` parameter for parameter, tied
    and per-row, with and without deleted-silence boundaries.

    | cell | start | sec | RSS GiB | reached histories | reached arcs |
    |---|---|---|---|---|---|
    | probe utterance (353 retained tokens) | controlled/reference | 0.92 | 1.32 | 60,879 | 2,435,160 |
    | probe utterance | real/espum_seed0_update30000 | 0.91 | 1.32 | 60,879 | 2,435,160 |
    | probe utterance | real/fingerprint | 0.90 | 1.32 | 60,879 | 2,435,160 |
    | probe utterance | real/pseudo_pair_seed0 | 0.92 | 1.32 | 60,879 | 2,435,160 |
    | probe utterance | real/random_map_seed1000 | 0.90 | 1.32 | 60,879 | 2,435,160 |
    | heaviest chunk (19,515 retained tokens, 201 utterances) | controlled/reference | 48.79 | 1.29 | . | . |
    | heaviest chunk | real/random_map_seed1000 | 48.88 | 1.29 | . | . |

    Projected from the heaviest chunk standing in for all 32 (an upper bound): 0.4345 h per whole-fold
    E-step over the 584,424 retained update tokens, five E-steps per count-4 curve, so 4 h for one
    curve and 17 h for all five starts in one process, both at 1.5x; 4 GiB either way.

21. **1g.12 experiments 2 and 3: the Gaussian cells at order 4 and the order-2 re-fit.** Ten jobs,
    one per (start, fitting order). Five fit the matched 4-gram; five re-run 1g.11's own operating
    point (the accepted add-one bigram), because 1g.11 persisted its decoded output and statistics but
    NOT the fitted means and variances. Every bigram cell asserts itself against the banked 1g.11 cell
    BEFORE writing any artifact at the declared bar (1e-9 / exactly zero disagreements); a corner that
    does not reproduce writes nothing. The decode here is the LM-BLIND local decoder only (1g.11's own,
    unchanged), the no-LM leg of clause 3's readout contrast. Requests are read from the gate artifact.
    The criterion is a log likelihood UNDER THE CELL'S OWN FITTING LM, so a matched-4g row and an
    accepted-2g row are in different currencies and must never be subtracted; within one fitting LM
    all five starts score the same 584,424 retained update tokens under the same model class, so a
    count-0-to-count-4 GAIN may be compared across starts (verdict 58).

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

22. **1g.12 experiment 4: the exact beam-free order-k one-best readout.** The maximizing twin of the
    accepted context recursion — same `(duration sub-state, BOS-padded history)` state space, same
    sub-stochastic path law, same BOS start and EOS terminal — with the path sum replaced by a path
    maximum and backpointers kept, reading the history algebra from the engine module. The banked
    prefix-beam decoder was not reusable: 1g.10b found no affordable beam meeting the stability duty
    (0 of 36 cells) and 1g.10c closed decode-parameter probing. DISCLOSED ESTIMATOR CHANGE, carried in
    the module docstring and every artifact: this maximizes over PATHS while the banked one-best
    maximizes over LABEL SEQUENCES, so 1g.12's decoded numbers are a new currency and are never
    compared to banked 1g.10 numbers. Acceptance before any cell existed: decoded score AND decoded
    phone sequence reproduce exhaustive enumeration of every legal (phone sequence, duration path) at
    orders 2, 3 and 4, with and without deleted-silence boundaries; every utterance's best-path score
    is at most its own exact forward log-likelihood; batching changes no decode. Measured cost at real
    sizes: 6.7 to 7.3 ms per retained token, about 7 minutes per cell over the 60,604 selection-fold
    tokens, 0.18 GiB peak. Twenty cells = five starts x four corners, all decoding under the SAME
    matched 4-gram automaton (the fitting-order contrast is only a fitting-order contrast if the
    decoder is held fixed); nothing is refitted and no label is read. Clause 4 is enforced in the
    producing job: on a nonzero violation count the artifacts are written and the job then fails.
    All twenty finished with ZERO exactness violations and an identical renormalized mass of
    1.274e-04 nats per emitted phone. `sym/tok` is per RETAINED TOKEN over the same 60,604 tokens as
    approach 21; `worst slack` is the largest nats by which a best-path score fell below the cell's
    own exact forward log likelihood (a bound on the decode, not a quality statistic).

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

23. **1g.13 experiment 1: the wav2vec-U v1-equivalent stream.** The subphase exists because every 1g
    arm to date runs on this project's own `seg12.5` construction, so no 1g result separates "the
    training paradigm is the binding constraint" from "the segmentation is". One CPU job fits the v1
    clustering and the v1 PCA on the dedicated-train role alone, segments every bed utterance at its
    cluster-ID change points and writes both twins — the discrete label sequence (the run's own
    cluster ID, a 128-symbol alphabet) and the continuous one (the run's mean PCA-512 vector). The two
    twins are the same segmentation BY CONSTRUCTION.

    BED PARTITION, established before any code and asserted inside the job: the two banked
    rVAD-trimmed layer-15 dumps partition the 8,416-utterance seed bed exactly — the 2,849
    dedicated-train utterances are the train dump's own intersection with the bed
    (`W2vu2FeatureDumpJob.HyHAk3OCbruI`, 28,539 utterances, 15,427,853 retained frames, 14.71% of
    frames dropped by the trim, zero utterances dropped), and the other 5,567 (3,565 update, 890
    selection, 1,112 evaluation) are the ENTIRE valid dump (`W2vu2FeatureDumpJob.WbaqNnxXpbRK`, 5,567
    utterances, 1,612,502 frames, 14.59%). No utterance is in both and none is missing, which is both
    the bed coverage the subphase assumes and the fact that puts the only fittable role in the train
    dump. Faithfulness is tested against fairseq 0.12.2 itself: the PCA matches
    `faiss.PCAMatrix(d, dim, eigen_power=0)` component for component up to eigenvector sign, the
    assignment matches `faiss.IndexFlatL2` exactly on every frame, the segmentation matches
    `torch.unique_consecutive` and `merge_clusters.py`'s mean pooling, and the substituted full-batch
    Lloyd k-means reaches 0.9979 of `faiss.Kmeans`'s within-cluster sum of squares at the same 50
    iterations and 3 restarts (reported, not asserted).

    MEASURED, `G13StreamBuildJob.Ob8Rh8y51x9M`. `seg/s` is segments per second of RETAINED audio at
    the dump's 50 Hz frame rate.

    | role | utterances | retained frames | segments | seg/s |
    |---|---|---|---|---|
    | update | 6,414 | 2,569,600 | 1,436,262 | 27.95 |
    | selection | 890 | 266,488 | 150,079 | 28.16 |
    | evaluation (sealed) | 1,112 | 308,759 | 175,269 | 28.38 |
    | dedicated_train (fit population, inside update) | 2,849 | 1,532,345 | 848,038 | 27.67 |
    | whole bed | 8,416 | 3,144,847 | 1,761,610 | 28.01 |

    Fit: k-means inertia 1.228e+11, best of three restarts, all 128 clusters used on the fit
    population and on the bed, rarest carrying 88 segments, zero reseeded; the run used its full
    50-iteration budget without the labels going stable, exactly as v1's fixed `niter=50` does.
    PCA-512 at eigen_power 0 keeps 0.9114 of the variance; zero degenerate components. Constants
    trace to the real v1 scripts at the line (`prepare_audio.sh:60-61`, `:68`, `:73-74`;
    `wav2vec_cluster_faiss.py:50-69`, `:192-200`; `apply_pca.py:71`). FOUR DECLARED DEVIATIONS from
    v1, copied into the artifact: feature-level rather than audio-level trim; no adjacent-pair pooling
    (`prepare_audio.sh:76-77` not applied, that leg unfunded); a substituted k-means implementation;
    the shared encoder tap. Silence is EMPTY BY DECISION and the boundary artifact is all-False.

24. **1g.13 experiment 2: H1's route read on the v1-equivalent stream, and the VAD-mask firewall.**
    Two CPU jobs; neither is a result, together they are the precondition for experiment 3. The route
    job emits an H1-SHAPED artifact, carrying the partition and text-side digests VERBATIM from the
    accepted H1 and PROVING the copy (every role hash re-derived by the accepted reader; `T_phi`
    re-read to reproduce the accepted line count and sequence digest — which is what makes "the
    matched 4-gram transports untouched" a checked claim). Its key is deliberately NOT
    `seg12.5/phones` and that key is refused outright, so a consumer reaching for the accepted key
    gets a KeyError instead of the wrong stream's duration. The silence mask is empty BY DECISION,
    with the reason in the artifact (the stream is already rVAD-trimmed, so H1's edge-enrichment split
    would select real speech units, and it is already run-collapsed, so the width statistic that split
    consumes is one for every token). The topology verdict is REPORTED, not asserted.

    The firewall exists because the banked dumps store no VAD mask, only a kept frame COUNT per
    utterance, while 1g.13's stream lives on the trimmed raster and gold alignments are rasterized on
    the untrimmed one. It recomputes the trim mask deterministically and proves it the only way
    available — recomputed kept count equal to the declared length for all 5,567 dump utterances —
    with the raster convention re-derived from the dump worker
    (`min(encoder feature-extractor length, subsampled kaldi MFCC length)`, then the rVADfast 0.4
    two-subframe majority mask truncated to it and tail-padded as silence,
    `unsupervised_asr/w2vu2/dump_w2vu2_data.py:175-200`). Gold leaves ROLE-SEPARATED in three files,
    so a fitting job naming the update file cannot reach the other two. Checked before either job was
    written: twelve real dev utterances put through this derivation reproduce their banked dump
    lengths exactly, encoder and MFCC lengths equal in all twelve.

    | run | p | mean duration | lag-1 MI (MM) | one-state ratio | two-state ratio | verdicts | job |
    |---|---|---|---|---|---|---|---|
    | v1-equivalent stream, empty silence mask | 0.68898090 | 3.2152 | 2.2498 | 1.199 | 0.906 | one ADMISSIBLE, two ADMISSIBLE | `hStPuE1UqLK6` |
    | seg12.5, accepted H1 (contrast, not a run here) | 0.23560298 | 1.3082 | 2.2315 | 3.185 | 1.819 | one REFUTED, two ADMISSIBLE | `Phase1gH1Job.HbxKiuBTJ8aN` |

    | run | headline | job |
    |---|---|---|
    | VAD-mask firewall | 5,567 utterances, 0 kept-count disagreements; 1,888,037 untrimmed frames to 1,612,502 kept (0.1459 trimmed, reproducing the dump's own recorded `vad_dropped_frac`); roles 3,565 / 890 / 1,112 | `Usfy2NF0LiSQ` |

25. **1g.13 experiment 3: the five registered start protocols re-derived on the v1-equivalent
    stream.** Four of five are the ACCEPTED job classes handed the new unit stream and the new
    H1-shaped route artifact and nothing else — seeds, Sinkhorn regularization, pseudo-pair length
    window, espum schedule, its label-free pick rule and the fitting text all transport verbatim. Only
    the controlled reference needed new code (it is produced inside `H4CalibrationPreparationJob`,
    whose recovery path cannot be re-run, and it counts gold on the UNTRIMMED raster while this stream
    segments the TRIMMED one). `num_units=128` is passed explicitly to every espum run: the module
    default is a hard-coded 500 and would silently build a generator over 372 unobserved units.
    Nothing here is a result about the channel. Every start is a row-stochastic 39 x 128 emission
    table with all entries strictly positive and largest row-sum deviation 8.9e-15; the smallest mean
    total variation between any two starts is 0.43 (espum against pseudo-pair) and the largest 0.97.
    `mean emission entropy` is descriptive shape in nats over each start's OWN alphabet; `normalised`
    divides by the log of that alphabet size only so the two streams' columns can sit side by side —
    it is not a comparison currency.

    | start | protocol, transported verbatim | peak RSS / declared (GiB) | wall clock | mean emission entropy, nats (normalised) | accepted seg12.5 counterpart, normalised | job |
    |---|---|---|---|---|---|---|
    | fingerprint | fixed-reg deterministic, reg 0.1, 6 position bins, hard argmax | 131.9 / 192 | 18 min | 1.0393 (0.2142) | 0.3434 | `lR5Q4q1xRtqV` |
    | random-map seed 1000 | canonical marginal-random | 131.7 / 192 | 13 min | 1.4849 (0.3060) | 0.3413 | `m4sNBqlCwK2Z` |
    | pseudo-pair seed 0 | length-matched proportional, window 16, text reuse | 132.2 / 192 | 14 min | 4.6476 (0.9579) | 0.9356 | `fGmIiECLQ2XW` |
    | controlled reference | gold counts on the trimmed raster, H3's emission floor | 2.2 / 32 | 13 min | 2.9038 (0.5985) | 0.5674 | `kG9pmxczOVgF` |
    | espum, projection of the picked checkpoint | full loss, label-free pick | 120.1 / 192 (per training) | 52 min per seed | 3.9151 (0.8069) | 0.6822 | `2EB1uTDlskOy` |

    The espum pick rule is label-free and unchanged: weighted phone-LM perplexity on the 890-utterance
    selection role (ordinary perplexity divided by squared emitted-inventory coverage, lower better),
    evaluated every 2,000 updates over 40,000.

    | run | loss | seed | selected update | weighted phone-LM perplexity | phone inventory covered | emitted tokens on the 890 | job |
    |---|---|---|---|---|---|---|---|
    | full seed 0, PICKED | full | 0 | 24,000 | 33.4666 | 39 of 39 | 146,029 | `oAOLIZZHVaVz` |
    | full seed 1 | full | 1 | 40,000 | 34.2041 | 39 of 39 | 146,650 | `18iF7DTcCNyF` |
    | full seed 2 | full | 2 | 32,000 | 33.8412 | 39 of 39 | 146,610 | `E9fojuqhcBDZ` |
    | bigram-only control | bigram_only | 0 | 14,000 | 64.1514 | 36 of 39 | 81,724 | `q59UQC0AW5Oc` |
    | accepted seg12.5 full seed 0 (contrast, not a run here) | full | 0 | 30,000 | 32.5352 | 39 of 39 | 59,751 | `97FwGhhItdpO` |
    | accepted seg12.5 bigram-only control (contrast, not a run here) | bigram_only | 0 | 40,000 | 55.4678 | 38 of 39 | 58,836 | `h4LngSZ4YvKL` |

    The controlled reference is the only new code, so its checks are listed. It reads gold from the
    firewall's UPDATE file only; the selection and evaluation files are never opened by it.

    | check | result |
    |---|---|
    | every labelled utterance lies inside the update role | 3,565 of 3,565 |
    | the re-derived frame assignment collapses to the banked segment sequence, position for position | 3,565 of 3,565 |
    | the route declares no silence unit | 0 |
    | T_phi reproduces the route artifact's line count | pass |
    | labelled trimmed frames, of which emitting | 1,037,255, of which 966,669 |
    | units with at least one labelled frame | 127 of 128; the remaining unit backs off to the T_phi phone prior |

26. **1g.13 experiment 4: the measured order-4 resource read on the v1-equivalent stream.** It runs
    1g.12 experiment 1's OWN job class rather than a copy, so the two contracts are comparable by
    construction: same 32-way sharding, same probe rule, same forked-child measurement, same 1.5
    multiplier, same 11.5 h / 256 GiB limits, same `size_request` arithmetic. Two inputs are adapted
    inside that class where the stream genuinely differs — the codebook, fitted on RAW pre-PCA
    features following v1, is carried into the observation space by the stream's own PCA (exact,
    because that map is affine); and the plain [phone, unit] starts are lifted into the two duration
    sub-states by duplicating each phone's row.

    THE FIRST RUN FOUND A BUG (a defect of experimental validity, not a reporting quirk).
    `G12ResourceGateJob.4iWPXMh9yoJN` sized PASS but reported ZERO reached histories for four of five
    starts. The backward recursion in `h4_context_engine.py` was rescaled by ALPHA's per-frame
    normalizer, which bounds `alpha * beta` inside float64 only while forward and backward masses stay
    near each other; a concentrated start over 512-dimensional observations breaks that, `raw / scale`
    overflows to `+inf`, and `alpha * beta` evaluates `inf * 0` to NAN. Three guards failed to stop
    it: the log-likelihood is read off the alpha recursion alone so it stayed finite and plausible;
    `mstep_from_statistics` guards `weight <= 0.0` and a NAN is not `<= 0`; and the gate's own
    `reached = occupancy > 0.0` counts NAN as unreached and prints a believable zero. Because
    `gaussian_context_pass` calls `context_forward_backward` with exactly the arguments the occupancy
    probe uses, the SAME gamma feeds the E-step's sufficient statistics — experiment 5 would have
    fitted NAN means and variances for four of five starts with every health indicator reading clean.
    Beta is now rescaled by its own per-frame maximum, which cancels exactly because `joint` is
    renormalized over its own frame before anything reads it (a change of normalizer, not of the
    quantity); two guards were added (the E-step raises on a non-finite sufficient statistic, the gate
    raises on a non-finite occupancy). The first run's numbers are SUPERSEDED, not merged: its
    occupancy column is unquotable and its timing measured code that no longer exists; `4iWPXMh9yoJN`
    is an orphan by hash and is superseded evidence, not debris. The registered anchor for the fix is
    `G12EngineEquivalenceJob.sWWDLbPKglfP` (verdict 68).

    | run | engine | chunk (s) | h per E-step | one curve | all 5 in one process | request (GiB) | reached histories | verdict |
    |---|---|---|---|---|---|---|---|---|
    | 1g.13, re-measured `cQ3wfqsTamPP` | fixed | 128.13 | 1.1389 | 9 h | 43 h | 30 | 59,204 - 60,879 | PASS one curve; one job per start |
    | 1g.13, first run `4iWPXMh9yoJN` (SUPERSEDED) | pre-fix | 124.81 | 1.1094 | 9 h | 42 h | 30 | 0 for four of five, NAN | occupancy unquotable |
    | 1g.12, accepted `3h2iIpk6lpaB` (contrast, not a run here) | pre-fix, unaffected | 48.88 | 0.4345 | 4 h | 17 h | 4 | 60,879 for all five | PASS one curve; one job per start |

    | measurement | 1g.13 (re-measured) | 1g.12 (accepted) |
    |---|---|---|
    | update fold, retained tokens | 1,436,262 | 584,424 |
    | observation dimension | 512 | 96 |
    | probe utterance | `422-122949-0013`, 893 tokens | `2902-9006-0015`, 353 tokens |
    | heaviest chunk (index 2 of 32) | 48,417 tokens, 201 utterances | 19,515 tokens |
    | engine peak, forked and isolated | 10.27 GiB | 1.32 GiB |
    | host peak, loading the twin and building the view | 9.14 GiB | 0.77 GiB |
    | time headroom on one curve | 2.5 h | 7.5 h |

27. **1g.12 experiment 5: the continuous observation null, fitted AND decoded as the arm is.** The
    null is the Gaussian repair job with ONE method overridden, the observation seam, so the fitting
    automaton, retained-token view, census, start means, constrained update, variance floor, local
    decoder and artifact schema are all the arm's, reached through the same calls. What experiment 5
    adds is the DECODE half of "null": the job persists its redrawn SELECTION-fold vectors in the
    segment twin's own shape (retained positions carry their redrawn vector, dropped positions NaN so
    a reader that ever took one gets a NaN and not a plausible number), records the draw seed, that
    file's content hash and a hash of the whole update+selection draw, and the readout cell is handed
    that file as `segments_pkl`. `g12_readout_jobs.py` is byte-identical, so the twenty banked cells
    are not re-certified. The null is fitted at BOTH fitting orders rather than only the registered
    matched 4-gram — one cheap job beyond the registration's letter, because clause 3's contrast (b)
    is a FITTING-ORDER contrast a single-order null cannot enter; the whole-draw hash is what makes
    that pair a fitting-order contrast rather than two beds, and experiment 6 refuses to run if the
    two cells disagree on it (RATIFIED by the planner).

    | cell | job | redrawn tokens | draw sha | selection artifact sha | exact readout | violations |
    |---|---|---|---|---|---|---|
    | accepted-2g | `G12ObservationNullJob.tDiHo9tPpn5Z` | 645,028 | 98a1cc7e | 38d68786 | `G12ExactReadoutJob.ij9vB58klqDW` | 0 |
    | matched-4g | `G12ObservationNullJob.QfLZEyTjxE6o` | 645,028 | 98a1cc7e | 38d68786 | `G12ExactReadoutJob.axh5u2jyP9Va` | 0 |

    The seam is confirmed end to end on the finished pair: the null's exact order-4 decode differs
    from the ARM's exact order-4 decode of the same start and fitting order
    (`G12ExactReadoutJob.CYMKVwReFIsN`) on ALL 890 utterances, 54,883 decoded symbols against 47,695.
    The two orders are ONE bed: the count-0 decodes agree exactly at 47,628 symbols (count 0 is the
    un-refitted start) and the count-4 decodes separate, 54,883 against 54,434. Verified beyond the
    implementer's checks: all 60,604 retained redrawn vectors are exact members of the real
    segment-vector pool and NONE equals the real vector at its own position, and `mu_0`/`var_0` are
    bit-identical between null and arm — so the count-0 row is a PURE observation swap while count 4
    mixes the swap with a refit. Two different comparisons, not a weak and a strong version of one.

28. **1g.13 experiment 5 step (b): the TABLE arm's own measured order-4 gate.** The table path was
    ported to the second stream with four widenings, each hash-excluded at its default: `route`
    (replacing the hardcoded `h1["routes"]["seg12.5/phones"]` and the default-route `_route_mask(h1)`
    in both modules); the topology assertion, now the shared per-route registry; a route-keyed START
    POPULATION (1g.13's start names are `controlled_reference`, `espum`, `fingerprint`,
    `pseudo_pair_seed0`, `random_map_seed1000`, NOT the seg12.5 names — mapping one onto the other in
    a config would have attributed a 1g.13 cell to a name meaning something else, e.g.
    `real/espum_seed0_update30000` names a seg12.5 update step while the 1g.13 espum pick is at update
    24,000); and the fitting-LM scope, now including `accepted-2g` built from the calibration artifact
    exactly as the Gaussian arm's own accepted-bigram cell builds it. A cross-stream start name is
    refused in either direction.

    STREAM-IDENTITY BINDING, corrected from the artifacts rather than assumed: the three registered
    start protocols write three manifest schemas that DO share `schema`, `phase`, `route` and
    `input_content_sha256`; what no single field spans is the UPDATE-ROLE BINDING
    (`H3InitializerJob` records it both ways, the espum projection only as `fit_ids_hash`, the
    controlled reference only as `h1_hashes["update"]`), so on a non-accepted route the binding is
    asserted through whichever field the manifest carries, beside the route key. The accepted-H1
    content digest is missing on ONE schema (the controlled reference), so four of five starts carry
    it; where present it is ASSERTED (a mismatch had been passing as a False flag) and all four were
    verified on disk; where absent the artifact says so. The accepted route keeps exactly its three
    original checks.

    `H4ContextResourceGateJob.8M4rSjaBlikH` reads PASS, and the headline is the opposite of the prior
    that made this gate look like a formality. The two hour figures are NOT comparable without the
    E-step column: the table curve records the criterion at count 0, AGAIN immediately after the
    symmetry-break perturbation before step 1, then at steps 1 to 4 (six E-steps); the Gaussian curve
    has no symmetry-break pass and evaluates at counts 0 to 4 (five).

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

    The v1 route's topology read fired as the amended guard requires: two-state ADMISSIBLE ASSERTED
    (ratio 0.9057 against an allowance of 2.4841) and one-state ADMISSIBLE REPORTED (1.1992 against
    1.8761), the reported class travelling into the artifact rather than deciding anything.

29. **1g.13 experiment 5 step (c): the four-corner factorial.** The Gaussian repair cell takes the
    SAME two adaptations its sizing gate already carried and nothing else (codebook carried into the
    observation space by the stream's own PCA; start read under its own key and lifted across the two
    duration sub-states), both hash-excluded at their 1g.12 defaults. The projection was MOVED to a
    module-level function rather than copied, so the gate that SIZES a cell and the cell it sizes read
    the codebook through the same call — a gate measuring one projection while the cell fits another
    would be a measurement of a job that never ran. ALL FOUR CORNERS ARE FITTED here (a table fitted
    on seg12.5 says nothing about this stream), which is the one structural difference from 1g.12
    experiment 4, where the two table corners were decode-only. Each arm's request is read from ITS
    OWN gate artifact and the build asserts both verdicts PASS and that the two gates measured the
    SAME update fold by their own recorded fold hash (`update_ids_hash 2d005933`,
    `accepted_h1_sha256 0ee96f33`, 6,414 update ids in both) — a factorial whose two arms fit
    different data is not a factorial.

    | pilot cell | job | request | source of the request |
    |---|---|---|---|
    | espum, gaussian, matched-4g | `G12GaussianContextRepairJob.mrmyPW7K6BJI` | 9 h, 30 GiB | `G12ResourceGateJob.cQ3wfqsTamPP` |
    | espum, table, matched-4g | `H4ContextRepairJob.ZOyDz3Lr5gvi` | 10 h, 3 GiB | `H4ContextResourceGateJob.8M4rSjaBlikH` |

    NO-LM LEG OF EVERY CELL. A GAUSSIAN corner's leg already had a named producer —
    `G12GaussianContextRepairJob.out_hypotheses`, the fitting cell's own selection-role local decode
    at every count, the same edge 1g.12's reader consumes — so it was unregistered, not unbuilt; the
    TABLE corners' leg (channel adapter plus local decoder) needed the same route widening. The table
    leg sits at the paired count, mirroring 1g.12, and the count-0 guard stands on purpose: at count 0
    the repaired table is the start duplicated across the two sub-states, not an occupancy-weighted
    channel, which is why the accepted method decodes count 0 as a direct `Q`. One substantive fix
    came out of the port: the channel adapter stamped its fitting-LM digest from the automaton alone,
    so an accepted-bigram cell (built from the calibration artifact, no automaton) would have carried
    a null there and looked like any other cell to anything comparing that field; the digest now comes
    from whichever form the fitting LM has, with the kind recorded beside it, and a manifest carrying
    neither is refused.

    E-STEP PARALLELIZATION (USER direction 2026-08-24 21:10), recorded because it changed what the
    clamp risk was. Every cell reserved four cores and used ONE at 100.5% measured; the E-step is a
    map-reduce over independent sub-batches in both arms and now forks workers, and the result is
    BIT-IDENTICAL to the single-core path (verified serial-vs-parallel at one and several workers, in
    both arms, at order 2 AND against the real banked matched 4-gram automaton
    `H4MatchedLmJob.VpVkGMMy7xKW`, sha256-identical raw arrays) — which is what lets these cells
    inherit the accepted computation's verification. Bit-identity is a statement about ORDER: both
    paths use `Pool.imap`, which yields in input order, and accumulate exactly the sequence of
    additions the loops they replace performed; the table arm's parallelism had to go INSIDE the shard
    because the repair job packs the whole update fold as one shard and splitting it would regroup the
    additions. The time request is left exactly as each gate sized it (9 h Gaussian, 10 h table), so
    the whole benefit lands as clamp headroom; the WIDTH is derived from the 256 GiB ceiling by
    charging each gate's whole-pass figure once per worker (7 workers / 234 GiB Gaussian, 16 / 56 GiB
    table). Only the ten order-4 cells were relaunched — the ten accepted-bigram cells had already
    finished in about thirteen minutes and were left alone — and the relaunched cells came back at
    their original hashes.

    | arm | gate projection (sequential) | request | workers | MEASURED | speed-up |
    |---|---|---|---|---|---|
    | table, order 4 | 6.6 h | 10 h | 16 | about 26 min | about 15x |
    | Gaussian, order 4 | 5.7 h | 9 h | 7 | about 45 min | about 7.6x |
    | accepted bigram, both arms | - | - | (serial) | about 13 min | - |

    The matched-4-gram observation null took 51 min. Cell artifacts read healthy: `espum|matched-4g`
    has floor share 0.0000, 38 distinct symbols at count 0 and 39 at count 4, and its likelihood rises
    from -1.0291e9 to -1.0098e9; a table order-4 cell's per-audio-unit log likelihood rises from
    -7.0748 at count 0 to -4.5685 at count 1 (`g2kLTuhpq0ps`). The registered fallback if a table cell
    had hit the clamp was the sharded shape the table gate separately passed at 1 h per shard with
    10.5 h of headroom — never a bigger request, since the queue caps at 11.5 h; it was not built and
    the measured speed-up retired the risk.

    CONTENT-FREE CONTROLS ON THIS BED (experiment 6), built BEFORE the evaluation that consumes them —
    1g.12's lesson, where the exact readout beat the local decode in every arm cell and the
    observation null beat it by MORE, so a contrast read before its controls existed would have looked
    like a result. `G12ObservationNullJob.UM72oLRoTEle` (matched-4g) and `.sakp81hAxfzB`
    (accepted-2g), each with its exact order-4 readout. The build asserts the null fits at the arm's
    own max_batch, route, start key and start channel: a control differing from its arm in anything
    but the observations is not a control.

30. **1g.12 experiment 6: the evaluation against gold, the first phone error rate in 1g.12.**
    `G12EvaluateJob.yJgxKex9peLp`, finished 2026-08-24 21:00 after 1 h 25 min. Seventy-eight scored
    rows: four corners by five starts plus the observation null at both fitting orders, each under
    BOTH decoders (the exact order-4 readout and that cell's own LM-blind local decode) at counts 0
    and 4, on the accepted H1 selection role — 890 utterances, dev-clean 432 and dev-other 458,
    61,032 gold phones. The ten table x local x count-0 rows are structurally absent (table local
    decodes are banked at the paired count only) and are marked in `cell_provenance.local_counts`. The
    1,112-utterance evaluation role is not opened anywhere in the job; the gold INPUT file physically
    holds all 5,567 dev utterances and sealing is enforced by the job's filter, verified arithmetically
    (the scored set is bit-exactly the 890-id selection role; the evaluation role would be 71,153
    phones and appears nowhere).

    Phone error rate at repair count 4 under the exact order-4 readout, the decision column:

    | start | gaussian x accepted-2g | gaussian x matched-4g | table x accepted-2g | table x matched-4g |
    |---|---|---|---|---|
    | controlled/reference (gold-informed) | 0.4500 | 0.4241 | 0.4339 | 0.4046 |
    | espum seed 0 / update 30,000 (selected) | 0.8308 | 0.8271 | 0.8180 | 0.8165 |
    | fingerprint | 0.8323 | 0.8382 | 0.8180 | 0.8154 |
    | pseudo-pair seed 0 | 0.8605 | 0.8555 | 0.8445 | 0.8406 |
    | random map seed 1000 (content-free) | 0.8625 | 0.8629 | 0.8358 | 0.8318 |
    | observation null (espum's own acoustics destroyed) | 0.8985 | 0.8929 | - | - |

    Clause 3's three contrasts, per-utterance correct-phone fraction, candidate minus baseline,
    positive meaning better, stratified within evaluation split:

    | contrast | selected real start | observation null | random-map control |
    |---|---|---|---|
    | (a) readout, exact order-4 minus local, matched-4g | +0.0201 [0.0169, 0.0234] | +0.0328 [0.0297, 0.0360] | +0.0132 [0.0105, 0.0160] |
    | (b) fitting order, matched-4g minus accepted-2g | +0.0056 [0.0035, 0.0078] | +0.0062 [0.0041, 0.0084] | +0.0004 [-0.0016, 0.0025] |
    | (c) emission model, Gaussian minus table at order 4 | -0.0092 [-0.0130, -0.0055] | structurally absent | -0.0272 [-0.0309, -0.0234] |

    Clause 4 passes outright: variance-floor share all zero, decoder exactness violations none. Four
    planned (b) rows are NOT computed and are NAMED in the artifact rather than dropped (the
    observation null exists only for the selected real start). Contrast (c) has no observation null by
    construction, recorded as structural — the table arm observes the frozen unit IDs, which the null
    preserves.

31. **1g.13 experiment 7: the evaluation against gold on the v1-equivalent stream.**
    `G12EvaluateJob.a3419LhkI7JT`, finished 2026-08-25 01:24 after 2 h 02 min. Seventy-eight scored
    rows on the SAME accepted H1 selection role as approach 30, so the two streams are scored on the
    same utterances; the evaluation role is not opened. Clauses 1, 2 and 4 are 1g.12's verbatim
    because they are the same code. Contrast (d) is this stream's cell minus the SAME cell of the
    other stream, paired per utterance over the 890 both score — the only contrast whose baseline is
    not a cell of its own table — and it wires THREE pairs (the arm, the content-free random-map
    start and the observation null, each against its own seg12.5 counterpart), because a segmentation
    difference read without its controls is the mistake 1g.12 had just demonstrated. The pairing is
    GIVEN to the job rather than derived, since the two subphases name their starts differently
    (`espum` against `real/espum_seed0_update30000`), and the job refuses a pairing naming a cell it
    was not handed, an unpaired cross-stream cell, and a pairing with no stream label; the other
    stream's cells are loaded but kept OUT of `arms` and go through the same symbol conversion against
    the same inventory, since a segmentation contrast read in two alphabets would be a comparison of
    alphabets.

    IDENTITY NOTE THAT MUST TRAVEL WITH (d): the two streams pin DIFFERENT espum checkpoints by
    construction — seg12.5 update 30,000, this stream update 24,000, each the stream's own label-free
    pick — so (d) compares each segmentation WITH its own registered start selection. The random-map
    pair, which pins the same seed on both sides, shows the same loss.

    Phone error rate at repair count 4 under the exact order-4 readout, the decision column. A rate
    above 1.0 is possible and means insertions outnumber the reference phones they were added to:

    | start | gaussian x accepted-2g | gaussian x matched-4g | table x accepted-2g | table x matched-4g |
    |---|---|---|---|---|
    | controlled reference (gold-informed) | 0.4617 | 0.4270 | 0.3890 | 0.3324 |
    | espum seed 0 / update 24,000 (selected) | 1.1058 | 1.0922 | 0.8610 | 0.8594 |
    | fingerprint | 1.0682 | 1.0550 | 0.8489 | 0.8278 |
    | pseudo-pair seed 0 | 0.9095 | 0.9091 | 0.8738 | 0.8725 |
    | random map seed 1000 (content-free) | 1.1568 | 1.1375 | 0.8937 | 0.9010 |
    | observation null (espum's own acoustics destroyed) | 1.3840 | 1.3494 | - | - |

    | contrast | selected real start | observation null | random-map control |
    |---|---|---|---|
    | (a) readout, exact order-4 minus local, matched-4g | +0.0870 [0.0785, 0.0962] | +0.6689 [0.6579, 0.6804] | +0.1210 [0.1100, 0.1326] |
    | (b) fitting order, matched-4g minus accepted-2g | +0.0161 [0.0120, 0.0201] | +0.0369 [0.0333, 0.0404] | +0.0412 [0.0336, 0.0492] |
    | (c) emission model, Gaussian minus table at order 4 | -0.2484 [-0.2592, -0.2379] | structurally absent | -0.2309 [-0.2419, -0.2201] |
    | (d) segmentation, this stream minus seg12.5, Gaussian x matched-4g | -0.2959 [-0.3085, -0.2839] | -0.5029 [-0.5186, -0.4876] | -0.2832 [-0.2952, -0.2714] |

    A negative (d) delta means the v1-equivalent stream is the worse of the two. The arm improves on
    only 0.8% of utterances, the random-map control on 2.1%, the observation null on none. Clause 4
    passes: variance-floor share all zero over the 48 cells that have variance components, exactness
    violations none; the 30 table cells have no variance to floor and the artifact records them as
    OUTSIDE that line rather than clean within it. The same four planned (b) rows are not computed,
    for the same reason, and are named in the artifact.

    REPORTING DEFECT IN THIS RUN, fixed in code and the re-run DECLINED by the planner 2026-08-25:
    `evaluate.txt` from `a3419LhkI7JT` contains NO section (d). The contrast was computed and IS
    banked in `evaluate.json` under `clause3_contrasts` — that is where the four (d) numbers above
    come from — but the report printer never rendered it, so the one comparison 1g.13 exists for is
    invisible on the page the gate is read from. Any future run of the job prints the section; THIS
    run's text file does not.

    THE GATE IS READ AND RULED by the planner 2026-08-25 (`archive/SAE_1g_spec_legacy.md` 1g.13
    Status): clause 2 fails every real start, clause 3 is NOT POSITIVE with the comparability ruling
    firing on TWO contrasts, clause 4 passes, and (d) is negative and not content-specific. The
    registered consequence fires verbatim — the failure license extends to "v1-equivalent segmentation
    does not rescue this channel family at this operating point", jointly with 1g.11/1g.12 further
    evidence toward the training paradigm as the binding constraint, NEVER "the paradigm cannot work".
    The wav2vec-U-faithful completion is NOT triggered; its trigger was (d) POSITIVE with clean
    controls. Nothing reopens 1g.11's or 1g.12's gates.

## Verdicts

Numbers live once, in the approach entry named (A<n>); verdicts carry the conclusion, its scope and
any correction. "WRONG" items are retained retractions of earlier conclusions.

1. **A1: one segment per text symbol is rejected** on all eight dev-other cells under both
   estimators. About that channel shape, not about every Phase-1 initializer.
2. **WRONG AS AN UNSCOPED CLAIM (old A1 conclusion): "independent duration is rejected".** At the
   historical gold-derived duration point it is rejected on every pooled stream, passes for raw
   phones, is split-dependent for raw characters, and subtracting the cross-utterance floor rescues
   no pooled cell. It neither fixes nor rejects a duration fitted prospectively without labels.
3. **WRONG AS STATED (old A1 two-state conclusion): "seven dev-clean cells pass and one is
   indeterminate".** The exception rows were attached incorrectly: all eight dev-other cells pass; on
   dev-clean five pass, `seg12.5`/characters and `seg9`/phones are indeterminate, `seg9`/characters
   is rejected. Gold-duration diagnostics only.
4. **A2: the spectral anchor fails its gate.** Text controls pass near ceiling, every audio stream
   misses 0.85; the failure survives the polarity correction.
5. **WRONG / UNVERIFIED (old Conclusions 2, 23, 24): "the silence pre-check demonstrated that the
   eigenvector tracked silence, and later uncertainty tests repaired it".** No catalogued artifact
   preserves those outputs. Not load-bearing: every stream fails the saved accuracy gate independently.
6. **WRONG AS STATED (old Conclusion 4): "the audio partition carries no information about the text
   partition".** Supported only narrowly: this fixed binary partition under these metrics recovered
   no useful correspondence.
7. **A3: the hard descriptor route also fails.** Best held-out result 0.8130 (energy on `seg16`)
   against 0.85; the +0.20-over-majority half passes on dev-other but the gate is a conjunction, and
   both evaluated splits fail.
8. **WRONG AND SUPERSEDED (old descriptor population read):** the first report used all 2,703/2,864
   labelled utterances, not the registered fixed fifth; the corrected 540/572 reports give the
   0.7588 to 0.8130 range. The all-utterance artifacts are provenance only.
9. **WRONG (old Conclusion 5): "the descriptor route replaces the failed spectral route"** — a
   temporary next-step statement; both exercised 1g.4 routes subsequently failed their gates.
10. **WRONG / NOT ANSWERABLE (old Conclusion 19): "the six-factor soft product failed its
    prerequisite".** Seven alternative descriptors for one binary target were counted, not six
    independent memberships, so no experimental verdict exists.
11. **WRONG AND SUPERSEDED (old E5 endpoint and hard-stop interpretation).** Retention 1 is the
    reference and 0 the random redraw; the same utterances fit and score it and no real seed or
    control ran. Endpoints moved 0.4865 -> 0.4589 -> 0.6699 (reference) and 1.0109 -> 0.8409
    (random). Non-decisive; fires no gate.
12. **A5 establishes the seed-provenance constraint.** The original fingerprint and ESPUM artifacts
    saw evaluation audio and are transductive rows only; neither qualifies for the held-out gate nor
    inherits its original headline.
13. **A5 localizes the preprocessing correction.** Frozen encoder, PCA/K-means and per-utterance
    pooling need no refit; the proxy-silence mask does, its historical construction having seen
    evaluation audio.
14. **A6 selects the two-state phone channel.** One-state decisively rejected, two-state admissible
    under both estimators. Freeze `p=0.23560298` (mean duration 1.308221) and the two-state topology
    for the H3/H4 phone path. The raw-character row makes the same choice but does not unblock H6's
    separately gated handoff.
15. **A7 freezes ESPUM seed 0 at update 30,000.** All three full-loss generators were fitted on the
    exact 6,414-utterance update population and selected without labels on the disjoint 890: weighted
    phone-LM perplexity 32.5352 (seed 0 / 30,000), 32.5912 (seed 1 / 38,000), 33.1554 (seed 2 /
    34,000). Not an evaluation error-rate result.
16. **A10 rejects every baseline sequence-decoder setting at the label-free beam boundary.** No
    adjacent beam pair passes both clauses on all three representatives. H4 is not failed by this: the
    admissible baseline surface reduces to the local decoder.
17. **A11 persists the five pre-label provisional maxima, and every one is a local winner.** All 85
    starts carry a finite own-minus-donor maximum computed with no label read and all 85 winners are
    `decoder.kind = "local"`, so the winner-audit precondition standing in front of the controlled
    labels is discharged BY CONSTRUCTION (ratified by the planner; the at-most-320 budgeted shard
    cells were not spent). The label-free half of the pre-evaluation-ready condition also reads
    positive: nonzero repair count for two of four real starts (`espum_seed0_update30000` and
    `pseudo_pair_seed0`, both count 4). The controlled safety read is a label read, so readiness is
    not decided here.
18. **THE FROZEN OWN-MINUS-DONOR SELECTOR FAILS 1g.2: not uninformative but systematically INVERTED,
    so H4 is unresolved and no baseline maximum may freeze.** Labels opened once
    (`H4ControlledValidationJob.Otv6GBVY8ZUj`; 81 controlled arms, 76 effective channels, 10,000
    resamples at seed 20260822). Reference channel 5.826478 against the strongest null
    `controlled/random_map_seed1007` 10.807694 and the strongest control of either family
    `controlled/map_q05_draw00` 10.848025: reference minus strongest control -5.0215, one-sided 95
    percent interval [-5.071922, -4.978244]. Rank agreement runs the wrong way at every scale: global
    Spearman(`Sel`, -error) -0.7493 [-0.826461, -0.622575]; inside the predeclared 0.80-0.93
    starting-PER band (48 channels) -0.5125 [-0.702978, -0.252155]; within trajectory [-0.928046,
    -0.789989]. All FIVE negative clauses (three correlation, two comparison) take the registered
    upper-bound-at-or-below-zero reading. (Corrections 2026-08-22: the point estimate is -5.0215 not
    -5.03; "all four correlation clauses" miscounted — three correlation plus two comparison.)
    It is a property of the frozen score, not the reader: it reproduces the pre-label cross-start
    ordering banked before any label existed. Instrument check at its own operating point: on the 458
    dev-other SELECTION-role utterances the reference channel's count-0 PER is 0.4149 and
    `random_map_seed1007` 0.9094, against SAE_1f's 0.4148 and 0.8946 on the DISJOINT 572-utterance
    dev-other evaluation fifth — corroboration across different utterance sets, NOT an identity check.
    (Correction 2026-08-22: the first version called it an "essentially exact" match without naming
    either set, and the numbers came from an unregistered console command; the job now emits
    `per_by_count_and_split` and `split_sizes` and the read was rerun at the same hash, reproducing
    every banked value.)
    The two margin clauses pass and do not rescue it: mean selection regret [0.015849, 0.020977],
    selected-minus-count-0 [-0.006181, 0.004722], both inside 0.05 — what an inverted score looks like
    when counts within one channel differ little.
    CONSEQUENCE, from the pre-registered gate: `Sel` has failed, H4 has no selector, likelihood cannot
    rescue it, no contrastive update may be invented after labels are read; the 85 maxima stay frozen
    and unreranked, no `H4SelectorFreezeJob` is built, and the 7,304-ID and 4,455-ID refits and the
    1,112-ID evaluation stay closed. This closes the tested score/channel-shape/decoder/representation
    combination only — a decision not to fund this selector, not a finding that repair cannot work.
19. **The count method-level safety read PASSES.** On the reference start's local decoder,
    PER(1)-PER(0) [-0.003457, -0.000787], PER(2)-PER(0) [0.008906, 0.012588], PER(4)-PER(0)
    [0.021156, 0.025474] — all upper bounds below 0.05, so all three counts are SAFE (reference PER by
    count 0.3934 / 0.3913 / 0.4042 / 0.4168). The H4-LM trigger therefore does NOT fire. Read with
    verdict 18: repair itself is not what failed here; the score that was to choose among repairs is.
20. **The sequence family is UNRESOLVED, untested rather than failed.** No mechanically eligible
    sequence tuple exists (verdict 16) and every provisional maximum is local, so the exemption
    applies and the sequence verdict would only have bound a sequence winner.
21. **DESCRIPTIVE (selects nothing): no real seed beats a content-free control on plain PER, and the
    best number in the table belongs to a control.** Pooled PER over the 890 selection utterances at
    best count, dev-other beside it: ESPUM seed-0/update-30,000 0.8528 / 0.8624 (count 4); fingerprint
    0.8586 / 0.8691 (4); pseudo-pair 0.8096 / 0.8105 (4); random-map 0.8921 / 0.9022 (4). At count 0
    the same rows read 0.8573 / 0.8673 / 0.9136 / 0.9015 pooled. Two rows are content-free CONTROLS:
    pseudo-pair moves furthest under repair (-0.104) and ends BELOW both candidate seeds, which move
    -0.0045 (ESPUM) and -0.0087 (fingerprint). Candidate-versus-random-map margins at best count:
    0.0393 and 0.0335, near the historical 0.0365 for selected ESPUM over the stronger control.
    PROVENANCE: the SAE_1f anchors (0.8580 / 0.8809 / 0.9239 / 0.8946) were computed on the DISJOINT
    572-utterance dev-other evaluation fifth at historical transductive operating points — same
    regime, not the same measurement. No held-out number exists; nothing here changes verdict 18.
    (Correction 2026-08-22: the bed is 890, not "892".)
22. **A14: exact order-4 repair CLEARS the measured resource gate with a wide margin.** 49.12 s and
    0.67 GiB per E-step on the heaviest of 32 update chunks; order 3 costs 0.05 s / 0.17 GiB per probe
    utterance against order 4's 0.90 s / 0.67 GiB. Reachability 60,879 / 2,435,160 at order 4 and
    1,560 / 62,400 at order 3 — one below `1+39+39^2+39^3` and `1+39+39^2`, the all-BOS history being
    unreachable once the first phone is emitted. Affordability on this machine, not a claim about
    repair. CORRECTION 2026-08-22, request clause only: the "1 h" first reported was the PER-SHARD
    request while item 4 runs the whole fold in one process, so it would have asked for a
    thirty-second of what it needs and died at the wall with nothing saved (no resume). The gate now
    emits both figures and a separate `whole_fold_verdict`; on re-measurement the heaviest chunk read
    50.82 s -> 1 h per shard and 5 h whole-fold at 2 GiB, both PASS, 6.5 h headroom; memory does not
    scale with the fold. Second correction, probe timing only: the order-4 probe cells read
    0.9366-0.9499 s, and the request is sized from the chunk rerun, not the probe.
23. **A14: the smoothing bridge is empirically NULL, so an order-3 or order-4 difference is
    attributable to ORDER rather than smoothing.** Add-one `legacy-2g` and matched MKN `matched-2g`
    agree to within 4e-5 per audio unit on all five starts at all four counts (largest gap
    `real/fingerprint` count 0, -7.293028 against -7.293065 = 3.7e-5), so rounded displays can still
    differ in the fourth decimal. (Corrected 2026-08-22: "agree to four decimals" was contradicted by
    its own example.) Expected — 39 phones over 39,630,169 lines leaves no sparse bigram — but it
    matters because the matched family exists to avoid confounding order with smoothing. SCOPE:
    label-free likelihood half, update role only.
24. **A14: matched order 3 and 4 are not a usable descriptive-PER gain over the baseline bigram, and
    the smoothing bridge is null here too.** (a) `legacy-2g` and `matched-2g` decode byte-identically
    in all 15 repaired cells. (b) On the prospective reference, higher order reduces the DAMAGE repair
    does rather than producing a gain: every fitting LM is worse at counts 2 and 4 than the unrepaired
    0.3934, and order 4 shrinks that loss from +0.0234 to +0.0051 at count 4. (c) On the four real
    starts count-4 repair helps under every fitting LM (largest, pseudo-pair 0.9136 -> 0.8096), but
    order moves PER by at most 0.0062 and not consistently: at count 4 order 4 beats the bigram on
    espum (-0.0037), fingerprint (-0.0022), random_map (-0.0047) and loses on pseudo_pair (+0.0006),
    while order 3 beats order 4 on three of four. Every real start stays in 0.81-0.91, so the START
    dominates fitting order by an order of magnitude. SCOPE: descriptive, label-reading; the gate
    forbids PER and perplexity from selecting order, so no fitting LM is chosen and the coherent
    matched-4 arm is neither authorized nor closed.
25. **A14: item 4 is COMPLETE on both trajectories and its two halves DISAGREE about fitting order,
    so the arm answers no order question.** (a) The smoothing bridge is null on own-minus-donor too:
    at most 2.7e-4 (`real/fingerprint` count 1) against cell-to-cell spreads of whole units — nonzero
    where decoded sequences were byte-identical, the expected asymmetry (the score reads the channel,
    the decode only its argmax). (b) Raising the order LOWERS own-minus-donor at count 4 at every
    start: -0.0033 (pseudo-pair), -0.4542 (reference), -0.4986 (espum), -1.2733 (fingerprint),
    -1.3931 (random map) — and higher is what the frozen selector maximizes, so the label-free
    statistic calls the higher-order model WORSE by a margin far larger than the smoothing bridge.
    (c) That is the OPPOSITE direction from verdict 24: within-start rank correlation between the two
    halves over the twelve repaired cells is +0.917, -0.629, +0.822, -0.993, +0.907 and +0.112 pooled
    over all 60 — no consistent sign. (Corrected 2026-08-22: the first values +0.902, -0.636, +0.734,
    -0.979, +0.853, +0.110 rested on no artifact and used positional instead of average ranks for
    ties; the corrected values are banked by `H4ContextAgreementJob.zd6RBdYcvzti` with the convention
    in its own output, and the positional convention reproduces the originals exactly, so the
    discrepancy is fully explained.) This reproduces the 1g.2 inversion finding with fitting order as
    the moving coordinate. CONSEQUENCE: a fixed-duration diagnostic at this operating point identifies
    no better fitting order; per the gate a negative fixed-duration result cannot close the coherent
    higher-order method, so the unrun matched-4 arm is untouched.
26. **A15: the phone-repair collapse is NOT in the posterior the 1g.9 constraints would act on.** At
    count 4 all five starts satisfy both targets (posterior TV 0.0108 to 0.0736 against 0.15; rate
    residual -5.5 to 0.0 percent against 20 percent), including the one whose decode collapses; and
    `lambda_equal` 8.1e+05 to 1.5e+08 says either term would need a weight six to eight orders of
    magnitude above the likelihood's scale before the optimizer could feel it. Clause 0 FIRED;
    experiments 2-3 stay unbuilt.
27. **A15: the collapse is DECODE-RESIDENT and specific to ONE start.** At count 4 decoded TV is
    0.0402 to 0.1357 for four starts emitting 36-38 of 39 phones; `real/pseudo_pair_seed0` emits 9 at
    0.6871 with a decoded rate 50.6 percent BELOW `r_target` while the others run 12-24 percent above,
    and at count 0 emits 3 at 0.8345, 85.6 percent below. Its AH overproduction against `p_text` is
    +0.835 at count 0 and +0.415 at count 4. (CORRECTION 2026-08-22: "independently reproducing the
    1g.2 audit's +0.417" overstated it — +0.415 is an excess over `p_text` and +0.417 over the gold
    890 unigram; two references agreeing to 0.002 is corroboration, not identity.) So the "babble" is
    a property of one start under the frozen local decoder — a per-unit argmax over `Q * prior` with
    run collapse, consulting neither the fitting LM nor the duration law — not something the repair
    objective produces.
28. **A15: a near-zero posterior TV is NECESSARY but NOT SUFFICIENT, and this table holds the
    counterexample.** `real/pseudo_pair_seed0` has the LOWEST posterior TV of all ten cells (0.0006 /
    0.0108), the worst decode and the worst count-4 per-unit likelihood (-5.8930 update against
    -5.2736 to -5.6547), barely moving from count 0 (-5.9438) while fingerprint moves -7.2930 ->
    -5.6262 and random-map -6.9771 -> -5.6547. A table carrying little audio information leaves the
    posterior on the fitting LM's marginal, which IS approximately `p_text`, so the clause-0 statistic
    is satisfied most easily by the least informative channel: any clause-0 reading must carry the
    likelihood column beside the divergence.
29. **A15: decoded unigram distance to `p_text` does not discriminate, because the registered
    random-map control passes it best** (0.0291 / 0.0402 against the reference's 0.0854 / 0.0658), so
    clause 1 ADMITS cells rather than evidencing them and clause 2 does all the discriminating.
    Reported, not proposed as a change.
30. **A16: 1g.10's table is BLOCKED by its own pre-registered explanation duty — the beams disagree
    while the score margins are WIDE, the decoder-defect branch rather than verdict 28's flat-score
    branch.** Zero of 36 cells reaches 0.999 agreement (min 0.2222) and zero has a median margin at or
    below 1e-3 nats (min 1.210e-03); the job prints "DECODER DEFECT SUSPECTED -- no cell of this table
    may be read until that is explained". LICENSES only "the registered sequence decoder does not
    currently produce a readable surface on these channels"; licenses NO comparison among the three
    rows, no statement about the repair route's viability and no reading of the correct-phone,
    total-variation or babble-null columns. Two observations, not rescues: 6.1 to 46.6 percent of
    utterances WITHIN a cell sit at or below the flat threshold, so the medians summarize a mixture;
    and agreement is read on 27 utterances, where one disagreement is 3.7 points.
31. **A16: the positive control's own reading is consistent with the blocked verdict** —
    `controlled/reference` gives 7 of 12 readable cells, best correct-phone 0.6010 against 0.5832 for
    the LM-blind local decoder. A decoder-health observation only; NOT a licence to read the control's
    cells as a result.
32. **A16: 1g.10a returns DISCHARGED — the instability is pruning reshuffle in a correct scorer, not
    a decoder defect.** TEST D zero violations at 1e-12 nats over 81 utterances; TEST U zero
    violations at 1e-6 nats over all 1,944 banked winners. LICENSES the beam-512 table as DESCRIPTIVE
    with each quoted cell carrying its own 256-vs-512 agreement; does NOT license cross-channel
    comparison and does not revisit verdict 30, which stands as written for its date.
33. **A16: beam 512 is NOT converged.** Of 384 differing-winner cases exact(w512) beats exact(w256) in
    352 and LOSES in 32 (8.3 percent), median gain +0.0109 nats per retained unit, range -0.0567 to
    +0.1657 — a wider beam merely searching a superset could never lose, so this is the non-nested
    kept-set effect of ranking whole prefixes by surviving mass. Context, not a gate.
34. **A16: 1g.10b's parity cell PASSES and ZERO of 36 cells clears the 26-of-27 bar, so cross-channel
    comparison stays closed and beam escalation is NOT funded.** Best cell 24 of 27, median 19 of 27;
    median agreement moved 0.6111 to 0.7037 per doubling (about +0.09), from which reaching 0.963
    would take several further doublings at a cost that doubles each time. Declined on the
    measurement; the within-channel paired read stays the standing currency.
35. **A16: the convergence is a TENDENCY, not a per-cell fact.** Agreement rises in 24 of 36 cells,
    FALLS in 10 (to -0.1852) and ties in 2; drift falls in 25 and RISES in 11, by up to +9.333e-03
    nats per unit (2.62x its own 256-vs-512 value). CORRECTS the planner's 2026-08-23 reading of the
    same artifact, which recorded drift as "down in every cell"; the medians it quoted and its ruling
    are unaffected. An extrapolation treating each doubling as monotone per cell is unsupported.
36. **A16: the positive insertion bonus recovers phones on the positive control and does NOT on the
    real arm — the rows split by SIGN, within channel and paired.** All four `controlled/reference`
    cells positive with intervals excluding zero (+0.0222 to +0.0555, best lambda 2 / beta +2,
    improving 74.0 percent of utterances); three of four `real/espum_seed0_update30000` cells NEGATIVE
    excluding zero (-0.0090 to -0.0294), the fourth straddling (-0.0021). Quotes no comparison between
    the rows. Answers the USER's question with: on this evidence, only where there was something to
    recover.
37. **A16: the stratified resampling convention made no material difference** — across all eight cells
    the primary and sensitivity intervals agree to within 1e-4 in every bound. The ruling was still
    right (convention fixed in advance) but no conclusion rests on it.
38. **A17: the continuous twin is the frozen pipeline's own, asserted rather than argued** — at the
    Ward segmentation the re-assigned segment means reproduce the frozen stream bit-for-bit on all
    8,416 utterances through `repr_pool.pool_utterance` itself, so the emission swap is ONE change.
39. **A17: the token-level twin costs nothing, by a theorem rather than luck.** All 919,248 tokens
    re-assign to their frozen code, including the 2,184 spanning absorbed Ward segments: an absorbed
    pair shares a centroid, a nearest-centroid cell is a convex polytope, and the token mean is a
    convex combination of two points inside it. The count-identity clause is satisfied with no
    residual to disclose.
40. **A17: the registered 128-component truncation is vacuous** — the frozen basis carries 96
    components, so the primary cell and the optional full-dimension sensitivity cell are the same cell.
41. **A18: the constrained update improves the criterion on every registered cell** (all six
    start-covariance cells rise from count 0 to count 4). Says the EM ascends on the real fold; says
    NOTHING about content.
    [2026-08-23 correction: the original parenthetical named `pseudo_pair_seed0` the largest rise and
    `controlled/reference` the smallest — both wrong, the cells having been ranked by count-0
    likelihood MAGNITUDE and that ranking mislabelled as one of rises. The six rises: +2,575,212
    (`pseudo_pair_seed0`), +3,529,805 (`controlled/reference`), +4,715,910 (`random_map_seed1000`),
    +4,995,400 (`fingerprint`), +5,064,753 (`espum` tied), +6,796,844 (`espum` per_row).]
42. **A18: clause 4's honesty line reads clean** — floor share 0.0000 in all twelve cells; the one
    numerical event is 68 clipped emissions of about 45.6 million on `espum per_row 4`, the disclosed
    relaxation widening the density past float64.
43. **A18: the count-0 decode of `real/pseudo_pair_seed0` is degenerate and repair recovers it**
    (0.0152 symbols per retained token over 3 phones at count 0, 0.7411 over 39 by count 4).
    DESCRIPTIVE ONLY: no gold is read, and a content-free control also produces a full inventory at a
    plausible rate (verdict 29).
44. **A18: no cross-arm comparison is licensed from this table** — `sym/tok` is per RETAINED TOKEN
    while the table arm's audited collapse is quoted as decoded length against GOLD.
45. **A19: clause 3, the decision read, FAILS — and on the CONTROL, not the arm.** The selected real
    start's Gaussian gain over its own banked table cell is +0.0098 [+0.0058, +0.0137], so the first
    condition passes; the content-free random-map control gains +0.0251 [+0.0202, +0.0302], 2.6 times
    larger with an interval ENTIRELY ABOVE the arm's. "Comparable" carries no number and none is
    needed. LICENSES "continuous emissions are not funded at this operating point; evidence toward the
    training paradigm as the binding constraint, jointly with the banked oracle gap (0.4148 achievable
    on this stream, 0.85+ found)" — never "the paradigm cannot work"; the attribution is conditional on
    the shared `seg12.5` segmentation both arms inherit.
    [2026-08-24 completion, not a correction: the deciding control is itself clause-1 INADMISSIBLE
    (`gaussian random_map tied 4` decodes 0.7832 of gold length, under the 0.80 floor) and that was not
    surfaced when the verdict was written. The planner ruled it counts as registered: filtering it out
    after seeing that doing so flips the verdict would be an unregistered gate edit and would delete
    the length pathology the control exists to expose. The verdict is unaffected either way because
    verdict 46's positive control fails independently.]
46. **A19: on the one channel known to carry content the Gaussian swap LOSES phones** —
    `controlled/reference` at count 4 is -0.0208 [-0.0260, -0.0154] paired against its own table cell
    (pooled PER 0.4168 table against 0.4429 Gaussian). The positive control runs OPPOSITE to the small
    positive deltas on the real starts, identifying those as a length or rate effect rather than the
    geometric inductive bias the subphase tested for.
47. **A19: no Gaussian cell shows content by clause 2, and the reference cell fails it for a reason a
    reader must not misread.** Only `table|controlled/reference` shows content (+0.4700, +0.4415 over
    the babble p99). `gaussian|controlled/reference` has margins +0.4952 and +0.3990 — far above the
    0.05 bar — but is recorded `c2 = .` because clause 2 reads READABLE cells only and clause 1
    excludes it on length (0.7947 / 0.7735 of gold, under [0.80, 1.25]). That channel CARRIES content;
    it is inadmissible on length, not empty.
48. **A19: clause 1 excluded the content and admitted the babble — verdict 29's warning in reverse.**
    Seven of 24 cells fail admission, two of them the Gaussian reference cells holding the two highest
    correct-phone fractions in the table (0.6502, 0.5571), while both null cells and every real-start
    cell within 0.04 of the babble bar pass. Clause 1 is admission only, in BOTH directions.
49. **A19: the criterion ascends where the PER worsens, including on observations with no structure
    left in them.** The observation null rises +4,119,582 (-79,970,371.2 to -75,850,789.7) against
    +5,064,753 for the real arm on the same start while its PER goes the wrong way (0.8946 to 0.9252);
    on `controlled/reference` the same repair rises +3,529,805 and worsens PER 0.3498 to 0.4429. About
    four fifths of the criterion improvement survives DESTROYING the observations, so criterion ascent
    is not evidence of content and verdict 41 must never be quoted as if it were.
50. **A19: the Gaussian arm DOES read the acoustics — which makes clause 3's failure informative
    rather than vacuous.** Redrawing observations costs the arm 0.0761 of pooled correct-phone fraction
    on the selected real start (0.1509 against 0.0748), and the null's paired delta against the banked
    table cell is -0.0772 [-0.0817, -0.0728]. The finding is that the categorical table already
    extracts as much of that signal, not that the swap was inert.
51. **A19: clause 4 reads clean across every cell of both experiments** (floor share 0.0000 in all
    twelve experiment-2 cells and both null cells), and the babble null is stable at its own operating
    point (100-draw bar reproduces the 1,000-draw bar to within 6.1e-04 in the worst cell, an order of
    magnitude below the smallest clause-2 gap).
52. **A20: the Gaussian order-4 repair curve is affordable one start at a time and NOT as a
    population.** 48.88 s on the heaviest of 32 chunks -> 0.4345 h per whole-fold E-step -> 4 h per
    count-4 curve at 1.5x, against 17 h for all five starts in one process and an 11.5 h clamp that
    cannot be raised. PASS for one curve, RESOURCE_INFEASIBLE for the single-process shape; the build
    shape follows the measurement rather than the reverse.
53. **A20: the continuous emission model does not narrow the context the recursion visits** — every
    start reaches 60,879 histories and 2,435,160 arcs, identical across starts and to the accepted
    order-4 TABLE gate's probe. Cost at order 4 is a property of the state space, not of how peaked the
    emission model is.
54. **A20: the Gaussian arm's order-4 cost is within 4% of the table arm's** (48.88 s against 50.82 s
    on the same sharding of the same update role), so neither corner of the order-4 column is
    handicapped by its own cost.
55. **A21: the order-2 instantiation of the new context path IS the banked 1g.11 code path, on the
    real fold and not only on fixtures.** All five bigram corners reproduce their banked cell at both
    counts: criterion relative differences 1.9e-16 to 2.6e-15, six orders inside the declared 1e-9 bar,
    and ZERO decoded-symbol disagreements over all 890 utterances in all ten cell-count pairs. The two
    paths differ in dynamic programming (dense 78-by-78 against the history contraction) and batching
    (256 utterances per call against one), so agreement at round-off is the strongest statement
    available; bit-identity was never claimed. This buys the attribution the subphase rests on.
56. **A21: the fitted Gaussian parameters 1g.11 never persisted now exist for all five starts**
    (`parameters.npz` with `mu` and `var` at counts 0 and 4), so nothing refits to read a bigram cell.
    1g.11's own numbers recovered; quotable only within their own arm.
57. **A23: the v1-equivalent stream reproduces the published wav2vec-U v1 token rate on this bed** —
    28.01 segments per second over the seed bed against the ~28 the v2 paper's Table 1 measures for the
    v1 pipeline on LibriSpeech dev-other. The anchor was written into the plan BEFORE the job ran and
    is a different split measured by different people, so hitting it checks the construction rather
    than fitting it; the rate is stable across roles (27.67 to 28.38), so no role is segmented differently
    from the fold the transform was fitted on. Consequences: the stream is 2.24x seg12.5's rate and its
    update role carries 1,436,262 segments against 584,424 retained update tokens (2.46x per E-step),
    which at 0.4345 h per E-step projects to about 8 h per curve against the 11.5 h clamp — inside it
    with little room, which is why experiment 4's resource read is mandatory and NOT discharged by this
    projection; and the alphabet does not collapse (128 clusters used, rarest 88 segments). (Verifier
    notes not folded into the verdict: the same ratio recomputes as 28.01/12.47 = 2.25x, and "~28" is a
    two-figure anchor, so "three significant figures" overstates its precision.)
58. **A21: at the matched 4-gram, as at the accepted bigram, the criterion ranks the content-free
    control ABOVE the gold-informed reference, so criterion magnitude carries no content read.** Gains
    at matched-4g, comparable across starts on the same 584,424 tokens under the same LM:
    `real/espum_seed0_update30000` +4,616,970, `real/fingerprint` +4,535,461,
    `real/random_map_seed1000` +4,439,137, `controlled/reference` +3,268,660,
    `real/pseudo_pair_seed0` +2,490,090 — the content-free control gains 1.36x what the only
    label-built start gains, and the accepted-bigram corner ranks them the same way (+4,715,910 against
    +3,529,805, 1.34x). NOT A CLAUSE VERDICT and not about decoded output; it licenses only "this
    criterion, at this operating point, does not rank starts by content".
59. **A22: the exact order-4 readout is certified on all twenty cells, and the fitting LM provably does
    not reach the decode.** Zero exactness violations and an identical renormalized mass of 1.274e-04
    nats per emitted phone everywhere. A second check falls out of the design: at count 0 nothing has
    been fitted, so a cell's accepted-2g and matched-4g rows must be identical — they are, to the
    symbol, for all five starts in both emission models. That is the positive control for "the
    fitting-order contrast is a fitting-order contrast".
60. **A22: the near-total collapse of the pseudo-pair channel is a READOUT property, not a channel
    property — but this says nothing yet about whether either readout is correct.** The same channel,
    fold and 60,604 tokens emit 0.0152 symbols per token over 3 phone types LM-blind and 0.9786 over 36
    under the exact order-4 readout; no parameter differs. Everywhere else the LM-aware decode emits
    FEWER symbols (reference 0.7832 -> 0.7570 at count 4, fingerprint 0.7964 -> 0.7019, random map
    0.7733 -> 0.6834, espum 0.8626 -> 0.7718 at matched-4g), so the rescue is specific to the collapsed
    cell. NOT a claim that the LM-aware output is better: rate and inventory are not correctness and no
    gold is read here.
61. **A24: on the v1-equivalent stream a one-state channel is NOT refuted, where on seg12.5 it is — the
    segmentation, not the phone inventory, is what made within-symbol duration structure necessary.**
    Lag-one mutual information is almost unchanged (2.2498 against 2.2315); what changed is what a
    one-state channel is ALLOWED to produce (1.876 nats against 0.701), so the ratio falls 3.185 ->
    1.199, inside the band. Mechanism: within-symbol pair rate 0.2356 -> 0.6890, mean duration 1.31 ->
    3.22 audio tokens per phone. Two-state stays ADMISSIBLE on both streams (0.906, 1.819), so nothing
    previously admitted is refuted. TOPOLOGY: none by itself — minimum-duration-2 is standing by the
    USER's ruling; what changes is only the EVIDENCE for it, absent rather than contrary on this
    stream. BUILD, AMENDED 2026-08-24: the original guard (one-state REFUTED and two-state ADMISSIBLE)
    would have stopped every 1g.13 cell; the planner amended it to the per-route registry.
62. **A25: all five start protocols transport to the v1-equivalent stream unchanged and produce five
    distinct, valid starts.** Four ran as the accepted classes with no code change; only the controlled
    reference needed new code, for a reason about where gold lives. Every start is strictly positive
    and row-stochastic (largest row-sum deviation 8.9e-15) and no two are close (minimum pairwise mean
    TV 0.43). The concentration ordering survives (fingerprint, random-map, reference, espum,
    pseudo-pair) at normalised entropies 0.2142 / 0.3060 / 0.5985 / 0.8069 / 0.9579 against seg12.5's
    0.3434 / 0.3413 / 0.5674 / 0.6822 / 0.9356; the one substantive shift is the markedly more diffuse
    espum start. USABILITY ONLY; nothing about content.
63. **A25: the espum selection perplexity is NOT comparable across streams, and inside this stream the
    full loss beats its control decisively.** 33.4666 here against 32.5352 read as "slightly worse"
    would be a currency error: the metric is per EMITTED token and this generator emits 146,029 tokens
    against 59,751 for the same 890 utterances (2.44x, tracking the token rate). Internally: 33.4666
    against its bigram-only control's 64.1514 with 39 of 39 phones covered against 36, reproducing the
    accepted stream's behaviour (32.5352 against 55.4678, 39 against 38). The three full seeds agree
    closely (33.4666, 33.8412, 34.2041).
64. **A25: the espum arm's cost does NOT scale with the stream's token rate, so experiment 4's resource
    question is untouched by these timings.** 52 minutes per full-loss training against 47 on the
    accepted stream despite 2.46x the tokens, because the schedule is a fixed 40,000 updates over TEXT;
    the construction-only starts ran 13-18 minutes at a 132 GiB peak against 192 declared, sized from
    H1's measured 141.52 GiB same-scale reference.
65. **A26: one order-4 repair curve on the v1-equivalent stream FITS, and the binding resource is TIME
    with 2.5 hours of margin.** 128.13 s on the heaviest chunk -> 1.1389 h per whole-fold E-step -> 9 h
    against the 11.5 h clamp; memory is not binding (10.27 GiB engine + 9.14 GiB host -> 30 GiB against
    256). All five starts in one process is 43 h and stays RESOURCE_INFEASIBLE, so the shape is one job
    per start. Funds experiment 5 at that shape and nothing else; the margin is thin in the direction
    that costs the whole fold (the job caps its request at the clamp and does not resume), so the first
    real cell's wall clock is read against this projection before the rest launch.
66. **A26: the order-4 cost tracks the TOKEN count and not the observation dimension.** The fold grew
    2.46x in tokens and 5.33x in dimension while chunk time grew 2.62x, about 7% above the token
    scaling — the context recursion, not the Gaussian density evaluation, is where the E-step spends
    its time (verdict 53 from the other direction). Memory rises 7.5x. Consequence for any future
    stream on this route: a rate change is a TIME risk and a dimension change is a MEMORY risk, and
    only the first is near a limit here.
67. **A26: the reachable order-4 context is fully visited from every start.** The five starts reach
    59,204 to 60,879 of 60,879, at most 2.8% below (1,675 of 60,879), the near-uniform pseudo-pair
    start reaching exactly 60,879 and the most concentrated (fingerprint) fewest; three of five report
    IDENTICAL counts (59,319 / 2,372,760), which is itself the caution that this column reads
    structural support rather than start quality. It is about which histories carry ANY posterior mass,
    not how much, and is not a cross-stream quality read. Incidental and recurring: on the probe
    utterance the fingerprint start leaves one of the 78 emission rows with exactly zero posterior mass
    — ordinary on one 893-token utterance, but at fold level `mstep_from_statistics` refuses a
    zero-weight row by design, and that refusal is a collapse to report rather than a guard to soften.
68. **A26: the backward-recursion fix changed the NORMALIZER and not the QUANTITY, measured against a
    banked artifact rather than argued from the algebra — so no 1g.11 or 1g.12 number moves.**
    `G12EngineEquivalenceJob.sWWDLbPKglfP` recomputes `G12ResourceGateJob.3h2iIpk6lpaB`'s own five
    probe cells under the current engine: log-likelihood difference exactly 0.0 and history occupancy
    60,879 exactly; under the superseded normalizer the two posteriors agree to at worst 8.882e-16
    against a 1e-12 tolerance. The separation is asserted in the same job and keeps the rest from being
    vacuous: on a peaked case in 1g.13's shape (893 tokens, log-density gap 700) the superseded
    normalizer is NOT finite and the current one is. The superseded engine is DERIVED from the live
    source by one substitution asserted to match exactly once, never kept as a copy. All ten banked
    1g.12 repair cells carry finite fitted parameters (minimum variance 3.96e-02, zero RuntimeWarnings
    in all ten logs). NO VERDICT IS MARKED WRONG, because none rested on a number that moved — what the
    bug destroyed was work not yet done.
69. **A28: the TABLE arm on the v1-equivalent stream is not cheaper in wall clock than the Gaussian arm
    — it asks for MORE hours and has LESS headroom — and is cheaper only in memory.** PASS at 10 h and
    3 GiB beside the Gaussian arm's 9 h and 30 GiB on the same stream, probe and heaviest chunk. Per
    E-step the table arm IS slightly cheaper (123.85 s against 128.13 s, 3.3%) but carries six E-steps
    against five. Both constants are right for their own driver; what is wrong is reading one arm's
    hour figure against the other's without that column. The TABLE cells have 1.5 h of clamp headroom
    where the Gaussian cells have 2.5.
70. **A30: on the accepted 890-utterance selection role, NO real start in 1g.12 shows content by clause
    2, at either fitting order, in either emission model, under either decoder.** Content is set only
    for the gold-informed controlled reference (margin 0.39 to 0.52 over the babble p99). Every real
    start's margin is between -0.03 and +0.03 against a required 0.05 AT THE COUNT-4 EXACT-READOUT
    ROWS, with the content-free random map inside the same band; several local rows spill to about
    +/-0.04 and the pseudo-pair count-0 row sits near -0.11, and no non-controlled cell approaches
    +0.05 anywhere (scope corrected 2026-08-24; conclusion unchanged). In plain PER the four real
    starts span 0.8154 to 0.8605 at count 4 under the exact readout while the random map reads 0.8318
    to 0.8629 — the whole real population within about 1.5 points of a start built to carry no content,
    against the reference's 0.4046 to 0.4500.
71. **A30: clause 3 does not separate the selected real start from a content-free control on ANY of its
    three contrasts.** (a) the exact readout gains +0.0201 over the local decode on the selected start
    and the observation null gains MORE, +0.0328, non-overlapping — the case the carried 1g.11 ruling
    calls beyond "comparable"; (b) +0.0056 against the null's +0.0062, overlapping; (c) the table beats
    the Gaussian by 0.0092 on the selected start and by 0.0272 on the random-map control, so the
    content-free start shows the larger effect. The gate VERDICT is the planner's, but on (a) the part
    already ruled fires against the arm.
72. **A30: the exact order-4 readout is a real improvement over the LM-blind local decode, and it is
    NOT evidence of content.** Every arm cell's (a) interval excludes zero — but so does the
    observation null's, by a larger margin. The readout recovers correct phones from the language
    model, which a null whose acoustics carry nothing benefits from at least as much. This is why
    contrast (a) needed its controls before it could be read at all, and it retires the reading that
    1g.10's collapse was a readout artifact.
73. **A31: no real start on the v1-equivalent stream shows content by clause 2, the same reading 1g.12
    gave on seg12.5.** Content is Y only for the gold-informed controlled reference (14 cells); all
    four real starts are content-free under both decoders, both fitting orders and both counts. Eleven
    cells are ADMISSIBLE without being content-bearing (21 of 78 admitted overall), so this is clause 2
    answering, not clause 1 refusing to admit them.
74. **A31: clause 3 fails on its control at the same place 1g.12 did, and by a far wider margin.** On
    (a) the observation null gains +0.6689 [0.6579, 0.6804] against the arm's +0.0870 [0.0785, 0.0962]
    — non-overlapping, nearly eight times the arm; on (b) BOTH content-free controls beat the arm
    (+0.0369 and +0.0412 against +0.0161). The carried 1g.11 ruling fires here on TWO contrasts rather
    than one. The gate verdict itself is the planner's (approach 31).
75. **A31: on the registered segmentation contrast the v1-equivalent stream is WORSE than seg12.5, and
    the loss is not content-specific.** The selected real start loses -0.2959 [-0.3085, -0.2839]
    against its pinned seg12.5 counterpart and the content-free random-map control loses -0.2832
    [-0.2952, -0.2714], intervals overlapping — two arms degrading by statistically indistinguishable
    amounts, and a control had no content to lose. The observation null falls furthest (-0.5029),
    consistent with the same reading: the further a cell's output already was from gold, the more this
    stream costs it.
## Evidence index

Job dirs for runs cited above. Hashes appear once here and are referenced by name in the approach
and verdict entries.

| evidence | concrete artifact |
|---|---|
| 1g.0 structure screen, dev-clean / dev-other (A1) | `work/speech_llm/sae/structure_screen/StructureScreenJob.Xyy7r1zTK9hU`; `.U3QYclOJHgq2` |
| spectral reads, duration polarity clean/other then registered polarity clean/other (A2) | `work/speech_llm/sae/spectral_split/SpectralVCJob.AK0OUD2QcPXz`; `.ZA7uvQ2s7Zta`; `.dP9A1geKgd45`; `.koxlC99UA0t6` |
| descriptor dump and canonical 540 / 572 reads (A3) | `work/speech_llm/sae/descriptors/UnitDescriptorsJob.cSmt6LY5WVOu`; `UnitClassReadJob.m0usvL4Oxlv2`; `.kvZb0zdRznOY`; superseded all-utterance reads `.yeB6P7J3rdwz`, `.FIwiUeQ5bgGv` |
| exploratory E5 rehearsal (A4) | `work/speech_llm/sae/seed_basin/SeedBasinJob.Zm3EuTveSGBL` |
| frozen encoder states and train-fit quantizer; raw and pooled unit streams | `work/speech_llm/sae/av_states/AvStatesJob.c4Ak1rACchRC`; `work/speech_llm/sae/quantize_states/QuantizeStatesJob.FWpGhC941JMi`; `MergeUnitsPklJob.ncxcd3vouD5E`; `work/speech_llm/sae/repr_pool/SegmentPoolUnitsJob.IHRNqQfnxrQ3` |
| historical full-bed silence-delimited stream (approach 5 defect) | `work/speech_llm/sae/lexfree_match/UnitWordStreamJob.eIxgmMh99RSE` |
| phone text `T_phi`; normalized character text | `work/i6_experiments/users/wu/experiments/posterior_hmm/data/phon_lm/TextToPhonemeJob.THKMON3k9LJQ`; `work/i6_core/tools/download/DownloadJob.g4jClO48cAvP` |
| accepted construction-only H1 read (A6); superseded H1 whose runtime source was not frozen | `work/speech_llm/sae/channel_h/Phase1gH1Job.HbxKiuBTJ8aN`; invalid `.Bz5bcz5grt8B` |
| H3 corrected calibration starts, ESPUM fan-out, frozen pick and projection (A7, verdict 15) | `H3MaskedEspumStreamJob.GqAphDUVZJ7f`; `H3InitializerJob.6ifXwi6C9o4b`, `.wP5OnAoxzDow`, `.gNAARAXeogOt`; `EspumMatchTrainJob.97FwGhhItdpO`, `.eQyuM6m4rPX2`, `.lk3V9mM67j0m`, `.h4LngSZ4YvKL`; `H3EspumPickJob.ezmw64E1JwzI`; `H3CalibrationEspumProjectionJob.s4GWy36bdWxZ`; resume equivalence `H3EspumResumeEquivalenceJob.yL2E4UjTDxQ6` |
| H3 construction-population final initializers, refit and strict projection | `H3InitializerJob.uKw59MBJC4Hj`, `.ABTGA9vIwwI8`, `.BS1nPUwf1fel`; `EspumMatchTrainJob.t1l7N4lQ9dtY`; `H3FinalEspumProjectionJob.PJMwUGUXUb7s` |
| H4 calibration preparation and the four real repair trajectories (A8) | `work/speech_llm/sae/h4_jobs/H4CalibrationPreparationJob.DPv4aIqwPEzM`; reference `H4RepairJob.x1TyHJMfEVpb`; fingerprint `.iUFh7IwniCMl`; random-map seed 1000 `.Ds0zM1NTY2C1`; pseudo-pair seed 0 `.aeetC3NfgPxB`; ESPUM seed 0/update 30,000 `.ViPSmq4Am8vX` |
| H4 recovery and decoder-resource contracts (A9) | `H4UpdateReferenceArtifactJob.DZa7gIj8rZNj`; `H4QRecoveryJob.ar34r8ltGTGW`; `H4RoleDonorTableJob.w2RMXcCJyGoy`; update contract `H4ResourceContractJob.kFA99bygctlt`; selection contract `.kyMk7fwm027C` |
| H4 global-beam boundary reducer (A10, verdict 16) | `work/speech_llm/sae/h4_beam_jobs/H4GlobalBeamTableJob.ro6L8QCnqYpx` |
| H4 pre-label selection surface and provisional maxima (A11, verdict 17) | `H4SelectionSurfaceJob.MKHfnUO9XwkU`; `H4ProvisionalMaximaJob.ejmy4sdTOcS3`; audited maxima `H4ProvisionalWinnerAuditJob.kBCapQOpk1Hj` |
| 1g.2 controlled validation read, the only label reader (verdicts 18-20) | `work/speech_llm/sae/h4_validation_jobs/H4ControlledValidationJob.Otv6GBVY8ZUj` |
| controlled reference gold phones (all gold reads in this log) | `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/eval/GoldPhonesJob.ZGSp0hxyd2YP` |
| 1g.2 descriptive real-seed PER (verdict 21; selects nothing) | `work/speech_llm/sae/h4_real_seed_per/H4RealSeedPerJob.vu6Dp6HkJ2pH` |
| 1g.2a fitting LMs and their ARPAs (A14) | `H4LegacyLmJob.lZI6TrYdVpev`; `H4MatchedLmJob.T8ImJUXHaB0l`, `.Jb2m4aM2fUTy`, `.VpVkGMMy7xKW`; `work/i6_core/lm/kenlm/KenLMplzJob.ef5FXMvv8af5`, `.tis71OtNidgL`, `.bg0iYRzBQynx` |
| 1g.2a item 3 measured resource gate (verdict 22) | `work/speech_llm/sae/h4_context_resource/H4ContextResourceGateJob.HA1vzRL7MEAz` |
| 1g.2a item 4 fixed-duration diagnostic cells, decodes, PER, own-minus-donor, rank agreement (A14, verdicts 23-25) | `work/speech_llm/sae/h4_context_diagnostic/H4ContextRepairJob.*` (20 cells, incl. matched-4g `mdA3sZp68iqz`); `work/speech_llm/sae/h4_context_decode/H4ContextLocalDecodeJob.*` (60); `H4ContextDiagnosticPerJob.IYHS4cX3j3XV`; `work/speech_llm/sae/h4_context_scores/H4ContextOwnMinusDonorJob.SygqXhY8F2Qt` with its 600 `H4FixedTextScoreJob` cells; `H4ContextAgreementJob.zd6RBdYcvzti` |
| 1g.9 experiment 1, locate the collapse (A15, verdicts 26-29) | `work/speech_llm/sae/h4_collapse_locate/H4CollapseLocateJob.gZ9d6e3E7ZGu` |
| 1g.10 full-model decode read, BLOCKED by its explanation duty (verdicts 30-31) | `work/speech_llm/sae/h4_full_model_decode/H4FullModelDecodeReadJob.MXhi20TtG1I0`; 1,152 beam-512 chunks + 36 merges + 36 single-shard beam-256 probes under `work/speech_llm/sae/h4_decode_jobs/` |
| 1g.10a cross-beam defect diagnostic, DISCHARGED (verdicts 32-33) | `work/speech_llm/sae/h4_cross_beam_defect/H4CrossBeamDefectJob.2pV5rHuWJW3d` |
| 1g.10b beam-1024 convergence probe, parity PASS, 0 of 36 quotable (verdicts 34-35) | `work/speech_llm/sae/h4_beam1024_probe/H4Beam1024ReadJob.tKbQ0MHLdX03` |
| 1g.10c insertion-bonus cells, parity PASS, sign split (verdicts 36-37) | `work/speech_llm/sae/h4_insertion_bonus/H4InsertionBonusReadJob.da3bGeQIkS0R` |
| 1g.11 continuous twin, Gaussian cells, observation null, gate table (A17-19, verdicts 38-51) | `work/speech_llm/sae/g11_continuous/G11ContinuousSegmentsJob.hImWJG0X4eZh`; `work/speech_llm/sae/g11_repair_jobs/G11GaussianRepairJob.NogH62uMEI7T`; `work/speech_llm/sae/g11_nulls/G11ObservationNullJob.orOc9h6K3cuR`; `work/speech_llm/sae/g11_evaluate/G11EvaluateJob.sWoS1bP4Nd12` |
| 1g.12 experiment 1 resource read (A20, verdicts 52-54) | `work/speech_llm/sae/g12_resource/G12ResourceGateJob.3h2iIpk6lpaB` — never clear it; its recorded code identity predates two edits at an unchanged hash |
| 1g.12 experiments 2-3, ten Gaussian repair cells (A21, verdicts 55-56, 58) | `work/speech_llm/sae/g12_repair_jobs/G12GaussianContextRepairJob.` accepted-2g `0nngx4f5pX69`, `iZaUwq3DQVjj`, `OBwHBeOmwYU5`, `OyooGnuVi7EK`, `uczGmykabX6i`; matched-4g `.8OzLoDv4PPlt`, `.BrQtRIAKaWwU`, `.dDKq6J6AQEIP`, `.DgOI3SI1cwph`, `.kHwPYElOcCPr` |
| 1g.12 experiment 4, twenty exact order-4 readouts (A22, verdicts 59-60) | `work/speech_llm/sae/g12_readout_jobs/G12ExactReadoutJob.*` (20 dirs, hashes in the A22 table) |
| 1g.12 experiment 5, continuous observation null at both fitting orders and its readouts (A27) | `work/speech_llm/sae/g12_nulls/G12ObservationNullJob.tDiHo9tPpn5Z`, `.QfLZEyTjxE6o`; `G12ExactReadoutJob.ij9vB58klqDW`, `.axh5u2jyP9Va` |
| 1g.12 experiment 6, the subphase's gate table (A30, verdicts 70-72) | `work/speech_llm/sae/g12_evaluate/G12EvaluateJob.yJgxKex9peLp` |
| 1g.13 experiment 1, the v1-equivalent stream (A23, verdict 57) | `work/speech_llm/sae/g13_jobs/G13StreamBuildJob.Ob8Rh8y51x9M`; input dumps `W2vu2FeatureDumpJob.HyHAk3OCbruI`, `.WbaqNnxXpbRK` |
| 1g.13 experiment 2, route read and VAD-mask firewall (A24, verdict 61) | `work/speech_llm/sae/g13_jobs/G13RoutesJob.hStPuE1UqLK6`; `work/speech_llm/sae/g13_firewall/G13VadFirewallJob.Usfy2NF0LiSQ` |
| 1g.13 experiment 3, the five starts and the espum fan-out (A25, verdicts 62-64) | `H3InitializerJob.lR5Q4q1xRtqV` (fingerprint), `.m4sNBqlCwK2Z` (random-map 1000), `.fGmIiECLQ2XW` (pseudo-pair 0); `G13ReferenceStartJob.kG9pmxczOVgF`; `H3CalibrationEspumProjectionJob.2EB1uTDlskOy`; `EspumMatchTrainJob.oAOLIZZHVaVz`, `.18iF7DTcCNyF`, `.E9fojuqhcBDZ`, `.q59UQC0AW5Oc`; `H3EspumPickJob.ud5adF5qEliC`; `H3MaskedEspumStreamJob.6OiRRPPXl1w8` |
| 1g.13 experiment 4, re-measured order-4 resource read plus the superseded pre-fix run (A26, verdicts 65-67) | `work/speech_llm/sae/g12_resource/G12ResourceGateJob.cQ3wfqsTamPP`; SUPERSEDED `.4iWPXMh9yoJN` (orphaned by hash; kept as the record of the pre-fix measurement) |
| the backward-recursion fix and its registered anchor (A26, verdict 68) | `work/speech_llm/sae/g12_engine_equivalence/G12EngineEquivalenceJob.sWWDLbPKglfP` |
| 1g.13 experiment 5 step (b), the table arm's own order-4 gate (A28, verdict 69) | `work/speech_llm/sae/h4_context_resource/H4ContextResourceGateJob.8M4rSjaBlikH` |
| 1g.13 experiment 5, factorial pilots and the no-LM leg of the table corners (A29) | `G12GaussianContextRepairJob.mrmyPW7K6BJI`; `H4ContextRepairJob.ZOyDz3Lr5gvi`; readouts `G12ExactReadoutJob.PXkdjfKVf0VA`, `.OOCRqqetyibP`; `H4ContextChannelAdapterJob.ruZ0Muc40Aaa`; `H4ContextLocalDecodeJob.6KblLtDciuiq`; a finished table order-4 cell quoted in A29 is `g2kLTuhpq0ps` |
| 1g.13 experiment 6, the content-free controls on this bed (A29) | `work/speech_llm/sae/g12_nulls/G12ObservationNullJob.UM72oLRoTEle` (matched-4g), `.sakp81hAxfzB` (accepted-2g) |
| 1g.13 experiment 7, the gate table and the segmentation contrast (A31, verdicts 73-75) | `work/speech_llm/sae/g12_evaluate/G12EvaluateJob.a3419LhkI7JT` — contrast (d) is in `evaluate.json` only; this run's `evaluate.txt` does not render it. Pinned seg12.5 counterparts: arm `h7dasAET4GnW`, random map `gpgV68WMinJF`, observation null `axh5u2jyP9Va` |

## Verifier feedback — unresolved

Resolved rounds are dropped; the items below still stand.

- **The minimum-duration-2 topology is a LABEL the transition law does not enforce (2026-08-25; on
  the USER's desk).** Every 1g.12/1g.13 fitted artifact carries an `operating_topology` string
  reading "two sub-states, minimum duration 2", but `channel_h.py` `repair_hmm` gives BOTH sub-states
  exit arcs, so duration-one paths are legal, and the 1g.13 Gaussian LM-blind local decode realizes a
  mean duration of 1.81 tokens per phone — below 2. Durations are one tied frozen scalar per route
  (p = 0.6890 on the v1-equivalent route, label-free from the length marginal, implied mean 3.215
  tokens per phone); no channel M-step touches transitions and no artifact stores any. No gate
  verdict moves (all compared cells share the same law and clause-2 margins are length-controlled by
  construction). Enforcement versus relabel, and the label-free length-repair levers, are the USER's
  call.
- **Uncommitted shared-tree edit at the label boundary (raised to the USER, unanswered).** An
  UNCOMMITTED `config_sae_1g_v1.py` edit wires a second `Phase1gH1Job` with `gold_json`. It does not
  enter the 1g.2 graph and the accepted H1 stays pinned, but its label-boundary status needs its
  author to explain it (neither the implementer nor the verifier wrote it). Also untracked:
  `config_sae_3e1_d6_swap_cont_v1.py`.
- **1g.13 experiment-3 start-artifact hygiene, NOT done and deliberately deferred.** (i) The espum
  projection emits `espum_calibration_start.npz`/`.json`, not `start.npz`/`start.json`, and its json
  carries no top-level `name`, so anything globbing `start.npz` across the five starts silently
  misses that arm. (ii) No `start.json` records the alphabet size as a named field (128 is implied
  only by array shapes). Both items change a FINISHED job's output — `num_units` is a `run()` change
  that the banked manifests would still lack, and renaming would leave the finished dir holding the
  old name while consumers resolve through its finished marker — so both belong to a REBUILD of those
  starts, not a hash-neutral edit. Mitigation already in place: the ported start population is read
  from the route's registered table, not from a glob.
- **Standing artifact cautions.** (a) Never clear `G12ResourceGateJob.3h2iIpk6lpaB`: its recorded
  code identity predates two edits made at an unchanged hash. (b) The gold INPUT file physically
  holds all 5,567 dev utterances; sealing of the 1,112-ID evaluation role is enforced by the
  evaluating job's own filter (verified arithmetically for both gate tables), not by the file. (c)
  The 1g.12 arm readout ran before the topology-guard commit and the null readout after, so their
  `readout.json` topology blocks differ in shape; the compared numbers are invariant, but the banked
  arm readout is not byte-reproducible from HEAD.
- **Accepted methodological caveats, no action taken.** The 1g.2 validation bootstrap draws its three
  levels independently per iteration and shares them across arms rather than literally nesting
  (defensible: utterances are shared across channels and every observed interval is far from its
  threshold). `evaluate.txt`'s clause-2 side table prints margins against the raw 99th percentile
  (the +0.05 lives in `shows_content`), so any "margin against the bar" quote is derived rather than
  a stored field. Per-node worker counts claimed for the parallel E-step are recorded in no file
  (sisyphus tracks the main process only) and stay unverified rather than confirmed.
