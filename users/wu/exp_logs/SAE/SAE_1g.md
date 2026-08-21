# SAE_1g — Evidence for a simple weak SAE initialization

## State
<!-- Overwritten in place, never appended; deleted at phase close. In-flight runs (job dir + the
question each answers), blockers, next action, proposals for the planner. -->

State as of 2026-08-22 -- nothing in flight for 1g:

- **H4 pre-label selection surfaces are COMPLETE** (`config/sae_1g_h4_prelabel_surfaces.py`). Both
  registered outputs resolve to finished job dirs: surface
  `work/speech_llm/sae/h4_selector_jobs/H4SelectionSurfaceJob.MKHfnUO9XwkU`, maxima
  `.../H4ProvisionalMaximaJob.ejmy4sdTOcS3`. The numbers and the local-winner reading are approach
  11 and verdict 17.
- Read this before believing a STALLED report on this config: the manager exits on sisyphus's
  interactive "All calculations are done, print verbose overview (v), update outputs and alias (u),
  cancel (c)?" prompt, which under `nohup` raises `EOFError` and looks exactly like a crash in the
  manager log. The one-shot console status then calls the finished consumer jobs `waiting`, so a
  watcher verdict reads `STATUS=STALLED ... work remains` on a graph that is fully finished. Grep
  the manager log for "All calculations are done" before restarting anything.
- Nothing else in 1g is running. The 821-job H4 prerequisite graph and the beam table are preserved.

Blockers: none.

Next action is the planner's, not the implementer's: the next step in `PLAN_1G.md` 1g.2 opens the
controlled labels, and the audit that had to precede them is discharged by construction (verdict
17). No 1g job should start until the planner rules on that.

Proposals for the planner:

1. The frozen-versus-next-beam winner audit standing in front of the controlled labels is satisfied
   without running it: all 85 provisional winners are `decoder.kind = "local"`, and `PLAN_1G.md`
   states that a local winner needs no beam audit. The at-most-320 shard cells budgeted for it are
   not needed at this boundary.
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
    provisional-maxima read, all on the 890 selection utterances and reading no label. Each tuple's
    statistic is the own-minus-donor fixed-text channel rate
    (`own_logp/own_retained_units - donor_logp/donor_retained_units`), summed with `math.fsum` in
    sorted-id order and weighted by split size (dev-clean 432, dev-other 458). The maximum over the
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


## Catalog

| evidence | concrete artifact or source |
|---|---|
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
  reference labels is in `PLAN_1G.md` Status. One wording hand-back (approach 11,
  implementer-owned): "all on the 890 selection utterances" overstates coverage — the statistic
  contributes from the 513 donor-eligible sources (235 clean / 278 other, identical in every
  tuple; the 377 `no_swap` sources are absent by construction and the fixed weights are the
  registered no-renormalization rule, so nothing is wrong beyond the phrase). Two observations
  for the validation stage, no action now: (i) only 79 distinct channel content hashes among the
  85 starts — `controlled/map_q09_draw00..04` are all one channel (draw seed has no effect at
  q09) and `controlled/soft_q09` is bit-identical to `controlled/reference` — so effective
  independent controls are 76 of 81, which any clustered CI or null spread over the controls
  must respect (implementer: please confirm in one line that both degeneracies are construction,
  not a copy error); (ii) the cross-start surface ranks the random-map null 9th of 85 (10.7753)
  and the reference 73rd (5.8265), with the registered five rows at ranks 9/72/73/76/84 — the
  controlled label validation must be read knowing the pre-label `Sel` ordering places nulls
  above the reference, and winning repair counts across all 85 starts are {0: 72, 4: 13} (counts
  1 and 2 never win).
