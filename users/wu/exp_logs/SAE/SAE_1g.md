# SAE_1g — Evidence for a simple weak SAE initialization

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

## Conclusion

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

  H4's first production stage is complete as a separate calibration graph. It freezes the
  label-fitted reference, the complete controlled start library, the same-speaker donor table, and
  update-only repair trajectories at counts 0/1/2/4 for all 81 controlled and four accepted H3
  calibration starts. The graph is provenance-bound to the accepted H1, pinned alignment snapshots,
  complete `T_phi`, unit raster, and runtime source hashes. All 85 repair manifests and NPZ bundles
  are present and the reconstructed graph has no unfinished or problem jobs.

  H4's corrected recovery mechanics are specified; selector semantics remain unresolved below. For a
  fixed source-decoded hypothesis of `N` phones, a donor with `T_d` retained units and `C_d`
  silence-delimited chunks has finite score exactly when `C_d <= N <= T_d`. The old preparation table did not enforce
  that support law; exact `-inf` values are correct model behavior and must not be clipped into
  apparent content evidence. The earlier 1,237/8,900 and 316/890 prevalence used the available
  `B`-times-text-prior surrogate, not the specified original-`Q` count-0 decoder. Reconstructing the
  pinned reference `Q` gives 1,156/8,900 affected pairs across 296/890 utterances. Across all 340
  existing local outputs, a common feasible subset of the old table retains only 3,398/8,900 pairs
  and leaves 267 utterances without a donor, so pair censoring is not the production fix.

  Rebuild immutable arm-independent audio-only donor tables requiring `T_d >= T_s` and
  `C_d <= C_s` before the existing same-speaker, duration-band, unit-rate, and fallback ranking. This
  guarantees support for every hypothesis that is valid on its source audio without conditioning the
  table on method output length. The selection table uses only the 890 selection IDs as sources and
  candidates; the evaluation table uses only the 1,112 evaluation IDs as sources and candidates.
  Update/construction IDs are not donors. The exploratory all-8,416-candidate preflight
  left 792/890 selection sources eligible and 98 `no_swap`, but that is not production-pool coverage
  and must be recomputed. Use one common eligible population per role for every candidate/control
  donor contrast, retain all sources in non-donor metrics, use fixed clean/other weights 432/890 and
  458/890 for selection and 540/1,112 and 572/1,112 for evaluation, report coverage and match
  distances, and assert every retained score is finite.

  Count-0 local decoding also requires the original `Q(phone | unit)`. The 71 non-soft controlled
  starts omitted it, while the four H3 starts already persist `Q` and `B`; `Q` cannot be recovered
  provenance-safely from `B` because row normalization loses phone scale and 103/500 units have zero
  update marginal. Regenerate and bind `Q` for the reference, retain/redraw, and marginal-random
  starts and require exact canonical `B`-hash reproduction; import and verify the four H3 pairs. The
  ten old soft B-space trajectories are superseded: define
  `pi_ref(y)=sum_u m(u)Q_ref(y|u)` and
  `Q_q(y|u)=q Q_ref(y|u)+(1-q)pi_ref(y)`, convert canonically so `q=1` is exactly the reference, and
  rerun those ten. Up to 75/85 other repair trajectories remain reusable after hash verification; a
  mismatching controlled trajectory is rerun alone from its verified canonical start, while a mismatch
  in one of the four persisted H3 pairs is a provenance blocker and does not authorize an H3 relaunch.

  Before selection, implement the direct-`Q` decoder, lossless repair-count adapter, no-swap-aware
  gate, measured decoder resource contract, immutable selector-freeze artifact, and explicit final-
  refit repair mode on all 7,304 construction utterances. Calibration repair uses only the 6,414
  update IDs and reads the 890 selection IDs without fitting; final refit starts only after the
  selector freezes. `JOB_AUTO_CLEANUP=True` remains effective for managers; console mode intentionally
  overrides it only for inspection.

- 2026-08-20 — The corrected H4 prerequisite graph is independently verified and launched. All 71
  regenerated non-soft controls reproduce their retained count-0 `B` exactly, and all four imported
  H3 pairs reproduce canonical `B` from persisted `Q`; therefore exactly 75 trajectories are reused
  and the ten soft Q-space trajectories are rerun. The graph exposes 85 direct-Q starts and the full
  85-by-4 lossless channel inventory. The production selection donor law yields 513/890 eligible
  sources (235/432 clean and 278/458 other), with 377 explicit `no_swap`; these are construction-time
  facts about the frozen table, not a content result. Resource preflight is exactly 288 probes: three
  representatives by 48 cells on each of update and selection, plus one global worst-cell shard rerun
  and contract per role. Evaluation audio, labels, donors, decodes, and scoring remain absent.

  Selector freeze is blocked by a normative conflict in `PLAN_1G.md`: Section 1g.2 defines the
  own-minus-donor contrast as the selection score, while experiment 6 makes construction-fold
  likelihood the primary checkpoint selector and calls own-minus-donor only its backup. The plan also
  does not freeze a single normalization for unequal source/donor retained lengths. The launched graph
  deliberately stops at raw sufficient statistics and resource contracts; no final refit or evaluation
  graph is authorized until the planner resolves those choices.

- 2026-08-20 — Higher-order `G_fit` is registered as conditional H4-LM work, not H6; H6 remains the
  character route after a valid H5 phone handoff. First complete the corrected baseline bigram H4
  assay prerequisites: mechanics, positive controls, donor-score calibration/correlation, and selector
  validity. The method-specific nonzero-count/update-health outcome is not a prerequisite. A failed
  prerequisite is fixed first and does not trigger an LM arm. Trigger H4-LM before evaluation only if
  those prerequisites pass and the selector assigns no safe nonzero count to any admissible real phone
  seed—including when no nonzero count is safe or both seeds choose count 0. A safe nonzero choice for
  at least one seed is merely pre-evaluation-ready, not a held-out content result. The motivating Ney
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
  conditioned duration/topology read on the 6,414 update IDs. The frozen label-free selector then
  compares that coherent arm with `legacy-2g` before any 7,304-ID refit. Held-out LM perplexity is
  descriptive and never selects order. This is planned work, not a result.
