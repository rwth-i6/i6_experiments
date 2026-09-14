# SAE §1f — statistics-matching initialization, prerequisite kill conditions

## State

State as of 2026-08-26. Active: **entry 9.1a, one arm (A9a) on the pairwise-merged stream** --
`uni+bi+tri`, seed 0, 40,000 updates, the only constructor variable against entry 7's banked
full-loss arm being `segment_dir`. Question: does the rate repair move a trained arm off 1.6828, and
where does it land against c5's 0.5421 ceiling and 0.9075 audio-free null. USER-funded 2026-08-26 as
an explicit override of gate 9.0's registered consequence.

Run pointers: `GuaTrainJob.N8moRyLEIytz` (live, 0.77 updates/s -> 14-16 h; `time_rqmt` 11.0 h, and
the job resumes) -> `GuaPinCheckpointJob.vSjIrfcoMvM1` (update 40,000 by declaration) ->
`GuaGenerateJob.ifQToLNPwMHR` (released viterbi, per-frame argmax, no LM) ->
`GuaScoreJob.wRFXGXZHdUoB` (manager registered in `sis_managers.sh`); clipping reader
`GuaFixedStreamReadJob.EcZEy9ZmD1eZ` (this arm and entry
7's two rows through ONE reader). Stream `GuaMergedSegmentsJob.aqDChbdylhLD` (train 8,416 utts at
13.977 seg/s, valid 2,292 at 14.255, test 572 at 14.375, gold 13.548/s; its internal frame check
passes, 2,699 of 2,699 of entry 9.3's c5 pool at fingerprint `93e6ee25c009`). Code speech-llm
`cfa9121` / `10bbd48` / `1ae0d86` / `b5ed8c0`; config `config/sae_1f_entry91a.py`.

Scope: ONE arm. A9b and the audio-swap control are NOT built, so gate 9.1 clauses (2) SIGNATURE and
(3) CONTENT are not fired; only clause (1) HEALTH is read, and it reports rather than gates. No
reading of 9.1a reopens gate 9.0, licenses entry 9.2, or counts as a reproduction of the published
0.473.

Next action: at update 40,000 read PER and the applied-step table through the one reader, applying
G8 verbatim. Early live evidence (updates 14-336: 91.7 % of epochs fully clipped, mean gradient norm
391, applied step near 6e-04 against the banked arm's 6.752e-05) is not a result.

Open for the planner: the two entry-8 constants of 2026-08-23 (`sil_weight` inert on this
vocabulary; the label-free selection rule anti-selects on the full-loss arms).

## Gates (pre-registered; amendments and voidings marked, originals kept)

- **G1 kill condition (i)** — dev-other oracle-map PER <= 0.50. FAILED on `enc50_raw` (verdict 1),
  CLEARED by data-driven pooling (verdict 6).
- **G2 kill condition (ii) / `tv_offdiag` bar** — span term >= 25 %, plus a position term. **VOID AS
  MEASURED** (planner 2026-08-16, verdict 9): unreadable as written on every representation; entries
  1/4 stay parked, no post-hoc replacement was registered.
- **G3 ladder entry 2** — sigma_min(P_X) must admit the closed-form estimator. Entry 2 CLOSED by it
  (verdict 8).
- **G4 arm gate** (entries 3, 5, 7 and ruling 3, unchanged throughout): **M1 >= 0.05 AND M2 >= 0.05
  on dev-other, plain PER as scored against the same sil-free gold phones.** Entry-5 operating point:
  banked `seg12.5` phone-side nulls n1 0.8946, n2 0.9239, memoryless ceiling 0.4148, so the M1 bar is
  0.8446 on the 572-utterance scored fifth.
- **G5 entry 3 kill-tests** — manner separation ~0.50 over majority baselines; admitted-pair
  precision ~0.70.
- **G6 entry 6 signature** (fixed before the run; the English reference row is scored under the same
  rule): top 20 by frequency, at or below the frequency-weighted mean unit-word length, opens an
  utterance >= 10 % of the time, closes one <= 2 % of the time.
- **G7 entry 7 stage-A signature** — bigram-only minus full-loss PER >= +0.10. **UNFIRABLE BY
  CONSTRUCTION** (verifier 2026-08-25 (A)): the bar came from the published 71.6 vs 39.2 delta, but
  both arms carry `pos_unigram_weight: 1.0` (default at `wav2vecu_graph.py:113`, set by no shipped
  config, `run.sh` line or job of ours; entry 5 hardcodes it at `espum_model.py:26/152`), so the pair
  run is `uni+bi` vs `uni+bi+tri`. Published Table 3 (verified twice, PDF and ar5iv): bigrams only
  71.6 / uni+bi 39.2 / uni+bi+tri 38.4, i.e. a separation of 0.8 PER points. Traps for a later fix:
  `skipgram_only` (line 69) is DEAD CODE, and the `posweight1_1` in the config filename is the
  segmenter BCE positive weight 1.1. Original bar kept; verdict 35.
- **G8 entry 9.1a reading rules** (registered pre-result 2026-08-26, at updates 126/476, before any
  PER existed). Applied-step ratio against the banked full arm: **above 1.25 AMBIGUOUS, 1.00 or below
  a fortiori, between them comparable within the declared band and NOT a fortiori** (1.00 is the
  2026-08-18 asymmetry of verdict 39; 1.25 is a declared convention erring toward AMBIGUOUS, as no
  applied-step-vs-PER calibration exists on this bed). (a) The RATIO travels with the verdict wherever
  it is quoted; no bare label. (b) Registered direction: an arm that does NOT improve on 1.6828 while
  applying materially more parameter movement per update **strengthens** the negative. (c) The label
  is direction-aware -- "APPLIED STEP MATERIALLY LARGER at Nx" for a non-improving arm, never
  "AMBIGUOUS at Nx". (d) **Neither label is asserted when PER is within 0.027 of 1.6828 in either
  direction; that band reports "NO MATERIAL CHANGE at Nx"** with the ratio attached (0.027 is the only
  seed-to-seed spread measured on a comparable arm, entry 5's full-loss seeds; 9.1a has one seed).
- **G9 gate 9.0**, in the producing module's docstring before any number existed: **mean H >= +0.05
  with H > 0 in at least 7 of every 8 batches, read per configuration.** FAILED on all five (42).
- **G10 entry 8** — no gate. Registered reporting rule (planner 2026-08-23): every entry-8 quote is
  the TRIPLE (label-free pick, label-oracle best as upper bound, grid range); the anti-selection is
  named wherever the pick appears; the label-oracle best never stands alone. The `sil_weight` axis is
  RETIRED (`RETIRED_GRID`); a repaired label-free selector would be a NEW registration, not a patch.
- **G11 E1 plumbing probe** — one-sided floor test: pass = at or below the 0.4148 memoryless
  oracle-map ceiling (a kernel-4 context window is a strictly richer class than a one-phone-per-unit
  map, so landing below it is the expected pass; the guarded failure was landing far above).

## Approach and results

**1. Kill condition (i): the §0a information audit re-read on the current `enc50_raw` inventory.**
`AuditAvUnitsJob` reused verbatim on the k-means-500 / PCA-96 codebook over the pretrained
wav2vec2-large-lv60 tap at 50 Hz, 500 LibriSpeech dev utterances against MFA gold, 80/20 seed-0
held-out protocol. The superseded 50 Hz codebook of the same K, PCA dim and seed is re-scored on the
identical utterances and reproduces its finished 0.424 / ins 0.189 row exactly, which pins the
protocol. Bar G1.
| inventory | split | oracle PER | sub | ins | del | purity | frame acc | H(phone\|unit) | PNMI | units/phone | dead |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `enc50_raw` (in use) | dev-clean | **0.712** | 0.115 | 0.591 | 0.006 | 0.694 | 0.680 | 1.046 | 0.682 | 2.794 | 11 |
| `enc50_raw` (in use) | dev-other | **0.832** | 0.132 | 0.692 | 0.008 | 0.659 | 0.657 | 1.186 | 0.633 | 2.824 | 10 |
| `enc50_prior` (superseded) | dev-clean | 0.424 | 0.185 | 0.189 | 0.050 | 0.659 | 0.644 | -- | -- | -- | 0 |
| `enc50_prior` (superseded) | dev-other | 0.451 | 0.195 | 0.209 | 0.046 | 0.637 | 0.632 | -- | -- | -- | 0 |

**2. Kill condition (ii): the §1a conclusion-6 channel-structure claim, measured.** Unit co-occurrence
graph projected to phones through the ORACLE unit-to-phone map fitted on the same gold frames (so
every number bounds the matcher from above), compared with the phone bigram of the same utterances.
Controls rebuild each utterance from its OWN real segments cut at gold boundaries: `seg_swap` draws
each segment from another occurrence of the same phone (the channel factorizing as a matcher assumes),
`seg_rand` from a random phone (the floor). Only 0.347 / 0.341 of adjacent pairs cross a boundary, so
`all` is what a matcher can observe and `cross` is the part that could carry the bigram; `tv_offdiag`
is the matcher's own L1 objective at a map it will never have, and the PMI versions are read because
raw cell correlations are confounded by shared phone-unigram frequencies.
| split / graph | row | diag_frac | spearman_pmi | pearson_pmi | tv_offdiag |
|---|---|---|---|---|---|
| dev-clean, all | real | 0.396 | 0.373 | 0.601 | 0.431 |
| | seg_swap | 0.320 | 0.413 | 0.707 | 0.409 |
| | seg_rand | 0.321 | 0.214 | 0.584 | 0.459 |
| dev-clean, cross | real | 0.261 | 0.515 | 0.665 | 0.299 |
| | seg_swap | 0.044 | 0.703 | 0.857 | 0.314 |
| | seg_rand | 0.046 | 0.029 | 0.006 | 0.430 |
| dev-other, all | real | 0.385 | 0.370 | 0.612 | 0.454 |
| | seg_swap | 0.308 | 0.398 | 0.700 | 0.432 |
| | seg_rand | 0.307 | 0.216 | 0.595 | 0.474 |
| dev-other, cross | real | 0.267 | 0.517 | 0.663 | 0.315 |
| | seg_swap | 0.045 | 0.682 | 0.841 | 0.338 |
| | seg_rand | 0.045 | 0.035 | 0.075 | 0.441 |

**3. The screen battery, one row per REPRESENTATION of the same stream.** Five frame-level 50 Hz
representations through the same three jobs; the raw row reproduces its registered PER exactly.
`seg16/12.5/9` pool encoder features by adjacency-constrained Ward merging to 16 / 12.5 / 9 tokens per
second and relabel each segment by the same codebook (re-quantizing 20 utterances asserts bit-exact
codes); `brown100` merges the inventory 500 -> 100 by bigram context at the raw rate; `ubpe12.5`
merges TOKENS by unit-BPE (8000 merges, the cap, reaching 14.1 tok/s against a 12.5 target). Both
coarsenings fit on the 28 539 train utterances only, never on the dev the screens read. Added to the
registered protocol: `PER_str` (a unit-to-phone-STRING map, the ceiling of a token map that can
delete) and sigma_min of P_X, the identifiability condition of arXiv:2306.07926's estimator (G3).
| repr | split | u/phone | PER | ins | del | PER_str | PNMI | cross | tv real | tv swap | tv rand | pmi real | pmi rand | sigma_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `raw` | dev-clean | 2.79 | 0.712 | 0.591 | 0.006 | 0.587 | 0.682 | 0.347 | 0.431 | 0.409 | 0.459 | 0.373 | 0.214 | 5e-33 |
| | dev-other | 2.82 | 0.832 | 0.692 | 0.008 | 0.705 | 0.633 | 0.341 | 0.454 | 0.432 | 0.474 | 0.370 | 0.216 | 5e-33 |
| `seg16` | dev-clean | 1.46 | 0.385 | 0.144 | 0.062 | 0.387 | 0.648 | 0.608 | 0.311 | 0.342 | 0.386 | 0.576 | 0.426 | 0 |
| | dev-other | 1.48 | 0.452 | 0.178 | 0.054 | 0.443 | 0.604 | 0.591 | 0.338 | 0.367 | 0.406 | 0.580 | 0.434 | 0 |
| `seg12.5` | dev-clean | 1.16 | **0.380** | 0.058 | 0.143 | 0.391 | 0.620 | 0.700 | 0.302 | 0.341 | 0.383 | 0.595 | 0.465 | 0 |
| | dev-other | 1.18 | **0.414** | 0.067 | 0.117 | 0.435 | 0.581 | 0.681 | 0.321 | 0.362 | 0.397 | 0.624 | 0.491 | 0 |
| `seg9` | dev-clean | 0.85 | 0.466 | 0.013 | 0.314 | 0.418 | 0.556 | 0.792 | 0.362 | 0.389 | 0.418 | 0.594 | 0.469 | 0 |
| | dev-other | 0.86 | 0.481 | 0.014 | 0.290 | 0.458 | 0.525 | 0.776 | 0.371 | 0.407 | 0.435 | 0.634 | 0.513 | 0 |
| `brown100` | dev-clean | 2.52 | 1.063 | 0.742 | 0.001 | 0.850 | 0.527 | 0.381 | 0.593 | 0.527 | 0.548 | 0.109 | 0.061 | 2e-17 |
| | dev-other | 2.55 | 1.152 | 0.833 | 0.003 | 0.891 | 0.491 | 0.374 | 0.587 | 0.523 | 0.544 | 0.103 | 0.054 | 8e-18 |
| `ubpe12.5` | dev-clean | 1.15 | 0.458 | 0.073 | 0.158 | 0.352 | 0.710 | 0.659 | 0.340 | 0.359 | 0.399 | 0.631 | 0.476 | 0 |
| | dev-other | 1.27 | 0.538 | 0.149 | 0.116 | 0.436 | 0.671 | 0.608 | 0.363 | 0.374 | 0.411 | 0.589 | 0.427 | 0 |

Planner ruling 2026-08-16: screens run on ALL pooled rungs plus `ubpe12.5`, `seg12.5` primary and
`seg9` the label-free-defensible rung; no label-selected rung enters the method. The pooled streams
cover only the seed dump's 8416 utterances (2849 train plus all 5567 LibriSpeech dev) -- funding a
matcher on one needs the same pooling pass over the assign-side tc100 / 960 h shards.

**4. Ladder entry 3: the unary fingerprint assignment, read against G4.** Per-unit transition-free
statistics from unpaired audio (log relative frequency, utterance-initial and -final rate, six-bin
position histogram) matched to the same statistics of `T_phi` phones by one entropic optimal-transport
solve on the two frequency marginals, regularization fixed at 0.1 before the run, after a label-free
two-means edge-enrichment split names silence units and maps them to SIL outside the solve. Candidate,
random null n1, pseudo-pair null n2 and the eval-only oracle are all fitted without labels on all 8416
utterances and scored through the registered oracle-map protocol on the same held-out fifth. Added
reads: the audio-swap control run on the oracle and both nulls (so M2 is a fraction of the span
between content-carrying and content-free), `pos` (re-solve without the frequency column), and
`top5`/`mrr` under the fingerprint cost itself (chance 0.128 / 0.109).
| repr | split | oracle | cand | pos | n1 | n2 | M1 | M2 | M2 n1 | M2 oracle | top5 | mrr | manner | major | admit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `raw` | dev-clean | 0.727 | 1.790 | 1.722 | 1.945 | 0.902 | -0.887 | 0.012 | 0.009 | 0.596 | 0.318 | 0.203 | 0.435 | 0.346 | 0.048 |
| | dev-other | 0.773 | 1.802 | 1.736 | 1.951 | 0.908 | -0.894 | 0.005 | 0.004 | 0.572 | 0.310 | 0.204 | 0.437 | 0.344 | 0.050 |
| `seg16` | dev-clean | 0.388 | 1.006 | 0.981 | 1.032 | 0.912 | -0.095 | 0.008 | 0.007 | 0.576 | 0.266 | 0.193 | 0.488 | 0.372 | 0.286 |
| | dev-other | 0.439 | 1.001 | 0.979 | 1.028 | 0.915 | -0.086 | 0.014 | 0.013 | 0.532 | 0.280 | 0.202 | 0.447 | 0.376 | 0.286 |
| `seg12.5` | dev-clean | 0.370 | 0.882 | 0.860 | 0.896 | 0.922 | **+0.014** | 0.012 | 0.005 | 0.518 | 0.251 | 0.181 | 0.475 | 0.413 | 0.000 |
| | dev-other | 0.405 | 0.882 | 0.866 | 0.897 | 0.924 | **+0.015** | 0.016 | 0.010 | 0.489 | 0.261 | 0.185 | 0.477 | 0.403 | 0.000 |
| `seg9` | dev-clean | 0.451 | 0.821 | 0.821 | 0.834 | 0.922 | +0.012 | 0.011 | 0.007 | 0.392 | 0.236 | 0.166 | 0.458 | 0.471 | 0.000 |
| | dev-other | 0.475 | 0.828 | 0.826 | 0.837 | 0.922 | +0.009 | 0.009 | 0.011 | 0.372 | 0.226 | 0.156 | 0.452 | 0.465 | 0.000 |
| `ubpe12.5` | dev-clean | 0.412 | 0.894 | 0.869 | 0.916 | 0.812 | -0.083 | 0.022 | 0.020 | 0.488 | 0.251 | 0.176 | 0.410 | 0.344 | 0.067 |
| | dev-other | 0.470 | 0.944 | 0.917 | 0.979 | 0.820 | -0.124 | 0.016 | 0.014 | 0.468 | 0.247 | 0.175 | 0.421 | 0.344 | 0.000 |

Frame limitations ratified 2026-08-16: the fingerprint set is narrower than the registered spec (no
duration, no mid-utterance silence adjacency) -- neither has a text-side counterpart, the spec
overpromised. The regularization sweep is diagnostic-only; its reg-1 "gain" is marginal collapse
(induced-marginal L1 1.2-1.4), the standing example of why the audio-swap read exists. Null n2 is the
1e pseudo-pair PROTOCOL transplanted to map form (the 1e init is an SFT run, not a map) -- an
implementer choice open to revision.

**5. Ladder entry 6's kill-test: is any frequent unit-word a function word?** The same label-free
edge-enrichment split names silence units, those cut each utterance into silence-delimited segments,
and ONE greedy unit-BPE merge list is learned over the segments toward the 2.8 unit-words/s English
word rate -- every row is a PREFIX of that single list. Signature per G6; `base` is the rate a
positionally indifferent unit-word would show; the last row applies the identical rule to the top 20
words of the raw LM corpus.
| row | word/s | units/word | zipf | base | hits | best init | best/base | hitting ids (units) |
|---|---|---|---|---|---|---|---|---|
| `seg12.5` @0 | 9.77 | 1.00 | -0.501 | 0.0117 | 0 | 0.073 | 6.3 | — |
| @0.25 | 5.31 | 1.84 | -1.035 | 0.0215 | 1 | 0.186 | 8.7 | 403 (1.0) |
| @0.5 | 4.81 | 2.03 | -1.008 | 0.0237 | **3** | 0.207 | 8.7 | 403, 423, 432 (1.0) |
| @0.75 | 4.52 | 2.16 | -0.986 | 0.0253 | 2 | 0.215 | 8.5 | 403, 423 (1.0) |
| @1 | 4.26 | 2.29 | -0.903 | 0.0268 | 1 | 0.227 | 8.5 | 403 (1.0) |
| `ubpe12.5` @0 | 10.68 | 1.00 | -0.987 | 0.0107 | 0 | 0.024 | 2.3 | — |
| @0.25 | 8.76 | 1.22 | -0.973 | 0.0130 | 0 | 0.034 | 2.6 | — |
| @0.5 | 8.07 | 1.32 | -0.988 | 0.0141 | 1 | 0.211 | 14.9 | 397 (1.0) |
| @0.75 | 7.63 | 1.40 | -0.979 | 0.0150 | **2** | 0.215 | 14.4 | 397, 608 (1.0) |
| @1 | 7.29 | 1.46 | -0.954 | 0.0157 | 1 | 0.215 | 13.8 | 397 (1.0) |
| TEXT words | — | 2.49 | -1.387 | 0.0503 | 2 | 0.287 | 5.7 | I, HE |

**6. Entry 6's onset control: does a hit mean something, or only mark where speech begins?** Eval-only
on entry 6's own stream, split and merge list, reading each hit's gold labels at its utterance-initial
occurrences against its other occurrences (`TV_ie`) and against the corpus's utterance-onset mixture
(`TV_i_onset`); silence and word-uncovered frames are held out and returned as `off_init`. Labels
restrict only the columns (stream, split, merge list and type selection come from all 8416 utterances,
5567 of which carry an alignment); every hit is read against `other`, the median over the remaining
top-20 types with >= 20 initial occurrences.
| row | id | n_init | n_else | off_init | phone init -> else | ph `TV_ie` | ph `TV_i_onset` | word init -> else | wd `TV_ie` | wd `TV_i_onset` |
|---|---|---|---|---|---|---|---|---|---|---|
| `seg12.5` @0.25 | 403 | 127 | 401 | 0.017 | AH .56 -> AH .53 | **0.074** | 0.674 | THE .79 -> THE .73 | **0.245** | 0.910 |
| @0.5 | 403 | 98 | 273 | 0.022 | AH .54 -> AH .53 | **0.079** | 0.657 | THE .77 -> THE .74 | **0.245** | 0.909 |
| @0.75 | 403 | 85 | 216 | 0.025 | AH .55 -> AH .52 | **0.092** | 0.654 | THE .76 -> THE .74 | **0.243** | 0.912 |
| @1 | 403 | 85 | 198 | 0.025 | AH .55 -> AH .52 | **0.094** | 0.655 | THE .76 -> THE .74 | **0.242** | 0.913 |
| @0.5 | 423 | 42 | 280 | 0 | T .72 -> T .66 | 0.187 | 0.751 | IT .21 -> BUT .11 | 0.715 | 0.864 |
| @0.5 | 432 | 27 | 264 | 0 | N .67 -> N .49 | 0.321 | 0.723 | IN .37 -> AND .68 | 0.630 | 0.917 |
| @0.5 other | — | 1 | — | 0.052 | — | 0.291 | 0.384 | — | 0.915 | 0.866 |
| `ubpe12.5` @0.5 | 397 | 79 | 94 | **1.000** | — | — | — | — | — | — |
| @1 | 397 | 79 | 92 | **1.000** | — | — | — | — | — | — |
| @0.75 | 608 | 24 | 157 | 0 | Y .49 -> Y .53 | **0.150** | 0.945 | YOU .76 -> YOU .63 | **0.338** | 0.983 |
| @0 other | — | 4 | — | 0.075 | — | 0.488 | 0.735 | — | 0.850 | 0.910 |

Caveat (verifier 2026-08-16, no conclusion flipped): the other-type median pool thins to 1-3 types at
the deciding prefixes; 403's call clears even the 12-type prefix-0 pool, so it stands.

**7. Ruling 3: lexicon-free text sides against the phone reference arm, on all four audio
representations.** Each text side is screened at the merge-list prefix whose MEASURED unit-word rate
sits closest to its own token rate (the argmin over each rung's measured curve reproduces on all 12
selections; no discretion), and the ceiling is refitted inside the candidate's own map space so oracle
and candidate price the same hypothesis set. The eval set is identical across rungs; the rungs differ
on the train side only. Every stream stopped on `no_pair_repeats` rather than on the merge cap or the
rate target, so no rung reached its target token rate and each `words` row is screened against audio
running 1.24-2.30x faster than its text. `max_fit_tokens` bit on `ubpe12.5` alone (1500017 of
3867583) and compresses merge-FITTING only. Gate G4, read per representation AND per text side.
| rung / text side | prefix | audio rate | text rate | M1 | M2 | gate |
|---|---|---|---|---|---|---|
| **`seg9`** — 8416 utts, 20.5 h, K=500, 30344 merges, fit 255006/206413 | | | | | | |
| phones | 0 | 7.04 | 9.86 | 0.0146 | 0.0146 | fail |
| bpe512 | 0.17 | 4.56 | 5.39 | -0.0768 | 0.0134 | fail |
| words | 1 | 3.46 | 2.80 | -0.0536 | 0.0141 | fail |
| **`seg12.5`** — 8416 utts, 20.5 h, K=500, 38228 merges, fit 313918/206413 | | | | | | |
| phones | 0 | 9.77 | 9.86 | 0.0137 | 0.0148 | fail |
| bpe512 | 0.17 | 5.62 | 5.39 | -0.1914 | 0.0173 | fail |
| words | 1 | 4.26 | 2.80 | -0.1343 | 0.0152 | fail |
| **`seg16`** — 8416 utts, 20.5 h, K=500, 43821 merges, fit 360623/206413 | | | | | | |
| phones | 0 | 12.30 | 9.86 | -0.0863 | 0.0202 | fail |
| bpe512 | 0.67 | 5.29 | 5.39 | -0.1462 | 0.0150 | fail |
| words | 1 | 4.89 | 2.80 | -0.2254 | 0.0120 | fail |
| **`ubpe12.5`** — 34106 utts, 111.0 h, K=8500, 108169 merges, fit 925126/433873, kept 1500017/3867583 | | | | | | |
| phones | 0 | 9.68 | 9.86 | -0.0108 | 0.0252 | fail |
| bpe512 | 1 | 6.45 | 5.39 | -0.3216 | 0.0049 | fail |
| words | 1 | 6.45 | 2.80 | -0.5201 | 0.0078 | fail |

Registered amendments (2026-08-16/17): 512 word types RATIFIED (adopted BPE-512 type count); the
unrestricted oracle OVERTURNED to the restricted one (the ceiling must live in the candidate's map
space) and the rung re-run, existing gate reads remaining valid; the cross-generation determinism
check PASSES exactly on all three seg rungs (old `oracle` == new `oracle_open`, oracle-independent
columns bit-for-bit). The `ubpe12.5` stream defect is CONFIRMED from ground truth (`learn_unit_bpe`
`max_merges` default 8000 not overridden at the build call; the stream stopped on the budget at 14.08
tok/s against the 12.5 target), so its vocabulary is a default's artifact, the rung is not rate-matched
to `seg12.5`, the matched-rate contrast is retired and the operating point is named on every read.
Reader note: the restricted oracle can read BELOW `oracle_open` on held-out rows (`seg12.5` words
dev-clean 0.672 vs 0.715) -- expected where the smaller map space generalizes better, not a defect.

`ubpe12.5` words cell, addendum (7a), 2026-08-17: its restricted transcript-built ceiling is
NON-FUNCTIONAL -- PER 1.0667 against the empty hypothesis's 1.000, word currency 1.4193, and it LOSES
to the content-free null n2 (0.8814) on the same held-out rows, so the cell cannot rank maps by
content (unique to ubpe words: the three seg words cells keep functional restricted ceilings
0.671/0.711/0.770 dev-other and ubpe's open ceiling is functional at 0.699). The row stays a gate
FAIL, annotated uninformative about matching quality; no word-level re-run is registered.
**RETRACTION (2026-08-17, later):** the "word-currency error floor ~1.32 regardless of assignment
quality" claim is WRONG and withdrawn -- the scored-fifth audio rate is 7.114 symbols/s (6.455/s is
the corpus-wide figure) and within-segment repeat collapse is map-dependent (restricted ceiling emits
1.742x reference tokens shedding 32 %, candidate 2.354x shedding 8 %, a constant map lands near 1.0),
so no class-wide floor above 1.0 follows from the rate. What stands: the rate mismatch (2.55x
scored-fifth emission capacity) dominates and dwarfs the coverage effect; token-level coverage on the
scored fifth is 0.7174, so the closed-vocabulary floor is 0.283 in word currency (the earlier 0.566
figure was wrong -- the `cover` column 0.4344 is FRAME-weighted over all 50 Hz frames including
silence; within-word-interval frame coverage is 0.5404, in-vocab words average 10.87 frames vs 23.47
OOV); and the crispest uninformativeness statement is n2 0.8814 beating the restricted ceiling 1.0667.
Gate reads are unaffected (M1/M2 are candidate-vs-null, ceiling-independent).

**8. Ladder entry 5: training-based statistics matching on the pooled seed stream (the funded batch).**
A one-layer convolution, kernel 4 over the one-hot 500-way unit stream, trained so the batch count
statistics of its segment-level phone posteriors (positional unigram at absolute segment index,
skipgrams at skips 1-6, tri-skipgrams) match those of unpaired phone text under an L1, with segment
boundaries FIXED from the battery's own pooling and a smoothness penalty on adjacent segments' logits.
Every checkpoint and the reported seed are chosen label-free by phone-LM perplexity of the arm's own
decodes weighted by inventory use; PER is computed afterwards by a separate job and the trainer is
handed no reference file. E1 supervises the same input-and-pooling path on the eval-only forced
alignment and discards its checkpoint (G11). Gate G4 at the entry-5 operating point.
| arm | label-free ppl (pre-`abc3d81`) | update | dev-other PER | sub | ins | del | hyp/ref | M1 | M2 | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 supervised probe (plumbing, ceiling-fit rows, checkpoint discarded) | — | — | **0.3565** | 0.1296 | 0.0213 | 0.2056 | — | — | — | — |
| full loss, seed 1 (**label-free selected**) | **31.41** | 40000 | **0.8580** | 0.6909 | 0.0699 | 0.0972 | 0.973 | +0.0365 | +0.0466 | **fail** |
| full loss, seed 2 | 31.49 | 40000 | 0.8848 | 0.7195 | 0.0696 | 0.0958 | 0.974 | +0.0098 | +0.0244 | fail |
| full loss, seed 0 | 33.04 | 40000 | 0.8770 | 0.7074 | 0.0717 | 0.0979 | 0.974 | +0.0175 | +0.0302 | fail |
| `bigram_only` arm = positional unigram + skip-1 bigram, seed 0 | 53.86 | 30000 | 0.8748 | 0.7096 | 0.0641 | 0.1011 | 0.963 | +0.0198 | +0.0254 | fail |
| ruling-3 unary candidate, same rung and text side (approach 7) | — | — | 0.8809 | 0.7157 | 0.0478 | 0.1174 | — | +0.0137 | +0.0148 | fail |

CURRENCY NOTE: the perplexity column is in the PRE-`abc3d81` convention (the N+1 bos/eos
log-probabilities divided by the N emitted phones) and may not be compared to any value produced
after that commit, which normalizes per scored event. The ranking it was read for (seed 1 ahead of
seed 2 by 0.08) is unaffected: the three seeds score the same utterances at hypothesis lengths within
0.1 percent, so the correction is monotone across them. `PPL_NORM` now travels in the metric dict and
is rendered in the three reports the number is read from, so a render without that line is pre-fix by
construction (the banked `GuaLmGridReadJob.SeNSdRhV1Wo3` report carries no such line).

The arm every job handle calls `bigram_only` is the published `uni+bi` configuration, not the
reference's collapse arm (G7, verdict 32); job handles keep their creation names. Standing caution
(2026-08-17): the selection metric's clean ablation separation (53.86 vs 31-33) with indistinguishable
PER licenses it for checkpoint/seed picking only, NEVER for method ranking. E1 watch item: deletions
dominate even supervised (0.2056 of 0.3565, insertions 0.0213, expected at 9.771 seg/s against a
9.86/s reference phone rate after duplicate collapse), so judge collapse signatures on the sub/ins/del
split alongside total PER. The `seg9` contingency was NOT exercised (identity-dominated failure at a
near-correct emission rate; rung choice not decision-relevant).

**9. Ladder entry 7: the published graph-based pipeline, run verbatim on our seed bed (USER ruling 6).**
GraphUnsupASR (arXiv:2310.02382) executed as published -- its own feature extraction, clustering, text
preparation and trainer -- on the same seed audio and unpaired text, so a gap against entry 5 is
attributable to the implementation rather than the bed. Two arms differ in one argument, `full`
against `bigram_only` (same misnomer; `pos_unigram_weight: 1.0` appears in each arm's resolved config
dump, e.g. `GuaTrainJob.OfNoESzNJykY/output/train/0/hydra_train.log`), 40,000 updates each with a
checkpoint every 2,000; a smoke arm and one alignment pass precede them and are discarded by
construction. Recognition: the selection split is decoded with each of the eighteen checkpoints the
trainer wrote, the checkpoint is pinned by entry 5's label-free metric, and only then is the scored
fifth decoded and scored; every decode is the released generation script through the conditional-gold
patch. The full arm additionally carries the reference's two relabeling passes. Each arm is also read
at update 40,000, an endpoint declared before the decode and pinned without consulting any metric.
Applied step is `lr * min(1, clip_norm / gnorm)`, clip_norm 20, lr constant 0.004, on the two update
windows both arms' surviving logs cover (999 and 58 epochs each; a timeout resubmit overwrote the full
arm's middle segment). Iteration 1, dev-other scored fifth, 572 utterances / 34,135 reference phones.
| iteration-1 arm | update read | PER | sub | ins | del | phones/utt | swap-control M2 | gnorm early / final | fully clipped early / final | applied step early / final |
|---|---|---|---|---|---|---|---|---|---|---|
| full loss, label-free pick | 2,000 | 1.6843 | 0.6046 | 1.0766 | 0.0031 | 123.7 | +0.0013 | 1,617 / 8,981 | 100.0 % / 100.0 % | 1.07e-04 / 3.64e-05 |
| full loss, fixed endpoint | 40,000 | 1.6828 | 0.6153 | 1.0650 | 0.0025 | 123.1 | +0.0039 | same arm | same arm | same arm |
| bigram only, label-free pick | 30,000 | 1.2449 | 0.6836 | 0.5547 | 0.0065 | 92.4 | +0.0079 | 1,256 / 20,395 | 50.7 % / 60.3 % | 5.59e-04 / 8.11e-05 |
| bigram only, fixed endpoint | 40,000 | 1.2409 | 0.6811 | 0.5535 | 0.0062 | 92.3 | +0.0107 | same arm | same arm | same arm |
| reference | -- | -- | -- | -- | -- | 59.7 | -- | -- | -- | -- |

Whole-run clipping reader (`GuaFixedStreamReadJob`, speech-llm `10bbd48`): the banked full arm reads
**6.752e-05 mean applied step over 2,858 epochs at 100.0 percent fully clipped**, independently
reproducing the planner's banked "about 6.8e-05".

SELECTION-GRID DEFECT AND ITS AMENDMENT (validity). The trainer writes no numbered snapshot where an
interval save lands on an EPOCH boundary, and the seed bed yields 14 updates per epoch, so updates
14000 and 28000 have only a `checkpoint_last` that is later overwritten -- verified absent at both
updates in BOTH arms (40000 is not epoch-aligned, so the final snapshot exists). The first ruling
(19-point grid, 14000 removed, blamed on the EDQUOT outage at update 14000) was WRONG and would have
crashed both iteration-1 sweeps at 28000; **the current grid is 18 points (14000 and 28000 removed)**,
derived from the epoch arithmetic rather than from observed holes, and the hole is symmetric across
arms so selection stays arm-fair. Iteration-2/3 grids need their own epoch arithmetic checked.

**Entry 8. The same four entry-7 decodes under a phone 4-gram beam search instead of per-frame argmax.**
The released generation script keeps emissions, batching and hypothesis writing; a patch anchor makes
its KenLM branch mirror fairseq's lexicon-free unit-LM branch (one dictionary word per phone, `KenLM`,
`LexiconFreeDecoder`), and emissions are log-softmaxed before decoding (the script's `no_softmax`
override is harmless for argmax, not for a beam summing acoustic and LM scores on one scale). Four
checkpoints x 12 grid points (lm_weight {0.5, 1, 2, 4} x sil_weight {-2, -1, 0}) x two splits: the
selection four-fifths for the registered label-free pick, the scored fifth for the read. Beam 50 plus
a beam-500 convergence probe on the fixed-endpoint arms; nothing is refit and no checkpoint is
selected. ANCHOR PIN (discharged, in `gua_lm_decode.py`'s docstring): the published 0.473 is argmax
currency, no LM, no self-training -- entry-8 numbers are a different currency with no like-for-like
published counterpart. RESULT, primary LM (SIL-free 4-gram, `CreateBinaryLMJob.hvZoC014xnIe`), beam
50, `GuaLmGridReadJob.SeNSdRhV1Wo3`; read every row as the registered TRIPLE (G10).
| arm | greedy PER | label-free pick (ANTI-SELECTING) | label-oracle best (upper bound) | grid PER range | hyp/ref phones at the oracle cell |
|---|---|---|---|---|---|
| bigram only, fixed endpoint | 1.2409 | 0.8481 (lm 0.5) | 0.8172 (lm 1) | 0.8172-0.8985 | 20836/34135 |
| bigram only, label-free pick | 1.2449 | 0.8520 (lm 0.5) | 0.8195 (lm 1) | 0.8195-0.8963 | 21082/34135 |
| full loss, fixed endpoint | 1.6828 | 1.5446 (lm 0.5) | 0.8444 (lm 4) | 0.8444-1.5446 | 17212/34135 |
| full loss, label-free pick | 1.6843 | 1.4805 (lm 0.5) | 0.8463 (lm 4) | 0.8463-1.4805 | 15949/34135 |

Error decomposition at the two ends of the full-loss endpoint arm's grid: at lm 0.5 still
insertion-dominated (sub 0.6466, ins 0.8950, del 0.0030, 64,586 phones emitted against 34,135
reference); at lm 4 insertions gone and deletions dominant (sub 0.3444, ins 0.0021, del 0.4979,
17,212 phones). The bigram-only arm shows the same trade at lower lm_scale. A pre-launch hand probe
(single cell, lm_weight 2 / sil 0 / beam 50, full loss fixed endpoint) read 0.9322 (sub 0.7624, ins
0.0979, del 0.0718) and is a mechanism check only, superseded by the table. Beam-500 convergence
probe, same cells, both fixed-endpoint arms: PER moves by at most 0.0195 in any cell (every delta
negative), while one-best agreement between the two beams runs 0.1136 to 0.4843. SENSITIVITY,
SIL-augmented 4-gram on the two fixed-endpoint arms (`GuaLmGridReadJob.I9lgMOqar8RO`): same shape,
differences small and in BOTH directions -- full loss best 0.8476 against 0.8444 (worse), bigram only
best 0.8145 against 0.8172 (better) -- so the vocabulary mismatch costs little and changes no
conclusion. USER-prompted length-matched swap control (2026-08-23; the registration asked for none):
| cell | own PER | length-matched swap PER | swap minus own |
|---|---|---|---|
| bigram only endpoint, greedy (banked) | 1.2409 | 1.2516 | +0.0107 |
| bigram only endpoint, lm 1 (best cell) | 0.8172 | 0.8252 | +0.0080 |
| full loss endpoint, greedy (banked) | 1.6828 | 1.6867 | +0.0039 |
| full loss endpoint, lm 2 | 0.9322 | 0.9488 | +0.0167 |
| full loss endpoint, lm 4 (best cell) | 0.8445 | 0.8461 | **+0.0017** |

**10. Entry 9.0: the identifiability gate -- does this objective prefer the reachable truth to the
strongest content-free answer?** One CPU read, no training and no GPU, of five configurations of
artifacts entries 5 and 7 already produced. Per paired batch of 640 audio utterances against 640 text
lines, the arm's own count statistics are computed for a ladder of explicit audio-side answers and
compared with the text side in two currencies: the RAW L1 the arms minimized, and a mass-normalized L1
(each statistic divided by its own total mass) which removes the irreducible token-mass constant and
is the currency the gate reads. **H = L(strongest content-free decoy) - L(best audio side the arm
could reach)**; the reachable-truth family is the perfect map and the oracle memoryless map ON THIS
SEGMENTATION, the content-free family is the trained decode (where a checkpoint exists), five
label-permuted oracle maps and five text-unigram draws at the arm's own per-utterance lengths; each
family contributes its LOWEST loss. H > 0 means the objective prefers the truth. Gate G9. All five
configurations are read on ONE pool (2,699 utterances, intersected over the three segmentations so H
is paired; the read job refuses to rank H when pool fingerprints differ), and before writing anything
the job asserts its bincount statistics against `count_statistics` element by element, its raw L1
against the arm's own `matching_loss`, and the pinned checkpoint's soft loss against the producing
trainer's own logged per-utterance band, on a real batch.
| configuration | segmentation (seg/s) | text side | best reachable truth | strongest content-free | H | sd | batches truth ahead | audio-free contrast (paired) | ceiling at perfect boundaries |
|---|---|---|---|---|---|---|---|---|---|
| c1 | entry-5 pooled (13.62) | as run | 5.6804 | trained decode 3.5683 | **-2.1121** | 0.0232 | 0/10 | -0.0752 +/- 0.0212, 0/10 | +0.2854 |
| c2 | entry-5 pooled (13.62) | length-matched | 5.6561 | trained decode 3.5845 | **-2.0716** | 0.0546 | 0/10 | -0.0541 +/- 0.0451, 1/10 | +0.2734 |
| c3 | released k-means runs (28.78) | as run | 7.7666 | unigram null 5.3320 | **-2.4407** | 0.0429 | 0/10 | -2.4407 +/- 0.0429, 0/10 | +2.0430 |
| c4 | pairwise-merged runs (14.46) | length-matched | 5.6052 | unigram null 5.5584 | **-0.0586** | 0.0362 | 0/10 | -0.0586 +/- 0.0362, 0/10 | +2.3013 |
| c5 | pairwise-merged runs (14.46) | as run | 5.6413 | unigram null 5.5618 | **-0.0891** | 0.0350 | 0/10 | -0.0891 +/- 0.0350, 0/10 | +2.2692 |

The audio-free contrast is a SEPARATE PAIRED read added 2026-08-25 after the planner's verification
(best text-unigram null against best reachable truth on the same batch). It coincides with H wherever
the null is the strongest content-free member (c3, c4, c5); on c1 and c2 the trained decode is
stronger, so H there is a statement about that DECODE and this column carries the audio-free claim.
Supporting measurements: gold phone rate on the same pool is 13.548/s throughout, so c4/c5 are the
rate-matched rungs and c3 is over-segmented by 2.12x; boundary F1 against gold at +/-20 ms is 0.7384
pooled, 0.6193 raw released (precision 0.4555 at recall 0.9674 -- the rate is wrong, not the
placement), 0.7542 pairwise-merged; the length-matched text resample costs total variation 0.0125 at
the phone unigram and 0.0398 at the bigram on c2, 0.0135 / 0.0368 on c4.

**REGISTERED SECOND READ NOT DELIVERABLE, and the reports say so on every page rather than showing a
flat curve:** H against training update cannot be run -- entry 5 retained only its pinned checkpoint
(`resume.pt` went with the job's automatic cleanup) and entry 7's eighteen need the released generator
on a GPU. **The STOPPING-RULE question therefore stays OPEN**, the closure is QUALIFIED by the
registration's own terms, and no reading of this gate closes it. For the same reason only the two
pooled configurations carry a trained decoy, so on the released stream the content-free family is the
constructed nulls alone.

**11. Entry 9.3: the disclosure decode -- gate 9.0's own ladder, priced in phone error rate.** Five CPU
jobs, one per configuration, no training and no GPU. Gate 9.0 ruled in mass-normalized L1 units, not
the currency this program reports in, and three of its five configurations had never been decoded; the
only decode reachable without GPU training is the one the gate constructs internally -- the arm's real
boundaries under the best memoryless unit-to-phone map, H's own second operand. Each job reads gate
9.0's pool through gate 9.0's own methods and asserts the pool fingerprint (2,699 utterances,
`93e6ee25c009`, checked against that configuration's banked `identifiability.json` for fingerprint AND
count AND segmentation AND text mode) before scoring; each row is scored as a DECODE (adjacent
duplicates collapsed within a chunk, asserted on 64 real utterances against `espum_jobs._decode_ids`)
with `quantize_states.phone_error_counts`. Currency: plain PER as scored, greedy per-segment argmax,
NO language model -- like-for-like with the published TIMIT 0.473 anchor and NOT with entry 8's
LM-decoded column. No bar; the disclosure funds nothing. Every "perfect" or "oracle" row READS GOLD
and is a CEILING, never an achievable result; the only label-free row is the trained entry-5 decode,
the only audio-free row the unigram null; read per configuration, nothing pooled or ranked.
| configuration | segmentation (seg/s) | perfect bnd + perfect map (CEILING) | real bnd + perfect map (CEILING) | real bnd + oracle map (CEILING) | trained entry-5 decode (label-free) | text-unigram null (audio-free) |
|---|---|---|---|---|---|---|
| c1 | entry-5 pooled (13.62), as run | 0.0503 | 0.2545 | **0.4168** (h/r 0.840) | **0.8522** | 0.8897 +/- 0.0007 |
| c2 | entry-5 pooled (13.62), length-matched | 0.0503 | 0.2545 | **0.4168** (h/r 0.840) | **0.8522** | 0.8907 +/- 0.0006 |
| c3 | released k-means runs (28.78), as run | 0.0503 | 0.1071 | **0.8370** (h/r 1.529) | -- | 1.6071 +/- 0.0005 |
| c4 | pairwise-merged runs (14.46), length-matched | 0.0503 | 0.2307 | **0.5421** (h/r 0.900) | -- | 0.9084 +/- 0.0013 |
| c5 | pairwise-merged runs (14.46), as run | 0.0503 | 0.2307 | **0.5421** (h/r 0.900) | -- | 0.9075 +/- 0.0011 |

Reading rules that travel with this table:
- **FLOOR 0.0503.** The perfect-boundaries perfect-map row is 0.0503 dev-other on all five
  configurations and pure deletion (sub 0.0000, ins 0.0000): collapsing adjacent duplicates within a
  chunk merges phones the speaker genuinely repeated. That is the currency's own cost, identical
  across configurations, so 0.4168 sits 0.366 above the floor and 0.5421 sits 0.492 above it.
- **THESE CEILINGS ARE MEMORYLESS AND DO NOT BOUND A TRAINED ARM.** Each oracle row is the best map
  assigning one phone per unit id with no context, so an ordering of two ceilings orders MAP quality.
  The trained arm reads 0.8522 against a 0.4168 ceiling on the same stream, so what it achieves is set
  by that 0.435 gap, which is unmeasured on the merged stream; and E1 reads 0.3565 against this
  stream's 0.4148 memoryless ceiling, the expected pass for a richer function class. So 0.5421 prices
  the released 128-cluster INVENTORY against entry 5's 500-unit codebook and prices nothing about a
  context-carrying generator on the merged stream. Any citation of verdict 46 carries this scope.
- **THE ROWS ARE GATE 9.0's RUNGS, SCORED AFTER THE COLLAPSE A PHONE ERROR RATE REQUIRES.** Gate
  9.0's L1 reads the segment stream with repeats intact (its adjacent-repeat diagnostic is 0.2134 on
  the perfect-map audio side); this table collapses first. Both sides of every comparison get
  identical treatment, so the readings are fair as measured, but the two tables score the same RUNG,
  one before and one after the decode step -- not the same SEQUENCES. This matters most on c3, where
  2.12 segments per gold phone leave the collapse doing almost all the work.
- All numbers dev-other; dev-clean runs 0.02-0.04 lower on every row. c1/c2 share an audio side (and
  c4/c5 likewise), so their audio rows are identical by construction -- design, not reproduction. The
  null spread is over five draws. The trained decode's decomposition on this pool: sub 0.6859, ins
  0.0520, del 0.1144 at 58,944 emitted against 62,866 reference phones.

## Verdicts

Entry 8 (cells 1-2):

- **E8.1.** LM decoding cuts the entry-7 arms' PER by a large margin, mostly by emitting fewer phones
  (best cell against greedy at the same checkpoint). The best cells emit 47-62 percent of the
  reference phone count and their error is deletion-dominated. The registered mechanism (an
  insertion-dominated rate attacked by a language model) is CONFIRMED as a mechanism and does NOT
  deliver a usable decode.
- **E8.2.** The second grid axis is INERT on this decoder: the three `sil_weight` values move PER by
  at most 8.8e-05. (AMENDED 2026-08-23 on the verifier's catch: the verdict first said "identical in
  every cell", which is false -- decodes differ at one `lm_weight` point per read by up to 8 emitted
  phones out of ~17,000; the within-lm_weight spread re-derived independently from both artifacts is
  8.79e-05 primary / 5.86e-05 sensitivity. "Inert" is the claim, "identical" was an overstatement.)
  Cause: the entry-7 generator's vocabulary has no silence symbol, so fairseq's index rule falls
  through to end-of-sentence and `sil_score` is charged on a token the decoder never emits. The grid
  that ran is 4 points, not 12; every 12-row table is 4 distinct decodes repeated three times.
  RETIRED by the planner 2026-08-23.
- **E8.3.** The registered label-free selector fails, in the OPPOSITE direction to the disclosed one:
  it picks lm 0.5 -- the LOWEST -- in all four arms, because weighted phone-LM perplexity per token
  rewards the long, insertion-heavy decode (sel wppl 43.03 at the picked lm 0.5 against 431.14 at the
  oracle cell on the full endpoint arm). Ranked over DISTINCT lm_weight points it picks the worst
  point on both full-loss arms (5 of 5 and 4 of 4) and the third of four on both bigram-only arms.
  CONSEQUENCE: no standalone entry-8 number exists (G10). The `abc3d81` perplexity fix does NOT
  rescue it -- the anti-selection is per-token perplexity not paying for length at all (first-order),
  while that normalizer is a 1-3 percent second-order correction.
- **E8.4.** Beam 50 is converged for the RATE and not for the SEQUENCE (PER within 0.0195, one-best
  agreement as low as 11.4 percent) -- the same signature §1g's decoder showed at its beam doublings.
  No beam escalation is warranted.
- **E8.5.** The LM-decoded gain is NOT content: scoring a hypothesis against a DIFFERENT utterance
  costs 0.2 to 1.7 percent of error rate, and the BEST cell by error rate has the SMALLEST swap margin
  (+0.0017). Decoding cut the full-loss arm from 1.6828 to 0.8445 while leaving the swap margin where
  it was -- the shape to expect if the gain comes from emitting a shorter, more phone-typical string.
  For the reference "we gazed for a moment silently into each oth-", the best bigram-only cell emits
  `S W T IH IY D IH DH N IH AO R B AH R IY IH Z`. CONSEQUENCE: no entry-8 cell may be described as a
  decode OF anything. Untouched: cell 4, which asks the same question against the registered nulls in
  the registered currency.

Numbered conclusions (approach in parentheses; tables above carry the numbers):

1. (1) **Kill condition (i) FAILS G1** at oracle-map PER 0.832 dev-other / 0.712 dev-clean, a harder
   cap than the 0.53-0.63 of the §0a inventories that closed §1a(i).
2. (1) The cap is **over-segmentation, not confusability**: substitutions 0.115 / 0.132 against the
   superseded codebook's 0.185 / 0.195 and §0a's k-means-500 0.245, insertions 0.591 / 0.692 at 2.79
   deduped units per gold phone. Frame-level phone information is the highest this program has
   measured (PNMI 0.682, H(phone|unit) 1.046 nats against H(phone) 3.292).
3. (2) §1a conclusion 6 holds operationally but **not literally**: the boundary-crossing graph carries
   the bigram (PMI rank correlation 0.515 / 0.517 against a 0.03 floor), but a matcher does not see
   boundaries, and on the observable graph the correlation falls to 0.373 / 0.370 between a floor of
   0.214 / 0.216 and a ceiling of 0.413 / 0.398.
4. (2) The matcher's objective barely separates truth from noise even at the oracle map: an 11 %
   relative `tv_offdiag` span on dev-clean with the real stream 57 % of the way floor to ceiling;
   dev-other 8.9 % and 46 %.
5. (2) **Coarticulation is measured, not assumed**: 26.1 % / 26.7 % of genuine transitions project
   onto the same phone against 4.4 % / 4.5 % when segments are drawn independently of their
   neighbours -- a 5.9x inflation, so a quarter of all transitions are invisible to the graph. The
   identifiability precondition fails by a measured amount.
6. (3) **Kill condition (i) is CLEARED by data-driven segment pooling**: every rung passes G1 on
   dev-other (0.414 / 0.452 / 0.481), the best measured on any inventory, so the arm stays at the UNIT
   level and the registered FEATURE-level redirect is not needed. Not a label-chosen rung: the gold
   phone rate is 9.8 / 9.4 per second, so the label-free rate-matched rung `seg9` passes on its own.
7. (3) The cap was the token RATE alone -- at fixed rate, coarsening the INVENTORY is catastrophic
   (`brown100` PER 1.063 / 1.152, PMI 0.109 / 0.103 against raw's 0.373 / 0.370) -- so the 500-way
   codebook's discriminability is load-bearing.
8. (3) Ladder entry 2 (ridge positional-unigram estimator) fails G3 everywhere -- sigma_min 5e-33 raw,
   exactly 0 wherever usable positions are fewer than units, 2e-17 on `brown100` (K=100 over 495
   position rows) -- and a SIMULATED perfectly recoverable channel reads 0 too, so the failure is
   structural. Entry 2 CLOSED.
9. (3) The `tv_offdiag` bar cannot be read as written: span term 3.8-11.4 % against a 25 % bar, and
   the position term exceeds 1 on every pooled row because the real stream beats the `seg_swap`
   ceiling -- coarticulation pushes correlated errors onto the DIAGONAL, so a channel erasing a
   quarter of its transitions scores BETTER on the matcher's objective than one that factorizes by
   construction. See G2 (VOID AS MEASURED).
10. (3) Where inversion does not confound, pooling buys real signal: transition-carrying adjacent
    pairs 0.347 -> 0.700, observable-graph PMI 0.373 -> 0.595 against a floor also rising 0.214 ->
    0.465, and the real stream's gap to that floor triples (6.2 % -> 21.2 %).
11. (3) Against simulated channels of known recoverability at MATCHED fertility the pooled stream is
    still worse than 35 % random emissions (`seg12.5` tv 0.322 / PMI 0.632 on the 500-utterance screen
    against 0.298 / 0.765 for fertility 1 at 35 % noise): pooling fixed the rate and left the emission
    ambiguity severe by construction-calibrated standards.
12. (3) The string map earns its complexity only for unit-BPE (0.538 -> 0.436 dev-other), buys nothing
    pooled (0.414 -> 0.435), and on raw only converts insertions into deletions (0.692 -> 0.646) -- a
    context-free token map cannot repair a rate error.
13. (4) **Ladder entry 3 FAILS G4 on every representation**: best margin +0.015 (`seg12.5` dev-other)
    against 0.05, and on `raw`, `seg16` and `ubpe12.5` it loses outright to n2. NOT FUNDED.
14. (4) The audio-swap control, anchored on the oracle map (0.372-0.596) and the random null
    (0.004-0.020), places the candidate at 0.9-4.6 % of the content-dependence span and at or below
    the random null's own movement on `seg9`: by the registered control the map is content-free.
15. (4) Both G5 kill-tests fail: manner separation 0.41-0.49 against ~0.50 over majority baselines of
    0.34-0.47, admitted-pair precision 0.00-0.29 against ~0.70, with the bootstrap admitting nearly
    the same set with and without the CSLS margin (10 vs 11 on `seg12.5`) and starting at chance from
    a PERFECT ten-pair seed, so the recounting step never engages.
16. (4) The failure is diffuseness, not a broken solver: the true phone is in the top five for
    0.23-0.32 of unit mass against a chance 0.128 (reciprocal rank 0.16-0.20 against 0.109) -- about
    twice chance, far short of what a 39-way many-to-one assignment needs.
17. (4) Frequency is a nuisance column, not the signal -- dropping it improves the solve on every row
    (0.882 -> 0.866 on `seg12.5` dev-other) while frequency alone is the worst of the three (0.885) --
    so the §1c single-marginal warning holds in the stronger form that the marginal hurts.
18. (4) CAVEAT on the ceiling this arm quotes: the registered frame-argmax oracle map is not
    PER-optimal on an over-segmented stream (forcing silence-proxy units to SIL beats it by 0.095 on
    `raw`, 0.773 -> 0.678), while on the pooled rungs the two agree within 0.026 either way -- the
    pooled ceilings stand, only the raw ceiling is loose.
19. (5) **Entry 6's kill-test does NOT kill**: at every granularity coarser than the bare unit at
    least one top-20 unit-word carries the G6 signature on `seg12.5` (peak 3) and on `ubpe12.5` (peak
    2), against the 2 the identical rule finds among the top 20 English words -- so ruling 3's
    lexicon-free text arm keeps its precondition. **CORRECTION (approach 6, 2026-08-16): the
    `ubpe12.5` half is WRONG** -- its hit at @0.5 and @1 is the all-silence unit 397, so that rung
    carries a non-silence hit at @0.75 only; the `seg12.5` half stands.
20. (5) The signature is positional only, not distributional: every hit is a single UNMERGED unit
    whose non-initial occurrences the merges absorbed into longer tokens, the top unit-word holds
    0.006 of tokens against THE's 0.061, and the unit-word Zipf slope is -0.90 to -1.01 against -1.39
    for words -- function-word POSITION without function-word MASS.
21. (5) The granularity curve turns over inside the swept range (hits 0/1/3/2/1 on `seg12.5`, 0/0/1/2/1
    on `ubpe12.5`), so the finest attainable rows (4.26 and 7.29 unit-words/s against the 2.8/s
    target) sit past the peak and do not understate the hit count.
22. (5) The hits are 8.5-14.9x their positionally indifferent base against 4.3-5.7x for the English
    hits -- MORE committed to the utterance onset than a real function word is, which no label-free
    read can separate from an utterance-onset acoustic effect. **CORRECTION (approach 6, 2026-08-16):
    the 14.9x end is WRONG as evidence about function-word-like units** (it is `ubpe12.5`'s
    all-silence 397); `seg12.5`'s 8.5x survives, and the clause stands since approach 6 separates them
    with labels.
23. (5) **`seg12.5` cannot be BPE-compressed to word rate at all, and the reason is recurrence, not
    budget**: 720315 tokens against an 800000-token fitting cap, stopping at 38228 merges against a
    50000 cap, so the loop exited on its no-pair-occurs-twice break -- recurrence is exhausted at 4.26
    unit-words/s, still 1.5x the English word rate. (`ubpe12.5`'s 7.29/s stall IS the merge cap, hit
    exactly at 50000.)
24. (6) **On `seg12.5` the signature is linguistic, not an utterance-onset effect**: the hit present at
    every prefix (403) reads AH -> AH on phones and THE -> THE on words, `TV_ie` 0.074-0.094 (phones)
    and 0.242-0.245 (words) against other-type medians of 0.291 and 0.915, and sits far from the
    corpus's onset mixture (`TV_i_onset` 0.65-0.66 against the other-type 0.384).
25. (6) **On `ubpe12.5` the signature is an artefact of a missed silence unit**: 397 has every frame it
    occupies labelled silence (`off_init` = `off_else` = 1.000), and since entry 6 builds segments by
    DELETING proxy-silence units, 397 reaching the merge list proves the label-free edge-enrichment
    proxy did not call it silence (ratified as proven by construction: `segment_tokens` deletes masked
    types before any merge). The one genuinely linguistic ubpe hit is 608 at @0.75.
26. (6) A hit can be positionally and phonetically stable without being a word: `seg12.5`'s 423 and 432
    hold one phone at both positions while their word reads disagree across positions (IT -> BUT, IN
    -> AND, `TV_ie` 0.715 and 0.630), so of the three `seg12.5` hits only 403 supports the
    function-word reading.
27. (7) **Ruling 3 FAILS G4 in all twelve cells**: M2 never exceeds 0.0252 and M1 is negative in 10 of
    12; the two positive M1 cells are both the phone reference side. NOT FUNDED; neither conditional
    follow-up triggered.
28. (7) The train-side corpus asymmetry did NOT decide it: `ubpe12.5` runs on 4.1x the utterances and
    5.4x the audio and is still the worst rung on every text side, so the corpus-matched `ubpe`
    control is not triggered. SCOPE GUARD (2026-08-17): the basis is the per-rung gate structure plus
    the all-fail table; read as representation attribution the "worst despite more data" clause is
    confounded by K=8500-vs-500 and the budget-stopped stream.
29. (7) The `words` cells price a rate-MISMATCHED arm: no stream reached the 2.8/s target, the mismatch
    shows up as candidate insertions (0.717 on `ubpe12.5` against 0.004 deletions), and a budget large
    enough to reach the target could only be spent on `ubpe12.5`, whose merges stopped on exhausted
    pair repeats at 6.45/s.
30. (8) **E1 PASSES G11** at 0.3565 dev-other against the 0.4148 ceiling, so the input and pooling path
    carries at least ceiling-level phone information and a poor unsupervised read cannot be charged to
    it. A supervised eval-only read, not an entry-5 performance claim.
31. (8) **Entry 5 FAILS G4 on both clauses**: the label-free-selected candidate reads 0.8580 dev-other
    (M1 +0.0365 against the 0.8446 bar, M2 +0.0466 against 0.05), so the only unkilled ladder entry
    closes and 1f returns to the user with none.
32. (8) The bigram-only ablation did NOT separate from the full loss: at 0.8748 it beats two of three
    full-loss seeds, and the seed spread (0.8580-0.8848) exceeds its 0.0015 gap to the ablation.
    **FRAME WRONG (verifier 2026-08-25, numbers unaffected):** the arm keeps the positional unigram, so
    it is the paper's `uni+bi` row (39.2), not `bigrams only` (71.6); the expected separation for the
    pair actually run is 0.8 PER points, not 32.4, so the absence of separation is what the reference
    predicts.
33. (8) The arms fail on IDENTITY, not rate or collapse: 0.963-0.974 of the reference phone count, all
    39 phone types used, 79-81 % of error as substitutions. (SUBSUMED by verdict 50; not the operative
    cause on its own.)
34. (8) Training-based matching beats the unary fingerprint solve on the identical rung, text side and
    fifth (0.8580 against 0.8809, both margins roughly tripled) -- which prices what the added
    machinery buys and leaves it short of the bar by more than it gained.
35. (9) **The entry-7 stage-A signature is ABSENT and its sign is REVERSED**: bigram-only minus
    full-loss PER is -0.4394 where G7 asked for at least +0.10. **WRONG as a test of the reference's
    claim (verifier 2026-08-25); the -0.4394 itself stands.** The arm's resolved config dump carries
    `pos_unigram_weight: 1.0` beside `skipgram_size: 1`, so it is `uni+bi`, published separation 0.8
    PER points -- one twelfth of the bar and of entry 5's seed noise. The clause was unfirable by
    construction on any bed; the measured -0.4394 is an observation about two of our own arms only,
    and the true collapse arm (`model.pos_unigram_weight=0.0`) has never been run (registered as
    entry 9.1 A9b, not built).
36. (9) That comparison is **not interpretable as a contrast between the two losses**: the label-free
    selector shows no signal on either arm and pinned them 28,000 updates apart -- weighted phone-LM
    perplexity spans 38.16-41.15 over the full arm's eighteen checkpoints and picked update 2,000, the
    FIRST one, against 30,000 for bigram-only. (Objection retired by verdict 40.)
37. (9) Both arms **over-generate massively** (123.7 and 92.4 phones per utterance against the
    reference's 59.7), so PER is insertion-dominated and the arm that emits more is mechanically the
    worse-scoring one. **The second clause is WRONG**: substitution-only (0.6046 against 0.6836) does
    not restore the predicted direction, it selects the length-favouring metric -- recall 0.3923
    against 0.3099 favours full, precision per emitted token 0.1892 against 0.2002 favours
    bigram-only, both artifacts of the 2.07x-against-1.55x length ratio. No decomposition of this pair
    of decodes carries a direction.
38. (9) The **audio-swap control is flat on both arms** (+0.0013 and +0.0079): no measurable
    utterance-specific information above a mismatched pairing -- though with hypotheses 1.5-2x longer
    than the reference the edit-distance match is largely length-driven, limiting the control here.
39. (9) Both arms ran at a **declining applied step under a constant declared learning rate** of 0.004
    (1.07e-04 -> 3.64e-05 full; 5.59e-04 -> 8.11e-05 bigram-only) because gradient norms rise 5.6x and
    16x into a fixed clip ceiling of 20. The arm expected to win applies the smaller step in both
    windows -- the direction that MASKS a real difference rather than manufacturing one. (Provenance
    of G8's 1.00 edge.)
40. (9) The fixed-endpoint read **reproduces the reversal at a common update with nothing in the
    selection loop**: -0.4419 at update 40,000 against -0.4394 at the selected checkpoints, so the
    28,000-update spread of verdict 36 is NOT what produced the sign and that objection is retired;
    over-generation is likewise stable across the run.
41. (9) **Stage A closes NOT ANSWERABLE** on the pre-registered interpretability condition: at the
    fixed endpoint both arms sit far above the 0.8446 arm-gate margin and both audio-swap controls stay
    flat (+0.0039 and +0.0107 against 0.05), so the contrast is between two uninformative decodes
    rather than between two losses. **SCOPE AMENDED (verifier 2026-08-25); the closure itself stands as
    applied** -- the cause it left open ("execution versus approach on this bed") is now measured and
    is neither (verdict 50).
42. (10) **GATE 9.0 FAILS ON ALL FIVE CONFIGURATIONS**: H negative everywhere, truth ahead in 0 of 10
    batches on every one, against G9's +0.05 and 7 of 8. On the registered reading that CLOSES the
    fixed low-order statistics-matching family ON THIS BED by measurement -- the consequence the
    registration pre-declared and the honest discharge of USER ruling 6. It licenses "not funding this
    family here"; it does not say the family could not work on another bed. Entry 9.1 was NOT licensed
    by it (9.1a runs only as an explicit USER override).
43. (10) **AN AUDIO-FREE NULL BEATS THE BEST REACHABLE TRUTH ON EVERY CONFIGURATION.** A sequence drawn
    i.i.d. from the text phone unigram at the arm's own per-utterance segment counts -- no audio, no
    units, no map -- scores 5.6230 against the truth's 5.6804 (c1), 5.3320 against 7.7666 (c3) and
    5.5584 against 5.6052 (c4). The objective's optimum on this bed is the phone marginal at the right
    length, and the truth is not it. **PAIRED READ ADDED 2026-08-25** (the c1/c2 half had rested on
    unpaired per-rung means): the verdict stands on all five, with the one qualification that c2 is 9
    of 10 batches rather than 10 of 10.
44. (10) **THE RATE REPAIR CLOSES 97 PERCENT OF THE GAP AND DOES NOT CHANGE THE SIGN.** The canonical
    `merge_clusters` + `mean_pool --subsample-rate 0.5` preprocessing entry 7 skipped moves the stream
    from 28.78 to 14.46 segments per second (gold 13.55) and H from -2.44 to -0.09, boundary F1 0.6193
    -> 0.7542 -- but H stays negative in 10 of 10 batches at a spread of 0.035, so the missing step is
    a real defect of entry 7 and is NOT what decided it. Length-matching the text side is worth a
    further +0.03 (c5 -> c4) and is likewise not decisive.
45. (10) **WITH A PERFECT SEGMENTER THE TRUTH WINS**, by little on the pooled stream and much on the
    merged one (ceiling H +0.2854 c1, +2.0430 c3, +2.3013 c4). The objective is not blind to
    transcription content -- the SEGMENTATION-induced audio side is what costs the truth its win, and
    0.29 is the whole margin a perfect segmenter would buy on the pooled stream.
46. (11) **THE CORRECTED MERGED STREAM PRICES WORSE AT ITS CEILING THAN THE STREAM IT CORRECTS, 0.5421
    AGAINST 0.4168 dev-other**, even though its matching-objective gap to the content-free answer is 24
    times smaller (H -0.09 against -2.11). The rate repair moves H; it does not buy a better memoryless
    unit-to-phone map, because the released 128-cluster inventory is coarser than entry 5's 500-unit
    codebook. The repair that most helps the objective is not the stream with the best reachable
    decode. SCOPE: ceilings are memoryless and order MAP quality, not achievable PER.
47. (11) **OVER-SEGMENTATION IS FREE UNDER A PERFECT MAP AND RUINOUS UNDER A MEMORYLESS ONE.** On the
    raw released runs real boundaries with a perfect map read 0.1071 -- the LOWEST real-boundary row of
    the three streams, because 2.12 segments per gold phone collapse back once duplicates merge --
    while the same boundaries under the best memoryless map read 0.8370 at 1.529 hypothesis phones per
    reference phone, insertion-dominated (ins 0.5428, del 0.0135). The cost of the wrong rate is paid
    entirely at the map.
48. (11) **THE INVERSION GATE 9.0 MEASURED IS AN INVERSION IN THE OBJECTIVE, NOT IN PHONE ERROR RATE.**
    The audio-free null the objective PREFERS (43) scores 0.8897-1.6071 PER while the audio side it
    beats scores 0.4168-0.8370: every ceiling row beats every null row on its own configuration, by
    0.35 to 0.80. The objective is not measuring what a decoder is measured by, which is why the
    closure rests on the objective and not on the streams being empty. NOTE: an arm-level comparison
    on the same 2,699 utterances, exempt from the paired-data rule on magnitude alone (smallest gap
    0.35 against a null draw spread of 0.0007); any FUTURE 9.3-style comparison at a margin near the
    noise must be paired.
49. (11) **THE TRAINED ENTRY-5 DECODE RE-READS 0.8522 dev-other ON GATE 9.0's POOL AGAINST THE BANKED
    0.8580 ON ENTRY 5's OWN SCORED FIFTH.** Same checkpoint, same decoder, 2,699 utterances instead of
    572; the 0.0058 difference is the utterance set. Nothing is selected and entry 5's gate does not
    move.
50. (audit 2026-08-25 (E); supersedes the autopsies in 33 and 41) **95-97 PERCENT OF THE MATCHING
    OBJECTIVE ON THIS BED IS AN IRREDUCIBLE CONSTANT.** Audio 179.57 segments per utterance against
    text 70.12 phones per line, ratio 2.54, against the reference bed's measured 1.022; the arms' own
    logged `loss_dense_g` sits 2.8-5.4 percent above that floor, i.e. both runs converged INTO the
    degenerate set whose only requirement is the phone marginal. In the arm's OWN loss at matched mass
    on gold-covered utterances the trained entry-5 decode scores 145,818 against the gold transcript's
    233,063 in 10 of 10 paired batches. Entry 5 did not fail to optimize. Our `seg12.5` boundaries are
    F1 0.762 at +/-20 ms -- the RATE right, the PLACEMENT wrong. **AMENDMENT (2026-08-25, same
    auditor):** "the truth only TIES the decoy at perfect boundaries" is SUPERSEDED -- on the
    registered paired pool the truth WINS there by 0.11 (3.4566 against the decoy's 3.5683) and by
    +0.2854 against the true reference sequence; the audit's banked c1 H of -1.77 is superseded by the
    registered -2.1121 (same ordering, same sign, every rung 0.15-0.49 higher on the intersected
    pool). Gate 9.0's registered values are the current ones.
51. (audit 2026-08-25 (4)) Two reported constants corrected in the frame, not in a number: the k-means
    proposal is **2.10x** the gold phone rate, not 2.9x (9.86/s was `WORD_RATE_HZ = 2.8` times 3.52
    phones per word, never a measured phone rate; the matched-basis gold rate is 13.652/s
    silence-stripped, 9.715/s full-audio); and the learned segmenter does NOT prune -- paired on the
    same 572 utterances the predicted rate is 126.916 (full) and 125.306 (bigram-only) against a
    125.149 proposal, r = 0.9996-0.99997, moving UP on every arm and checkpoint. The entire length
    difference between the arms is duplicate collapse at decode (3.0 versus 26.3 percent of adjacent
    segment pairs), not segmentation.
52. (audit 2026-08-25 (3)) Recorded as a negative result so it is not re-audited: `count_statistics`,
    `tri_skip_pairs`, `matching_loss`, logit-pooling-then-softmax, the smoothness term on post-pooling
    logits, the batch tail truncation, the zero-transition Viterbi identity, the conditional-gold patch
    and the edit-distance scorer were compared line by line against the release and are FAITHFUL.

The kill conditions, the battery, entry 3, ruling 3 and entry 6 are reported, not acted on: entry 6's
kill-test clears so the lexicon-free arm may be funded, with conclusions 20 and 22 bounding what it
establishes.

## Standing constraints and unresolved findings

- **Not built, and the constraint that follows.** Entry 8 cell 3 (CTC-student decoder-sanity control),
  cell 4 (re-banking the argmax nulls and the memoryless oracle-map ceiling under this decode) and
  cell 5 (the entry-5 ESPUM checkpoints). **Until cell 4 exists no margin against 0.8946 / 0.9239 /
  0.4148 may be quoted** and the "stage A answerable after all" trigger cannot be evaluated. Cells 3-4
  stay with the user; E8.5 raises cell 4's stakes.
- **The stopping-rule question is OPEN** (approach 10), so gate 9.0's closure is qualified.
- **Text-sample coverage defect (registered fix 7b); affects entry 3 and all twelve ruling-3 cells,
  NOT entry 6.** `sample_text_lines(stride=80, max_lines=300000)` stops at source line 23,999,921 of
  39,630,169 (60.6 %), and `librispeech-lm-norm` is ALPHABETICALLY sorted by sentence, so sentences
  beginning roughly P-Z never entered those screens' text-side statistics -- fingerprints, marginals,
  n1 quotas and n2 pairing draws alike. Candidate and both nulls consumed the same truncated sample,
  so every gate comparison stays internally consistent and the all-fail verdict is NOT reopened; the
  defect must be disclosed wherever those text statistics are quoted. Entry 6's text row (stride=400,
  max_lines=100000) is unaffected. Standing rule: every future text-side statistics pass must satisfy
  stride*max_lines >= corpus lines.
- **Hash drift orphans entry 5's own runs.** Reconstructing `EspumMatchTrainJob` with entry 5's own
  arguments yields `EspumMatchTrainJob.LJnNQh8wVbVu`, NOT the banked `.lALR9ldNG8f1` (the constructor
  gained the H3 arguments after entry 5 ran, and sisyphus hashes the full bound signature). The first
  graph load had that unfinished job runnable, one `-r` away from funding a fresh 32-GPU-hour
  training. Consume the checkpoint as a frozen path; the artifact list below is the only protection
  for those job dirs.
- **Pooling trap, unresolved and potentially live for 9.1a.** Under entry 7's l1 config (BINARY
  segmentation) the boundary head pools label-free at training and generation alike, and entry 7's
  data dirs carry no `.src` file; but the bundle's FIXED/JOIN branches DO silently change pooling when
  `bin_labels` is absent. Unreachable for entry 7, load-bearing for any segmentation-type change --
  and entry 9.1a ships `segments/<split>.src`, with nothing in this document recording which branch
  its arm takes.
- Verified-and-closed defect fixes are not logged here; the only one that changes how a banked number
  reads is the `abc3d81` perplexity convention (approach 8 CURRENCY NOTE).

## Artifacts

| entry | config | key jobs |
|---|---|---|
| prerequisite (approaches 1-2) | `config/sae_1f_prereq.py` | codebook `QuantizeStatesJob.FWpGhC941JMi`; unit stream `MergeUnitsPklJob.ncxcd3vouD5E`; superseded codebook `QuantizeStatesJob.c5H3nY2G1VIz`; audits `AuditAvUnitsJob.hPXeQoupqWBa` (dev-clean) / `.GFdWI6Kzfhkr` (dev-other); channel structure `ChannelStructureJob.a2QG6jbb5Fzp` / `.G98AobA396ha`; 12.5 Hz rate-reduced comparison rows `AuditAvUnitsJob.zzZk9wq8vBfe` (oracle PER 0.423, ins 0.013, del 0.287 under its own grid-rasterization caveat) |
| battery (approach 3) | `config/sae_1f_battery.py` | features `AvStatesJob.c4Ak1rACchRC`; pooled streams `SegmentPoolUnitsJob.IHRNqQfnxrQ3`; `brown100` / `ubpe12.5` `RelabelUnitsJob.BA6DGayY7B5F` / `.mkk17SxDKjG2`; audits `AuditAvUnitsJob.LhqtknKw7dzh` / `.KBO9vGKFDjPT`; screens `MatchScreenJob.8Dw01CNHAXYZ` / `.tmCr93GgmkVH`; per-representation channel structure dev-clean `ChannelStructureJob.zoS0O0t0pB5Z` (seg16) / `.m06rcKTBVYuq` (seg12.5) / `.Pvg3LvrVVgU4` (seg9) / `.x0kMX4nLb0Rb` (brown100) / `.seXXeYp8aOyZ` (ubpe12.5), dev-other `.BxyUz8Fha84d` / `.Xf4J9E9gNiz4` / `.hisUE5DAz6EF` / `.NgWhyKTWYYMV` / `.bCFN8Q1pORtz` |
| entry 3 (approach 4) | `config/sae_1f_entry3.py` (+ `_fp`) | `FingerprintMatchJob.O4dpJTesB66u` / `.MHmUIV85g8Ry`; text `TextToPhonemeJob.THKMON3k9LJQ`; gold phones `GoldPhonesJob.ZGSp0hxyd2YP` |
| entry 6 (approaches 5-6) | `config/sae_1f_onset.py` | `UnitWordProfileJob.vULzsMp1oise`; onset control `OnsetControlJob.wGkGvMmpWF5V`; word text `DownloadJob.g4jClO48cAvP`, BPE-512 `ReturnnTrainBpeJob.su17F7YEcwEr` |
| ruling 3 (approach 7) | `config/sae_1f_ruling3.py` | streams `UnitWordStreamJob.jwEGIPgoOuy5` (seg9) / `.mPnLApAbYnVG` (**seg16** -- 43821 merges, fit 360623, kept 906940) / `.eIxgmMh99RSE` (**seg12.5** -- 38228 merges, fit 313918, kept 720336) / `.5XnmEvOqh0TK` (ubpe12.5); gate reads `LexFreeMatchJob.PQGETAeQAVaZ` (seg9) / `.rk48Zk5U6jzW` (seg12.5) / `.iY14L0buio5T` (seg16) / `.qw7Q0eDiq2hW` (ubpe12.5). **The two stream hashes were SWAPPED in the original Catalog** (correction 2026-08-17); the `LexFreeMatchJob` labels and their stream wiring are correct |
| entry 5 (approach 8) | `config/sae_1f_entry5.py` | E1 probe `EspumProbeJob.5KJjR2SsYBJT`; training arms `EspumMatchTrainJob.mIXXRFodAMKs` (full s0) / `.lALR9ldNG8f1` (full s1) / `.ydfMFa6NdL4f` (full s2) / `.zokDElC71cF9` (bigram-only s0); eval `EspumEvalJob.LPVdtT35Jzzw` / `.WdKc8iOgFSzP` / `.k9x16RkG3w5M` / `.uvJWVebI6ffu`; pick `EspumPickJob.W9HzeOEviPO4`; selection ids `EspumDevIdsJob.IqaCdokey92g`; phone 4-gram `KenLMplzJob.0aJeN88X6EdW` -> `CreateBinaryLMJob.hvZoC014xnIe` |
| entry 7 (approach 9) | -- | chain `GuaAudioManifestJob.rdVx8r37h78h`, `GuaFeaturesJob.RIxCDjD5XPqW`, `GuaClusterJob.baYUouLc37Ay`, `GuaTextJob.4mttbvA9Ut8f`; smoke `GuaTrainJob.3xMSCoAtUuDz` + `GuaGenerateJob.RsHiB5ueKWdz` (discarded by construction); arms `GuaTrainJob.PZo12D74ij2M` (full) / `.OfNoESzNJykY` (bigram-only). Iteration-1 recognition (DERIVED 18-point grid, re-read from the graph 2026-08-18; the earlier 20-point generation is superseded and must not be quoted): full `GuaDecodeSweepJob.bVvWrIiZ2MwL` -> `GuaSelectJob.GQOGrA6sd9Ax` -> `GuaGenerateJob.P7c0GNgIt8LK` -> `GuaScoreJob.CLeBGY6k2NnY`; bigram-only `GuaDecodeSweepJob.yxg5kfzIJB33` -> `GuaSelectJob.EehyOxVHXBp7` -> `GuaGenerateJob.yM1kMat8ofMD` -> `GuaScoreJob.TZSFN5FLubRJ`. Fixed endpoint (declared in advance): full `GuaPinCheckpointJob.knPFtsTVTdMN` -> `GuaGenerateJob.1FazO6bRnEb0` -> `GuaScoreJob.kClihi9eGTuy`; bigram-only `GuaPinCheckpointJob.738PI5sL150y` -> `GuaGenerateJob.ROUHUjlQOiKX` -> `GuaScoreJob.ykS6g9QXLRHv`. Full-arm relabeling iteration 2 `GuaGenerateJob.0jGtVQIDcorO` -> `GuaSegmentsJob.gqJOWVD6TtrW` -> `GuaTrainJob.EwdQgD4XqYPI` -> `GuaDecodeSweepJob.wkYQ0fx9YeM1` -> `GuaSelectJob.b4LpanAratYe` -> `GuaGenerateJob.EXCk3cLpB4kz` -> `GuaScoreJob.UeD8Vt2mL2fy`; iteration 3 `GuaGenerateJob.fxueYq266ZpF` -> `GuaSegmentsJob.ZmaDEaqnTAyS` -> `GuaTrainJob.NOS8BQ4IoPlM` -> `GuaDecodeSweepJob.mWJV4jphL6g4` -> `GuaSelectJob.2XISV1CViuoE` -> `GuaGenerateJob.YeAgEwsze6io` -> `GuaScoreJob.iYigGwYyzwIN`. Each arm's optimizer trace is in its own job dir across `log.run.1.backup-quota` + `log.run.1` (the restart replays epochs; read one record per `train_num_updates`) |
| entry 8 | `config/sae_1f_entry8.py` | reads `GuaLmGridReadJob.SeNSdRhV1Wo3` (primary) / `.I9lgMOqar8RO` (SIL-augmented sensitivity); grid decodes `GuaLmDecodeGridJob.*` (8 jobs: 4 arms x 2 LMs + 2 beam probes); code `gua_lm_decode.py` (carries the discharged anchor pin) |
| entry 9.0 | `config/sae_1f_entry9.py` | `EspumIdentifiabilityJob.fzOQ9UKTnLh1` (c1) / `.NMDdH7owD52u` (c2) / `.ffAQBEntKvBe` (c3) / `.POCnVeDHejYU` (c4) / `.JnsqE57Ui4XQ` (c5); ruling `EspumIdentifiabilityReadJob.r4PXlgWX8uwY` (`gate.txt`, `gate.json`; re-run after the read-side fixes at speech-llm `0f12982`, every H unchanged) |
| entry 9.3 | `config/sae_1f_entry93.py` | `EspumDisclosureDecodeJob.BSCM3ZOXy0eI` (c1) / `.kPcyX1XzCcZg` (c2) / `.AfLEEL9bKOn9` (c3) / `.1ZODsDfIH31h` (c4) / `.AEerv7uGWqeX` (c5) (`disclosure.txt`, `disclosure.json`) |
| entry 9.1a (live) | `config/sae_1f_entry91a.py` | see State |
