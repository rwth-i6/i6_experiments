# SAE §1f — statistics-matching initialization, prerequisite kill conditions

## State
<!-- Overwritten in place, never appended; deleted at phase close. In-flight runs (job dir + the
question each answers), blockers, next action, proposals for the planner. -->

State as of 2026-08-23 -- entry 5 and entry 7 stage A are closed; PLAN_1F entry 8 (LM-decoded phone
error rate) is BUILT, TESTED AND LAUNCHED on the USER's word for cells 1 and 2 only.

**ENTRY 8 CELLS 1-2 ARE COMPLETE** (verdicts E8.1-E8.4; primary read
`GuaLmGridReadJob.SeNSdRhV1Wo3`, SIL-augmented sensitivity `GuaLmGridReadJob.I9lgMOqar8RO`; all
eight grid-decode jobs and both reads finished, ZERO error markers). The headline: language-model
decoding does cut the arms' error rate a long way (full loss 1.6828 -> 0.8444, bigram only
1.2409 -> 0.8172 at the label-oracle cell) and it does it largely by emitting half as many phones,
so the mechanism is confirmed and a usable decode is not. Sizing note for anyone repeating this:
the whole grid cost about an hour of GPU per arm; 572-utterance lexicon-free decoding at beam 50 is
51 seconds.

- THREE THINGS THE RESULT ITSELF EXPOSED, all now in the verdicts. (1) `sil_weight` is INERT --
  identical decodes at all three values -- because the vocabulary has no silence symbol and
  fairseq's index rule falls through to end-of-sentence, so the ruled second axis is a no-op and
  the grid is 4 points, not 12. (2) The registered label-free selector picks the WORST cell on
  both full-loss arms, and in the opposite direction to the circularity I disclosed: it prefers
  low lm_scale because per-token perplexity rewards the long insertion-heavy decode. (3) Beam 50
  is converged for the RATE (a ten-fold wider beam moves PER by at most 0.0195) and nowhere near
  converged for the SEQUENCE (one-best agreement as low as 0.1136).
- TWO GAPS OF MINE THAT THE FIRST READ EXPOSED, both fixed at speech-llm `55045ed`, both
  re-keying only the read jobs: the beam-500 probe finished with NOTHING consuming it -- a probe
  that is registered and never read is a probe that was not run -- and the payload still printed
  the anchor pin as UNDISCHARGED after the planner discharged it.
- FOR THE PLANNER, since two ruled constants turned out to be inert or harmful in practice rather
  than in principle: ruling (1)'s `sil_weight` axis buys nothing on this vocabulary and a future
  grid should drop it; and the registered label-free rule of ruling (3) is not just circular but
  actively anti-selecting on the full-loss arms, so if any entry-8 number is to be quoted as a
  single figure, the label-oracle cell with its range is the honest one and the rule's pick is
  not. Neither is mine to change.
- STILL NOT BUILT: cell 3 (the CTC-student decoder-sanity control) and cell 4 (re-banking the
  argmax nulls and the memoryless oracle-map ceiling under this decode). Until cell 4 exists no
  margin against 0.8946 / 0.9239 / 0.4148 may be quoted, and the pre-registered "stage A
  answerable after all" trigger cannot be evaluated. Cell 5 (the entry-5 ESPUM checkpoints) is
  untouched. The ANCHOR PIN IS DISCHARGED and recorded in `gua_lm_decode.py`: the published TIMIT
  0.473 is greedy/argmax currency, so no entry-8 number may be quoted against it in either
  direction.

## Approach

**1. Kill condition (i): the §0a information audit re-read on the current `enc50_raw` inventory.**
`AuditAvUnitsJob` is reused verbatim on the k-means-500 / PCA-96 codebook over the pretrained
wav2vec2-large-lv60 encoder tap at 50 Hz, on 500 LibriSpeech dev utterances against MFA gold with the
80/20 seed-0 held-out protocol, and the superseded 50 Hz codebook of the same K, PCA dim and seed is
re-scored as a second row on the identical utterances — it reproduces its finished 0.424 / ins 0.189
row exactly, which is what pins the protocol. `ChannelStructureJob`'s header adds the entropies and
the over-segmentation ratio that job does not report. The registered bar is dev-other oracle-map
PER <= 0.50.

| inventory | split | oracle PER | sub | ins | del | purity | frame acc | H(phone\|unit) | PNMI | units/phone | dead |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `enc50_raw` (in use) | dev-clean | **0.712** | 0.115 | 0.591 | 0.006 | 0.694 | 0.680 | 1.046 | 0.682 | 2.794 | 11 |
| `enc50_raw` (in use) | dev-other | **0.832** | 0.132 | 0.692 | 0.008 | 0.659 | 0.657 | 1.186 | 0.633 | 2.824 | 10 |
| `enc50_prior` (superseded) | dev-clean | 0.424 | 0.185 | 0.189 | 0.050 | 0.659 | 0.644 | -- | -- | -- | 0 |
| `enc50_prior` (superseded) | dev-other | 0.451 | 0.195 | 0.209 | 0.046 | 0.637 | 0.632 | -- | -- | -- | 0 |

**2. Kill condition (ii): the §1a conclusion-6 channel-structure claim, measured.** The unit
co-occurrence graph of the deduped stream is projected to phones through the ORACLE unit-to-phone map
fitted on the same gold frames — no unsupervised matcher can do better, so every number bounds the
matcher from above — and compared with the phone bigram of the same utterances. Two controls rebuild
each utterance from its OWN real segments cut at the gold phone boundaries and differ only in where
each segment is drawn from: `seg_swap` from another occurrence of the same phone (the channel
factorizing exactly as a matcher assumes, at this stream's own durations, emission ambiguity and
within-phone dynamics) and `seg_rand` from a phone drawn at random (the floor).

The graph is split at the gold phone boundary before anything is correlated. Pairs inside one phone
cannot carry a transition at all: they land either on the diagonal (duration and flicker) or, wherever
the oracle map disagrees across a single phone's own frames, in exactly the off-diagonal cells a
distribution matcher reads as transitions. Only 0.347 (dev-clean) / 0.341 (dev-other) of adjacent
pairs cross a boundary, so `all` is the graph a matcher can observe and `cross` is the part of it that
could carry the bigram. Correlations run on the gold graph's support; `tv_offdiag` is the
total-variation gap over all off-diagonal cells, which is the matcher's own L1 objective evaluated at
a map it will never have. The raw cell correlation is confounded — both graphs inherit the same
phone-unigram frequencies — so the marginal-free pointwise-mutual-information versions are read.

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

**3. The screen battery, one row per REPRESENTATION of the same stream (PLAN_1F queue item 0).**
Five frame-level 50 Hz representations of the enc50 stream are screened by the same three jobs, so
every row is like-for-like and the raw row reproduces its registered PER exactly: `seg16/12.5/9` pool
the encoder features by adjacency-constrained Ward merging to 16 / 12.5 / 9 tokens per second and
relabel each segment by the same codebook (nearest centroid to the segment mean; the job re-quantizes
20 utterances frame by frame and asserts the shipped codes come back bit-exactly), `brown100` merges
the inventory 500 -> 100 by bigram context at the raw rate, and `ubpe12.5` merges TOKENS by unit-BPE
(8000 merges, the cap, reaching 14.1 tok/s against a 12.5 target). Both coarsenings fit on the 28 539
train utterances only, never on the dev the screens read. Two reads are added to the registered
protocol: a unit-to-phone-STRING map (`PER_str`, each gold phone run given to the token holding most
of its frames, so a token may spell several phones or none -- the ceiling of a token map that can
delete), and sigma_min of the positional-unigram matrix P_X, which is the identifiability condition
of arXiv:2306.07926's closed-form estimator.

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

**4. Ladder entry 3: the unary fingerprint assignment, read against the arm gate (PLAN_1F queue
item 1).** Per-unit transition-free statistics from the unpaired audio -- log relative frequency,
utterance-initial and utterance-final rate, and a six-bin histogram of position in the utterance --
are matched to the same statistics of `T_phi` phones by one entropic optimal-transport solve on the
two frequency marginals with the regularization fixed at 0.1 before the run, after units whose frame
occupancy within 0.2 s of an utterance edge far exceeds their corpus share are called silence by a
label-free two-means split, mapped to SIL and left out of the solve. Every map -- candidate, the
marginal-matched random null n1, the 1e pseudo-pair null n2 in map form (each audio utterance paired
with a length-matched `T_phi` line, the map counted off the proportional correspondence), and the
eval-only oracle -- is fitted without labels on all 8416 utterances the streams cover and scored
through the registered oracle-map protocol on the same held-out fifth of each split. Three reads
make the verdict legible rather than a bare threshold comparison: the registered audio-swap control
is run on the oracle and on both nulls as well, so M2 is a fraction of the span between a
content-carrying and a content-free map; `pos` re-solves without the frequency column; and
`top5`/`mrr` are the rank the true phone takes under the fingerprint cost itself, chance 0.128 /
0.109.

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

**5. Ladder entry 6's kill-test: is any frequent unit-word a function word (PLAN_1F queue item 3,
and ruling 3's precondition for the lexicon-free text arm).** The label-free edge-enrichment split
that entry 3 uses names the silence units, those cut each utterance into silence-delimited segments,
and ONE greedy unit-BPE merge list is learned over the segments toward the 2.8 unit-words/s English
word rate -- every row below is a PREFIX of that single list, so the rows are one granularity curve
rather than independent builds. A unit-word carries the registered signature when it is in the top
20 by frequency, is at or below the frequency-weighted mean unit-word length, opens an utterance at
least 10 % of the time and closes one at most 2 % of the time; `base` is the rate a positionally
indifferent unit-word would show (one over unit-words per utterance) and `best/base` the enrichment
over it. The last row applies the identical rule to the top 20 words of the raw LM corpus, so the
hit count and the enrichment are read against the scale English itself sets rather than against a
bare threshold.

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
on entry 6's own stream, silence split and merge list, reading each hit's gold labels at its
utterance-initial occurrences against its other occurrences (`TV_ie`, small = the same thing wherever
it occurs) and against the corpus's own utterance-onset mixture (`TV_i_onset`, small = it looks like
whatever starts an utterance); silence and word-uncovered frames are held out of both and returned as
`off_init`, because a unit that only marks speech onset sits on exactly those frames at exactly the
position the signature rewards, and leaving them in would read as a stable meaning. The labels
restrict only the columns -- the stream, the split, the merge list and the type selection come from
all 8416 utterances, of which 5567 carry an alignment -- and every hit is read against `other`, the
median over the remaining top-20 types with at least 20 initial occurrences, since neither distance
means anything on its own.

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

**7. Ruling 3: lexicon-free text sides against the phone reference arm, on all four audio
representations.** Each text side is screened at the merge-list prefix whose MEASURED unit-word rate
sits closest to its own token rate, and the ceiling is refitted inside the candidate's own map space
so oracle and candidate price the same hypothesis set; the eval set is identical across all four
rungs -- both unit sets cover both dev splits completely and `seg` is a strict subset of `ubpe` -- so
the rungs differ on the train side only. Every stream stopped on `no_pair_repeats` rather than on the
merge cap or on the rate target, so no rung reached its target token rate and each `words` row is
screened against audio running 1.24-2.30x faster than the text it is matched to. The
`max_fit_tokens` bound bit on `ubpe12.5` alone (1500017 of 3867583), and it compresses the
merge-FITTING asymmetry only -- the unigram fingerprints and the pseudo-pair null still consume the
full 34106-utterance stream, so kept/total does not close the train-side gap.

Arm gate, unchanged and read per representation AND per text side: M1 >= 0.05 AND M2 >= 0.05 on
dev-other, plain PER as scored against the same sil-free gold phones.

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

**8. Ladder entry 5: training-based statistics matching on the pooled seed stream (the funded
batch).** A one-layer convolution, kernel 4 over the one-hot 500-way unit stream, is trained to make
the batch count statistics of its segment-level phone posteriors -- positional unigram at absolute
segment index, skipgrams at skips 1-6, tri-skipgrams -- match those of unpaired phone text under an
L1, with the segment boundaries FIXED from the battery's own pooling and a smoothness penalty on
adjacent segments' logits. Every checkpoint and the reported seed are chosen label-free, by phone
language-model perplexity of the arm's own decodes weighted by how much of the phone inventory those
decodes use; the phone error rate is computed only afterwards, by a separate job, and the training
job is handed no reference file of any kind. The plumbing check (E1) supervises the same
input-and-pooling path on the eval-only forced alignment and discards its checkpoint.

Arm gate unchanged: M1 >= 0.05 AND M2 >= 0.05 on dev-other against the banked `seg12.5` phone-side
nulls (n1 0.8946, n2 0.9239, ceiling 0.4148), plain PER as scored on the same 572-utterance fifth.

| arm | label-free ppl | update | dev-other PER | sub | ins | del | hyp/ref | M1 | M2 | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 supervised probe (plumbing, ceiling-fit rows, checkpoint discarded) | — | — | **0.3565** | 0.1296 | 0.0213 | 0.2056 | — | — | — | — |
| full loss, seed 1 (**label-free selected**) | **31.41** | 40000 | **0.8580** | 0.6909 | 0.0699 | 0.0972 | 0.973 | +0.0365 | +0.0466 | **fail** |
| full loss, seed 2 | 31.49 | 40000 | 0.8848 | 0.7195 | 0.0696 | 0.0958 | 0.974 | +0.0098 | +0.0244 | fail |
| full loss, seed 0 | 33.04 | 40000 | 0.8770 | 0.7074 | 0.0717 | 0.0979 | 0.974 | +0.0175 | +0.0302 | fail |
| bigram-only, seed 0 | 53.86 | 30000 | 0.8748 | 0.7096 | 0.0641 | 0.1011 | 0.963 | +0.0198 | +0.0254 | fail |
| ruling-3 unary candidate, same rung and text side (approach 7) | — | — | 0.8809 | 0.7157 | 0.0478 | 0.1174 | — | +0.0137 | +0.0148 | fail |

**9. Ladder entry 7: the published graph-based pipeline, run verbatim on our seed bed (USER ruling
6).** Entry 5 is our own statistics-matching implementation; this one is the reference method
(GraphUnsupASR, arXiv:2310.02382) executed as published -- its own feature extraction, its own
clustering, its own text preparation, its own trainer -- on the same seed audio and the same
unpaired text, so a gap between the two is attributable to the implementation rather than to the
bed. Two arms differ in one argument only, which statistics the matching objective is asked to
match: `full` against `bigram_only`, the same contrast entry 5 ran, at 40,000 updates each with a
checkpoint every 2,000; a smoke arm of a few hundred updates and one end-to-end alignment pass
precede them and are discarded by construction. Recognition: the split reserved for
selection is decoded with each of the eighteen checkpoints the trainer actually wrote, the
checkpoint is pinned by entry 5's own label-free metric (phone-LM perplexity of the decode,
weighted by how much of the inventory it uses), and only then is the scored fifth decoded and
scored -- reference phones are opened in one job, downstream of a pinned checkpoint.
Every decode is the released generation script through the conditional-gold patch, so the
recognizer is the reference's, not ours. The full arm additionally carries the reference's two
relabeling passes -- align every split off the pinned checkpoint, refit on the boundaries it
predicted -- which is what makes the iteration-3 read exist alongside the matched iteration-1 one. Each
arm is additionally read at update 40,000, an endpoint declared before the decode and pinned without
consulting any metric, so the same contrast exists once with the label-free selector in the loop and
once with nothing in the loop at all.

Iteration 1, dev-other scored fifth, 572 utterances / 34,135 reference phones. Applied step is
lr * min(1, clip_norm / gnorm) with clip_norm 20 and lr constant at 0.004, on the two update
windows both arms' surviving logs cover (999 and 58 epochs each; a timeout resubmit overwrote the
full arm's middle segment).

| iteration-1 arm | update read | PER | sub | ins | del | phones/utt | swap-control M2 | gnorm early / final | fully clipped early / final | applied step early / final |
|---|---|---|---|---|---|---|---|---|---|---|
| full loss, label-free pick | 2,000 | 1.6843 | 0.6046 | 1.0766 | 0.0031 | 123.7 | +0.0013 | 1,617 / 8,981 | 100.0 % / 100.0 % | 1.07e-04 / 3.64e-05 |
| full loss, fixed endpoint | 40,000 | 1.6828 | 0.6153 | 1.0650 | 0.0025 | 123.1 | +0.0039 | same arm | same arm | same arm |
| bigram only, label-free pick | 30,000 | 1.2449 | 0.6836 | 0.5547 | 0.0065 | 92.4 | +0.0079 | 1,256 / 20,395 | 50.7 % / 60.3 % | 5.59e-04 / 8.11e-05 |
| bigram only, fixed endpoint | 40,000 | 1.2409 | 0.6811 | 0.5535 | 0.0062 | 92.3 | +0.0107 | same arm | same arm | same arm |
| reference | -- | -- | -- | -- | -- | 59.7 | -- | -- | -- | -- |

**Entry 8. The same four entry-7 decodes under a phone 4-gram beam search instead of per-frame
argmax.** The released generation script keeps the emissions, the batching and the hypothesis
writing; one further patch anchor makes its KenLM branch return a mirror of fairseq's lexicon-free
unit-language-model branch (one dictionary word per phone, `KenLM`, `LexiconFreeDecoder`), because
the branch the script itself imports no longer exists in this fairseq. Emissions are log-softmaxed
before decoding -- the script overrides the model with `no_softmax`, which is harmless for argmax
and not for a beam that adds acoustic and language-model scores on one scale. Four checkpoints
(both arms x label-free pick and fixed endpoint 40,000) x 12 grid points (lm_weight in
{0.5, 1, 2, 4} x sil_weight in {-2, -1, 0}) x two splits: the selection four-fifths for the
registered label-free grid pick, the scored fifth for the read. Beam 50 with a beam-500 convergence
probe on the fixed-endpoint arms. Nothing is refit and no checkpoint is selected.

| decode | arm | update read | PER | sub | ins | del |
|---|---|---|---|---|---|---|
| greedy (banked, entry 7) | full loss, fixed endpoint | 40,000 | 1.6828 | 0.6153 | 1.0650 | 0.0025 |
| pre-launch hand probe, lm_weight 2 / sil 0 / beam 50 | full loss, fixed endpoint | 40,000 | 0.9322 | 0.7624 | 0.0979 | 0.0718 |

The second row was a single cell run by hand to prove the chain end to end before any job was
submitted; it is superseded by the table below. Both rows are plain rates on the same 572
utterances against the same references.

RESULT, primary language model (SIL-free 4-gram, `CreateBinaryLMJob.hvZoC014xnIe`), beam 50,
`GuaLmGridReadJob.SeNSdRhV1Wo3`. Every arm's full 12-cell grid is in the artifact; each row here
carries the registered label-free pick, the label-oracle best as its upper bound, and the grid's
own range, per the planner's ruling (3):

| arm | greedy PER | label-free pick | label-oracle best | grid PER range | hyp/ref phones at the oracle cell |
|---|---|---|---|---|---|
| bigram only, fixed endpoint | 1.2409 | 0.8481 (lm 0.5) | 0.8172 (lm 1) | 0.8172-0.8985 | 20836/34135 |
| bigram only, label-free pick | 1.2449 | 0.8520 (lm 0.5) | 0.8195 (lm 1) | 0.8195-0.8963 | 21082/34135 |
| full loss, fixed endpoint | 1.6828 | 1.5446 (lm 0.5) | 0.8444 (lm 4) | 0.8444-1.5446 | 17212/34135 |
| full loss, label-free pick | 1.6843 | 1.4805 (lm 0.5) | 0.8463 (lm 4) | 0.8463-1.4805 | 15949/34135 |

Error decomposition at the two ends of the full-loss endpoint arm's grid: at lm 0.5 the decode is
still insertion-dominated (sub 0.6466, ins 0.8950, del 0.0030, 64,586 phones emitted against
34,135 reference); at lm 4 the insertions are gone and deletions have taken over (sub 0.3444, ins
0.0021, del 0.4979, 17,212 phones). The bigram-only arm shows the same trade at lower lm_scale.

Beam-500 convergence probe, same cells, same 572 utterances, both fixed-endpoint arms: PER moves by
at most 0.0195 in any cell (every delta negative, i.e. the wider beam is never worse), while
one-best agreement between the two beams runs 0.1136 to 0.4843. The rate is converged at beam 50;
the SEQUENCE is nowhere near converged.

SENSITIVITY, SIL-augmented 4-gram on the two fixed-endpoint arms
(`GuaLmGridReadJob.I9lgMOqar8RO`): same shape, differences small and in BOTH directions -- full
loss best 0.8476 against 0.8444 (worse), bigram only best 0.8145 against 0.8172 (better; verifier
correction of "slightly worse throughout", 2026-08-23) -- so the vocabulary mismatch costs little
here and changes no conclusion.

## Verdicts

**Entry 8 cells 1-2, four verdicts.**

E8.1. **Language-model decoding cuts the entry-7 arms' phone error rate by a large margin, and it
does it mostly by emitting fewer phones.** Best cell against the greedy rate at the same
checkpoint: full loss 1.6828 -> 0.8444, bigram only 1.2409 -> 0.8172. But the best cells emit
between 47 and 62 percent of the reference phone count, and their error is deletion-dominated
(full loss at lm 4: ins 0.0021, del 0.4979). Both arms land in the same 0.82-0.85 band from very
different starting points, which is the band a decode reaches by saying little. The registered
mechanism -- an insertion-dominated rate attacked by a language model -- is CONFIRMED as a
mechanism and does NOT deliver a usable decode.

E8.2. **The second grid axis is INERT on this decoder: all three `sil_weight` values give the same
decode in every cell.** Not merely "not an insertion score" (the ruled amendment), but a no-op.
The cause is the same fact that inverted the language-model pin: the entry-7 generator's
vocabulary has no silence symbol, so fairseq's own index rule falls through to the
end-of-sentence index, and `sil_score` is charged on a token the decoder never emits. The grid
that ran is therefore 4 points, not 12. Every 12-row table in the artifact is 4 distinct decodes
repeated three times, and it should be read that way.

E8.3. **The registered label-free selector fails here, and in the OPPOSITE direction to the one I
disclosed.** I flagged it as near-circular and predicted it would be dragged toward high
lm_weight. It picks lm 0.5 -- the LOWEST -- in all four arms, because weighted phone-LM perplexity
per token rewards the long, insertion-heavy decode. On the two full-loss arms that is the WORST
cell of the grid: 1.5446 picked against 0.8444 available, and 1.4805 against 0.8463. On the
bigram-only arms the damage is small (0.8481 against 0.8172). CONSEQUENCE: the quotable number
under the registered rule is 1.5446 for the full-loss arm, which is worse than three quarters of
its own grid; quoting it without the range beside it would misrepresent the decode in the harmful
direction. The planner's ruling (3) reporting rule is what makes that visible, and it is now
printed by the producing job.

E8.4. **Beam 50 is converged for the RATE and not for the SEQUENCE.** A ten-fold wider beam moves
PER by at most 0.0195 in any cell while the two beams agree on the one-best in as few as 11.4
percent of utterances. The same signature the 1g decoder showed at its own beam doublings
(PLAN_1G verdicts 33-35): many near-equal hypotheses, so which one wins is unstable while what it
scores is not. No beam escalation is warranted for this measurement.
<!-- One line per answered experimental question, resting on a number in that approach's table. A
wrong verdict is marked WRONG with a one-line correction below it, never rewritten. (Migrated from
the pre-format "Conclusion" heading, 2026-08-23; entries below predate the one-line form.) -->

1. (1) **Kill condition (i) FAILS its registered bar**: the inventory the loops run on reads oracle-map
   PER 0.832 on dev-other and 0.712 on dev-clean against a bar of 0.50, so it caps a unit-level token
   mapping harder than the 0.53-0.63 of the §0a inventories that closed §1a(i).
2. (1) The cap is **over-segmentation, not confusability**, and that localizes what a redirect would
   have to fix: substitutions are 0.115 / 0.132, lower than both the superseded 50 Hz codebook
   (0.185 / 0.195) and §0a's k-means-500 (0.245), while insertions are 0.591 / 0.692 at 2.79 deduped
   units per gold phone token. Frame-level phone information is the highest this program has measured
   on a unit stream (PNMI 0.682, H(phone|unit) 1.046 nats against H(phone) 3.292).
3. (2) §1a conclusion 6 holds operationally but **not literally**: the boundary-crossing part of the
   graph does carry the bigram (rank correlation on pointwise mutual information 0.515 / 0.517 against
   a 0.03 floor), but a matcher does not see boundaries, and on the graph it does see the correlation
   falls to 0.373 / 0.370 between a floor of 0.214 / 0.216 and a ceiling of 0.413 / 0.398.
4. (2) The matcher's own objective barely separates truth from noise even at the oracle map:
   total variation of the observable off-diagonal bigram runs 0.409 (perfectly factorizing channel) /
   0.431 (real) / 0.459 (no correspondence at all) on dev-clean, an 11 % relative span with the real
   stream 57 % of the way from floor to ceiling; dev-other is 8.9 % and 46 %.
5. (2) **Coarticulation is measured, not assumed**: 26.1 % / 26.7 % of genuine phone transitions
   project onto the same phone in the real stream against 4.4 % / 4.5 % when the same segments are
   drawn independently of their neighbours — a 5.9x inflation, so a quarter of all transitions are
   invisible to the graph. This is the identifiability precondition (the channel factorizing as the
   model assumes) failing by a measured amount rather than by inference.

6. (3) **Kill condition (i) is CLEARED by data-driven segment pooling**: every rung passes the
   registered dev-other bar (0.414 `seg12.5` / 0.452 `seg16` / 0.481 `seg9` against 0.50; dev-clean
   0.380 / 0.385 / 0.466), the best this program has measured on any inventory, so the arm stays at
   the UNIT level and the registered FEATURE-level redirect is not needed. The pass does not rest on
   a label-chosen rung: the gold phone rate is 9.8 / 9.4 per second, so the label-free rate-matched
   rung is `seg9`, which passes on its own.
7. (3) The cap was the token RATE alone — at a fixed rate, coarsening the INVENTORY is catastrophic
   (`brown100` PER 1.063 / 1.152 with PMI rank correlation 0.109 / 0.103 against raw's 0.373 / 0.370),
   so the 500-way codebook's discriminability is load-bearing and only over-segmentation ever had to
   be fixed.
8. (3) Ladder entry 2 (the ridge positional-unigram estimator) fails its own registered gate on every
   representation — sigma_min(P_X) is 5e-33 raw, exactly 0 wherever usable positions are fewer than
   units, and 2e-17 on the one representation where full column rank is possible at all (`brown100`,
   K=100 over 495 position rows) — and a SIMULATED channel with perfect recoverability reads 0 too,
   so the failure is structural (utterances do not supply as many estimable positions as there are
   units) rather than a property of this channel.
9. (3) The tv_offdiag bar pre-registered in `PLAN_1F.md` cannot be read as written on any
   representation: its span term is 3.8-11.4 % everywhere against a 25 % bar, while its position term
   exceeds 1 on every pooled row because the real stream now beats the `seg_swap` ceiling —
   coarticulation pushes correlated errors onto the DIAGONAL instead of spreading them off-diagonal,
   so a channel that erases a quarter of its transitions scores BETTER on the matcher's own objective
   than one that factorizes by construction.
10. (3) On the reads that inversion does not confound, pooling buys real signal: the share of adjacent
    pairs that can carry a phone transition rises 0.347 -> 0.700, the observable-graph PMI rank
    correlation 0.373 -> 0.595 against a floor that also rises (0.214 -> 0.465), and the real stream's
    gap to that floor on the matcher's objective triples (6.2 % -> 21.2 %).
11. (3) Calibrated against simulated channels of known recoverability at MATCHED fertility, the pooled
    stream is still worse than 35 % random emissions (`seg12.5` reads tv 0.322 / PMI 0.632 on the
    500-utterance screen against 0.298 / 0.765 for fertility 1 at 35 % noise), so pooling fixed the
    rate and left the emission ambiguity severe by construction-calibrated standards.
12. (3) The string map earns its complexity only for unit-BPE (0.538 -> 0.436 dev-other) and buys
    nothing on a pooled stream (0.414 -> 0.435), while on the raw stream it only converts insertions
    into deletions (0.692 -> 0.646) — a context-free token map cannot repair a rate error, which is
    why pooling and not mapping was the fix.

13. (4) **Ladder entry 3 FAILS the arm gate on every representation**: its best margin over the
    better null is +0.015 (`seg12.5`, dev-other) against the pre-registered 0.05, and on `raw`,
    `seg16` and `ubpe12.5` it loses outright to the pseudo-pair null n2.
14. (4) The audio-swap control, anchored by running it on the oracle map (0.372-0.596) and on the
    random null (0.004-0.020), places the candidate at 0.9-4.6 % of the content-dependence span and
    at or below the random null's own movement on `seg9` — by the registered control the map is
    content-free, which is the gate's own criterion answered directly rather than by a threshold.
15. (4) Both of entry 3's registered kill-tests fail: manner separation is 0.41-0.49 against a ~0.50
    bar over majority baselines of 0.34-0.47, and admitted-pair precision is 0.00-0.29 against ~0.70,
    with the bootstrap admitting nearly the same set with and without the CSLS margin (10 vs 11 on
    `seg12.5`) and starting at chance from a PERFECT ten-pair seed rather than decaying from a good
    start, so the recounting step never engages at all.
16. (4) The failure is diffuseness, not a broken solver: under the fingerprint cost the true phone
    is in the top five for 0.23-0.32 of unit mass against a chance 0.128 (reciprocal rank 0.16-0.20
    against 0.109), so the transition-free statistics do carry about twice chance information about
    phone identity and it is far short of what a 39-way many-to-one assignment needs.
17. (4) Frequency is a nuisance column rather than the signal — dropping it improves the solve on
    every row (0.882 -> 0.866 on `seg12.5` dev-other) while frequency alone is the worst of the three
    (0.885) — so the transport marginal already carries what the frequency marginal says, and the §1c
    single-marginal warning holds in the stronger form that the marginal hurts as a feature.
18. (4) CAVEAT on the ceiling this arm quotes throughout: the registered frame-argmax oracle map is
    not PER-optimal on an over-segmented stream, since forcing the silence-proxy units to SIL beats
    it by 0.095 on `raw` (0.773 -> 0.678), while on the pooled rungs the two agree to within 0.026 in
    either direction —
    the battery's pooled ceilings stand and only the raw ceiling is loose.
19. (5) **Entry 6's kill-test does NOT kill**: at every granularity coarser than the bare unit at
    least one top-20 unit-word carries the registered function-word signature on `seg12.5` (peak 3
    at half the merge list) and on `ubpe12.5` (peak 2), against the 2 that the identical rule finds
    among the top 20 English words — so ruling 3's lexicon-free text arm keeps its precondition.
    CORRECTION (approach 6, 2026-08-16): the `ubpe12.5` half is WRONG — its hit at @0.5 and @1 is
    the all-silence unit 397, so that rung carries a non-silence hit at @0.75 only, and the `seg12.5`
    half stands.
20. (5) The signature is positional only and not distributional: every hit is a single UNMERGED
    unit whose non-initial occurrences the merges absorbed into longer tokens, the top unit-word
    holds 0.006 of tokens against THE's 0.061, and the unit-word Zipf slope is -0.90 to -1.01
    against -1.39 for words — the streams have function-word POSITION without function-word MASS.
21. (5) The granularity curve turns over inside the swept range (hits 0/1/3/2/1 on `seg12.5`,
    0/0/1/2/1 on `ubpe12.5`), so the finest attainable rows — 4.26 and 7.29 unit-words/s against the
    2.8/s word-rate target — sit past the peak and do not understate the hit count.
22. (5) The hits are 8.5-14.9x their positionally indifferent base against 4.3-5.7x for the English
    hits, i.e. MORE committed to the utterance onset than a real function word is, which no
    label-free read can separate from an utterance-onset acoustic effect riding on the same units.
    CORRECTION (approach 6, 2026-08-16): the 14.9x end is WRONG as evidence about function-word-like
    units — it is `ubpe12.5`'s all-silence 397 — while `seg12.5`'s 8.5x survives as a real unit; the
    "no label-free read can separate them" clause stands, since approach 6 separates them with labels.
23. (5) **`seg12.5` cannot be BPE-compressed to word rate at all, and the reason is recurrence, not
    budget**: its silence-delimited stream is 720315 tokens against the 800000-token fitting cap and
    it stopped at 38228 merges against the 50000 cap, so with both budgets slack the merge loop can
    only have exited on its no-pair-occurs-twice break — after 38228 merges NO adjacent unit-word
    pair repeats anywhere in the corpus, and the stream's recurrence is exhausted at 4.26
    unit-words/s, still 1.5x the English word rate (`ubpe12.5`'s 7.29/s stall is separate and IS the
    merge cap, hit exactly at 50000).
24. (6) **On `seg12.5` the signature is linguistic, not an utterance-onset effect**: the hit present
    at every prefix (403) reads AH initially and AH elsewhere on phones and THE -> THE on words, with
    `TV_ie` 0.074-0.094 (phones) and 0.242-0.245 (words) against other-type medians of 0.291 and
    0.915, and it sits far from the corpus's own onset mixture (`TV_i_onset` 0.65-0.66 on phones
    against the other-type 0.384) — the direction that says "same thing wherever it occurs".
25. (6) **On `ubpe12.5` the signature is an artefact of a missed silence unit**: 397, whose 13.8-14.9x
    enrichment is the largest in approach 5's table, has every frame it ever occupies labelled silence
    (`off_init` = `off_else` = 1.000), and since entry 6 builds its segments by DELETING proxy-silence
    units, 397 reaching the merge list at all proves the label-free edge-enrichment proxy did not call
    it silence; the one genuinely linguistic ubpe hit is 608 at @0.75 (Y -> Y, YOU -> YOU).
26. (6) A hit can be positionally and phonetically stable without being a word: `seg12.5`'s 423 and
    432 hold one phone at both positions (T .72 -> .66, N .67 -> .49) while their word reads disagree
    across positions (IT -> BUT, IN -> AND, `TV_ie` 0.715 and 0.630), so of the three `seg12.5` hits
    only 403 supports the function-word reading the kill-test was about.
27. (7) **Ruling 3 FAILS the arm gate in all twelve cells**: M2 never exceeds 0.0252 against the 0.05
    bar and M1 is negative in 10 of 12, so no lexicon-free text side clears the bar on any of the four
    audio representations, and the two positive M1 cells are both the phone reference side.
28. (7) The train-side corpus asymmetry did NOT decide the ruling: `ubpe12.5` runs on 4.1x the
    utterances (34106 vs 8416) and 5.4x the audio (111.0 h vs 20.5 h) and is nonetheless the worst
    rung on every text side, so the corpus-matched `ubpe` control that PLAN_1F amendment (6) makes
    conditional is not triggered.
29. (7) The `words` cells price a rate-MISMATCHED arm rather than a rate-matched one: no stream reached
    the 2.8/s target, the mismatch shows up directly as candidate insertions (0.717 on `ubpe12.5`
    against 0.004 deletions), and a budget large enough to reach the target could only be spent on
    `ubpe12.5`, whose merges stopped on exhausted pair repeats at 6.45/s.
30. (8) **E1 PASSES its one-sided floor**: the supervised plumbing probe reaches 0.3565 dev-other PER,
    below the 0.4148 memoryless oracle-map ceiling, so the input and pooling path carries at least
    ceiling-level phone information and a poor unsupervised read on this arm cannot be charged to it.
31. (8) **Entry 5 FAILS the arm gate on both clauses**: the label-free-selected candidate reads 0.8580
    dev-other against an M1 bar of 0.8446 (margin +0.0365) and an M2 of +0.0466 against 0.05, so the
    only unkilled ladder entry closes and 1f returns to the user with none.
32. (8) The bigram-only ablation did NOT separate from the full loss, so the health checkpoint's
    expected signature is absent: at 0.8748 it beats two of the three full-loss seeds, and the
    seed spread of the full loss (0.8580-0.8848) exceeds its 0.0015 gap to the ablation.
33. (8) The arms fail on IDENTITY, not on rate or on collapse: they emit 0.963-0.974 of the reference
    phone count, use all 39 phone types, and carry 79-81% of their error as substitutions, so the
    matching objective fixed the marginal statistics it is written on while leaving the unit-to-phone
    assignment near chance.
34. (8) Training-based matching does beat the unary fingerprint solve on the identical rung, text side
    and fifth (0.8580 against 0.8809, both margins roughly tripled), which prices what the added
    machinery buys and leaves it short of the bar by more than it gained.

35. (9) **The entry-7 stage-A signature is ABSENT and its sign is REVERSED**: bigram-only minus
    full-loss phone error rate is -0.4394 (1.2449 against 1.6843) where the registered bar asked for
    at least +0.10, so the ablation the reference method's own loss design predicts would hurt is the
    arm that scored better.
36. (9) That comparison is **not interpretable as a contrast between the two losses**, because the
    label-free selector shows no signal on either arm and pinned them 28,000 updates apart: weighted
    phone-LM perplexity spans 38.16-41.15 over the full arm's eighteen checkpoints and picked update
    2,000, the FIRST one, against 30,000 for bigram-only.
37. (9) Both arms **over-generate massively** -- 123.7 and 92.4 phones per utterance against the
    reference's 59.7 -- so the phone error rate is insertion-dominated (1.0766 and 0.5547) and the
    arm that emits more is mechanically the worse-scoring one; on substitutions alone the ordering
    reverses again to the predicted direction, 0.6046 against 0.6836, by 0.079. **The second clause
    is WRONG.** Substitution-only does not restore the predicted direction, it selects the
    length-favouring metric: recall against the reference is 0.3923 against 0.3099 favouring full,
    precision per emitted token is 0.1892 against 0.2002 favouring bigram-only, and both orderings
    are artifacts of the 2.07x-against-1.55x length ratio, so no decomposition of this pair of
    decodes carries a direction.
38. (9) The **audio-swap control is flat on both arms**: re-pairing each hypothesis with a different
    utterance's reference moves the phone error rate by +0.0013 and +0.0079, so neither arm shows
    measurable utterance-specific information above a mismatched pairing -- though with hypotheses
    1.5-2x longer than the reference the edit-distance match is largely length-driven, which limits
    how much this control can discriminate here.
39. (9) Both arms ran at a **declining applied step under a constant declared learning rate** of
    0.004: 1.07e-04 to 3.64e-05 (full) and 5.59e-04 to 8.11e-05 (bigram-only) between the two matched
    windows, because gradient norms rise 5.6x and 16x into a fixed clip ceiling of 20. The arm
    expected to win applies the smaller step in both windows, which is the direction that MASKS a
    real difference rather than manufacturing one.

40. (9) The fixed-endpoint read **reproduces the reversal at a common update with nothing in the
    selection loop**: at update 40,000 on both arms the signature is -0.4419 (1.2409 bigram-only
    against 1.6828 full) where the selected checkpoints gave -0.4394, so the 28,000-update spread
    conclusion 36 names is NOT what produced the sign, and that objection is retired -- over-generation
    is likewise stable across the run, 123.1 and 92.3 phones per utterance against 123.7 and 92.4.
41. (9) **Stage A closes NOT ANSWERABLE** on the pre-registered interpretability condition, for the
    remaining reason rather than the checkpoint one: at the fixed endpoint both arms sit far above
    the 0.8446 arm-gate margin (1.6828 and 1.2409) and both audio-swap controls stay flat (+0.0039
    and +0.0107 against 0.05), so neither decode carries measurable utterance-specific information
    and the contrast is between two uninformative decodes rather than between two losses.

The kill conditions and the battery are reported, not acted on. `seg12.5` leads both splits and
`seg9` is the label-free rate-matched rung; which one the ladder runs on, and whether the tv_offdiag
bar survives the ceiling inversion in conclusion 9, are planner calls. The pooled streams cover only
the seed dump's 8416 utterances (2849 train plus all 5567 LibriSpeech dev), which is what the screens
need — funding a matcher on one needs the same pooling pass over the assign-side tc100 / 960 h shards.
Entry 3 is likewise reported, not acted on: it fails the arm gate and both of its own kill-tests on
every rung, and conclusion 16 says the shortfall is the diffuseness of transition-free statistics
rather than the solver, which is the fact the remaining ladder entries have to be weighed against.
Two operational choices behind the entry-3 table are the implementer's and are open to revision:
null n2 is the 1e pseudo-pair PROTOCOL transplanted to map form (the 1e init itself is an SFT run,
not a map), and the silence class comes from a label-free edge-enrichment proxy whose cost against
the unrestricted ceiling is at most 0.026 in either direction on the pooled rungs (conclusion 18).
Entry 6's kill-test is reported the same way: it clears, so the lexicon-free arm may be funded, but
conclusions 20 and 22 say what it does not establish, and whether the surviving signature is worth
a funded arm is a planner call. Two of its operating points are the implementer's: the signature
thresholds (top 20, initial >= 0.10, final <= 0.02, at or below the mean unit-word length) were
fixed before the run and are the ones the English reference row is also scored under, and the merge
list stops at 4.26 / 7.29 unit-words/s rather than the 2.8/s target — conclusion 21 is why that
ceiling does not change the kill-test verdict, and conclusion 23 is why the two rungs stop for
different reasons, only one of which a larger budget could move.

## Catalog

| artifact | path |
|---|---|
| entry 8 LM decode code (+ tests) | `recipe/2025-10-speech-llm/src/speech_llm/sae/gua_lm_decode.py`, `scripts/gua_lm_decode_test.py` (27/27) at speech-llm `4fa256c` |
| entry 8 config | `config/sae_1f_entry8.py` -> `.../librispeech/configs/config_sae_1f_entry8_v1.py` |
| entry 8 reads, primary / SIL-augmented sensitivity | `work/speech_llm/sae/gua_lm_decode/GuaLmGridReadJob.SeNSdRhV1Wo3` / `.I9lgMOqar8RO` (`entry8_lm_per.txt`, `.json`) |
| entry 8 grid decodes (8 jobs, 4 arms x 2 LMs + 2 beam probes) | `work/speech_llm/sae/gua_lm_decode/GuaLmDecodeGridJob.*` |
| §1f prerequisite config | `config/sae_1f_prereq.py` -> `.../librispeech/configs/config_sae_1f_prereq_v1.py` |
| measurement code (+ CPU tests) | `recipe/2025-10-speech-llm/src/speech_llm/sae/channel_audit.py`, `test_channel_audit.py` |
| audited inventory (codebook) | `work/speech_llm/sae/quantize_states/QuantizeStatesJob.FWpGhC941JMi` |
| audited unit stream (tc100 + LBS dev) | `work/speech_llm/sae/quantize_states/MergeUnitsPklJob.ncxcd3vouD5E` |
| superseded 50 Hz codebook (comparison row) | `work/speech_llm/sae/quantize_states/QuantizeStatesJob.c5H3nY2G1VIz` |
| gold sil-free phone sequences | `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/eval/GoldPhonesJob.ZGSp0hxyd2YP` |
| (i) audit, dev-clean / dev-other | `work/speech_llm/sae/quantize_states/AuditAvUnitsJob.hPXeQoupqWBa` / `.GFdWI6Kzfhkr` |
| (ii) channel structure, dev-clean / dev-other | `work/speech_llm/sae/channel_audit/ChannelStructureJob.a2QG6jbb5Fzp` / `.G98AobA396ha` |
| rate-reduced rows of the same encoder tap (12.5 Hz pooling, prior codebook) | `work/speech_llm/sae/quantize_states/AuditAvUnitsJob.zzZk9wq8vBfe` |
| battery config | `config/sae_1f_battery.py` -> `.../librispeech/configs/config_sae_1f_battery_v1.py` |
| battery code (+ CPU tests) | `recipe/2025-10-speech-llm/src/speech_llm/sae/repr_pool.py`, `match_screen.py`, `test_repr_pool.py`, `test_match_screen.py` |
| encoder features the pooling reads (10 h seed dump, train+dev) | `work/speech_llm/sae/av_states/AvStatesJob.c4Ak1rACchRC` |
| pooled streams `seg16` / `seg12.5` / `seg9` | `work/speech_llm/sae/repr_pool/SegmentPoolUnitsJob.IHRNqQfnxrQ3` |
| `brown100` / `ubpe12.5` streams | `work/speech_llm/sae/repr_pool/RelabelUnitsJob.BA6DGayY7B5F` / `.mkk17SxDKjG2` |
| battery audit, dev-clean / dev-other | `work/speech_llm/sae/quantize_states/AuditAvUnitsJob.LhqtknKw7dzh` / `.KBO9vGKFDjPT` |
| battery screen, dev-clean / dev-other | `work/speech_llm/sae/match_screen/MatchScreenJob.8Dw01CNHAXYZ` / `.tmCr93GgmkVH` |
| channel structure per representation, dev-clean | `ChannelStructureJob.zoS0O0t0pB5Z` (seg16) / `.m06rcKTBVYuq` (seg12.5) / `.Pvg3LvrVVgU4` (seg9) / `.x0kMX4nLb0Rb` (brown100) / `.seXXeYp8aOyZ` (ubpe12.5), under `work/speech_llm/sae/channel_audit/` |
| channel structure per representation, dev-other | `ChannelStructureJob.BxyUz8Fha84d` (seg16) / `.Xf4J9E9gNiz4` (seg12.5) / `.hisUE5DAz6EF` (seg9) / `.NgWhyKTWYYMV` (brown100) / `.bCFN8Q1pORtz` (ubpe12.5) |
| entry-3 config | `config/sae_1f_entry3.py` (+ `sae_1f_entry3_fp.py`) -> `.../librispeech/configs/config_sae_1f_entry3_v1.py` |
| entry-3 code (+ CPU tests) | `recipe/2025-10-speech-llm/src/speech_llm/sae/fingerprint_match.py`, `test_fingerprint_match.py` |
| unpaired text the match reads (`T_phi`) | `work/i6_experiments/users/wu/experiments/posterior_hmm/data/phon_lm/TextToPhonemeJob.THKMON3k9LJQ` |
| entry 3, dev-clean / dev-other | `work/speech_llm/sae/fingerprint_match/FingerprintMatchJob.O4dpJTesB66u` / `.MHmUIV85g8Ry` |
| unpaired word text the kill-test reads | `work/i6_core/tools/download/DownloadJob.g4jClO48cAvP` (`librispeech-lm-norm.txt.gz`), BPE-512 codes `work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.su17F7YEcwEr` |
| entry 6 kill-test (unit-words) | `work/speech_llm/sae/fingerprint_match/UnitWordProfileJob.vULzsMp1oise` |
| entry 6 onset control (approach 6) | `work/speech_llm/sae/onset_control/OnsetControlJob.wGkGvMmpWF5V` |
| onset-control config and code (+ 19 CPU tests) | `config/sae_1f_onset.py` -> `.../librispeech/configs/config_sae_1f_onset_v1.py`; `recipe/2025-10-speech-llm/src/speech_llm/sae/onset_control.py`, `test_onset_control.py` |
| ruling-3 config and code (+ CPU tests) | `config/sae_1f_ruling3.py` -> `.../librispeech/configs/config_sae_1f_ruling3_v1.py`; `recipe/2025-10-speech-llm/src/speech_llm/sae/lexfree_match.py`, `test_lexfree_match.py` |
| ruling-3 unit-word streams (approach 7) | `work/speech_llm/sae/lexfree_match/UnitWordStreamJob.jwEGIPgoOuy5` (seg9) / `.eIxgmMh99RSE` (seg12.5) / `.mPnLApAbYnVG` (seg16) / `.5XnmEvOqh0TK` (ubpe12.5) |
| ruling-3 arm-gate reads (approach 7) | `work/speech_llm/sae/lexfree_match/LexFreeMatchJob.PQGETAeQAVaZ` (seg9) / `.rk48Zk5U6jzW` (seg12.5) / `.iY14L0buio5T` (seg16) / `.qw7Q0eDiq2hW` (ubpe12.5) |
| entry-5 config and code (+ 18 CPU tests) | `config/sae_1f_entry5.py` -> `.../librispeech/configs/config_sae_1f_entry5_v1.py`; `recipe/2025-10-speech-llm/src/speech_llm/sae/espum_match.py`, `espum_model.py`, `espum_jobs.py`, `test_espum_match.py`, `test_espum_model.py`, `test_espum_jobs.py` |
| entry-5 phone language model (4-gram over `T_phi`) | `work/i6_core/lm/kenlm/KenLMplzJob.0aJeN88X6EdW` -> `work/i6_core/lm/kenlm/CreateBinaryLMJob.hvZoC014xnIe` |
| entry-5 selection ids (dev-other, scored fifth excluded) | `work/speech_llm/sae/espum_jobs/EspumDevIdsJob.IqaCdokey92g` |
| entry-5 E1 supervised probe (approach 8) | `work/speech_llm/sae/espum_jobs/EspumProbeJob.5KJjR2SsYBJT` |
| entry-5 training arms (approach 8) | `work/speech_llm/sae/espum_jobs/EspumMatchTrainJob.mIXXRFodAMKs` (full s0) / `.lALR9ldNG8f1` (full s1) / `.ydfMFa6NdL4f` (full s2) / `.zokDElC71cF9` (bigram-only s0) |
| entry-5 gate reads (approach 8) | `work/speech_llm/sae/espum_jobs/EspumEvalJob.LPVdtT35Jzzw` (full s0) / `.WdKc8iOgFSzP` (full s1) / `.k9x16RkG3w5M` (full s2) / `.uvJWVebI6ffu` (bigram-only s0); label-free pick `EspumPickJob.W9HzeOEviPO4` |

Entry 7 (approach 9), the reference pipeline's own chain, in order:
`work/speech_llm/sae/gua_jobs/GuaAudioManifestJob.rdVx8r37h78h`, `GuaFeaturesJob.RIxCDjD5XPqW`,
`GuaClusterJob.baYUouLc37Ay`, `GuaTextJob.4mttbvA9Ut8f`; smoke `GuaTrainJob.3xMSCoAtUuDz` +
`GuaGenerateJob.RsHiB5ueKWdz` (discarded by construction); arms `GuaTrainJob.PZo12D74ij2M` (full)
and `.OfNoESzNJykY` (bigram-only). Recognition off each arm. Every hash below is the DERIVED
18-point-grid generation, re-read from the graph on 2026-08-18 after the grid change; the
20-point generation cited earlier is superseded and must not be quoted.
Full `GuaDecodeSweepJob.bVvWrIiZ2MwL` -> `GuaSelectJob.GQOGrA6sd9Ax` ->
`GuaGenerateJob.P7c0GNgIt8LK` -> `GuaScoreJob.CLeBGY6k2NnY`; bigram-only
`GuaDecodeSweepJob.yxg5kfzIJB33` -> `GuaSelectJob.EehyOxVHXBp7` -> `GuaGenerateJob.yM1kMat8ofMD`
-> `GuaScoreJob.TZSFN5FLubRJ`. Every pick runs on 18 checkpoints, not 20: the trainer writes no numbered snapshot where an interval save lands on an
EPOCH boundary, and the seed bed yields 14 updates per epoch, so updates divisible by both 14 and
2000 -- 14000 and 28000 -- leave only a `checkpoint_last` that is later overwritten. Verified absent
at both updates in both arms; 40000 is not epoch-aligned, so the final snapshot exists. The grid is
derived from that structure rather than listing the holes, so the relabeled arms inherit it.
Full-arm relabeling: iteration 2 `GuaGenerateJob.0jGtVQIDcorO` -> `GuaSegmentsJob.gqJOWVD6TtrW` ->
`GuaTrainJob.EwdQgD4XqYPI` -> `GuaDecodeSweepJob.wkYQ0fx9YeM1` -> `GuaSelectJob.b4LpanAratYe` ->
`GuaGenerateJob.EXCk3cLpB4kz` -> `GuaScoreJob.UeD8Vt2mL2fy`; iteration 3
`GuaGenerateJob.fxueYq266ZpF` -> `GuaSegmentsJob.ZmaDEaqnTAyS` -> `GuaTrainJob.NOS8BQ4IoPlM` ->
`GuaDecodeSweepJob.mWJV4jphL6g4` -> `GuaSelectJob.2XISV1CViuoE` -> `GuaGenerateJob.YeAgEwsze6io` ->
`GuaScoreJob.iYigGwYyzwIN`. Fixed-endpoint read at update 40,000, declared in advance and beside the selected one on the same arms: full `GuaPinCheckpointJob.knPFtsTVTdMN` -> `GuaGenerateJob.1FazO6bRnEb0` -> `GuaScoreJob.kClihi9eGTuy`; bigram-only `GuaPinCheckpointJob.738PI5sL150y` -> `GuaGenerateJob.ROUHUjlQOiKX` -> `GuaScoreJob.ykS6g9QXLRHv`. Each arm's optimizer trace is in its own job dir across
`log.run.1.backup-quota` + `log.run.1` (the file-count-quota restart replays epochs, so read one
record per `train_num_updates`).

## Verifier feedback

- 2026-08-16: internal-consistency checks pass (sub+ins+del reproduces every oracle-PER row;
  the superseded-codebook row reproduces its finished 0.424/0.189 exactly). The cataloged
  12.5 Hz rate-reduced rows (prior codebook, AuditAvUnitsJob.zzZk9wq8vBfe) read oracle PER
  0.423 with ins 0.013 but del 0.287 under that job's own grid-rasterization caveat —
  fixed-rate pooling removes the insertion mass at a confounded deletion cost, so the
  data-driven pooled rows of the CURRENT codebook are the deciding measurement. Planner fork,
  kill-(ii) bar, and the USER's gate replacement recorded in PLAN_1F.md (2026-08-16).
- 2026-08-16 (battery): rows verified against the job outputs directly
  (MatchScreenJob.tmCr93GgmkVH, AuditAvUnitsJob.KBO9vGKFDjPT: seg12.5 0.4135/0.0666/0.1172,
  ubpe12.5 string-map 0.436, brown100 1.152, the sigma_min column) — the log reproduces them.
  The two open planner calls are decided in PLAN_1F.md: matcher screens run on ALL pooled
  rungs plus ubpe12.5 (no label-selected rung enters the method; seg12.5 named primary, seg9
  the label-free-defensible rung); the tv_offdiag bar is VOID AS MEASURED per conclusion 9,
  entries 1/4 stay parked with no post-hoc replacement. Entry 2 is CLOSED by its sigma_min
  gate (conclusion 8). Arm-gate margin pre-registered before any matcher run.
- 2026-08-16 (entry 3): both FingerprintMatchJob outputs reproduce the logged table exactly;
  M1 arithmetic checks as min(n1,n2)-cand on every row; M2.frac confirmed as candidate/oracle
  swap-delta. The fingerprint set is narrower than the registered spec (no duration, no
  mid-utterance silence adjacency) — ratified, since neither has a text-side counterpart;
  the registered spec overpromised. The regularization sweep is diagnostic-only as claimed,
  and its reg-1 "gain" is marginal collapse (induced-marginal L1 1.2-1.4) — keep as the
  standing example of why the audio-swap read exists. Verdict recorded in PLAN_1F.md /
  PLAN.md: entry 3 NOT FUNDED; next per plan is entry 6's kill-test + the lexicon-free arm.
- 2026-08-16 (entry 6 + ruling-3 frame): approach-5 table reproduces
  UnitWordProfileJob.vULzsMp1oise exactly (hit counts, hitting ids, enrichments; the
  English row's clause logic checks — BUT length-excluded, IT final-excluded); every hit
  row confirmed units=1 in the job's own top-20 tables, and the two stall causes in
  conclusion 23 confirmed (38228 < 50000 vs exactly 50000). Verdict in PLAN_1F.md: kill-test
  CLEARS with conclusions 20/22 as recorded scope; the proposed utterance-initial oracle
  read on the hitting units is green-lit as an eval-only diagnostic. LexFreeMatchJob frame
  checked in code before any output was opened: [OTHER] exclusion covers candidate and both
  nulls, budgets 120000/1500000 as claimed, fingerprint identical to entry 3's. Two frame
  rulings recorded in PLAN_1F.md: 512 word types RATIFIED (adopted BPE-512 type count);
  unrestricted oracle OVERTURNED to restricted (ceiling must live in the candidate's map
  space; coverage column already prices the closed vocabulary) — re-run required, existing
  gate reads remain valid.
- 2026-08-16 (approach 6, onset control): the table reproduces OnsetControlJob.wGkGvMmpWF5V
  bit-for-bit on every hit row, and the silence-proxy inference in conclusion 25 is ratified
  as proven by construction — segment_tokens deletes masked types before any merge, verified
  in lexfree_match.py, so no sil_mask dump is needed. The conclusion-19/22 corrections and
  the standing "no label-free read" clause are endorsed as written (the control is eval-only
  and does not hand the separation to a label-free pipeline). Two flags, neither flipping a
  conclusion: the other-type median pool thins to 1-3 types at the deciding prefixes (403's
  call clears even the 12-type prefix-0 pool, so it stands); the approach-6 header says
  38228 merges where conclusion 23 says 38230 — pin which count the artifact carries.
  Amended verdict (precondition STANDS on direct evidence; ubpe12.5 proxy defect and its
  consequence for the running ruling-3 rung — no mid-flight change) recorded in PLAN_1F.md
  and PLAN.md.
- 2026-08-16 (merge-count resolution): the artifact carries 38228 (unitwords.json
  rows["seg12.5@1"].merges, re-read directly; onset control and the lexfree report agree)
  — conclusion 23's 38230 was a transcription slip off the txt table's 3.823e+04. The
  implementer's in-place fix is RATIFIED as a transcription-slip exception to the WRONG-
  marker rule (substance untouched; this bullet is the audit trail); the stale 38230 in the
  earlier verifier bullet and in PLAN_1F.md is corrected the same way.
- 2026-08-17 (ruling-3 batch amendments, verified before any screen output is read): the
  `ubpe12.5` stream defect CONFIRMED from ground truth — learn_unit_bpe's max_merges
  default 8000 (repr_pool.py:129) is not overridden at the build call (:437), and
  RelabelUnitsJob.mkk17SxDKjG2/output/units.stats.txt reads "8000 merges, vocab 500 ->
  8500" at measured 14.08 tok/s vs the 12.5 target: the stream stopped on the budget, so
  its vocabulary is a default's artifact and the rung is not rate-matched to `seg12.5`.
  Rulings (no rebuild this batch; operating point named on every read; matched-rate
  contrast retired; conditional rebuild follow-up) in PLAN_1F.md 2026-08-17. The
  rewritten LexFreeMatchJob verified in code: one restricted pass emits both ceilings
  from the same counts on the same held-out rows, and the resume/merge-checkpoint change
  is rng-safe (merge learning draws nothing from the job rng; its single consumer sits
  after the cache boundary) — determinism check pre-registered in PLAN_1F.md: seg-rung
  oracle-independent columns reproduce bit-for-bit, old `oracle` equals new
  `oracle_open` exactly. The prefix-1 floors (4.892/4.258/3.459) are consistent with
  conclusion 23's 4.26 but await batch-close verification; the 11 h timeout cause stays
  open pending the job's new per-stage timing.
- 2026-08-17 (ruling-3 batch close): every approach-7 body cell verified against the four
  lexfree.json artifacts (M1/M2/rates/prefixes exact; conclusion 29's 0.717/0.004
  confirmed), and the closest-rate prefix selection re-derived as the argmin over each
  rung's measured meta.json curve — all 12 selections reproduce, the rule leaves no
  discretion. Amendment (1)'s cross-generation determinism check PASSES exactly on all
  three seg rungs (oracle-independent columns bit-for-bit vs the open runs; old oracle
  == new oracle_open), and the job split reproduces the whole restricted generation
  bit-for-bit (md5-identical pairs on all three seg rungs). CORRECTION for the Catalog
  row and the approach-7 headers, no gate number touched: the seg12.5/seg16
  UnitWordStreamJob hashes are SWAPPED — mPnLApAbYnVG IS seg16 (43821 merges, fit
  360623, kept 906940), eIxgmMh99RSE IS seg12.5 (38228 merges, fit 313918, kept
  720336, matching entry 6's independent 38228) — while the LexFreeMatchJob labels and
  their stream wiring are correct (each diag matches its named stream). Conclusions
  27-29 ratified; scope guard on 28: its basis is the per-rung gate structure plus the
  all-fail table — the "worst despite more data" clause is confounded by K=8500-vs-500
  and the budget-stopped stream if ever read as representation attribution. Note for
  later readers: the restricted oracle can read BELOW oracle_open on held-out rows
  (seg12.5 words dev-clean 0.672 vs 0.715) — both are fitted on the ceiling-fit rows
  and scored held-out, where the smaller map space can generalize better; expected,
  not a defect. Verdict in PLAN_1F.md amendment (7) / PLAN.md: ruling 3 NOT FUNDED in
  all twelve cells; neither conditional follow-up triggered; the entry-5-vs-new-
  screen-vs-close-1f fork goes to the user.
- 2026-08-17 (implementer probe on the ubpe12.5 words cell, verified): the probe
  (scripts/probe_ruling3_map_ubpe_words.py) rebuilds both maps through the job's own library
  functions at the registered constants and anchors EXACTLY on the artifact (cand 1.4014,
  oracle 1.0667, live_a 114118 = K_audio); its stream is read from the meta name field
  (ubpe12.5). CONFIRMED and cell-specific: the restricted transcript-built ceiling in that
  cell is non-functional — PER 1.0667 vs the empty hypothesis's 1.000, word currency 1.4193,
  and it LOSES to the content-free pseudo-pair null n2 (0.8814) on the same held-out rows,
  so the cell cannot rank maps by content. Unique to ubpe words: the three seg words cells
  keep functional restricted ceilings (0.671/0.711/0.770 dev-other) and ubpe's open ceiling
  is functional too (0.699) — the 2026-08-16 restricted-ceiling overturn is what made this
  visible. Mechanism, by construction at the operating point: 114118 always-emitting audio
  symbols at 6.455/s against a MEASURED 2.786 reference words/s on the scored fifth
  (planner recomputation, speech_llm env) put any map in the candidate's class at a
  word-currency error floor of ~1.32 regardless of assignment quality; the implementer's
  probe-derived 2.35x net over-generation and function-word collapse (top-mass types ->
  THE, mean emitted word 2.65 phones vs 3.57 reference) corroborate the entry-6 signature.
  ONE NUMBER CORRECTED before it enters any record: the claimed 0.566 coverage floor is
  wrong — the cover column (0.4344) is FRAME-weighted by construction (gold_text_frames),
  and token-level coverage on the scored fifth is 0.7174, so the closed-vocabulary floor is
  0.283 in word currency; the rate, not the coverage, is the binding cause. Gate reads
  unaffected (M1/M2 are candidate-vs-null, ceiling-independent; the cell's M1 -0.520 row
  stands); conclusions 27-29 unaffected, 29's pricing sharpened. Planner ruling recorded in
  PLAN_1F.md addendum (7a): the row stays a gate FAIL, annotated uninformative about
  matching quality; the 512-type vocabulary and the 6.455/s floor are traceable registered
  constants needing no re-derivation; no word-level re-run is registered — reopening one is
  a frame redesign (rate gap, abstention/coverage) and a user decision.
- 2026-08-17 (later; CORRECTION to the previous bullet, implementer counter-argument
  verified): the "word-currency error floor ~1.32" claim is WRONG and retracted — the
  scored-fifth audio rate is 7.114 symbols/s (planner re-derived exact; 6.455 is the
  corpus-wide figure), and within-segment repeat collapse is map-dependent (implementer
  measurements on the anchored maps: restricted ceiling emits 1.742x reference tokens,
  shedding 32%; candidate 2.354x, shedding 8%; a constant map collapses each chunk to one
  token and lands near 1.0), so no class-wide error floor above 1.0 follows from the rate.
  What stands: the rate mismatch (2.55x scored-fifth emission capacity) is the dominant
  structural cause and dwarfs the coverage effect (token floor 0.283); the
  collapse-fraction contrast is itself diagnostic — consecutive symbols inside one spoken
  word share the ceiling's target but not the adjacency-blind fingerprint candidate's; and
  the crispest uninformativeness statement remains n2 0.8814 beating the restricted
  ceiling 1.0667 on the same held-out rows. Refinement on the cover column: it is the
  fraction of ALL 50 Hz frames (silence included) carrying an in-vocabulary word — a third
  quantity (implementer: 0.5404 within-word-interval frame coverage vs 0.4344 recorded;
  in-vocab words average 10.87 frames vs 23.47 OOV). PLAN_1F.md (7a) amended in place.
- 2026-08-17 (text-sample coverage defect, surfaced by the entry-5 grounding sweep;
  affects entry 3 and all twelve ruling-3 cells, NOT entry 6): sample_text_lines(stride=80,
  max_lines=300000) stops at source line 23,999,921 of 39,630,169 (60.6%), and
  librispeech-lm-norm is ALPHABETICALLY sorted by sentence, so sentences beginning roughly
  P-Z never entered the screens' text-side statistics — fingerprints, marginals, n1
  quotas, and n2 pairing draws alike. Candidate and both nulls consumed the same truncated
  sample, so every gate comparison stays internally consistent and the all-fail verdict is
  NOT reopened; the defect is a frame limitation to disclose wherever those text
  statistics are quoted. Entry 6's text row (stride=400, max_lines=100000; 40.0M >= 39.6M
  lines) is unaffected. Standing fix registered in PLAN_1F.md (7b): every future
  text-side statistics pass must satisfy stride*max_lines >= corpus lines; the entry-5
  spec pins the proven full-coverage stride-400 sample.

- 2026-08-17 (E1 verified, interpretation registered): EspumProbeJob.5KJjR2SsYBJT
  probe.json read first-hand — name entry5_seg12.5_probe, dev-other, n=572, PER 0.35647
  (sub 0.12957, ins 0.02133, del 0.20557) — matches the log row exactly; all four
  train-arm hash-to-arm mappings re-verified from each job's own info name field
  (entry5_full_s0/s1/s2, entry5_bigram_s0) — Catalog correct. RULING: E1 is a one-sided
  floor test and PASSES at 0.3565 vs the 0.4148 memoryless oracle-map ceiling — the
  probe's kernel-4 context window is a strictly richer function class than a
  one-phone-per-unit map, so landing BELOW the ceiling is the expected pass outcome, not
  an overshoot; the failure E1 guards against was landing far above it. 0.3565 is a
  supervised eval-only read, not an entry-5 performance claim; no gate constant changes.
  Watch item for the health read: deletions dominate even supervised (0.2056 of 0.3565,
  insertions 0.0213; expected at 9.771 seg/s vs 9.86/s reference phone rate after
  duplicate collapse) — judge the four arms' collapse signature on the sub/ins/del split
  alongside total PER, since under-generation has little slack before deletions alone
  breach the M1 bar.

- 2026-08-17 (entry-5 batch verified at close): all five artifacts read first-hand
  (EspumEvalJob.WdKc8iOgFSzP/.LPVdtT35Jzzw/.k9x16RkG3w5M/.uvJWVebI6ffu,
  EspumPickJob.W9HzeOEviPO4) and every table number matches to the fourth decimal; M1/M2
  re-derived from the raw PERs and swap PERs against the banked rk48Zk5U6jzW nulls —
  arithmetic exact; hyp/ref ratios and the 80.5% substitution share reproduce. Rulings
  (recorded in PLAN_1F.md entry-5 Status): gate FAIL both clauses on the label-free pick
  full_s1 (M2 named close, M1 not); health checkpoint passed as written, absent
  bigram-vs-full separation recorded as observation, not verdict; seg9 contingency NOT
  exercised (identity-dominated failure at near-correct emission rate — rung choice not
  decision-relevant); entry 5 closed, BPE follow-up condition unmet, 1f returned to the
  user. Caution registered: the selection metric's clean ablation separation (53.86 vs
  31-33) with indistinguishable PER licenses it for checkpoint/seed picking only, never
  method ranking. Implementer's stop-and-report at the failed bar was the registered
  behavior — correct.

- 2026-08-18 (entry-7 recognition chain, submission verified pre-run; commit 4e7550b): code
  read in full and independently re-tested — the full sae/ sweep reproduces exactly 630
  passed, and the 7 new gua_recog tests cover the claimed traps (2000-vs-12000 checkpoint
  resolution, argmin pick + symlink, sub/ins/del decomposition, short-decode count assert).
  Selection-metric parity with entry 5 PROVEN from the banked artifact: the committed espum
  code (first checked in by this same commit; worktree == HEAD) recomputes
  EspumPickJob.W9HzeOEviPO4's three curves to <= 1.2e-4 relative with the picked arm's token
  count exact and the winner full_s1 reproducing; the residual is quantified GPU-vs-CPU
  argmax near-ties (12-73 of 143,307 selection segments under 1e-3 logit margin), not code
  drift. Pooling claim CONFIRMED at source: under the l1 config's BINARY segmentation the
  boundary head pools label-free at training and generation alike (wav2vecu_graph.py:1299-1307;
  :1145-1146 forces BINARY back at generate because segment_weight > 0; bin_labels feeds only
  the boundary BCE at :1686-1690); our data dirs carry no .src file. Recorded trap: the
  bundle's FIXED/JOIN branches DO silently change pooling when bin_labels is absent —
  unreachable here, load-bearing for any future segmentation-type change. Margin claim proven
  at source: margin reaches only the segment outputs (w2vu_generate.py:330-333, :569), never
  gen_hypos, so the symmetric margin-0.0 scored decodes are exact.
- 2026-08-18 (same submission — two status corrections and one DEFECT, all from the arms'
  own logs): (1) the EDQUOT halt was at update 14000, not 12000 — both backup logs reach
  num_updates 14000 before the Errno 122 crash killed the checkpoint save; 12000 was the
  last persisted checkpoint and hence the resume point, ~2000 updates recomputed per arm.
  (2) The update-14000 numbered checkpoint is permanently ABSENT in BOTH arms: on resume the
  14000 boundary coincided with an epoch end, fairseq wrote only checkpoint_last (since
  overwritten at 20000). DEFECT in the chain as wired: SELECT_GRID includes 14000 and
  _checkpoint_at asserts exactly one hit, so both iteration-1 sweeps would crash — grid
  amendment ruled in PLAN_1F.md (19-point grid for the iteration-1 arms only; the hole is
  symmetric across arms, so selection stays arm-fair). Arms verified RUNNING (SLURM
  1402939/1402940), numbered checkpoints through 20000 and live logs past update
  20700/21000 at 14:42; the clipping regime persists (full arm gnorm ~2332 at 100 % clip,
  bigram-only ~593 at 95-100 %) — the registered clip/gnorm table columns stay mandatory.
- 2026-08-18 (grid amendment verified; commit 52fedaa, 14:55): the ruling is implemented
  exactly — SELECT_UPDATES_ITER1 (19 points, 14000 absent) reaches only the two
  iteration-1 sweep/pick pairs, iteration-2/3 keep 20 points, _checkpoint_at's
  exactly-one assert is untouched, and the pick artifact now names any grid hole
  (updates + missing_from_grid in pick.json, hole line above the printed curve, with the
  two-point-grid limitation honestly declared); 632 tests reproduce, the two new ones
  read as claimed. Catalog hashes are graph-computed pre-run — labels to be re-checked
  off each job's own name field once the dirs exist. Sequencing fact: the entry-7
  manager was restarted at 14:39, BEFORE the amendment landed, so the live graph held
  the pre-amendment jobs at verification time; a second restart (user action) was
  requested to supersede them before the arms finish (~22:30) — the pre-amendment
  recognition jobs never ran and no dirs exist.
- 2026-08-18 (CORRECTION to the two bullets above; implementer-found, planner-verified from
  both arms' checkpoint listings): the missing update-14000 snapshot was NOT caused by the
  EDQUOT outage. The cause is deterministic — at ~14 updates per epoch (2000 updates = 143
  epochs), any update divisible by both the 2000-update save interval and the per-epoch count
  falls exactly on an epoch end, where the trainer writes only `checkpoint_last` under
  no_epoch_checkpoints, so the numbered file is never created. It therefore RECURS: update
  28000 is also absent in BOTH arms (verified directly — each lists 2000-12000, 16000-26000,
  30000-36000, with 14000 and 28000 the only gaps), and 28000 was lost with no quota event
  anywhere near it, which is the observation separating the two explanations. The outage did
  strike at update 14000, and that coincidence is exactly what made a one-off cause look
  sufficient — a recurring rule was inferred from a single symptom, against the standing
  principle that repeated failures point upstream to a shared cause. CONSEQUENCE: the
  19-point ruling was still wrong and would have crashed both iteration-1 sweeps at 28000;
  the corrected grid is 18 points (minus 14000 and 28000), recorded in PLAN_1F.md by
  replacement. Iteration-2/3 grids need their own epoch arithmetic checked, not assumed.
- 2026-08-23 (entry 8 cells 1-2 launch round VERIFIED AND ACCEPTED; three proposals ruled and
  the anchor pin discharged in `PLAN_1F.md` entry 8 Status 2026-08-23 launch ruling).
  Verified: 8 grid-decode dirs on disk plus the two reads and two LM jobs, matching the
  twelve-job accounting; the module docstring carries the sil_weight and SIL-free-primary
  decisions verbatim; the env guard fired correctly under the default environment (the
  suite runs only in the w2vu env -- the guard firing IS the env discipline observable, and
  the implementer's 'tested before launch' is accepted on that basis plus the disclosures
  verified in source). The hand probe (PER 1.6828 -> 0.9322, ins 1.0650 -> 0.0979 on the
  full arm's scored fifth) is accepted as a mechanism check only. All three proposals were
  the right kind of stop: each names a registered constant the code or data cannot support,
  none was worked around silently. Rulings and the discharged anchor pin (0.473 = argmax
  currency, no LM, no self-training; entry-8 numbers are a different currency with no
  like-for-like published counterpart) are in the plan; the pin text goes into
  `gua_lm_decode.py`'s docstring before any result is read.
- 2026-08-23 (entry 8 cells 1-2 result VERIFIED; three result rulings in `PLAN_1F.md` entry 8
  Status 2026-08-23 result ruling). Both read artifacts reproduced line for line by the
  planner: every Approach table number, decomposition, range, pick marker, probe delta and
  agreement matches `entry8_lm_per.txt` in `GuaLmGridReadJob.SeNSdRhV1Wo3` / `.I9lgMOqar8RO`,
  and the payload prints the discharged anchor pin and consumes the beam-500 probe (both
  `55045ed` gaps closed pre-read as claimed; zero error markers on all ten jobs). Verdicts
  E8.1, E8.3, E8.4 are faithful to the artifact. TWO PRECISION NOTES, neither flipping a
  verdict: (i) E8.2's "identical decodes at all three values ... in every cell" is slightly
  too strong -- at lm 4 the three sil_weight rows differ by up to 0.0001 PER and 2 hypothesis
  phones (full endpoint 17212/17213/17214), which is the near-tie reshuffling the
  eos-fall-through mechanism itself predicts; the axis is inert to any decision-relevant
  precision, and the retirement ruling stands on the measured 0.0001 bound, but the verdict's
  universal phrasing should carry that bound (implementer's amendment). (ii) The SIL-augmented
  sensitivity paragraph said "slightly worse throughout" while its own numbers show the
  bigram-only best BETTER (0.8145 vs 0.8172); objectively wrong reference corrected in place
  by the verifier, conclusion unchanged. The E8.3 anti-selection finding is verified in the
  artifact (sel wppl 43.03 at the picked lm 0.5 vs 431.14 at the oracle cell on the full
  endpoint arm -- per-token perplexity rewards the 64,586-phone decode) and its mechanism
  matches the banked per-token-mean-pays-for-length principle; the tightened TRIPLE quoting
  rule and the sil-axis retirement are in the plan. Cells 3-4 stay with the user; cell 4's
  stakes are raised, as ruled.
