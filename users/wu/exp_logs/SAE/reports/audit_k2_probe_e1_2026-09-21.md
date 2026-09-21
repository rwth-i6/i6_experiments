# Audit: the five k2 settling probes against efficiency gate E1 (2026-09-21)

Fresh-context, read-only. Nothing was edited, rerun or repaired. Every number below was
re-derived from `output/per_batch.json` and `output/probe.json` with an independent script, not
read off `summary.txt`.

Artifacts audited (all five job dirs under
`/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/lexlat_k2_jobs/`):
`LexlatK2ProbeJob.Tkl94t85j4pV` (in-house HLG `LexlatHLGBuildJob.rtX44PBJFNy1`, all-states
back-off loops), `.by7UYEjtYdel` (in-house `LexlatHLGBuildJob.cdcxYJMjiYj5`, `backoff_loops` key
present = word-boundary only), `.R7QzD6vYBLD3` (`LexlatOfficialHLGBuildJob.YJsdBTcEQJz9`, 3-gram
pruned 3e-7), `.5dP7W2NMFq4T` (`.GxCkDk90bQpT`, 3-gram pruned 1e-7), `.DtWYToXTPh6w`
(`.NrghE6fnf5hc`, 4-gram, theta 5.0). Code read at
`recipe/2025-10-speech-llm/src/speech_llm/sae/emc/lexlat_k2.py` (`run_probe`,
`chunked_tot_scores`), `lexlat_k2_jobs.py` (`LexlatK2ProbeJob.run`, `probe_reads_and_summary`) and
`lexlat_k2_train.py` (`LexlatK2Runtime`), at commits 3454206 and 0c812b1.

## Verdict

**The two in-house PASS verdicts: CONFIRMED.** Every per-rung number reproduces exactly, all 27
cells are real, and the PASS survives the stricter of the two readings of the bar.

**The three official-LM FAIL verdicts: the arithmetic is CONFIRMED but the verdict word does not
answer the gate's question.** All three fail on one clause of one rung (`max_active` 10000
exceeds 80 GiB). At the rung the design actually selects -- amendment 9.3, "the smallest probe
rung that PASSES the E1 bar AND E0's stability rule" -- all three pass both clauses and the
stability rule. `summary.txt`'s `VERDICT` is computed by a stricter rule of the writer's own
("every planned cell measured AND *every* rung passes"), which is not the registered gate.

## 1. Do the numbers reproduce? Yes, exactly; 27/27 cells are real

Re-derived mean seconds per batch x 57 and max peak reserved over the 9 cells of each rung:

| job | rung | mean s/batch | x57 (bar 1202) | peak reserved GiB (bar 80) | median gap/retained frame |
|---|---|---|---|---|---|
| Tkl94t85j4pV | 1000 | 5.422 | **309.0** PASS | **43.28** PASS | 0.0337 |
| Tkl94t85j4pV | 3000 | 6.184 | 352.5 PASS | 46.29 PASS | 0.0141 |
| Tkl94t85j4pV | 10000 | 8.350 | 476.0 PASS | 62.25 PASS | 0.0000 |
| by7UYEjtYdel | 1000 | 3.185 | **181.5** PASS | **23.38** PASS | 0.0284 |
| by7UYEjtYdel | 3000 | 3.822 | 217.9 PASS | 27.19 PASS | 0.0125 |
| by7UYEjtYdel | 10000 | 5.434 | 309.7 PASS | 38.11 PASS | 0.0000 |
| R7QzD6vYBLD3 | 1000 | 12.627 | 719.7 PASS | 62.43 PASS | 0.0285 |
| R7QzD6vYBLD3 | 3000 | 13.709 | 781.4 PASS | 67.82 PASS | 0.0126 |
| R7QzD6vYBLD3 | 10000 | 16.716 | 952.8 PASS | **83.30 FAIL** | 0.0000 |
| 5dP7W2NMFq4T | 1000 | 7.985 | 455.1 PASS | 59.72 PASS | 0.0298 |
| 5dP7W2NMFq4T | 3000 | 9.240 | 526.7 PASS | 70.59 PASS | 0.0129 |
| 5dP7W2NMFq4T | 10000 | 12.357 | 704.4 PASS | **88.90 FAIL** | 0.0000 |
| DtWYToXTPh6w | 1000 | 17.213 | 981.1 PASS | 69.71 PASS | 0.0316 |
| DtWYToXTPh6w | 3000 | 18.079 | 1030.5 PASS | 72.64 PASS | 0.0135 |
| DtWYToXTPh6w | 10000 | 20.858 | 1188.9 PASS | **88.54 FAIL** | 0.0000 |

Every figure matches `summary.txt` to the printed digit. The x57 rule is the gate's
(`probe_reads_and_summary` uses `fmean(seconds) * n_total` because `covers_all` is false).
Cell integrity: 27 of 27 planned cells present in all five jobs; no zero or missing timing; the
nine per-rung second values are all distinct; `seconds` equals
`seconds_intersect + seconds_tot_scores + seconds_backward` to the last bit in all 135 cells;
seconds, peak memory and lattice arc counts rise monotonically with `max_active` in every batch;
`tot_scores` vectors differ between rungs and between graphs (spot-checked), so no cell is reused
across rungs or copied between jobs. `log.run.1` (or the copy inside `finished.tar.gz`) carries
one printed line per cell agreeing with the json. The five jobs ran on five distinct exclusive
nodes (jpbo-040-23, -068-03, -035-46, -029-46, -037-19), so there was no GPU contention between
them.

## 2. The timed region is what the bar says

`chunked_tot_scores` (lexlat_k2.py:976-1041) brackets, per chunk: `k2.DenseFsaVec` +
`intersect_dense_pruned`, then `get_tot_scores(log_semiring=True)`, then `tot.sum().backward()`.
Each of the three is preceded and followed by `torch.cuda.synchronize()` (`_sync()`, active
because `dense_in.is_cuda`), and `torch.cuda.synchronize()` is called once more immediately
before the region opens (lexlat_k2.py:1173). The CPU-side `lens.cpu()` and the `seg` `torch.stack`
sit outside the clock. The chunk loop is *inside* the timer and the three stage times are summed
over chunks, which is the true serial cost. Nothing else is inside: the max-plus sanity pass is
taken after the region, behind a synchronisation, with its own timer and its own peak reset, and
is reported separately (0.02-0.03 s/batch, about 0.3% of the timed figure).

Memory: `torch.cuda.empty_cache()` then `torch.cuda.reset_peak_memory_stats()` per cell, then
`torch.cuda.max_memory_reserved()` read at the end of the chunk loop -- torch's running maximum,
i.e. the MAX over chunks, never their sum, as the summary states. The bar is read on reserved,
the stricter statistic. Because `reset_peak_memory_stats` resets the peak to the *current* value,
the HLG already resident on the card is counted in the peak. This item is clean.

## 3. The batches are the planned ones, at the run shape

`plan = [0, 1, 14, 15, 28, 29, 43, 44, 48]` in all five jobs, and I reproduced it from
`LexlatEfficiencyProbeJob._plan` with `n_steps=100`, `n_points=4`, `max_timed_batches=9`:
`width = max(1, 8//4) = 2`, window starts `round(i*57/4)` = 0, 14, 28, 43, each contributing two
batches, plus `longest = 48`. Sub-epoch 10 of the arm's own RETURNN config
(`output/returnn.config`: `batch_size = {"features": 88000}`, `max_seqs = 128`,
`seq_ordering = "laplace:.1000"`, `partition_epoch = 4`), asserted against the declared values
before a frame is dumped. `random_seed = 42` is recorded in `probe.json`; `random_seed_offset` is
`null` (the train dataset carries no seed offset). The loader is walked twice and the second
walk's shapes are asserted equal to the first.

Actual shapes (identical in all five jobs, confirming a deterministic loader):

| index | B | T padded | T recognizer | chunks |
|---|---|---|---|---|
| 0 | 114 | 770 | 257 | 8 |
| 1 | 127 | 689 | 230 | 8 |
| 14 | 128 | 687 | 229 | 8 |
| 15 | 113 | 764 | 255 | 8 |
| 28 | 128 | 615 | 205 | 8 |
| 29 | 128 | 651 | 217 | 8 |
| 43 | 128 | 582 | 194 | 8 |
| 44 | 128 | 619 | 207 | 8 |
| 48 | 77 | 1137 | 379 | 5 |

B x T padded is 74,496-87,936 in every batch, i.e. at the 88,000-frame budget and at or below
`max_seqs` 128. The longest-T batch of the sub-epoch (index 48, T = 1137) is present and flagged
`is_longest`. 1071 utterances in total.

**The gate's premise about that batch is empirically false, and the sample is biased upward.**
Index 48 is the *cheapest* cell in every rung of every graph (e.g. 6.147 s against 7.7-9.2 s on
Tkl94t85j4pV at 10000) and it does not set the memory peak on Tkl94t85j4pV, by7UYEjtYdel or
5dP7W2NMFq4T. Chunking at 16 sequences is why: the batch has only 77 sequences and 5 chunks.
Separately, the four sampling windows all land on the long side of the laplace sawtooth: over the
whole 57-batch sub-epoch the mean of the proxy `ceil(B/16) x T_recognizer` is 1529, while over the
9 sampled batches it is 1805, a factor 1.18 (mean real frames: factor 1.16). The troughs of the
period (indices 7-8, 23-24, 39-40, T padded 257-352) are never sampled. Fitting seconds against
that proxy on the 9 cells and summing over all 57 batches gives, e.g. for Tkl94t85j4pV at 10000,
459 s against the reported 476 s, and for DtWYToXTPh6w at 10000, 1197 s against 1189 s. **The
direction of this bias is conservative** -- the reported x57 figure is at or above a
shape-corrected estimate in 14 of 15 cells -- so it cannot have manufactured a PASS. It does mean
the x57 number is an upper estimate rather than an unbiased one.

Also worth stating plainly: Design 7 as written asks for "100 steps sampled at four evenly spaced
points of a sub-epoch". The sub-epoch has only 57 batches, so 100 is unreachable; the job caps the
sample at `max_timed_batches = 9` (16% of the sub-epoch) rather than timing all 57, which would
have given the exact sum and no extrapolation at all. The cap is a constant of the probe job, not
of the gate.

## 4. The emissions are the real posterior, not a cached or random tensor

`LexlatK2ProbeJob.run` builds the arm's model through RETURNN's `Engine`, loads
`PackedBlankfreeTrainJob.ks7CbtlvpcIL/output/ctrl_50/models/epoch.010.pt` (listed as an INPUT in
`info` and in `log.run.1`), calls `model.train()`, sets `model.lexlat = None`, asserts
`permute_frames_seed is None`, and for each planned batch seeds `rf.set_random_seed` the way the
train step does, then evaluates `model.recognizer(feats.raw_tensor.float(), lengths)`. The dumped
`log_q` has shape `(B, ceil(T/3), 40)`, asserted against `lattice_cfg.n_phones`. `tau` is
`model.temperature(10)`; the config's `temperature_schedule` is `[8.0, 5.0397, 3.1748, 2.0, ...]`,
so index 9 is 2.0 and `probe.json` records `temperature = 2.0`. Both the dense emissions and the
HLG arc scores are divided by tau (lexlat_k2.py:1116 and :1158), which is what the bed's DP does.
The graph's `n_phones` / `sil_id` / `d_min` / `recognizer_stride` are asserted against the live
`LatticeConfig` before a frame is dumped. `grad_l2` is 142-146 on batch 0 in all five jobs, and
`grad_finite` / `grad_nonzero` are true in all 135 cells. This item is clean.

## 5. The stability column, and which rung the rule selects

Computed per utterance as `abs(z(ma) - z(10000)) / retained`, where `retained` is the utterance's
feature-frame count -- the same quantity `l_tau` divides by, which is E0's declared convention
(`lexlat_jobs.py:613`: "`retained` is the utterance's unit-frame count, the divisor `l_tau` itself
uses"). Medians re-derived (they match the summary to four decimals); pooled values added because
E0's clause 2 was registered as a pooled read:

| job | ma 1000 median / pooled / max | ma 3000 median / pooled / max |
|---|---|---|
| Tkl94t85j4pV | 0.0337 / 0.0351 / 0.0881 (n=1070) | 0.0141 / 0.0152 / 0.0526 |
| by7UYEjtYdel | 0.0284 / 0.0299 / 0.1156 | 0.0125 / 0.0132 / 0.0475 |
| R7QzD6vYBLD3 | 0.0285 / 0.0296 / 0.0954 | 0.0126 / 0.0129 / 0.0343 |
| 5dP7W2NMFq4T | 0.0298 / 0.0307 / 0.0823 | 0.0129 / 0.0134 / 0.0321 |
| DtWYToXTPh6w | 0.0316 / 0.0326 / 0.1283 | 0.0135 / 0.0142 / 0.0558 |

Median and pooled agree to within 0.002 everywhere, so the choice of statistic does not move the
rule. **The rule of amendment 9.3 selects `max_active = 1000` on all five graphs**: on each graph
1000 is the smallest rung, it passes both clauses of the bar, and its median is <= 0.05.

Two caveats on the column. First, the denominator matters a lot: `retained` is at the 50 Hz
feature rate, while the lattice runs at the recognizer's stride-3 rate. Had the gap been divided
by recognizer frames instead, the ma=1000 medians would be about 3x larger (roughly 0.085-0.10)
and would miss 0.05 on every graph. The convention used is E0's own, so this is not a defect, but
the rule's margin is a factor of 3 in the choice of frame rate. Second, the reference is
`max_active = 10000`, the top of the ladder, not an unpruned intersection; the residual gap from
10000 to unpruned is not measured here (E0 did carry an unpruned arm on its 20 shortest
utterances; this probe does not).

## 6. What would make the probe's seconds and GiB an underestimate

Four items, in decreasing order of how much they matter. None of them is hidden -- the summary
discloses the leg-only scope in words -- but none is quantified either.

(a) **The training runtime does strictly more per step than the probe times.**
`LexlatK2Runtime.step` (lexlat_k2_train.py:399-400) calls `log_z_hlg` *and* `log_z_h`, a second,
UNPRUNED `k2.intersect_dense` against the bed's H topology, because amendment 9.1's term is
`log Z_HLG - log Z_H`. It then calls `_expected_words`, which runs `lattice.get_arc_post(...)` --
a full log-semiring forward-backward over every chunk's lattice -- for the amendment 9.9 monitors.
Neither is in the probe's timed region. H is small (40 phones, d_min 2) and the monitors run under
`no_grad`, so both are probably modest next to a 24.7M-state HLG intersection, but "probably" is
all the artifacts support: the cost is unmeasured.

(b) **The probe's memory read does not bound the arm's.** The probe calls
`chunked_tot_scores(..., backward=True)`, which runs `backward()` per chunk and frees that chunk's
autograd graph before the next chunk starts; the arm calls it with the default `backward=False`
(lexlat_k2_train.py:343-348), deliberately, because a training step needs one backward over the
whole loss. All 8 chunks' autograd graphs are therefore alive simultaneously in the arm and only
one at a time in the probe. The probe's retained lattices are in its peak, so the difference is
the saved-tensor tape of 7 extra chunks, not a factor of 8 on the whole figure -- but it is
positive and unmeasured.

(c) **The memory clause is measured in an isolated process.** The k2 child is a separate process
that holds the HLG, the dense tensor and the lattices and nothing else; `del model, engine, ...`
and `torch.cuda.empty_cache()` run before it starts. A real arm holds the recognizer, the reverse
model, the bed's lattice DP tables and the optimizer state on the same card at the same time, and
the 80 GiB bar was written "against the per-arm marker of 96", i.e. about the arm's card. The
leg-only reading is registered for route A (`SAE_4A_lexlat.md` line 314), so this is not a moved
bar, but under it the memory clause no longer bounds what the arm will actually reserve.

(d) **`empty_cache()` before every cell** removes allocator fragmentation that a continuously
running training step would carry. Small, same direction.

Two things that are *not* underestimates, contrary to the usual suspects: the graph being held on
GPU across batches is correct (the arm holds it resident too, and `reset_peak_memory_stats` counts
it in the peak); and the backward stopping at the emissions is correct, because the recognizer's
own backward already exists inside the bed's 601 s and the lexicon term only adds to the emission
gradient. All 1071 sequences of each batch are intersected -- no sequences are dropped.

## 7. The one non-finite total score

It is on `Tkl94t85j4pV`, `max_active = 1000`, batch index 15, position 95 in the batch:
`log Z(1000) = -inf` while `log Z(10000) = -218.084` (finite), `retained = 733` frames. Meaning:
at that pruning budget the intersection lost *every* word-decomposable path for that utterance --
an empty lattice, amendment 9.6's `lexlat_k2_n_empty` case -- so in training it would contribute
no lexicon term for that step. At 1 in 1071 (0.09%) it is far below amendment 9.6's 2% sub-epoch
and 10% per-batch thresholds, so it does not read the rung uninformative. It does not appear at
3000 or 10000 on that graph, and no other graph has one.

**What it means for the read** is narrower than the count suggests: `probe_reads_and_summary`'s
gap loop skips any pair where either score is non-finite, so this utterance is *excluded* from the
ma=1000 stability statistic (n = 1070 against 1071 at the other two rungs). The one utterance
where pruning did the most damage -- an infinite disagreement -- is the one dropped from the
median that gates the rung choice. With 1070 samples the median moves by nothing, so the
selection does not change; but the statistic is, by construction, blind to exactly the failure
mode it is meant to detect, and the count is only disclosed on the adjacent line.

## 8. The frame: is the comparison the gate's question?

Three findings, one of which is decisive for how the three official-LM verdicts should be read.

**(i) The bar is applied to the lexicon leg alone, and that reading is registered.** Design 7's
1202 s is "2.00 x the bed's 601 s" for a sub-epoch; the k2 leg is *additive* to the bed (amendment
9.1: two graphs, the bed's lattice term unchanged), so an arm's sub-epoch would be about
601 + leg. Applying 1202 to the leg alone is therefore the more permissive of the two readings.
It is not a post-hoc restatement -- `SAE_4A_lexlat.md` line 314 registers route A as "the lexicon
leg measured against the SAME E1 bar (<= 21 s per step, <= 80 GiB)". Under the stricter
`601 + leg <= 1202` reading the two in-house graphs still pass at *every* rung (worst case
601 + 476 = 1077 s on Tkl94t85j4pV at 10000), while of the official graphs only 5dP7W2NMFq4T would
still pass at ma=1000 (1056 s); R7QzD6vYBLD3 would be 1321 s and DtWYToXTPh6w 1582 s, both FAIL on
time. **The in-house conclusion is stable under either reading of the bar; the official-LM
conclusion is not.**

**(ii) The `VERDICT` word is not the gate's question.** `probe_reads_and_summary` writes PASS only
when every planned cell was measured AND every rung of the ladder passes. The registered rule
(amendment 9.3) is the opposite shape: the arm takes "the smallest probe rung that PASSES the E1
bar AND E0's stability rule". Under the registered rule all five graphs have a passing rung, and
on all five it is `max_active = 1000`. The three "VERDICT: FAIL" lines are produced entirely by
the ma=10000 rung -- a rung the design would never select once 1000 passes -- exceeding 80 GiB by
3.3, 8.9 and 8.5 GiB. Anyone transcribing those three FAILs as "this graph fails E1" would be
stating something the measurements do not say. Conversely the two in-house PASSes are the
*stronger* claim (all three rungs pass) and are correct as measured.

**(iii) Constants traced.** batch 88,000 / max_seqs 128 / laplace / partition_epoch 4 / tau from
the arm's own anneal / ctrl_50 ep10 / search beam 20 / output beam 8 / min_active 30 / the ladder
1000-3000-10000 / d_min 2 via `min_frames = 1` at stride 3 / escape graph, undeterminized: all
either asserted in the job against the live config or recorded in `build.json`, and all trace to
Design 7, amendment 8 or amendment 9.3. The two constants that do *not* trace to the gate text are
`max_timed_batches = 9` (section 3) and `CHUNK_SEQS = 16`. The latter is a launch-granularity knob
introduced by commit 3454206 to work around a k2 int32 overflow; it changes no per-sequence score
(the chunk guard logged a worst-case window sum 89x-894x over 2^31 on every rung and the run did
not fault, so the guard is a bound, not a prediction), but it does change the cost profile, and
it is the reason the longest-T batch is the cheapest rather than the most expensive one.

## What I would need to close the remaining gaps

1. A timed read of `LexlatK2Runtime.step` at the same shape -- it is the quantity the arm will
   actually pay, and it includes `log_z_h` and the `get_arc_post` monitors that the probe omits.
2. A peak-memory read with `backward=False` (the arm's path), which is what bounds the card.
3. If the memory clause is meant to bound the arm rather than the leg, the bed's own peak reserved
   at this shape, to add to the leg's 43.28 GiB (in-house, ma=1000).

## Files

- `.../lexlat_k2_jobs/LexlatK2ProbeJob.{Tkl94t85j4pV,by7UYEjtYdel,R7QzD6vYBLD3,5dP7W2NMFq4T,DtWYToXTPh6w}/output/{summary.txt,probe.json,per_batch.json,gpu_check.txt}`
- `log.run.1` in the first and last job dirs; inside `finished.tar.gz` for the other three.
- `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/lexlat_k2.py:976-1270`,
  `lexlat_k2_jobs.py:517-1089`, `lexlat_k2_train.py:333-425`.
