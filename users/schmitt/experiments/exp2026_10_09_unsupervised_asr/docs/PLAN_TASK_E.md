# Task E (re-scoped, 2026-09-17): the **emission model**

Supersedes the original Task E ("run-length / duration model, unsegmented setting"). The reason for
the re-scope is II.C.2: on the real clus128 stream the **supervised** ceiling of the model class we
have been optimizing is **71.1% PER**. A duration model sits inside that cap, so the original task
could not have produced a usable result no matter how well it worked.

    criterion  : essentially solved on cheat-seg  (label-free 26.4% vs a 26.0% ceiling, II.D.3)
    emission   : the binding constraint on real audio (71.1% supervised cap, II.C.2)

Task E's job is therefore **not** a better objective. It is: *find an audio-side representation whose
supervised ceiling is good, then hand it to the criterion stack that already works.*

---

## E.0 What is actually on disk (verified 2026-09-17)

The featurize job keeps **every** intermediate stage of fairseq's `prepare_audio.sh`, for the full
train-other-960 (278,400 utts — the same utterance set as zyang's cheat-seg clusters and
`GMM_SEGMENT_PHONEME_HDFS`). Under

    work/.../audio_preprocessing/Wav2VecUFeaturizeAudioJob.mkGrrp0YWy8y/output/audio_features/

| stage | content | size | tokens/utt | `R` vs 119.55 phonemes |
|---|---|---|---|---|
| `precompute_pca512/` | one 512-d vector per wav2vec frame (20 ms) | 277 GB | **514.4** | 4.30 |
| `precompute_pca512_cls128_mean/` | pooled within cls128 runs | 171 GB | **317.5** | 2.66 |
| `precompute_pca512_cls128_mean_pooled/` | + pairwise (2x) subsample | 86 GB | **159.0** | 1.33 |
| `CLUS128/{train,valid}.src` + `centroids.npy` | the 128 cluster ids + centroids | 222 MB | 317.5 | 2.65 |
| `pca/512_pca_A.npy`, `_b.npy` | the PCA projection | 2.1 MB | | |
| *(zyang)* cheat-seg k512 segment clusters | one token per oracle phoneme segment | | ~96–123 | ~0.98 |

Each stage is `train.npy` (flat `[sum_T, 512]` float32) + `train.lengths` + `train.tsv`, i.e.
`np.load(..., mmap_mode="r")` with offsets from `cumsum(lengths)`; `valid.*` is the fairseq 1%
split and must be concatenated (see the CLAUDE.md gotcha).

**Two facts this makes concrete, both verified numerically:**
1. `clus_len == 2*feat_len` or `2*feat_len - 1` per utterance (checked on 2794 paired seqs of shard
   0: exact for 49.9%, off by one for the rest — the odd-length case). So the wav2vec-U feature
   stream everyone calls "the features" is a **2x pooled** version of the collapsed cluster stream,
   not a frame-synchronous partner of it. The unsegmented work in II.C.2 used the 317-token discrete
   stream; the 159-token continuous one was never measured.
2. At 159 tokens/utt the feature stream is at **`R = 1.33`**, i.e. much closer to one-token-per-
   phoneme than clus128's 2.65 — without any oracle segmentation. It is the nearest available
   analogue of the cheat-seg granularity.

---

## E.1 The supervised-ceiling ladder (the gate — do this first, do not skip to E.2)

**Measure the supervised ceiling of each rung before running any unsupervised search on it.** This is
the direct lesson of II.C.2: four `lam_z` arms x 4 seeds were spent on a setting that could not have
exceeded 71% even with a perfect search.

**Fixed harness for every rung** (so rungs are comparable, and comparable to II.C.2):
- Same splits as II.C.2: 20k audio utts / 60k disjoint text utts / **1000 paired eval utts** /
  3000 paired utts for the supervised fits, all disjoint. Phoneme references from the existing
  `PHONEME_HDF`; pairing by seq tag as `calc_unsegmented_map.py` already does.
- Decode = map each audio token to its argmax phoneme, collapse adjacent repeats, edit distance
  against the reference -> PER. Same decoder for every rung, so only the emission changes.
- **Supervised fit protocol.** For a *discrete* rung: count map from position-proportional
  alignment, + 1 Viterbi realignment, then `per_coordinate_descent` (the direct PER hill-climb) to
  convergence. Report all three. For a *continuous* rung: multinomial logistic regression on frames
  with targets from proportional alignment + 1 Viterbi realignment. **Do not use the maximum-
  likelihood fit under the collapse model as the reference** — II.C.2 showed it decodes to 7 tokens
  per utterance; it is the wrong supervised comparator here.
- Report per rung: supervised PER, decoded length vs reference length, the random-map floor, and the
  information diagnostic **NLL/token against a tied (audio-ignoring) map** — the number that gave
  "a clus128 id is worth 0.26 nats", which is model-class-free and comparable across rungs.

**Rungs, cheapest first.** Every one is offline CPU work on data that already exists; none needs a
sisyphus job, a GPU, or a training run.

| rung | what it isolates | construction |
|---|---|---|
| **E.1a** *(control)* | — | clus128, 317 tokens/utt. Reproduce **71.14%**; validates the harness. |
| **E.1b** granularity | 317 -> 159 tokens | k-means (k=128) on `_mean_pooled`; same memoryless table. |
| **E.1c** resolution | alphabet size | k-means on `_mean_pooled` at k ∈ {128, 256, 512, 1024, 2048}. **k=512 is the direct analogue of cheat-seg's k512**, differing only in where the segment boundaries come from. |
| **E.1d** discretization | symbol vs vector | linear + 1-hidden-layer probe on the raw 512-d `_mean_pooled` vectors. Upper bound for "no quantization at this granularity". |
| **E.1e** context | memorylessness | log-linear emission over `[onehot(c_{t-1}); onehot(c_t); onehot(c_{t+1})]` (3·k·40 params) and ±2. Cheap, and it keeps a finite symbol alphabet -> the II.B/II.D machinery still applies (see E.2). |
| **E.1f** segmentation | boundary quality | agglomerative merge of the `_mean_pooled` sequence down to `T/R` segments (`R` label-free from A2), pool, re-cluster at k=512. This is the cheat-seg *construction* with unsupervised boundaries. |
| **E.1g** *(upper bound)* | boundary quality, oracle end | same as E.1f but boundaries from a supervised Viterbi alignment of the fitted frame model. Labels used — an upper bound, not a result. |

*Why E.1g rather than importing zyang's boundaries:* our stream is rVAD-silence-removed and twice
pooled, zyang's GMM alignment is on the original audio. Aligning the two frame grids is a separate
(and error-prone) piece of work; a self-contained pseudo-oracle segmentation on our own stream
answers the same question — "how much does boundary quality cost?" — without it.

### E.1 OUTCOME (2026-09-17) — G1 cleared; full tables in `RESULTS.md` II.E.1

`calc_emission_ladder.py`, three SLURM jobs (~1 h each, `plan_runs/taskE/`). Control validated: the
clus128 rung reproduces II.C.2 (73.57 vs 73.55 for +1 Viterbi, 72.09 vs 71.14 for the hill-climb).

| | best ceiling on this stage |
|---|---|
| `pca512` (514 tok/utt) | 47.92 (segmented probe) — **75.62 unsegmented** |
| `cls_mean` (317) | 42.74 (segmented probe) — 53.56 unsegmented |
| `pooled` (159) | **34.74** (linear probe), best discrete **41.89** (k=2048) |
| clus128 control (317) | 72.09 |

Granularity dominates; resolution pays only at the right granularity (k 128→2048 is −14.7 points on
`pooled` but **+10.8** on `pca512`); segmentation rescues a fine-grained stream and does nothing for
an already-pooled one; quantization costs ~7 points, not 40. The rungs that clear G1 and the
E.2 candidate list are in `RESULTS.md` II.E.1.

### Decision gate G1

Rank the rungs by supervised PER. A rung is eligible for E.2 if its supervised PER is **≤ 45%**.
Rationale: the criterion lands within 0.4 points of its ceiling on cheat-seg (26.4 vs 26.0), so a
45% ceiling plausibly yields a ~45–50% label-free result — clearly above chance and in the range of
the supervised frozen-encoder probe (33.6–41%). A rung at 71% cannot.

- **If some rung clears 45%:** proceed to E.2 with the best one, and with any rung within 5 points of
  it (cheaper rungs preferred at equal ceiling).  **← this is what happened; see E.1 OUTCOME above.**
- **If the best rung is 45–55%:** proceed, but state in the report that the result is bounded and
  that the emission model, not the criterion, is still the limiting factor.
- **If nothing clears 55%:** **stop and report.** The lookup-table family is then exhausted and the
  next step is a learned generator (E.3), which is a training task, not a script task.

### Honesty note on E.1 (must appear in the report)

E.1 uses labels by design — it selects a *representation*, not a model. That is legitimate as a
diagnostic and is what fairseq's own w2vu pipeline did, but it means **any headline label-free number
from E.2 inherits a label-dependent choice of representation**. State it whenever that number is
quoted. It does not license using labels anywhere inside E.2 (see the guardrails).

---

## E.2 Carry the validated criterion to the winning rung

No new objective. Take the winner of E.1 and run the II.D.3 two-stage procedure unchanged:

1. **Stage 1** — `calc_hmm_map_search.py --mode em --init uniform`, 4 seeds, ≥400 iterations, select
   by hardened objective.
2. **Stage 2** — `calc_soft_map_search.py --init assign --init-assign <stage-1 npz> --w 5,5,10,20
   --four-backoff 0.2 --tau-start 1.0 --keep-best`.
3. Plus the Task C machinery, which is what the unsegmented setting needs: `--collapse` for the
   repeat pushforward and `--lam-z` / `--z-target` for the length term, with `z*` from the **unpaired**
   corpus length ratio. II.C.2 established that the pushforward alone is nearly a no-op and the length
   term is what makes it work, so `lam_z` must be swept, not fixed.

**If the winner is E.1c or E.1f** (a finite symbol alphabet), the machinery applies verbatim — only
the count tables `C_k` are rebuilt over the new symbol set. Note the cost: the map is `k x 40`, so
k=2048 is 16x the parameters of the clus128 table against the same amount of text, and the
higher-order tables get sparser; report whether the criterion's own basin behaviour survives the
larger alphabet (4/4 seeds at stage 1 is the benchmark).

**If the winner is E.1e** (context), the audio symbol becomes a tuple and the same applies with
`x = (c_{t-1}, c_t)`; 128² = 16,384 symbols, most of them observed.

**Reporting (unchanged from the plan):** all seeds with hardened loss and hardened accuracy, plus the
loss-selected one; never only the best seed. Separate **identifiability** (oracle-start distance to
the rung's own supervised ceiling) from **basin** (cold-start result and seed success rate). And run
`calc_shuffle_control.py`'s test in PER form on the loss-selected map — a result is not a result
until the matched/shuffled gap is positive.

---

## E.3 Conditional: a learned generator with the n-gram objective instead of a GAN

Only if G1 sends us here, or if E.1d shows a large gap between the continuous probe and the best
quantized rung (i.e. quantization is what costs).

The project already has a wav2vec-U implementation (`models/definitions/wav2vec_u.py`,
`train_steps/wav2vec_u.py`) whose generator is exactly the right emission class — a Conv1d over the
continuous features — and which trains end to end but was never shown to beat chance. The experiment
is to **keep that generator and replace the GAN discriminator with the objective we have proved
identifies the map**: forward `KL(p_n || q_n)` at orders up to 4 with back-off, the collapse
pushforward, and the length term.

Mechanically this is `train_steps/output_stats.py` (which already accumulates a soft frame-level
bigram in the right KL direction) extended to orders 3–4 and given the Task C pushforward. The one
real design change is that `q_n` can no longer be a contraction of a fixed count table — it has to be
accumulated from the generator's per-frame posteriors over minibatches, which makes the loss
stochastic. The HMM route (II.D) does not port as directly, since its audio side is a frozen finite
Markov chain; the n-gram route does.

This is a RETURNN training task (GPU, sisyphus), not a script task, and should not be started before
E.1 and E.2 are reported.

---

### E.3 OUTCOME (2026-09-24) — identifiability strong, basin 0/12; `RESULTS.md` II.E.3

Run as a GPU script (`calc_e3_generator.py`), not yet ported to RETURNN: the ceilings, shuffle control
and eval split live in the E.1/E.2 harness, and a port is only worth it once a cold start works.
Linear generator (kernel 1; wav2vec-U's kernel 4 has a worse supervised ceiling) + the unchanged
order-4 criterion via an EMA surrogate; `--no-repeat` (repeat cells dropped + length term) for the
collapse decode. From the supervised generator the objective moves 32.10 → **23.53% PER** (x1.25),
i.e. its optimum is better than the supervised reference; all 12 cold seeds are at chance.
Next: basin width (corrupt-the-supervised sweep), then a label-free initializer.

---

## Dropped from the original Task E

The standalone **duration / run-length model** on the unsegmented clus128 stream. It lives entirely
under the 71.1% cap, and under E.1f duration becomes the segmenter's job rather than a separate model
term. It can come back as a refinement *after* a rung clears G1, where it would be measured against
that rung's own ceiling.

---

## Guardrails (carried over verbatim; still in force)

- Do not change the direction of any KL to `KL(q || p)`. The reverse direction frees `H(q)` and
  rewards collapse — structural, not a tuning failure.
- Do not select seeds, tune hyperparameters, or early-stop using accuracy or PER. Hardened loss only.
  (E.1 is the stated exception and is *supervised by design*; nothing inside E.2 may use labels.)
- Do not refactor the existing count-table or criterion code beyond what a task needs; the II.B/II.C/
  II.D results must stay reproducible for comparison.
- Do not add a blank symbol.
- Fix `OMP_NUM_THREADS` across arms — the anneal is chaotic and the same seed lands elsewhere under a
  different BLAS thread count (II.C.2).
- If a decision gate says stop, stop and report rather than proceeding.
- Keep the running results table in `RESULTS.md` (`## II.E`).

## Cost

E.1 is CPU-only and reads existing `.npy` via `mmap`; the k-means fits are `MiniBatchKMeans` on a
frame subsample, the probes are a few minutes each. Estimate ~1 day of `cpu_modern` wall clock for the
whole ladder including E.1f. E.2 is ~1.5 h per stage-1 seed + ~7 h for stage 2, as in Task D. E.3 is
GPU training and is not costed here.
