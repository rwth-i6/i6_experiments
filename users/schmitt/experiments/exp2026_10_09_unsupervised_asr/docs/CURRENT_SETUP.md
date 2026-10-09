# Current setup: Task E.3 — a continuous generator trained with an n-gram matching objective

Status as of 2026-10-09. Self-contained description of the most recent experiment line: what it
does, why, how to run it, where everything lives, and what it showed. Full tables also in
`RESULTS.md` (sections II.E.3 and II.E.3b); the plan it belongs to is `PLAN_TASK_E.md` (E.3).

---

## 1. Goal and background

**Goal:** unsupervised phoneme recognition on LibriSpeech. Learn a map from audio to phonemes using
audio and phoneme text that are **never paired** (they come from disjoint utterances).

**Where this sits in the project** (one line each; details in `RESULTS.md` Part II):

- On **oracle (cheat) segmentation**, one audio token per phoneme with 512 clusters, a soft
  cluster→phoneme map trained with a **forward-KL n-gram matching** objective works without labels:
  - with the order-4 criterion and weights `5,5,10,20` (II.B) it reaches ~28% PER;
  - chained with HMM-EM it reaches 26.4% (II.D), against a 26.0% supervised ceiling.
- On **real audio** the obstacle is the emission model:
  - The discrete 128-cluster stream caps at 71% PER even with labels (II.C.2).
  - E.1 showed that the continuous wav2vec features, segmented label-free to roughly phoneme rate,
    give a supervised linear-probe ceiling of 32–42% PER.
  - E.2 ran the validated criterion on a discrete k=512 stream of those segments. It still
    identified the map, but cold starts stayed at chance and the optimum was displaced by 11–16 PER
    points. The tokens are not informative enough: 0.73 nats/token of mutual information with the
    phoneme, against ~2.3 on cheat-seg.
- **E.3 (this setup)** keeps the criterion unchanged. It replaces the discrete token→phoneme table
  with a **learned generator over the continuous 512-d segment features**. This is the
  wav2vec-U architecture with its GAN replaced by the n-gram objective.

Everything is computed by one script, **`scripts/calc_e3_generator.py`** (paths in this document are
relative to this package, `recipe/i6_experiments/users/schmitt/experiments/exp2026_10_09_unsupervised_asr/`). It is plain
PyTorch on GPU, with no Sisyphus and no RETURNN.

---

## 2. Data

### Audio features
- **Source:** the intermediate stages of fairseq's wav2vec-U `prepare_audio.sh`, as kept by the
  featurize job:
  `work/i6_experiments/users/schmitt/experiments/exp2025_10_02_shared_enc/librispeech/data/audio_preprocessing/Wav2VecUFeaturizeAudioJob.mkGrrp0YWy8y/output/audio_features/`
- **Pipeline:** wav2vec2 (vox 60kh, layer 14) → PCA 512 → mean-pooled within runs of identical
  cls128 k-means ids → pairwise 2× pooling.
- **Stage used:** `pooled` (`precompute_pca512_cls128_mean_pooled/`), i.e. 512-d vectors at about
  **158 tokens/utterance**. The `train.npy`/`valid.npy` + `.lengths` + `.tsv` files are read through
  `calc_emission_ladder.FeatureStore`, which includes fairseq's 1% valid split.
- **Corpus:** train-other-960 utterances. The phoneme references are
  `DumpPhonemeIndicesToHdfJob.V3rHWoSUS9Hf` (g2p lexicon, no silence, 40 phonemes actually occurring).

### Label-free segmentation
- **Ratio R:** mean feature tokens per utterance (from the audio split) divided by mean phonemes per
  utterance (from the *disjoint* text split): 158.18 / 118.65 = **R = 1.333**. Both are unpaired
  corpus averages.
- **Segments:** each utterance is cut into `round(x · T / R)` contiguous segments by
  `calc_emission_ladder.agglomerate`. This repeatedly merges the adjacent pair with the smallest
  squared centroid distance (lazy heap, O(T log T)). Each segment's frames are then averaged into one
  512-d vector.
- **Two arms:**

| arm | segments/utt (audio / eval / ceil) | meaning |
|---|---|---|
| **x1.00** | 119.4 / 120.0 / 118.4 | about one segment per phoneme |
| **x1.25** | 149.1 / 149.9 / 147.9 | 25% over-segmented: a phoneme may span 2 segments, merged at decode time by collapsing repeats |

### Splits
The same seed and harness as E.1 and E.2 (`--split-seed 1`): the tags common to features and
phonemes are shuffled and cut into the following sets.

| set | size | use |
|---|---|---|
| audio | 20,000 (19,998 kept) | unpaired training audio |
| text | 60,000 pool, **first 30,000 used** | unpaired training text (different utterances from the audio) |
| eval | 1,000 | paired, scoring only |
| ceil | 3,000 | paired, supervised reference only |

- Two audio utterances have NaN or |x| > 1e5 features. They are dropped, mirroring the wav2vec-U
  dump's `max_abs_value=1e5` filter. The paired sets are asserted clean.
- Standardization statistics (`mu`, `sd`) come from the unpaired audio set only.

### Cache
Everything above is stored **float32** in:

```
/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskE/e3/pooled_seg1.00/features.npz   (5.9 GB)
/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskE/e3/pooled_seg1.25/features.npz   (7.3 GB)
```

- float16 does not work: some features exceed the float16 range of 65504.
- The features stay in host RAM; each batch is padded and moved to the GPU on demand.
- All checkpoints (`*.pt`) are written next to the respective `features.npz`.

---

## 3. Model: the generator

```
x [B, T, 512] → (x − mu) / sd → dropout (0) → Conv1d(512 → 40, kernel k, bias) → logits [B, T, 40]
posteriors P = softmax(logits / τ)
```

- **Kernel 1 is used everywhere.** It is exactly E.1's per-segment linear probe, with 20,520
  parameters.
  - wav2vec-U's kernel 4 does worse under the same supervised protocol: 52.30 vs 42.21% PER at x1.00,
    and 40.78 vs 32.10% at x1.25.
  - With kernel > 1 the padding is `(k//2, k−1−k//2)`, so the output stays one per segment.
- **Decoding:** per-segment argmax, then collapse adjacent repeats.

---

## 4. Objective: forward-KL n-gram matching

### Text targets
From the 30k unpaired text utterances, compute the joint n-gram distributions `t1..t4`:
- dense, of size 40, 40², 40³, 40⁴;
- windows never cross utterance boundaries.

### Induced statistics
`q_n` is the **soft n-gram** of the generator posteriors, averaged over all valid windows:
`q_n = mean_t p_t ⊗ p_{t+1} ⊗ … ⊗ p_{t+n−1}`. The 4-gram is computed as `(P0⊗P1)ᵀ(P2⊗P3)`, a
1600×1600 matmul.

### Criterion
This is the order-4 conditional criterion of II.B, unchanged:

```
L = Σ_k w_k · C_k,   w = (5, 5, 10, 20)
C_1 = KL(t1 ‖ q1)
C_k = KL(t_k ‖ q_k) − KL(marg(t_k) ‖ marg(q_k))    (k ≥ 2; marg = sum over the last symbol)
```

- Each `C_k` is the exact conditional KL of order k.
- The KL direction is **forward**, text ‖ model. It is mode-covering: every phoneme and transition
  must be produced. The reverse direction (an LM score) provably prefers degenerate maps. **Never
  flip it.**
- **Back-offs:**
  - trigram: `q3 ← 0.95·q3 + 0.05·q2(a,b)·q1(c)`;
  - 4-gram: `q4 ← 0.8·q4 + 0.2·q3(a,b,c)·q3(b,c,d)/q2(b,c)`.
- Computed in float64 with `EPS = 1e-9`. Class: `NgramCriterion`.

### `--no-repeat` arm
This is the right form for a decoder that collapses repeats.
- Every n-gram cell with an **adjacent repeat** is removed on both sides and the rest renormalized.
  In the text this drops 0.57% / 1.15% / 1.73% of the mass at n = 2/3/4.
- The unigram uses Task C's collapse pushforward, `q1 ← (q1 − diag q2)/·`.
- A **length term** `lam_z · (Z − z*)²` is added, with `lam_z = 100`:
  - `Z = 1 − trace q2` is the predicted rate of label changes;
  - `z* = min(1, mean text length / mean segment count)`, again from unpaired averages. It is
    **0.9939** at x1.00 and **0.7958** at x1.25.
- This is exact for n ≤ 2 and an approximation for n = 3, 4: only windows whose interior tokens have
  run length 1 are counted.

### Minibatch estimator (EMA surrogate)
A single 256-utterance batch gives a very noisy 4-gram estimate, and the log of a noisy estimate is
biased. The loss is therefore evaluated at a running corpus estimate, with the gradient taken
through the current batch:

```
Q_n = EMA_n.detach() + (q_n^batch − q_n^batch.detach())      # value = loss at EMA, grad via batch
EMA_n ← 0.95·EMA_n + 0.05·q_n^batch                           # after each step
```

### Hardened loss (the only model-selection signal)
1. Decode all 19,998 training audio utterances by argmax.
2. Count exact n-grams of the resulting integer streams (`hard_ngrams`).
3. Evaluate the same criterion on those counts.

This is used for keep-best within a run and for choosing among seeds. Accuracy and PER are
**never** used for selection, stopping or tuning.

---

## 5. Training

| setting | value |
|---|---|
| optimizer | Adam, lr 1e-3 |
| steps | 6000 |
| batch | 256 audio utterances per step (random, without replacement within a step) |
| temperature | τ annealed geometrically 1.0 → 0.02 |
| eval interval | hardened loss every 250 steps (+ PER for logging with `--score-every-eval`) |
| keep-best | the step with the lowest hardened loss |
| runtime | about 1.3–4 h per run on one GPU (RTX 2080 Ti / GTX 1080, SLURM `gpu_11gb`) |

**Initializations (`--init`):**

| init | labels? | purpose |
|---|---|---|
| `uniform` | no | **the real unsupervised run**: weights N(0, 1e-3), bias 0, so every output starts near uniform and the seed only breaks the symmetry |
| `ceiling` | **yes** | identifiability probe: start from the supervised generator and ask whether the objective's optimum stays near it |
| `corrupt` | **yes** | basin probe: start from the supervised generator damaged by `--corrupt-kind`/`--corrupt-frac` |

**Corruption modes:**
- **`perm f`:** a derangement of `round(f·40)` output rows (weight and bias). The emission stays
  sharp, but the labels of that subset are wrong. Uses `rng = default_rng(1000 + seed)`.
- **`mix a`:** `W ← (1−a)·W + a·N`, where N is Gaussian rescaled to W's Frobenius norm; the bias is
  treated the same way. Labels stay in place, and the signal is diluted by noise.

---

## 6. Evaluation (on the 1000 paired eval utterances; after training only)

- **PER:** argmax → collapse repeats → Levenshtein distance against the reference
  (`calc_unsegmented_map.per`).
- **Shuffle control** (`per_shuffled`): each hypothesis is scored against a *different*,
  length-matched reference.
  - **gap = shuffled PER − PER.** This is the evidence that the output depends on the audio at all.
  - PER alone is misleading, because two unrelated phoneme strings already align by chance.
  - A gap near 0 means chance level.
- **Supervised reference** (`ceiling` mode; labels; reference only, never a result). It is E.1's
  protocol:
  1. proportional alignment of segments to the collapsed reference;
  2. frame-level cross-entropy (15 epochs, Adam 3e-3, 32 utterances/batch);
  3. one Viterbi realignment and a retrain;
  4. the better of the two stages on eval is kept.

  Utterances with fewer segments than phonemes are skipped. Kernel 1 gives **42.21%** (x1.00) and
  **32.10%** (x1.25), reproducing E.1's 41.67 / 32.27.
- **Reporting rules:** report **all seeds** with their hardened loss and PER, plus the loss-selected
  one. Keep **identifiability** (how far the optimum sits from a supervised start) separate from
  the **basin** (whether cold starts reach it).

---

## 7. How to run

**Environment:**
- GPU venv: `/work/asr4/schmitt/venvs/torch-2.11/bin/python3` (torch 2.5.1+cu121). The project's
  `returnn_torch` venv is CPU-only.
- Submit with `sbatch`; `srun` does not work from the desktop nodes.
- In zsh, a command stored in a variable such as `$S` is not word-split, so wrap such submissions in
  `bash -c`.
- Scripts and logs that SLURM jobs use must live on `/work` or `/u`, never in node-local `/var/tmp`.
- Large data goes under `/work/asr4/schmitt/...`; the home directory has only a few GB.

**Commands** (run from `scripts/`, since the scripts import each other as siblings; the launcher is
`launchers/taskE/launch_e3.sh`, and logs and checkpoints go to
`/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskE/`):

```bash
# 1. features (CPU, ~4 min per arm with 16 workers)
python3 calc_e3_generator.py features --overseg 1.00
python3 calc_e3_generator.py features --overseg 1.25

# 2. mandatory correctness gate (CPU)
python3 calc_e3_generator.py selftest

# 3. supervised reference (LABELS) -> ceiling_k1.pt
python3 calc_e3_generator.py ceiling --overseg 1.25 --kernel 1

# 4. label-free run (one seed); the x1.00 plain arm omits --no-repeat
python3 calc_e3_generator.py train --overseg 1.25 --kernel 1 --no-repeat --init uniform --seed 1 --score-every-eval

# identifiability probe (LABELS)
python3 calc_e3_generator.py train --overseg 1.25 --kernel 1 --no-repeat --init ceiling --seed 1 --score-every-eval

# basin probe (LABELS)
python3 calc_e3_generator.py train --overseg 1.25 --kernel 1 --no-repeat --init corrupt --corrupt-kind mix --corrupt-frac 0.8 --seed 1 --score-every-eval

# 5. list all finished runs of an arm sorted by hardened loss
python3 calc_e3_generator.py summary --overseg 1.25
```

**Launcher cases:**
- `bash launchers/taskE/launch_e3.sh ceiling|smoke|train|basin`
- `train` runs 4 uniform seeds plus the ceiling probe per arm. The x1.00 `--no-repeat` arm was
  submitted by hand with the same command; its logs are `e3_train_1.00nr_*`.
- Jobs use `sbatch -p gpu_11gb --gres=gpu:1 -c 4 --mem 24G` with `OMP_NUM_THREADS=4`.

**Files:**
- Checkpoints: `plan_runs/taskE/e3/pooled_seg{1.00,1.25}/{norep|plain}_k1_{init}_seed{N}.pt`, holding
  the state dict, hardened loss, PER, gap and hyp length.
- Logs: `plan_runs/taskE/e3_*.log`.

**Other CLI defaults:**
- The script defaults to `--kernel 4` and `--overseg 1.25`; always pass `--kernel 1` explicitly.
- Remaining defaults: `--w 5,5,10,20 --tri-backoff 0.05 --four-backoff 0.2 --lam-z 100 --steps 6000
  --batch-utts 256 --lr 1e-3 --ema 0.95 --tau-start 1.0 --tau-end 0.02 --eval-every 250
  --init-noise 1e-3`.

---

## 8. Results

### Gates (all passed)
- **`selftest`:** on hard maps of the E.2 discrete stream, both `hard_ngrams` and `soft_ngrams`
  reproduce `calc_soft_map_search.SoftMapLoss.hard` to 1e-6. The EMA surrogate's gradient equals the
  full gradient when EMA = batch (relative error 0).
- **Supervised generator:** kernel 1 reproduces E.1 (42.21 / 32.10% vs 41.67 / 32.27%).

### E.3: cold starts vs supervised start (kernel 1; hardened loss is the kept value)

| arm | init | hardened loss | PER % | gap |
|---|---|---|---|---|
| x1.00 plain | uniform s1–s4 | 18.43 / **17.87**← / 18.18 / 18.02 | 88.2 / 87.6 / 86.8 / 87.8 | −0.55 … +0.76 |
| x1.00 plain | supervised (labels) | **12.99** | 37.52 | +48.5 |
| x1.00 no-repeat | uniform s1–s4 | 16.67 / **16.27**← / 16.60 / 16.61 | 87.1 / 86.0 / 85.7 / 86.6 | −0.50 … +0.76 |
| x1.00 no-repeat | supervised (labels) | **10.26** | 32.85 | +52.0 |
| x1.25 no-repeat | uniform s1–s4 | **14.94**← / 15.53 / 15.48 / 14.97 | 90.8 / 89.9 / 89.9 / 90.3 | −0.21 … +0.58 |
| x1.25 no-repeat | supervised (labels) | **6.59** | **23.53** | **+63.5** |

← marks the loss-selected seed. The supervised starts were at 42.21% (x1.00) and 32.10% (x1.25).

**What this shows:**
- **Identifiability is strong.**
  - In every arm, training from the supervised generator ends far below every cold optimum in
    hardened loss.
  - The objective *improves* on the supervised reference: 32.10 → **23.53% PER** at x1.25. The x1.25
    trajectory passes 20.8% at step 250, then drifts to ~24.5 while the loss keeps falling: a small
    residual displacement of ~3 points.
- **The basin fails completely.** 0 of 12 cold seeds leave chance, and all of them converge to
  nearly the same wrong optimum.
- **`no-repeat` is the correct form for a collapse decode.** It lowers every loss and gives the
  better supervised-start optimum.
- **Over-segmentation (x1.25) now helps.** It does so once repeats are removed from the statistics.

### E.3b: basin sweep (x1.25 no-repeat; start = damaged supervised generator; labels used)

| start | start PER / gap | kept hardened loss | final PER / gap |
|---|---|---|---|
| perm 0.2 | 46.75 / +38.6 | 6.74 | **23.58** / +63.7 |
| perm 0.4 | 61.85 / +24.8 | 6.90 | **23.93** / +63.4 |
| perm 0.6 s1 | 76.01 / +13.2 | 7.19 | **24.52** / +62.9 |
| perm 0.6 s2 | 70.95 / +14.7 | 14.74 | 68.35 / +20.9 (stuck partway) |
| perm 0.8 | 84.33 / +3.7 | 16.30 | 89.14 / +0.6 (chance) |
| perm 1.0 s1 / s2 | 88.2 / 89.3, gap ≈ 0 | 16.00 / 17.18 | 89.4 / 89.8 (chance) |
| mix 0.5 | 59.15 / +26.7 | 6.71 | **23.66** / +63.6 |
| **mix 0.8** | **89.58 / +2.95** | 6.85 | **23.61** / +63.6 |
| mix 0.95 | 93.19 / +0.45 | 16.77 | 90.23 / +0.7 (chance) |

**What this shows:**
- **Recovery is all-or-nothing.** Every success lands at 23.5–24.5% PER, with hardened loss
  6.6–7.2.
- **The hardened loss ranks all outcomes in PER order**, so choosing among restarts by loss remains
  valid. The partial basin (14.74) is only barely below the best cold optimum (14.94), however.
- **Wrong labels:** the basin tolerates about 60% of phonemes relabeled (1 of 2 seeds). 80% and 100%
  fail: the criterion cannot undo a sharp but permuted emission.
- **Weak signal:** much more tolerant. A start at 89.6% PER with only a **+2.95** shuffle gap
  recovers fully; +0.45 does not.

---

## 9. Conclusion and open next step

- The objective is right for this generator: it identifies the map, and its optimum beats the
  supervised reference.
- **The only remaining problem is the basin.** A label-free initializer must deliver a **weak but
  consistent signal on every phoneme, about a +3 shuffle gap**; a good PER is not needed.
  - The uniform start gives 0 by construction.
  - The best label-free maps from E.2 (HMM-EM on the discrete stream) reached +1.13.
- **Candidate next experiments** (not started; need approval):
  - distil E.2's HMM-EM maps into the generator as its initialization;
  - a longer or slower high-temperature phase from uniform;
  - a lower-order curriculum (orders 1–2 first, then 3–4);
  - later, a port of the working recipe to RETURNN.

**Guardrails for continuing this line:**
- never use `KL(q‖p)`;
- never select, tune or stop by accuracy — hardened loss only;
- do not refactor the existing criterion or count-table code (`calc_soft_map_search.py`,
  `calc_e2_unsup_map.py`), so earlier results stay reproducible;
- stop and report when a decision gate says stop.
