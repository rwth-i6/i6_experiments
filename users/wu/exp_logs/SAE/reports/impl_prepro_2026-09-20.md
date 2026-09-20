# Implementer report — SAE 4a prepro (wav2vec-U 2.0 waveform cut), 2026-09-20

Status: **DONE_WITH_CONCERNS**. Deliverable 1 is written, tested end to end and committed.
Deliverable 2 is written and committed, but **two experimental constants are undetermined and were
NOT chosen**; the pack cannot be built until the planner sets them. The data jobs — including the
design review's pre-funding check — are complete and launchable now.

Commit `d92109b` on `recipe/2025-10-speech-llm`, branch `haotian_modality_matching_jupiter`
(explicit paths only; the other implementers' untracked files were left alone). Not pushed.

## Files

| file | what |
| --- | --- |
| `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/trimmed_audio_data.py` (new, 755 lines) | `TrimmedAudioBlankfreeDataJob` + the pure helpers `speech_segments` / `concat_segments` / `splice_offsets` / `trimmed_raw_index` / `splice_distance_frames` / `aggregate_labels_to_silence` / `or_retained_indices` |
| `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_trimmed_audio_data.py` (new, 299 lines) | the four-part test below |
| `.../librispeech/configs/config_sae_4a_prepro_pack_v1.py` (new, 515 lines) | `data()` / `py_dev_other()` / `build()` / `py()` |
| `config/sae_4a_prepro_pack.py` (setup dir, new) | three-line shim → `py` (the repo's convention for `config/`; these are plain shims, not symlinks) |
| `config/sae_4a_prepro_devother.py` (setup dir, new) | three-line shim → `py_dev_other`, the pre-funding check alone |

No existing file was edited. No blank-free module, no `av_states.py` / `build_av_states.py` /
`quantize_states.py` / `feature_dump.py`, no live config, nothing of the concurrently-written
`prior_probe` / `blankfree_sampler` / `blankfree_probe_jobs` / `neural_phone_lm_v2` / `text_filter`
set. The new job class is in a NEW module, so no neighbouring job re-hashed (confirmed by census).

## The two undetermined constants — please decide, I did not

### 1. `LR_WARM_FRAC` — the N = 20 learning-rate schedule does not exist

`budget._schedules("ctrl", 20)` **raises**. `blankfree_budget_jobs.budget_learning_rates` guards
`2 <= ceil(warm_frac*n) <= floor(hold_until*n) < n`; at N = 20 the pre-registered `warm_frac = 0.05`
gives `ceil(0.05*20) = 1` warmup entry, which its own message refuses ("the pre-registered shape
needs at least two warmup entries"). Measured directly:

```
20 FAIL ValueError warm_frac=0.05 / hold_until=0.6 put the end of the warmup at entry 1 and the end
        of the hold at entry 12 of 20; the pre-registered shape needs at least two warmup entries...
50 ok  [1e-05, 5.5e-05, 0.0001, ...]        100 ok  [1e-05, 3.25e-05, ...]
w(20)=1  h(20)=12  anneal(20)=4
```

`SAE_4A_budget.md` "Sub-epoch count for future arms" states exactly the refused shape ("anneal 4
sub-epochs 8 -> 2, **warmup 1**, hold to 12, linear decay to 20"). So the prose and the function
contradict each other **at N = 20** and the value is not derivable from either. Two resolutions:

* pre-register a `warm_frac` for N = 20 with `ceil(warm_frac*20) >= 2` (e.g. 0.10 → a two-sub-epoch
  warmup, hold to 12, decay to 20 — the documented shape with warmup 2 instead of 1); or
* amend the guard in `emc.blankfree_budget_jobs` if "warmup 1" is to be kept (a shared-module edit,
  which this round's constraints forbid me anyway).

The **temperature** schedule at N = 20 needs nothing: `budget_temperature_schedule(20)` already
gives the documented `[8.0, 5.04, 3.17, 2.0, 2.0, ...]`.

Every arm's hash moves with this value.

### 2. `CTRL_S1_SEED` — the blank-free bed has no registered seed mechanism

`ctrl_20_s1` is "ctrl_20 with a second seed" (design review amendment 1), but there are **two**
different seeds and they measure **different bands**:

* `init_jobs.FlatRecognizerInitJob(seed=…)` — the bed uses `seed=0`. Moves theta's non-logit
  initialisation. phi's init and the batch order do NOT move.
* RETURNN's `random_seed` config key — `build_blankfree_train_config` writes no such key, so every
  arm silently runs RETURNN's default 42. Moves phi's init AND the batch/shuffle order. theta's
  init does not move.

Neither `SAE_4A_prepro.md` nor the design review names one. Because the band is the gate's
reference spread, the mechanism is part of the experiment, not an implementation detail. **Both are
plumbed**; set `CTRL_S1_SEED = {"flat_seed": <int>}` and/or `{"random_seed": <int>}`.

### How the config behaves meanwhile

`data()` needs neither constant. `py()` registers the three data jobs and prints
`[sae/4a/prepro] ARMS NOT BUILT -- <the exact reason>`; `build()` raises before registering
anything. This is the correct intermediate state, not a degraded one: the design review funds the
pack only AFTER the dev-other reads.

## Deliverable 1 — what the job does

Chain 1–6 of the Design, with the design review's amendments folded in:

1. HF ogg decode at 16 kHz — same dataset object (`control.HF_DATASET`) and the same
   `cast_column("audio", Audio(sampling_rate=16000))` as `blankfree_data.py` / `build_av_states.py`.
   The **tag set, shard membership and within-shard write order are read off the BANKED L15 feature
   HDFs**, so the emitted shards carry exactly the bed's utterances in the bed's order; the audio
   rows are then fetched by an id→row map and `ds.select(...)` in that order.
2. `rVADfast(vad_threshold=0.4)` RAW 10 ms labels, no 50 Hz aggregation applied to the cut.
   On the first utterance of each shard the job asserts that its own re-derivation of the
   aggregation **is** `vad_port.rvad_silence(..., subframes=2)`.
3. `speech_segments` = `vads.py:80-92` verbatim (stride 160, 0→1 opens, 1→0 closes, open tail to
   `len(wav)`, empty list = whole file kept), concatenated in memory (`remove_silence.py:44-51`).
4. `Wav2Vec2EncoderV1(hf_model_dir=<pinned>, encoder_layer=15, trainable=False)`, `.eval()`,
   `no_grad`, batch 1. Numerically the bed's own encoder: the bed constructs it with
   `trainable=True` inside `SpeechLmV2` and then calls `model.eval()`, which routes to the same
   `self.model.eval()`; `truncate_unused_layers=True` is the default on both sides and the module
   documents `hidden_states[15]` as byte-identical under truncation. The front-end
   zero-mean/unit-variance normalisation therefore acts on the TRIMMED waveform, as the paper's
   extractor does.
5. Units: `reconstruct_quantizer` / `standardize_frames` / `quantize_utt` — `AssignUnitsJob`'s own
   call — applied to the **fp16-rounded** states.
6. `raw_index`: centre sample `320 t' + 200` (receptive field 400 — the design review's amendment,
   not the spec's +160) mapped back through the segment offsets, `// 320`, clipped to
   `orig_length - 1`; asserted int32, non-decreasing, `< orig_length`. `orig_length` is
   `Wav2Vec2EncoderV1.get_output_lengths(len(full_wav))`, cross-asserted per utterance against the
   banked feature HDF's own `seqLengths`.

Outputs: `feats/units/raw_index/orig_length.{split}.shard{k}.hdf` through `SimpleHDFWriter` with the
bed's dims (1024 / 500 / 1 / 1) and insert pattern, plus `manifest.json` and `summary.txt`. Output
attribute names are the bed's (`out_feature_hdfs` etc.), so a consumer that takes the VAD job takes
this one. Every input is a `tk.Path` with `hash_overwrite` (HF dataset, unit store, quantizer,
w2v2 repo, vad_port) or a banked job output (the L15 feature shards).

Asserts that STOP the job (per split, in `collect`): the re-derived OR mask's retained total equals
781,130 / 831,372 / 15,427,853 exactly; `sum(orig_length)` equals 919,980 / 968,057 / 18,088,388;
the utterance count equals 2864 / 2703 / 28,539; no utterance with T' < 2 (a trimmed waveform under
400 samples is recorded as an offender rather than crashing the encoder); and, per shard, that
applying this module's assignment path to one BANKED utterance's states reproduces that utterance's
banked units frame for frame with K = 500.

Reported (`summary.txt` + `manifest.json`): bed-vs-new unit agreement on `raw_index`-matched frames
overall and by distance to the nearest splice; k-means distortion (mean squared PCA distance to the
assigned centroid) new vs bed, the bed side computed on ITS OR-retained frames with ITS banked
units; unit unigram entropy and dead-unit counts new vs bed; the OR-vs-raw retained delta; T'/T;
utterances kept whole; segments per utterance; and the disclosed differences from the paper's own
scripts.

**Assumption I had to make and am naming**: the splice-distance bin edges. The design review asks
for "binned by distance to the nearest splice" without naming edges. I used dyadic bins in 50 Hz
frames — `[0,1) [1,2) [2,4) [4,8) [8,16) [16,inf)` — declared as a rendering choice in the module
(`SPLICE_DISTANCE_BINS_FRAMES`) and printed beside the edge-free overall agreement. Change the
tuple to re-bin; nothing else depends on it.

rqmt per shard task: `{"gpu": 1, "gpu_mem": 24, "cpu": 4, "mem": 64, "time": 4}`, plus a `collect`
mini task. Shards: train 4, dev-clean 1, dev-other 1 (the banked feature shard counts).

## Deliverable 2 — the config

Three data jobs, then one packed node with `ctrl_20`, `ctrl_20_s1`, `prepro_20` at N = 20, kept
checkpoints 1 / 4 / 10 / 20; per kept epoch and both dev splits the budget pack's own reads
(theta/phi extraction, posterior dump, `BlankfreeGreedyPerJob` with its decode stats, and
`BlankfreeDerangementGapJob`); and the paired contrasts `prepro_20 vs ctrl_20`,
`prepro_20 vs ctrl_20_s1` and `ctrl_20 vs ctrl_20_s1` (the band) at every kept epoch and both
splits, with `PairedPerDeltaJob`'s own `per_a` = baseline convention.

**Deviation from the brief, deliberate**: the brief asked me to reuse the budget pack config's
helpers by import. `budget_pack._arm_train_config` looks the arm up in `budget.ARMS` (no N = 20
entry) and always takes `base["vad"]`, so it cannot be called at all;
`budget_pack._register_epoch_reads` hardcodes `name=f"budget_pack/{arm}/..."`, which would LABEL
this phase's decodes as the budget round's — exactly the mis-labelling these reads must survive;
`budget.register_paired_reads` is hardwired to `budget.ARMS` / `budget.KEEP_EPOCHS` /
`budget.paired_arm_pairs`. I wrote three local copies that make the same calls with the same
arguments and read every constant from those modules. The duplication is checked by diff (below),
which is the same guard `test_blankfree_pack` applies to the budget pack's own copy.

`py_dev_other()` + `config/sae_4a_prepro_devother.py` implement amendment (d): a graph with the
dev-other data job ALONE (3 jobs total, the other two already finished upstream), so the ~2 h train
instance is not funded alongside the pre-funding check. I added this second shim because
registering all three data jobs in one graph would submit the train job too.

## Checks run and their results

* `python -m speech_llm.sae.emc.test_trimmed_audio_data --no-smoke` → cut rule ok, raw_index
  mapping ok, aggregation ok. Covers all-speech, all-silence, open tail, single-frame segments,
  leading/trailing silence, alternating labels, non-binary label refusal; the identity map, the
  constant shift, the across-splice shift with the exact `320 t' + 200` centre, the
  `orig_length - 1` clip, monotonicity, `inf` splice distance without a seam; and the aggregation
  against an independently written form of `vad_port`'s rule at n = 0, 1, 2, 3, 7, 101 plus the
  silence padding and truncation to `orig_length`.
* Full test (CPU, real encoder / real frozen quantizer / real unit store / real HF audio), three
  banked dev-other utterances `116-288045-000{0,1,2}` (532 / 431 / 481 frames):
  * **quantizer agreement on the banked utterance: 532/532 exact, K = 500** — the call into the
    frozen codebook is right;
  * 1444 raw → 1194 trimmed frames; the OR mask re-derived from the same labels would keep 1202
    (an 8-frame OR-vs-raw delta on these three);
  * bed-vs-new unit agreement 881/1194 = **0.738** on these three utterances. Anecdote, not a
    result — but it is the opposite of the "near 1.00 ⇒ empty treatment" case, which is what the
    dev-other job is being funded to measure properly;
  * HDF field parity against the BANKED `BlankfreeVadHdfJob.SAjz8y1cT06g` dev-other shard:
    identical for all four streams — `feats` `inputs float16 (F,1024)` / `numLabels 1024`,
    `units` `int32 (F,)` / 500, `raw_index` `int32 (F,)` / 1, `orig_length` `int32 (N,)` / 1 with
    `seqLengths` second column all 0; same attrs (`inputPattSize`, `numDims`, `numLabels`), same
    dataset names and dtypes, same tags.
* Config load from the setup dir, `py()` with both constants unset: loads, prints
  `ARMS NOT BUILT -- LR_WARM_FRAC is UNDETERMINED: ...`, registers the three data jobs and six
  outputs. `py_dev_other()`: 3 jobs in the graph, the dev-other data job at the same hash.
* **Hash census** (`config_sae_4a_budget_pack_v1.py()` + `config_sae_4a_budget_v1.py()` alone =
  1545 jobs; with this config loaded too): **no banked id disappeared in either state**. Data-only
  state adds exactly the 3 data jobs (1548). With provisional constants the pack state adds 125
  jobs (1670) and still moves nothing: 3 data, 1 pack, 1 `FlatRecognizerInitJob`, 24 forward /
  24 `BlankfreeGreedyPerJob` / 24 `BlankfreeDerangementGapJob` / 24 `ExtractSubmoduleCheckpointJob`
  / 24 `PairedPerDeltaJob` — the expected 3 arms x 4 epochs x 2 splits and 3 pairs x 4 x 2.
  `node_a`'s `PackedBlankfreeTrainJob.ks7CbtlvpcIL` is untouched.
* `python -m speech_llm.sae.emc.test_blankfree_budget_config` → all budget config tests still pass
  (`node_b` / `node_c` hashes unchanged).
* **Duplication guard** (probe, provisional constants): the budget pack's `ctrl_50` written RETURNN
  config vs my `ctrl_20` differ ONLY in `num_epochs`, `cleanup_old_models.keep` and the two
  schedule lists — no other key drifted, so the local builder copy is the budget pack's call.
* **Arm-to-arm diff** (probe): `ctrl_20 → ctrl_20_s1` changes exactly 3 lines (the flat checkpoint
  path and a new `random_seed = 43`); `ctrl_20 → prepro_20` changes exactly 48 lines, all of them
  HDF path swaps `BlankfreeVadHdfJob.SAjz8y1cT06g → TrimmedAudioBlankfreeDataJob.<train hash>`.
  Nothing else differs between the arms.
* **Wall clock**: `pack.rqmt = {gpu 4, cpu 64, mem 256, time 11.5, gpu_mem 96}`. The three arms run
  side by side, one per GPU, so the allocation is ONE arm's clock: 20 x 601 s = **3.34 h** inside
  11.5 h without a resume (asserted in `build()`). Even read serially, 3 x 20 x 601 s = 10.0 h
  still fits. (The brief's "2 x 20 x ~600 s" assumed two arms; with the review's third arm both
  readings hold.)

The probe constants (`LR_WARM_FRAC = 0.10`, `CTRL_S1_SEED = {"flat_seed": 1, "random_seed": 43}`)
were set **only in a throwaway script** to exercise the pack path; they are not in the committed
config, and every hash below the pack will move when the real values are set.

## Job hashes (data jobs — final, independent of both undetermined constants)

```
train      speech_llm/sae/emc/trimmed_audio_data/TrimmedAudioBlankfreeDataJob.qb4o6dlW3urA  (4 shards)
dev-clean  speech_llm/sae/emc/trimmed_audio_data/TrimmedAudioBlankfreeDataJob.hxIx0ItTvx15  (1 shard)
dev-other  speech_llm/sae/emc/trimmed_audio_data/TrimmedAudioBlankfreeDataJob.0IOLr6hZnYWj  (1 shard)
```

Pack/arm/read hashes are not final and are deliberately not quoted here: they all move with
`LR_WARM_FRAC` (and `ctrl_20_s1`'s with `CTRL_S1_SEED`).

## What a reviewer should look at first

1. The two undetermined constants above.
2. The splice-distance bin edges (a rendering choice I had to make).
3. That `raw_index`'s approximation after a splice is acceptable: `BlankfreeGreedyPerJob` and
   `BlankfreeDerangementGapJob` do not read it (design review Q1.6), but the gold-frame
   diagnostics do.
4. The second shim `config/sae_4a_prepro_devother.py` — added so amendment (d) is real rather than
   nominal. If a separate entry point is unwanted, say so and I will fold it into a flag.

## What I did NOT do

Launch anything; touch `settings.py`; push; edit any project document; edit any shared module,
including `blankfree_budget_jobs.py` whose guard is one of the two blockers.
