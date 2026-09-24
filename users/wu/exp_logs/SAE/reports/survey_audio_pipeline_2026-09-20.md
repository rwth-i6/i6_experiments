# Survey: current audio/feature pipeline of the blank-free bed (for a waveform-trimmed wav2vec-U 2.0 variant)

Read-only survey (Explore agent, 2026-09-20; the agent could not write files, transcribed by the
orchestrator verbatim). Setup root `S = /e/project1/spell/wu24/2026-07-13_unsupervised`. Paths
relative to `S` unless absolute.

## 1. Current blank-free bed pipeline

### 1a. Raw layer-15 feature extraction (GPU)
- Job `AvStatesJob` — `recipe/2025-10-speech-llm/src/speech_llm/sae/av_states.py:55`; ctor `:83-131`; worker `recipe/2025-10-speech-llm/src/speech_llm/sae/build_av_states.py`.
- Config args for the bed's dumps: `tap="encoder"`, `encoder_layer=15`, `av_checkpoint=None`, `downsampling_factor=4` (unused on the encoder tap), `shards=4` — `configs/config_sae_3a_enc50_units_v1.py:86` (`_states`), `:103` (`seed_codebook`), `:118` (`tc100_units`).
- `av_checkpoint=None` => no AV SFT; the encoder is the pretrained **facebook/wav2vec2-large-lv60** built by `Wav2Vec2EncoderV1` (`av_states.py:57-62`; log line "NO AV CHECKPOINT: dumping the raw pretrained wav2vec2-large-lv60 encoder"). HF weights dir: `work/i6_experiments/users/schmitt/external_models/huggingface/DownloadHuggingFaceRepoJobV2.hUcBAmkrRK83/output/content`.
- Tap = `hidden_states[encoder_layer]`, **1024-d @ 50 Hz** (`av_states.py:74-77`, `build_av_states.py:27,92`). Layer convention: `hidden_states[15]` = output of block 14 = fairseq `layer=14` = paper's "15th layer" (`w2vu2/dump_w2vu2_data.py:50-54`).
- Inference is **batch size 1**, one utterance at a time, `datasets` audio cast to 16 kHz (`build_av_states.py:107-131`).
- Output format: pickled `{seq_tag: float16 [T,1024]}` per shard + `av_state_stats_global.npy` / `av_state_stats_perutt.pkl` (raw, unstandardized).
- Banked outputs (consumed as frozen `tk.Path` with `hash_overwrite`, `configs/config_sae_4a_s0b_inits_v1.py:48-61`):
  - `work/speech_llm/sae/av_states/AvStatesJob.Dsynh5MqmgjY/output/av_states.shard{0..3}.pkl` — train-clean-100, 28,539 utts, 18,088,388 frames, 4 x ~9.3 GB (alias `sae/3a/enc50/states_tc100full_sh4`).
  - `work/speech_llm/sae/av_states/AvStatesJob.c4Ak1rACchRC/output/av_states.pkl` — 2,849 seed + 5,567 dev, 3,685,941 frames (alias `sae/3a/substrate/raw/seed_states50`).
- Repack to RETURNN HDF: `emc.feature_dump.L15FeatureHdfJob` — `sae/emc/feature_dump.py:125`, fp16 byte-for-byte, `dim=1024, ndim=2` (`:192-199`); constants `L15_DIM=1024`, `L15_FRAME_HZ=50.0` (`:77-78`); `EXPECTED_UTTS = {dev-clean:2703, dev-other:2864, test-clean:2620, test-other:2939}` (`:75`). Registered `configs/config_sae_4a_s0b_inits_v1.py:119` (train) and `:128` (dev).

### 1b. rVAD mask job — `BlankfreeVadHdfJob.SAjz8y1cT06g`
- Class `sae/emc/blankfree_data_jobs.py:10`; implementation `prepare_blankfree_data` in `sae/emc/blankfree_data.py:44-166`.
- Registered in `configs/config_sae_4a_blankfree_v1.py:38-43` (`prepare()`), alias `sae/4a/blankfree/vad`.
- Inputs (`config_sae_4a_blankfree_v1.py:14-21, 36-40`):
  - `hf_dataset` = `work/i6_core/datasets/huggingface/TransformAndMapHuggingFaceDatasetJob.OYvh9012Pgkb/output/dataset` (splits `train`, `dev`).
  - `feature_hdfs` = `L15FeatureHdfJob` outputs (train 4 shards, dev-clean/dev-other 1 shard each).
  - `units_store` = `work/speech_llm/sae/quantize_states/PackUnitsJob.I0uzRMfUrKWC/output/units_store`.
  - `reproduction_dir` = `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/features/MergeW2vu2DataJob.wxBxCIaJpqS2/output/data` (`train.ids/.lengths`, `valid.ids/.lengths`) — the §1c reference length check.
  - `vad_port_path` = `recipe/i6_experiments/users/wu/experiments/unsupervised_asr/vad_port.py`.
- rVAD: `rVADfast(vad_threshold=0.4)` (`blankfree_data.py:82`). Installed package `env/conda/envs/speech_llm/lib/python3.11/site-packages/rVADfast/rVADfast.py:17-18` defaults: window 0.025 s, shift 0.01 s, n_fft 512, sft_threshold 0.5, vad_threshold 0.4. Manifest string: `"rVADfast 0.0.3 via vad_port.rvad_silence, threshold=0.4, 25ms window/10ms shift, subframes=2, tail=silence"` (`blankfree_data.py:111-112`).
- 10 ms -> 50 Hz aggregation, `vad_port.py:30-43`: `labels[: n*2].reshape(n,2).mean(1) < 0.5` with `subframes=2`. With 2 subframes a 50 Hz frame is SILENCE only if **both** 10 ms frames are non-speech (speech = logical OR). The trailing remainder frame is truncated.
- Reconciliation (`blankfree_data.py:95-98`): pad tail with silence up to `original`; `indices = np.flatnonzero(~silence[:original]).astype(np.int32)`.
- Outputs per split/shard (`blankfree_data_jobs.py:26-43`): `feats.{split}.shard{k}.hdf` (fp16, dim 1024), `units.{split}.shard{k}.hdf` (sparse, dim 500), `raw_index.{split}.shard{k}.hdf` (int32, retained -> ORIGINAL frame index), `orig_length.{split}.shard{k}.hdf` (1 int per utt), `manifest.json`, `summary.txt`. Written with `SimpleHDFWriter` (`blankfree_data.py:139-161`).
- Output dir: `work/speech_llm/sae/emc/blankfree_data_jobs/BlankfreeVadHdfJob.SAjz8y1cT06g/output`. `summary.txt`: train 28,539 IDs, 18,088,388 raw -> 15,427,853 kept; dev-clean 2,703 IDs, 968,057 -> 831,372; dev-other 2,864 IDs, 919,980 -> 781,130; 0 reference mismatches, 0 utterances with <2 retained frames (hard release gate, `blankfree_data.py:104-126`).
- Cost: CPU-only task, `{"cpu":4,"mem":96,"time":8}` (`blankfree_data_jobs.py:46`); measured `used_time` 1.75 h.

### 1c. enc50 K=500 reverse units
- Quantizer `QuantizeStatesJob.FWpGhC941JMi` — class `sae/quantize_states.py:360`; **PCA(96) + MiniBatchKMeans(K=500, random_state=42, batch_size=4096, n_init=3, max_iter=100)** (`fit_pca_kmeans`, `:60-70`), fit on <= 500,000 frames of the 10 h seed encoder-tap dump, `dedup=False`, per-frame assignment, no pooling, `per_utt_standardize=False`. Constants `NUM_UNITS=500, PCA_DIM=96, KM_SEED=42` (`configs/config_sae_3a_enc50_units_v1.py:49-51`).
- Assignment over tc100: `AssignUnitsJob.ARG5PMIPji84` (`quantize_states.py:524`); merge `MergeUnitsPklJob.iryDwagsbJkH` (`:617`, frame agreement 1.000000); pack `PackUnitsJob.I0uzRMfUrKWC` (`:694`) -> `units_store/{data.npy,uids.npy,offsets.npy,lengths.npy}`, read mmap in `blankfree_data.py:128-131`. Per-utterance `T == S` guarantee documented `feature_dump.py:16-27`.

### 1d. Training HDF assembly / consumers
- `blankfree_train_jobs._dataset` (`sae/emc/blankfree_train_jobs.py:68-88`): `MetaDataset` with `data_map = {"features": ("feats","data"), "units": ("units","data"), "original_length": ("original","data")}`, `seq_order_control_dataset="feats"`, optional `MultiProcDataset`.
- `extern_data` (`blankfree_train_jobs.py:156-160`): `features {dim 1024, float16}`, `units {sparse, dim 500, int32}`, `original_length {sparse, dim 1, int32}`.
- Segments: `init_jobs.CvHoldoutSplitJob` seed-0 1 % holdout of the 28,539 GAN-pseudo-label ids (`config_sae_4a_s0b_inits_v1.py:105-106,143`), used as `train_segments` / `dev_segments` (`config_sae_4a_blankfree_v1.py:61-62`).
- Train step `prefix_lm/model/train_steps/sae_blankfree.py:36-44`: `original = extern_data["original_length"]`, asserts `lengths == unit_lens` and `original >= lengths`; recognizer output length `(T_retained + 2) // 3` (`:59`).
- Packed variant: `PackedBlankfreeTrainJob` — `sae/emc/blankfree_pack_jobs.py:132` (subclass of `pack_jobs.PackedEmcTrainJob`), per-arm `mem 64`, `gpu_mem 96` (`:97-98`), `packed_blankfree_training` `:171`.
- Registering configs (all under `prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/`): `config_sae_4a_blankfree_v1.py` (bed), `config_sae_4a_s0b_inits_v1.py` (features/inits), `config_sae_4a_attrib_v1.py` (`:201,:230` raw_index), `config_sae_4a_private_code_v1.py` (`:72,:116-125` pinned VAD streams), `config_sae_4a_budget*/infomax/pack/seed` variants.

## 2. Where the original audio comes from
- No bliss corpus and no flac/wav tree. Source = HF `openslr/librispeech_asr` parquet snapshot under `HF_HOME=/e/project1/spell/common_hf_home` (`data/huggingface.py:60-81`).
- `TransformAndMapHuggingFaceDatasetJob` + `_map_func_audio_to_ogg` (`prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/data/huggingface.py:209-275`): ffmpeg `-ar 16000 -ac 1 -c:a libvorbis -q 3 -f ogg` => stored audio is **16 kHz mono Ogg Vorbis q3** inside HF arrow shards. A resampling step exists (to 16 kHz); no amplitude normalisation at this stage.
- Datasets in use: `TransformAndMapHuggingFaceDatasetJob.OYvh9012Pgkb` (train-clean-100 -> train/dev/test; builder `get_librispeech_train_clean_100_hf_ogg`, `data/huggingface.py:125-153`), `...DlfDYnmTEkGZ` (seed-10h + dev), `...mQmb6aW1IDH5` (10 h seed dir).
- Decoding always `cast_column("audio", Audio(sampling_rate=16000))` -> float array (`blankfree_data.py:84`, `build_av_states.py:119`).
- Per-utterance **waveform** standardisation (fairseq layer-norm style, `do_normalize=True`) is applied by the encoder front-end, not by the dataset — documented `sae/emc/feature_dump.py:36-46`; explicit in the §1c worker `w2vu2/dump_w2vu2_data.py:73` (`Wav2Vec2FeatureExtractor`). The dumped 1024-d states themselves are raw/unstandardized.

## 3. Existing waveform-trimmed audio or fairseq `.vads` on disk — NONE
- `find S -maxdepth 8 -name "*.vads"` -> no hits. `work/` search for `vads|remove_silence|without_silence` -> only an unrelated job-input symlink name.
- `work/i6_experiments/users/` contains only `schmitt`, `wu`, `zeyer` — Enrique's jobs have never run in this setup.
- The §1c wav2vec-U 2.0 reproduction never cut waveforms: `W2vu2FeatureDumpJob` (`recipe/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/features.py:141`) + worker `w2vu2/dump_w2vu2_data.py`. It encodes the **full** waveform (`dump_w2vu2_data.py:186`), computes Kaldi MFCC 39-d on the continuous waveform (`:93-105`), then drops silent frames from the feature matrix (`:195-199`, `_sil_mask` `:107-112`). The module docstring (`:18-27`) states the deliberate choice to subsample+trim in feature space rather than fairseq-style waveform trimming.
- Banked §1c outputs (train `W2vu2FeatureDumpJob.HyHAk3OCbruI`, valid `...WbaqNnxXpbRK`, merged `MergeW2vu2DataJob.wxBxCIaJpqS2`): `train.stats.txt` = 28,539 utts, frames_raw 18,088,388 -> frames_kept 15,427,853, `vad_dropped_frac=0.1471`, dim 1024, fp16, `encoder_layer=15`, `trim_silence=True`, `mfcc_downsample=2`, `vad_subframes=2`; `valid.stats.txt` = 5,567 utts, 1,888,037 -> 1,612,502, `vad_dropped_frac=0.1459`.
- Fairseq-literal path exists **as code only**: `recipe/i6_experiments/users/enrique/jobs/fairseq/wav2vec/audio_preprocessing.py` — `Wav2VecUDeleteSilencesInAudioJob` (`:53`) runs `vads.py` (`:210-228`) then `remove_silence.py` (`:231-252`); `Wav2VecUFeaturizeAudioJob` (`:286`) runs fairseq `wav2vec_extract_features.py` with `--layer 14` (0-based = paper's 15) (`:304`, `:447-483`). Twin at `recipe/i6_experiments/users/enrique/jobs/gan/process_audio.py`. Drivers: `experiments/wav2vec_u/{full_pipeline,rasr_decoding_pipeline,bpe_pipeline}.py` via `process_audio(...)`.

## 4. Exact fairseq cut rule
`env/conda/envs/w2vu/lib/python3.9/site-packages/fairseq/examples/wav2vec/unsupervised/scripts/`
- `vads.py:31-64` (`rvad`): `soundfile.read(path)`; asserts `fs == 16000`; `winlen=0.025, ovrlen=0.01, pre_coef=0.97, nfilter=20, nftt=512, ftThres=0.5, vadThres=0.4, opts=1`; pre-emphasis filter `b=[0.9770,-0.9770], a=[1.0,-0.9540]`; returns one 0/1 label per **10 ms** frame.
- `vads.py:75, 80-92`: `stride = 160` samples. Loop over labels: on `0->1` set `start = i*stride`; on `1->0` append `(start, i*stride)`; if still open after the loop append `(start, len(wav))`. Printed as `"s0:e0 s1:e1 ..."`, one line per utterance in tsv order.
- **No padding, no margin, no min-segment length, no merging of neighbours.** Boundaries are exact multiples of 160 samples except the final end, which is the sample count.
- `remove_silence.py:34-39` parses `s:e` pairs; `:44-51`: `torchaudio.load`, `torch.cat([data[0][it[0]:it[1]] for it in intervals]).unsqueeze(0)`; if the interval list is empty the file is kept whole; `torchaudio.save(outpath, data_filtered, sample_rate=16000)`.
- Reproducing from our side: use the **raw 10 ms rVAD labels** with stride 160. Our banked 50 Hz masks are lossy w.r.t. this rule — `subframes=2` ORs pairs of 10 ms frames, truncates the odd trailing frame, and pads the tail as silence — so the naive map frame `t` -> samples `[320t, 320t+320)` is an approximation, and `rVADfast` (pip refactor) is not bit-identical code to fairseq's `RVAD_ROOT/speechproc`.

## 5. MFA gold loader and index streams needed by evaluation
- `private_code.load_gold_frames` — `sae/emc/private_code.py:332-377`. Reads gilkeyio parquet (`id`, `phonemes`) from `$HF_HOME/hub/datasets--gilkeyio--librispeech-alignments/snapshots/*/data/{dev_clean,dev_other}-*.parquet` (`:354-356`, `MFA_SPLIT` `:64`). Rasterised by `i6_experiments/users/wu/experiments/ssl/analysis/repr_audit.frame_phone_labels` (`repr_audit.py:53-70`) on the **original** 50 Hz grid, frame `t` takes centre `(t+0.5)/50` (`FRAME_RATE_HZ = 50.0`, `private_code.py:60`). An utterance whose MFA coverage exceeds the feature length by more than **one** frame is dropped, not truncated (`:370-374`).
- Mapping to retained frames (`private_code.py:812-826`): `gret = g[raw_index[tag]]`, then per output frame a majority vote over blocks of `UNITS_PER_OUTPUT_FRAME = 3` (`:62`, ties -> lowest id).
- Consistency asserts: `len(raw_index[tag]) == T_retained` (`:718`); posterior length `== ceil(T_retained/3)` (`:717`); collapsed argmax must equal the banked greedy string (`:712-715`).
- PER job `BlankfreeGreedyPerJob` (`sae/emc/blankfree_eval_jobs.py:46-119`) consumes `features` (lengths), `originals` (`orig_length`) and reports `output_frames`, `retained_frames`, `original_frames`, `phone_rate_original_hz`, `phone_rate_retained_hz` (`:92-107`); asserts `len(q) == (retained + 2)//3` (`:96`).
- Pinned streams for analysis: `config_sae_4a_private_code_v1.py:116-125` (`units`, `raw_index`, `orig_length` per dev split, shard0).
- => A trimmed-waveform variant must still emit `raw_index` (retained-frame -> ORIGINAL 50 Hz frame index) and `orig_length` (untrimmed T) per utterance, otherwise PER/gold-frame diagnostics break. Caveat: after waveform concatenation the encoder's new frames no longer correspond 1:1 to original frames (conv receptive field crosses each splice), so `raw_index` becomes an approximate map derived from the segment sample offsets, and `T_trimmed` will generally differ from the 15,427,853 / 831,372 / 781,130 kept-frame counts.

## 6. GPU cost precedent
- `AvStatesJob.Dsynh5MqmgjY` (tc100 L15, 4 shards, batch 1): each `run` task **3:55** wall (~3:05 pure encode for 7,135 utts / 4,537,368 frames); `rqmt {gpu:1, gpu_mem:24, cpu:4, mem:64, time:8}`, exclusive booster node; 4 tasks in parallel + a trivial `collect`. Source: `work/.../AvStatesJob.Dsynh5MqmgjY/log.run.1`, `usage.run.*`.
- `AvStatesJob.c4Ak1rACchRC` (2,849 seed + 5,567 dev, 1 shard): ~**5:50**; `usage.run.1 used_time 0.0882 h`; `gpu:1, mem:96`.
- §1c `W2vu2FeatureDumpJob` (HF `Wav2Vec2Model` + Kaldi MFCC + rVAD in one per-utterance loop): train `HyHAk3OCbruI` **1:51:26** (`used_time 1.857 h`), valid `WbaqNnxXpbRK` **0:14:43** (`0.245 h`); `rqmt {gpu:1, mem:32, cpu:8, time:8/4}`. The realistic precedent for a fresh per-utterance dump over trimmed audio.
- `BlankfreeVadHdfJob.SAjz8y1cT06g`: CPU-only, `used_time 1.75 h` (rVAD over 34,106 utterances + HDF rewrite of ~35 GB).
