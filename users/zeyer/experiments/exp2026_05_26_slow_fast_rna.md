# Frame-synchronous streaming ASR: three parallel implementations, code differences

Three separate implementations of the same idea,
with separate experiments and separate results.
This file lists the **code differences**,
so numbers are not compared naively.
All facts were read from the code on `origin/main`,
which since 2026-09-05 also carries V's work (merge of `ba-thesis-volodymyr`).

| | who | code |
| --- | --- | --- |
| **A** | Albert Zeyer | `i6_experiments/users/zeyer/nn_rf/decoder/streaming/*`, recipe `exp2026_05_26_slow_fast_rna_rz.py` |
| **R** | Robin Schmitt | `2025-10-speech-llm`, branch `main`, `src/speech_llm/prefix_lm/` |
| **V** | Volodymyr Sheremeta | same repo, branch `ba-thesis-volodymyr`, `src/speech_llm/delayed_stream/` |

R has two model generations,
`DelayedStreamsModel` (V1) and `DelayedStreamsModelV2` (separate fast/slow predictors),
each with its own train step and search.
V has DSM, AED and MonoRNN-T,
and his MonoRNN-T has both a full-sum and a forced-alignment path.

## The DSM / RNA decoder is not the same in the three

This is the biggest difference, and it is easy to miss:
all three are "one label-or-blank per encoder frame,
audio frame added to the previous-emission embedding".
That additive fusion is the same.
The decoder consuming it is not.

| | decoder | size |
| --- | --- | --- |
| A | own RF Transformer++ (RMSNorm, RoPE, gated FF), `FramewiseDecoder` | 6 layers x 1024, ~100M |
| R | **HF `Qwen/Qwen2-0.5B` causal LM** (`Qwen2DecoderV2`) | **~0.5B**, randomly initialized |
| V | own `CausalTransformerDecoderV1` | 6 layers x 512 |

R's decoder dominates his model and is ~5x ours;
V's is half our width.
R does not load pretrained Qwen weights:
`load_checkpoint_on_init` defaults to False,
and is never set in his configs.
What he preloads is his own earlier ASR baseline
(`config_loquacious_25k_albert_qwen_lowercase_v1/baseline_bs-50000_downsample-1`, epoch 25,
`init_for_train=True`, `ignore_missing=True` for the new blank head),
so his DSM runs start from a model that already saw 25 epochs.
No LoRA and no freezing in his DSM path;
the `decoder_lora_opts` / `freeze_decoder_params` defaults live in the older baseline config.

## Vocabulary

- A: spm10k plus one for blank/eoc, 10241 classes.
- R, **DSM line only**: the **Qwen tokenizer, `vocab_size=151646`**, lowercased,
  with `<blank>` added as a special token.
  His output softmax is therefore ~15x wider than ours,
  so his CE and WER are not comparable cell-for-cell.
  He also has non-DSM configs on **spm10k with a 16-layer encoder**
  (`config_loquacious_25k_albert_qwen_lowercase_spm_v1.py`),
  so "Robin uses the Qwen vocab" holds for the DSM experiments,
  not for his work in general.
- V: spm, 4k and 10k arms,
  with the SPM `PAD` id doubling as the blank.

## Encoder

- A: `ChunkedConformerEncoderV2`, **16 layers**, 1024 dim, 8 heads,
  C=5 / L=80 / R=4, dynamic chunk/history/lookahead train pools,
  aux CTC on layers 4, 10, 16.
- R: `ChunkedConformerEncoder`, **18 layers**, 1024 dim, 8 heads, downsampling 6,
  chunk stride / history / input-chunk dims set explicitly,
  aux CTC on layer **18 only**.
  Same family as ours, one size up.
- V, current Loquacious runs: Conformer **16 layers, 1024 dim**, 8 heads, ff 4096,
  frame rate 16.67 Hz, downsampling 6,
  and **chunked L80 / C5 / R4**, i.e. our chunk settings,
  with a causal variant alongside; aux CTC on layers **4, 10**.
  His older LibriSpeech baselines are 12 layers, 512 dim, offline;
  do not quote those against ours.

## Corpus and schedule

All three are on **Loquacious**;
V started on LibriSpeech-960 and moved over.

- A: `partition_epoch=25`, 100 sub-epochs = **4 full epochs**.
- R: `partition_epoch=100`, 25 sub-epochs per run,
  **preloaded from a 25-epoch checkpoint**,
  so his effective training is much longer than ours.
- V, current Loquacious runs: `train_partition_epoch=25`, 100 sub-epochs,
  i.e. **4 full epochs, the same as ours**.
  His LibriSpeech runs were `train_partition_epoch=1`, 100 epochs.

## The loss is the same loss

It is cross-entropy over the same event space in all three.
Factorizing the blank is a **modelling** choice,
i.e. how the distribution is parameterized,
not a different loss.
The real differences are these:

| | blank parameterization | label-loss normalization | blank weighting |
| --- | --- | --- | --- |
| A | extra class in one softmax | per **frame** (/T) | none, blank is just a class |
| R | separate sigmoid head, `P(y)=P(not blank)*P_label(y)` | per **label** (/L) | separate `blank_ce_scale`, **0.5** in the DSM configs |
| V | SPM `PAD` in the softmax, optional factorized branch | per **frame** | `pad_loss_scale` **0.5**, i.e. blank down-weighted |

R's blank BCE is summed over T frames but divided by L,
so relative to a per-frame normalization
the blank term carries an extra factor T/L (~2-3x here),
before `blank_ce_scale` is applied.
Both R and A use the numerically stable `log_sigmoid(+-x)` form,
never `log(1 - sigmoid(x))`.

## Alignment

- A: **fixed** RNA alignment from a separate offline base model (`ChunkAlignDataset`);
  a self-aligned variant (`rf.ctc_best_path`, `label_loop=False`) is now training.
- R: **CTC Viterbi recomputed every step from the model's own aux CTC head**, under `no_grad`,
  torchaudio or RETURNN-native,
  `label_loop` True (CTC plus centre-frame reduction) or False (RNA).
- V, current Loquacious runs: `alignment_source="ctc_offline"`,
  a fixed alignment from his own frozen CTC aligner, stored once and reused;
  same idea as ours, different aligner.
  MFA is the older LibriSpeech path; on-the-fly re-alignment exists but is off by default.

## Delay

- A: decoder-side `delay_frames` (42 = 2.5 s),
  encoder padded and target shifted inside the model.
- R: `num_delay_frames`, **0 in the DSM configs read**;
  alignment left-padded with blank, audio right-padded,
  the first frames sliced off the loss,
  forced-blank prefix at search.
- V: **data-side**, the dataset shifts onsets and pads audio,
  **2.5 s**, the same value as our best model.

## Search

- A: beam plus max recombination (frame-sync),
  length normalization for the label-sync AED,
  decoder-only.
- R: beam only, **no length normalization**,
  recombination V1-only and off by default,
  decoder-only,
  except a slow-predictor variant where CTC is the acoustic model by construction.
- V: greedy by default, beam sweeps 4/8/16/32;
  recombination default `none` for DSM but **`sum` for MonoRNN-T**;
  **CTC fusion at search** (`ctc_joint_scale`, `ctc_prefix_scale`) and 4-gram LM rescoring,
  with the reported number the **min WER over checkpoints x beams x scales on dev**.
  LM-rescored numbers are not comparable to ours and should be ignored.

## What still differs from us in his current Loquacious runs

These are close enough to compare, so the remaining differences are the ones that matter:

- Aux CTC on layers **4, 10**, ours on **4, 10, 16**.
- `pad_loss_scale` **0.5**, so blank frames count half; ours weights blank like any label.
- `label_smoothing_mode="gradient"` at 0.1, ours applies smoothing on the value.
- Target stream is word-packed (`CTC_WORD_PACKED`), carrying word structure our RNA target lacks.
- **No speed perturbation**, ours follows the base regime.
- `behavior_version` 30, ours 24.
- adamw with epsilon 1e-16 and gradient clip 5.0, peak learning rate 5e-4,
  against our OCLR base 0.5 regime.
- His full-sum transducer path silently drops sequences with more labels than frames.

## Still unverified

- Which R generation (V1 vs V2) and which slow-predictor variant his reported numbers come from.
- Whether V's current Loquacious runs keep the LibriSpeech-era settings listed above.
