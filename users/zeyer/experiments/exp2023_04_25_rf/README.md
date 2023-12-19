* Import works, recog works, produces mostly same WERs
* Training speedup (https://github.com/rwth-i6/returnn/issues/1402).
  Mostly done now, we get almost same speed as TF?
* [Check older experiments on Conformer](../exp2022_07_21_transducer/exp_fs_base/README.md),
  example: conformer_ln_pre10_d384_h6_blstmf2_fix245_wdro_specaugweia_attdrop01_posdrop01_aux48ff_mhsapinit05_lsxx01
* ESPnet example: https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/conf/tuning/train_asr_conformer10_hop_length160.yaml

Reference:

From Mohammad, 2023-06-29
  dev-clean  2.27
  dev-other  5.39
  test-clean  2.41
  test-other  5.51
_returnn_tf_config_filename = (
   "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.1oORPHJTAcW0/output/returnn.config")
E.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
LR file (train/dev/devtrain): /work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.SAh74CLCNJQi/work/learning_rates

Experiments:

- CTC/AED loss scales? unclear results, all worse than 1.0/1.0. also tried with loss normalization.
- aux48ff, or aux4812ff: aux CTC. variant just aux12 (on top) did not work
- gradient clipping? global norm variant. 5.0. seems to solve hiccups. 
- bfloat16 vs float16 vs float32: float16 gives nan, bfloat16 seems to work just like float32 
- grad_scaler: not needed for bfloat16, even better without (nogradscaler)

TODO:

- posdrop01: base-24gb-v4-posdrop01
- wdf?
- wdro
- wdblacklist: BatchNorm, LayerNorm etc not part of weight decay. unclear, worse?
- lsxx01, or other lsxx
- specaugweia? yes but generalize... also could be optimized
- twarp?
- rndresize09_12, or other
- mhsapinit05
- chunk, some variant, unclear which is best...

- data loading, shuffling, batching, more like ESPnet?
- optimizer, weight decay
- LR scheduling:
  - std: Noam (like ESPnet): lin warmup, inv sqrt decay 
  - lrlin: lin warmup, lin decay, lin decay finetune
  - lrcos: multiply with cos
- loss normalization, mean batch, mean all, or sum? also div by grad accum?
- feature normalization?
- param init. Linear is different
- higher batch size / more grad accum, or schedule.
  in the very beginning (warmup phase), we might need high batch size (grad accum) for convergence.
  but then, smaller batch size (bs30k) actually seem to converge faster initially?
  for finetuning at the end, we might want to have higher batch size again.
- less regularization, augmentation in beginning, schedule (seems very important for convergence)
- schedule downsampling factor, high in beginning

- dropout mask like TF, broadcast over time?
- mixup (port over TF code)

- adamw vs adam? note that weight decay param needs to be retuned. adam needs another test.
- Nadam? https://github.com/rwth-i6/returnn/issues/1440
- weight decay? (also dependent on whether adam or adamw) adamw-wd1e_3 better?
- weight decay only on selected layers/modules, like in TF, e.g. not so much on decoder
- adam eps? 1e-16 is what we had in TF, maybe better? moh as 1e-8 though.
- try CTC only
- CTC is incorrectly trained with EOS label at end - fix?
- specaugment_steps still not like original (0, 1000, 2000), much later.
  but necessary for convergence without pretrain.
  also looking at overfitting-ratio, not sure if this is a big problem.
- attention dropout broadcast: check no broadcast over batch/head (base-24gb-v4-attdropfixbc, rf_att_dropout_broadcast=False)
- dropout broadcast in general: what effect? disable? running: base-24gb-v4-nodropbc. compare also base-24gb-v4-attdropfixbc
- embedding init is probably bad. check embInit1
- gradient noise
- variational noise
- compare FER to Moh, we overfit more! need more regularization
- label smoothing for CTC?

- fine tune:
  - linspace vs geomspace?
  - step-base vs epoch-base?
  - start LR = initial LR or most recent?
  - final LR?
  - how long? 50? 100? 200 subepochs?

- model average


TODO model changes:

- Second decoder LSTM
- ZoneoutLSTM use_zoneout_output=True
- (cnnblstmf2)
- QK Norm (as in QK Norm paper with L2 norm, or as in Scaling ViT paper with LayerNorm)
- Transformer decoder, probably makes training more stable, reaches better WER without ext LM/ILM
- Chunked AED
- LayerNorm no bias
- check Zipformer, ScaledAdam, etc
- check E-Branchformer
- flash attention

