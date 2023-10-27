* Import works, recog works, produces mostly same WERs
* Training speedup (https://github.com/rwth-i6/returnn/issues/1402).
  Mostly done now, we get almost same speed as TF?
  (CTC still missing)
* [Check older experiments on Conformer](../exp2022_07_21_transducer/exp_fs_base/README.md),
  example: conformer_ln_pre10_d384_h6_blstmf2_fix245_wdro_specaugweia_attdrop01_posdrop01_aux48ff_mhsapinit05_lsxx01

TODO:

- aux48ff, or aux4812ff: aux CTC
- attdrop01
- posdrop01
- wdf?
- wdro
- lsxx01, or other lsxx
- specaugweia? yes but generalize... also could be optimized
- twarp?
- rndresize09_12, or other
- mhsapinit05
- chunk, some variant, unclear which is best...

- data loading, shuffling, batching, more like ESPnet?
- optimizer, weight decay
- LR scheduling
- loss normalization, mean batch, mean all, or sum? also div by grad accum?
- feature normalization?
- param init. Linear is different
- higher batch size / more grad accum, or schedule
- less regularization, augmentation in beginning, schedule (seems very important for convergence)
- schedule downsampling factor, high in beginning

- dropout mask like TF, broadcast over time?
- mixup (port over TF code)

- CTC/AED loss scales?
- gradient clipping? global norm variant. 5.0. seems to solve hiccups. 
- bfloat16 vs float16 vs float32: float16 gives nan, bfloat16 seems to work just like float32 
- grad_scaler: not needed for bfloat16
- adamw vs adam?
- weight decay? (also dependent on whether adam or adamw)
- weight decay only on selected layers/modules, like in TF, e.g. not so much on decoder
- adam eps? 1e-16 is what we had in TF, maybe better?
- try CTC only
- try no specaugment (should overfit then, i.e. reach train loss 0)

TODO model changes:

- Second decoder LSTM
- ZoneoutLSTM use_zoneout_output=True
- (cnnblstmf2)
