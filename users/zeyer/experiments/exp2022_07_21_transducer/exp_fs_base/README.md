The starting point (`start`) of the reproduction of the transducer pipeline
(still only the first step, the full sum training)
is far off the original config.

Here we will do experiments,
both from this start point and also from the original (`orig`) config,
towards each other, to compare and find out the relevant differences.

See `sis_config_main` for the main Sis entry point.
It will automatically collect all configs in this directory and run them.
Every config is just a separate Python file where `sis_run_with_prefix` is defined.

---

Tracks:

- Reproduce original RNA full-sum results:
  [Paper](https://arxiv.org/abs/2005.09319),
  [config](https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config),
  [train scores](https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/scores/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.train.info.txt),
  17.5% WER Hub500, 16.5% WER Hub501, after 25 full epochs (150 subepochs).
  - orig_pre20bs5k_native_fixsepblanklast: "hub5e_00": 18.4, "hub5e_01": 17.2
  - start_slowpretrain: "hub5e_00": 18.0, "hub5e_01": 17.1
  Not reached orig, why? Noise? Hardware? RETURNN version? TF version?

- Conformer, get good results, run existing old RETURNN configs, new RETURNN-common Conformer, maybe import old to new:
  - old_moh_hybrid_conformer: "hub5e_00": 27.7, "hub5e_01": 23.0
  - old_nick_att_conformer_lrs2: "hub5e_00": 23.3, "hub5e_01": 19.0
  - conformer_d384_h6_wd0_nopre_blstmf_specaug: "hub5e_00": 23.6, "hub5e_01": 20.8
  Far worse than BLSTM. What's the problem?

  Check: `/u/jxu/setups/switchboard/2022-05-24-speaker-adaptation-for-neural-acoustic-modeling/work/jxu/crnn/sprint_training/do_not_delete/oclr_baseline_10_4/CRNNSprintTrainingJob.IoaZg3yRZi9L/output/crnn.config` (Via Tina)
  - CE with focal_loss_factor 2
  - 12 layers
  - frontend: conv-based, downsampling 3 via strides
  - 384 / 1536 dim
  - 6 heads
  - depthwise_conv filter size 8
  - layer norm instead of BN
  - dropout 0.1 / attention dropout 0.1
  - "one-cycle" learning rate, 0.002 to 0.02 to 0.002 to 1e-07
  - num sub-epochs 260, partition epoch 6 (swb)
  - aux loss layer 4, layer 8
  - no L2
  - no pretrain
  - batch size 14k, no grad accum
  - chunking 500

  Check: `/work/asr4/zhou/asr-exps/swb1/2021-12-09_phoneme-transducer/work/crnn/custom_sprint_training/CustomCRNNSprintTrainingJob.C7DaO3jCzO1K/output/crnn.config` (via Wei)
  - full-sum training
  - 12 layers
  - frontend: 3 layer 2d conv, downsampling 4 via 2*2 strides
  - 512 / 2048 dim
  - 8 heads
  - depthwise_conv filter size 32
  - Batch norm: 'delay_sample_update': True, 'epsilon': 1e-05, 'momentum': 0.0, 'update_sample_only_in_training': True, 'use_sample': 1.0
  - L2: 5e-06 (but not everywhere)
  - Dropout 0.2 / attention dropout 0.2
  - batch_size 3000, accum_grad_multiple_step 3
  - no aux loss
  - custom one cycle LR
  - custom spec augment
  - no pretrain
  - num sub-epochs 240, partition epoch 6

  Conformer now seems to perform well on Hub500 Swb and Hub501 and RT03s
  but bad on Hub500 CH. Why?

- Understand cause(s) of non-determinism.
  - https://github.com/rwth-i6/returnn/issues/1210
  - Maybe grad accum in behavior version 15 helps? https://github.com/rwth-i6/returnn/pull/1206
  - Native-CTC is known to be non-deterministic. But how much?
  - Other aspects?

---

RNA differences/Observations:

- pretrain: growing, repeat, etc.

- adam vs nadam
- accum grad vs larger batch size
- batching "random"?
- model really the same?
- L2, dropout, etc?
- batch_size was 5000 (and grad accum 3) in the first 20 subepochs in the original config before it was edited...

- joint CTC? -> both no.
- correct dataset? normalized? kazuki version?
  -> both use `/work/asr3/irie/data/switchboard/corpora`, same BPE
- native vs pure TF -> no diff
- lr schedule? no
- lr schedule error key? error but no actual diff

...

Conformer:

- model:
  - frontend: conv (and its variants) or BLSTM
  - num layers
  - dim, ff dim
  - num heads
  - depthwise_conv filter size
  - layer norm vs batch norm
    - layer norm allows for much higher LR?
  - pos encoding type: old, new, other...
- LR schedule
- pretrain
- aux loss (CTC or CE or whatever)
- batch_size / grad accum
- param init
- weight decay (L2)
- chunking
  - can even do that on whole seq, inside the network (only for training)
  - alternatively, similar effect: restrict self-attention to local context

I think a big problem is overfitting, esp on Switchboard.
Should use some better regularization:
- time warping (e.g. see also ESPnet, or our old twarp variants, and other variants)
- weight dropout
- variational weight noise

Other aspects:

- Use full train corpora, including current CV set, and use hub500 as CV set. (Via Tina, matters for full-sum training.)
  - Maybe only the non-OOV segments from Hub500?
  `/u/raissi/experiments/lm-sa-swb/dependencies/zhou-dev-segments-hub`
  - Maybe cleanup Hub500. Might have different text processing...
