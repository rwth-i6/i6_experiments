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

- CTX aux loss...?
- LR schedule
- pretrain
