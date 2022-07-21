
# My BPE transducer Switchboard setups

Old location, my old multi-setup: /u/zeyer/setups/switchboard/2019-10-22--e2e-bpe1k

Paper Latex (with config filenames): /u/zeyer/Documents-archive/2020-rnnt-paper/rnnt-paper.tex

RETURNN-experiments configs and readmes: https://github.com/rwth-i6/returnn-experiments/tree/master/2020-rnn-transducer

## Pipeline to best result

Final config: rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.fixmask.rna-align-blank0-scratch-swap.encctc.devtrain.retrain1

Import last model from config: rna3c-lm4a.convtrain.switchout6.l2a_1e_4.nohdf.encbottle256.attwb5_am.dec1la-n128.decdrop03.decwdrop03.pretrain_less2_rep6.mlr50.emit2.fl2.fixmask.rna-align-blank0-scratch-swap.encctc.devtrain

Alignment: rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap

Alignment comes from RNA full-sum where blank idx was 0, thus there was a conversion step in between, `tf.where(alignment==0, 1030, alignment)`.

Alignment code: https://github.com/rwth-i6/returnn-experiments/tree/master/2020-rnn-transducer/dump-align

RNA full-sum config for alignment model: rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50

