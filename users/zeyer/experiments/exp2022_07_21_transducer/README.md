

# My BPE transducer Switchboard setups from 2020

This is the baseline.

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

## TODO missing

- [x] L2 in pred net
- [x] L2 in enc ctx
- [x] att weighs energy_factor
- [x] att weights dropout bc/noise
- [x] batch size
- [x] learning rate scheduling


# Design of setup

Still experimenting with how to best structure it...

Current thoughts:

Model def / training (recog?), one file, or one class/object?
Or maybe even better, each pipeline, one file? Each experiment is one file.
All common building blocks would be elsewhere, so such file could be quite short.
These building blocks will probably be changed over time.
We can add options to them, or just make ..._v2, ..._v3 versions.
The building blocks should always stay compatible (both hash and behavior).

Following this thought, one file for the whole thing, some open questions:

- The model def, also in there? Or is this supposed to be a building block?
Actually, to decide what should be directly in the file, and what should be a building block:
Depends on the type of experiments, what needs to be changed.
In this case, it would be the multi-step pipeline itself, and the model defs (per step). 
- What is the model def? Some func which returns an instance of `nn.Module`? Or better our custom class derived from `nn.Module`.
It gets `train: bool` as an argument. Or maybe not, and we rely on the PyTorch-like train flag of `nn.Module`?
At this point, no tensors are really created except of the parameters.
Although, the lazy param init is important to be considered, so once we did not feed any input, it might not have created all params!
- We need two additional separate functions, defining the training and the recog, based on the model def.
- How to handle pretraining?
