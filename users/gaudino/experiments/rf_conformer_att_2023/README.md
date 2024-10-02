# Recipes for joint search with attention and CTC using Pytorch

Uses RETURNN frontend with Pytorch backend

Most results of my master thesis are obtained with this setup.

I imported tensorflow models with this script: 
https://github.com/rwth-i6/i6_experiments/blob/main/users/gaudino/experiments/rf_conformer_att_2023/tedlium2/_import_model.py
and store the checkpoints in this folder: 
`/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/`

A very similar version of this setup is used for the experiments at AppTek.

Some of the recipes include experiments, model definition, training logic and recognition logic in one file.
This is what Albert also does. I find the file get too long, so I started to separate them for the recent experiments.
Just backtrack from the experiments in the entry point file what is actually used.

## Recognition logic

One-pass label-synchronous search: 
https://github.com/rwth-i6/i6_experiments/blob/main/users/gaudino/experiments/rf_conformer_att_2023/librispeech_960/model_recogs/model_recog.py

One-pass time-synchronous search: 
https://github.com/rwth-i6/i6_experiments/blob/main/users/gaudino/experiments/rf_conformer_att_2023/librispeech_960/model_recogs/model_recog_time_sync.py

## Librispeech 960h

### Sisyphus entry points

Train new attention models on Librispeech 960h and do recognition with the models:
https://github.com/rwth-i6/i6_experiments/blob/main/users/gaudino/experiments/rf_conformer_att_2023/librispeech_960/conformer_import_moh_att_train.py

Recognition with the baseline model (5.6 test-other) from tensorflow:
https://github.com/rwth-i6/i6_experiments/blob/main/users/gaudino/experiments/rf_conformer_att_2023/librispeech_960/conformer_import_moh_att_2023_06_30.pyhttps://github.com/rwth-i6/i6_experiments/blob/main/users/gaudino/experiments/rf_conformer_att_2023/librispeech_960/conformer_import_moh_att_2023_06_30.py

Combining two models (system combination) in recognition, either new or imported:
https://github.com/rwth-i6/i6_experiments/blob/main/users/gaudino/experiments/rf_conformer_att_2023/librispeech_960/conformer_import_sep_enc.py

## Tedlium 2

All the scales and results for the experiments on Tedlium 2: 
https://github.com/rwth-i6/i6_experiments/blob/main/users/gaudino/experiments/rf_conformer_att_2023/tedlium2/scales.py

### Sisyphus entry points

Train new attention models on Tedlium 2:
https://github.com/rwth-i6/i6_experiments/blob/main/users/gaudino/experiments/rf_conformer_att_2023/tedlium2/conformer_import_moh_att_train.py

Recognition with all the imported models:
https://github.com/rwth-i6/i6_experiments/blob/main/users/gaudino/experiments/rf_conformer_att_2023/tedlium2/conformer_import_moh_att_2023_10_19.py

Combining two models (system combination) in recognition with the imported models:
https://github.com/rwth-i6/i6_experiments/blob/main/users/gaudino/experiments/rf_conformer_att_2023/tedlium2/conformer_sep_enc_recogs_2024_02_02.py


## Search errors

Check here how I do it:
https://github.com/rwth-i6/i6_experiments/blob/b1cc67532e319317d062c8dd8b224eb5c4c96c57/users/gaudino/experiments/rf_conformer_att_2023/librispeech_960/conformer_import_moh_att_2023_06_30.py#L346


## ILM training (WIP)

Entry point in this file: 
https://github.com/rwth-i6/i6_experiments/blob/d26413dd4f0ca91a998d7b509b1e6e689d234248/users/gaudino/experiments/rf_conformer_att_2023/librispeech_960/conformer_import_moh_att_train.py#L405





