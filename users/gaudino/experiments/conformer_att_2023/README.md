# Recipes for joint search with attention and CTC using Tensorflow

Uses RETURNN network dictionaries

Only one-pass time-synchronous joint search is supported.

The code generates a config file with the dictionary

Most the logic of adding the layers for joint time-synchronous search happens in this file: `users/gaudino/models/asr/decoder/ctc_decoder.py`.

## Librispeech 960h

### Sisyphus entry points

Recognition: `users/gaudino/experiments/conformer_att_2023/librispeech_960/configs/ctc_att_search_w_recombine.py`


## Tedlium 2

### Sisyphus entry points

Training: `users/gaudino/experiments/conformer_att_2023/tedlium2/configs/ted2_att_baseline.py`

Recognition: `users/gaudino/experiments/conformer_att_2023/tedlium2/configs/ted2_recogs_lm.py`