# Experiments on the ILM of CTC

This `users` folder is dedicated to experiments on the ILM of the CTC model for the Master's thesis "Investigating the Internal Language Model Estimation and Suppression Including the Encoder for Automatic Speech Recognition". Some results in this thesis are also published in the paper "Label-Context-Dependent Internal Language Model Estimation for CTC" at INTERSPEECH 2025. The experiments are implemented using RETURNN-Frontend (RF) with Torch backend and are run via the Sisyphus workflow manager.

Notes: The "smoothing" method in the paper was initially referred to as "sampling". This can be commonly found in older code namings. Newer codes refer to this method correctly as "smoothing".

To rerun an experiment using a Sisyphus config, use the command:
```
/work/tools/users/zeyer/py-envs/py3.11-torch2.1/bin/python3.11 ./sis m <relative path to the Sisyphus config>
```
Configs are usually located under `recipe/i6_experiments/users/phan/configs/`.

The work directory for these experiments can be found in the i6 cluster under `/work/asr3/zyang/share/mnphan/work_rf_ctc/work`. You can import Sisyphus jobs from this directory before running further experiments.

An overview of the important modules:

- Main functions for training and recognition using transcribed audio data can be located in `users/yang/torch/luca_ctc/train.py` and `users/yang/torch/luca_ctc/recog.py`.
- `users/yang/torch/loss/ctc_pref_scores_loss.py` contains several important functions and losses:
    - `log_ctc_pref_beam_scores` computes the CTC prefix probability.
    - `kldiv_ctc_lm_loss` implements the KL-Divergence loss for ILM estimation.
    - `kldiv_ctc_lm_sample_batch_loss` implements KL-Divergence loss with smoothing for ILM estimation.
    - `ctc_double_softmax_loss` implements double softmax for ILM suppression.
- `alignment/` contains functions related to converting a GMM alignment to a word-level alignment (with pseudo-words). This is mainly for the ILM training with masking.
- `configs/` contains Sisyphus configs for all experiments.
- `datasets/` contains definition and utilities for some datasets.
- `forward_misc/` contains RETURNN forward step and callbacks for certain tasks such as computing the PPL of an LM.
- `lbs_transcription_bpe10k/`, `train/`, `train_transcription_lm/` contains RETURNN training configs for transcription LMs on LibriSpeech using text-only dataset.
- `prior/` contains RETURNN forward callback and configs for computing the frame-level prior.
- `recog/` contains several search implementation. `ctc_time_sync_recomb_first_v2.py` is used in the final results in the thesis and the paper. Others are for earlier experiments and debugging.
- `rescoring/` contains recognition functions for a two-pass rescoring approach.
- `rf_models/` contains model and train step definitions in the RF framework.
- `utils/` contains some common utility functions about tensor ops, sequence mask and padding, masking alignment, and computing losses for bidirectional ILM.
- `ctc_ilm_sequence_level_loss.py` implements sequence-level KD training criterion for ILM estimation.
- `ctc_lf_mmi.py` implements the lattice-free MMI training criterion for ILM suppression.
- `ctc_masked_score.py` computes the CTC posterior probabilities when some target labels are masked out (more details are in the thesis' appendix). This is used for training of bidirectional ILM. Related losses are implemented in `ctc_masked_score_loss.py`.
