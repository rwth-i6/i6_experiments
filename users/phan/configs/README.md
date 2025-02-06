Main configs for the experiments in this thesis.

In all of these configs, the ILM is a 1-layer LSTM with all same hyperparameters. These results were already reported in the thesis.

For the experiments with different architectures of the ILM, checkout the folder `ilm_archs/`.

For the experiments where ILM estimation is done using cross-domain data (Tedlium 2), please checkout the configs in the folder `estimation_ted2/` (This is anyway not good because there is a strong domain mismatch).

In general, please uncomment the respective code blocks to run the recognitions with different parameters.

`conformer_baseline.py`: recognitions on the baseline CTC model

`conformer_baseline_retrain_v2.py`: retrain the baseline with standard CTC loss (no aux CTC loss)

<!-- `conformer_ilm_kldiv_v4_fix_eos_posterior.py`: The experiment of `conformer_ilm_kldiv_v4` has a bug where the posterior of eos is all set to 0. This is to verify whether that bug greatly affects the results. The impact should not be too big, since the bug only hinders the ILM's ability to predict EOS, but there is length normalization and the extrenal LM anyway prefers shorter sequences. -->

`conformer_ilm_kldiv_v4_fixEos_noSpecAug.py`: ILM estimation using standard and sampling method.

`conformer_ilm_kldiv_masking.py`: ILM estimation using the masking method.

`conformer_ilm_kldiv_sequence_level.py`: ILM estimation using the sequence-level method.

`conformer_double_softmax_final.py`: Double softmax. The transcription LM has the same config as the ILMs. Different versions this config just change the training hyperparameters.

`conformer_lfmmi_final.py`: LF-MMI. Use a transcription bigram. Different versions this config just change the training hyperparameters.
