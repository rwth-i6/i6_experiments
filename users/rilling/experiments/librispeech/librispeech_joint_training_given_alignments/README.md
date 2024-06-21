# Joint Training using given alignments

The naming of this folder might be confusing: Despite the name "joint training" this does not contain experiments jointly training TTS and ASR like it is done in "librispeech_joint_training", but this folder contains experiments using an additional auxiliary loss to influence the latent space of Glow-TTS during TTS training. 

Additionally this setup is the newest of the setups dealing with joint modelling of TTS and ASR and therefore contains functions to create datasets for TTS and ASR with additional external durations and x-vector speaker embeddings for TTS forwarding as well as explicit model configs that are written to the Returnn Config to enforce that all model parameters are stored in the Returnn config and considered for hash computation of the returnn jobs. 

Therefore the folder [`exp_tts`](./exp_tts/) contains additional TTS-only experiments similar to the experiments in [`librispeech_glowtts`](../librispeech_glowtts/). 

The folder [`exp_joint`](./exp_joint/) contains TTS trainings with auxiliary loss, where the aux. loss is computed on the phoneme labels that are upsampled using an external  Glow-TTS Viterbi alignment, given as an HDF. 

The folder [`exp_joint_flow_ga`](./exp_joint_flow_ga/) contains similar experiments but instead of using MAS to compute the Viterbi alignment during training the external alignment is also used for the TTS itself. 

[`exp_joint_flow_ga_frozen_glowtts`](./exp_joint_flow_ga_frozen_glowtts/) additionally freezes the Glow-TTS parameters and only trains the phoneme reconstruction from different parts of the latent space, making it similar to the "encoder_test/decoder_test/encoder_sample" experiments in [`librispeech_glowtts`](../librispeech_glowtts/) and [`librispeech_glow_asr`](../librispeech_glow_asr/). 

[`exp_joint_2step`](./exp_joint_2step/) contains a mixture of experiments using two steps of training. The first block in the respective experiment.py contains trainings where in the first step a very strong auxiliary loss was used, after which the TTS is then further trained without an aux. loss in a second training. Additionally, it contains further ASR trainings using BLSTM or Conformer on a TTS trained with aux. loss to see the effect of the aux. loss on WER. 
