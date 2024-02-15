This is the project folder for the experiments conduction during the internship at the HA3CI Research Lab at JAIST.

The goal of this work was to define a consistent experiment environment in order to test different Text-To-Speech (TTS) decoder architectures for synthetic data generation.
In particular, it is about the measurement of the synthetic data quality by training automatic speech recognition systems (ASR).

As overview, this experiment folder provides the following models and pipelines:
 - 3 different ASR Baseline models for LibriSpeech 100h and LibriSpeech 960h
   - Phoneme-based Conformer CTC
   - BPE-based Conformer CTC
   - BPE-based Conformer RNN-T
 - A uniform TTS encoder, duration predictor and upsampling module that are related to FastSpeech
 - Implementations for 4 different TTS decoder architectures:
   - FastSpeech-style Transformer decoder
   - GlowTTS Flow network decoder
   - GradTTS diffusion network decoder
   - Autoregressive LSTM decoder (Tacotron2-style)
 - Provides a training and evalation pipeline to evaluate TTS/ASR combinations in multiple scenarios:
   1. Train TTS on train-clean-100h, synthesize using text and speaker labels of train-clean-100h, train ASR on the synthetic data (training data replication test)
   2. as i. but synthesize using text of train-clean-100h but shuffle the speakers (new speaker condition)
   3. as ii. but use an equivalent amount of text randomly drawn from train-clean-360 (unseen text condition)
   4. synthesize all of train-clean-360, train the ASR on both original train-clean-100 and the synthetic data (combined training with new text condition)

All models are implemented using PyTorch and trained with the lightweight "Mini-RETURNN" fork of our RETURNN deep learning toolkit.
For simplicity we use the torchaudio CTC and RNN-T decoders. The torchaudio CTC decoder uses Flashlight (written in c++) for the lexicon
based search. The torchaudio RNN-T decoder does lexicon-free beam search and is implemented purely in Python and PyTorch.


Model Structure
===============

Text-To-Speech
--------------

All TTS systems are designed to use the same encoder and duration predictor structure.
The code and structure are derived from https://github.com/jaywalnut310/glow-tts,
which follows the FastSpeech Transformer architecture.

<table>
<tr>
<td><img src="docs/figures/encoder.svg" alt="Glow-TTS at training" height="400"></td>
<td>
The code parts for the encoder can be found under `pytorch_networks/tts_shared/encoder <https://github.com/rwth-i6/i6_experiments/tree/main/users/rossenbach/experiments/jaist_project/pytorch_networks/tts_shared/encoder>`_.
A template class for all the TTS models is under `pytorch_networks/tts_shared/tts_base_model <https://github.com/rwth-i6/i6_experiments/tree/main/users/rossenbach/experiments/jaist_project/pytorch_networks/tts_shared/tts_base_model>`_
</td>
</tr>
</table>
