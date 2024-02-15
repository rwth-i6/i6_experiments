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


Experiment Design
=================

A simple sketch of "Scenario 1" (i., ii. and iii. from above) looks as follows:

![pipeline_synthetic](/users/rossenbach/experiments/jaist_project/docs/figures/pipeline_synthetic.svg)

A simple sketch of "Scenario 4" (iv. from above) looks as follows:

![pipeline_synthetic](/users/rossenbach/experiments/jaist_project/docs/figures/pipeline_combined.svg)


Model Structure
===============

Text-To-Speech
--------------

All TTS systems are designed to use the same encoder and duration predictor structure.
The code and structure are derived from [GlowTTS](https://github.com/jaywalnut310/glow-tts),
which follows the FastSpeech Transformer architecture.
Please note that in contrast to the FastSpeech2 implementation no F0 and no energy prediction is used.

![Encoder](/users/rossenbach/experiments/jaist_project/docs/figures/encoder.svg)

The code parts for the encoder can be found under [pytorch_networks/tts_shared/encoder](/users/rossenbach/experiments/jaist_project/pytorch_networks/tts_shared/encoder) .

A template class for all the TTS models is under [pytorch_networks/tts_shared/tts_base_model](/users/rossenbach/experiments/jaist_project/pytorch_networks/tts_shared/tts_base_model)

The setup includes an implementation of the following 4 TTS decoder architectures, depicted in a simplifying sketch:


![Encoder](/users/rossenbach/experiments/jaist_project/docs/figures/tts_deocder.svg)

a) Transformer TTS: This encoder is a simple Transformer with convolution layers instead of feed-forward layers.

b) GlowTTS: This model follows exactly the GlowTTS code.
The code is taken from the [GlowTTS repo](https://github.com/jaywalnut310/glow-tts) and only refactored to fit
the code style and structure of this setup.

c) GradTTS: As for the GlowTTS, also this model is derived from the original code base published with the paper.
But in addition, we had to add a speaker condition mean prediction network, as the original architecture requires conditioning the full encoder on the speakers.
In order to avoid this, we used 2 extra Transformer layers as prediction network following the same style as the encoder itself.

d) Autoregressive LSTM (Tacotron2-style): This model is derived from the ESPNet Tacotron2 implementation.
The attention mechanism was completely removed, and the instead for each time frame we just the upsampled encoder state as input.


Automatic Speech Recognition
----------------------------

All ASR systems use a conformer encoder with 12 layers.
For training on train-clean-100h and on synthetic data, we use a base size of 384.
For training on the full LibriSpeech data, we use a base size of 512.

For the exact parameters, please look into the respective [experiment files](/users/rossenbach/experiments/jaist_project/exp_asr).

### ASR References:

A starting point for information on the CTC decoder can be found [here (torchaudio documentation)](https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html).

The C++ code for the Torchaudio / Flashlight decoder implementation can be found [here (Flashlight/text)](https://github.com/flashlight/text/blob/main/flashlight/lib/text/decoder/LexiconDecoder.cpp).

Literature:
 - [Conformer paper, Gulati et al. 2020](https://arxiv.org/abs/2005.08100)
 - [CTC decoder paper, Collobert et al. 2016](https://arxiv.org/pdf/1609.03193.pdf)


Language Modelling
------------------

The setup currently only allows LM combination via ARPA models for CTC decoding.
The official 4-gram LM from LibriSpeech will automatically be downloaded when running the setup.
Neural language model support, and language model support for RNN-T is planned for the future.
