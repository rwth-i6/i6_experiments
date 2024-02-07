This is the project folder for the experiments conduction during the internship at the HA3CI Research Lab at JAIST.

As overview, the following experiments are covered:
 - torchaudio CTC and RNN-T baseline experiments for LibriSpeech 100h and 960h
 - TTS implementations for FastSpeech, GlowTTS and GradTTS aimed to be technically comparable
 - A pipeline using TTS to add synthetic training data using LibriSpeech 360h as text-only data and trained ASR combined with LibriSpeech 100h
 - A pipeline using only TTS to create a synthetic 100h corpus and evaluate the quality via ASR training