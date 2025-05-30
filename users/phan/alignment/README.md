This module contains codes related to the CTC alignment. The most important config is `convert.py`, where the LibriSpeech GMM alignment is converted to the 78 augmented CTC phonemes, and then the CTC phoneme alignment is converted to word level alignment. Other modules have codes for aligning a Bi-LSTM CTC, which was anyway discarded.

The alignments coming out of `convert.py` can be found under `/work/asr3/zyang/share/mnphan/alignment_data`.
