"""
Parameters, to be imported from Sisyphus so make sure there is nothing unnecessary loaded
"""

from dataclasses import dataclass

@dataclass
class ConvBlstmRecParams:
    audio_emb_size: int
    speaker_emb_size: int
    conv_hidden_size: int
    enc_lstm_size: int
    rec_lstm_size: int
    dropout: float
    reconstruction_scale: float
    training: bool