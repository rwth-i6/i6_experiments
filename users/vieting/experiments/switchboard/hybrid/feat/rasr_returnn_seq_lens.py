"""
Check size of context window which always results in the same output sequence lengths.
"""
from typing import Callable


# Case 1: ogg waveforms are synched with feature cache
# such that the waveform length is always 80 times the alignment length
def check_context_window(c_win: int, fn: Callable) -> bool:
    for seq_len_classes in range(1, 1000):
        if fn(seq_len_classes * 80 + c_win) != seq_len_classes:
            print(f"resulting len: {fn(seq_len_classes * 80 + c_win)}, ref: {seq_len_classes}")
            return False
    return True


# Gammatone
def gammatone_len(len_in):
    len_out = (len_in - 320) // 1 + 1
    len_out = (len_out - 200) // 80 + 1
    return len_out


# for single convolution: context_window = window_size - window_shift + 1
# use here: 320 - 1 + 1 + (200 - 80 + 1) * 1 = 441
assert check_context_window(441, gammatone_len)


# SCF
def scf_len(len_in):
    len_out = (len_in - 128) // 5 + 1
    len_out = (len_out - 40) // 16 + 1
    return len_out


# for single convolution: context_window = window_size - window_shift + 1
# use here: 128 - 5 + 1 + (40 - 16 + 1) * 5 = 249
assert check_context_window(249, scf_len)


# log Mel
def log_mel_len(len_in):
    len_out = (len_in - 200) // 80 + 1
    return len_out


# for single convolution: context_window = window_size - window_shift + 1
# use here: 200 - 80 + 1 = 121
assert check_context_window(121, log_mel_len)

