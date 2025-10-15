import torch
import numpy as np

def ctc_decoder(log_probs, audio_len, blank_idx=9323
, vocab=None):

    log_probs = log_probs.numpy()
    
    max_indices = np.argmax(log_probs, axis=1)
    
    decoded_sequence_idx = []
    hypothesis = []
    prev_label = blank_idx

    len_int = audio_len.detach().cpu().type(torch.int64).numpy()
    print(len_int)
    
    for i in range(len_int):
        current_label = max_indices[i]
        if current_label != blank_idx and current_label != prev_label:
            hyp = vocab[current_label[3:]]
            if hyp[-2:] == "@@":
                hypothesis.append(hyp)
            else:
                hypothesis.append(f"{hyp} ")
            decoded_sequence.append(current_label)
        prev_label = current_label
    
    return hypothesis, decoded_sequence