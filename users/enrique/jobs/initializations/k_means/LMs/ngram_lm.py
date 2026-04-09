

#########################################
# Marten count-based bigram model
#LM: "/work/smt4/marten.mueller/setups-data/ctc_baseline/work/i6_experiments/users/mueller/experiments/language_models/n_gram/ConvertARPAtoTensor.wuVkNuDg8B55/output/lm.pt"
# vocab: "/work/smt4/marten.mueller/setups-data/ctc_baseline/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.P1DXd9G7EdsU/output/bpe.vocab"

import math
import torch
import ast

lm_tensor = torch.load("/work/smt4/marten.mueller/setups-data/ctc_baseline/work/i6_experiments/users/mueller/experiments/language_models/n_gram/ConvertARPAtoTensor.wuVkNuDg8B55/output/lm.pt")

bpe_vocab_path = "/work/smt4/marten.mueller/setups-data/ctc_baseline/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.P1DXd9G7EdsU/output/bpe.vocab"

with open(bpe_vocab_path, 'r') as f:
    vocab_string = f.read()
    bpe_vocab = ast.literal_eval(vocab_string)


print(bpe_vocab["V"], bpe_vocab["Z"], bpe_vocab["A"])
print(list(enumerate(bpe_vocab)))

for idx, token in enumerate(bpe_vocab):
    print(f"Index: {idx}, Token: {token}")
    try:
        if idx == 184:
            print(f"Token at index {idx} is 'A': {token}")
        pr_c_n = lm_tensor[idx, bpe_vocab["A"]]
        
    except IndexError:
        print(f"Index {idx} is out of bounds for the tensor with shape {lm_tensor.shape}")



# idx_dog = bpe_vocab["V"]      # 82
# idx_barks = bpe_vocab["Z"]  # 451
# idx_runs = bpe_vocab["J"]    # 512


# log_prob_barks = lm_tensor[idx_dog, idx_barks]

# log_prob_runs = lm_tensor[idx_dog, idx_runs]

# print(f"Log probability of 'barks' given 'dog': {log_prob_barks.item()}")
# print(f"Log probability of 'runs' given 'dog': {log_prob_runs.item()}")
# print(f"Log probability of 'barks': {lm_tensor[idx_barks].item()}")

def log_lm_pr(c_n, previous_cs):
    return  1

class LM_n_gram():
    def __init__(self, lm_tensor_path, vocab_path):
        self.lm_tensor = torch.load(lm_tensor_path)
        with open(vocab_path, 'r') as f:
            vocab_string = f.read()
            self.vocab = ast.literal_eval(vocab_string)
        self.vocab_size = len(self.vocab)
        self.n_gram_order = self.lm_tensor.shape[0]

    def get_index(self, token):
        return self.vocab.get(token, None)

    def log_pr(self, c_n, previous_cs):
        previous_indices = [self.get_index(c) for c in previous_cs]
        
        c_n_index = self.get_index(c_n)
        
        return self.lm_tensor[previous_indices[-1], c_n_index]