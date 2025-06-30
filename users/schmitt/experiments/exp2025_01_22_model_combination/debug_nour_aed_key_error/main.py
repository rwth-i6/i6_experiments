import json
import sys

sys.path.insert(0, "/u/schmitt/experiments/2025_01_22_model_combination/recipe/returnn")

import returnn
from returnn.datasets.util.vocabulary import BytePairEncoding

vocab = BytePairEncoding(
  seq_postfix=[0],
  unknown_label=None,
  vocab_file="/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.vocab",
  bpe_file="/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.codes"
)

fh_path = "/work/asr3/zeyer/schmitt/debug/debug_nour_1kbpe_aed_key_error/zeft.py"
with open(fh_path, "r") as file:
  fh_data = json.load(file)

seq_tag = "dev-other/7601-291468-0006/7601-291468-0006"

hypotheses = fh_data[seq_tag]
sentence_split = hypotheses[0].split()
print("sentence_split:", sentence_split)
segments = vocab.bpe.segment_sentence(hypotheses[0])
print("segments:", segments)
tokenized_hypothesis = [vocab.get_seq(hypothesis) for hypothesis in hypotheses]

print(tokenized_hypothesis)
