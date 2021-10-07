import os
from sisyphus import tk

from .data import get_silence_processed_dataset_group

def test():
    path = "experiments/librispeech/librispeech_100_ar_tts/prototype_pipeline"
    dataset_group = get_silence_processed_dataset_group()
    train_corpus = dataset_group.get_segmented_corpus_object("train-clean-100-tts-train")[0].corpus_file
    tk.register_output(os.path.join(path, "processed_train_corpus.xml.gz"), train_corpus)