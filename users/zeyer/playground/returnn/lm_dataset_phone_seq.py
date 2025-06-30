"""
Start via returnn/tools/dump-dataset.py.
Eg:

    ~/work/py-envs/py3.12-torch2.5/bin/python tools/returnn/tools/dump-dataset.py recipe/i6_experiments/users/zeyer/playground/returnn/lm_dataset_phone_seq.py

"""

train = {
    "class": "LmDataset",
    "corpus_file": "work/i6_core/tools/download/DownloadJob.g4jClO48cAvP/output/librispeech-lm-norm.txt.gz",
    "dtype": "int32",
    "phone_info": {
        "add_silence_beginning": 0.01,
        "add_silence_between_words": 0.9,
        "add_silence_end": 0.01,
        "allo_num_states": 1,
        "extra_begin_lemma": {
            "phons": [
                {"phon": "[start]"},
            ]
        },
        "extra_end_lemma": {
            "phons": [
                {"phon": "[end]"},
            ]
        },
        "lexicon_file": "work/i6_core/lexicon/modification/MergeLexiconJob.NCxgAZ1OMbJ1/output/lexicon.xml.gz",
        "phoneme_vocab_file": "work/i6_core/returnn/vocabulary/ReturnnVocabFromPhonemeInventory.z2RlZd9Y0jWQ/output/vocab.pkl",
        "repetition": 0.01,
        "silence_lemma_orth": "[space]",
        "silence_repetition": 0.05,
    },
    "seq_end_symbol": None,
    "unknown_symbol": None,
}
