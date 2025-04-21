# dicts for librispeech-wise tedlium2 lm data


train_path = '/work/asr4/zyang/lm/tedlium2/dataset/librispeech_process/train/tedlium_lm_training.cleaned.bpe.txt.gz'
vocab_path = '/work/asr4/zyang/lm/tedlium2/dataset/librispeech_process/trans.bpe.vocab.lm.txt'
bpe_code_path = '/work/asr4/zyang/lm/tedlium2/dataset/librispeech_process/librispeech.trans.bpe.codes'
dev_path = '/work/asr4/zyang/lm/tedlium2/dataset/librispeech_process/dev/tedlium_dev.cleaned.bpe.txt.gz'

train_data_dict = {
    "class": "LmDataset",
    "corpus_file": train_path,
    "orth_symbols_map_file": vocab_path,
    "orth_replace_map_file": None,
    "word_based": True,
    "seq_end_symbol": "<sb>",
    "auto_replace_unknown_symbol": True,
    "unknown_symbol": "<unk>",
    "add_delayed_seq_data": True,
    "delayed_seq_data_start_symbol": "<sb>",
    "seq_ordering": "random",
    "partition_epoch": 4,
    "use_cache_manager": True,
}

dev_data_dict = {
    "class": "LmDataset",
    "corpus_file": dev_path,
    "orth_symbols_map_file": vocab_path,
    "orth_replace_map_file": None,
    "word_based": True,
    "seq_end_symbol": "<sb>",
    "auto_replace_unknown_symbol": True,
    "unknown_symbol": "<unk>",
    "add_delayed_seq_data": True,
    "delayed_seq_data_start_symbol": "<sb>",
    "seq_ordering": "random",
    "partition_epoch": 4,
    "use_cache_manager": True,
}

from returnn.tensor import Dim, batch_dim
out_spatial_dim = Dim(None, name="out-spatial", kind=Dim.Types.Spatial)
vocab_dim = Dim(description="vocab", dimension=10025, kind=Dim.Types.Spatial)
extern_data_raw = {
    "data":{
    "dim_tags": [batch_dim, out_spatial_dim],
    "sparse_dim": vocab_dim,
        "vocab": {
            "bpe_file": bpe_code_path,
            "vocab_file": vocab_path,
            "unknown_label": None,
            "bos_label": 0,
            "eos_label": 0,

        }
    }
}

def get_ted_librispeech_wise_lm_data():
    return train_data_dict, dev_data_dict, extern_data_raw
