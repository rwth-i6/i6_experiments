import copy
import inspect

from sisyphus import gs, tk, delayed_ops

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.util import default_tools_v2
from i6_core.returnn.config import CodeWrapper






# ********** Return Config generators **********

def get_dataset(path, vocab, epoch_split=10, special_symbols={}, seq_order='random'):
    eos = special_symbols.get('eos', '<sb>')
    bos = special_symbols.get('bos', eos)
    unk = special_symbols.get('unk', '<UNK>')


    return {
        "class": "LmDataset",
        #"corpus_file": CodeWrapper(delayed_ops.DelayedFormat("lambda: cf(%s)", path)),
        #"orth_symbols_map_file": CodeWrapper(delayed_ops.DelayedFormat("lambda: cf(%s)", vocab)),
        "corpus_file": path,
        "orth_symbols_map_file": vocab,
        "orth_replace_map_file": None,
        "word_based": True,
        "seq_end_symbol": eos,
        "auto_replace_unknown_symbol": True,
        "unknown_symbol": unk,
        "add_delayed_seq_data": True,
        "delayed_seq_data_start_symbol": bos,
        "seq_ordering": seq_order,
        "partition_epoch": epoch_split
    }

def get_librispeech_lm_data_word_200k(train_epoch_split=10):
    train_path = "/work/asr4/zyang/lm/librispeech/dataset/kazuki_word/train.lm.all.unk.txt.gz"
    dev_path = "/work/asr4/zyang/lm/librispeech/dataset/kazuki_word/dev.clean.other.unk.txt.gz"
    test_path = "/work/asr4/zyang/lm/librispeech/dataset/kazuki_word/test.clean.other.unk.txt.gz"
    vocab = "/work/asr4/zyang/lm/librispeech/dataset/kazuki_word/vocab.word.freq_sorted.200k.txt"
    special_symbols= {'eos': '<sb>', 'unk': '<UNK>'}
    train_num_seqs = 40699501
    seq_order_bin_size = 100
    train_seq_order = "laplace:%i" % ((train_num_seqs // train_epoch_split) // seq_order_bin_size)
    train_data = get_dataset(train_path, vocab, epoch_split=train_epoch_split, special_symbols=special_symbols, seq_order="random")
    dev_data = get_dataset(dev_path, vocab, epoch_split=1, special_symbols=special_symbols, seq_order='sorted')
    return train_data, dev_data




