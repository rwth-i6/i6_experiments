import copy
import inspect

from sisyphus import gs, tk, delayed_ops

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.util import default_tools_v2
from i6_core.returnn.config import CodeWrapper


# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 120
num_lstm_layers = 1
embed_dim = 128
dropout = 0.1
hidden_dim = 640
init_learning_rates = [1e-2, 1e-3, 1e-4]
train_split_epoch = 10

tools = copy.deepcopy(default_tools_v2)
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


# ********** Return Config generators **********

def get_dataset(path, vocab, epoch_split=10, special_symbols={}, seq_order='random'):
    eos = special_symbols.get('eos', '<sb>')
    bos = special_symbols.get('bos', eos)
    unk = special_symbols.get('unk', '<unk>')


    return {
        "class": "LmDataset",
        #"corpus_file": CodeWrapper(delayed_ops.DelayedFormat("lambda: cf(%s)", path)),
        #"orth_symbols_map_file": CodeWrapper(delayed_ops.DelayedFormat("lambda: cf(%s)", vocab)),
        "corpus_file": path,
        "orth_symbols_map_file": vocab,
        "orth_replace_map_file": None,
        "word_based": True,
        "seq_end_symbol": eos,
        "auto_replace_unknown_symbol": False,
        "unknown_symbol": unk,
        "add_delayed_seq_data": True,
        "delayed_seq_data_start_symbol": bos,
        "seq_ordering": seq_order,
        "partition_epoch": epoch_split
    }

def get_librispeech_lm_data_bpe_10k(train_epoch_split=4):
    train_path = "/work/asr4/zyang/lm/librispeech/dataset/kazuki_bpe/librispeech-lm-norm.bpe.txt.gz"
    dev_path = "/work/asr4/zyang/lm/librispeech/dataset/kazuki_bpe/dev.clean.other.bpe.txt.gz"
    test_path = "/work/asr4/zyang/lm/librispeech/dataset/kazuki_bpe/test.clean.other.txt.gz"
    vocab = "/work/asr4/zyang/lm/librispeech/dataset/kazuki_bpe/trans.bpe.vocab.lm.txt"
    special_symbols= {'eos': '<sb>', 'unk': '<UNK>'}
    train_num_seqs = 40418260
    seq_order_bin_size = 100
    train_seq_order = "laplace:%i" % ((train_num_seqs // train_epoch_split) // seq_order_bin_size)
    train_data = get_dataset(train_path, vocab, epoch_split=train_epoch_split, special_symbols=special_symbols, seq_order=train_seq_order)
    dev_data = get_dataset(dev_path, vocab, epoch_split=1, special_symbols=special_symbols, seq_order='sorted')
    return train_data, dev_data




