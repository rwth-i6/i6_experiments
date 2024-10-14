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
        "auto_replace_unknown_symbol": True,
        "unknown_symbol": unk,
        "add_delayed_seq_data": True,
        "delayed_seq_data_start_symbol": bos,
        "seq_ordering": seq_order,
        "partition_epoch": epoch_split
    }

def get_tedlium2_lm_data_word_152k(train_epoch_split=4, pretrain=True):
    if pretrain:
        train_path = ["/work/asr4/zyang/lm/tedlium2/dataset/kazuki_word/train.en.gz",
              "/work/asr4/zyang/lm/tedlium2/dataset/kazuki_word/commoncrawl-9pc.en.gz",
              "/work/asr4/zyang/lm/tedlium2/dataset/kazuki_word/europarl-v7-6pc.en.gz",
              "/work/asr4/zyang/lm/tedlium2/dataset/kazuki_word/giga-fren-4pc.en.gz",
              "/work/asr4/zyang/lm/tedlium2/dataset/kazuki_word/news-18pc.en.gz",
              "/work/asr4/zyang/lm/tedlium2/dataset/kazuki_word/news-commentary-v8-9pc.en.gz",
              "/work/asr4/zyang/lm/tedlium2/dataset/kazuki_word/yandex-1m-31pc.en.gz"]
    else:

        train_path = ["/work/asr4/zyang/lm/tedlium2/dataset/kazuki_word/train.en.gz",
                      "/work/asr4/zyang/lm/tedlium2/dataset/kazuki_word/commoncrawl-9pc.en.gz"]
    dev_path = "/work/asr4/zyang/lm/tedlium2/dataset/kazuki_word/dev.en.gz"
    test_path = "/work/asr4/zyang/lm/tedlium2/dataset/kazuki_word/eval.en.gz"
    vocab = "/work/asr4/zyang/lm/tedlium2/dataset/kazuki_word/vocab.returnn.txt"
    special_symbols= {'eos': '<sb>', 'unk': '<unk>'}
    seq_order_bin_size = 100
    train_seq_order = "random"
    train_data = get_dataset(train_path, vocab, epoch_split=train_epoch_split, special_symbols=special_symbols, seq_order=train_seq_order)
    dev_data = get_dataset(dev_path, vocab, epoch_split=1, special_symbols=special_symbols, seq_order='sorted')
    return train_data, dev_data




