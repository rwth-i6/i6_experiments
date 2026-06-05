import ast
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
from sisyphus import tk

from ...data.phmm_common import TrainingDatasets
from ...phmm_default_tools import LIBRASR_WHEEL, MINI_RETURNN_ROOT, RETURNN_EXE
from ...phmm_pipeline import training
from ...phmm_rasr import CreateLibrasrVenvJob
from ...pytorch_networks.phmm.ngram_conv_lm_v2_cfg import ModelConfig


class _ReturnnDataset:
    def __init__(self, opts: Dict):
        self.opts = opts

    def as_returnn_opts(self):
        return copy.deepcopy(self.opts)


def _get_vocab_size(vocab_file: str) -> int:
    vocab = ast.literal_eval(Path(vocab_file).read_text())
    return max(vocab.values()) + 1


def _get_symbol_id(vocab_file: str, symbol: str) -> int:
    vocab = ast.literal_eval(Path(vocab_file).read_text())
    return int(vocab[symbol])


def _make_lr_schedule(num_epochs: int, const_epochs: int, start_lr: float, end_lr: float):
    decay = list(np.linspace(start_lr, end_lr, num_epochs - const_epochs + 1))[1:]
    return [start_lr] * const_epochs + decay


def _make_lm_dataset(
    *,
    corpus_file: str,
    vocab_file: str,
    partition_epoch: int,
    seq_ordering: str,
    unknown_symbol: str | None = "<unk>",
    use_cache_manager: bool = False,
):
    opts = {
        "class": "LmDataset",
        "corpus_file": corpus_file,
        "orth_symbols_map_file": vocab_file,
        "word_based": True,
        "seq_start_symbol": "<s>",
        "seq_end_symbol": "</s>",
        "auto_replace_unknown_symbol": False,
        "add_delayed_seq_data": False,
        "partition_epoch": partition_epoch,
        "seq_ordering": seq_ordering,
    }
    if unknown_symbol is not None:
        opts["unknown_symbol"] = unknown_symbol
    if use_cache_manager:
        opts["use_cache_manager"] = True
    return _ReturnnDataset(opts)


def _make_training_datasets(*, train_corpus: str, dev_corpus: str, vocab_file: str, unknown_symbol: str | None = "<unk>",
                            use_cache_manager: bool=False):
    train = _make_lm_dataset(
        corpus_file=train_corpus,
        vocab_file=vocab_file,
        partition_epoch=10,
        seq_ordering="laplace:.100",
        unknown_symbol=unknown_symbol,
        use_cache_manager=use_cache_manager,
    )
    dev = _make_lm_dataset(
        corpus_file=dev_corpus,
        vocab_file=vocab_file,
        partition_epoch=1,
        seq_ordering="sorted",
        unknown_symbol=unknown_symbol,
    )
    return TrainingDatasets(
        train=train,
        cv=dev,
        devtrain=dev,
        datastreams={},
        prior=None,
    )


def eow_phon_phmm_ls960_phoneme_ngram_conv_lm_v2():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_phoneme_ngram_conv_lm_v2"

    returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["ninja", "transformers"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
    ).out_python_bin

    data_base = "/work/asr4/zyang/corpora/librispeech/960/transcript_phonemes"
    train_corpus = f"{data_base}/splits/train-merged-960.phon-corpus.no_eow.train99.txt"
    dev_corpus = f"{data_base}/splits/train-merged-960.phon-corpus.no_eow.dev1.txt"
    vocab_file = f"{data_base}/vocab/train-merged-960.phon-corpus.no_eow.lm.vocab"

    datasets = _make_training_datasets(
        train_corpus=train_corpus,
        dev_corpus=dev_corpus,
        vocab_file=vocab_file,
    )

    num_epochs = 200
    train_config = {
        "optimizer": {"class": "RAdam"},
        "learning_rates": _make_lr_schedule(
            num_epochs=num_epochs,
            const_epochs=100,
            start_lr=5e-5,
            end_lr=1e-6,
        ),
        "batch_size": 80_000,
        "max_seqs": 200,
        "max_seq_length": {"data": 250},
        "num_workers_per_gpu": 0,
        "accum_grad_multiple_step": 1,
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 4,
            "keep": [num_epochs],
        },
    }

    model_config = ModelConfig(
        vocab_size=_get_vocab_size(vocab_file),
        embedding_dim=128,
        conv_channels=256,
        conv_kernel_size=3,
        projection_dim=256,
        dropout=0.3,
        pad_token_id=0,
        bos_token_id=_get_symbol_id(vocab_file, "<s>"),
    )
    train_job = training(
        training_name=(
            prefix_name
            + "/ngram_conv_lm_v2.phoneme_no_eow.context3_e128_c256_radam_lr5e-5_const100_decay1e-6_ep200"
        ),
        datasets=datasets,
        train_args={
            "network_module": "phmm.ngram_conv_lm_v2",
            "config": train_config,
            "net_args": {"model_config_dict": asdict(model_config)},
            "use_training_config_v2": True,
        },
        num_epochs=num_epochs,
        returnn_exe=returnn_exe,
        returnn_root=MINI_RETURNN_ROOT,
    )
    train_job.rqmt["gpu_mem"] = 24

    eow_sil_train_corpus = f"{data_base}/splits/train-merged-960.phon-corpus.sil-augmented.eow.train99.txt"
    eow_sil_dev_corpus = f"{data_base}/splits/train-merged-960.phon-corpus.sil-augmented.eow.dev1.txt"
    eow_sil_vocab_file = f"{data_base}/vocab/train-merged-960.phon-corpus.sil-augmented.eow.lm.vocab"
    eow_sil_datasets = _make_training_datasets(
        train_corpus=eow_sil_train_corpus,
        dev_corpus=eow_sil_dev_corpus,
        vocab_file=eow_sil_vocab_file,
        unknown_symbol=None,
    )
    eow_sil_model_config = ModelConfig(
        vocab_size=_get_vocab_size(eow_sil_vocab_file),
        embedding_dim=128,
        conv_channels=256,
        conv_kernel_size=3,
        projection_dim=256,
        dropout=0.3,
        pad_token_id=0,
        bos_token_id=_get_symbol_id(eow_sil_vocab_file, "<s>"),
    )
    eow_sil_num_epochs = num_epochs * 2
    eow_sil_train_config = {
        **train_config,
        "learning_rates": _make_lr_schedule(
            num_epochs=eow_sil_num_epochs,
            const_epochs=200,
            start_lr=5e-5,
            end_lr=1e-6,
        ),
        "batch_size": 160_000,
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 4,
            "keep": [eow_sil_num_epochs],
        },
    }
    eow_sil_train_job = training(
        training_name=(
            prefix_name
            + "/ngram_conv_lm_v2.phoneme_sil_eow.context3_e128_c256_radam_lr5e-5_const200_decay1e-6_ep400"
        ),
        datasets=eow_sil_datasets,
        train_args={
            "network_module": "phmm.ngram_conv_lm_v2",
            "config": eow_sil_train_config,
            "net_args": {"model_config_dict": asdict(eow_sil_model_config)},
            "use_training_config_v2": True,
        },
        num_epochs=eow_sil_num_epochs,
        returnn_exe=returnn_exe,
        returnn_root=MINI_RETURNN_ROOT,
    )
    eow_sil_train_job.rqmt["gpu_mem"] = 48

    no_eow_sil_p30_train_corpus = f"{data_base}/splits/train-merged-960.phon-corpus.no_eow.sil_p30.train99.txt"
    no_eow_sil_p30_dev_corpus = f"{data_base}/splits/train-merged-960.phon-corpus.no_eow.sil_p30.dev1.txt"
    no_eow_sil_p30_datasets = _make_training_datasets(
        train_corpus=no_eow_sil_p30_train_corpus,
        dev_corpus=no_eow_sil_p30_dev_corpus,
        vocab_file=vocab_file,
        unknown_symbol=None,
    )
    no_eow_sil_p30_model_config = ModelConfig(
        vocab_size=_get_vocab_size(vocab_file),
        embedding_dim=128,
        conv_channels=256,
        conv_kernel_size=3,
        projection_dim=256,
        dropout=0.3,
        pad_token_id=0,
        bos_token_id=_get_symbol_id(vocab_file, "<s>"),
    )
    no_eow_sil_p30_train_job = training(
        training_name=(
            prefix_name
            + "/ngram_conv_lm_v2.phoneme_no_eow_sil_p30.context3_e128_c256_radam_lr5e-5_const200_decay1e-6_ep400"
        ),
        datasets=no_eow_sil_p30_datasets,
        train_args={
            "network_module": "phmm.ngram_conv_lm_v2",
            "config": eow_sil_train_config,
            "net_args": {"model_config_dict": asdict(no_eow_sil_p30_model_config)},
            "use_training_config_v2": True,
        },
        num_epochs=eow_sil_num_epochs,
        returnn_exe=returnn_exe,
        returnn_root=MINI_RETURNN_ROOT,
    )
    no_eow_sil_p30_train_job.rqmt["gpu_mem"] = 48

    external_no_eow_train_corpus = "/work/asr4/zyang/corpora/librispeech/lm_text/phon_no_eow.txt.gz"
    external_no_eow_datasets = _make_training_datasets(
        train_corpus=external_no_eow_train_corpus,
        dev_corpus=dev_corpus,
        vocab_file=vocab_file,
        unknown_symbol=None,
        use_cache_manager=True,
    )
    external_no_eow_num_epochs = 20
    external_no_eow_train_config = {
        "optimizer": {"class": "RAdam"},
        "learning_rates": _make_lr_schedule(
            num_epochs=external_no_eow_num_epochs,
            const_epochs=10,
            start_lr=5e-5,
            end_lr=1e-6,
        ),
        "batch_size": 640_000,
        "max_seqs": 5000,
        "max_seq_length": {"data": 250},
        "num_workers_per_gpu": 0,
        "accum_grad_multiple_step": 1,
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 4,
            "keep": [external_no_eow_num_epochs],
        },
    }
    external_no_eow_model_config = ModelConfig(
        vocab_size=_get_vocab_size(vocab_file),
        embedding_dim=128,
        conv_channels=512,
        conv_kernel_size=8,
        projection_dim=512,
        dropout=0.0,
        pad_token_id=0,
        bos_token_id=_get_symbol_id(vocab_file, "<s>"),
    )
    external_no_eow_train_job = training(
        training_name=(
            prefix_name
            + "/ngram_conv_lm_v2.phoneme_no_eow_external_context8_e128_c512_radam_lr5e-5_const10_decay1e-6_ep20"
        ),
        datasets=external_no_eow_datasets,
        train_args={
            "network_module": "phmm.ngram_conv_lm_v2",
            "config": external_no_eow_train_config,
            "net_args": {"model_config_dict": asdict(external_no_eow_model_config)},
            "use_training_config_v2": True,
        },
        num_epochs=external_no_eow_num_epochs,
        returnn_exe=returnn_exe,
        returnn_root=MINI_RETURNN_ROOT,
    )
    external_no_eow_train_job.rqmt["gpu_mem"] = 48

    #




py = eow_phon_phmm_ls960_phoneme_ngram_conv_lm_v2
