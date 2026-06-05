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
from ...pytorch_networks.phmm.ngram_conv_lm_v3_cfg import ModelConfig


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
    unknown_symbol: str | None = None,
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


def _make_training_datasets(*, train_corpus: str, dev_corpus: str, vocab_file: str):
    train = _make_lm_dataset(
        corpus_file=train_corpus,
        vocab_file=vocab_file,
        partition_epoch=10,
        seq_ordering="laplace:.100",
        unknown_symbol=None,
        use_cache_manager=True,
    )
    dev = _make_lm_dataset(
        corpus_file=dev_corpus,
        vocab_file=vocab_file,
        partition_epoch=1,
        seq_ordering="sorted",
        unknown_symbol=None,
        use_cache_manager=False,
    )
    return TrainingDatasets(
        train=train,
        cv=dev,
        devtrain=dev,
        datastreams={},
        prior=None,
    )


def eow_phon_phmm_ls960_phoneme_ngram_conv_lm_v3():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_phoneme_ngram_conv_lm_v3"

    returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["ninja", "transformers"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
    ).out_python_bin

    data_base = "/work/asr4/zyang/corpora/librispeech/960/transcript_phonemes"
    train_corpus = "/work/asr4/zyang/corpora/librispeech/lm_text/phon_no_eow.txt.gz"
    dev_corpus = f"{data_base}/splits/train-merged-960.phon-corpus.no_eow.dev1.txt"
    vocab_file = f"{data_base}/vocab/train-merged-960.phon-corpus.no_eow.lm.vocab"

    datasets = _make_training_datasets(
        train_corpus=train_corpus,
        dev_corpus=dev_corpus,
        vocab_file=vocab_file,
    )

    num_epochs = 20
    train_config = {
        "optimizer": {"class": "RAdam"},
        "learning_rates": _make_lr_schedule(
            num_epochs=num_epochs,
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
            "keep": [num_epochs],
        },
    }

    def run_lm(*, run_suffix: str, conv_kernel_sizes: tuple[int, ...], conv_dilations: tuple[int, ...]):
        model_config = ModelConfig(
            vocab_size=_get_vocab_size(vocab_file),
            embedding_dim=128,
            conv_channels=512,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_dilations=conv_dilations,
            num_conv_layers=len(conv_kernel_sizes),
            projection_dim=512,
            dropout=0.0,
            pad_token_id=0,
            bos_token_id=_get_symbol_id(vocab_file, "<s>"),
        )
        train_job = training(
            training_name=prefix_name + "/" + run_suffix,
            datasets=datasets,
            train_args={
                "network_module": "phmm.ngram_conv_lm_v3",
                "config": train_config,
                "net_args": {"model_config_dict": asdict(model_config)},
                "use_training_config_v2": True,
            },
            num_epochs=num_epochs,
            returnn_exe=returnn_exe,
            returnn_root=MINI_RETURNN_ROOT,
        )
        train_job.rqmt["gpu_mem"] = 48

    run_lm(
        run_suffix="ngram_conv_lm_v3.phoneme_no_eow_external_context8_k4_5_d1_1_e128_c512_radam_lr5e-5_const10_decay1e-6_ep20",
        conv_kernel_sizes=(4, 5),
        conv_dilations=(1, 1),
    )
    run_lm(
        run_suffix="ngram_conv_lm_v3.phoneme_no_eow_external_context8_k2_4_d1_2_e128_c512_radam_lr5e-5_const10_decay1e-6_ep20",
        conv_kernel_sizes=(2, 4),
        conv_dilations=(1, 2),
    )


py = eow_phon_phmm_ls960_phoneme_ngram_conv_lm_v3
