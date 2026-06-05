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
from ...pytorch_networks.phmm.ngram_conv_lm_cfg import ModelConfig


class _ReturnnDataset:
    def __init__(self, opts: Dict):
        self.opts = opts

    def as_returnn_opts(self):
        return copy.deepcopy(self.opts)


def _get_vocab_size(vocab_file: str) -> int:
    vocab = ast.literal_eval(Path(vocab_file).read_text())
    return max(vocab.values()) + 1


def _make_lr_schedule(num_epochs: int, const_epochs: int, start_lr: float, end_lr: float):
    decay = list(np.linspace(start_lr, end_lr, num_epochs - const_epochs + 1))[1:]
    return [start_lr] * const_epochs + decay


def _make_lm_dataset(*, corpus_file: str, vocab_file: str, partition_epoch: int, seq_ordering: str):
    return _ReturnnDataset(
        {
            "class": "LmDataset",
            "corpus_file": corpus_file,
            "orth_symbols_map_file": vocab_file,
            "word_based": True,
            "seq_start_symbol": "<s>",
            "seq_end_symbol": "</s>",
            "unknown_symbol": "<unk>",
            "auto_replace_unknown_symbol": False,
            "add_delayed_seq_data": False,
            "partition_epoch": partition_epoch,
            "seq_ordering": seq_ordering,
        }
    )


def _make_training_datasets(*, train_corpus: str, dev_corpus: str, vocab_file: str):
    train = _make_lm_dataset(
        corpus_file=train_corpus,
        vocab_file=vocab_file,
        partition_epoch=10,
        seq_ordering="laplace:.100",
    )
    dev = _make_lm_dataset(
        corpus_file=dev_corpus,
        vocab_file=vocab_file,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    return TrainingDatasets(
        train=train,
        cv=dev,
        devtrain=dev,
        datastreams={},
        prior=None,
    )


def eow_phon_phmm_ls960_phoneme_ngram_conv_lm():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_phoneme_ngram_conv_lm"

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

    def run_ngram_training(*, kernel_size: int, num_epochs: int, learning_rates, dropout: float, suffix: str):
        train_config = {
            "optimizer": {"class": "RAdam"},
            "learning_rates": learning_rates,
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
            conv_kernel_size=kernel_size,
            projection_dim=256,
            dropout=dropout,
            pad_token_id=0,
        )
        train_job = training(
            training_name=(
                prefix_name
                + f"/ngram_conv_lm.phoneme_no_eow.kernel{kernel_size}_e128_c256_radam_{suffix}"
            ),
            datasets=datasets,
            train_args={
                "network_module": "phmm.ngram_conv_lm",
                "config": train_config,
                "net_args": {"model_config_dict": asdict(model_config)},
                "use_training_config_v2": True,
            },
            num_epochs=num_epochs,
            returnn_exe=returnn_exe,
            returnn_root=MINI_RETURNN_ROOT,
        )
        train_job.rqmt["gpu_mem"] = 24

    for kernel_size in (1, 2, 3):
        run_ngram_training(
            kernel_size=kernel_size,
            num_epochs=50,
            learning_rates=[5e-5],
            dropout=0.0,
            suffix="lr5e-5_ep50",
        )

    lr_schedule_ep200 = _make_lr_schedule(
        num_epochs=200,
        const_epochs=100,
        start_lr=5e-5,
        end_lr=1e-6,
    )
    for kernel_size in (1, 2, 3):
        run_ngram_training(
            kernel_size=kernel_size,
            num_epochs=200,
            learning_rates=lr_schedule_ep200,
            dropout=0.3,
            suffix="lr5e-5_const100_decay1e-6_ep200",
        )


py = eow_phon_phmm_ls960_phoneme_ngram_conv_lm
