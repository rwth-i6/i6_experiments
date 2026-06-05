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
from ...pytorch_networks.phmm.bert_mlm_cfg import ModelConfig
from ...pytorch_networks.phmm.decoder_lm_cfg import ModelConfig as DecoderLmConfig


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
            "seq_start_symbol": "[CLS]",
            "seq_end_symbol": "[SEP]",
            "unknown_symbol": "[UNK]",
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


def eow_phon_phmm_ls960_bert_mlm():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_bert_mlm"

    returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["ninja", "transformers"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
    ).out_python_bin

    data_base = "/work/asr4/zyang/corpora/librispeech/960"
    runs = {
        "phoneme": {
            "train_corpus": f"{data_base}/mlm_splits/train-merged-960.phon-corpus.sil-augmented.no_hash.train99.txt",
            "dev_corpus": f"{data_base}/mlm_splits/train-merged-960.phon-corpus.sil-augmented.no_hash.dev1.txt",
            "vocab_file": f"{data_base}/vocab/train-merged-960.phon-corpus.sil-augmented.no_hash.bert.vocab",
        },
        "cluster_k60_merged": {
            "train_corpus": (
                f"{data_base}/mlm_splits/"
                "train-merged-960.layer15_pca512_peak_prom0p02.k60.cluster_ids.merged.train99.txt"
            ),
            "dev_corpus": (
                f"{data_base}/mlm_splits/"
                "train-merged-960.layer15_pca512_peak_prom0p02.k60.cluster_ids.merged.dev1.txt"
            ),
            "vocab_file": f"{data_base}/vocab/train-merged-960.layer15_pca512_peak_prom0p02.k60.cluster_ids.merged.bert.vocab",
        },
        "cluster_k40_merged": {
            "train_corpus": (
                f"{data_base}/mlm_splits/"
                "train-merged-960.layer15_pca512_peak_prom0p02.k40.cluster_ids.merged.train99.txt"
            ),
            "dev_corpus": (
                f"{data_base}/mlm_splits/"
                "train-merged-960.layer15_pca512_peak_prom0p02.k40.cluster_ids.merged.dev1.txt"
            ),
            "vocab_file": f"{data_base}/vocab/train-merged-960.layer15_pca512_peak_prom0p02.k40.cluster_ids.merged.bert.vocab",
        },
        "cluster_k128_merged": {
            "train_corpus": (
                f"{data_base}/mlm_splits/"
                "train-merged-960.layer15_pca512_peak_prom0p02.k128.cluster_ids.merged.train99.txt"
            ),
            "dev_corpus": (
                f"{data_base}/mlm_splits/"
                "train-merged-960.layer15_pca512_peak_prom0p02.k128.cluster_ids.merged.dev1.txt"
            ),
            "vocab_file": f"{data_base}/vocab/train-merged-960.layer15_pca512_peak_prom0p02.k128.cluster_ids.merged.bert.vocab",
        },
    }

    def run_mlm_training(
        *,
        run_name: str,
        run_data: Dict[str, str],
        num_epochs: int,
        batch_size: int,
        gpu_mem: int,
        embedding_dim: int,
        hidden_dropout: float,
        attention_dropout: float,
        suffix: str,
    ):
        const_epochs = num_epochs // 4
        learning_rates = _make_lr_schedule(
            num_epochs=num_epochs,
            const_epochs=const_epochs,
            start_lr=5e-5,
            end_lr=1e-6,
        )
        train_config = {
            "optimizer": {"class": "RAdam"},
            "learning_rates": learning_rates,
            "batch_size": batch_size,
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
        datasets = _make_training_datasets(**run_data)
        model_config = ModelConfig(
            vocab_size=_get_vocab_size(run_data["vocab_file"]),
            embedding_dim=embedding_dim,
            hidden_size=512,
            num_hidden_layers=8,
            hidden_dropout_prob=hidden_dropout,
            attention_probs_dropout_prob=attention_dropout,
            max_position_embeddings=512,
            mlm_probability=0.15,
        )
        train_job = training(
            training_name=prefix_name + f"/bert_mlm.{run_name}.layers8_h512_e{embedding_dim}_radam_lr5e-5_decay_{suffix}",
            datasets=datasets,
            train_args={
                "network_module": "phmm.bert_mlm",
                "config": train_config,
                "net_args": {"model_config_dict": asdict(model_config)},
                "use_training_config_v2": True,
            },
            num_epochs=num_epochs,
            returnn_exe=returnn_exe,
            returnn_root=MINI_RETURNN_ROOT,
        )
        train_job.rqmt["gpu_mem"] = gpu_mem

    def run_decoder_lm_training(
        *,
        run_name: str,
        run_data: Dict[str, str],
        num_epochs: int,
        batch_size: int,
        gpu_mem: int,
        embedding_dim: int,
        dropout: float,
        suffix: str,
    ):
        const_epochs = num_epochs // 4
        learning_rates = _make_lr_schedule(
            num_epochs=num_epochs,
            const_epochs=const_epochs,
            start_lr=5e-5,
            end_lr=1e-6,
        )
        train_config = {
            "optimizer": {"class": "RAdam"},
            "learning_rates": learning_rates,
            "batch_size": batch_size,
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
        datasets = _make_training_datasets(**run_data)
        model_config = DecoderLmConfig(
            vocab_size=_get_vocab_size(run_data["vocab_file"]),
            embedding_dim=embedding_dim,
            hidden_size=512,
            num_hidden_layers=8,
            dropout=dropout,
            max_position_embeddings=512,
        )
        train_job = training(
            training_name=prefix_name + f"/decoder_lm.{run_name}.layers8_h512_e{embedding_dim}_radam_lr5e-5_decay_{suffix}",
            datasets=datasets,
            train_args={
                "network_module": "phmm.decoder_lm",
                "config": train_config,
                "net_args": {"model_config_dict": asdict(model_config)},
                "use_training_config_v2": True,
            },
            num_epochs=num_epochs,
            returnn_exe=returnn_exe,
            returnn_root=MINI_RETURNN_ROOT,
        )
        train_job.rqmt["gpu_mem"] = gpu_mem

    for run_name in ("phoneme", "cluster_k60_merged"):
        run_data = runs[run_name]
        run_mlm_training(
            run_name=run_name,
            run_data=run_data,
            num_epochs=200,
            batch_size=8_000,
            gpu_mem=24,
            embedding_dim=256,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            suffix="ep200_bs8000",
        )
        run_mlm_training(
            run_name=run_name,
            run_data=run_data,
            num_epochs=600,
            batch_size=16_000,
            gpu_mem=24,
            embedding_dim=256,
            hidden_dropout=0.3,
            attention_dropout=0.0,
            suffix="ep600_bs16000",
        )

    for run_name in ("phoneme", "cluster_k40_merged"):
        run_mlm_training(
            run_name=run_name,
            run_data=runs[run_name],
            num_epochs=600,
            batch_size=80_000,
            gpu_mem=48,
            embedding_dim=128,
            hidden_dropout=0.3,
            attention_dropout=0.0,
            suffix="ep600_bs16000",
        )

    for run_name in ("phoneme", "cluster_k60_merged", "cluster_k40_merged", "cluster_k128_merged"):
        run_decoder_lm_training(
            run_name=run_name,
            run_data=runs[run_name],
            num_epochs=600,
            batch_size=80_000,
            gpu_mem=48,
            embedding_dim=128,
            dropout=0.3,
            suffix="ep600_bs80000",
        )


py = eow_phon_phmm_ls960_bert_mlm
