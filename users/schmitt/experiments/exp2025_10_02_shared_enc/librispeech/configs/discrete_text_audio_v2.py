"""
 AED model with shared encoder and separate decoder for phoneme and audio seqs.
 Inputs: discrete phoneme and audio indices.
 Outputs: distribution over phoneme and audio indices.
 Trainin objective: denoising masked input seqs with CE loss.
"""

import copy

from sisyphus import tk
from typing import Any, Dict, Optional, Union, List, Tuple

from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms.librispeech.aed import model_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import learning_rate_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import optimizer_configs
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn.datastreams.base import Datastream
from i6_experiments.users.schmitt.hdf import GetHdfDatasetStatisticsJob
from i6_experiments.users.schmitt.text.normalize import NormalizeLBSLMDataJob
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset, get_subepoch_dataset
from i6_experiments.users.schmitt.datasets.distrib_files import DistributedFilesDataset
from i6_experiments.users.schmitt.datasets.combine import CombinedDataset
from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from i6_core.tools.download import DownloadJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.returnn import CodeWrapper, ReturnnConfig

from ..data.common import DatasetSettings, TrainingDatasets
from ..data.wav2vec import run_meta_experiments
from ..data.text import get_phonemized_lm_data, get_dev_text
from ..pipeline import training
from ...default_tools import RETURNN_EXE, RETURNN_ROOT
# from ..aed.tune_eval import eval_model_v2
# from ...recognition.aed.beam_search import DecoderConfig


class LabelDatastreamWoVocab(Datastream):
    """
    Defines a datastream for labels represented by indices using the default `Vocabulary` class of RETURNN

    This defines a word-(unit)-based vocabulary
    """

    def __init__(
        self,
        available_for_inference: bool,
        vocab_size: Union[tk.Variable, int],
    ):
        """

        :param available_for_inference:
        :param vocab: word vocab file path (pickle containing dictionary)
        :param vocab_size: used for the actual dimension
        :param unk_label: unknown label
        """
        super().__init__(available_for_inference)
        self.vocab_size = vocab_size

    def as_returnn_extern_data_opts(self, available_for_inference: Optional[bool] = None, **kwargs) -> Dict[str, Any]:
        """
        :param tk.Variable|int vocab_size: number of labels
        :rtype: dict[str]
        """
        d = {
            **super().as_returnn_extern_data_opts(available_for_inference=available_for_inference),
            "shape": (None,),
            "dim": self.vocab_size,
            "sparse": True,
        }
        d.update(kwargs)
        return d


default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}
base_num_epochs = 80
base_config = {
    "__network_module": "pytorch_networks.conformer_aed_discrete_shared_v1",
    "__train_step_module": "training.aed_denoising_discrete_shared",
    "__baseline_alias": "v1",
    "__decoder_module": "recognition.aed",
    "train_rqmt": {
        "cpu_rqmt": 24,
    },
    "general": {
        "torch_dataloader_opts": {"num_workers": 1},  # for multi proc dataset
    },
    "training": {
        "__num_gpus": 1,
        "__num_epochs": base_num_epochs,
        "__lr_opts": {
            "n_ep": base_num_epochs
        },
        # "torch_amp": "bfloat16",  # no effect for 11gb GPUs
        "batch_size": {"data": 9000, "phon_indices": 9000},
        **optimizer_configs.v1,
        "max_seq_length": {"data": 700, "phon_indices": 700},  # 30 seconds
        "accum_grad_multiple_step": 2,  # alternate batching
        "gradient_clip_global_norm": 5.0,
        "torch_batching": CodeWrapper("alternate_batching"),
    },
    "model_args": {
        "aux_loss_layers": (),
        "num_enc_layers": 6,
        "num_text_dec_layers": 3,
        "num_audio_dec_layers": 3,
        "num_heads": 8,
        "model_dim": 512,
    },
    "train_args": {
        "text_ce_loss_scale": 0.5,
        "text_masked_ce_loss_scale": 1.0,
        "audio_ce_loss_scale": 0.5,
        "audio_masked_ce_loss_scale": 1.0,
        "text_masking_opts": {
            "mask_prob": 0.3,
            "min_span": 1,
            "max_span": 3,
        },
        "audio_masking_opts": {
            "mask_prob": 0.3,
            "min_span": 1,
            "max_span": 3,
        },
    }
}


def py():
    prefix_name = f"experiments/librispeech/aed/ls960/{__name__.split('.')[-1]}"

    _, clusters_train, cluster_ids_train_hdf = run_meta_experiments(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=10,
        vad_concurrent=10,
        max_abs_value=1e4,
    )
    _, _, cluster_ids_devtrain_hdf = run_meta_experiments(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=1,
        vad_concurrent=10,
        fixed_random_subset=2832,  # number of seqs in dev_other  # 3000,
    )
    _, _, cluster_ids_dev_hdf = run_meta_experiments(
        librispeech_key="dev-other",
        existing_clusters=clusters_train,
    )
    _, _, cluster_ids_dev_correct_seq_tags_hdf = run_meta_experiments(
        librispeech_key="dev-other",
        existing_clusters=clusters_train,
        use_tsv_for_cluster_ids=True
    )

    raw_lm_corpus = DownloadJob(
        url="https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm_corpus/librispeech_lm_corpus.minus_librivox.metadata_and_manual_and_missing.corpus.txt"
    ).out_file
    tk.register_output("data/librispeech/lm/lbs_lm_minus_librivox.raw", raw_lm_corpus)

    wav2letter_repo = CloneGitRepositoryJob(
        "https://github.com/flashlight/wav2letter",
        commit="e5a4b62d87f15fde6a963d9ac174c8db8eb67fbc",
        checkout_folder_name="wav2letter",
    ).out_repository
    normalized_lm_data = NormalizeLBSLMDataJob(
        wav2letter_root=wav2letter_repo,
        wav2letter_python_exe=tk.Path("/work/asr4/schmitt/venvs/wav2letter_lm_corpus/bin/python3"),
        librispeech_lm_corpus=raw_lm_corpus,
    ).out_corpus_norm

    phoneme_train_hdfs, phoneme_vocab, lexicon_file, phoneme_file = get_phonemized_lm_data(
        alias="train-960h-filtered",
        text_file=normalized_lm_data,
        dump_hdf_concurrent=40,
    )
    phoneme_devtrain_hdfs, _, _, _ = get_phonemized_lm_data(
        text_file=normalized_lm_data,
        dump_hdf_concurrent=1,
        fixed_random_subset=3000,
    )
    phoneme_dev_hdfs, _, _, _ = get_phonemized_lm_data(
        alias="dev",
        # text_file=get_corpus_text("dev-other"),
        text_file=get_dev_text(),
        dump_hdf_concurrent=1,
        fixed_random_subset=3000,
        lexicon_file=lexicon_file,
        phoneme_file=phoneme_file,
        # vocab_size=100?
    )

    train_data = TrainingDatasets(
        train=CombinedDataset(
            datasets={
                "phonemes": DistributedFilesDataset(
                    files=phoneme_train_hdfs,
                    partition_epoch=40,
                    get_subepoch_dataset=get_subepoch_dataset
                ),
                "audio_features": DistributedFilesDataset(
                    # upsample audio data to match phoneme data size
                    # phoneme data has 18x more frames than audio data
                    # set partition_epoch 4x smaller and use 5x replication -> 20x
                    files=cluster_ids_train_hdf * 5,
                    partition_epoch=10,
                    get_subepoch_dataset=get_subepoch_dataset
                ),
            },
            data_map={
                ("phonemes", "data"): "phon_indices",
                ("audio_features", "data"): "data",
            },
            seq_ordering="interleave",
            partition_epoch=1,
        ),
        cv=CombinedDataset(
            datasets={
                "phonemes": HdfDataset(
                    files=phoneme_dev_hdfs
                ),
                "audio_features": HdfDataset(
                    files=cluster_ids_dev_hdf
                ),
            },
            data_map={
                ("phonemes", "data"): "phon_indices",
                ("audio_features", "data"): "data",
            },
            seq_ordering="sorted",
            partition_epoch=1,
        ),
        devtrain=CombinedDataset(
            datasets={
                "phonemes": HdfDataset(
                    files=phoneme_devtrain_hdfs
                ),
                "audio_features": HdfDataset(
                    files=cluster_ids_devtrain_hdf
                ),
            },
            data_map={
                ("phonemes", "data"): "phon_indices",
                ("audio_features", "data"): "data",
            },
            seq_ordering="sorted",
            partition_epoch=1,
        ),
        datastreams={
            "features": LabelDatastreamWoVocab(
                available_for_inference=False,
                vocab_size=128,
            ),
            "labels": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        }
    )

    for config, train_name in [
        *[(dict_update_deep(
            copy.deepcopy(base_config),
            {
                "training.batch_size": {"data": batch_size, "phon_indices": batch_size},
                # "training.grad_scaler": None,
                "train_args.text_masking_opts.mask_prob": mask_prob,
                "train_args.text_masking_opts.min_span": min_span,
                "train_args.text_masking_opts.max_span": max_span,
                "train_args.audio_masking_opts.mask_prob": mask_prob,
                "train_args.audio_masking_opts.min_span": min_span,
                "train_args.audio_masking_opts.max_span": max_span,
                "model_args.text_out_dim": train_data.datastreams["labels"].vocab_size,
                "model_args.audio_out_dim": train_data.datastreams["features"].vocab_size,
            }
        ),
            f"baseline_bs-{batch_size}"
        ) for batch_size, mask_prob, min_span, max_span in (
                (9_000, 0.3, 1, 3),
        )]
    ]:

        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=copy.deepcopy(config),
            train_data=train_data,
            dev_audio_hdf=cluster_ids_dev_correct_seq_tags_hdf,
            keep_epochs=[5],
        )

    for config, train_name in [
        *[(dict_update_deep(
            copy.deepcopy(base_config),
            {
                "training.batch_size": {"data": batch_size, "phon_indices": batch_size},
                # "training.grad_scaler": None,
                "train_args.text_masking_opts.mask_prob": text_mask_prob,
                "train_args.text_masking_opts.min_span": text_min_span,
                "train_args.text_masking_opts.max_span": text_max_span,
                "train_args.audio_masking_opts.mask_prob": audio_mask_prob,
                "train_args.audio_masking_opts.min_span": audio_min_span,
                "train_args.audio_masking_opts.max_span": audio_max_span,
                "model_args.text_out_dim": train_data.datastreams["labels"].vocab_size,
                "model_args.audio_out_dim": train_data.datastreams["features"].vocab_size,
            }
        ),
            f"baseline_bs-{batch_size}_tm-{text_mask_prob}_ts-{text_min_span}-{text_max_span}_am-{audio_mask_prob}_as-{audio_min_span}-{audio_max_span}"
        ) for batch_size, text_mask_prob, text_min_span, text_max_span, audio_mask_prob, audio_min_span, audio_max_span in (
                (9_000, 0.3, 1, 3, 0.3, 10, 40),
                (9_000, 0.3, 1, 3, 0.3, 5, 15),
                (9_000, 0.3, 5, 15, 0.3, 10, 40),
                (9_000, 0.4, 5, 15, 0.4, 10, 40),
                (9_000, 0.4, 5, 15, 0.4, 5, 15),
        )]
    ]:

        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=copy.deepcopy(config),
            train_data=train_data,
            dev_audio_hdf=cluster_ids_dev_hdf,
            keep_epochs=list(range(base_config["training"]["__num_epochs"] // 5, base_config["training"]["__num_epochs"] + 1, base_config["training"]["__num_epochs"] // 5)),
        )


def run_experiment(
    training_name: str,
    config: Dict,
    train_data,
    dev_audio_hdf: List[tk.Path],
    keep_epochs: Optional[List[int]] = None,
):
    num_epochs = config["training"].pop("__num_epochs")
    network_module = config.pop("__network_module")
    train_step_module = config.pop("__train_step_module")
    decoder_module = config.pop("__decoder_module")
    lr_opts = config["training"].pop("__lr_opts")

    if keep_epochs is None:
        keep_epochs = [num_epochs]

    # batch size, adamw, speed pert, gradient clip,
    train_args = {
        "config": {
            **config["training"],
            **learning_rate_configs.get_cfg_lrlin_oclr_by_bs_nep_v4(**lr_opts),
            **config["general"]
        },
        "post_config": {
            "tensorboard_opts": {
                # uneven so that both text and audio losses get logged (alternated batching)
                "log_every_n_train_steps": 51,
            },
            "use_tensorboard": True
        },
        "python_prolog": [
            "from i6_experiments.users.zeyer.returnn.alternate_batching import alternate_batching"
        ],
        "network_module": network_module,
        "train_step_module": train_step_module,
        "net_args": config["model_args"],
        "train_args": config["train_args"],
        "rqmt": config.pop("train_rqmt", {}),
        "debug": True,
    }
    train_job = training(
        training_name,
        train_data,
        train_args,
        num_epochs=num_epochs,
        **default_returnn
    )

    recognition_dataset = HdfDataset(files=dev_audio_hdf)

    # eval_model_v2(
    #     training_name=training_name,
    #     train_job=train_job,
    #     train_args=train_args,
    #     train_data=train_data,
    #     decoder_config=DecoderConfig(beam_size=12),
    #     decoder_module=decoder_module,
    #     recognition_dataset=recognition_dataset,
    #     specific_epoch=1
    # )

    return train_job
