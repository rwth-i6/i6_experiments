import copy
from sisyphus import tk
from typing import Any, Dict, Optional, Union

from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms.librispeech.aed import model_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import learning_rate_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import optimizer_configs
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn.datastreams.base import Datastream
from i6_experiments.users.schmitt.hdf import GetHdfDatasetStatisticsJob
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset, get_subepoch_dataset
from i6_experiments.users.schmitt.datasets.distrib_files import DistributedFilesDataset

from ..data.common import DatasetSettings, TrainingDatasets
from ..data.wav2vec import run_meta_experiments
from ..pipeline import training
from ...default_tools import RETURNN_EXE, RETURNN_ROOT
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


def py():
    prefix_name = f"experiments/librispeech/aed/ls960/{__name__.split('.')[-1]}"
    network_module = "pytorch_networks.conformer_aed_discrete_v1"

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

    audio_stats_job = GetHdfDatasetStatisticsJob(hdf_files=cluster_ids_dev_hdf)
    audio_stats_job.add_alias("data/librispeech/wav2vec/dev-other/audio_cluster_stats")
    tk.register_output(audio_stats_job.get_one_alias(), audio_stats_job.out_statistics)

    audio_stats_job = GetHdfDatasetStatisticsJob(hdf_files=cluster_ids_train_hdf)
    audio_stats_job.add_alias("data/librispeech/wav2vec/train/audio_cluster_stats")
    tk.register_output(audio_stats_job.get_one_alias(), audio_stats_job.out_statistics)

    train_data = TrainingDatasets(
        train=DistributedFilesDataset(
            files=cluster_ids_train_hdf,
            partition_epoch=10,
            get_subepoch_dataset=get_subepoch_dataset
        ),
        cv=HdfDataset(
            files=cluster_ids_dev_hdf
        ),
        devtrain=HdfDataset(
            files=cluster_ids_devtrain_hdf
        ),
        datastreams={
            "labels": LabelDatastreamWoVocab(
                available_for_inference=False,
                vocab_size=128,
            ),
        }
    )

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": RETURNN_ROOT,
    }
    batch_size = 10_000

    for model_alias, model_config, mask_prob, min_span, max_span, epochs in [
        ("v1", copy.deepcopy(model_configs.discrete_audio_v1), 0.3, 1, 3, 80),
        ("v1", copy.deepcopy(model_configs.discrete_audio_v1), 0.3, 4, 10, 80),
        ("v1", copy.deepcopy(model_configs.discrete_audio_v1), 0.5, 4, 10, 80),
        ("v1", copy.deepcopy(model_configs.discrete_audio_v1), 0.5, 10, 40, 80),
    ]:
        model_config["out_dim"] = train_data.datastreams["labels"].vocab_size

        train_config = {
            **optimizer_configs.v1,
            **learning_rate_configs.get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep=epochs,),
            #############
            "batch_size": batch_size,
            "max_seq_length": {"data": 700},
            "accum_grad_multiple_step": 1,
            "gradient_clip_global_norm": 5.0,
            "__num_gpus": 1,  # 4, TODO: only for debugging: set to 1 GPU
            "torch_dataloader_opts": {"num_workers": 1},  # for multi proc dataset
        }
        train_args = {
            "config": train_config,
            "post_config": {
                "torch_log_memory_usage": True
            },
            "network_module": network_module,
            "train_step_module": "training.aed_denoising_discrete",
            "net_args": model_config,
            "train_args": {
                "ce_loss_scale": 0.5,
                "masked_ce_loss_scale": 1.0,
                "masking_opts": {
                    "mask_prob": mask_prob,
                    "min_span": min_span,
                    "max_span": max_span,
                },
            },
            "debug": True,
        }

        training_name = (
            prefix_name
            + f"/{model_alias}_"
            + f"bs{batch_size}_ep{epochs}_mask{mask_prob}-spans{min_span}-{max_span}"
        )
        training(
            training_name, train_data, train_args, num_epochs=epochs, **default_returnn
        )
