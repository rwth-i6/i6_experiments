"""
Same as v2 but removing repetitions for audio indices and using different noising strategies.
"""

import copy

from i6_core.text import HeadJob
from sisyphus import tk
from typing import Any, Dict, Optional, Union, List, Tuple
import functools

from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms.librispeech.aed import model_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import learning_rate_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import optimizer_configs
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn.datastreams.base import Datastream
from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
from i6_experiments.users.schmitt.hdf import GetHdfDatasetStatisticsJob
from i6_experiments.users.schmitt.text.normalize import NormalizeLBSLMDataJob
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset, get_subepoch_dataset
from i6_experiments.users.schmitt.datasets.distrib_files import DistributedFilesDataset
from i6_experiments.users.schmitt.datasets.combine import CombinedDataset
from i6_experiments.users.schmitt.datasets.postprocessing import PostprocessingDataset
from i6_experiments.users.schmitt.util.dict_update import dict_update_deep
from i6_experiments.users.schmitt.corpus.seq_tags import GetSeqTagsFromCorpusJob

from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict

from i6_core.serialization import CallImport
from i6_core.tools.download import DownloadJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.returnn import CodeWrapper, ReturnnConfig
from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
from i6_core.text.processing import TakeNRandomLinesJob

from ..data.common import DatasetSettings, TrainingDatasets
from ..data.wav2vec import run_meta_experiments
from ..data.text import get_phonemized_lm_data, get_dev_text, get_960_text
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
base_num_epochs = 2000
base_config = {
    "__network_module": "pytorch_networks.conformer_aed_discrete_shared_v1",
    "__train_step_module": "training.aed_denoising_discrete",
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
        # "max_seq_length": {"data": 700, "phon_indices": 700},
        # "accum_grad_multiple_step": 2,  # alternate batching
        "gradient_clip_global_norm": 5.0,
        # "torch_batching": CodeWrapper("alternate_batching"),
        "max_seqs": 200,
    },
    "model_args": {
        "text_aux_loss_layers": (),
        "audio_aux_loss_layers": (),
        "num_enc_layers": 3,
        "num_text_dec_layers": 3,
        "num_audio_dec_layers": 3,
        "num_heads": 8,
        "model_dim": 512,
        "share_decoder": True,
        # "discriminator_type": "mlp",
    },
    "train_args": {
        "aux_loss_scales": (),
        "ce_loss_scale": 1.0,
        "masked_ce_loss_scale": 0.0,
        "masking_opts": {
            "mask_prob": 0.0,
            "min_span": 0,  # 1
            "max_span": 0,  # 3
        },
    }
}


label_noiser = CallImport(
    code_object_path="i6_experiments.users.schmitt.experiments.exp2025_10_02_shared_enc.postprocessing.AddLabelNoiseV1",
    hashed_arguments={
        "alpha": 5,
    },
    unhashed_arguments={},
    unhashed_package_root="",
)


def get_train_data(
    phoneme_train_hdfs: List[tk.Path],
    phoneme_dev_hdfs: List[tk.Path],
    cluster_ids_train_hdf: List[tk.Path],
    cluster_ids_dev_hdf: List[tk.Path],
    train_seq_tags: tk.Path,
    devtrain_seq_tags: tk.Path,
    dev_seq_tags: tk.Path,
    phoneme_vocab: tk.Path,
):
    return TrainingDatasets(
        train=MetaDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=cluster_ids_train_hdf,
                    segment_file=train_seq_tags,
                ),
                "phon_indices": HdfDataset(
                    files=phoneme_train_hdfs,
                    segment_file=train_seq_tags,
                    # set here because this controls which seqs are loaded
                    partition_epoch=20,
                ),
            },
            data_map={
                "data": ("feature_clusters", "data"),
                "target": ("phon_indices", "data"),
            },
            seq_order_control_dataset="phon_indices",
        ),
        devtrain=MetaDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=cluster_ids_train_hdf,
                    segment_file=devtrain_seq_tags,
                ),
                "phon_indices": HdfDataset(
                    files=phoneme_train_hdfs,
                    segment_file=devtrain_seq_tags,
                ),
            },
            data_map={
                "data": ("feature_clusters", "data"),
                "target": ("phon_indices", "data"),
            },
            seq_order_control_dataset="phon_indices",
        ),
        cv=MetaDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=cluster_ids_dev_hdf,
                    segment_file=dev_seq_tags,
                ),
                "phon_indices": HdfDataset(
                    files=phoneme_dev_hdfs,
                    segment_file=dev_seq_tags,
                ),
            },
            data_map={
                "data": ("feature_clusters", "data"),
                "target": ("phon_indices", "data"),
            },
            seq_order_control_dataset="phon_indices",
        ),
        datastreams={
            "data": LabelDatastreamWoVocab(
                available_for_inference=False,
                vocab_size=128,
            ),
            "target": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        }
    )



def py():
    prefix_name = f"experiments/librispeech/aed/ls960/{__name__.split('.')[-1]}"

    _, clusters_train, cluster_ids_train_hdf = run_meta_experiments(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=10,
        vad_concurrent=10,
        max_abs_value=1e4,
        use_correct_seq_tags_for_cluster_ids=True,
        remove_cluster_repetitions=True,
    )
    audio_stats_job = GetHdfDatasetStatisticsJob(hdf_files=cluster_ids_train_hdf)
    audio_stats_job.add_alias("data/librispeech/statistics/audio-clusters/rem-reps/train-other")
    tk.register_output(audio_stats_job.get_one_alias(), audio_stats_job.out_statistics)

    _, _, cluster_ids_dev_other_hdf = run_meta_experiments(
        librispeech_key="dev-other",
        existing_clusters=clusters_train,
        use_correct_seq_tags_for_cluster_ids=True,
        remove_cluster_repetitions=True,
    )
    _, _, cluster_ids_dev_clean_hdf = run_meta_experiments(
        librispeech_key="dev-clean",
        existing_clusters=clusters_train,
        use_correct_seq_tags_for_cluster_ids=True,
        remove_cluster_repetitions=True,
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

    text_data_960h, seq_tags_960 = get_960_text()

    phoneme_train_hdfs, phoneme_vocab, lexicon_file, phoneme_file, _ = get_phonemized_lm_data(
        alias="lm-data-wo-960h",
        text_file=normalized_lm_data,
        dump_hdf_concurrent=100,
    )
    phoneme_960h_hdfs, _, _, _, train_seq_tags = get_phonemized_lm_data(
        alias="960h",
        text_file=text_data_960h,
        dump_hdf_concurrent=10,
        lexicon_file=lexicon_file,
        phoneme_file=phoneme_file,
        seq_tag_file=seq_tags_960,
    )
    dev_text, dev_seq_tags = get_dev_text()
    phoneme_dev_hdfs, _, _, _, dev_seq_tags = get_phonemized_lm_data(
        alias="dev",
        # text_file=get_corpus_text("dev-other"),
        text_file=dev_text,
        dump_hdf_concurrent=1,
        fixed_random_subset=3000,
        lexicon_file=lexicon_file,
        phoneme_file=phoneme_file,
        seq_tag_file=dev_seq_tags
        # vocab_size=100?
    )

    devtrain_seq_tags = TakeNRandomLinesJob(
        text_file=train_seq_tags,
        num_lines=3000
    ).out
    dev_seq_tags = TakeNRandomLinesJob(
        text_file=dev_seq_tags,
        num_lines=3000
    ).out

    train_data = get_train_data(
        phoneme_train_hdfs=phoneme_960h_hdfs,
        phoneme_dev_hdfs=phoneme_dev_hdfs,
        cluster_ids_train_hdf=cluster_ids_train_hdf,
        cluster_ids_dev_hdf=cluster_ids_dev_other_hdf + cluster_ids_dev_clean_hdf,
        train_seq_tags=train_seq_tags,
        dev_seq_tags=dev_seq_tags,
        devtrain_seq_tags=devtrain_seq_tags,
        phoneme_vocab=phoneme_vocab,
    )

    for config, train_name in [
        *[(dict_update_deep(
            copy.deepcopy(base_config),
            {
                "training.batch_size": batch_size,
                "training.grad_scaler": None,
                # TODO: set this when using H100
                # "training.torch_amp": "bfloat16",
                # includes most data according to histogram
                "training.max_seq_length": {"data": 500},
                "model_args.text_out_dim": train_data.datastreams["target"].vocab_size,
                "model_args.audio_out_dim": train_data.datastreams["data"].vocab_size,
            },
        ),
            f"baseline_bs-{batch_size}"
        ) for batch_size in (
                15_000,
        )]
    ]:
        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=copy.deepcopy(config),
            train_data=train_data,
            dev_audio_hdf=cluster_ids_dev_other_hdf,
            keep_epochs=[10, 50, 100, 200, 500, 1000, 1500, 2000],
            # itc=True,
        )


def run_experiment(
    training_name: str,
    config: Dict,
    train_data,
    dev_audio_hdf: List[tk.Path],
    keep_epochs: Optional[List[int]] = None,
    itc: bool = False,
    recog_epoch: int = 1,
):
    num_epochs = config["training"].pop("__num_epochs")
    network_module = config.pop("__network_module")
    train_step_module = config.pop("__train_step_module")
    decoder_module = config.pop("__decoder_module")
    lr_opts = config["training"].pop("__lr_opts")
    lr_type = lr_opts.pop("type", "ocrl")
    if lr_type == "ocrl":
        lr_opts = learning_rate_configs.get_cfg_lrlin_oclr_by_bs_nep_v4(**lr_opts)
    else:
        assert lr_type == "constant"
        lr_opts = {"learning_rate": lr_opts["learning_rate"]}

    if keep_epochs is None:
        keep_epochs = [num_epochs]

    # batch size, adamw, speed pert, gradient clip,
    train_args = {
        "config": {
            **config["training"],
            **lr_opts,
            **config["general"]
        },
        "post_config": {
            # "tensorboard_opts": {
            #     # uneven so that both text and audio losses get logged (alternated batching)
            #     "log_every_n_train_steps": 51,
            # },
            "use_tensorboard": True,
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": keep_epochs,
            },
        },
        "python_prolog": [
            # "from i6_experiments.users.zeyer.returnn.alternate_batching import alternate_batching"
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
        use_v2_serialization=True,
        **default_returnn
    )
    if itc:
        train_job.hold()
        train_job.move_to_hpc = True

    recognition_dataset = HdfDataset(files=dev_audio_hdf)

    # eval_model_v2(
    #     training_name=training_name,
    #     train_job=train_job,
    #     train_args=train_args,
    #     train_data=train_data,
    #     decoder_config=DecoderConfig(beam_size=12),
    #     decoder_module=decoder_module,
    #     recognition_dataset=recognition_dataset,
    #     specific_epoch=recog_epoch
    # )

    return train_job
