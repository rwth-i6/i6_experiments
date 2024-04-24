"""
experiments
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy
from sisyphus import tk

# from i6_core.datasets.tf_datasets import DownloadAndPrepareTfDatasetJob
# from i6_core.datasets.huggingface import DownloadAndPrepareHuggingFaceDatasetJob
from i6_core.returnn import ReturnnConfig, ReturnnTrainingJob
from i6_experiments.common.setups.returnn_common import serialization
from returnn_common import nn


def run():
    """run"""
    for name, path in librispeech_ogg_zip_dict.items():
        tk.register_output(f"librispeech/dataset/{name}", path)

    tk.register_output("librispeech/sentencepiece-2k.model", spm_2k)

    # tk.register_output(
    #  "librispeech/huggingface-dataset-clean",
    #  DownloadAndPrepareHuggingFaceDatasetJob("librispeech_asr", "clean").out_dir)

    input_dim = nn.FeatureDim("input", 80)
    time_dim = nn.SpatialDim("time")
    targets_time_dim = nn.SpatialDim("targets-time")
    output_dim = nn.FeatureDim("output", 2000)

    from returnn_common.asr import specaugment

    class Model(nn.Module):
        """model"""

        def __init__(self, out_dim: nn.Dim, conformer_dim: Optional[nn.Dim] = None, **kwargs):
            super(Model, self).__init__()
            # Medium size default...
            if conformer_dim is None:
                conformer_dim = nn.FeatureDim("conformer", 256)
            kwargs.setdefault("num_layers", 16)
            kwargs.setdefault("num_heads", 4)
            self.conformer = nn.ConformerEncoder(conformer_dim, **kwargs)
            self.out_dim = out_dim
            self.output = nn.Linear(out_dim)

        def __call__(self, x: nn.Tensor, *, in_spatial_dim: nn.Dim) -> Tuple[nn.Tensor, nn.Dim]:
            x = specaugment.specaugment_v2(x, spatial_dim=in_spatial_dim)
            x, out_spatial_dim_ = self.conformer(x, in_spatial_dim=in_spatial_dim)
            assert isinstance(out_spatial_dim_, nn.Dim)
            if out_spatial_dim_ != in_spatial_dim:
                out_spatial_dim_.declare_same_as(nn.SpatialDim("downsampled-time"))
            x = self.output(x)
            return x, out_spatial_dim_

    model = Model(out_dim=output_dim + 1)  # +1 for blank
    inputs = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim]))
    logits, out_spatial_dim = model(inputs, in_spatial_dim=time_dim)

    targets = nn.get_extern_data(nn.Data("classes", dim_tags=[nn.batch_dim, targets_time_dim], sparse_dim=output_dim))
    loss = nn.ctc_loss(logits=logits, targets=targets)
    loss.mark_as_loss("ctc")

    decoded, decoded_spatial_dim = nn.ctc_greedy_decode(logits, in_spatial_dim=out_spatial_dim)
    error = nn.edit_distance(a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_time_dim)
    error.mark_as_loss("label", as_error=True, custom_inv_norm_factor=nn.length(targets_time_dim))
    model_py_code_str = nn.get_returnn_config().get_complete_py_code_str(model)

    returnn_train_config_dict = dict(
        use_tensorflow=True,
        # flat_net_construction=True,
        **default_dataset_config,
        batching="random",
        batch_size=20000,
        max_seqs=200,
        max_seq_length={"classes": 75},
        gradient_clip=0,
        # gradient_clip_global_norm = 1.0
        optimizer={"class": "nadam", "epsilon": 1e-8},
        gradient_noise=0.0,
        learning_rate=0.0008,
        learning_rates=[0.0003] * 10 + list(numpy.linspace(0.0003, 0.0008, num=10)),
        learning_rate_control="newbob_multi_epoch",
        # learning_rate_control_error_measure = "dev_score_output"
        learning_rate_control_relative_error_relative_lr=True,
        learning_rate_control_min_num_epochs_per_new_lr=3,
        use_learning_rate_control_always=True,
        newbob_multi_num_epochs=default_train_epoch_split,
        newbob_multi_update_interval=1,
        newbob_learning_rate_decay=0.9,
    )

    returnn_train_config = ReturnnConfig(
        returnn_train_config_dict,
        python_epilog=[
            serialization.Collection(
                [
                    serialization.ExplicitHash("my_model"),
                    serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                    serialization.NonhashedCode(model_py_code_str),
                ]
            )
        ],
        post_config=dict(
            log_batch_size=True,
            tf_log_memory_usage=True,
            tf_session_opts={"gpu_options": {"allow_growth": True}},
            cleanup_old_models=True,
            # debug_add_check_numerics_ops = True
            # debug_add_check_numerics_on_output = True
            # stop_on_nonfinite_train_score = False,
        ),
        sort_config=False,
    )
    returnn_train_job = ReturnnTrainingJob(
        returnn_train_config, log_verbosity=5, num_epochs=100, time_rqmt=80, mem_rqmt=15, cpu_rqmt=4
    )
    tk.register_output("librispeech/ctc-model/learning-rates", returnn_train_job.out_learning_rates)


# Note: I copied i6_experiments.users.zeyer.datasets.librispeech here,
# because I cleaned up some of the things there, and wanted to keep the old setup here functional.


from typing import Optional, Any, Union, Tuple, Dict
from copy import deepcopy

from sisyphus import tk
from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.text.label.sentencepiece.train import TrainSentencePieceJob, SentencePieceType
from returnn.util.basic import NotSpecified
from returnn_common.datasets_old_2022_10.interface import DatasetConfig, VocabConfig
from i6_experiments.common.datasets import librispeech
from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.speed_pert.librosa_09_10_11_kaiser_fast import (
    speed_pert_librosa_09_10_11_kaiser_fast as _default_train_audio_preprocess,
)
from i6_experiments.users.zeyer.datasets.task import Task, MeasureType, RecogOutput, ScoreResult
from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel


librispeech_ogg_zip_dict = librispeech.get_ogg_zip_dict()

# Get Bliss corpus. Same audio format as in ogg_zip, so already there anyway due to how we created the ogg_zip.
# WARNING: Do not use these directly... It will keep another ogg copy of the audio...
_bliss_corpus_dict = librispeech.get_bliss_corpus_dict(audio_format="ogg")  # TODO bad deps...
_bliss_train_corpus = _bliss_corpus_dict["train-other-960"]  # TODO bad deps...

_train_corpus_text = CorpusToTxtJob(_bliss_train_corpus, gzip=False).out_txt  # TODO avoid...

# https://github.com/google/sentencepiece/blob/master/doc/options.md
_spm_train_job = TrainSentencePieceJob(
    training_text=_train_corpus_text,
    vocab_size=2000,
    model_type=SentencePieceType.UNIGRAM,
    additional_options={
        "split_digits": True,
        "unk_id": 2,  # default is 0
        "bos_id": 1,  # default is 1
        "eos_id": 0,  # default is 2
    },
)
spm_2k = _spm_train_job.out_model  # TODO bad deps...

# common
bpe10k = Bpe(
    dim=10_025,
    eos_idx=0,
    bos_idx=0,
    codes=generic_job_output("i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.codes"),
    vocab=generic_job_output("i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab"),
    # unknown_label="<unk>",
    unknown_label=None,
)


_Parts = ["train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other", "test-clean", "test-other"]


# https://github.com/rwth-i6/returnn-experiments/blob/master/2020-librispeech-data-prepare/returnn.config
def _get_dataset(key: str, *, subset=None, train_partition_epoch=None, training: bool = False, targets, audio):
    files = []
    parts = [part for part in _Parts if part.startswith(key)]
    assert parts, f"invalid key {key!r}"
    for part in parts:
        files += [librispeech_ogg_zip_dict[part]]
    d = {
        "class": "OggZipDataset",
        "path": files,
        "use_cache_manager": True,
        "targets": targets,
        "audio": audio,
    }
    if key.startswith("train") and training:
        d["partition_epoch"] = train_partition_epoch
        if key == "train":
            d["epoch_wise_filter"] = {
                (1, 5): {"max_mean_len": 200},
                (6, 10): {"max_mean_len": 500},
            }
        # if audio is not None:
        #   d["audio"]["random_permute"] = True  # play around. note that this can be slow
        d["seq_ordering"] = "laplace:.1000"
    else:
        d["fixed_random_seed"] = 1
        d["seq_ordering"] = "sorted_reverse"
    if subset:
        d["fixed_random_subset"] = subset  # faster
    return d


# _default_audio_opts_no_stats = dict(features="mfcc", num_feature_filters=40, window_len=0.025, step_len=0.010)
_default_audio_opts_log_mel_fbank_no_stats = dict(
    features="log_mel_filterbank", num_feature_filters=80, window_len=0.025, step_len=0.010
)
# _returnn_train_full_no_stats_dict = _get_dataset("train", audio=_default_audio_opts_no_stats)
# _audio_stats_job = ExtractDatasetMeanStddevJob(ReturnnConfig(config={"train": _returnn_train_full_no_stats_dict}))
# default_audio_opts = {
#  **_default_audio_opts_no_stats,
#  "norm_mean": _audio_stats_job.out_mean_file, "norm_std_dev": _audio_stats_job.out_std_dev_file}
default_audio_opts = _default_audio_opts_log_mel_fbank_no_stats

# https://returnn.readthedocs.io/en/latest/api/datasets.util.vocabulary.html#returnn.datasets.util.vocabulary.SentencePieces
default_targets_opts = {
    "class": "SentencePieces",
    "model_file": spm_2k,
    # If your model (e.g. enc-dec) needs EOS, add "add_eos".
}
default_targets_train_opts = default_targets_opts.copy()
default_targets_train_opts.update(
    {
        "enable_sampling": True,  # might be played around with, along with nbest_size, alpha.
    }
)

default_train_epoch_split = 20

default_dataset_config = {
    "train": _get_dataset(
        "train",
        training=True,
        train_partition_epoch=default_train_epoch_split,
        audio=default_audio_opts,
        targets=default_targets_train_opts,
    ),
    "dev": _get_dataset("dev", subset=3000, audio=default_audio_opts, targets=default_targets_opts),
    "eval_datasets": {
        "devtrain": _get_dataset("train", subset=2000, audio=default_audio_opts, targets=default_targets_opts),
    },
}
