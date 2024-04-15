"""
Tedlium2 dataset
"""

from __future__ import annotations
from typing import Optional, Any, Union, Tuple, Dict
from copy import deepcopy

from sisyphus import tk
from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.text.label.sentencepiece.train import (
    TrainSentencePieceJob,
    SentencePieceType,
)
from returnn.util.basic import NotSpecified
from returnn_common.datasets_old_2022_10.interface import DatasetConfig, VocabConfig
from i6_experiments.common.datasets.tedlium2 import corpus as tedlium2
from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.speed_pert.librosa_09_10_11_kaiser_fast import (
    speed_pert_librosa_09_10_11_kaiser_fast as _default_train_audio_preprocess,
)
from .task import Task, MeasureType, RecogOutput, ScoreResult
from .utils.bpe import Bpe


tedlium2_ogg_zip_dict = tedlium2.get_ogg_zip_dict()

# Get Bliss corpus. Same audio format as in ogg_zip, so already there anyway due to how we created the ogg_zip.
bliss_corpus_dict = tedlium2.get_bliss_corpus_dict(audio_format="ogg")
bliss_train_corpus = bliss_corpus_dict["train"]

train_corpus_text = CorpusToTxtJob(bliss_train_corpus, gzip=False).out_txt

# https://github.com/google/sentencepiece/blob/master/doc/options.md
# spm_train_job = TrainSentencePieceJob(
#     training_text=train_corpus_text,
#     vocab_size=2000,
#     model_type=SentencePieceType.UNIGRAM,
#     additional_options={
#         "split_digits": True,
#         "unk_id": 2,  # default is 0
#         "bos_id": 1,  # default is 1
#         "eos_id": 0,  # default is 2
#     },
# )
# spm_2k = spm_train_job.out_model

# common
bpe1k = Bpe(
    dim=1_057,
    eos_idx=0,
    bos_idx=0,
    codes=generic_job_output(
        "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.codes"
    ),
    vocab=generic_job_output(
        "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab"
    ),
    # unknown_label="<unk>",
    unknown_label=None,
)


_Parts = ["train", "dev", "test"]


# https://github.com/rwth-i6/returnn-experiments/blob/master/2020-librispeech-data-prepare/returnn.config
def _get_dataset(
    key: str,
    *,
    subset=None,
    train_partition_epoch=None,
    training: bool = False,
    targets,
    audio,
):
    files = []
    parts = [part for part in _Parts if part.startswith(key)]
    assert parts, f"invalid key {key!r}"
    for part in parts:
        files += [tedlium2_ogg_zip_dict[part]]
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
    features="log_mel_filterbank",
    num_feature_filters=80,
    window_len=0.025,
    step_len=0.010,
)
# _returnn_train_full_no_stats_dict = _get_dataset("train", audio=_default_audio_opts_no_stats)
# _audio_stats_job = ExtractDatasetMeanStddevJob(ReturnnConfig(config={"train": _returnn_train_full_no_stats_dict}))
# default_audio_opts = {
#  **_default_audio_opts_no_stats,
#  "norm_mean": _audio_stats_job.out_mean_file, "norm_std_dev": _audio_stats_job.out_std_dev_file}
default_audio_opts = _default_audio_opts_log_mel_fbank_no_stats

# https://returnn.readthedocs.io/en/latest/api/datasets.util.vocabulary.html#returnn.datasets.util.vocabulary.SentencePieces
# default_targets_opts = {
#     "class": "SentencePieces",
#     "model_file": spm_2k,
#     # If your model (e.g. enc-dec) needs EOS, add "add_eos".
# }
# default_targets_train_opts = default_targets_opts.copy()
# default_targets_train_opts.update(
#     {
#         "enable_sampling": True,  # might be played around with, along with nbest_size, alpha.
#     }
# )

# default_train_epoch_split = 4
default_train_epoch_split = 20 # TODO
#
# default_dataset_config = {
#     "train": _get_dataset(
#         "train",
#         training=True,
#         train_partition_epoch=default_train_epoch_split,
#         audio=default_audio_opts,
#         targets=default_targets_train_opts,
#     ),
#     "dev": _get_dataset("dev", subset=3000, audio=default_audio_opts, targets=default_targets_opts),
#     "eval_datasets": {
#         "devtrain": _get_dataset("train", subset=2000, audio=default_audio_opts, targets=default_targets_opts),
#     },
# }

_default_train_epoch_wise_filter = {
    (1, 5): {"max_mean_len": 1000},  # better? # TODO
    # older settings:
    # (1, 5): {"max_mean_len": 200},
    # (6, 10): {"max_mean_len": 500},
}


class Tedlium2OggZip(DatasetConfig):
    """
    Tedlium2 dataset in OggZip format.
    """

    def __init__(
        self,
        *,
        audio: Optional[Dict[str, Any]] = None,
        audio_dim: Optional[int] = None,
        vocab: Optional[VocabConfig] = None,
        with_eos_postfix: bool = False,
        main_key: Optional[str] = None,
        train_epoch_split: int = default_train_epoch_split,
        train_sort_laplace_num_seqs: int = 1000,
        train_epoch_wise_filter: Optional[
            Dict[Tuple[int, int], Dict[str, Any]]
        ] = NotSpecified,
        train_audio_preprocess: Optional[Any] = NotSpecified,
        train_audio_random_permute: Union[bool, Dict[str, Any]] = False,
        eval_subset: Optional[int] = 3000,
    ):
        """
        :param with_eos_postfix: For RETURNN train/dev/eval datasets, mostly relevant for training.
            For recognition, our score function uses the Bliss corpus directly, so this has no influence.
        """
        super(Tedlium2OggZip, self).__init__()
        self.audio = audio
        self.audio_dim = audio_dim
        self.vocab = vocab
        self.with_eos_postfix = with_eos_postfix
        self.main_key = main_key
        self.train_epoch_split = train_epoch_split
        self.train_sort_laplace_num_seqs = train_sort_laplace_num_seqs
        if train_epoch_wise_filter is NotSpecified:
            train_epoch_wise_filter = deepcopy(_default_train_epoch_wise_filter)
        if train_audio_preprocess is NotSpecified:
            if train_audio_random_permute:
                train_audio_preprocess = None
            else:
                train_audio_preprocess = _default_train_audio_preprocess
        self.train_audio_preprocess = train_audio_preprocess
        # By default, audio random_permute is False
        # because we use the specific speed perturbation variant above instead.
        # A common setting otherwise is {"rnd_zoom_order": 0}.
        self.train_audio_random_permute = train_audio_random_permute
        self.train_epoch_wise_filter = train_epoch_wise_filter
        self.eval_subset = eval_subset

    def get_extern_data(self) -> Dict[str, Dict[str]]:
        """
        Get extern data
        """
        from returnn.tensor import Dim, batch_dim

        opts = {}

        if self.audio is not None:
            assert self.audio_dim is not None
            time_dim = Dim(None, name="time", kind=Dim.Types.Spatial)
            feature_dim = Dim(self.audio_dim, name="audio", kind=Dim.Types.Feature)
            opts["data"] = {"dim_tags": [batch_dim, time_dim, feature_dim]}

        if self.vocab is not None:
            out_spatial_dim = Dim(None, name="out-spatial", kind=Dim.Types.Spatial)
            classes_dim = Dim(
                self.vocab.get_num_classes(), name="vocab", kind=Dim.Types.Spatial
            )
            opts["classes"] = {
                "dim_tags": [batch_dim, out_spatial_dim],
                "sparse_dim": classes_dim,
                "vocab": self.vocab.get_opts(),
            }

        return opts

    def get_train_dataset(self) -> Dict[str]:
        return self.get_dataset("train", training=True)

    def get_eval_datasets(self) -> Dict[str, Dict[str]]:
        return {
            "dev": self.get_dataset("dev", subset=self.eval_subset),
            "devtrain": self.get_dataset("train", subset=self.eval_subset),
        }

    def get_main_name(self) -> str:
        return self.main_key

    def get_main_dataset(self) -> Dict[str, Any]:
        return self.get_dataset(self.main_key)

    def get_dataset(
        self, key: str, *, training: bool = False, subset: Optional[int] = None
    ) -> Dict[str, Any]:
        files = []
        parts = [part for part in _Parts if part.startswith(key)]
        assert parts, f"invalid key {key!r}"
        for part in parts:
            files += [tedlium2_ogg_zip_dict[part]]
        d = {
            "class": "OggZipDataset",
            "path": files,
            "use_cache_manager": True,
        }
        if self.audio is not None:
            d["audio"] = self.audio.copy()
        if self.vocab is not None:
            d["targets"] = self.vocab.get_opts().copy()
            assert "seq_postfix" not in d["targets"], d  # we are handling this here
            if self.with_eos_postfix:
                eos_id = self.vocab.get_eos_idx()
                assert (
                    eos_id is not None
                ), f"{self}: vocab {self.vocab} does not define EOS"
                d["targets"]["seq_postfix"] = [eos_id]
        if training:
            d["partition_epoch"] = self.train_epoch_split
            if self.train_epoch_wise_filter is not None:
                d["epoch_wise_filter"] = self.train_epoch_wise_filter
            if self.train_audio_preprocess is not None:
                assert self.audio is not None, "train_audio_preprocess needs audio"
                d["audio"]["pre_process"] = self.train_audio_preprocess
            if self.train_audio_random_permute:
                assert self.audio is not None, "train_audio_random_permute needs audio"
                d["audio"]["random_permute"] = self.train_audio_random_permute
            d["seq_ordering"] = f"laplace:.{self.train_sort_laplace_num_seqs}"
        else:
            d["fixed_random_seed"] = 1
            d["seq_ordering"] = "sorted_reverse"
        if subset:
            d["fixed_random_subset"] = subset  # faster
        return d


class Tedlium2MetaDataset(DatasetConfig):
    """
    Tedlium2 dataset Meta Dataset with Ogg Zip.
    """

    def __init__(
        self,
        *,
        ogg_zip_opts: Dict[str, Any] = None,
        additional_dataset=None,
    ):
        """
        :param ogg_zip_opts: Options for the oggzip dataset.
        """
        super(Tedlium2MetaDataset, self).__init__()
        self.oggzip_dataset = Tedlium2OggZip(**ogg_zip_opts)
        self.additional_dataset = additional_dataset

    def get_extern_data(self) -> Dict[str, Dict[str]]:
        """
        Get extern data
        """

        from returnn.tensor import Dim, batch_dim

        opts = {}

        audio_dim = self.oggzip_dataset.audio_dim
        vocab = self.oggzip_dataset.vocab

        time_dim = Dim(None, name="time", kind=Dim.Types.Spatial)
        out_spatial_dim = Dim(None, name="out-spatial", kind=Dim.Types.Spatial)

        if self.oggzip_dataset.audio is not None:
            assert audio_dim is not None
            feature_dim = Dim(audio_dim, name="audio", kind=Dim.Types.Feature)
            opts["audio_features"] = {"dim_tags": [batch_dim, time_dim, feature_dim]}

        if self.oggzip_dataset.vocab is not None:
            classes_dim = Dim(
                vocab.get_num_classes(), name="vocab", kind=Dim.Types.Spatial
            )
            opts["bpe_labels"] = {
                "dim_tags": [batch_dim, out_spatial_dim],
                "sparse_dim": classes_dim,
                "vocab": vocab.get_opts(),
            }

            out_spatial_dim_1 = Dim(None, name="out-spatial-1", kind=Dim.Types.Spatial)
            classes_dim_1 = Dim(
                vocab.get_num_classes(), name="vocab-1", kind=Dim.Types.Feature
            )
            opts["hdf_features"] = {
                "dim_tags": [batch_dim, out_spatial_dim_1],
                "sparse_dim": classes_dim_1,
                "vocab": vocab.get_opts(),
            }

        return opts

    def get_train_dataset(self) -> Dict[str]:
        return self.oggzip_dataset.get_dataset("train", training=True)

    def get_eval_datasets(self) -> Dict[str, Dict[str]]:
        return {
            "dev": self.oggzip_dataset.get_dataset("dev", subset=self.eval_subset),
            "devtrain": self.oggzip_dataset.get_dataset(
                "train", subset=self.eval_subset
            ),
        }

    def get_main_name(self) -> str:
        return self.oggzip_dataset.main_key

    def get_main_dataset(self) -> Dict[str, Any]:
        return self.get_dataset(self.oggzip_dataset.main_key)

    def get_default_input(self) -> Optional[str]:
        return "audio_features"

    def get_default_target(self) -> Optional[str]:
        return "hdf_features"

    def get_dataset(
        self, key: str, *, training: bool = False, subset: Optional[int] = None
    ) -> Dict[str, Any]:

        ogg_dict = self.oggzip_dataset.get_dataset(
            key, training=training, subset=subset
        )

        d = {
            "class": "MetaDataset",
            "data_map": {
                "audio_features": ("zip_dataset", "data"),
                "bpe_labels": ("zip_dataset", "classes"),
                "hdf_features": ("hdf_dataset", "data"),
            },
            "datasets": {
                "zip_dataset": ogg_dict,
                "hdf_dataset": self.additional_dataset.get_main_dataset(),
            },
            "seq_order_control_dataset": "zip_dataset",
        }

        return d


_raw_audio_opts = dict(
    features="raw",
    sample_rate=16_000,
    peak_normalization=True,
    preemphasis=None,
)


def get_tedlium2_task_spm2k() -> Task:
    """
    Tedlium2
    """
    # TODO ...


def get_tedlium2_task_raw(*, vocab: VocabConfig, **dataset_train_opts) -> Task:
    """
    Tedlium2
    """
    dataset_common_opts = dict(audio=_raw_audio_opts.copy(), audio_dim=1, vocab=vocab)
    # We expect that all kwargs are only relevant for the training, thus we only pass them here.
    train_dataset = Tedlium2OggZip(**dataset_common_opts, **dataset_train_opts)
    dev_dataset = Tedlium2OggZip(**dataset_common_opts, main_key="dev")
    eval_datasets = {
        "dev": dev_dataset,
        "test": Tedlium2OggZip(**dataset_common_opts, main_key="test"),
    }

    return Task(
        name="ted2_bpe1k",
        train_dataset=train_dataset,
        train_epoch_split=train_dataset.train_epoch_split,
        dev_dataset=dev_dataset,
        eval_datasets=eval_datasets,
        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name="dev",
        score_recog_output_func=score,
        recog_post_proc_funcs=[_bpe_to_words],
    )


def get_tedlium2_task_bpe1k_raw(**dataset_train_opts) -> Task:
    return get_tedlium2_task_raw(vocab=bpe1k, **dataset_train_opts)


def _bpe_to_words(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    from i6_core.returnn.search import SearchBPEtoWordsJob

    words = SearchBPEtoWordsJob(bpe.output, output_gzip=True).out_word_search_results
    return RecogOutput(output=words)


def _score(*, hyp_words: tk.Path, corpus_name: str) -> ScoreResult:
    # We use sclite now.
    # Could also use ReturnnComputeWERJob.

    from i6_core.returnn.search import SearchWordsToCTMJob
    from i6_core.corpus.convert import CorpusToStmJob
    from i6_core.recognition.scoring import ScliteJob

    recognition_bliss_corpus = bliss_corpus_dict[corpus_name]

    search_ctm = SearchWordsToCTMJob(
        recog_words_file=hyp_words,
        bliss_corpus=recognition_bliss_corpus,
    ).out_ctm_file

    stm_file = CorpusToStmJob(bliss_corpus=recognition_bliss_corpus).out_stm_path

    score_job = ScliteJob(
        ref=stm_file,
        hyp=search_ctm,
        sctk_binary_path=tools_paths.get_sctk_binary_path(),
        precision_ndigit=2,
    )

    return ScoreResult(
        dataset_name=corpus_name,
        main_measure_value=score_job.out_wer,
        report=score_job.out_report_dir,
    )


def score(dataset: DatasetConfig, recog_output: RecogOutput) -> ScoreResult:
    """score"""
    return _score(hyp_words=recog_output.output, corpus_name=dataset.get_main_name())
