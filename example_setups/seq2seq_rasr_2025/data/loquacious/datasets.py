import functools
from typing import List, Literal, get_args
from .phoneme import get_phoneme_target_hdf_file
from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.text.processing import HeadJob
from i6_core.corpus.filter import FilterCorpusBySegmentsJob
from i6_experiments.common.datasets.loquacious.corpus import get_bliss_corpus_dict as get_bliss_corpus_dict_
from i6_experiments.common.datasets.loquacious.corpus import get_ogg_zip_dict
from sisyphus import tk

from ...model_pipelines.common.corpus import ScorableCorpus
from ..base import HdfDataConfig, MetaOggZipDataConfig, MetaOggZipHdfTargetDataConfig, OggZipDataConfig
from .bpe import get_default_bpe_target_config
from ...tools import returnn_root, returnn_python_exe

TrainSet = Literal["train.medium", "train.small"]

EvalSet = Literal[
    "dev.all",
    "dev.short",
    "dev.commonvoice",
    "dev.librispeech",
    "dev.voxpopuli",
    "dev.yodas",
    "test.all",
    "test.commonvoice",
    "test.librispeech",
    "test.voxpopuli",
    "test.yodas",
]

EVAL_SETS: List[EvalSet] = get_args(EvalSet)  # type: ignore


def _get_dev_short_segments():
    dev = get_bliss_corpus_dict_()["dev.all"]
    dev_all_segments = SegmentCorpusJob(dev, 1).out_single_segment_files[1]

    def shuffle_and_head(segment_file: tk.Path, num_lines: int):
        # only shuffle, this is deterministic
        shuffle_segment_file_job = ShuffleAndSplitSegmentsJob(
            segment_file=segment_file, split={"shuffle": 1.0}, shuffle=True
        )
        segment_file = shuffle_segment_file_job.out_segments["shuffle"]
        return HeadJob(segment_file, num_lines=num_lines).out

    dev_all_subset = shuffle_and_head(dev_all_segments, 3000)
    return dev_all_subset


@functools.lru_cache
def get_bliss_corpus_dict(**kwargs) -> dict:
    bliss_corpus_dict = get_bliss_corpus_dict_(**kwargs)
    bliss_corpus_dict["dev.short"] = FilterCorpusBySegmentsJob(
        bliss_corpus_dict["dev.all"], segment_file=_get_dev_short_segments()
    ).out_corpus

    return bliss_corpus_dict


def get_small_bpe_train_data(bpe_size: int, speed_perturb: bool = True) -> MetaOggZipDataConfig:
    return MetaOggZipDataConfig(
        oggzip_files=[
            get_ogg_zip_dict(returnn_root=returnn_root, returnn_python_exe=returnn_python_exe)["train.small"]
        ],
        speed_perturbation=speed_perturb,
        partition_epoch=5,
        seq_ordering="laplace:.1000",
        target_config=get_default_bpe_target_config(bpe_size, corpus_key="train.small"),
    )


def get_small_phoneme_train_data(speed_perturb: bool = True) -> MetaOggZipHdfTargetDataConfig:
    return MetaOggZipHdfTargetDataConfig(
        oggzip_config=OggZipDataConfig(
            oggzip_files=[
                get_ogg_zip_dict(returnn_root=returnn_root, returnn_python_exe=returnn_python_exe)["train.small"]
            ],
            speed_perturbation=speed_perturb,
            partition_epoch=5,
            seq_ordering="laplace:.1000",
            target_config=None,
        ),
        oggzip_target_name=None,
        hdf_config=HdfDataConfig(files=[get_phoneme_target_hdf_file("train.small")]),
        hdf_target_name="classes",
    )


def get_medium_bpe_train_data(bpe_size: int) -> MetaOggZipDataConfig:
    return MetaOggZipDataConfig(
        oggzip_files=[
            get_ogg_zip_dict(returnn_root=returnn_root, returnn_python_exe=returnn_python_exe)["train.medium"]
        ],
        speed_perturbation=True,
        partition_epoch=50,
        seq_ordering="laplace:.1000",
        target_config=get_default_bpe_target_config(bpe_size, corpus_key="train.medium"),
    )


def get_medium_phoneme_train_data() -> MetaOggZipHdfTargetDataConfig:
    return MetaOggZipHdfTargetDataConfig(
        oggzip_config=OggZipDataConfig(
            oggzip_files=[
                get_ogg_zip_dict(returnn_root=returnn_root, returnn_python_exe=returnn_python_exe)["train.medium"]
            ],
            speed_perturbation=True,
            partition_epoch=50,
            seq_ordering="laplace:.1000",
            target_config=None,
        ),
        oggzip_target_name=None,
        hdf_config=HdfDataConfig(files=[get_phoneme_target_hdf_file("train.medium")]),
        hdf_target_name="classes",
    )


def get_medium_bpe_phoneme_train_data(bpe_size: int) -> MetaOggZipHdfTargetDataConfig:
    return MetaOggZipHdfTargetDataConfig(
        oggzip_config=OggZipDataConfig(
            oggzip_files=[
                get_ogg_zip_dict(returnn_root=returnn_root, returnn_python_exe=returnn_python_exe)["train.medium"]
            ],
            speed_perturbation=True,
            partition_epoch=50,
            seq_ordering="laplace:.1000",
            target_config=get_default_bpe_target_config(bpe_size, corpus_key="train.medium"),
        ),
        oggzip_target_name="bpe",
        hdf_config=HdfDataConfig(files=[get_phoneme_target_hdf_file("train.medium")]),
        hdf_target_name="phoneme",
    )


def get_small_bpe_cv_data(bpe_size: int) -> MetaOggZipDataConfig:
    return MetaOggZipDataConfig(
        oggzip_files=[get_ogg_zip_dict(returnn_root=returnn_root, returnn_python_exe=returnn_python_exe)["dev.all"]],
        speed_perturbation=False,
        partition_epoch=1,
        seq_ordering="sorted",
        target_config=get_default_bpe_target_config(bpe_size, corpus_key="train.small"),
        segment_file=_get_dev_short_segments(),
    )


def get_medium_bpe_cv_data(bpe_size: int) -> MetaOggZipDataConfig:
    return MetaOggZipDataConfig(
        oggzip_files=[get_ogg_zip_dict(returnn_root=returnn_root, returnn_python_exe=returnn_python_exe)["dev.all"]],
        speed_perturbation=False,
        partition_epoch=1,
        seq_ordering="sorted",
        target_config=get_default_bpe_target_config(bpe_size, corpus_key="train.medium"),
        segment_file=_get_dev_short_segments(),
    )


def get_phoneme_cv_data() -> MetaOggZipHdfTargetDataConfig:
    return MetaOggZipHdfTargetDataConfig(
        oggzip_config=OggZipDataConfig(
            oggzip_files=[
                get_ogg_zip_dict(returnn_root=returnn_root, returnn_python_exe=returnn_python_exe)["dev.all"]
            ],
            speed_perturbation=False,
            partition_epoch=1,
            seq_ordering="sorted",
            target_config=None,
            segment_file=_get_dev_short_segments(),
        ),
        oggzip_target_name=None,
        hdf_config=HdfDataConfig(files=[get_phoneme_target_hdf_file("dev.all")]),
        hdf_target_name="classes",
    )


def get_small_bpe_phoneme_cv_data(bpe_size: int) -> MetaOggZipHdfTargetDataConfig:
    return MetaOggZipHdfTargetDataConfig(
        oggzip_config=OggZipDataConfig(
            oggzip_files=[
                get_ogg_zip_dict(returnn_root=returnn_root, returnn_python_exe=returnn_python_exe)["dev.all"]
            ],
            speed_perturbation=False,
            partition_epoch=1,
            seq_ordering="sorted",
            target_config=get_default_bpe_target_config(bpe_size, corpus_key="train.small"),
            segment_file=_get_dev_short_segments(),
        ),
        oggzip_target_name="bpe",
        hdf_config=HdfDataConfig(files=[get_phoneme_target_hdf_file("dev.all")]),
        hdf_target_name="phoneme",
    )


def get_medium_bpe_phoneme_cv_data(bpe_size: int) -> MetaOggZipHdfTargetDataConfig:
    return MetaOggZipHdfTargetDataConfig(
        oggzip_config=OggZipDataConfig(
            oggzip_files=[
                get_ogg_zip_dict(returnn_root=returnn_root, returnn_python_exe=returnn_python_exe)["dev.all"]
            ],
            speed_perturbation=False,
            partition_epoch=1,
            seq_ordering="sorted",
            target_config=get_default_bpe_target_config(bpe_size, corpus_key="train.medium"),
            segment_file=_get_dev_short_segments(),
        ),
        oggzip_target_name="bpe",
        hdf_config=HdfDataConfig(files=[get_phoneme_target_hdf_file("dev.all")]),
        hdf_target_name="phoneme",
    )


def get_prior_data(train_corpus_key: TrainSet) -> MetaOggZipDataConfig:
    # use 10% of the training corpus to estimate the prior
    train_corpus_file = get_bliss_corpus_dict()[train_corpus_key]
    segment_file = SegmentCorpusJob(train_corpus_file, 10).out_single_segment_files[1]

    return MetaOggZipDataConfig(
        oggzip_files=[
            get_ogg_zip_dict(returnn_root=returnn_root, returnn_python_exe=returnn_python_exe)[train_corpus_key]
        ],
        speed_perturbation=False,
        partition_epoch=1,
        seq_ordering="sorted",
        segment_file=segment_file,
    )


def get_default_recog_data(corpus_name: EvalSet) -> OggZipDataConfig:
    return OggZipDataConfig.from_bliss(
        bliss_corpus_files=[get_bliss_corpus_dict()[corpus_name]],
        speed_perturbation=False,
        ogg_segments=1,
        partition_epoch=1,
        seq_ordering="sorted",
    )


def get_default_score_corpus(corpus_name: EvalSet) -> ScorableCorpus:
    return ScorableCorpus(
        corpus_name=corpus_name,
        bliss_corpus_file=get_bliss_corpus_dict()[corpus_name],
    )
