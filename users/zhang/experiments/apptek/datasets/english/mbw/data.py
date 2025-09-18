import copy
from dataclasses import dataclass
from functools import cache
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from sisyphus import Job, Path, Task, tk
from sisyphus.delayed_ops import DelayedFormat

from apptek_asr.users.mgunz.postprocessing.oggzip import BlissToSplitOggZipJob
from apptek_asr.users.mgunz.postprocessing.seq_concat_speed_pert import DataPostprocessor
from i6_core.corpus import FilterCorpusBySegmentDurationJob, SegmentCorpusJob
from i6_core.returnn.config import CodeWrapper
from i6_core.util import uopen

from .corpus_lex import Corpora
from .gigaspeech import GigaspeechCorpusToOggZipJob
from .large_scale_asr import LargeScaleASRCorpusToOggZipJob
from .libriheavy import LibriheavyCorpusToOggZipJob


@dataclass(frozen=True)
class EnglishData:
    train: DatasetConfig
    cv: DatasetConfig

    dev: Dict[str, DatasetConfig]
    test: Dict[str, DatasetConfig]


_default_train_epoch_wise_filter = {
    (1, 5): {"max_mean_len": 1000},  # better?
    # older settings:
    # (1, 5): {"max_mean_len": 200},
    # (6, 10): {"max_mean_len": 500},
}


class CreateTrainDataset:
    def __init__(
        self,
        *,
        epoch_wise_filter: Optional[Any] = None,
        num_processes: Optional[int] = None,
        sample_rate: int,
        vocab_opts: Dict[str, Any],
    ):
        assert num_processes is None or num_processes > 0
        assert sample_rate > 0

        self.epoch_wise_filter = epoch_wise_filter
        self.num_processes = num_processes
        self.sample_rate = sample_rate
        self.vocab_opts = vocab_opts

    def __call__(self, files_subepoch: List[Path]) -> Dict[str, Any]:
        from returnn.util.file_cache import CachedFile

        from i6_core.util import instanciate_delayed

        dataset = {
            "class": "OggZipDataset",
            "audio": {
                "features": "raw",
                "peak_normalization": True,
                "preemphasis": None,
                "pre_process": None,
                "sample_rate": self.sample_rate,
            },
            "epoch_wise_filter": self.epoch_wise_filter,
            "path": [CachedFile(f) for f in files_subepoch],
            "seq_ordering": "random",
            "targets": instanciate_delayed(self.vocab_opts.copy()),
            "use_cache_manager": False,
        }
        dataset = {
            "class": "PostprocessingDataset",
            "dataset": dataset,
            "map_seq_stream": DataPostprocessor(
                speed_pert_kwargs={
                    "data_key": "data",
                    "factors": [0.7, 0.8, 0.9, 1.0, 1.1],
                    "sample_rate": self.sample_rate,
                },
                concat_seqs_kwargs={
                    "classes_key": "classes",
                    "classes_sep": 3,  # <sep> index in spm_sep10k vocab
                    "data_key": "data",
                    "max_num_seqs": 3,
                    "max_seq_len": 45 * self.sample_rate,
                    "num_seqs_dist": ("geometric", 0.3),
                    "seq_tag_sep": ":",
                },
                laplace_kwargs=None,
            ),
            "seq_ordering": "default",
        }
        if self.num_processes is not None:
            dataset = {
                "buffer_size": 100,
                "class": "MultiProcDataset",
                "dataset": dataset,
                "num_workers": self.num_processes,
                "sharding_method": "dedicated",
            }
        return dataset


class EnglishTrainDataset(DatasetConfig):
    def __init__(
        self,
        train_oggzip: List[Path],
        cv_oggzip: List[Path],
        spm: SentencePieceModel,
        main_key: str,
        train_partition_epoch: int,
        train_num_processes: Optional[int] = None,
        train_concat_seqs: bool = True,
        train_sort_laplace_num_seqs: Optional[int] = None,
        train_audio_preprocess: Optional[Any] = None,
        train_epoch_wise_filter: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None,
        filter_invalid_ctc_seq_for_frame_rate: Optional[int] = None,
    ) -> None:
        super().__init__()

        assert train_partition_epoch > 0
        assert train_sort_laplace_num_seqs is None or train_sort_laplace_num_seqs > 0
        assert train_num_processes is None or train_num_processes > 0

        assert isinstance(train_oggzip, list) and isinstance(cv_oggzip, list)
        self.oggzips = {"train": train_oggzip, "cv": cv_oggzip}
        self.spm = spm
        self.main_key = main_key
        self.train_audio_preprocess = train_audio_preprocess
        self.train_partition_epoch = train_partition_epoch
        self.train_concat_seqs = train_concat_seqs
        self.train_num_processes = train_num_processes
        self.train_sort_laplace_num_seqs = train_sort_laplace_num_seqs
        self.train_epoch_wise_filter = train_epoch_wise_filter or copy.deepcopy(_default_train_epoch_wise_filter)
        self.filter_invalid_ctc_seq_for_frame_rate = filter_invalid_ctc_seq_for_frame_rate

    def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
        from returnn.tensor import Dim, batch_dim

        classes_spatial_dim = Dim(None, name="out-spatial", kind=Dim.Types.Spatial)
        classes_dim = Dim(self.spm.get_num_classes(), name="vocab", kind=Dim.Types.Spatial)

        time_dim = Dim(None, name="time", kind=Dim.Types.Spatial)
        feature_dim = Dim(1, name="audio", kind=Dim.Types.Feature)

        return {
            "data": {
                "dim_tags": [batch_dim, time_dim, feature_dim],
            },
            "classes": {
                "dim_tags": [batch_dim, classes_spatial_dim],
                "sparse_dim": classes_dim,
                "vocab": self.spm.get_opts(),
            },
        }

    def get_train_dataset(self) -> Dict[str, Any]:
        return self.get_dataset("train", training=True)

    def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
        return {key: self.get_dataset(key) for key in ["cv"]}

    def get_main_dataset(self) -> Dict[str, Any]:
        return self.get_dataset(self.get_main_name())

    def get_main_name(self) -> str:
        return self.main_key

    def get_dataset(self, key: str, training=False) -> Dict[str, Any]:
        spm_opts = self.spm.get_opts()

        if not training:
            from returnn.util.file_cache import CachedFile

            obj = {
                "class": "OggZipDataset",
                "audio": {
                    "features": "raw",
                    "peak_normalization": True,
                    "preemphasis": None,
                    "sample_rate": 16_000,
                },
                "fixed_random_seed": 1,
                "path": [CachedFile(file.get_path()) for file in self.oggzips[key]],
                "partition_epoch": 1,
                "seq_ordering": "sorted_reverse",
                "targets": spm_opts,
                "use_cache_manager": False,
            }
            return obj

        assert self.train_sort_laplace_num_seqs is None, (
            "only random sorting supported, please use bucket batching or LaplaceOrdering in PP Dataset!"
        )
        assert self.train_partition_epoch <= len(self.oggzips[key]), (
            f"too few oggzips for DFD ({len(self.oggzips[key])} ogg zips for part ep: {self.train_partition_epoch})"
        )

        distrib_files_dataset = {
            "buffer_size": 1_000,
            "class": "DistributeFilesDataset",
            "distrib_shard_files": True,
            "files": self.oggzips[key],
            "partition_epoch": self.train_partition_epoch,
            "get_sub_epoch_dataset": CreateTrainDataset(
                epoch_wise_filter=self.train_epoch_wise_filter,
                num_processes=self.train_num_processes,
                sample_rate=16_000,
                vocab_opts=spm_opts,
            ),
            "seq_ordering": "random",
        }
        if self.filter_invalid_ctc_seq_for_frame_rate is not None:
            raise NotImplementedError
            # valid_seqs_job = FilterInvalidCtcSeqsJob(
            #     self.oggzips[key], self.filter_invalid_ctc_seq_for_frame_rate, self.spm.model_file
            # )
            # obj["seq_list_filter_file"] = valid_seqs_job.out_valid_seqs
        return distrib_files_dataset


class EvalOggZip(DatasetConfig):
    def __init__(self, oggzip_data: Path, main_key: str, spm: SentencePieceModel, gpu_mem: Optional[int] = None):
        super().__init__()

        assert isinstance(oggzip_data, Path)
        self.main_key = main_key
        self.oggzip = oggzip_data
        self.spm = spm

        if gpu_mem is not None:
            self.gpu_mem = gpu_mem

    def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
        from returnn.tensor import Dim, batch_dim

        classes_spatial_dim = Dim(None, name="out-spatial", kind=Dim.Types.Spatial)
        classes_dim = Dim(self.spm.get_num_classes(), name="vocab", kind=Dim.Types.Spatial)

        time_dim = Dim(None, name="time", kind=Dim.Types.Spatial)
        feature_dim = Dim(1, name="audio", kind=Dim.Types.Feature)

        return {
            "data": {
                "dim_tags": [batch_dim, time_dim, feature_dim],
            },
            "classes": {
                "dim_tags": [batch_dim, classes_spatial_dim],
                "sparse_dim": classes_dim,
                "vocab": self.spm.get_opts(),
            },
        }

    def get_train_dataset(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
        return {self.main_key: self.get_dataset()}

    def get_main_dataset(self) -> Dict[str, Any]:
        return self.get_dataset()

    def get_main_name(self) -> str:
        return self.main_key

    def get_dataset(self) -> Dict[str, Any]:
        return {
            "class": "OggZipDataset",
            "audio": {
                "features": "raw",
                "peak_normalization": True,
                "preemphasis": None,
                "sample_rate": 16_000,
            },
            "path": CodeWrapper(DelayedFormat('CachedFile("{file}")', file=self.oggzip)),
            "partition_epoch": 1,
            "seq_ordering": "sorted_reverse",
            "targets": self.spm.get_opts(),
            "fixed_random_seed": 1,
        }


@cache
def _get_ogg_zip(
    corpus: tk.Path,
    name: str,
    returnn_root: Union[str, tk.Path],
    alias_prefix: str,
    duration: Optional[float] = None,
    split_per_hours: Optional[int] = None,
) -> Union[List[tk.Path]]:
    """
    Creates .ogg.zip files from the given corpus.

    Splits the corpus into `split` many files for compatibility with DistributeFilesDataset.

    To get an as random as possible distribution of the segments in the DistributeFilesDataset
    we shuffle + split the corpus segments, so that every created .ogg.zip gets segments from all
    the different source corpora (by chance).
    """

    segments = None
    if split_per_hours is not None:
        assert duration is not None
        assert split_per_hours > 0
        segments = SegmentCorpusJob(corpus, math.ceil(duration / split_per_hours)).out_segment_path
    # corpus = FilterCorpusBySegmentDurationJob(
    #     corpus,
    #     min_duration=0.01,
    #     max_duration=float("inf"),
    #     delete_empty_recordings=True,
    # ).out_corpus
    oggzip_job = BlissToSplitOggZipJob(
        corpus,
        corpus_name=name.replace("/", "."),
        resample_to=16_000,
        returnn_root=returnn_root,
        segments=segments,
    )
    oggzip_job.add_alias(f"{alias_prefix}/oggzip/{name}")
    for i, p in enumerate(oggzip_job.out_zips):
        tk.register_output(f"{alias_prefix}/oggzip/{name}/{i}.ogg.zip", p)
    return oggzip_job.out_zips


def get_task_data(
    *,
    corpora: Corpora,
    spm: SentencePieceModel,
    returnn_root: Union[str, tk.Path],
    train_partition_epoch: int,
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    train_dataset_kwargs: Optional[Dict[str, Any]] = None,
    alias_prefix: str,
    include_hf_large_scale_asr_corpus: bool = True,  # add another 25kh of data
    include_hf_libriheavy_corpus: bool = False,  # add another 50kh of data
    include_hf_gigaspeech_corpus: bool = False,  # add another 10kh of data
) -> EnglishData:
    train_oggzips = {
        key: _get_ogg_zip(
            corpus_def.bliss_corpus,
            name=f"train/{key}",
            split_per_hours=10,
            duration=corpus_def.duration,
            returnn_root=returnn_root,
            alias_prefix=alias_prefix,
        )
        for key, corpus_def in corpora.train.items()
    }
    if include_hf_large_scale_asr_corpus:
        ls_hf_job = LargeScaleASRCorpusToOggZipJob("large")
        ls_hf_job.add_alias(f"{alias_prefix}/oggzip/train/corpus.EN.f16kHz.SB-LargeScaleASR")
        for i, p in enumerate(ls_hf_job.out_oggzips):
            tk.register_output(f"{alias_prefix}/oggzip/train/corpus.EN.f16kHz.SB-LargeScaleASR/{i}.ogg.zip", p)
        train_oggzips["corpus.EN.f16kHz.SB-LargeScaleASR"] = ls_hf_job.out_oggzips
    if include_hf_gigaspeech_corpus:
        gs_hf_job = GigaspeechCorpusToOggZipJob("xl")
        gs_hf_job.add_alias(f"{alias_prefix}/oggzip/train/corpus.EN.f16kHz.GigaSpeech")
        for i, p in enumerate(gs_hf_job.out_oggzips):
            tk.register_output(f"{alias_prefix}/oggzip/train/corpus.EN.f16kHz.GigaSpeech/{i}.ogg.zip", p)
        train_oggzips["corpus.EN.f16kHz.GigaSpeech"] = gs_hf_job.out_oggzips
    if include_hf_libriheavy_corpus:
        lbh_hf_job = LibriheavyCorpusToOggZipJob("large")
        lbh_hf_job.add_alias(f"{alias_prefix}/oggzip/train/corpus.EN.f16kHz.LibriHeavy")
        for i, p in enumerate(lbh_hf_job.out_oggzips):
            tk.register_output(f"{alias_prefix}/oggzip/train/corpus.EN.f16kHz.LibriHeavy/{i}.ogg.zip", p)
        train_oggzips["corpus.EN.f16kHz.LibriHeavy"] = lbh_hf_job.out_oggzips
    cv_oggzip = _get_ogg_zip(corpora.cv, name="cv", returnn_root=returnn_root, alias_prefix=alias_prefix)
    dataset_train_cv_common_opts = {
        "cv_oggzip": cv_oggzip,
        "train_oggzip": [oggzip for oggzips_per_corp in train_oggzips.values() for oggzip in oggzips_per_corp],
        "spm": spm,
        "train_partition_epoch": train_partition_epoch,
    }
    cv = EnglishTrainDataset(**dataset_train_cv_common_opts, main_key="cv")
    dataset_train_opts = {**dataset_train_cv_common_opts, "spm": spm.copy(**(train_vocab_opts or {}))}
    if train_dataset_kwargs is not None:
        dataset_train_opts.update(train_dataset_kwargs)
    train = EnglishTrainDataset(**dataset_train_opts, main_key="train")

    dev_datas = {
        k: EvalOggZip(
            _get_ogg_zip(
                eval_info.segmented_corpus,
                name=f"seg.{eval_info.segmenter_type}/{k}",
                returnn_root=returnn_root,
                alias_prefix=alias_prefix,
            )[0],
            main_key=k,
            spm=spm,
        )
        for k, eval_info in corpora.dev.items()
    }
    test_datas = {
        k: EvalOggZip(
            _get_ogg_zip(
                eval_info.segmented_corpus,
                name=f"seg.{eval_info.segmenter_type}/{k}",
                returnn_root=returnn_root,
                alias_prefix=alias_prefix,
            )[0],
            main_key=k,
            spm=spm,
            gpu_mem=80 if any(id in k for id in ["EN_GB.f8kHz.eval-v3"]) else None,
        )
        for k, eval_info in corpora.test.items()
    }

    result = EnglishData(cv=cv, train=train, dev=dev_datas, test=test_datas)
    return result


class FilterInvalidCtcSeqsJob(Job):
    """
    Creates a seq len filter file that removes segments that cannot be CTC force-aligned
    under the given subsampling/frame rate reduction parameters.
    """

    def __init__(
        self,
        oggzip_file: Path,
        frame_shift_secs: float,
        spm: Path,
        sample_rate: int = 8000,
        feat_extract_shift_secs: float = 0.1,
        feat_extract_window_size_secs: float = 0.025,
        gzip: bool = True,
    ):
        self.oggzip_file = oggzip_file
        assert 0 < feat_extract_shift_secs <= frame_shift_secs
        self.frame_shift_secs = frame_shift_secs
        assert sample_rate > 0
        self.sample_rate = sample_rate
        assert feat_extract_shift_secs > 0
        self.feat_extract_shift_secs = feat_extract_shift_secs
        assert feat_extract_window_size_secs > 0
        self.feat_extract_window_size_secs = feat_extract_window_size_secs
        self.spm = spm

        self.out_valid_seqs = self.output_path("valid_seqs.txt" + (".gz" if gzip else ""))
        self.out_invalid_seqs = self.output_path("invalid_seqs.txt" + (".gz" if gzip else ""))

    def tasks(self) -> Iterator[Task]:
        return Task("run", mini_task=True)

    def run(self):
        from zipfile import ZipFile

        from returnn.util.literal_py_to_pickle import literal_eval
        from sentencepiece import SentencePieceProcessor

        sp = SentencePieceProcessor(self.spm)
        name, _ = os.path.splitext(os.path.basename(self.oggzip_file))
        with ZipFile(self.oggzip_file) as oggzip:
            data: List[Dict[str, Any]] = literal_eval(oggzip.read(f"{name}.txt"))

        downsampling_factor = self.frame_shift_secs / self.feat_extract_shift_secs
        red_subtrahend = self.feat_extract_window_size_secs * self.sample_rate - 1
        red_factor = self.sample_rate * self.feat_extract_shift_secs * downsampling_factor

        with uopen(self.out_valid_seqs, "wt") as out_valid, uopen(self.out_invalid_seqs, "wt") as out_invalid:
            for seq in data:
                num_frames = (seq["duration"] - red_subtrahend + red_factor + 1) / red_factor
                tokens = sp.Encode(seq["text"])
                seq_tag = seq["seq_name"]
                repetitions = sum(tokens[i] == tokens[i - 1] for i in range(1, len(tokens)))
                if num_frames >= len(tokens) + repetitions:
                    out_valid.write(f"{seq_tag}\n")
                else:
                    out_invalid.write(f"{seq_tag}\n")
