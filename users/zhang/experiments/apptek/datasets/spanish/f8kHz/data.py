import copy
from dataclasses import dataclass
from functools import cache
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from sisyphus import Job, Path, tk
from sisyphus.delayed_ops import DelayedFormat
from sisyphus.task import Task

from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.returnn.config import CodeWrapper
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.util import uopen

from .corpus_lex import Corpora


@dataclass(frozen=True)
class SpanishData:
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


class SpanishOggZip(DatasetConfig):
    def __init__(
        self,
        train_oggzip: Path,
        cv_oggzip: Path,
        spm: SentencePieceModel,
        main_key: str,
        train_partition_epoch: int,
        train_sort_laplace_num_seqs: Optional[int] = 1000,
        train_audio_preprocess: Optional[Any] = None,
        train_epoch_wise_filter: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None,
        filter_invalid_ctc_seq_for_frame_rate: Optional[int] = None,
    ) -> None:
        super().__init__()

        assert train_partition_epoch > 0
        assert train_sort_laplace_num_seqs is None or train_sort_laplace_num_seqs > 0

        self.oggzips = {"train": train_oggzip, "cv": cv_oggzip}
        self.spm = spm
        self.main_key = main_key
        self.train_audio_preprocess = train_audio_preprocess
        self.train_partition_epoch = train_partition_epoch
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
        obj = {
            "class": "OggZipDataset",
            "audio": {
                "features": "raw",
                "peak_normalization": True,
                "preemphasis": None,
                "sample_rate": 8_000,
            },
            "path": CodeWrapper(DelayedFormat('CachedFile("{file}")', file=self.oggzips[key])),
            "partition_epoch": self.train_partition_epoch if training else 1,
            "seq_ordering": (
                "sorted_reverse"
                if not training
                else (
                    f"laplace:.{self.train_sort_laplace_num_seqs}"
                    if self.train_sort_laplace_num_seqs is not None
                    else "random"
                )
            ),
            "targets": self.spm.get_opts(),
            "use_cache_manager": False,
        }

        if training:
            obj["audio"]["pre_process"] = self.train_audio_preprocess
            obj["epoch_wise_filter"] = self.train_epoch_wise_filter

            if self.filter_invalid_ctc_seq_for_frame_rate is not None:
                valid_seqs_job = FilterInvalidCtcSeqsJob(
                    self.oggzips[key], self.filter_invalid_ctc_seq_for_frame_rate, self.spm.model_file
                )
                obj["seq_list_filter_file"] = valid_seqs_job.out_valid_seqs
        else:
            obj["fixed_random_seed"] = 1

        return obj


class EvalOggZip(DatasetConfig):
    def __init__(self, oggzip_data: tk.Path, main_key: str, spm: SentencePieceModel):
        super().__init__()

        self.main_key = main_key
        self.oggzip = oggzip_data
        self.spm = spm

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
                "sample_rate": 8_000,
            },
            "path": CodeWrapper(DelayedFormat('CachedFile("{file}")', file=self.oggzip)),
            "partition_epoch": 1,
            "seq_ordering": "sorted_reverse",
            "targets": self.spm.get_opts(),
            "fixed_random_seed": 1,
        }


@cache
def _get_ogg_zip(
    corpus: tk.Path, name: str, split: int, returnn_root: Union[str, tk.Path], alias_prefix: str
) -> tk.Path:
    segment_job = SegmentCorpusJob(corpus, split)
    oggzip_job = BlissToOggZipJob(corpus, segments=segment_job.out_segment_path, returnn_root=returnn_root)
    oggzip_job.rqmt = {"cpu": 1, "mem": 2}
    oggzip_job.merge_rqmt = None  # merge on local machine, to be more robust against slowness due to slow FS
    oggzip_job.add_alias(f"{alias_prefix}/oggzip/{name}")
    tk.register_output(f"{alias_prefix}/oggzip/{name}.ogg.zip", oggzip_job.out_ogg_zip)
    return oggzip_job.out_ogg_zip


def get_task_data(
    *,
    corpora: Corpora,
    spm: SentencePieceModel,
    returnn_root: Union[str, tk.Path],
    train_partition_epoch: int,
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    train_dataset_kwargs: Optional[Dict[str, Any]] = None,
    alias_prefix: str,
) -> SpanishData:  #
    train_oggzip = _get_ogg_zip(
        corpora.train,
        name="train",
        split=300,
        returnn_root=returnn_root,
        alias_prefix=alias_prefix,
    )
    cv_oggzip = _get_ogg_zip(
        corpora.cv,
        name="cv",
        split=10,
        returnn_root=returnn_root,
        alias_prefix=alias_prefix,
    )
    dataset_train_cv_common_opts = {
        "cv_oggzip": cv_oggzip,
        "train_oggzip": train_oggzip,
        "spm": spm,
        "train_partition_epoch": train_partition_epoch,
    }
    cv = SpanishOggZip(**dataset_train_cv_common_opts, main_key="cv")
    dataset_train_opts = {**dataset_train_cv_common_opts, "spm": spm.copy(**(train_vocab_opts or {}))}
    if train_dataset_kwargs is not None:
        dataset_train_opts.update(train_dataset_kwargs)
    train = SpanishOggZip(**dataset_train_opts, main_key="train")

    dev_datas = {
        k: EvalOggZip(
            _get_ogg_zip(
                eval_info.segmented_corpus,
                name=f"seg.{eval_info.segmenter_type}/{k}",
                split=10,
                returnn_root=returnn_root,
                alias_prefix=alias_prefix,
            ),
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
                split=10,
                returnn_root=returnn_root,
                alias_prefix=alias_prefix,
            ),
            main_key=k,
            spm=spm,
        )
        for k, eval_info in corpora.test.items()
    }

    result = SpanishData(cv=cv, train=train, dev=dev_datas, test=test_datas)
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
