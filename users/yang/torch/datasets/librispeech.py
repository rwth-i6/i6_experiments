"""
Librispeech dataset
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Union, Tuple, Dict
from copy import deepcopy
import re
from functools import cache

from sisyphus import tk, Task as SisTask
from i6_core.corpus.convert import CorpusToTextDictJob
from i6_core.text.convert import TextDictToTextLinesJob
from i6_core.text.label.sentencepiece.train import TrainSentencePieceJob, SentencePieceType
from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob
from returnn.util.basic import NotSpecified
from returnn_common.datasets_old_2022_10.interface import DatasetConfig, VocabConfig
from i6_experiments.common.datasets import librispeech
from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.utils.basic import make_hashable
from i6_experiments.users.zeyer.speed_pert.librosa_09_10_11_kaiser_fast import (
    speed_pert_librosa_09_10_11_kaiser_fast as _default_train_audio_preprocess,
)
from i6_experiments.users.zeyer.datasets.task import Task, MeasureType, RecogOutput, ScoreResult
from .utils.bpe import Bpe
from .utils.spm import SentencePieceModel

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim
    from i6_experiments.users.zeyer.collect_model_dataset_stats import StatisticsOutput


_alias_prefix = "datasets/LibriSpeech/"


@cache
def _get_librispeech_ogg_zip_dict() -> Dict[str, tk.Path]:
    return librispeech.get_ogg_zip_dict()


@cache
def _get_bliss_corpus_dict() -> Dict[str, tk.Path]:
    # Get Bliss corpus. Same audio format as in ogg_zip, so already there anyway due to how we created the ogg_zip.
    # WARNING: Do not use these directly... It will keep another ogg copy of the audio...
    # However, these are used later in the scoring, so when changing them, make sure it's optional,
    # to not break hashes of old setups.
    return librispeech.get_bliss_corpus_dict(audio_format="ogg")


@cache
def _get_corpus_text_dict(key: str) -> tk.Path:
    job = CorpusToTextDictJob(_get_bliss_corpus_dict()[key], gzip=True)
    job.add_alias(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_dict")
    tk.register_output(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_dict.py.gz", job.out_dictionary)
    return job.out_dictionary


@cache
def _get_train_corpus_text() -> tk.Path:
    key = "train-other-960"
    train_corpus_text_dict = _get_corpus_text_dict(key)
    job = TextDictToTextLinesJob(train_corpus_text_dict, gzip=True)
    job.add_alias(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_lines")
    tk.register_output(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_lines.txt.gz", job.out_text_lines)
    return job.out_text_lines


@cache
def _get_spm_vocab(
    *, dim: Union[int, str], model_type: SentencePieceType = SentencePieceType.UNIGRAM
) -> SentencePieceModel:
    dim_str = str(dim)
    if isinstance(dim, str):
        # Not sure if power-of-two or just multiple-of-64, but 10240 has more 2s in it (2048*5) than 10048.
        dim = {"20k": 20_480, "10k": 10_240, "5k": 5_120, "4k": 4_096, "1k": 1_024, "512": 512, "128": 128}[dim]
    assert isinstance(dim, int) and dim >= 10

    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    _spm_train_job = TrainSentencePieceJob(
        training_text=_get_train_corpus_text(),
        vocab_size=dim,
        model_type=model_type,
        additional_options={
            "split_digits": True,
            "unk_id": 2,  # default is 0
            "bos_id": 1,  # default is 1
            "eos_id": 0,  # default is 2
        },
    )
    _spm_train_job.add_alias(_alias_prefix + f"vocab/spm_{model_type.value}_{dim_str}_train")
    tk.register_output(_alias_prefix + f"vocab/spm_{model_type.value}_{dim_str}_train.model", _spm_train_job.out_model)
    tk.register_output(
        _alias_prefix + f"vocab/spm_{model_type.value}_{dim_str}_train.vocab",
        ExtractSentencePieceVocabJob(_spm_train_job.out_model).out_vocab,
    )
    spm = SentencePieceModel(
        dim=dim,
        model_file=_spm_train_job.out_model,
        unknown_label="<unk>",
        bos_idx=1,
        eos_idx=0,
    )
    return spm


# common, this is the BPE10k that many of us use
bpe10k = Bpe(
    dim=10_025,
    eos_idx=0,
    bos_idx=0,
    codes=generic_job_output("i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.codes"),
    vocab=generic_job_output("i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab"),
    # unknown_label="<unk>",
    unknown_label=None,
)

# character vocab, but in a bpe style

bpe0 = Bpe(
    dim=56,
    eos_idx=0,
    bos_idx=0,
    unknown_label='<unk>',
    codes=tk.Path("/u/zyang/setups/vocab/dummy_codes.bpe"),
    vocab=tk.Path("/u/zyang/setups/vocab/char_bpe.vocab")
)


@cache
def get_vocab_by_str(vocab: str) -> Union[SentencePieceModel, Bpe]:
    """
    Get vocab
    """
    if re.match("^spm[0-9]+.*$", vocab):
        return _get_spm_vocab(dim=vocab[len("spm") :], model_type=SentencePieceType.UNIGRAM)
    elif re.match("^spm_bpe[0-9]+.*$", vocab):
        return _get_spm_vocab(dim=vocab[len("spm_bpe") :], model_type=SentencePieceType.BPE)
    elif vocab == "bpe10k":  # predefined
        return bpe10k
    elif vocab == "bpe0":
        return bpe0
    else:
        raise ValueError(f"invalid vocab {vocab!r}")


# ESPnet uses this SPM. However, it does not use the vocab directly from it.
# It has some custom code to generate its own vocab based from this:
# https://github.com/espnet/espnet/blob/d0047402e830a3c53e8b590064af4bf70415fb3b/egs2/TEMPLATE/asr1/asr.sh#L878
# Specifically, it removes <unk>, <s>, </s>, then adds back <blank> and <unk> at the beginning,
# and a single token for both EOS/SOS at the end.
spm_espnet_5k_wrong = SentencePieceModel(
    dim=5_000,
    model_file=tk.Path(
        "/u/zeineldeen/setups/ubuntu_22_setups/2024-02-12--aed-beam-search/work/downloaded_models/"
        "models--asapp--e_branchformer_librispeech/snapshots/f50914447c48b091738b3e020023ac69dbde9ea9/"
        "data/en_token_list/bpe_unigram5000/bpe.model",
        hash_overwrite="ESPnet-Librispeech-sentencepieces-5k",
    ),
    unknown_label="<unk>",  # idx 0
    bos_idx=1,
    eos_idx=2,
)


class CustomVocab(SentencePieceModel):
    """HACK: behaves like SPM, but this is actually some custom token list"""

    # noinspection PyMissingConstructor
    def __init__(self, *, dim: int, token_list: tk.Path, unknown_label: str, bos_idx: int, eos_idx: int):
        self.dim = dim
        self.token_list = token_list
        self.unknown_label = unknown_label
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def get_opts(self) -> Dict[str, Any]:
        return {
            "vocab_file": self.token_list,
            "num_labels": self.dim,
            "unknown_label": self.unknown_label,
            "bos_label": self.bos_idx,
            "eos_label": self.eos_idx,
        }

    def get_bos_idx(self) -> Optional[int]:
        return self.bos_idx

    def get_eos_idx(self) -> Optional[int]:
        return self.eos_idx


spm_espnet_5k = CustomVocab(
    dim=5_000,
    token_list=tk.Path(
        "/u/zeineldeen/setups/ubuntu_22_setups/2024-02-12--aed-beam-search/sym",
        hash_overwrite="ESPnet-Librispeech-sentencepieces-5k-tokenlist",
    ),
    unknown_label="<unk>",
    bos_idx=4999,
    eos_idx=4999,
)


_Parts = ["train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other", "test-clean", "test-other"]


# https://github.com/rwth-i6/returnn-experiments/blob/master/2020-librispeech-data-prepare/returnn.config
def _get_dataset(key: str, *, subset=None, train_partition_epoch=None, training: bool = False, targets, audio):
    files = []
    parts = [part for part in _Parts if part.startswith(key)]
    assert parts, f"invalid key {key!r}"
    for part in parts:
        files += [_get_librispeech_ogg_zip_dict()[part]]
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


default_train_epoch_split = 20

_default_train_epoch_wise_filter = {
    (1, 5): {"max_mean_len": 1000},  # better?
    # older settings:
    # (1, 5): {"max_mean_len": 200},
    # (6, 10): {"max_mean_len": 500},
}


class LibrispeechOggZip(DatasetConfig):
    """
    Librispeech dataset in OggZip format.
    """

    def __init__(
        self,
        *,
        audio: Optional[Dict[str, Any]] = None,
        audio_dim: Optional[int] = None,
        vocab: Optional[VocabConfig] = None,
        train_vocab: Optional[VocabConfig] = None,
        with_eos_postfix: bool = False,
        main_key: Optional[str] = None,
        train_epoch_split: int = default_train_epoch_split,
        train_sort_laplace_num_seqs: int = 1000,
        train_epoch_wise_filter: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = NotSpecified,
        train_audio_preprocess: Optional[Any] = NotSpecified,
        train_audio_random_permute: Union[bool, Dict[str, Any]] = False,
        eval_subset: Optional[int] = 3000,
    ):
        """
        :param with_eos_postfix: For RETURNN train/dev/eval datasets, mostly relevant for training.
            For recognition, our score function uses the Bliss corpus directly, so this has no influence.
        """
        from returnn.tensor import Dim

        super(LibrispeechOggZip, self).__init__()
        self.audio = audio
        self.audio_dim = audio_dim
        self.vocab = vocab
        self.train_vocab = train_vocab
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

        self._time_dim = None
        self._feature_dim = None
        if self.audio is not None:
            assert self.audio_dim is not None
            self._time_dim = Dim(None, name="time", kind=Dim.Types.Spatial)
            self._feature_dim = Dim(self.audio_dim, name="audio", kind=Dim.Types.Feature)

        self._out_spatial_dim = None
        self._classes_dim = None
        if self.vocab is not None:
            self._out_spatial_dim = Dim(None, name="out-spatial", kind=Dim.Types.Spatial)
            self._classes_dim = Dim(self.vocab.get_num_classes(), name="vocab", kind=Dim.Types.Spatial)

    def _sis_hash(self) -> bytes:
        # Note: Currently our GetBestRecogTrainExp job / _RecogAndScoreFunc sis hash
        # includes this instance in the hash
        # (unfortunately, as this is not really needed, as it is already part of the train job anyway).
        # Thus make sure any future changes here keep the old hash consistent.
        import hashlib
        from sisyphus.hash import sis_hash_helper

        # Keep consistent to the hash of the old LibrispeechOggZip via sis_hash_helper.
        state = self.__dict__.copy()
        if not self.train_vocab:
            state.pop("train_vocab")  # backward compat
        state = {k: v for k, v in state.items() if not k.startswith("_")}
        byte_list = [b"LibrispeechOggZip", sis_hash_helper(state)]

        # Same as sis_hash_helper.
        byte_str = b"(" + b", ".join(byte_list) + b")"
        if len(byte_str) > 4096:
            return hashlib.sha256(byte_str).digest()
        else:
            return byte_str

    def __sis_state__(self):
        # Avoid that any Dim instances are in here.
        return {k: v for (k, v) in self.__dict__.items() if not k.startswith("_")}

    def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get extern data
        """
        from returnn.tensor import Dim, batch_dim

        opts = {}

        if self.audio is not None:
            assert self.audio_dim is not None
            opts["data"] = {"dim_tags": [batch_dim, self._time_dim, self._feature_dim]}

        if self.vocab is not None:
            opts["classes"] = {
                "dim_tags": [batch_dim, self._out_spatial_dim],
                "sparse_dim": self._classes_dim,
                "vocab": self.vocab.get_opts(),
            }

        return opts

    def get_train_dataset(self) -> Dict[str, Any]:
        return self.get_dataset("train", training=True)

    def get_train_dataset_for_forward(self) -> Dict[str, Any]:
        return self.get_dataset("train")

    def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
        return {
            "dev": self.get_dataset("dev", subset=self.eval_subset),
            "devtrain": self.get_dataset("train", subset=self.eval_subset),
        }

    def get_main_name(self) -> str:
        return self.main_key

    def get_main_dataset(self) -> Dict[str, Any]:
        return self.get_dataset(self.main_key)

    def get_dataset(self, key: str, *, training: bool = False, subset: Optional[int] = None) -> Dict[str, Any]:
        files = []
        parts = [part for part in _Parts if part.startswith(key)]
        assert parts, f"invalid key {key!r}"
        for part in parts:
            files += [_get_librispeech_ogg_zip_dict()[part]]
        d: Dict[str, Any] = {
            "class": "OggZipDataset",
            "path": files,
            "use_cache_manager": True,
        }
        if self.audio is not None:
            d["audio"] = self.audio.copy()
        else:
            d["audio"] = None
        if self.vocab is not None:
            vocab = self.train_vocab if training and self.train_vocab else self.vocab
            d["targets"] = vocab.get_opts().copy()
            assert "seq_postfix" not in d["targets"], d  # we are handling this here
            if self.with_eos_postfix:
                eos_id = vocab.get_eos_idx()
                assert eos_id is not None, f"{self}: vocab {vocab} does not define EOS"
                d["targets"]["seq_postfix"] = [eos_id]
        else:
            d["targets"] = None
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


class LibrispeechOldFlacTarZip(DatasetConfig):
    """
    Librispeech dataset using the old LibriSpeechCorpus RETURNN dataset with use_zip=True,
    i.e. the original tar files repacked into zip files,
    i.e. keeping the original flac files inside the zip files.
    """

    # $ ls -la /u/zeyer/setups/librispeech/dataset/tars/
    # -rw-r--r-- 1 zeyer assi   360977013 Feb 26  2018 dev-clean.zip
    # -rw-r--r-- 1 zeyer assi   338709788 Feb 26  2018 dev-other.zip
    # -rw-r--r-- 1 zeyer assi        1024 Feb 27  2018 .history.zeyer
    # -rw-r--r-- 1 zeyer assi   369096021 Feb 26  2018 test-clean.zip
    # -rw-r--r-- 1 zeyer assi   353841318 Feb 26  2018 test-other.zip
    # -rw-r--r-- 1 zeyer assi  6625963133 Feb 26  2018 train-clean-100.zip
    # -rw-r--r-- 1 zeyer assi 23919296392 Feb 26  2018 train-clean-360.zip
    # -rw-r--r-- 1 zeyer assi 31839925140 Feb 26  2018 train-other-500.zip
    _librispeech_tars_zip_base_path = tk.Path(
        "/u/zeyer/setups/librispeech/dataset/tars", hash_overwrite="Librispeech-tars-zip-base-path"
    )

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
        train_epoch_wise_filter: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = NotSpecified,
        train_audio_preprocess: Optional[Any] = NotSpecified,
        train_audio_random_permute: Union[bool, Dict[str, Any]] = False,
        eval_subset: Optional[int] = 3000,
    ):
        """
        :param with_eos_postfix: For RETURNN train/dev/eval datasets, mostly relevant for training.
            For recognition, our score function uses the Bliss corpus directly, so this has no influence.
        """
        super(LibrispeechOldFlacTarZip, self).__init__()
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

    def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
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
            classes_dim = Dim(self.vocab.get_num_classes(), name="vocab", kind=Dim.Types.Spatial)
            opts["classes"] = {
                "dim_tags": [batch_dim, out_spatial_dim],
                "sparse_dim": classes_dim,
                "vocab": self.vocab.get_opts(),
            }

        return opts

    def get_train_dataset(self) -> Dict[str, Any]:
        return self.get_dataset("train", training=True)

    def get_train_dataset_for_forward(self) -> Dict[str, Any]:
        return self.get_dataset("train")

    def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
        return {
            "dev": self.get_dataset("dev", subset=self.eval_subset),
            "devtrain": self.get_dataset("train", subset=self.eval_subset),
        }

    def get_main_name(self) -> str:
        return self.main_key

    def get_main_dataset(self) -> Dict[str, Any]:
        return self.get_dataset(self.main_key)

    def get_dataset(self, key: str, *, training: bool = False, subset: Optional[int] = None) -> Dict[str, Any]:
        d = {
            "class": "LibriSpeechCorpus",
            "path": self._librispeech_tars_zip_base_path,
            "use_zip": True,
            "prefix": key,
            "use_cache_manager": True,
            # Keep seq tags consistent with our Bliss corpus and with the OggZipDataset.
            "seq_tag_format": "%(subdir)s/%(speaker)i-%(chapter)i-%(seq)04i/%(speaker)i-%(chapter)i-%(seq)04i",
        }
        if self.audio is not None:
            d["audio"] = self.audio.copy()
        if self.vocab is not None:
            d["targets"] = self.vocab.get_opts().copy()
            assert "seq_postfix" not in d["targets"], d  # we are handling this here
            if self.with_eos_postfix:
                eos_id = self.vocab.get_eos_idx()
                assert eos_id is not None, f"{self}: vocab {self.vocab} does not define EOS"
                d["targets"]["seq_postfix"] = [eos_id]
        if training:
            d["partition_epoch"] = self.train_epoch_split
            if self.train_epoch_wise_filter is not None:
                d["epoch_wise_filter"] = {"use_new_filter": True, **self.train_epoch_wise_filter}
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


_raw_audio_opts = dict(
    features="raw",
    sample_rate=16_000,
    peak_normalization=True,
    preemphasis=None,
)


def get_librispeech_task_raw(
    *,
    dataset_cls: Union[
        type[LibrispeechOggZip], type[LibrispeechOldFlacTarZip], type[DatasetConfig]
    ] = LibrispeechOggZip,
    vocab: VocabConfig,
    audio_opts: Optional[Dict[str, Any]] = None,
    audio_dim: int = 1,
    **dataset_train_opts,
) -> Task:
    """
    Librispeech
    """
    if isinstance(vocab, Bpe):
        vocab_to_words = _bpe_to_words_v1
    elif isinstance(vocab, SentencePieceModel):
        vocab_to_words = _spm_to_words
    else:
        raise TypeError(f"unhandled vocab type {type(vocab)}")

    audio_opts_ = _raw_audio_opts.copy()
    if audio_opts:
        audio_opts_.update(audio_opts)
    dataset_common_opts = dict(audio=audio_opts_, audio_dim=audio_dim, vocab=vocab)
    # We expect that all kwargs are only relevant for the training, thus we only pass them here.
    train_dataset = dataset_cls(**dataset_common_opts, **dataset_train_opts)
    dev_dataset = dataset_cls(**dataset_common_opts, main_key="dev-other")
    eval_datasets = {
        "dev-clean": dataset_cls(**dataset_common_opts, main_key="dev-clean"),
        "dev-other": dev_dataset,
        "test-clean": dataset_cls(**dataset_common_opts, main_key="test-clean"),
        "test-other": dataset_cls(**dataset_common_opts, main_key="test-other"),
    }

    return Task(
        name="librispeech",
        train_dataset=train_dataset,
        train_epoch_split=train_dataset.train_epoch_split,
        dev_dataset=dev_dataset,
        eval_datasets=eval_datasets,
        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name="dev-other",
        score_recog_output_func=_score_recog_out_v1,
        recog_post_proc_funcs=[vocab_to_words],
    )


def get_librispeech_task_bpe10k_raw(**dataset_train_opts) -> Task:
    return get_librispeech_task_raw(vocab=bpe10k, **dataset_train_opts)


_librispeech_task_raw_v2_cache = {}


def get_librispeech_task_raw_v2(
    *,
    dataset_cls: Union[
        type[LibrispeechOggZip], type[LibrispeechOldFlacTarZip], type[DatasetConfig]
    ] = LibrispeechOggZip,
    vocab: Union[VocabConfig, str],
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    audio_opts: Optional[Dict[str, Any]] = None,
    audio_dim: int = 1,
    **dataset_train_opts,
) -> Task:
    """
    Librispeech.

    Version 2:
    Use _bpe_to_words_v2 and _score_recog_out_v2 which does not use the Bliss corpus anymore directly,
    so it is easier to copy this setup to a new environment.
    """
    vocab_ = vocab
    if isinstance(vocab, str):
        vocab = get_vocab_by_str(vocab)

    cache_key = make_hashable((dataset_cls, vocab, train_vocab_opts, audio_opts, audio_dim, dataset_train_opts))
    if cache_key in _librispeech_task_raw_v2_cache:
        return _librispeech_task_raw_v2_cache[cache_key]

    if isinstance(vocab, Bpe):
        vocab_to_words = _bpe_to_words_v2
    elif isinstance(vocab, SentencePieceModel):
        vocab_to_words = _spm_to_words
    else:
        raise TypeError(f"unhandled vocab type {type(vocab)}")

    audio_opts_ = _raw_audio_opts.copy()
    if audio_opts:
        audio_opts_.update(audio_opts)
    dataset_common_opts = dict(audio=audio_opts_, audio_dim=audio_dim, vocab=vocab)
    if train_vocab_opts:
        dataset_common_opts["train_vocab"] = vocab.copy(**train_vocab_opts)
    # We expect that all kwargs are only relevant for the training, thus we only pass them here.
    train_dataset = dataset_cls(**dataset_common_opts, **dataset_train_opts)
    _extract_audio_seq_len_file(train_dataset)
    _extract_text_seq_len_file(train_dataset, vocab_, name="target")
    eval_datasets = {
        "dev-clean": dataset_cls(**dataset_common_opts, main_key="dev-clean"),
        "dev-other": dataset_cls(**dataset_common_opts, main_key="dev-other"),
        "test-clean": dataset_cls(**dataset_common_opts, main_key="test-clean"),
        "test-other": dataset_cls(**dataset_common_opts, main_key="test-other"),
    }
    dev_dataset = eval_datasets["dev-other"]

    task = Task(
        name="librispeech",
        train_dataset=train_dataset,
        train_epoch_split=train_dataset.train_epoch_split,
        dev_dataset=dev_dataset,
        eval_datasets=eval_datasets,
        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name="dev-other",
        score_recog_output_func=_score_recog_out_v2,
        recog_post_proc_funcs=[vocab_to_words],
    )
    _librispeech_task_raw_v2_cache[cache_key] = task
    return task


def _extract_audio_seq_len_file(train_dataset: DatasetConfig):
    """
    Extract audio seq len file
    """
    from sisyphus import tk
    from i6_core.returnn.dataset import ExtractSeqLensJob

    ds_dict = train_dataset.get_train_dataset()
    # The code is semi-generic. But anyway double check for now. Later to be extended...
    assert ds_dict["class"] in {"OggZipDataset", "LibriSpeechCorpus"}
    ds_dict.pop("partition_epoch")
    ds_dict["targets"] = None
    ds_dict.pop("epoch_wise_filter", None)
    ds_dict.pop("seq_ordering")
    # Originally, I extracted seq len stats with the pre_process.
    # But this complicates the code below for naming the file,
    # and also it's redundant.
    # The seq len stats can easily be inferred from the original stats for a given pre_process function.
    ds_dict["audio"].pop("pre_process", None)
    post_ds_dict = {}
    if "use_cache_manager" in ds_dict:
        post_ds_dict["use_cache_manager"] = ds_dict.pop("use_cache_manager")
    name_parts = []
    for k, v in ds_dict["audio"].items():
        if v is None:
            continue
        k_s = re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), k)
        name_parts.append(f"{k_s}={v}")
    job = ExtractSeqLensJob(ds_dict, post_ds_dict, key=train_dataset.get_default_input(), output_format="txt")
    job.rqmt["time"] = 3
    tk.register_output(_alias_prefix + "seq_len_audio-%s.txt" % "-".join(name_parts), job.out_file)
    return job.out_file


def _extract_text_seq_len_file(train_dataset: DatasetConfig, vocab_cfg: Union[str, VocabConfig], *, name: str):
    """
    Extract target seq len file
    """
    from sisyphus import tk
    from i6_core.returnn.dataset import ExtractSeqLensJob

    name_parts = []
    if isinstance(vocab_cfg, str):
        name_parts.append(vocab_cfg)
    elif isinstance(vocab_cfg, VocabConfig):
        name_parts.append(vocab_cfg.__class__.__name__)
        for k, v in vocab_cfg.get_opts().items():
            name_parts.append(f"{k}={v}")
    else:
        raise TypeError(f"invalid vocab_cfg {vocab_cfg!r} type {type(vocab_cfg)}")

    ds_dict = train_dataset.get_train_dataset()
    # The code is semi-generic. But anyway double check for now. Later to be extended...
    assert ds_dict["class"] in {"OggZipDataset", "LibriSpeechCorpus", "LmDataset"}
    vocab_key = "targets" if ds_dict["class"] in {"OggZipDataset", "LibriSpeechCorpus"} else "orth_vocab"
    ds_dict.pop("partition_epoch")
    if ds_dict["class"] in {"OggZipDataset", "LibriSpeechCorpus"}:
        assert "audio" in ds_dict
        ds_dict["audio"] = None
    ds_dict.pop("epoch_wise_filter", None)
    ds_dict.pop("seq_ordering")
    post_ds_dict = {}
    if "use_cache_manager" in ds_dict:
        post_ds_dict["use_cache_manager"] = ds_dict.pop("use_cache_manager")
    for k, v in ds_dict[vocab_key].items():
        if k in {
            "bpe_file",
            "vocab_file",
            "model_file",
            "unknown_label",
            "bos_label",
            "eos_label",
            "word_prefix_symbol",
        }:  # ignore those here
            continue
        if k == "class" and v in {"SentencePieces", "Utf8ByteTargets"}:
            continue
        k_s = re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), k)
        name_parts.append(f"{k_s}={v}")
    job = ExtractSeqLensJob(ds_dict, post_ds_dict, key=train_dataset.get_default_target(), output_format="txt")
    tk.register_output(_alias_prefix + f"seq_len_{name}-" + "%s.txt" % "-".join(name_parts), job.out_file)
    return job.out_file


def _bpe_to_words_v1(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    from i6_core.returnn.search import SearchBPEtoWordsJob

    words = SearchBPEtoWordsJob(bpe.output, output_gzip=True).out_word_search_results
    return RecogOutput(output=words)


def _bpe_to_words_v2(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    from i6_core.returnn.search import SearchOutputRawReplaceJob

    words = SearchOutputRawReplaceJob(bpe.output, [("@@ ", "")], output_gzip=True).out_search_results
    return RecogOutput(output=words)


def _spm_to_words(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    from i6_core.returnn.search import SearchOutputRawReplaceJob

    words = SearchOutputRawReplaceJob(bpe.output, [(" ", ""), ("▁", " ")], output_gzip=True).out_search_results
    return RecogOutput(output=words)


def _score_recog_out_v1(dataset: DatasetConfig, recog_output: RecogOutput) -> ScoreResult:
    """score"""
    # We use sclite now.
    # Could also use ReturnnComputeWERJob.
    from i6_core.corpus.convert import CorpusToStmJob
    from i6_core.returnn.search import SearchWordsToCTMJob
    from i6_core.recognition.scoring import ScliteJob

    hyp_words = recog_output.output
    corpus_name = dataset.get_main_name()

    bliss_corpus = _get_bliss_corpus_dict()[corpus_name]
    search_ctm = SearchWordsToCTMJob(recog_words_file=hyp_words, bliss_corpus=bliss_corpus).out_ctm_file
    stm_file = CorpusToStmJob(bliss_corpus=bliss_corpus).out_stm_path

    score_job = ScliteJob(
        ref=stm_file, hyp=search_ctm, sctk_binary_path=tools_paths.get_sctk_binary_path(), precision_ndigit=2
    )

    return ScoreResult(dataset_name=corpus_name, main_measure_value=score_job.out_wer, report=score_job.out_report_dir)


def _score_recog_out_v2(dataset: DatasetConfig, recog_output: RecogOutput) -> ScoreResult:
    """score"""
    # We use sclite now.
    # Could also use ReturnnComputeWERJob.
    from i6_core.returnn.search import SearchWordsDummyTimesToCTMJob
    from i6_core.text.convert import TextDictToStmJob
    from i6_core.recognition.scoring import ScliteJob

    hyp_words = recog_output.output
    corpus_name = dataset.get_main_name()

    corpus_text_dict = _get_corpus_text_dict(corpus_name)
    # Arbitrary seg length time. The jobs SearchWordsDummyTimesToCTMJob and TextDictToStmJob
    # serialize two points after decimal, so long seqs (>1h or so) might be problematic,
    # and no reason not to just use a high value here to avoid this problem whenever we get to it.
    seg_length_time = 1000.0
    search_ctm = SearchWordsDummyTimesToCTMJob(
        recog_words_file=hyp_words, seq_order_file=corpus_text_dict, seg_length_time=seg_length_time
    ).out_ctm_file
    stm_file = TextDictToStmJob(text_dict=corpus_text_dict, seg_length_time=seg_length_time).out_stm_path

    score_job = ScliteJob(
        ref=stm_file, hyp=search_ctm, sctk_binary_path=tools_paths.get_sctk_binary_path(), precision_ndigit=2
    )

    return ScoreResult(dataset_name=corpus_name, main_measure_value=score_job.out_wer, report=score_job.out_report_dir)


def get_librispeech_raw_audio_only(*, main_key: str = "train") -> LibrispeechOggZip:
    """librispeech with raw audio"""
    return LibrispeechOggZip(audio=_raw_audio_opts, audio_dim=1, main_key=main_key)


def get_librispeech_log_mel_stats(dim: int, **kwargs) -> StatisticsOutput:
    """
    Get feature stats

    :param dim: feature dim
    :param kwargs: all passed to rf.audio.log_mel_filterbank_from_raw.
        Default sampling_rate is 16_000, which is exactly also what we have for Librispeech usually.
        Note on log_base: Default is 10.0.
            Note that in some earlier setups, and also Mohammads original AED setup,
            we used log_base=math.exp(2.3026), which is almost 10.0 but not exactly...
    """
    from i6_experiments.users.zeyer.collect_model_dataset_stats import collect_log_mel_feature_statistics

    return collect_log_mel_feature_statistics(dataset=get_librispeech_raw_audio_only(), dim=dim, **kwargs)


def _librispeech_log_mel_stats_returnn_forward(
    source: Tensor, /, in_spatial_dim: Dim, model: Any
) -> Tuple[Tensor, Dim]:
    from returnn.config import get_global_config
    import returnn.frontend as rf
    from returnn.tensor import Dim

    model  # noqa # unused
    config = get_global_config()
    feat_dim = config.int("_audio_feature_dim", -1)
    assert feat_dim > 0
    feat_dim = Dim(feat_dim, name="audio", kind=Dim.Types.Feature)
    opts = config.typed_value("_audio_feature_opts", None)
    assert isinstance(opts, dict)

    source, out_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
        source, in_spatial_dim=in_spatial_dim, out_dim=feat_dim, **opts
    )
    return source, out_spatial_dim


def seq_list_960_to_split_100_360_500(seq_list: tk.Path) -> tk.Path:
    """
    :param seq_list:
        E.g. contains (in combined 960h dataset) "train-other-960/1034-121119-0049/1034-121119-0049",
        but it's actually "train-clean-100/1034-121119-0049/1034-121119-0049".
    :return: correct seq tags list with split parts
    """
    dataset = LibrispeechOggZip(main_key="train").get_main_dataset()
    return ConvertSeqList960ToSplit100_360_500(seq_list=seq_list, returnn_dataset=dataset).out_seq_list


def seq_list_split_100_360_500_to_single_960(seq_list: tk.Path) -> tk.Path:
    """
    :param seq_list:
        E.g. contains (in combined 960h dataset) "train-other-960/1034-121119-0049/1034-121119-0049",
        but it's actually "train-clean-100/1034-121119-0049/1034-121119-0049".
    :return: correct seq tags list with split parts
    """
    dataset = LibrispeechOggZip(main_key="train").get_main_dataset()
    dataset["path"] = _get_librispeech_ogg_zip_dict()["train-other-960"]
    return ConvertSeqList960ToSplit100_360_500(seq_list=seq_list, returnn_dataset=dataset).out_seq_list


class ConvertSeqList960ToSplit100_360_500(tk.Job):
    """
    E.g. contains (in combined 960h dataset) "train-other-960/1034-121119-0049/1034-121119-0049",
    but it's actually "train-clean-100/1034-121119-0049/1034-121119-0049".

    (Note: the job name is maybe misleading, it can also do other ways. it compares based on the seq tag base name.)
    """

    def __init__(
        self,
        *,
        seq_list: tk.Path,
        returnn_dataset: Dict[str, Any],  # to get all seq tags
        returnn_root: Optional[tk.Path] = None,
    ):
        self.seq_list = seq_list
        self.returnn_dataset = returnn_dataset
        self.returnn_root = returnn_root

        self.out_seq_list = self.output_path("out_seq_list.txt")

    def tasks(self):
        yield SisTask("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        import sys
        import os
        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(1, returnn_root.get_path())

        seq_list = open(self.seq_list.get_path()).read().splitlines()

        from returnn.config import set_global_config, Config
        from returnn.datasets import init_dataset
        from returnn.log import log

        config = Config()
        set_global_config(config)

        if not config.has("log_verbosity"):
            config.typed_dict["log_verbosity"] = 4
        log.init_by_config(config)

        import tree

        dataset_dict = self.returnn_dataset
        dataset_dict = tree.map_structure(lambda x: x.get_path() if isinstance(x, tk.Path) else x, dataset_dict)
        print("RETURNN dataset dict:", dataset_dict)
        assert isinstance(dataset_dict, dict)
        dataset = init_dataset(dataset_dict)

        all_tags = set(dataset.get_all_tags())
        all_tags_wo_prefix = {}
        for tag in all_tags:
            tag_wo_prefix = tag.split("/", 2)[-1]
            assert tag_wo_prefix not in all_tags_wo_prefix
            all_tags_wo_prefix[tag_wo_prefix] = tag
        seq_list_ = []
        for seq_tag in seq_list:
            tag_wo_prefix = seq_tag.split("/", 2)[-1]
            if seq_tag in all_tags:
                seq_list_.append(seq_tag)
            elif tag_wo_prefix in all_tags_wo_prefix:
                seq_list_.append(all_tags_wo_prefix[tag_wo_prefix])
            else:
                print(f"seq tag {seq_tag} not found in dataset")

        with open(self.out_seq_list.get_path(), "w") as f:
            for seq_tag in seq_list_:
                print(seq_tag, file=f)


class LibrispeechLmDataset(DatasetConfig):
    """
    Librispeech LM dataset
    """

    def __init__(
        self,
        *,
        vocab: VocabConfig,
        train_vocab: Optional[VocabConfig] = None,
        main_key: Optional[str] = None,
        train_epoch_split: int = default_train_epoch_split,
        train_sort_laplace_num_seqs: int = 1000,
        eval_subset: Optional[int] = 3000,
    ):
        super().__init__()
        self.vocab = vocab
        self.train_vocab = train_vocab
        self.main_key = main_key
        self.train_epoch_split = train_epoch_split
        self.train_sort_laplace_num_seqs = train_sort_laplace_num_seqs
        self.eval_subset = eval_subset

    def _sis_hash(self) -> bytes:
        import hashlib
        from sisyphus.hash import sis_hash_helper

        # Keep consistent once we do any changes.
        state = self.__dict__.copy()
        if not self.train_vocab:
            state.pop("train_vocab")  # backward compat
        byte_list = [b"LibrispeechLmDataset", sis_hash_helper(state)]

        # Same as sis_hash_helper.
        byte_str = b"(" + b", ".join(byte_list) + b")"
        if len(byte_str) > 4096:
            return hashlib.sha256(byte_str).digest()
        else:
            return byte_str

    def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get extern data
        """
        from returnn.tensor import Dim, batch_dim

        out_spatial_dim = Dim(None, name="out-spatial", kind=Dim.Types.Spatial)
        classes_dim = Dim(self.vocab.get_num_classes(), name="vocab", kind=Dim.Types.Spatial)

        return {
            "data": {
                "dim_tags": [batch_dim, out_spatial_dim],
                "sparse_dim": classes_dim,
                "vocab": self.vocab.get_opts(),
            }
        }

    def get_default_input(self) -> Optional[str]:
        """data"""
        return "data"

    def get_default_target(self) -> Optional[str]:
        """data"""
        return "data"

    def get_train_dataset(self) -> Dict[str, Any]:
        return self.get_dataset("train", training=True)

    def get_train_dataset_for_forward(self) -> Dict[str, Any]:
        return self.get_dataset("train")

    def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
        return {
            "dev": self.get_dataset("transcriptions-dev-other", subset=self.eval_subset),
            "devtrain": self.get_dataset("transcriptions-train-clean-100", subset=self.eval_subset),
        }

    def get_main_name(self) -> str:
        return self.main_key

    def get_main_dataset(self) -> Dict[str, Any]:
        return self.get_dataset(self.main_key)

    def get_dataset(self, key: str, *, training: bool = False, subset: Optional[int] = None) -> Dict[str, Any]:
        vocab = self.train_vocab if training and self.train_vocab else self.vocab
        if key == "train":
            from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data

            d: Dict[str, Any] = {
                "class": "LmDataset",
                "corpus_file": [get_librispeech_normalized_lm_data(), _get_train_corpus_text()],
                "use_cache_manager": True,
                "orth_vocab": vocab.get_opts().copy(),
                "seq_end_symbol": None,  # handled via orth_vocab
                "unknown_symbol": None,  # handled via orth_vocab
            }
        elif key.startswith("transcriptions-"):
            files = []
            parts = [part for part in _Parts if part.startswith(key[len("transcriptions-") :])]
            assert parts, f"invalid key {key!r}"
            for part in parts:
                files += [_get_librispeech_ogg_zip_dict()[part]]
            d: Dict[str, Any] = {
                "class": "OggZipDataset",
                "path": files,
                "use_cache_manager": True,
                "audio": None,
                "targets": vocab.get_opts().copy(),
            }
        else:
            raise ValueError(f"invalid key {key!r}")
        if training:
            d["partition_epoch"] = self.train_epoch_split
            d["seq_ordering"] = f"laplace:.{self.train_sort_laplace_num_seqs}"
        else:
            if d["class"] == "OggZipDataset":
                d["fixed_random_seed"] = 1
            d["seq_ordering"] = "sorted_reverse"
        if subset:
            d["fixed_random_subset"] = subset  # faster
        if d["class"] == "OggZipDataset":
            d = {
                "class": "MetaDataset",
                "datasets": {"ogg_zip": d},
                "data_map": {"data": ("ogg_zip", "classes")},
                "seq_order_control_dataset": "ogg_zip",
            }
        return d


_librispeech_lm_dataset_raw_cache = {}
_librispeech_lm_raw_seq_lens = False


def get_librispeech_lm_dataset(
    *,
    vocab: Union[VocabConfig, str],
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    **opts,
) -> LibrispeechLmDataset:
    """
    Librispeech LM.
    """
    vocab_ = vocab
    if isinstance(vocab, str):
        vocab = get_vocab_by_str(vocab)

    cache_key = make_hashable((vocab, train_vocab_opts, opts))
    if cache_key in _librispeech_lm_dataset_raw_cache:
        return _librispeech_lm_dataset_raw_cache[cache_key]

    opts = opts.copy()
    if train_vocab_opts:
        assert "train_vocab" not in opts
        opts["train_vocab"] = vocab.copy(**train_vocab_opts)
    # We expect that all kwargs are only relevant for the training, thus we only pass them here.
    train_dataset = LibrispeechLmDataset(vocab=vocab, **opts)
    _extract_text_seq_len_file(train_dataset, vocab_, name="lm_text")

    global _librispeech_lm_raw_seq_lens
    if not _librispeech_lm_raw_seq_lens:
        from i6_experiments.users.zeyer.datasets.utils.bytes import Utf8BytesVocab

        _librispeech_lm_raw_seq_lens = True
        _extract_text_seq_len_file(LibrispeechLmDataset(vocab=Utf8BytesVocab()), vocab_cfg="utf8bytes", name="lm_text")

    _librispeech_lm_dataset_raw_cache[cache_key] = train_dataset
    return train_dataset


def tests():
    from sisyphus.hash import sis_hash_helper

    task = get_librispeech_task_raw_v2(vocab="bpe10k")
    model = ...  # dummies, not relevant here
    recog_def = ...

    from i6_experiments.users.zeyer.recog import _RecogAndScoreFunc

    # This is used in GetBestRecogTrainExp. Make sure the hash is stable.
    recog_and_score_func = _RecogAndScoreFunc(
        prefix_name="test_recog_and_score_func",  # should not matter
        task=task,
        model=model,
        recog_def=recog_def,
    )
    h1 = sis_hash_helper(recog_and_score_func)
    assert (
        h1 == b"(dict, (tuple, (str, 'class'), (str, '_RecogAndScoreFunc')), (tuple, (str, 'model'),"
        b" (ellipsis, (NoneType))), (tuple, (str, 'recog_def'), (ellipsis, (NoneType))),"
        b" (tuple, (str, 'task.train_dataset'),"
        b" (LibrispeechOggZip, (dict, (tuple, (str, 'audio'),"
        b" (dict, (tuple, (str, 'features'), (str, 'raw')), (tuple, (str, 'peak_normalization'), (bool, True)),"
        b" (tuple, (str, 'preemphasis'), (NoneType)), (tuple, (str, 'sample_rate'), (int, 16000)))),"
        b" (tuple, (str, 'audio_dim'), (int, 1)), (tuple, (str, 'eval_subset'),"
        b" (int, 3000)), (tuple, (str, 'main_key'), (NoneType)),"
        b" (tuple, (str, 'train_audio_preprocess'),"
        b" (function, (tuple, (str, 'i6_experiments.users.zeyer.speed_pert.librosa_09_10_11_kaiser_fast'),"
        b" (str, 'speed_pert_librosa_09_10_11_kaiser_fast')))),"
        b" (tuple, (str, 'train_audio_random_permute'), (bool, False)),"
        b" (tuple, (str, 'train_epoch_split'), (int, 20)), (tuple, (str, 'train_epoch_wise_filter'),"
        b" (dict, (tuple, (tuple, (int, 1), (int, 5)), (dict, (tuple, (str, 'max_mean_len'), (int, 1000)))))),"
        b" (tuple, (str, 'train_sort_laplace_num_seqs'), (int, 1000)), (tuple, (str, 'vocab'),"
        b" (Bpe, (dict, (tuple, (str, 'bos_idx'), (int, 0)), (tuple, (str, 'codes'),"
        b" (Path, (tuple, (str, 'i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output'),"
        b" (str, 'bpe.codes')))), (tuple, (str, 'dim'), (int, 10025)), (tuple, (str, 'eos_idx'),"
        b" (int, 0)), (tuple, (str, 'other_opts'), (NoneType)), (tuple, (str, 'unknown_label'),"
        b" (NoneType)), (tuple, (str, 'vocab'),"
        b" (Path, (tuple, (str, 'i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output'),"
        b" (str, 'bpe.vocab'))))))), (tuple, (str, 'with_eos_postfix'), (bool, False))))),"
        b" (tuple, (str, 'task.train_epoch_split'), (int, 20)))"
    )
