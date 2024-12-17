"""
Librispeech dataset
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Union, Tuple, Dict
from copy import deepcopy, copy
import re
import os
from enum import Enum
from functools import cache

from sisyphus import tk, Task as SisTask
from sisyphus.delayed_ops import DelayedBase
from i6_core.util import instanciate_delayed
from i6_core.corpus.convert import CorpusToTextDictJob
from i6_core.text.convert import TextDictToTextLinesJob
from i6_core.text.label.subword_nmt.train import ReturnnTrainBpeJob
from i6_core.text.label.sentencepiece.train import TrainSentencePieceJob, SentencePieceType
from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob
from returnn.util.basic import NotSpecified
from returnn_common.datasets_old_2022_10.interface import DatasetConfig, VocabConfig, VocabConfigStatic
from i6_experiments.common.datasets import librispeech
from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.utils.basic import make_hashable
from i6_experiments.users.zeyer.speed_pert.librosa_09_10_11_kaiser_fast import (
    speed_pert_librosa_09_10_11_kaiser_fast as _default_train_audio_preprocess,
)
from .task import Task, MeasureType, RecogOutput, ScoreResult
from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from i6_experiments.users.zeyer.datasets.utils.bytes import Utf8BytesVocab
from i6_experiments.users.zeyer.datasets.utils.char import get_char_vocab

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim
    from i6_experiments.users.zeyer.collect_model_dataset_stats import StatisticsOutput
    
from .utils import CorpusReplaceOrthFromPyDictJob, get_ogg_zip_dict_pseudo_labels, MetaDataset

_alias_prefix = "datasets/LibriSpeech/"


@cache
def _get_librispeech_ogg_zip_dict() -> Dict[str, tk.Path]:
    return librispeech.get_ogg_zip_dict()


@cache
def _get_bliss_corpus_dict(pseudo_labels_path: tk.Path, part: str) -> Dict[str, tk.Path]:
    # Get Bliss corpus. Same audio format as in ogg_zip, so already there anyway due to how we created the ogg_zip.
    # WARNING: Do not use these directly... It will keep another ogg copy of the audio...
    # However, these are used later in the scoring, so when changing them, make sure it's optional,
    # to not break hashes of old setups.
    if pseudo_labels_path:
        assert part is not None
        print("Made it to pseudo label combination", pseudo_labels_path)
        bliss_corpus_dict = librispeech.get_bliss_corpus_dict(audio_format="ogg")
        # load pseudo labels and replace here
        bliss_corpus = bliss_corpus_dict[part]
        replace_job = CorpusReplaceOrthFromPyDictJob(bliss_corpus, pseudo_labels_path)
        replace_job.add_alias(os.path.join("datasets", "LibriSpeech-PseudoLabels", "%s_replace_orth" % part.replace('-', '_')))
        bliss_corpus = replace_job.out_corpus
        return {part: bliss_corpus}
    else:
        return librispeech.get_bliss_corpus_dict(audio_format="ogg")


@cache
def _get_librispeech_ogg_zip_dict_pseudo_labels(pseudo_labels_path: tk.Path, part: str) -> Dict[str, tk.Path]:
    # print("Convert pseudo labels to ogg")
    
    bliss_corpus_dict = _get_bliss_corpus_dict(pseudo_labels_path, part)

    return get_ogg_zip_dict_pseudo_labels(bliss_corpus_dict)


@cache
def _get_corpus_text_dict(key: str) -> tk.Path:
    job = CorpusToTextDictJob(_get_bliss_corpus_dict(None, key)[key], gzip=True)
    job.add_alias(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_dict")
    tk.register_output(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_dict.py.gz", job.out_dictionary)
    return job.out_dictionary


@cache
def _get_train_corpus_text(train_small: bool = False) -> tk.Path:
    if train_small:
        key = "train-clean-100"
    else:
        key = "train-other-960"
    train_corpus_text_dict = _get_corpus_text_dict(key)
    job = TextDictToTextLinesJob(train_corpus_text_dict, gzip=True)
    job.add_alias(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_lines")
    tk.register_output(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_lines.txt.gz", job.out_text_lines)
    return job.out_text_lines


@cache
def _get_spm_vocab(
    *, dim: Union[int, str], model_type: SentencePieceType = SentencePieceType.UNIGRAM, train_full: bool = False, train_small: bool = False
) -> SentencePieceModel:
    dim_str = str(dim)
    if isinstance(dim, str):
        # Not sure if power-of-two or just multiple-of-64, but 10240 has more 2s in it (2048*5) than 10048.
        dim = {"20k": 20_480, "10k": 10_240, "5k": 5_120, "4k": 4_096, "1k": 1_024, "512": 512, "128": 128, "64": 64}[
            dim
        ]
    assert isinstance(dim, int) and dim >= 10

    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    _spm_train_job = TrainSentencePieceJob(
        training_text=get_librispeech_lm_combined_txt(train_small) if train_full else _get_train_corpus_text(train_small),
        vocab_size=dim,
        model_type=model_type,
        additional_options={
            "split_digits": True,
            "unk_id": 2,  # default is 0
            "bos_id": 1,  # default is 1
            "eos_id": 0,  # default is 2
            **(
                {
                    "train_extremely_large_corpus": True,
                    "shuffle_input_sentence": True,
                    "input_sentence_size": 10_000_000,  # oom otherwise, with full (40M), it takes more than 126GB
                }
                if train_full
                else {}
            ),
        },
    )
    name_postfix = "_full" if train_full else ""
    if train_full:
        _spm_train_job.rqmt.update({"time": 12, "mem": 126})  # needs much more mem, maybe little longer
    _spm_train_job.add_alias(f"{_alias_prefix}vocab/spm_{model_type.value}_{dim_str}_train{name_postfix}")
    tk.register_output(
        f"{_alias_prefix}vocab/spm_{model_type.value}_{dim_str}_train{name_postfix}.model", _spm_train_job.out_model
    )
    tk.register_output(
        f"{_alias_prefix}vocab/spm_{model_type.value}_{dim_str}_train{name_postfix}.vocab",
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


@cache
def _get_bpe_vocab(*, bpe_size: Union[int, str], train_small: bool = False) -> Bpe:
    bpe_size_str = str(bpe_size)
    if isinstance(bpe_size, str):
        bpe_size = {"128": 128, "64": 64, "0": 0}[bpe_size]
    assert isinstance(bpe_size, int)
    # Note: Once we need another size, put some dummy value here first,
    # then run the BPE training, then get the real vocab size, then update this.
    # We need to do it this manual way until we properly handle _DelayedDim,
    # but for _DelayedDim, we also need the new serialization_v2,
    # and update the whole train/recog pipeline...
    dim = {128: 184, 64: 120, 0: 56}[bpe_size]

    from i6_core.tools.git import CloneGitRepositoryJob

    subword_nmt_job = CloneGitRepositoryJob(
        url="https://github.com/rwth-i6/subword-nmt",
        commit="5015a45e28a958f800ef1c50e7880c0c9ef414cf",
        checkout_folder_name="subword-nmt",
    )
    subword_nmt_repo = subword_nmt_job.out_repository
    subword_nmt_repo.hash_overwrite = "I6_SUBWORD_NMT_V2"  # this is what most other people use as well

    _bpe_train_job = ReturnnTrainBpeJob(
        text_file=_get_train_corpus_text(train_small),
        bpe_size=bpe_size,
        unk_label="<unk>",
        subword_nmt_repo=subword_nmt_repo,
    )
    _bpe_train_job.add_alias(f"{_alias_prefix}vocab/bpe_{bpe_size_str}_train")
    tk.register_output(f"{_alias_prefix}vocab/bpe_{bpe_size_str}_train.vocab", _bpe_train_job.out_bpe_vocab)
    tk.register_output(f"{_alias_prefix}vocab/bpe_{bpe_size_str}_train.codes", _bpe_train_job.out_bpe_codes)
    tk.register_output(f"{_alias_prefix}vocab/bpe_{bpe_size_str}_train.vocab_size", _bpe_train_job.out_vocab_size)
    bpe = Bpe(
        dim=dim,
        codes=_bpe_train_job.out_bpe_codes,
        vocab=_bpe_train_job.out_bpe_vocab,
        # unknown_label="<unk>",
        unknown_label=None,
        bos_idx=0,
        eos_idx=0,
    )
    return bpe


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


@cache
def get_vocab_by_str(vocab: str, train_small: bool = False) -> Union[SentencePieceModel, Bpe, VocabConfigStatic, Utf8BytesVocab]:
    """
    Get vocab
    """
    if re.match("^spm[0-9]+.*$", vocab):
        return _get_spm_vocab(dim=vocab[len("spm") :], model_type=SentencePieceType.UNIGRAM, train_small=train_small)
    elif re.match("^spmLm[0-9]+.*$", vocab):
        return _get_spm_vocab(dim=vocab[len("spmLm") :], model_type=SentencePieceType.UNIGRAM, train_full=True, train_small=train_small)
    elif re.match("^spm_bpe[0-9]+.*$", vocab):
        return _get_spm_vocab(dim=vocab[len("spm_bpe") :], model_type=SentencePieceType.BPE, train_small=train_small)
    elif vocab == "bpe10k":  # predefined
        if train_small:
            raise ValueError(f"bpe10k not available for train_small")
        else:
            return bpe10k
    elif re.match("^bpe[0-9]+.*$", vocab):
        return _get_bpe_vocab(bpe_size=vocab[len("bpe") :], train_small=train_small)
    elif vocab == "char":
        return get_char_vocab(
            get_librispeech_lm_combined_txt(train_small=train_small), num_classes=29, extra_labels=("\x00",), eos_label="\x00"
        )
    elif vocab == "utf8":
        return Utf8BytesVocab(eos_label=0)
    else:
        raise ValueError(f"invalid vocab {vocab!r}")


def get_librispeech_lm_combined_txt(train_small: bool = False) -> tk.Path:
    from i6_core.text.processing import ConcatenateJob
    from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data

    return ConcatenateJob([get_librispeech_normalized_lm_data(), _get_train_corpus_text(train_small)]).out


def get_bpe_lexicon(bpe_vocab: Bpe) -> tk.Path:
    """
    Create BPE lexicon without unknown and silence

    :return: path to a lexicon bliss xml file
    """
    from i6_core.tools.git import CloneGitRepositoryJob
    from i6_core.lexicon.bpe import CreateBPELexiconJob
    from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
    from i6_experiments.common.datasets.librispeech import get_bliss_lexicon

    subword_nmt_job = CloneGitRepositoryJob(
        url="https://github.com/rwth-i6/subword-nmt",
        commit="5015a45e28a958f800ef1c50e7880c0c9ef414cf",
        checkout_folder_name="subword-nmt",
    )
    subword_nmt_repo = subword_nmt_job.out_repository
    subword_nmt_repo.hash_overwrite = "I6_SUBWORD_NMT_V2"  # this is what most other people use as well
    
    bpe_lexicon = CreateBPELexiconJob(
        base_lexicon_path=get_bliss_lexicon(add_unknown_phoneme_and_mapping=False, add_silence=False),
        bpe_codes=bpe_vocab.codes,
        bpe_vocab=bpe_vocab.vocab,
        subword_nmt_repo=subword_nmt_repo,
        unk_label="<unk>",
    ).out_lexicon
    
    word_lexicon = BlissLexiconToG2PLexiconJob(
        bpe_lexicon,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon

    return word_lexicon


_Parts = ["train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other", "test-clean", "test-other"]

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
        train_subset: Optional[int] = None,
        train_ds_key: Optional[str] = None,
        pseudo_label_path: tk.Path = None,
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
        self.train_ds_key = train_ds_key
        self.pseudo_label_path = pseudo_label_path
        self.test_self_training_on_small_dataset = 0 # Old param, not used. Needed for compatibility.
        if train_epoch_wise_filter is NotSpecified:
            train_epoch_wise_filter = deepcopy(_default_train_epoch_wise_filter)
        if train_audio_preprocess is NotSpecified:
            if not audio:
                train_audio_preprocess = None
            elif train_audio_random_permute:
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
        self.train_subset = train_subset

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
            num_classes = self.vocab.get_num_classes()
            if isinstance(num_classes, int):
                self._classes_dim = Dim(num_classes, name="vocab", kind=Dim.Types.Feature)
            elif isinstance(num_classes, tk.Variable):
                self._classes_dim = _DelayedDim(num_classes, name="vocab", kind=Dim.Types.Feature)
            else:
                raise TypeError(f"unexpected type {type(num_classes)} for {num_classes}")

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
        if self.train_subset is None:
            state.pop("train_subset")
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
        if not self.train_ds_key:
            raise ValueError("train_ds_key not set")
        else:
            return self.get_dataset(self.train_ds_key, training=True, subset=self.train_subset)

    def get_train_dataset_for_forward(self) -> Dict[str, Any]:
        if not self.train_ds_key:
            raise ValueError("train_ds_key not set")
        else:
            return self.get_dataset(self.train_ds_key, subset=self.train_subset)
    
    def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
        ds = {
            "dev": self.get_dataset("dev", subset=self.eval_subset),
        }
        if not self.pseudo_label_path:
            ds["devtrain"] = self.get_dataset("train", subset=self.eval_subset)
        return ds

    def get_main_name(self) -> str:
        return self.main_key

    def get_main_dataset(self) -> Dict[str, Any]:
        assert self.main_key is not None, f"{self}: main_dataset not defined, main_key is None"
        return self.get_dataset(self.main_key)
    
    def get_sharded_main_dataset(self, shard_index: int, num_shards: int) -> Dict[str, Any]:
        assert self.main_key is not None, f"{self}: main_dataset not defined, main_key is None"
        assert 0 <= shard_index < num_shards, f"{self}: invalid shard_index 0 <= {shard_index} < {num_shards}"
        return self.get_dataset(self.main_key, sharding=(shard_index, num_shards))

    def get_dataset(self, key: str, *, training: bool = False, subset: Optional[int] = None, sharding: tuple[int, int] | None = None) -> Dict[str, Any]:
        files = []
        if key == "train-other-860":
            parts = ["train-clean-360", "train-other-500"]
        else:
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
            if self.train_ds_key == "train-clean-100":
                d["partition_epoch"] = 2
            elif self.train_ds_key == "train-other-860":
                d["partition_epoch"] = 18
            else:
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
            if sharding:
                d["_num_shards"] = sharding[1]
                d["_shard_index"] = sharding[0]
        if subset:
            if training:
                d["fixed_random_subset_seed"] = 1
            d["fixed_random_subset"] = subset  # faster
        
        # Combine pseudo labels into MetaDataset
        if training and self.pseudo_label_path:
            files_new = []
            for part in parts:
                files_new += [_get_librispeech_ogg_zip_dict_pseudo_labels(self.pseudo_label_path, part)[part]]
            d_pseudo = copy(d)
            d.pop("fixed_random_subset", None)
            d_pseudo["audio"] = None
            d_pseudo["path"] = files_new
            d_comb = {"zip_dataset": d, "pseudo_labels_dataset": d_pseudo}
            data_map = {
                "data": ("zip_dataset", "data"),
                "classes": ("pseudo_labels_dataset", "classes"),
            }
            d = MetaDataset(data_map, d_comb, "pseudo_labels_dataset").as_returnn_opts()
        return d


class _DelayedDim(DelayedBase):
    """
    TODO this is currently planned but not yet used...
    """

    def __init__(self, dimension: tk.Variable, **opts):
        # suppress init warning
        super().__init__(None)
        assert isinstance(dimension, DelayedBase)
        self.dimension = dimension
        self.opts = opts

    def get(self):
        from sisyphus.toolkit import running_in_worker
        from returnn.tensor import Dim

        assert running_in_worker(), "_DelayedDim: get() should only be called in worker"
        assert self.dimension.is_set(), f"_DelayedDim: dimension not set: {self.dimension}"
        dimension = instanciate_delayed(self.dimension)
        assert isinstance(
            dimension, int
        ), f"unexpected type {type(dimension)} for {dimension}, {self.dimension}, {self.dimension.get_path()}"
        return Dim(dimension, **instanciate_delayed(self.opts))


_raw_audio_opts = dict(
    features="raw",
    sample_rate=16_000,
    peak_normalization=True,
    preemphasis=None,
)


_librispeech_task_raw_v2_cache = {}

class TrainDatasetSel(Enum):
    train_100h = 1
    train_860h = 2
    train_960h = 3

def get_librispeech_task_raw_v2(
    *,
    vocab: Union[VocabConfig, str],
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    audio_opts: Optional[Dict[str, Any]] = None,
    audio_dim: int = 1,
    save_pseudo_labels: bool = False,
    ds_sel: TrainDatasetSel,
    with_prior: bool,
    **dataset_train_opts,
) -> tuple[Task, dict, Optional[LibrispeechOggZip]]:
    """
    Librispeech.

    Version 2:
    Use _bpe_to_words_v2 and _score_recog_out_v2 which does not use the Bliss corpus anymore directly,
    so it is easier to copy this setup to a new environment.
    """
    assert isinstance(ds_sel, TrainDatasetSel)
    
    vocab_ = vocab
    if isinstance(vocab, str):
        vocab = get_vocab_by_str(vocab, train_small=True if (ds_sel == TrainDatasetSel.train_100h or ds_sel == TrainDatasetSel.train_860h) else False)
        
    if ds_sel == TrainDatasetSel.train_860h:
        if dataset_train_opts:
            dataset_train_opts["train_epoch_wise_filter"] = None
        else:
            dataset_train_opts = dict(train_epoch_wise_filter=None)

    cache_key = make_hashable((LibrispeechOggZip, vocab, train_vocab_opts, audio_opts, audio_dim, save_pseudo_labels, ds_sel, with_prior, dataset_train_opts))
    if cache_key in _librispeech_task_raw_v2_cache:
        return _librispeech_task_raw_v2_cache[cache_key]

    if isinstance(vocab, Bpe):
        vocab_to_words = [_bpe_to_words_v2]
    elif isinstance(vocab, SentencePieceModel):
        vocab_to_words = [_spm_to_words]
    elif isinstance(vocab, (Utf8BytesVocab, VocabConfigStatic)):
        vocab_to_words = []  # assume it can just stay that way
    else:
        raise TypeError(f"unhandled vocab type {type(vocab)}")
    
    # Read out which datasets to use during training
    if ds_sel == TrainDatasetSel.train_100h:
        train_ds_key = "train-clean-100"
    elif ds_sel == TrainDatasetSel.train_860h:
        train_ds_key = "train-other-860" 
    else:
        train_ds_key = "train"

    audio_opts_ = _raw_audio_opts.copy()
    if audio_opts:
        audio_opts_.update(audio_opts)
    dataset_common_opts = dict(audio=audio_opts_, audio_dim=audio_dim, vocab=vocab)
    if train_vocab_opts:
        dataset_common_opts["train_vocab"] = vocab.copy(**train_vocab_opts)
    # We expect that all kwargs are only relevant for the training, thus we only pass them here.
    train_dataset = LibrispeechOggZip(**dataset_common_opts, **dataset_train_opts, train_ds_key=train_ds_key)
    _extract_audio_seq_len_file(train_dataset)
    _extract_text_seq_len_file(train_dataset, vocab_, name="target")
    eval_datasets = {
        "dev-clean": LibrispeechOggZip(**dataset_common_opts, main_key="dev-clean"),
        "dev-other": LibrispeechOggZip(**dataset_common_opts, main_key="dev-other"),
        "test-clean": LibrispeechOggZip(**dataset_common_opts, main_key="test-clean"),
        "test-other": LibrispeechOggZip(**dataset_common_opts, main_key="test-other"),
    }
    dev_dataset = eval_datasets["dev-other"]
    
    pseudo_labels_ds = {}
    train_100_ds = None
    if save_pseudo_labels:
        for ds_name in ["train-clean-360", "train-other-500"]:
            pseudo_labels_ds[ds_name] = LibrispeechOggZip(**dataset_common_opts, main_key=ds_name)
        train_100_ds = LibrispeechOggZip(**dataset_common_opts, main_key="train-clean-100")
            
    if with_prior:
        prior_dataset = LibrispeechOggZip(**dataset_common_opts, main_key=train_ds_key)
    else:
        prior_dataset = None

    task = Task(
        name="librispeech",
        train_dataset=train_dataset,
        train_epoch_split=train_dataset.train_epoch_split,
        dev_dataset=dev_dataset,
        eval_datasets=eval_datasets,
        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name="dev-other",
        score_recog_output_func=_score_recog_out_v2,
        prior_dataset=prior_dataset,
        recog_post_proc_funcs=vocab_to_words,
    )
    _librispeech_task_raw_v2_cache[cache_key] = (task, pseudo_labels_ds, train_100_ds)
    
    return task, pseudo_labels_ds, train_100_ds


def _extract_audio_seq_len_file(train_dataset: DatasetConfig):
    """
    Extract audio seq len file
    """
    from sisyphus import tk
    from i6_core.returnn.dataset import ExtractSeqLensJob

    ds_dict = train_dataset.get_train_dataset()
    # The code is semi-generic. But anyway double check for now. Later to be extended...
    if ds_dict["class"] == "MetaDataset":
        ds_dict = ds_dict["datasets"]["zip_dataset"]
    assert ds_dict["class"] in {"OggZipDataset", "LibriSpeechCorpus"}
    if ds_dict["audio"] is None:
        return None
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
    if ds_dict["class"] == "MetaDataset":
        ds_dict = ds_dict["datasets"]["pseudo_labels_dataset"]
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
