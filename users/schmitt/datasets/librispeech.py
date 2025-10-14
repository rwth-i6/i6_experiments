"""
Librispeech dataset
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Union, Tuple, Dict, Sequence
from copy import deepcopy, copy
import re
import os
from enum import Enum
from functools import cache
import numpy as np

from sisyphus import tk, Task as SisTask
from sisyphus.delayed_ops import DelayedBase
from i6_core.util import instanciate_delayed
from i6_core.corpus.convert import CorpusToTextDictJob
from i6_core.text.convert import TextDictToTextLinesJob
from i6_core.text.label.subword_nmt.train import ReturnnTrainBpeJob
from i6_core.text.label.sentencepiece.train import TrainSentencePieceJob, SentencePieceType
from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob
from i6_core.lib import corpus
from returnn.util.basic import NotSpecified
from returnn_common.datasets_old_2022_10.interface import DatasetConfig, VocabConfig, VocabConfigStatic
from i6_experiments.common.datasets import librispeech
from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.utils.basic import make_hashable
from i6_experiments.users.zeyer.speed_pert.librosa_09_10_11_kaiser_fast import (
    speed_pert_librosa_09_10_11_kaiser_fast as _default_train_audio_preprocess,
)
from i6_experiments.users.mueller.datasets.task import Task, MeasureType, RecogOutput, ScoreResult
from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from i6_experiments.users.zeyer.datasets.utils.bytes import Utf8BytesVocab
from i6_experiments.users.zeyer.datasets.utils.char import get_char_vocab

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim
    from i6_experiments.users.zeyer.collect_model_dataset_stats import StatisticsOutput

from i6_experiments.users.mueller.datasets.utils import CorpusReplaceOrthFromPyDictJob, get_ogg_zip_dict_pseudo_labels, \
    MetaDataset, GetAlignmentTargets, TargetsToHDF, CorpusToHDF, DummyHDF, ScoresHDF

_alias_prefix = "datasets/LibriSpeech/"


@cache
def _get_librispeech_ogg_zip_dict() -> Dict[str, tk.Path]:
    ogg_zip_dict = librispeech.get_ogg_zip_dict()

    from i6_core.returnn.oggzip import BlissToOggZipJob
    for name in ["_train-clean-100-short", "train-clean-100-max-10s", "train-1", "train-10min"]:
        # name = "_train-clean-100-short"
        ogg_zip_job = BlissToOggZipJob(
            _get_bliss_corpus_dict(None, name)[name],
            no_conversion=True,
        )
        ogg_zip_job.add_alias(os.path.join("datasets", "LibriSpeech", "%s_ogg_zip_job" % name))
        ogg_zip_dict[name] = ogg_zip_job.out_ogg_zip

    return ogg_zip_dict


@cache
def _get_bliss_corpus_dict(pseudo_labels_path: tk.Path, part: str) -> Dict[str, tk.Path]:
    # Get Bliss corpus. Same audio format as in ogg_zip, so already there anyway due to how we created the ogg_zip.
    # WARNING: Do not use these directly... It will keep another ogg copy of the audio...
    # However, these are used later in the scoring, so when changing them, make sure it's optional,
    # to not break hashes of old setups.

    from i6_experiments.users.schmitt.corpus.segment_ends import AugmentCorpusSegmentEndsJob
    from i6_experiments.users.schmitt.corpus.rename import RenameBlissCorpusJob
    from i6_experiments.users.schmitt.corpus.statistics import GetBlissCorpusStatisticsJob
    from i6_core.corpus.filter import FilterCorpusBySegmentDurationJob

    if pseudo_labels_path:
        assert part is not None
        # bliss_corpus_dict = librispeech.get_bliss_corpus_dict(audio_format="ogg")
        bliss_corpus_dict = _get_bliss_corpus_dict(None, part)
        # load pseudo labels and replace here
        bliss_corpus = bliss_corpus_dict[part]
        replace_job = CorpusReplaceOrthFromPyDictJob(bliss_corpus, pseudo_labels_path)
        replace_job.add_alias(
            os.path.join("datasets", "LibriSpeech-PseudoLabels", "%s_replace_orth" % part.replace('-', '_')))
        bliss_corpus = replace_job.out_corpus
        return {part: bliss_corpus}
    elif part == "_train-clean-100-short":
        augmented_corpus = AugmentCorpusSegmentEndsJob(
            bliss_corpous=librispeech.get_bliss_corpus_dict(audio_format="ogg")["train-clean-100"],
            oggzip_path=librispeech.get_ogg_zip_dict()["train-clean-100"],
            corpus_key="train-clean-100",
        ).out_bliss_corpus
        filtered_corpus = FilterCorpusBySegmentDurationJob(
            bliss_corpus=augmented_corpus,
            min_duration=0.0,
            max_duration=10.0,  # 10 seconds
        ).out_corpus
        # filtered_corpus = FilterCorpusBySegmentDurationJob(
        #     bliss_corpus=augmented_corpus,
        #     min_duration=0.0,
        #     max_duration=10.0,  # 10 seconds
        #     delete_empty_recordings=True
        # ).out_corpus
        # renamed_filtered_corpus = RenameBlissCorpusJob(
        #     bliss_corpus=filtered_corpus,
        #     new_name=part
        # ).out_corpus
        corpus_stats = GetBlissCorpusStatisticsJob(
            bliss_corpus=filtered_corpus,
        )
        tk.register_output(f"datasets/LibriSpeech/statistics/{part}_statistics.txt", corpus_stats.out_statistics)
        return {part: filtered_corpus}
    elif part == "train-clean-100-max-10s":
        augmented_corpus = AugmentCorpusSegmentEndsJob(
            bliss_corpous=librispeech.get_bliss_corpus_dict(audio_format="ogg")["train-clean-100"],
            oggzip_path=librispeech.get_ogg_zip_dict()["train-clean-100"],
            corpus_key="train-clean-100",
        ).out_bliss_corpus
        filtered_corpus = FilterCorpusBySegmentDurationJob(
            bliss_corpus=augmented_corpus,
            min_duration=0.0,
            max_duration=10.0,
            delete_empty_recordings=True
        ).out_corpus
        renamed_filtered_corpus = RenameBlissCorpusJob(
            bliss_corpus=filtered_corpus,
            new_name=part
        ).out_corpus
        corpus_stats = GetBlissCorpusStatisticsJob(
            bliss_corpus=filtered_corpus,
        )
        tk.register_output(f"datasets/LibriSpeech/statistics/{part}_statistics.txt", corpus_stats.out_statistics)
        return {part: renamed_filtered_corpus}
    elif part in ["train-1", "train-10min"]:
        if part == "train-1":
            num_seconds = 3600  # 1 hour
        else:
            num_seconds = 600  # 10 minutes

        augmented_corpora = {
            corpus_key: AugmentCorpusSegmentEndsJob(
                bliss_corpous=librispeech.get_bliss_corpus_dict(audio_format="ogg")[corpus_key],
                oggzip_path=librispeech.get_ogg_zip_dict()[corpus_key],
                corpus_key=corpus_key,
            ).out_bliss_corpus for corpus_key in ["train-clean-100", "train-clean-360", "train-other-500"]
        }
        filtered_corpus = GetLimitedResourceTrainingSetJob(
            num_seconds=num_seconds,
            clean100_bliss_corpus=augmented_corpora["train-clean-100"],
            clean360_bliss_corpus=augmented_corpora["train-clean-360"],
            other500_bliss_corpus=augmented_corpora["train-other-500"],
            delete_empty_recordings=True
        ).out_corpus
        renamed_filtered_corpus = RenameBlissCorpusJob(
            bliss_corpus=filtered_corpus,
            new_name=part
        ).out_corpus
        corpus_stats = GetBlissCorpusStatisticsJob(
            bliss_corpus=filtered_corpus,
        )
        corpus_stats.add_alias(f"datasets/LibriSpeech/statistics/{part}_statistics.txt")
        tk.register_output(corpus_stats.get_one_alias(), corpus_stats.out_statistics)
        return {part: renamed_filtered_corpus}
    else:
        bliss_corpus_dict =librispeech.get_bliss_corpus_dict(audio_format="ogg")
        corpus_stats = GetBlissCorpusStatisticsJob(
            bliss_corpus=bliss_corpus_dict[part],
        )
        corpus_stats.add_alias(f"datasets/LibriSpeech/statistics/{part}_statistics.txt")
        tk.register_output(corpus_stats.get_one_alias(), corpus_stats.out_statistics)
        return bliss_corpus_dict


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
        *, dim: Union[int, str], model_type: SentencePieceType = SentencePieceType.UNIGRAM, train_full: bool = False,
        train_small: bool = False
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
        training_text=get_librispeech_lm_combined_txt(train_small) if train_full else _get_train_corpus_text(
            train_small),
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
def get_vocab_by_str(vocab: str, train_small: bool = False) -> Union[
    SentencePieceModel, Bpe, VocabConfigStatic, Utf8BytesVocab]:
    """
    Get vocab
    """
    if re.match("^spm[0-9]+.*$", vocab):
        return _get_spm_vocab(dim=vocab[len("spm"):], model_type=SentencePieceType.UNIGRAM, train_small=train_small)
    elif re.match("^spmLm[0-9]+.*$", vocab):
        return _get_spm_vocab(dim=vocab[len("spmLm"):], model_type=SentencePieceType.UNIGRAM, train_full=True,
                              train_small=train_small)
    elif re.match("^spm_bpe[0-9]+.*$", vocab):
        return _get_spm_vocab(dim=vocab[len("spm_bpe"):], model_type=SentencePieceType.BPE, train_small=train_small)
    elif vocab == "bpe10k":  # predefined
        if train_small:
            raise ValueError(f"bpe10k not available for train_small")
        else:
            return bpe10k
    elif re.match("^bpe[0-9]+.*$", vocab):
        return _get_bpe_vocab(bpe_size=vocab[len("bpe"):], train_small=train_small)
    elif vocab == "char":
        return get_char_vocab(
            get_librispeech_lm_combined_txt(train_small=train_small), num_classes=29, extra_labels=("\x00",),
            eos_label="\x00"
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


_Parts = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "_train-clean-100-short",
    "train-clean-100-max-10s",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]

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
            forward_subset: Optional[int] = None,
            train_ds_key: Optional[str] = None,
            pseudo_label_path: tk.Path = None,
            pseudo_label_alignment: bool = -1,
            pseudo_label_nbest: int = 1,
            pseudo_label_scores: bool = False,
            pseudo_label_sentences: bool = False,
            keep_small_labels: bool = False,
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
        self.pseudo_label_alignment = pseudo_label_alignment
        self.pseudo_label_nbest = pseudo_label_nbest
        self.pseudo_label_sentences = pseudo_label_sentences
        self.keep_small_labels = keep_small_labels
        self.pseudo_label_scores = pseudo_label_scores
        self.test_self_training_on_small_dataset = 0  # Old param, not used. Needed for compatibility.
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
        self.forward_subset = forward_subset

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
        if self.forward_subset is None:
            state.pop("forward_subset")
        if self.pseudo_label_alignment == -1:
            state.pop("pseudo_label_alignment")
        if self.pseudo_label_nbest == 1:
            state.pop("pseudo_label_nbest")
        if not self.pseudo_label_scores:
            state.pop("pseudo_label_scores")
        if not self.pseudo_label_sentences:
            state.pop("pseudo_label_sentences")
        if not self.keep_small_labels:
            state.pop("keep_small_labels")
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

        if self.pseudo_label_alignment > 0:
            opts["targets_indices"] = {
                "dim_tags": [batch_dim, Dim(None, name="out-spatial-grad", kind=Dim.Types.Spatial),
                             Dim(self.pseudo_label_alignment, name="grad_best")],
                "sparse_dim": self._classes_dim,
            }
        nbest_dim = Dim(self.pseudo_label_nbest, name="pseudo_nbest")
        if self.pseudo_label_nbest > 1:
            opts["nbest_lengths"] = {
                "dim_tags": [batch_dim, nbest_dim],
            }
        if self.pseudo_label_scores:
            opts["scores"] = {
                "dim_tags": [batch_dim, nbest_dim],
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
        if self.eval_subset > 0:
            ds = {
                "dev": self.get_dataset("dev", subset=self.eval_subset),
            }
            if not self.pseudo_label_path:
                ds["devtrain"] = self.get_dataset("train", subset=self.eval_subset)
        else:
            ds = {}
        return ds

    def get_main_name(self) -> str:
        return self.main_key

    def get_main_dataset(self) -> Dict[str, Any]:
        assert self.main_key is not None, f"{self}: main_dataset not defined, main_key is None"
        return self.get_dataset(self.main_key, subset=self.forward_subset)

    def get_sharded_main_dataset(self, shard_index: int, num_shards: int) -> Dict[str, Any]:
        assert self.main_key is not None, f"{self}: main_dataset not defined, main_key is None"
        assert 0 <= shard_index < num_shards, f"{self}: invalid shard_index 0 <= {shard_index} < {num_shards}"
        return self.get_dataset(self.main_key, sharding=(shard_index, num_shards))

    def get_dataset(self, key: str, *, training: bool = False, subset: Optional[int] = None,
                    sharding: tuple[int, int] | None = None) -> Dict[str, Any]:
        files = []
        if key == "train-other-860":
            parts = ["train-clean-360", "train-other-500"]
        elif key == "train":
            parts = ["train-clean-100", "train-clean-360", "train-other-500"]
        elif key in ["dev", "test"]:
            parts = [part for part in _Parts if part.startswith(key)]
        else:
            parts = [key]
            # parts = [part for part in _Parts if part.startswith(key)]
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
            if self.train_ds_key in ["train-1", "train-10min"]:
                d["partition_epoch"] = 1
            elif self.train_ds_key in ["train-clean-100", "_train-clean-100-short", "train-clean-100-max-10s"]:
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
            if self.pseudo_label_alignment > -1:
                vocab = self.train_vocab if training and self.train_vocab else self.vocab
                assert vocab is not None
                for part in parts:
                    files_new += [GetAlignmentTargets(part, self.pseudo_label_path, vocab.vocab).out_file]
                d_pseudo = {
                    "class": "HDFDataset",
                    "files": files_new,
                    "use_cache_manager": True,
                }
                order_key = "zip_dataset"
            elif self.pseudo_label_nbest > 1:
                vocab = self.train_vocab if training and self.train_vocab else self.vocab
                assert vocab is not None
                for part in parts:
                    if part == "train-clean-100" and self.keep_small_labels:
                        train100 = librispeech.get_bliss_corpus_dict(audio_format="ogg")[part]
                        files_new += [CorpusToHDF(train100, vocab, self.pseudo_label_nbest).out_file]
                    else:
                        files_new += [TargetsToHDF(part, self.pseudo_label_path, vocab.vocab, self.pseudo_label_nbest,
                                                   vocab if self.pseudo_label_sentences else None).out_file]
                d_pseudo = {
                    "class": "HDFDataset",
                    "files": files_new,
                    "use_cache_manager": True,
                }
                order_key = "zip_dataset"
            else:
                for part in parts:
                    if part == "train-clean-100" and self.keep_small_labels:
                        files_new += [_get_librispeech_ogg_zip_dict()[part]]
                    else:
                        ogg_files = _get_librispeech_ogg_zip_dict_pseudo_labels(self.pseudo_label_path, part)
                        files_new += [ogg_files[part]]
                d_pseudo = copy(d)
                if not self.pseudo_label_scores:
                    d.pop("fixed_random_subset", None)
                d_pseudo["audio"] = None
                d_pseudo["path"] = files_new
                order_key = "pseudo_labels_dataset"
            d_comb = {"zip_dataset": d, "pseudo_labels_dataset": d_pseudo}
            data_map = {
                "data": ("zip_dataset", "data"),
                "classes": ("pseudo_labels_dataset", "classes"),
            }
            if self.pseudo_label_alignment > -1:
                data_map["classes"] = ("pseudo_labels_dataset", "data")
                if self.pseudo_label_alignment > 0:
                    data_map["targets_indices"] = ("pseudo_labels_dataset", "indices")
            if self.pseudo_label_nbest > 1:
                data_map["classes"] = ("pseudo_labels_dataset", "data")
                data_map["nbest_lengths"] = ("pseudo_labels_dataset", "lengths")
            if self.pseudo_label_scores:
                # We are decoding every step (NOTE: works only for max approx so far)
                score_files = []
                for part in parts:
                    if part == "train-clean-100" and self.keep_small_labels:
                        train100 = librispeech.get_bliss_corpus_dict(audio_format="ogg")[part]
                        score_files += [ScoresHDF(None, train100, None, 1).out_file]
                    else:
                        score_files += [ScoresHDF(part, None, self.pseudo_label_path, self.pseudo_label_nbest).out_file]
                d_scores = {
                    "class": "HDFDataset",
                    "files": score_files,
                    "use_cache_manager": True,
                }

                # def _get_scores_dataset(*, epoch: int, **_):
                #     import returnn.frontend as rf
                #     from returnn.config import get_global_config
                #     config = get_global_config()
                #     train_dataset_dict = config.typed_value("train")
                #     partition_epoch = train_dataset_dict["datasets"]["zip_dataset"]["partition_epoch"]

                #     if 1 <= epoch <= partition_epoch or not rf.get_run_ctx().train_flag:
                #         # return {
                #         #     "class": "AnythingDataset",
                #         #     "data_keys": {
                #         #         "data": {
                #         #             "dim": 1,
                #         #             "shape": (None,),
                #         #             "dtype": "int32",
                #         #         },
                #         #     }
                #         # }
                #         return d_scores
                #     else:
                #         n_finished_full_epochs = (epoch - 1) // partition_epoch
                #         hdf_files = [f"scores-epoch-{n_finished_full_epochs}.hdf"]
                #         return {
                #             "class": "HDFDataset",
                #             "files": hdf_files,
                #             "use_cache_manager": True,
                #         }

                # d_scores_variable = {
                #     "class": "VariableDataset",
                #     "get_dataset": _get_scores_dataset
                # }

                d_comb["scores_dataset"] = d_scores
                data_map["scores"] = ("scores_dataset", "data")

                if self.pseudo_label_alignment > -1 or self.pseudo_label_nbest > 1:
                    d_tmp = d_pseudo
                else:
                    d_tmp = d_pseudo.copy()
                    d_tmp.pop("partition_epoch")
                    d_tmp = MetaDataset({"data": ("pseudo_labels_dataset", "classes")},
                                        {"pseudo_labels_dataset": d_tmp}, "pseudo_labels_dataset").as_returnn_opts()
                d_comb["init_pseudo_labels_dataset"] = d_tmp

                # always_same_tags
                d_targets_variable = {
                    "class": "VariableDataset",
                    "get_dataset": _get_targets_dataset
                }

                d_comb["pseudo_labels_dataset"] = d_targets_variable
                data_map["classes"] = ("pseudo_labels_dataset", "data")
                order_key = "zip_dataset"
            d = MetaDataset(data_map, d_comb, order_key).as_returnn_opts()
        elif self.pseudo_label_nbest > 1 or self.pseudo_label_scores:
            d_comb = {"zip_dataset": d}
            data_map = {
                "data": ("zip_dataset", "data"),
                "classes": ("zip_dataset", "classes"),
            }
            order_key = "zip_dataset"
            if self.pseudo_label_nbest > 1:
                files_new = []
                for part in parts:
                    corp = librispeech.get_bliss_corpus_dict(audio_format="ogg")[part]
                    files_new += [DummyHDF(corp, self.pseudo_label_nbest).out_file]
                dummy_ds = {
                    "class": "HDFDataset",
                    "files": files_new,
                    "use_cache_manager": True,
                }
                d_comb["dummy"] = dummy_ds
                data_map["nbest_lengths"] = ("dummy", "lengths")
            if self.pseudo_label_scores:
                score_files = []
                for part in parts:
                    corp = librispeech.get_bliss_corpus_dict(audio_format="ogg")[part]
                    score_files += [ScoresHDF(None, corp, None, 1).out_file]
                d_scores = {
                    "class": "HDFDataset",
                    "files": score_files,
                    "use_cache_manager": True,
                }
                d_comb["scores_dataset"] = d_scores
                data_map["scores"] = ("scores_dataset", "data")
            d = MetaDataset(data_map, d_comb, order_key).as_returnn_opts()
        return d


# return {
#     "class": "AnythingDataset",
#     "data_keys": {
#         "data": {
#             "dim": 1,
#             "shape": (None,),
#             "dtype": "int32",
#         },
#     }
# }
def _get_targets_dataset(*, epoch: int, **_):
    import os
    import returnn.frontend as rf
    from returnn.config import get_global_config
    config = get_global_config()
    train_dataset_dict = config.typed_value("train")
    partition_epoch = train_dataset_dict["dataset"]["datasets"]["zip_dataset"]["partition_epoch"]

    if 1 <= epoch <= partition_epoch:
        print("load OLD")
        return train_dataset_dict["dataset"]["datasets"]["init_pseudo_labels_dataset"]
    else:
        n_finished_full_epochs = (epoch - 1) // partition_epoch
        print("load NEW epoch", n_finished_full_epochs)
        hdf_files = os.listdir(".")
        hdf_files = [f for f in hdf_files if
                     f.startswith(f"targets-epoch-{n_finished_full_epochs}-") and f.endswith(".hdf")]
        assert hdf_files, f"no target files found for epoch {n_finished_full_epochs}"
        return {
            "class": "HDFDataset",
            "files": hdf_files,
            "use_cache_manager": True,
        }


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
            train_sort_order: Optional[Any] = "laplace",
            train_sort_laplace_num_seqs: Optional[int] = 1000,
            eval_subset: Optional[int] = 3000,
    ):
        super().__init__()
        self.vocab = vocab
        self.train_vocab = train_vocab
        self.main_key = main_key
        self.train_epoch_split = train_epoch_split
        self.train_sort_order = train_sort_order
        self.train_sort_laplace_num_seqs = train_sort_laplace_num_seqs
        self.eval_subset = eval_subset

    def _sis_hash(self) -> bytes:
        import hashlib
        from sisyphus.hash import sis_hash_helper

        # Keep consistent once we do any changes.
        state = self.__dict__.copy()
        if not self.train_vocab:
            state.pop("train_vocab")  # backward compat
        if self.train_sort_order == "laplace":
            state.pop("train_sort_order")  # backward compat
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
        return self.get_dataset("full", training=True)

    def get_train_dataset_for_forward(self) -> Dict[str, Any]:
        return self.get_dataset("full")

    def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
        return {
            "dev": self.get_dataset("dev-other", subset=self.eval_subset),
            "devtrain": self.get_dataset("train-clean-100", subset=self.eval_subset),
        }

    def get_main_name(self) -> str:
        return self.main_key

    def get_main_dataset(self) -> Dict[str, Any]:
        return self.get_dataset(self.main_key)

    def get_dataset(self, key: str, *, training: bool = False, subset: Optional[int] = None) -> Dict[str, Any]:
        vocab = self.train_vocab if training and self.train_vocab else self.vocab
        if key == "full":
            from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data

            d: Dict[str, Any] = {
                "class": "LmDataset",
                "corpus_file": [get_librispeech_normalized_lm_data(), _get_train_corpus_text()],
                "use_cache_manager": True,
                "orth_vocab": vocab.get_opts().copy(),
                "seq_end_symbol": None,  # handled via orth_vocab
                "unknown_symbol": None,  # handled via orth_vocab
            }
        else:
            files = []
            parts = [part for part in _Parts if part.startswith(key)]
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
        if training:
            d["partition_epoch"] = self.train_epoch_split
            if self.train_sort_order == "laplace":
                if self.train_sort_laplace_num_seqs is not None:
                    d["seq_ordering"] = f"laplace:.{self.train_sort_laplace_num_seqs}"
                else:
                    d["seq_ordering"] = "random"
            else:
                d["seq_ordering"] = self.train_sort_order
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
    _train_clean_100_short = 4
    train_clean_100_max_10s = 5
    train_1h = 6
    train_10min = 7


def _is_char_vocab(vocab: VocabConfig) -> bool:
    if isinstance(vocab, Utf8BytesVocab):
        return True
    if isinstance(vocab, VocabConfigStatic):
        return vocab.opts.get("class") == "CharacterTargets"
    return False


def get_librispeech_task_raw_v2(
        *,
        vocab: Union[VocabConfig, str],
        train_vocab_opts: Optional[Dict[str, Any]] = None,
        audio_opts: Optional[Dict[str, Any]] = None,
        audio_dim: int = 1,
        save_pseudo_labels: Optional[TrainDatasetSel] = None,
        ds_sel: TrainDatasetSel,
        init_small: bool,
        with_prior: bool,
        empirical_prior: bool,
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
        vocab = get_vocab_by_str(vocab, train_small=init_small)

    cache_key = make_hashable(
        (LibrispeechOggZip, vocab, train_vocab_opts, audio_opts, audio_dim, save_pseudo_labels, ds_sel, init_small,
         with_prior, empirical_prior, dataset_train_opts))
    if cache_key in _librispeech_task_raw_v2_cache:
        return _librispeech_task_raw_v2_cache[cache_key]

    if isinstance(vocab, Bpe):
        vocab_to_words = [_bpe_to_words_v2]
    elif _is_char_vocab(vocab):
        vocab_to_words = [_char_to_words]
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
    elif ds_sel == TrainDatasetSel._train_clean_100_short:
        train_ds_key = "_train-clean-100-short"
    elif ds_sel == TrainDatasetSel.train_clean_100_max_10s:
        train_ds_key = "train-clean-100-max-10s"
    elif ds_sel == TrainDatasetSel.train_1h:
        train_ds_key = "train-1"
    elif ds_sel == TrainDatasetSel.train_10min:
        train_ds_key = "train-10min"
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
    if not dataset_train_opts or (
            dataset_train_opts and dataset_train_opts.get("pseudo_label_alignment") == -1 and dataset_train_opts.get(
            "pseudo_label_nbest") == 1):
        _extract_text_seq_len_file(train_dataset, vocab_, name="target")
    eval_datasets = {
        "dev-clean": LibrispeechOggZip(**dataset_common_opts, main_key="dev-clean"),
        "dev-other": LibrispeechOggZip(**dataset_common_opts, main_key="dev-other"),
        "test-clean": LibrispeechOggZip(**dataset_common_opts, main_key="test-clean"),
        "test-other": LibrispeechOggZip(**dataset_common_opts, main_key="test-other"),
        "_train-clean-100-short": LibrispeechOggZip(**dataset_common_opts, main_key="_train-clean-100-short"),
        # "train-clean-100-max-10s": LibrispeechOggZip(**dataset_common_opts, main_key="train-clean-100-max-10s"),
    }
    dev_dataset = eval_datasets["dev-other"]

    pseudo_labels_ds = {}
    train_100_ds = None
    if save_pseudo_labels is not None:
        if save_pseudo_labels == TrainDatasetSel.train_860h:
            ds_ls = ["train-clean-360", "train-other-500"]
        elif save_pseudo_labels == TrainDatasetSel._train_clean_100_short:
            ds_ls = ["_train-clean-100-short"]
        elif save_pseudo_labels == TrainDatasetSel.train_clean_100_max_10s:
            ds_ls = ["train-clean-100-max-10s"]
        else:
            ds_ls = ["train-clean-100", "train-clean-360", "train-other-500"]
        for ds_name in ds_ls:
            pseudo_labels_ds[ds_name] = LibrispeechOggZip(**dataset_common_opts, main_key=ds_name)
        train_100_ds = LibrispeechOggZip(**dataset_common_opts, main_key="train-clean-100")

    if with_prior:
        if empirical_prior:
            prior_dataset = LibrispeechOggZip(**dataset_common_opts, main_key="train")
        else:
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


def _extract_audio_seq_len_file(train_dataset: DatasetConfig, *, use_main_ds: bool = False):
    """
    Extract audio seq len file
    """
    from sisyphus import tk
    from i6_core.returnn.dataset import ExtractSeqLensJob

    if use_main_ds:
        ds_dict = train_dataset.get_main_dataset()
    else:
        ds_dict = train_dataset.get_train_dataset()
    # The code is semi-generic. But anyway double check for now. Later to be extended...
    if ds_dict["class"] == "MetaDataset":
        ds_dict = ds_dict["datasets"]["zip_dataset"]
    assert ds_dict["class"] in {"OggZipDataset", "LibriSpeechCorpus"}
    if ds_dict["audio"] is None:
        return None
    ds_dict.pop("partition_epoch", None)
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


def _extract_text_seq_len_file(train_dataset: DatasetConfig, vocab_cfg: Union[str, VocabConfig], *, name: str,
                               use_main_ds: bool = False):
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

    if use_main_ds:
        ds_dict = train_dataset.get_main_dataset()
    else:
        ds_dict = train_dataset.get_train_dataset()
    if ds_dict["class"] == "MetaDataset":
        ds_dict = ds_dict["datasets"]["pseudo_labels_dataset"]
    # The code is semi-generic. But anyway double check for now. Later to be extended...
    if ds_dict["class"] == "VariableDataset":
        return
    assert ds_dict["class"] in {"OggZipDataset", "LibriSpeechCorpus", "LmDataset"}
    vocab_key = "targets" if ds_dict["class"] in {"OggZipDataset", "LibriSpeechCorpus"} else "orth_vocab"
    ds_dict.pop("partition_epoch", None)
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


def _char_to_words(bpe: RecogOutput) -> RecogOutput:
    """Char to words"""
    from i6_core.returnn.search import SearchOutputRawReplaceJob

    # Note, our standard search uses :func:`_returnn_v2_get_forward_callback`,
    # and that uses ``hyp_serialized = hyps.sparse_dim.vocab.get_seq_labels(hyp_ids)``.
    # Utf8ByteTargets/CharacterTargets would output non-white-space delimited labels
    # (CharacterTargets.get_seq_labels: ``"".join(map(self._labels.__getitem__, seq))``).
    # However, most of the CTC models then do sth like this:
    #   vocab_labels = list(target_dim.vocab.labels) + [model_recog.output_blank_label]
    #   wb_target_dim.vocab = Vocabulary.create_vocab_from_labels(
    #      vocab_labels, user_defined_symbols={model_recog.output_blank_label: blank_idx})
    # And that create_vocab_from_labels has some special logic,
    # but with output_blank_label = "<blank>", i.e. len(output_blank_label) > 1,
    # this will result in a static Vocabulary where get_seq_labels is ``" ".join(map(labels.__getitem__, seq))``,
    # i.e. white-space delimited.
    # utf8/char, after SearchRemoveLabelJob, produces: "H I S  A B O D E  W H I C H  H E  H A D  F I X E D ..."
    # This is somewhat an artefact of the processing because it assumed white-space separated words,
    # and it used txt.split(" ") in SearchCollapseRepeatedLabelsJob and SearchRemoveLabelJob.
    # So any whitespace labels in the search output stays as two spaces.
    # That's why we can just do the SearchOutputRawReplaceJob below.
    # If we have do deal with non-white-space delimited outputs at some point (might occur with AED models?),
    # some solutions:
    # - In _returnn_v2_get_forward_callback, maybe don't use the vocab-dependend get_seq_labels,
    #   but just always output it white-space delimited.
    #   We can make this optional, and only apply for AED models (?),
    #   such that this does not break all existing hashes.
    # - Maybe we can handle it here? But it might need some further modifications...
    words = SearchOutputRawReplaceJob(
        bpe.output, [("  ", "▁"), (" ", ""), ("▁", " ")], output_gzip=True
    ).out_search_results
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


class GetLimitedResourceTrainingSetJob(tk.Job):
    def __init__(
            self,
            num_seconds: float,
            clean100_bliss_corpus: tk.Path,
            clean360_bliss_corpus: tk.Path,
            other500_bliss_corpus: tk.Path,
            delete_empty_recordings: bool = True,
            random_seed: int = 42,
    ):
        self.num_seconds = num_seconds
        self.clean100_bliss_corpus = clean100_bliss_corpus
        self.clean360_bliss_corpus = clean360_bliss_corpus
        self.other500_bliss_corpus = other500_bliss_corpus
        self.delete_empty_recordings = delete_empty_recordings
        self.random_seed = random_seed

        self.out_corpus = self.output_path("corpus.xml.gz", cached=True)
        if self.delete_empty_recordings:
            self.out_removed_recordings = self.output_path("removed_recordings.log")

    def tasks(self):
        yield SisTask("run", rqmt={"cpu": 4, "mem": 16, "time": 4, "gpu": 0})

    def run(self):
        import random

        clean100_corpus = corpus.Corpus()
        clean100_corpus.load(self.clean100_bliss_corpus.get_path())
        clean360_corpus = corpus.Corpus()
        clean360_corpus.load(self.clean360_bliss_corpus.get_path())
        other500_corpus = corpus.Corpus()
        other500_corpus.load(self.other500_bliss_corpus.get_path())
        clean_recordings = []
        other_recordings = []
        num_clean_recordings = 0
        num_other_recordings = 0

        for r in clean100_corpus.recordings + clean360_corpus.recordings:
            assert len(r.segments) == 1, "expected one segment per recording"
            clean_recordings.append(r)
            num_clean_recordings += 1

        for r in other500_corpus.recordings:
            assert len(r.segments) == 1, "expected one segment per recording"
            other_recordings.append(r)
            num_other_recordings += 1

        selected_recordings = []
        selected_recordings_duration = 0.0
        clean_recordings_indices = list(range(num_clean_recordings))
        other_recordings_indices = list(range(num_other_recordings))

        rand = random.Random(self.random_seed)
        rand.shuffle(clean_recordings_indices)
        rand.shuffle(other_recordings_indices)

        i = 0
        # alternate between clean and other recordings until we reach the target duration
        while selected_recordings_duration <= self.num_seconds:
            i_ = i // 2
            if i % 2 == 0 and (i_ < num_clean_recordings):
                r = clean_recordings[clean_recordings_indices[i_]]
            elif i % 2 == 1 and (i_ < num_other_recordings):
                r = other_recordings[other_recordings_indices[i_]]
            else:
                raise ValueError("not enough data to select from")
            selected_recordings.append(r)
            selected_recordings_duration += r.segments[0].end - r.segments[0].start
            i += 1
        print(f"selected {len(selected_recordings)} segments, total duration {selected_recordings_duration:.1f}s")

        new_corpus = corpus.Corpus()
        new_corpus.recordings = selected_recordings
        new_corpus.dump(self.out_corpus.get_path())
