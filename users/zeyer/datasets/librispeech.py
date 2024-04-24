"""
Librispeech dataset
"""

from __future__ import annotations
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
from .task import Task, MeasureType, RecogOutput, ScoreResult
from .utils.bpe import Bpe
from .utils.spm import SentencePieceModel


librispeech_ogg_zip_dict = librispeech.get_ogg_zip_dict()

# $ ls -la /u/zeyer/setups/librispeech/dataset/tars/
# -rw-r--r-- 1 zeyer assi   360977013 Feb 26  2018 dev-clean.zip
# -rw-r--r-- 1 zeyer assi   338709788 Feb 26  2018 dev-other.zip
# -rw-r--r-- 1 zeyer assi        1024 Feb 27  2018 .history.zeyer
# -rw-r--r-- 1 zeyer assi   369096021 Feb 26  2018 test-clean.zip
# -rw-r--r-- 1 zeyer assi   353841318 Feb 26  2018 test-other.zip
# -rw-r--r-- 1 zeyer assi  6625963133 Feb 26  2018 train-clean-100.zip
# -rw-r--r-- 1 zeyer assi 23919296392 Feb 26  2018 train-clean-360.zip
# -rw-r--r-- 1 zeyer assi 31839925140 Feb 26  2018 train-other-500.zip
librispeech_tars_zip_base_path = tk.Path(
    "/u/zeyer/setups/librispeech/dataset/tars", hash_overwrite="Librispeech-tars-zip-base-path"
)

# Get Bliss corpus. Same audio format as in ogg_zip, so already there anyway due to how we created the ogg_zip.
# WARNING: Do not use these directly... It will keep another ogg copy of the audio...
# Note: These are used later in the scoring, so when changing them, make sure it's optional,
# to not break hashes of old setups.
_bliss_corpus_dict = librispeech.get_bliss_corpus_dict(audio_format="ogg")  # TODO bad deps...
_bliss_train_corpus = _bliss_corpus_dict["train-other-960"]  # TODO bad deps...

# TODO change this here... we can change it as no code is currently using it
_train_corpus_text = CorpusToTxtJob(_bliss_train_corpus, gzip=False).out_txt  # TODO...

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
_spm_2k = _spm_train_job.out_model  # TODO bad deps...

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
        super(LibrispeechOggZip, self).__init__()
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
            files += [librispeech_ogg_zip_dict[part]]
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
                assert eos_id is not None, f"{self}: vocab {self.vocab} does not define EOS"
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


class LibrispeechOldFlacTarZip(DatasetConfig):
    """
    Librispeech dataset using the old LibriSpeechCorpus RETURNN dataset with use_zip=True,
    i.e. the original tar files repacked into zip files,
    i.e. keeping the original flac files inside the zip files.
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
            "path": librispeech_tars_zip_base_path,
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


def get_librispeech_task_spm2k() -> Task:
    """
    Librispeech
    """
    # TODO ...


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
        vocab_to_words = _bpe_to_words
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
        name="swb_bpe1k",
        train_dataset=train_dataset,
        train_epoch_split=train_dataset.train_epoch_split,
        dev_dataset=dev_dataset,
        eval_datasets=eval_datasets,
        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name="dev-other",
        score_recog_output_func=_score_recog_out,
        recog_post_proc_funcs=[vocab_to_words],
    )


def get_librispeech_task_bpe10k_raw(**dataset_train_opts) -> Task:
    return get_librispeech_task_raw(vocab=bpe10k, **dataset_train_opts)


def _bpe_to_words(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    from i6_core.returnn.search import SearchBPEtoWordsJob

    words = SearchBPEtoWordsJob(bpe.output, output_gzip=True).out_word_search_results
    return RecogOutput(output=words)


def _spm_to_words(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    from i6_experiments.users.zeyer.returnn.search import SearchOutputRawReplaceJob  # TODO move to i6_core

    words = SearchOutputRawReplaceJob(bpe.output, [(" ", ""), ("▁", " ")], output_gzip=True).out_search_results
    return RecogOutput(output=words)


def _score_recog_out(dataset: DatasetConfig, recog_output: RecogOutput) -> ScoreResult:
    """score"""
    # We use sclite now.
    # Could also use ReturnnComputeWERJob.
    from i6_core.returnn.search import SearchWordsToCTMJob
    from i6_core.corpus.convert import CorpusToStmJob
    from i6_core.recognition.scoring import ScliteJob

    hyp_words = recog_output.output
    corpus_name = dataset.get_main_name()

    recognition_bliss_corpus = _bliss_corpus_dict[corpus_name]

    search_ctm = SearchWordsToCTMJob(
        recog_words_file=hyp_words,
        bliss_corpus=recognition_bliss_corpus,
    ).out_ctm_file

    stm_file = CorpusToStmJob(bliss_corpus=recognition_bliss_corpus).out_stm_path

    score_job = ScliteJob(
        ref=stm_file, hyp=search_ctm, sctk_binary_path=tools_paths.get_sctk_binary_path(), precision_ndigit=2
    )

    return ScoreResult(dataset_name=corpus_name, main_measure_value=score_job.out_wer, report=score_job.out_report_dir)
