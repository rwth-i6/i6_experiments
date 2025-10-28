import copy
from dataclasses import dataclass
from functools import cache
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from returnn_common.datasets_old_2022_10.interface import DatasetConfig, VocabConfig, VocabConfigStatic
from sisyphus import Job, Path, tk
from sisyphus.delayed_ops import DelayedFormat
from sisyphus.task import Task

from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.corpus.convert import CorpusToTextDictJob
from i6_core.text.convert import TextDictToTextLinesJob
from i6_core.returnn.config import CodeWrapper
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.util import uopen

from .corpus_lex import Corpora, dev_corpora_def, test_corpora_def
DEV_KEYS = [f"{k}.{vi}" for k,v in dev_corpora_def.items() for vi in v]
TEST_KEYS = [f"{k}.{vi}" for k,v in test_corpora_def.items() for vi in v]

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

LM_DATA_PATH = "/nas/models/asr/hzhang/setups/2025-07-20--combined/data/ES/lm_text.gz"
default_train_epoch_split = 20


# TODO: LM dataset should be independent of frequency, consider move it separate
class SpainishLmDataset(DatasetConfig):
    """
    Spainish LM dataset
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
        byte_list = [b"SpainishLmDataset", sis_hash_helper(state)]

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
            "dev": self.get_dataset("dev", subset=self.eval_subset),
            "test": self.get_dataset("test", subset=self.eval_subset),
        }

    def get_main_name(self) -> str:
        return self.main_key

    def get_main_dataset(self) -> Dict[str, Any]:
        return self.get_dataset(self.main_key)

    def get_dataset(self, key: str, *, training: bool = False, subset: Optional[int] = None) -> Dict[str, Any]:
        vocab = self.train_vocab if training and self.train_vocab else self.vocab
        if key == "train":
            d: Dict[str, Any] = {
                "class": "LmDataset",
                "corpus_file": [tk.Path(LM_DATA_PATH)], # A zipped file, named out_text
                "use_cache_manager": True,
                "orth_vocab": vocab.get_opts().copy(),
                "seq_end_symbol": None,  # handled via orth_vocab
                "unknown_symbol": None,  # handled via orth_vocab
            }
        # elif key == "dev":
        #     d: Dict[str, Any] = {
        #         "class": "OggZipDataset",
        #         "path": _get_lm_eval_ogg_zip(key=key, alias_prefix=f"Spainish_LM_dataset_{key}"),
        #         "use_cache_manager": True,
        #         "audio": None,
        #         "targets": vocab.get_opts().copy(),
        #     }
        # elif key == "test":
        #     d: Dict[str, Any] = {
        #         "class": "OggZipDataset",
        #         "path": _get_lm_eval_ogg_zip(key=key, alias_prefix=f"Spainish_LM_dataset_{key}"),
        #         "use_cache_manager": True,
        #         "audio": None,
        #         "targets": vocab.get_opts().copy(),
        #     }
        else:
            assert key in (DEV_KEYS + TEST_KEYS + ["test", "dev"]), f"invalid eval/dev key {key!r}"
            d: Dict[str, Any] = {
                "class": "OggZipDataset",
                "path": _get_lm_eval_ogg_zip(key=key, alias_prefix=f"Spainish_LM_dataset_{key}"),
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

class SpainishLmEvalDataset(DatasetConfig):
    """
    Spainish LM dataset
    """

    def __init__(
        self,
        *,
        vocab: VocabConfig,
        main_key: Optional[str] = "test",
        eval_subset: Optional[int] = 3000,
    ):
        super().__init__()
        self.vocab = vocab
        self.main_key = main_key
        self.eval_subset = eval_subset

    def _sis_hash(self) -> bytes:
        import hashlib
        from sisyphus.hash import sis_hash_helper

        # Keep consistent once we do any changes.
        state = self.__dict__.copy()
        byte_list = [b"SpainishLmEvalDataset", sis_hash_helper(state)]

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
        pass

    def get_train_dataset_for_forward(self) -> Dict[str, Any]:
        pass

    def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
        return {
            "dev": self.get_dataset("dev", subset=self.eval_subset),
            "test": self.get_dataset("test", subset=self.eval_subset),
        }

    def get_main_name(self) -> str:
        return self.main_key

    def get_main_dataset(self) -> Dict[str, Any]:
        return self.get_dataset(self.main_key)

    def get_dataset(self, key: str, *, training: bool = False, subset: Optional[int] = None) -> Dict[str, Any]:
        vocab = self.train_vocab if training and self.train_vocab else self.vocab
        assert key in (DEV_KEYS + TEST_KEYS + ["test", "dev"]), f"invalid eval/dev key {key!r}"
        d: Dict[str, Any] = {
            "class": "OggZipDataset",
            "path": _get_lm_eval_ogg_zip(key=key, alias_prefix=f"Spainish_LM_dataset_{key}"),
            "use_cache_manager": True,
            "audio": None,
            "targets": vocab.get_opts().copy(),
        }

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
                "dtype": "int16"
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
                #"num_feature_filters": 1,
                "peak_normalization": False,
                "preemphasis": None,
                "sample_rate": 16_000,
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
                "dtype": "int16"
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
                #"num_feature_filters": 1,
                "peak_normalization": False,
                "preemphasis": None,
                "sample_rate": 16_000,
            },
            "path": CodeWrapper(DelayedFormat('CachedFile("{file}")', file=self.oggzip)),
            "partition_epoch": 1,
            "seq_ordering": "sorted_reverse",
            "targets": self.spm.get_opts(),
            "fixed_random_seed": 1,
        }

AVAILABLE_NBEST = {
    "mtp_dev_heldout-v2": "/nas/models/asr/am/ES/16kHz/20250423-hwu-mbw-ctc-conformer/work/i6_core/recognition/advanced_tree_search/AdvancedTreeSearchJob.1aMajLp4pxh3/output/recognition.res.1",
    "common_voice_two_speakers-v1": "/nas/models/asr/am/ES/16kHz/20250423-hwu-mbw-ctc-conformer/work/i6_core/recognition/advanced_tree_search/AdvancedTreeSearchJob.B9h0VCvQDrKL/output/recognition.res.1",
    "dev_conversation": "/nas/models/asr/am/ES/16kHz/20250423-hwu-mbw-ctc-conformer/work/i6_core/recognition/advanced_tree_search/AdvancedTreeSearchJob.GK76328ZVDPE/output/recognition.res.1",
    "eval_voice_call-v2": "/nas/models/asr/am/ES/16kHz/20250423-hwu-mbw-ctc-conformer/work/i6_core/recognition/advanced_tree_search/AdvancedTreeSearchJob.TZlV2LDMj9YH/output/recognition.res.1",
    "movies_tvshows": "/nas/models/asr/am/ES/16kHz/20250423-hwu-mbw-ctc-conformer/work/i6_core/recognition/advanced_tree_search/AdvancedTreeSearchJob.meNIq27euq1T/output/recognition.res.1",
    "napoli": "/nas/models/asr/am/ES/16kHz/20250423-hwu-mbw-ctc-conformer/work/i6_core/recognition/advanced_tree_search/AdvancedTreeSearchJob.9IqWQf5Qhj5r/output/recognition.res.1",
    "eval_callcenter_lt": "/nas/models/asr/am/ES/16kHz/20250423-hwu-mbw-ctc-conformer/work/i6_core/recognition/advanced_tree_search/AdvancedTreeSearchJob.hxnOkkCauvRW/output/recognition.res.1",
    "mtp_eval_p1": "/nas/models/asr/am/ES/16kHz/20250423-hwu-mbw-ctc-conformer/work/i6_core/recognition/advanced_tree_search/AdvancedTreeSearchJob.yqKcreqI1KtW/output/recognition.res.1",
    "mtp_eval_p2": "/nas/models/asr/am/ES/16kHz/20250423-hwu-mbw-ctc-conformer/work/i6_core/recognition/advanced_tree_search/AdvancedTreeSearchJob.tImCQc0pjkhE/output/recognition.res.1",
    "mtp_eval_p3": "/nas/models/asr/am/ES/16kHz/20250423-hwu-mbw-ctc-conformer/work/i6_core/recognition/advanced_tree_search/AdvancedTreeSearchJob.NhRrBALsTOMr/output/recognition.res.1",
    "mtp_eval_p4": "/nas/models/asr/am/ES/16kHz/20250423-hwu-mbw-ctc-conformer/work/i6_core/recognition/advanced_tree_search/AdvancedTreeSearchJob.1s67YARvbumn/output/recognition.res.1",
}
class NbestListDataset(DatasetConfig):
    def __init__(self, spm: SentencePieceModel, replace_list: List[Tuple[str,str]] = None):
        self.spm = spm
        self.replace_list = replace_list

    def _get_match_Nbest(self, name: str, N: int = 80):
        from i6_experiments.users.zhang.experiments.apptek.datasets.tools import ConvertNbestTextToDictJob
        assert N == 80, f"Only have N=80 for now, given is N={N}"
        for key, path in AVAILABLE_NBEST.items():
            if key in name:
                return ConvertNbestTextToDictJob(in_text=tk.Path(path), nbest_size=N, replace_list=self.replace_list).out_nbest_dict
        raise ValueError(f"Nbest key {name} not found in AVAILABLE_NBEST")
    def get_dataset(self, key: str, *, tokenize: bool = True, N: int = 80) -> Tuple[tk.Path,tk.Path]:
        assert "ref" in key, f"Only have Nbest for ref.seg for now, given: {key}"
        from i6_experiments.users.zhang.datasets.vocab import ApplySentencepieceToWordOutputJob
        Nbest = self._get_match_Nbest(key, N)
        from i6_core.returnn.search import SearchRemoveLabelJob
        Nbest_for_lm = SearchRemoveLabelJob(Nbest, remove_label={"<unk>","<sep>","▁mes"}).out_search_results
        if tokenize:
            Nbest = ApplySentencepieceToWordOutputJob(search_py_output=Nbest,sentencepiece_model=self.spm.model_file,enable_unk=True).out_search_results
            Nbest_for_lm = ApplySentencepieceToWordOutputJob(search_py_output=Nbest_for_lm,sentencepiece_model=self.spm.model_file,enable_unk=True).out_search_results
        return Nbest, Nbest_for_lm

NEED_FIX_OGG_ZIP_DATASET_NAME = ["test_set.ES_ES.f16kHz.eval_voice_call-v2",
                                 "test_set.ES_ES.f16kHz.eval_napoli_202210-v3",
                                 "test_set.ES_ES.f16kHz.eval_voice_call-v3",
                                 "test_set.ES_ES.f8kHz.mtp_eval-v2",
                                 "test_set.ES.f8kHz.mtp_dev_heldout-v2.aptk_leg.ff_wer"]
from i6_experiments.users.zhang.experiments.apptek.datasets.tools import OggZipFixTxtTextualJob, BlissStripOrthPunctJob
@cache
def _get_ogg_zip(
    corpus: tk.Path, name: str, split: int, returnn_root: Union[str, tk.Path], alias_prefix: str,
    keep_training_hash: bool = False,
) -> tk.Path:
    segment_job = SegmentCorpusJob(corpus, split)
    if (any(infix in name for infix in ["dev_conversation","common_voice_two_speakers"])  # Actually this should be done right after the creation of corpus
            and not keep_training_hash):
        corpus = BlissStripOrthPunctJob(corpus).out_corpus
        if "common_voice_two_speakers" in name:
            from i6_experiments.users.zhang.experiments.apptek.datasets.tools import FixInfEndInBlissJob
            corpus = FixInfEndInBlissJob(in_corpus=corpus).out_corpus
    oggzip_job = BlissToOggZipJob(corpus, segments=segment_job.out_segment_path, returnn_root=returnn_root)
    oggzip_job.rqmt = {"cpu": 1, "mem": 2}
    oggzip_job.merge_rqmt = None  # merge on local machine, to be more robust against slowness due to slow FS
    #print(name)
    if any(name_prefix in name for name_prefix in NEED_FIX_OGG_ZIP_DATASET_NAME):
        oggzip_job = OggZipFixTxtTextualJob(oggzip_job.out_ogg_zip)
    oggzip_job.add_alias(f"{alias_prefix}/oggzip{'_old' if keep_training_hash else ''}/{name}")
    tk.register_output(f"{alias_prefix}/oggzip{'_old' if keep_training_hash else ''}/{name}.ogg.zip", oggzip_job.out_ogg_zip)
    return oggzip_job.out_ogg_zip

def get_spm_lexicon() -> tk.Path:
    """
    Create SPM lexicon without unknown and silence

    :return: path to a lexicon bliss xml file
    """
    #from i6_core.tools.git import CloneGitRepositoryJob
    #from i6_core.lexicon.bpe import CreateBPELexiconJob
    from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
    #from i6_experiments.common.datasets.librispeech import get_bliss_lexicon

    # subword_nmt_job = CloneGitRepositoryJob(
    #     url="https://github.com/rwth-i6/subword-nmt",
    #     commit="5015a45e28a958f800ef1c50e7880c0c9ef414cf",
    #     checkout_folder_name="subword-nmt",
    # )
    # subword_nmt_repo = subword_nmt_job.out_repository
    # subword_nmt_repo.hash_overwrite = "I6_SUBWORD_NMT_V2"  # this is what most other people use as well

    # bpe_lexicon = CreateBPELexiconJob(
    #     base_lexicon_path=get_bliss_lexicon(add_unknown_phoneme_and_mapping=False, add_silence=False),
    #     bpe_codes=bpe_vocab.codes,
    #     bpe_vocab=bpe_vocab.vocab,
    #     subword_nmt_repo=subword_nmt_repo,
    #     unk_label="<unk>",
    # ).out_lexicon

    word_lexicon = BlissLexiconToG2PLexiconJob(
        tk.Path("/nas/models/asr/artefacts/lex/ES/20250717-tel-spm-recognition_lex-v1/recognition.lex.v1.xml.gz"),
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon

    return word_lexicon

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

def _get_lm_eval_ogg_zip( # any data set inside "dev" or "test" will affect the hash for training Job!
    *,
    returnn_root: Union[str, tk.Path] = tk.Path("/home/mgunz/setups/2024-07-08--zeyer-setup-apptek/recipe/returnn"),
    alias_prefix: str,
    key: str,
) -> List[Path]:  #
    from .corpus_lex import get_corpora
    lm_corpora = get_corpora(for_lm=True)
    corpora = get_corpora()
    if key == "dev": # During Training
        ogg_zip_files = []
        for k, eval_info in lm_corpora.dev.items():
            if str(eval_info.segmenter_type) == "ref":
                ogg_zip_files.append(_get_ogg_zip(
                    eval_info.segmented_corpus,
                    name=f"seg.{eval_info.segmenter_type}/{k}",
                    split=10,
                    returnn_root=returnn_root,
                    alias_prefix=alias_prefix,
                    keep_training_hash=True,
                )
                )
    elif key == "test": # During Training
        ogg_zip_files = []
        for k, eval_info in lm_corpora.test.items():
            if str(eval_info.segmenter_type) == "ref":
                ogg_zip_files.append(_get_ogg_zip(
                    eval_info.segmented_corpus,
                    name=f"seg.{eval_info.segmenter_type}/{k}",
                    split=10,
                    returnn_root=returnn_root,
                    alias_prefix=alias_prefix,
                    keep_training_hash=True,
                )
                )
    else:
        ogg_zip_files = []
        if "dev" in key:
            for k, eval_info in corpora.dev.items():
                if key in k and str(eval_info.segmenter_type) == "ref":
                    ogg_zip_files.append(_get_ogg_zip(
                        eval_info.segmented_corpus,
                        name=f"seg.{eval_info.segmenter_type}/{k}",
                        split=10,
                        returnn_root=returnn_root,
                        alias_prefix=alias_prefix,
                    )
                    )
        elif "eval" in key or "mbw" in key:
            for k, eval_info in corpora.test.items():
                if key in k and str(eval_info.segmenter_type) == "ref":
                    ogg_zip_files.append(_get_ogg_zip(
                        eval_info.segmented_corpus,
                        name=f"seg.{eval_info.segmenter_type}/{k}",
                        split=10,
                        returnn_root=returnn_root,
                        alias_prefix=alias_prefix,
                    )
                    )
        else:
            raise NotImplementedError(f"Can not determine which part {key} is belonging to (dev/eval)")
        assert ogg_zip_files, f"No data find for {key}"
    return ogg_zip_files

_alias_prefix = "datasets/Spainish/"

@cache
def _get_corpus_text_dict(corpus_file: tk.Path, key: str) -> tk.Path:
    job = CorpusToTextDictJob(corpus_file, gzip=True)
    job.add_alias(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_dict")
    tk.register_output(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_dict.py.gz", job.out_dictionary)
    return job.out_dictionary

@cache
def _get_corpus_text(corpus_file: tk.Path, key: str) -> tk.Path:
    if any(infix in key for infix in ["dev_conversation",
                                    "common_voice_two_speakers"]):  # Actually this should be done right after the creation of corpus
        corpus_file = BlissStripOrthPunctJob(corpus_file).out_corpus
    corpus_text_dict = _get_corpus_text_dict(corpus_file, key)
    job = TextDictToTextLinesJob(corpus_text_dict, gzip=True)
    job.add_alias(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_lines")
    tk.register_output(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_lines.txt.gz", job.out_text_lines)
    return job.out_text_lines

@cache
def get_corpus_text_dict(key: str) -> tk.Path:
    from .corpus_lex import get_corpora
    corpora = get_corpora()
    if "dev" in key:
        for k, eval_info in corpora.dev.items():
            if key in k and str(eval_info.segmenter_type) == "ref":
                return _get_corpus_text_dict(eval_info.segmented_corpus, k)

    elif "eval" in key or "mbw" in key:
        for k, eval_info in corpora.test.items():
            if key in k and str(eval_info.segmenter_type) == "ref":
                corpus = eval_info.segmented_corpus
                if any(infix in k for infix in ["dev_conversation",
                                                   "common_voice_two_speakers"]):  # Actually this should be done right after the creation of corpus
                    corpus = BlissStripOrthPunctJob(corpus).out_corpus
                return _get_corpus_text_dict(corpus, k)
    else:
        raise NotImplementedError(f"Can not determine which part {key} is belonging to (dev/eval)")
    assert False, f"No data find for {key}"


@cache
def get_lm_eval_text(
    *,
    key: str,
) -> Path:  #
    from .corpus_lex import get_corpora
    from i6_core.text.processing import ConcatenateJob
    corpora = get_corpora()
    if key == "dev":
        text_files = []
        for k, eval_info in corpora.dev.items():
            if str(eval_info.segmenter_type) == "ref":
                text_files.append(_get_corpus_text(eval_info.segmented_corpus, key))

    elif key == "test":
        text_files = []
        for k, eval_info in corpora.test.items():
            if str(eval_info.segmenter_type) == "ref":
                text_files.append(_get_corpus_text(eval_info.segmented_corpus, key))
    else:
        text_files = []
        if "dev" in key:
            for k, eval_info in corpora.dev.items():
                if key in k and str(eval_info.segmenter_type) == "ref":
                    text_files.append(_get_corpus_text(eval_info.segmented_corpus, key))
        elif "eval" in key or "mbw" in key:
            for k, eval_info in corpora.test.items():
                if key in k and str(eval_info.segmenter_type) == "ref":
                    text_files.append(_get_corpus_text(eval_info.segmented_corpus, key))
        else:
            raise NotImplementedError(f"Can not determine which part {key} is belonging to (dev/eval)")
        assert text_files, f"No data find for {key}"
    return ConcatenateJob(text_files).out

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
        sample_rate: int = 16000,
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
