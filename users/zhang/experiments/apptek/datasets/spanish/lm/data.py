from dataclasses import dataclass
from returnn_common.datasets_old_2022_10.interface import DatasetConfig, VocabConfig
from sisyphus import Path, tk

from functools import cache
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from i6_experiments.users.zhang.experiments.apptek.datasets.tools import OggZipFixTxtTextualJob, BlissStripOrthPunctJob
from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import NEED_FIX_OGG_ZIP_DATASET_NAME, DEV_KEYS, TEST_KEYS
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.corpus_lex import get_corpora


ALIAS_PREFIX = "datasets/spanish/lm"
default_train_epoch_split = 20
#LM_TRAIN_DATA = Path("/nas/models/asr/mgunz/2024-07-08--zeyer-setup-apptek/lm_train.bgd.unk.subsampled.100m.txt.gz")
LM_TRAIN_DATA = Path("/nas/models/asr/hzhang/setups/2025-07-20--combined/data/ES/lm_text.gz")
LM_TRANS_TRAIN_DATA = Path("/nas/models/asr/hzhang/setups/2025-07-20--combined/data/ES/trans.train.merged.txt.gz")

# 16kHz ?
class SpanishLmDataset(DatasetConfig):
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
        only_transcripts: Optional[bool] = False,
    ):
        super().__init__()
        self.vocab = vocab
        self.train_vocab = train_vocab
        self.main_key = main_key
        self.train_epoch_split = train_epoch_split
        self.train_sort_order = train_sort_order
        self.train_sort_laplace_num_seqs = train_sort_laplace_num_seqs
        self.eval_subset = eval_subset
        self.only_transcripts = only_transcripts

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
                "corpus_file": [LM_TRANS_TRAIN_DATA] if self.only_transcripts else [LM_TRAIN_DATA, LM_TRANS_TRAIN_DATA], # A zipped file, named out_text
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

@cache
def _get_ogg_zip(
    corpus: tk.Path, name: str, split: int, returnn_root: Union[str, tk.Path], alias_prefix: str,
) -> tk.Path:
    segment_job = SegmentCorpusJob(corpus, split)
    if any(infix in name for infix in ["dev_conversation","common_voice_two_speakers"]): # Actually this should be done right after the creation of corpus
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
    oggzip_job.add_alias(f"{alias_prefix}/oggzip/{name}")
    tk.register_output(f"{alias_prefix}/oggzip/{name}.ogg.zip", oggzip_job.out_ogg_zip)
    return oggzip_job.out_ogg_zip

def _get_lm_eval_ogg_zip( # any data set inside "dev" or "test" will affect the hash for training Job!
    *,
    returnn_root: Union[str, tk.Path] = tk.Path("/home/mgunz/setups/2024-07-08--zeyer-setup-apptek/recipe/returnn"),
    alias_prefix: str,
    key: str,
) -> List[Path]:  #

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
                )
                )
    elif key == "test": # During Training
        ogg_zip_files = []
        for k, eval_info in lm_corpora.test.items():
            if str(eval_info.segmenter_type) == "ref" and "mbw" not in k:
                ogg_zip_files.append(_get_ogg_zip(
                    eval_info.segmented_corpus,
                    name=f"seg.{eval_info.segmenter_type}/{k}",
                    split=10,
                    returnn_root=returnn_root,
                    alias_prefix=alias_prefix,
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