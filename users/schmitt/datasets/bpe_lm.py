from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional

from i6_core.text.label.subword_nmt.apply import ApplyBPEToTextJob
from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.text.processing import ConcatenateJob
from i6_core.returnn.config import CodeWrapper

from i6_experiments.common.setups.returnn.datasets import MetaDataset, ControlDataset, Dataset
from i6_experiments.common.setups.returnn.datastreams.base import Datastream
from i6_experiments.common.setups.returnn.datastreams.vocabulary import BpeDatastream
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe_v2
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data



SOURCE_DATASTREAM_KEY = "data"
TARGET_DATASTREAN_KEY = "delayed"


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    datastreams: Dict[str, Datastream]


class LmDataset(ControlDataset):

    def __init__(
        self,
        *,
        corpus_file: tk.Path,
        vocab_file: tk.Path,
        # super parameters
        partition_epoch: Optional[int] = None,
        segment_file: Optional[tk.Path] = None,
        seq_ordering: Optional[str] = None,
        random_subset: Optional[int] = None,
        additional_options: Optional[Dict] = None,
    ):
        super().__init__(
            partition_epoch=partition_epoch,
            segment_file=segment_file,
            seq_ordering=seq_ordering,
            random_subset=random_subset,
            additional_options=additional_options
        )

        self.corpus_file = corpus_file
        self.vocab_file = vocab_file

    def as_returnn_opts(self) -> Dict[str, Any]:
        d = {
            "class": "LmDataset",
            "corpus_file": self.corpus_file,
            "orth_symbols_map_file": self.vocab_file,
            "orth_replace_map_file": "",
            "word_based": True,
            "seq_end_symbol": "</s>",
            "auto_replace_unknown_symbol": False,
            "unknown_symbol": "<unk>",
            "add_delayed_seq_data": True,
            "delayed_seq_data_start_symbol": "<s>",
            "use_cache_manager": True,
        }
        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s"
            % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d

@dataclass()
class LMDatasetSettings:
    train_partition_epoch: int
    train_seq_ordering: str


def get_subword_repo():
    """
    This is a for now very ugly helper to get the same subword_nmt repo
    as the get_subword_nmt_bpe_v2 is using
    :return: 
    """
    subword_nmt_repo = get_returnn_subword_nmt(
        commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf", output_prefix=""
    )
    # overwrite hash for future bugfixes, it is unlikely the logic will ever be changed
    subword_nmt_repo.hash_overwrite = "I6_SUBWORD_NMT_V2"
    return subword_nmt_repo

def build_lm_training_datasets(prefix, librispeech_key, bpe_size, settings: LMDatasetSettings):
    
    #data_map = {SOURCE_DATASTREAM_KEY: ("lm_dataset", "data"), TARGET_DATASTREAN_KEY: ("lm_dataset", "delayed")}
    #def make_meta(dataset: LmDataset):
    #    return MetaDataset(
    #        data_map=data_map, datasets={"lm_dataset": dataset}, seq_order_control_dataset="lm_dataset"
    #    )
    
    bpe_settings = get_subword_nmt_bpe_v2(corpus_key=librispeech_key, bpe_size=bpe_size, unk_label='<unk>')
    ls_bliss_corpus_dict = get_bliss_corpus_dict()
    bpe_datastream = BpeDatastream(available_for_inference=False, bpe_settings=bpe_settings)

     #### Training Data ####

    lm_data = get_librispeech_normalized_lm_data()
    ls_train_bliss = ls_bliss_corpus_dict["train-other-960"]
    ls_train_text = CorpusToTxtJob(
        bliss_corpus=ls_train_bliss,
        gzip=True,
    ).out_txt
    full_train_text = ConcatenateJob(
        text_files=[lm_data, ls_train_text],
        zip_out=True,
    ).out
    lm_bpe_data_job = ApplyBPEToTextJob(
        text_file=full_train_text,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_count_vocab,
        gzip_output=True,
        subword_nmt_repo=get_subword_repo(),
        mini_task=False,  # this is a large file, so run in cluster
    )
    lm_bpe_data_job.add_alias(os.path.join(prefix, "apply_bpe_to_train"))

    #### Dev Data ####

    dev_clean_text = CorpusToTxtJob(bliss_corpus=ls_bliss_corpus_dict["dev-clean"], gzip=True).out_txt
    dev_other_text = CorpusToTxtJob(bliss_corpus=ls_bliss_corpus_dict["dev-other"], gzip=True).out_txt
    cv_text = ConcatenateJob(
        text_files=[dev_clean_text, dev_other_text],
        zip_out=True,
    ).out
    cv_bpe_data_job = ApplyBPEToTextJob(
        text_file=cv_text,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_count_vocab,
        gzip_output=True,
        subword_nmt_repo=get_subword_repo(),
    )

    #### datasets ####
    lm_train_dataset = LmDataset(
        corpus_file=lm_bpe_data_job.out_bpe_text,
        vocab_file=bpe_settings.bpe_vocab,
        partition_epoch=settings.train_partition_epoch,
        segment_file=None,
        seq_ordering=settings.train_seq_ordering
    )

    lm_cv_dataset = LmDataset(
        corpus_file=cv_bpe_data_job.out_bpe_text,
        vocab_file=bpe_settings.bpe_vocab,
        partition_epoch=1,
        segment_file=None,
        seq_ordering="sorted"
    )

    lm_devtrain_dataset = LmDataset(
        corpus_file=lm_bpe_data_job.out_bpe_text,
        vocab_file=bpe_settings.bpe_vocab,
        partition_epoch=1,
        segment_file=None,
        seq_ordering="sorted",
        random_subset=3000,
    )

    return TrainingDatasets(
        train=lm_train_dataset,
        cv=lm_cv_dataset,
        # devtrain=lm_devtrain_dataset,
        # TODO: Ultra hack for now
        devtrain=lm_cv_dataset,
        datastreams={"data": bpe_datastream, "delayed": bpe_datastream},
    )

