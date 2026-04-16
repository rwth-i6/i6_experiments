from sisyphus import tk

from dataclasses import dataclass
from typing import Any, Dict, Optional

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.text.processing import ConcatenateJob

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
from i6_experiments.common.setups.returnn.datasets import Dataset, ControlDataset

# LMDataset does not work well with MetaDataset, thus we have to fix the keys to the internal RETURNN defaults
SOURCE_DATASTREAM_KEY = "delayed"
TARGET_DATASTREAN_KEY = "data"

@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    extern_data: Dict[str, Dict[str, Any]]


def get_librispeech_train_cv_data():
    """
    Build the LibriSpeech training/cv text for BPE training
    """
    ls_bliss_corpus_dict = get_bliss_corpus_dict()

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

    #### CV Data ####

    dev_clean_text = CorpusToTxtJob(bliss_corpus=ls_bliss_corpus_dict["dev-clean"], gzip=True).out_txt
    dev_other_text = CorpusToTxtJob(bliss_corpus=ls_bliss_corpus_dict["dev-other"], gzip=True).out_txt
    cv_text = ConcatenateJob(
        text_files=[dev_clean_text, dev_other_text],
        zip_out=True,
    ).out

    return full_train_text, cv_text


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
