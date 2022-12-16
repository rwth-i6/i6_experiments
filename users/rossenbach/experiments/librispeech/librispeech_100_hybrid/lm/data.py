from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.text.processing import ConcatenateJob
from i6_core.returnn.config import CodeWrapper

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
from i6_experiments.common.datasets.librispeech.vocab import get_lm_vocab

from returnn_common.datasets import Dataset, ControlDataset
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LmLabelDatastream


SOURCE_DATASTREAM_KEY = "delayed"
TARGET_DATASTREAM_KEY = "data"


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    extern_data: Dict[str, Dict[str, Any]]


class LmDataset(ControlDataset):

    def __init__(
        self,
        *,
        corpus_file: tk.Path,
        vocab_file: tk.Path,
        unknown_symbol: Union[str, tk.Variable] = "<unk>",
        auto_replace_unknown_symbol: bool = False,
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
        self.unknown_symbol = unknown_symbol
        self.auto_replace_unknown_symbol = auto_replace_unknown_symbol

    def as_returnn_opts(self) -> Dict[str, Any]:
        d = {
            "class": "LmDataset",
            "corpus_file": CodeWrapper(DelayedFormat('lambda: cf("{}")', self.corpus_file)),
            "orth_symbols_map_file": self.vocab_file,
            "orth_replace_map_file": "",
            "word_based": True,
            "seq_end_symbol": "</s>",
            "auto_replace_unknown_symbol": self.auto_replace_unknown_symbol,
            "unknown_symbol": self.unknown_symbol,
            "add_delayed_seq_data": True,
            "delayed_seq_data_start_symbol": "<s>",
        }
        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s"
            % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d


def build_training_data(output_prefix="", partition_epoch=20):
    # conversion factor for PPL computation is 1.448
    ls_bliss_corpus_dict = get_bliss_corpus_dict()
    lm_vocab = get_lm_vocab()
    label_datastream = LmLabelDatastream(
        available_for_inference=True,
        lm_index_vocab=lm_vocab
    )

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

    #### Dev Data ####

    dev_clean_text = CorpusToTxtJob(bliss_corpus=ls_bliss_corpus_dict["dev-clean"], gzip=True).out_txt
    dev_other_text = CorpusToTxtJob(bliss_corpus=ls_bliss_corpus_dict["dev-other"], gzip=True).out_txt
    cv_text = ConcatenateJob(
        text_files=[dev_clean_text, dev_other_text],
        zip_out=True,
    ).out

    #### datasets ####
    lm_train_dataset = LmDataset(
        corpus_file=full_train_text,
        vocab_file=lm_vocab.vocab,
        unknown_symbol=lm_vocab.unknown_token,
        auto_replace_unknown_symbol=True,
        partition_epoch=partition_epoch,
        segment_file=None,
        seq_ordering="sort_bin_shuffle:.32"
    )

    lm_cv_dataset = LmDataset(
        corpus_file=cv_text,
        vocab_file=lm_vocab.vocab,
        unknown_symbol=lm_vocab.unknown_token,
        auto_replace_unknown_symbol=True,
        partition_epoch=1,
        segment_file=None,
        seq_ordering="sorted"
    )

    lm_devtrain_dataset = LmDataset(
        corpus_file=full_train_text,
        vocab_file=lm_vocab.vocab,
        unknown_symbol=lm_vocab.unknown_token,
        auto_replace_unknown_symbol=True,
        partition_epoch=1,
        segment_file=None,
        seq_ordering="sorted",
        random_subset=3000,
    )

    return TrainingDatasets(
        train=lm_train_dataset,
        cv=lm_cv_dataset,
        devtrain=lm_devtrain_dataset,
        extern_data={
            SOURCE_DATASTREAM_KEY: label_datastream.as_returnn_extern_data_opts(available_for_inference=True),
            TARGET_DATASTREAM_KEY: label_datastream.as_returnn_extern_data_opts(available_for_inference=False)
        }
    )
