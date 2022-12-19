from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional

from i6_core.text.label.subword_nmt.apply import ApplyBPEToTextJob
from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.text.processing import ConcatenateJob
from i6_core.returnn.config import CodeWrapper

from i6_experiments.common.datasets.librispeech import get_subword_nmt_bpe, get_bliss_corpus_dict
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data

from returnn_common.datasets import MetaDataset, Dataset, ControlDataset
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import BpeDatastream


SOURCE_DATASTREAM_KEY = "delayed"
TARGET_DATASTREAN_KEY = "data"


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
            "corpus_file": CodeWrapper(DelayedFormat('lambda: cf("{}")', self.corpus_file)),
            "orth_symbols_map_file": self.vocab_file,
            "orth_replace_map_file": "",
            "word_based": True,
            "seq_end_symbol": "</s>",
            "auto_replace_unknown_symbol": False,
            "unknown_symbol": "<unk>",
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


def build_training_data(output_prefix="", corpus_key="train-clean-100", bpe_size=2000, partition_epoch=4):
    bpe_settings = get_subword_nmt_bpe(corpus_key=corpus_key, bpe_size=bpe_size, unk_label='<unk>')
    # conversion factor for PPL computation is 1.448
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
        subword_nmt_repo=get_returnn_subword_nmt()
    )
    lm_bpe_data_job.add_alias(os.path.join(output_prefix, "apply_bpe_to_train"))
    tk.register_output("test_lm_bpe.txt.gz", lm_bpe_data_job.out_bpe_text)

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
        subword_nmt_repo=get_returnn_subword_nmt()
    )

    #### datasets ####
    lm_train_dataset = LmDataset(
        corpus_file=lm_bpe_data_job.out_bpe_text,
        vocab_file=bpe_settings.bpe_vocab,
        partition_epoch=partition_epoch,
        segment_file=None,
        seq_ordering="sort_bin_shuffle:.32"
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
        devtrain=lm_devtrain_dataset,
        extern_data={
            SOURCE_DATASTREAM_KEY: bpe_datastream.as_returnn_extern_data_opts(available_for_inference=True),
            TARGET_DATASTREAN_KEY: bpe_datastream.as_returnn_extern_data_opts(available_for_inference=False)
        }
    )









