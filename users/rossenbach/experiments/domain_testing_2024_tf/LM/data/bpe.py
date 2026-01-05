from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

import os
from typing import Any, Dict, Optional

from i6_core.text.label.subword_nmt.apply import ApplyBPEToTextJob

from i6_core.returnn.config import CodeWrapper

from i6_experiments.common.datasets.librispeech import get_subword_nmt_bpe, get_bliss_corpus_dict
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt

from i6_experiments.common.setups.returnn.datastreams.vocabulary import BpeDatastream


from .common import LmDataset, TrainingDatasets, SOURCE_DATASTREAM_KEY, TARGET_DATASTREAN_KEY

class LmBpeDataset(LmDataset):

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
        allow_unknown=False,
    ):
        super().__init__(
            corpus_file=corpus_file,
            vocab_file=vocab_file,
            partition_epoch=partition_epoch,
            segment_file=segment_file,
            seq_ordering=seq_ordering,
            random_subset=random_subset,
            additional_options=additional_options,
        )
        self.allow_unknown = allow_unknown

    def as_returnn_opts(self) -> Dict[str, Any]:
        d = {
            "class": "LmDataset",
            "corpus_file": CodeWrapper(DelayedFormat('lambda: cf("{}")', self.corpus_file)),
            "orth_symbols_map_file": self.vocab_file,
            "orth_replace_map_file": "",
            "word_based": True,
            "seq_end_symbol": "</s>",
            "auto_replace_unknown_symbol": self.allow_unknown,
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


def build_training_data(
        prefix: str,
        train_text: tk.Path,
        cv_text: tk.Path,
        bpe_size: int,
        partition_epoch: int,
        seq_ordering: str,
        allow_unknown: bool = False,
):
    bpe_settings = get_subword_nmt_bpe(corpus_key="train-other-960", bpe_size=bpe_size, unk_label='<unk>')
    bpe_datastream = BpeDatastream(available_for_inference=False, bpe_settings=bpe_settings)

    # Train data
    lm_bpe_data_job = ApplyBPEToTextJob(
        text_file=train_text,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_count_vocab,
        gzip_output=True,
        subword_nmt_repo=get_returnn_subword_nmt()
    )
    lm_bpe_data_job.add_alias(os.path.join(prefix, "apply_bpe_to_train"))

    #### Dev Data ####
    cv_bpe_data_job = ApplyBPEToTextJob(
        text_file=cv_text,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_count_vocab,
        gzip_output=True,
        subword_nmt_repo=get_returnn_subword_nmt()
    )

    #### datasets ####
    lm_train_dataset = LmBpeDataset(
        corpus_file=lm_bpe_data_job.out_bpe_text,
        vocab_file=bpe_settings.bpe_vocab,
        partition_epoch=partition_epoch,
        segment_file=None,
        seq_ordering=seq_ordering,
        allow_unknown=allow_unknown,
    )

    lm_cv_dataset = LmBpeDataset(
        corpus_file=cv_bpe_data_job.out_bpe_text,
        vocab_file=bpe_settings.bpe_vocab,
        partition_epoch=1,
        segment_file=None,
        seq_ordering="sorted"
    )

    lm_devtrain_dataset = LmBpeDataset(
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
    ), bpe_datastream, bpe_settings
