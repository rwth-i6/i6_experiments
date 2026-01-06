from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional

from i6_core.text.label.subword_nmt.apply import ApplyBPEToTextJob
from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
from i6_core.text.processing import ConcatenateJob, HeadJob
from i6_core.returnn.config import CodeWrapper

from i6_experiments.common.setups.returnn.datasets import MetaDataset, ControlDataset, Dataset
from i6_experiments.common.setups.returnn.datastreams.base import Datastream
from i6_experiments.common.setups.returnn.datastreams.vocabulary import BpeDatastream
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt

from i6_experiments.common.datasets.loquacious.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.loquacious.vocab import get_subword_nmt_bpe

from .common import get_dev_short_bliss


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
            "corpus_file": CodeWrapper(DelayedFormat('lambda: cf("{}")', self.corpus_file)),
            "orth_symbols_map_file": self.vocab_file,
            "orth_replace_map_file": "",
            "word_based": True,
            "seq_end_symbol": "</s>",
            "auto_replace_unknown_symbol": True,
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

def build_lm_training_datasets(prefix, loquacious_key, bpe_size, settings: LMDatasetSettings):
    
    #data_map = {SOURCE_DATASTREAM_KEY: ("lm_dataset", "data"), TARGET_DATASTREAN_KEY: ("lm_dataset", "delayed")}
    #def make_meta(dataset: LmDataset):
    #    return MetaDataset(
    #        data_map=data_map, datasets={"lm_dataset": dataset}, seq_order_control_dataset="lm_dataset"
    #    )
    
    bpe_settings = get_subword_nmt_bpe(corpus_key=loquacious_key, bpe_size=bpe_size, unk_label='<unk>')
    bpe_datastream = BpeDatastream(available_for_inference=False, bpe_settings=bpe_settings)

     #### Training Data ####

    lm_train_text = tk.Path("/work/asr4/rossenbach/corpora/loquacious/LoquaciousAdditionalResources/loquacious-large.txt", hash_overwrite="LOQUACIOUS_LARGE_TEXT")
    lm_bpe_data_job = ApplyBPEToTextJob(
        text_file=lm_train_text,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_count_vocab,
        gzip_output=True,
        subword_nmt_repo=get_subword_repo(),
        mini_task=False,  # this is a large file, so run in cluster
    )
    lm_bpe_data_job.add_alias(os.path.join(prefix, "apply_bpe_to_train"))

    #### Dev Data ####

    dev_short_text = CorpusToTxtJob(bliss_corpus=get_dev_short_bliss(), gzip=True).out_txt
    cv_bpe_data_job = ApplyBPEToTextJob(
        text_file=dev_short_text,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_count_vocab,
        gzip_output=True,
        subword_nmt_repo=get_subword_repo(),
    )

    # statistics
    from i6_core.text.processing import PipelineJob
    wordcount = PipelineJob(dev_short_text, ["wc -w"], mini_task=True).out
    bpecount = PipelineJob(cv_bpe_data_job.out_bpe_text, ["wc -w"], mini_task=True).out
    tk.register_output(prefix + "/wordcount", wordcount)
    tk.register_output(prefix + "/bpecount", bpecount)

    #### Dev Train Data ####

    # only shuffle, this is deterministic
    shuffle_segment_file_job = ShuffleAndSplitSegmentsJob(
        segment_file=lm_train_text,
        split={"shuffle": 1.0},
        shuffle=True
    )
    segment_file = shuffle_segment_file_job.out_segments["shuffle"]
    devtrain_text = HeadJob(segment_file, num_lines=3000).out
    devtrain_bpe_data_job = ApplyBPEToTextJob(
        text_file=devtrain_text,
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
        corpus_file=devtrain_bpe_data_job.out_bpe_text,
        vocab_file=bpe_settings.bpe_vocab,
        partition_epoch=1,
        segment_file=None,
        seq_ordering="sorted",
    )

    return TrainingDatasets(
        train=lm_train_dataset,
        cv=lm_cv_dataset,
        devtrain=lm_devtrain_dataset,
        datastreams={"data": bpe_datastream, "delayed": bpe_datastream},
    )

