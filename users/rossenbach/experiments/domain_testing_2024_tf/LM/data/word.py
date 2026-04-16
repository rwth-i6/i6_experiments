from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

import os
from typing import Any, Dict, Optional

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.text.processing import ConcatenateJob
from i6_core.returnn.config import CodeWrapper

from i6_experiments.common.datasets.librispeech import get_subword_nmt_bpe, get_bliss_corpus_dict
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LmLabelDatastream, LmIndexVocabulary


from .common import LmDataset, TrainingDatasets, SOURCE_DATASTREAM_KEY, TARGET_DATASTREAN_KEY

class LmWordDataset(LmDataset):

    def as_returnn_opts(self) -> Dict[str, Any]:
        d = {
            "class": "LmDataset",
            "corpus_file": CodeWrapper(DelayedFormat('lambda: cf("{}")', self.corpus_file)),
            "orth_symbols_map_file": self.vocab_file,
            "orth_replace_map_file": "",
            "word_based": True,
            "seq_end_symbol": "</s>",
            "auto_replace_unknown_symbol": True,
            "unknown_symbol": "<UNK>",
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
        lexicon_bliss: tk.Path, 
        train_text: tk.Path, 
        cv_text: tk.Path, 
        partition_epoch: int,
        seq_ordering: str,
):

    from i6_core.lm import LmIndexVocabularyFromLexiconJob

    lm_index_vocab_job = LmIndexVocabularyFromLexiconJob(bliss_lexicon=lexicon_bliss, count_ordering_text=train_text)
    lm_index_vocab = LmIndexVocabulary(
        vocab=lm_index_vocab_job.out_vocab,
        vocab_size=lm_index_vocab_job.out_vocab_size,
        unknown_token="<UNK>"
    )
    source_datstream = LmLabelDatastream(
        available_for_inference=True,
        lm_index_vocab=lm_index_vocab
    )
    target_datastream = LmLabelDatastream(
        available_for_inference=False,
        lm_index_vocab=lm_index_vocab
    )

    #### datasets ####
    lm_train_dataset = LmWordDataset(
        corpus_file=train_text,
        vocab_file=lm_index_vocab.vocab,
        partition_epoch=partition_epoch,
        segment_file=None,
        seq_ordering=seq_ordering
    )

    lm_cv_dataset = LmWordDataset(
        corpus_file=cv_text,
        vocab_file=lm_index_vocab.vocab,
        partition_epoch=1,
        segment_file=None,
        seq_ordering="sorted"
    )

    lm_devtrain_dataset = LmWordDataset(
        corpus_file=train_text,
        vocab_file=lm_index_vocab.vocab,
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
            SOURCE_DATASTREAM_KEY: source_datstream.as_returnn_extern_data_opts(available_for_inference=True),
            TARGET_DATASTREAN_KEY: target_datastream.as_returnn_extern_data_opts(available_for_inference=False)
        }
    ), lm_index_vocab
