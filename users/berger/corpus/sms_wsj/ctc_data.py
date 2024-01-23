import copy
from typing import List

from i6_core import returnn, corpus
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from . import data
from ..general import CTCSetupData
from sisyphus import tk


def get_wsj_data_hdf(
    returnn_root: tk.Path,
    train_key: str = "train_si284",
    cv_key: str = "cv_dev93",
    dev_keys: List[str] = ["cv_dev93"],
    test_keys: List[str] = ["test_eval92"],
    freq_kHz: int = 16,
) -> CTCSetupData:
    # ********** Data inputs **********

    train_data_input, cv_data_input, dev_data_inputs, test_data_inputs = data.get_data_inputs(
        train_key=train_key,
        cv_key=cv_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        ctc_lexicon=True,
        add_all_allophones=True,
        freq=freq_kHz,
        lm_name="64k_3gram",
        recog_lex_name="nab-64k",
    )

    # ********** Train data **********
    train_corpus = train_data_input.corpus_object.corpus_file
    assert train_corpus is not None

    train_sample_hdf_job = returnn.BlissToPcmHDFJob(
        train_corpus, rounding=returnn.BlissToPcmHDFJob.RoundingScheme.rasr_compatible, returnn_root=returnn_root
    )
    train_sample_hdf_job.rqmt["mem"] = 8
    train_sample_hdf_job.rqmt["time"] = 24
    train_sample_hdf = train_sample_hdf_job.out_hdf

    train_data_config = {
        "class": "HDFDataset",
        "files": [train_sample_hdf],
        "partition_epoch": 3,
        "seq_ordering": "laplace:.1000",
        "use_cache_manager": True,
    }

    # ********** CV data **********
    cv_corpus = cv_data_input.corpus_object.corpus_file
    cv_sample_hdf = returnn.BlissToPcmHDFJob(
        cv_data_input.corpus_object.corpus_file,
        rounding=returnn.BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
        returnn_root=returnn_root,
    ).out_hdf

    cv_data_config = {
        "class": "HDFDataset",
        "files": [cv_sample_hdf],
        "partition_epoch": 1,
        "seq_ordering": "sorted",
        "use_cache_manager": True,
    }

    # ********** Loss corpus **********

    loss_corpus = corpus.MergeCorporaJob(
        [train_corpus, cv_corpus],
        name="loss-corpus",
        merge_strategy=corpus.MergeStrategy.SUBCORPORA,
    ).out_merged_corpus
    loss_lexicon = train_data_input.lexicon.filename

    # ********** Recog lexicon **********

    dev_data_inputs = copy.deepcopy(dev_data_inputs)
    test_data_inputs = copy.deepcopy(test_data_inputs)

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon.filename = AddEowPhonemesToLexiconJob(rasr_input.lexicon.filename).out_lexicon

    # ********** Align data **********

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input)
        for key, data_input in [(train_key, train_data_input), (cv_key, cv_data_input)]
    }
    for data_input in align_data_inputs.values():
        data_input.lexicon.filename = AddEowPhonemesToLexiconJob(data_input.lexicon.filename).out_lexicon

    all_data_inputs = {
        train_key: train_data_input,
        cv_key: cv_data_input,
        **dev_data_inputs,
        **test_data_inputs,
        **align_data_inputs
    }

    return CTCSetupData(
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        align_keys=[f"{train_key}_align", f"{cv_key}_align"],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        loss_corpus=loss_corpus,
        loss_lexicon=loss_lexicon,
        data_inputs=all_data_inputs,
    )
