import copy
from typing import List

from i6_core import returnn, corpus
from i6_core.audio.encoding import BlissChangeEncodingJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from . import data
from ..general import CTCSetupData
from sisyphus import tk


def get_wsj_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    train_key: str = "train_si284",
    cv_key: str = "cv_dev93",
    dev_keys: List[str] = ["cv_dev93"],
    test_keys: List[str] = ["test_eval92"],
    freq_kHz: int = 16,
) -> CTCSetupData:
    # ********** Data inputs **********

    (train_data_input, cv_data_input, dev_data_inputs, test_data_inputs,) = data.get_data_inputs(
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

    train_ogg = BlissChangeEncodingJob(
        train_corpus,
        output_format="ogg",
        codec="libvorbis",
    ).out_corpus

    train_ogg_zip = returnn.BlissToOggZipJob(
        train_ogg,
        no_conversion=True,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    ).out_ogg_zip

    train_data_config = {
        "class": "OggZipDataset",
        "audio": {"features": "raw", "sample_rate": freq_kHz * 1000},
        "targets": None,
        "partition_epoch": 3,
        "path": train_ogg_zip,
        "seq_ordering": "random",
        "use_cache_manager": True,
    }

    # ********** CV data **********
    cv_corpus = cv_data_input.corpus_object.corpus_file
    assert cv_corpus is not None

    cv_ogg = BlissChangeEncodingJob(
        cv_corpus,
        output_format="ogg",
        codec="libvorbis",
    ).out_corpus

    cv_ogg_zip = returnn.BlissToOggZipJob(
        cv_ogg,
        no_conversion=True,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    ).out_ogg_zip

    cv_data_config = {
        "class": "OggZipDataset",
        "audio": {"features": "raw", "sample_rate": freq_kHz * 1000},
        "targets": None,
        "partition_epoch": 1,
        "path": cv_ogg_zip,
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

    all_data_inputs = {
        f"{train_key}_align": train_data_input,
        f"{cv_key}_align": cv_data_input,
        **dev_data_inputs,
        **test_data_inputs,
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
