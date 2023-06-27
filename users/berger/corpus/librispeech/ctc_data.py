from typing import List
import copy

from i6_core import returnn, corpus
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from . import data
from ..general import CTCSetupData
from sisyphus import tk


def get_librispeech_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    train_key: str = "train-other-960",
    dev_keys: List[str] = ["dev-clean", "dev-other"],
    test_keys: List[str] = ["test-clean", "test-other"],
    add_unknown: bool = False,
    augmented_lexicon: bool = False,
) -> CTCSetupData:
    # ********** Data inputs **********

    train_data_inputs, dev_data_inputs, _ = data.get_data_inputs(
        train_key=train_key,
        dev_keys=dev_keys,
        ctc_lexicon=True,
        use_augmented_lexicon=augmented_lexicon,
        add_all_allophones=True,
        audio_format="ogg",
        add_unknown_phoneme_and_mapping=add_unknown,
    )

    wav_train_data_inputs, wav_dev_data_inputs, wav_test_data_inputs = data.get_data_inputs(
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        ctc_lexicon=True,
        use_augmented_lexicon=augmented_lexicon,
        add_all_allophones=True,
        audio_format="wav",
        add_unknown_phoneme_and_mapping=add_unknown,
    )

    # ********** Train data **********

    train_corpus = train_data_inputs[train_key].corpus_object.corpus_file
    train_lexicon = train_data_inputs[train_key].lexicon.filename
    assert train_corpus is not None

    if not add_unknown:
        train_corpus = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
            train_corpus,
            train_lexicon,
            all_unknown=False,
        ).out_corpus

    train_ogg_zip = returnn.BlissToOggZipJob(
        train_corpus,
        no_conversion=True,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    ).out_ogg_zip

    train_data_config = {
        "class": "OggZipDataset",
        "audio": {"features": "raw", "sample_rate": 16_000},
        "targets": None,
        "partition_epoch": 20,
        "path": train_ogg_zip,
        "seq_ordering": "random",
        "use_cache_manager": True,
    }

    # ********** CV data **********

    if not add_unknown:
        for corpus_object in [dev_data_inputs[key].corpus_object for key in dev_keys]:
            assert corpus_object.corpus_file is not None
            corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
                corpus_object.corpus_file,
                train_lexicon,
                all_unknown=False,
            ).out_corpus

    cv_corpus = corpus.MergeCorporaJob(
        [dev_data_inputs[key].corpus_object.corpus_file for key in dev_keys],
        name="dev_combine",
        merge_strategy=corpus.MergeStrategy.CONCATENATE,
    ).out_merged_corpus

    cv_ogg_zip = returnn.BlissToOggZipJob(
        cv_corpus,
        no_conversion=True,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    ).out_ogg_zip

    cv_data_config = {
        "class": "OggZipDataset",
        "audio": {"features": "raw", "sample_rate": 16_000},
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
    loss_lexicon = train_lexicon

    # ********** Recog lexicon **********

    recog_lexicon = AddEowPhonemesToLexiconJob(loss_lexicon).out_lexicon

    for rasr_input in {**wav_dev_data_inputs, **wav_test_data_inputs}.values():
        rasr_input.lexicon.filename = recog_lexicon

    # ********** Align data **********

    align_lexicon = copy.deepcopy(recog_lexicon)

    align_data_inputs = {f"{key}_align": copy.deepcopy(data_input) for key, data_input in {**wav_train_data_inputs, **wav_dev_data_inputs}.items()}
    for data_input in align_data_inputs.values():
        data_input.lexicon.filename = align_lexicon

        if not add_unknown:
            assert data_input.corpus_object.corpus_file is not None
            data_input.corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
                data_input.corpus_object.corpus_file,
                align_lexicon,
                all_unknown=False,
            ).out_corpus

    return CTCSetupData(
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        align_keys=[f"{train_key}_align", *[f"{key}_align" for key in dev_keys]],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        loss_corpus=loss_corpus,
        loss_lexicon=loss_lexicon,
        data_inputs={
            **train_data_inputs,
            **wav_dev_data_inputs,
            **wav_test_data_inputs,
            **align_data_inputs,
        },
    )
