from typing import Callable, Dict, List, Optional
import copy

from i6_core import returnn, corpus
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from recipe.i6_core.returnn.hdf import BlissToPcmHDFJob
from recipe.i6_core.text.processing import ConcatenateJob
from recipe.i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder
from recipe.i6_experiments.users.berger.recipe.corpus.filter import (
    FilterMismatchedSequencesJob,
)
from recipe.i6_experiments.users.berger.systems.dataclasses import AlignmentData
from . import data
from ..general import BasicSetupData
from sisyphus import tk


def get_librispeech_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    alignments: Dict[str, AlignmentData],
    train_key: str = "train-other-960",
    dev_keys: List[str] = ["dev-clean", "dev-other"],
    test_keys: List[str] = ["test-clean", "test-other"],
    add_unknown: bool = False,
    augmented_lexicon: bool = True,
    length_mismatch_check_function: Optional[Callable[[int, int], bool]] = None,
) -> BasicSetupData:
    # ********** Data inputs **********

    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = data.get_data_inputs(
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        ctc_lexicon=True,
        use_augmented_lexicon=augmented_lexicon,
        add_all_allophones=True,
        audio_format="wav",  # Note: OGGZip dataset lead to length mismatches between features and alignment
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

    train_sample_hdf_job = BlissToPcmHDFJob(train_corpus, returnn_root=returnn_root)
    train_sample_hdf_job.rqmt["mem"] = 8
    train_sample_hdf_job.rqmt["time"] = 24
    train_sample_hdf = train_sample_hdf_job.out_hdf
    train_alignment_hdf = alignments[f"{train_key}_align"].get_hdf(
        returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
    )

    if length_mismatch_check_function is not None:
        segment_whitelist = FilterMismatchedSequencesJob(
            feature_hdf=train_sample_hdf,
            target_hdf=train_alignment_hdf,
            check_mismatch_func=length_mismatch_check_function,
            returnn_root=returnn_root,
        ).out_segment_whitelist
    else:
        segment_whitelist = None

    train_dataset_builder = MetaDatasetBuilder()
    train_dataset_builder.add_hdf_dataset(
        name="data",
        hdf_files=train_sample_hdf,
        key_mapping={"data": "data"},
    )

    train_dataset_builder.add_hdf_dataset(
        name="classes",
        hdf_files=train_alignment_hdf,
        seq_ordering="laplace:25",
        dataset_config={
            "partition_epoch": 20,
            "seq_list_filter_file": segment_whitelist,
        },
        key_mapping={"data": "classes"},
        control=True,
    )
    train_data_config = train_dataset_builder.get_dict()

    # ********** CV data **********

    cv_data_inputs = copy.deepcopy(dev_data_inputs)

    if not add_unknown:
        for corpus_object in [cv_data_inputs[key].corpus_object for key in dev_keys]:
            assert corpus_object.corpus_file is not None
            corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
                corpus_object.corpus_file,
                train_lexicon,
                all_unknown=False,
            ).out_corpus

    cv_sample_hdfs = [
        BlissToPcmHDFJob(
            data_input.corpus_object.corpus_file, returnn_root=returnn_root
        ).out_hdf
        for key, data_input in cv_data_inputs.items()
        if key in dev_keys
    ]
    cv_alignment_hdfs = [
        alignments[f"{dev_key}_align"].get_hdf(
            returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
        )
        for dev_key in dev_keys
    ]

    if length_mismatch_check_function is not None:
        segment_whitelists = [
            FilterMismatchedSequencesJob(
                feature_hdf=cv_sample_hdf,
                target_hdf=cv_alignment_hdf,
                check_mismatch_func=length_mismatch_check_function,
                returnn_root=returnn_root,
            ).out_segment_whitelist
            for cv_sample_hdf, cv_alignment_hdf in zip(
                cv_sample_hdfs, cv_alignment_hdfs
            )
        ]
        segment_whitelist = ConcatenateJob(segment_whitelists).out
    else:
        segment_whitelist = None

    cv_dataset_builder = MetaDatasetBuilder()
    cv_dataset_builder.add_hdf_dataset(
        name="data",
        hdf_files=cv_sample_hdfs,
        key_mapping={"data": "data"},
    )

    cv_dataset_builder.add_hdf_dataset(
        name="classes",
        hdf_files=cv_alignment_hdfs,
        seq_ordering="sorted",
        dataset_config={
            "partition_epoch": 1,
            "seq_list_filter_file": segment_whitelist,
        },
        key_mapping={"data": "classes"},
        control=True,
    )
    cv_data_config = cv_dataset_builder.get_dict()

    # ********** Align data **********

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input)
        for key, data_input in {**train_data_inputs, **dev_data_inputs}.items()
    }
    for data_input in align_data_inputs.values():
        data_input.lexicon.filename = train_lexicon

        if not add_unknown:
            assert data_input.corpus_object.corpus_file is not None
            data_input.corpus_object.corpus_file = (
                corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
                    data_input.corpus_object.corpus_file,
                    train_lexicon,
                    all_unknown=False,
                ).out_corpus
            )

    # ********** Recog lexicon **********

    recog_lexicon = AddEowPhonemesToLexiconJob(train_lexicon).out_lexicon

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon.filename = recog_lexicon

    return BasicSetupData(
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        align_keys=[f"{train_key}_align", *[f"{key}_align" for key in dev_keys]],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
            **align_data_inputs,
        },
    )
