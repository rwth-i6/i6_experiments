from typing import Dict, List
import copy

from i6_core import corpus
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core import returnn
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder
from i6_experiments.users.berger.systems.dataclasses import AlignmentData
from . import data
from ..general import BasicSetupData
from sisyphus import tk


def get_wsj_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    alignments: Dict[str, AlignmentData],
    train_key: str = "train_si284",
    cv_key: str = "cv_dev93",
    dev_keys: List[str] = ["cv_dev93"],
    test_keys: List[str] = ["test_eval92"],
    freq_kHz: int = 16,
    **kwargs,
) -> BasicSetupData:
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
    train_alignment_hdf = alignments[f"{train_key}_align"].get_hdf(
        returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
    )

    train_dataset_builder = MetaDatasetBuilder()
    train_dataset_builder.add_hdf_dataset(
        name="data",
        hdf_files=train_sample_hdf,
        key_mapping={"data": "data"},
    )

    train_dataset_builder.add_hdf_dataset(
        name="classes",
        hdf_files=train_alignment_hdf,
        dataset_config={
            "partition_epoch": 3,
            "seq_ordering": "laplace:.1000",
        },
        key_mapping={"data": "classes"},
        control=True,
    )
    train_data_config = train_dataset_builder.get_dict()

    # ********** CV data **********

    cv_sample_hdf = returnn.BlissToPcmHDFJob(
        cv_data_input.corpus_object.corpus_file,
        rounding=returnn.BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
        returnn_root=returnn_root,
    ).out_hdf
    cv_alignment_hdf = alignments[f"{cv_key}_align"].get_hdf(
        returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
    )

    cv_dataset_builder = MetaDatasetBuilder()
    cv_dataset_builder.add_hdf_dataset(
        name="data",
        hdf_files=[cv_sample_hdf],
        key_mapping={"data": "data"},
    )

    cv_dataset_builder.add_hdf_dataset(
        name="classes",
        hdf_files=[cv_alignment_hdf],
        dataset_config={
            "partition_epoch": 1,
            "seq_ordering": "sorted",
        },
        key_mapping={"data": "classes"},
        control=True,
    )
    cv_data_config = cv_dataset_builder.get_dict()

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
        **align_data_inputs,
    }

    return BasicSetupData(
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        align_keys=[f"{train_key}_align", f"{cv_key}_align"],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        data_inputs=all_data_inputs,
    )
