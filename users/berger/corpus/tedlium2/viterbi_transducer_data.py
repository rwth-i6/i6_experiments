import copy
from typing import Dict, List, Optional

from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from sisyphus import tk

from i6_experiments.users.berger.systems.dataclasses import AlignmentData, FeatureType
from i6_experiments.users.berger.corpus.general.experiment_data import BasicSetupData
from i6_experiments.users.berger.corpus.general.hdf import build_feature_alignment_meta_dataset_config

from . import data


def get_tedlium2_data(
    alignments: Dict[str, AlignmentData],
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    train_key: str = "train",
    cv_keys: Optional[List[str]] = None,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    add_unknown: bool = False,
    augmented_lexicon: bool = True,
    eow_lexicon: bool = True,
    feature_type: FeatureType = FeatureType.GAMMATONE_16K,
    dc_detection: bool = False,
    **kwargs,
) -> BasicSetupData:
    if cv_keys is None:
        cv_keys = ["dev"]
    if dev_keys is None:
        dev_keys = ["dev"]
    if test_keys is None:
        test_keys = ["test"]

    # ********** Data inputs **********
    train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs = copy.deepcopy(
        data.get_data_inputs(
            ctc_lexicon=True,
            use_augmented_lexicon=augmented_lexicon,
            add_all_allophones=True,
            add_unknown_phoneme_and_mapping=add_unknown,
            filter_unk_from_corpus=True,
            eow_lexicon=eow_lexicon,
            **kwargs,
        )
    )

    # ********** Train data **********

    train_data_config = build_feature_alignment_meta_dataset_config(
        data_inputs=[train_data_inputs[train_key]],
        feature_type=feature_type,
        alignments=[alignments[f"{train_key}_align"]],
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
        rasr_binary_path=rasr_binary_path,
        rasr_arch=rasr_arch,
        dc_detection=dc_detection,
        extra_config={
            "partition_epoch": 5,
            "seq_ordering": "laplace:.1000",
        },
    )

    # ********** CV data **********

    cv_data_config = build_feature_alignment_meta_dataset_config(
        data_inputs=[cv_data_inputs[cv_key] for cv_key in cv_keys],
        feature_type=feature_type,
        alignments=[alignments[f"{cv_key}_align"] for cv_key in cv_keys],
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
        rasr_binary_path=rasr_binary_path,
        rasr_arch=rasr_arch,
        dc_detection=dc_detection,
        single_hdf=True,
        extra_config={
            "partition_epoch": 1,
            "seq_ordering": "sorted",
        },
    )

    # ********** Align data **********

    train_lexicon = train_data_inputs[train_key].lexicon.filename
    align_lexicon = AddEowPhonemesToLexiconJob(train_lexicon).out_lexicon

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input) for key, data_input in {**train_data_inputs, **cv_data_inputs}.items()
    }
    for data_input in align_data_inputs.values():
        data_input.lexicon.filename = align_lexicon

    return BasicSetupData(
        train_key=train_key,
        dev_keys=list(dev_data_inputs.keys()),
        test_keys=list(test_data_inputs.keys()),
        align_keys=[f"{train_key}_align", *[f"{key}_align" for key in cv_keys]],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
            **align_data_inputs,
        },
    )
