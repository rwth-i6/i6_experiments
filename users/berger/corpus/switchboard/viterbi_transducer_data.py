from typing import Dict, List, Optional
import copy

from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_experiments.users.berger.systems.dataclasses import AlignmentData, FeatureType
from . import data
from ..general import BasicSetupData, build_feature_alignment_meta_dataset_config, filter_unk_in_corpus_object
from sisyphus import tk


def get_librispeech_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    alignments: Dict[str, AlignmentData],
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    train_key: str = "train-other-960",
    cv_keys: Optional[List[str]] = None,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    use_wei_lexicon: bool = False,
    feature_type: FeatureType = FeatureType.SAMPLES,
    dc_detection: bool = False,
    **kwargs,
) -> BasicSetupData:
    if cv_keys is None:
        cv_keys = ["dev-clean", "dev-other"]
    if dev_keys is None:
        dev_keys = ["dev-clean", "dev-other"]
    if test_keys is None:
        test_keys = ["test-clean", "test-other"]

    # ********** Data inputs **********

    (
        train_data_inputs,
        cv_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = data.get_data_inputs(
        train_key=train_key,
        cv_keys=cv_keys,
        dev_keys=dev_keys,
        test_keys=test_keys,
        ctc_lexicon=True,
        use_wei_lexicon=use_wei_lexicon,
        add_all_allophones=True,
        audio_format="wav",  # Note: OGGZip dataset lead to length mismatches between features and alignment
        **kwargs,
    )

    # ********** Train data **********

    train_lexicon = train_data_inputs[train_key].lexicon.filename

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
            "partition_epoch": 20,
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

    # ********** Recog lexicon **********

    if use_wei_lexicon:
        recog_lexicon = tk.Path("/work/asr4/berger/dependencies/librispeech/lexicon/recog.lexicon.wei.xml")
    else:
        recog_lexicon = AddEowPhonemesToLexiconJob(train_lexicon).out_lexicon

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon.filename = recog_lexicon

    # ********** Align data **********

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input) for key, data_input in {**train_data_inputs, **cv_data_inputs}.items()
    }
    align_lexicon = AddEowPhonemesToLexiconJob(train_lexicon).out_lexicon
    for data_input in align_data_inputs.values():
        data_input.lexicon.filename = align_lexicon
        filter_unk_in_corpus_object(data_input.corpus_object, train_lexicon)  # TODO: Remove!

    return BasicSetupData(
        train_key=train_key,
        dev_keys=list(dev_data_inputs.keys()),
        test_keys=list(test_data_inputs.keys()),
        align_keys=[f"{train_key}_align", *[f"{cv_key}_align" for cv_key in cv_keys]],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
            **align_data_inputs,
        },
    )
