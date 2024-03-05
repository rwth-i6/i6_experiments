from typing import List, Optional
import copy

from i6_core import corpus
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_experiments.users.berger.systems.dataclasses import FeatureType
from . import data
from ..general import CTCSetupData, filter_unk_in_corpus_object, build_feature_hdf_dataset_config
from sisyphus import tk


def get_librispeech_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
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
) -> CTCSetupData:
    if cv_keys is None:
        cv_keys = ["dev-clean", "dev-other"]
    if dev_keys is None:
        dev_keys = ["dev-clean", "dev-other"]
    if test_keys is None:
        test_keys = ["test-clean", "test-other"]

    # ********** Data inputs **********

    train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs = data.get_data_inputs(
        train_key=train_key,
        cv_keys=cv_keys,
        dev_keys=dev_keys,
        test_keys=test_keys,
        ctc_lexicon=True,
        use_wei_lexicon=use_wei_lexicon,
        add_all_allophones=True,
        audio_format="wav",
        **kwargs,
    )

    # ********** Train data **********

    train_lexicon = train_data_inputs[train_key].lexicon.filename

    train_data_config = build_feature_hdf_dataset_config(
        data_inputs=[train_data_inputs[train_key]],
        feature_type=feature_type,
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

    cv_data_config = build_feature_hdf_dataset_config(
        data_inputs=[cv_data_inputs[key] for key in cv_keys],
        feature_type=feature_type,
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

    # ********** Loss corpus **********

    loss_corpus = corpus.MergeCorporaJob(
        [train_data_inputs[train_key].corpus_object.corpus_file]
        + [cv_data_inputs[key].corpus_object.corpus_file for key in cv_keys],
        name="loss-corpus",
        merge_strategy=corpus.MergeStrategy.SUBCORPORA,
    ).out_merged_corpus
    loss_lexicon = train_lexicon

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
        # filter_unk_in_corpus_object(data_input.corpus_object, train_lexicon)  # TODO: Remove!

    return CTCSetupData(
        train_key=train_key,
        dev_keys=list(dev_data_inputs.keys()),
        test_keys=list(test_data_inputs.keys()),
        align_keys=[f"{train_key}_align", *[f"{key}_align" for key in cv_keys]],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        loss_corpus=loss_corpus,
        loss_lexicon=loss_lexicon,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
            **align_data_inputs,
        },
    )
