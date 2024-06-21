import copy
from i6_core import corpus
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.returnn.hdf import BlissToPcmHDFJob
from i6_experiments.users.berger.args.jobs.rasr_init_args import (
    get_feature_extraction_args_16kHz,
)
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder, hdf_config_dict_for_files
from i6_experiments.users.berger.corpus.general.experiment_data import (
    CTCSetupData,
    PytorchCTCSetupData,
)
from i6_experiments.users.berger.helpers.hdf import build_rasr_feature_hdfs
from i6_experiments.users.berger.recipe.returnn.hdf import BlissCorpusToTargetHdfJob
from i6_experiments.users.berger.systems.dataclasses import FeatureType
from i6_experiments.users.jxu.corpus.tedlium2 import data
from sisyphus import tk


def get_tedlium2_pytorch_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    add_unknown: bool = False,
    augmented_lexicon: bool = True,
    feature_type: FeatureType = FeatureType.GAMMATONE_16K,
) -> PytorchCTCSetupData:
    # ********** Data inputs **********
    train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs = copy.deepcopy(
        data.get_data_inputs(
            ctc_lexicon=True,
            use_augmented_lexicon=augmented_lexicon,
            add_all_allophones=True,
            add_unknown_phoneme_and_mapping=add_unknown,
        )
    )

    # ********** Train data **********
    train_corpus_object = train_data_inputs["train"].corpus_object
    eow_lexicon = AddEowPhonemesToLexiconJob(train_data_inputs["train"].lexicon.filename).out_lexicon
    assert train_corpus_object.corpus_file is not None

    if not add_unknown and not augmented_lexicon:
        train_corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
            train_corpus_object.corpus_file,
            eow_lexicon,
            all_unknown=False,
        ).out_corpus

    train_dataset_builder = MetaDatasetBuilder()
    if feature_type == FeatureType.GAMMATONE_16K:
        gt_args = get_feature_extraction_args_16kHz()["gt"]
        train_feature_hdf = build_rasr_feature_hdfs(
            train_corpus_object,
            split=train_data_inputs["train"].concurrent,
            feature_type="gt",
            feature_extraction_args=gt_args,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
        )
    elif feature_type == FeatureType.SAMPLES:
        train_feature_hdf = BlissToPcmHDFJob(train_corpus_object.corpus_file, returnn_root=returnn_root).out_hdf
    else:
        raise NotImplementedError

    train_dataset_builder.add_dataset(
        dataset_config=hdf_config_dict_for_files([train_feature_hdf]),
        name="features",
        key_mapping={"data": "data"},
    )

    train_targets_hdf = BlissCorpusToTargetHdfJob(
        train_corpus_object.corpus_file,
        bliss_lexicon=eow_lexicon,
        returnn_root=returnn_root,
    ).out_hdf
    train_dataset_builder.add_dataset(
        dataset_config=hdf_config_dict_for_files([train_targets_hdf], {"partition_epoch": 5, "seq_ordering":"laplace:.1000"}),
        name="targets",
        key_mapping={"data": "targets"},
        control=True,
    )

    train_data_config = train_dataset_builder.get_dict()

    # ********** CV data **********
    cv_corpus_object = copy.deepcopy(dev_data_inputs["dev_4gram"].corpus_object)

    if not add_unknown:
        cv_corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
            cv_corpus_object.corpus_file,
            eow_lexicon,
            all_unknown=False,
        ).out_corpus

    cv_dataset_builder = MetaDatasetBuilder()

    if feature_type == FeatureType.GAMMATONE_16K:
        gt_args = get_feature_extraction_args_16kHz()["gt"]
        cv_feature_hdf = build_rasr_feature_hdfs(
            cv_corpus_object,
            split=1,
            feature_type="gt",
            feature_extraction_args=gt_args,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
        )
    elif feature_type == FeatureType.SAMPLES:
        cv_feature_hdf = BlissToPcmHDFJob(cv_corpus_object.corpus_file, returnn_root=returnn_root).out_hdf
    else:
        raise NotImplementedError

    cv_dataset_builder.add_dataset(
        dataset_config=hdf_config_dict_for_files([cv_feature_hdf]),
        name="features",
        key_mapping={"data": "data"},
    )

    cv_targets_hdf = BlissCorpusToTargetHdfJob(
        cv_corpus_object.corpus_file,
        bliss_lexicon=eow_lexicon,
        returnn_root=returnn_root,
    ).out_hdf
    cv_dataset_builder.add_dataset(
        dataset_config=hdf_config_dict_for_files([cv_targets_hdf], {"partition_epoch": 1, "seq_ordering": "sorted"}),
        name="targets",
        key_mapping={"data": "targets"},
        control=True,
    )

    cv_data_config = cv_dataset_builder.get_dict()

    # ********** Recog lexicon **********

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon.filename = eow_lexicon

    return PytorchCTCSetupData(
        train_key="train",
        dev_keys=["dev_4gram"],
        test_keys=["test_4gram"],
        align_keys=["train", "dev"],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
        },
    )
