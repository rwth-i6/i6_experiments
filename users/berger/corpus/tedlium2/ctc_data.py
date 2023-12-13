import copy
from i6_core import corpus
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.returnn.hdf import BlissToPcmHDFJob
from i6_experiments.users.berger.args.jobs.rasr_init_args import (
    get_feature_extraction_args_16kHz,
)
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder
from i6_experiments.users.berger.corpus.general.experiment_data import (
    CTCSetupData,
    PytorchCTCSetupData,
)
from i6_experiments.users.berger.helpers.hdf import build_rasr_feature_hdfs
from i6_experiments.users.berger.recipe.returnn.hdf import BlissCorpusToTargetHdfJob
from i6_experiments.users.berger.systems.dataclasses import FeatureType
from sisyphus import tk

from . import data


def get_tedlium2_pytorch_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    add_unknown: bool = False,
    augmented_lexicon: bool = True,
    feature_type: FeatureType = FeatureType.GAMMATONE,
) -> PytorchCTCSetupData:
    # ********** Data inputs **********
    train_data_inputs, dev_data_inputs, test_data_inputs = copy.deepcopy(
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
    if feature_type == FeatureType.GAMMATONE_16K or feature_type == FeatureType.GAMMATONE_CACHED_16K:
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
            single_hdf=True,
        )
    elif feature_type == FeatureType.SAMPLES:
        train_feature_hdf = BlissToPcmHDFJob(
            train_corpus_object.corpus_file,
            rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
            returnn_root=returnn_root,
        ).out_hdf
    else:
        raise NotImplementedError

    train_dataset_builder.add_hdf_dataset(
        train_feature_hdf,
        name="features",
        key_mapping={"data": "data"},
        dataset_config={
            "partition_epoch": 5,
            "seq_ordering": "laplace:.1000",
        },
        control=True,
    )

    train_targets_hdf = BlissCorpusToTargetHdfJob(
        train_corpus_object.corpus_file,
        bliss_lexicon=eow_lexicon,
        returnn_root=returnn_root,
    ).out_hdf
    train_dataset_builder.add_hdf_dataset(
        train_targets_hdf,
        name="targets",
        key_mapping={"data": "targets"},
        control=False,
    )

    train_data_config = train_dataset_builder.get_dict()

    # ********** CV data **********
    cv_corpus_object = copy.deepcopy(dev_data_inputs["dev"].corpus_object)

    if not add_unknown:
        cv_corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
            cv_corpus_object.corpus_file,
            eow_lexicon,
            all_unknown=False,
        ).out_corpus

    cv_dataset_builder = MetaDatasetBuilder()

    if feature_type == FeatureType.GAMMATONE_16K or feature_type == FeatureType.GAMMATONE_CACHED_16K:
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
            single_hdf=True,
        )
    elif feature_type == FeatureType.SAMPLES:
        cv_feature_hdf = BlissToPcmHDFJob(
            cv_corpus_object.corpus_file,
            rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
            returnn_root=returnn_root,
        ).out_hdf
    else:
        raise NotImplementedError

    cv_dataset_builder.add_hdf_dataset(
        cv_feature_hdf,
        name="features",
        key_mapping={"data": "data"},
        dataset_config={
            "partition_epoch": 1,
            "seq_ordering": "sorted",
        },
        control=True,
    )

    cv_targets_hdf = BlissCorpusToTargetHdfJob(
        cv_corpus_object.corpus_file,
        bliss_lexicon=eow_lexicon,
        returnn_root=returnn_root,
    ).out_hdf
    cv_dataset_builder.add_hdf_dataset(
        cv_targets_hdf,
        name="targets",
        key_mapping={"data": "targets"},
        control=False,
    )

    cv_data_config = cv_dataset_builder.get_dict()

    # ********** Recog lexicon **********

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon.filename = eow_lexicon

    return PytorchCTCSetupData(
        train_key="train",
        dev_keys=["dev"],
        test_keys=["test"],
        align_keys=["train", "dev"],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
        },
    )


def get_tedlium2_tf_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    add_unknown: bool = False,
    augmented_lexicon: bool = True,
    feature_type: FeatureType = FeatureType.GAMMATONE,
) -> CTCSetupData:
    # ********** Data inputs **********
    train_data_inputs, dev_data_inputs, test_data_inputs = copy.deepcopy(
        data.get_data_inputs(
            ctc_lexicon=True,
            use_augmented_lexicon=augmented_lexicon,
            add_all_allophones=True,
            add_unknown_phoneme_and_mapping=add_unknown,
        )
    )

    # ********** Train data **********
    train_corpus_object = train_data_inputs["train"].corpus_object
    train_lexicon = train_data_inputs["train"].lexicon.filename
    eow_lexicon = AddEowPhonemesToLexiconJob(train_lexicon).out_lexicon
    assert train_corpus_object.corpus_file is not None

    if not add_unknown and not augmented_lexicon:
        train_corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
            train_corpus_object.corpus_file,
            eow_lexicon,
            all_unknown=False,
        ).out_corpus

    if feature_type == FeatureType.GAMMATONE_16K or feature_type == FeatureType.GAMMATONE_CACHED_16K:
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
            single_hdf=True,
        )
    elif feature_type == FeatureType.SAMPLES:
        train_feature_hdf = BlissToPcmHDFJob(
            train_corpus_object.corpus_file,
            rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
            returnn_root=returnn_root,
        ).out_hdf
    else:
        raise NotImplementedError

    train_data_config = {
        "class": "HDFDataset",
        "files": train_feature_hdf,
        "partition_epoch": 5,
        "seq_ordering": "laplace:.30",
        "use_cache_manager": True,
    }

    # ********** CV data **********
    cv_corpus_object = copy.deepcopy(dev_data_inputs["dev"].corpus_object)

    if not add_unknown:
        cv_corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
            cv_corpus_object.corpus_file,
            eow_lexicon,
            all_unknown=False,
        ).out_corpus

    if feature_type == FeatureType.GAMMATONE_16K or feature_type == FeatureType.GAMMATONE_CACHED_16K:
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
            single_hdf=True,
        )
    elif feature_type == FeatureType.SAMPLES:
        cv_feature_hdf = BlissToPcmHDFJob(
            cv_corpus_object.corpus_file,
            rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
            returnn_root=returnn_root,
        ).out_hdf
    else:
        raise NotImplementedError

    cv_data_config = {
        "class": "HDFDataset",
        "files": cv_feature_hdf,
        "partition_epoch": 1,
        "seq_ordering": "sorted",
        "use_cache_manager": True,
    }

    # ********** Loss corpus **********

    loss_corpus = corpus.MergeCorporaJob(
        [train_corpus_object.corpus_file, cv_corpus_object.corpus_file],
        name="TED-LIUM-realease2",
        merge_strategy=corpus.MergeStrategy.CONCATENATE,
    ).out_merged_corpus
    loss_lexicon = train_lexicon

    # ********** Recog lexicon **********

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon.filename = eow_lexicon

    # ********** Align data **********

    align_lexicon = copy.deepcopy(eow_lexicon)

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input)
        for key, data_input in {**train_data_inputs, **dev_data_inputs}.items()
    }
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
        train_key="train",
        dev_keys=["dev"],
        test_keys=["test"],
        align_keys=["train_align", "dev_align"],
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
