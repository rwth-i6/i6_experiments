from i6_core import corpus
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_experiments.users.berger.args.jobs.rasr_init_args import (
    get_feature_extraction_args_16kHz,
)
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder
from i6_experiments.users.berger.corpus.general.experiment_data import (
    PytorchCTCSetupData,
)
from i6_experiments.users.berger.helpers.hdf import build_rasr_feature_hdf
from i6_experiments.users.berger.recipe.returnn.hdf import BlissCorpusToTargetHdfJob
from sisyphus import tk

from . import data


def get_tedlium2_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    add_unknown: bool = False,
    augmented_lexicon: bool = False,
) -> PytorchCTCSetupData:
    # ********** Data inputs **********

    train_data_inputs, dev_data_inputs, test_data_inputs = data.get_data_inputs(
        ctc_lexicon=True,
        use_augmented_lexicon=augmented_lexicon,
        add_all_allophones=True,
        add_unknown_phoneme_and_mapping=add_unknown,
    )

    # ********** Train data **********
    train_corpus_object = train_data_inputs["train"].corpus_object
    eow_lexicon = AddEowPhonemesToLexiconJob(train_data_inputs["train"].lexicon.filename).out_lexicon
    assert train_corpus_object.corpus_file is not None

    if not add_unknown:
        train_corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
            train_corpus_object.corpus_file,
            eow_lexicon,
            all_unknown=False,
        ).out_corpus

    gt_args = get_feature_extraction_args_16kHz()["gt"]

    train_dataset_builder = MetaDatasetBuilder()
    train_feature_hdf = build_rasr_feature_hdf(
        train_corpus_object,
        split=train_data_inputs["train"].concurrent,
        feature_type="gt",
        feature_extraction_args=gt_args,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        rasr_binary_path=rasr_binary_path,
        rasr_arch=rasr_arch,
    )
    train_dataset_builder.add_hdf_dataset(
        train_feature_hdf,
        name="features",
        key_mapping={"data": "data"},
    )

    train_targets_hdf = BlissCorpusToTargetHdfJob(
        train_corpus_object.corpus_file,
        bliss_lexicon=eow_lexicon,
        returnn_root=returnn_root,
    ).out_hdf
    train_dataset_builder.add_hdf_dataset(
        train_targets_hdf,
        seq_ordering="random",
        name="targets",
        key_mapping={"data": "targets"},
        dataset_config={"partition_epoch": 5},
        control=True,
    )

    train_data_config = train_dataset_builder.get_dict()

    # ********** CV data **********
    cv_corpus_object = dev_data_inputs["dev"].corpus_object

    if not add_unknown:
        cv_corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
            cv_corpus_object.corpus_file,
            eow_lexicon,
            all_unknown=False,
        ).out_corpus

    cv_dataset_builder = MetaDatasetBuilder()
    cv_feature_hdf = build_rasr_feature_hdf(
        cv_corpus_object,
        split=1,
        feature_type="gt",
        feature_extraction_args=gt_args,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        rasr_binary_path=rasr_binary_path,
        rasr_arch=rasr_arch,
    )
    cv_dataset_builder.add_hdf_dataset(
        cv_feature_hdf,
        name="features",
        key_mapping={"data": "data"},
    )

    cv_targets_hdf = BlissCorpusToTargetHdfJob(
        cv_corpus_object.corpus_file,
        bliss_lexicon=eow_lexicon,
        returnn_root=returnn_root,
    ).out_hdf
    cv_dataset_builder.add_hdf_dataset(
        cv_targets_hdf,
        seq_ordering="sorted",
        name="targets",
        key_mapping={"data": "targets"},
        dataset_config={"partition_epoch": 1},
        control=True,
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
