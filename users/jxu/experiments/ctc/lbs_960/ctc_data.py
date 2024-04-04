import copy
from typing import List

from sisyphus import tk


from i6_core import returnn, corpus
from i6_core.returnn.hdf import BlissToPcmHDFJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_experiments.users.berger.args.jobs.rasr_init_args import (
    get_feature_extraction_args_16kHz,
)
from i6_experiments.users.berger import helpers
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.helpers.hdf import build_rasr_feature_hdfs
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder
from i6_experiments.users.berger.corpus.general.experiment_data import PytorchCTCSetupData
from i6_experiments.users.berger.corpus.librispeech import data
from i6_experiments.users.berger.systems.dataclasses import FeatureType


def get_librispeech_data_hdf(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    train_key: str = "train-other-960",
    dev_keys: List[str] = ["dev-clean", "dev-other"],
    test_keys: List[str] = ["test-clean", "test-other"],
    add_unknown: bool = False,
    augmented_lexicon: bool = False,
    feature_type: FeatureType = FeatureType.GAMMATONE,
    blank_index_last: bool = False,
) -> PytorchCTCSetupData:

    if blank_index_last:
        from i6_experiments.users.jxu.experiments.ctc.lbs_960.utils.hdf import BlissCorpusToTargetHdfJob
    else:
        from i6_experiments.users.berger.recipe.returnn.hdf import BlissCorpusToTargetHdfJob
    # ********** Data inputs **********

    (train_data_inputs, dev_data_inputs, test_data_inputs,) = copy.deepcopy(data.get_data_inputs(
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        ctc_lexicon=True,
        use_augmented_lexicon=augmented_lexicon,
        add_all_allophones=True,
        audio_format="wav",
        add_unknown_phoneme_and_mapping=add_unknown,
    ))

    # ********** Train data **********

    train_corpus_object = train_data_inputs[train_key].corpus_object
    eow_lexicon = AddEowPhonemesToLexiconJob(train_data_inputs[train_key].lexicon.filename).out_lexicon
    assert train_corpus_object is not None

    if not add_unknown and not augmented_lexicon:
        train_corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
            train_corpus_object.corpus_file,
            eow_lexicon,
            all_unknown=False,
        ).out_corpus

    train_dataset_builder = MetaDatasetBuilder()
    if feature_type == FeatureType.GAMMATONE:
        gt_args = get_feature_extraction_args_16kHz()["gt"]
        train_feature_hdf = build_rasr_feature_hdfs(
            train_corpus_object,
            split=train_data_inputs[train_key].concurrent,
            feature_type="gt",
            feature_extraction_args=gt_args,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            single_hdf=True,
        )
    elif feature_type == FeatureType.SAMPLES:
        bliss_to_pcm_hdf_job = BlissToPcmHDFJob(
            train_corpus_object.corpus_file,
            rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
            returnn_root=returnn_root,
        )
        bliss_to_pcm_hdf_job.rqmt = {"cpu": 2, "mem": 8, "time": 8}
        train_feature_hdf = bliss_to_pcm_hdf_job.out_hdf
    else:
        raise NotImplementedError

    train_dataset_builder.add_hdf_dataset(
        train_feature_hdf,
        name="features",
        key_mapping={"data": "data"},
        dataset_config={
            "partition_epoch": 20,
            "seq_ordering": "laplace:.1000",
        },
        control=True,
    )
    if blank_index_last:
        train_targets_hdf = BlissCorpusToTargetHdfJob(
            train_corpus_object.corpus_file,
            bliss_lexicon=eow_lexicon,
            returnn_root=returnn_root,
            blank_index_last=True
        ).out_hdf
    else:
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

    cv_data_inputs = copy.deepcopy(dev_data_inputs)
    if not add_unknown:
        for corpus_object in [cv_data_inputs[key].corpus_object for key in dev_keys]:
            assert corpus_object.corpus_file is not None
            corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
                corpus_object.corpus_file,
                eow_lexicon,
                all_unknown=False,
            ).out_corpus

    cv_dataset_builder = MetaDatasetBuilder()
    cv_feature_hdf = []
    for key in dev_keys:
        if feature_type == FeatureType.GAMMATONE:
            gt_args = get_feature_extraction_args_16kHz()["gt"]
            cv_feature_hdf.extend(build_rasr_feature_hdfs(
                cv_data_inputs[key].corpus_object,
                split=cv_data_inputs[key].concurrent,
                feature_type="gt",
                feature_extraction_args=gt_args,
                returnn_python_exe=returnn_python_exe,
                returnn_root=returnn_root,
                rasr_binary_path=rasr_binary_path,
                rasr_arch=rasr_arch,
                single_hdf=True,
            ))
        elif feature_type == FeatureType.SAMPLES:
            bliss_to_pcm_hdf_job = BlissToPcmHDFJob(
                cv_data_inputs[key].corpus_object.corpus_file,
                rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
                returnn_root=returnn_root,
            )
            cv_feature_hdf.append(bliss_to_pcm_hdf_job.out_hdf)
        else:
            raise NotImplementedError

    cv_dataset_builder.add_hdf_dataset(
        cv_feature_hdf,
        name="features",
        key_mapping={"data": "data"},
        dataset_config={
            "partition_epoch": 1,
            "seq_ordering": "laplace:.1000",
        },
        control=True,
    )

    cv_targets_hdf = []
    for key in dev_keys:
        if blank_index_last:
            cv_targets_hdf.append(BlissCorpusToTargetHdfJob(
                cv_data_inputs[key].corpus_object.corpus_file,
                bliss_lexicon=eow_lexicon,
                returnn_root=returnn_root,
                blank_index_last=True
            ).out_hdf)
        else:
            cv_targets_hdf.append(BlissCorpusToTargetHdfJob(
                cv_data_inputs[key].corpus_object.corpus_file,
                bliss_lexicon=eow_lexicon,
                returnn_root=returnn_root,
            ).out_hdf)

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
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        align_keys=["train", "dev"],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
        },
    )
