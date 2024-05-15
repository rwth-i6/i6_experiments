from typing import List, Optional
import copy

from i6_core import corpus
from i6_core.returnn.hdf import BlissToPcmHDFJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_experiments.users.berger.systems.dataclasses import FeatureType
from i6_experiments.users.jxu.experiments.ctc.swb.data import data
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder, hdf_config_dict_for_files
from i6_experiments.users.berger.recipe.returnn.hdf import BlissCorpusToTargetHdfJob
from i6_experiments.users.berger.corpus.general.experiment_data import PytorchCTCSetupData

from sisyphus import tk


def get_switchboard_data(
        returnn_root: tk.Path,
        train_key: str = "train",
        cv_keys: Optional[List[str]] = None,
        dev_keys: Optional[List[str]] = None,
        test_keys: Optional[List[str]] = None,
        feature_type: FeatureType = FeatureType.SAMPLES,
        dc_detection: bool = False,
        add_unknown: bool = False,
        augmented_lexicon: bool = False,
        **kwargs,
) -> PytorchCTCSetupData:
    if cv_keys is None:
        cv_keys = ["hub5e00"]
    if dev_keys is None:
        dev_keys = ["hub5e00"]
    if test_keys is None:
        test_keys = ["hub5e01", "rt03s"]

    # ********** Data inputs **********

    train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs = data.get_data_inputs(
        train_key=train_key,
        cv_keys=cv_keys,
        dev_keys=dev_keys,
        test_keys=test_keys,
        ctc_lexicon=True,
        add_all_allophones=True,
        augmented_lexicon=augmented_lexicon,
        **kwargs,
    )

    # ********** Train data **********
    train_corpus_object = train_data_inputs[train_key].corpus_object
    eow_lexicon = AddEowPhonemesToLexiconJob(train_data_inputs[train_key].lexicon.filename,
                                             nonword_phones=["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"]).out_lexicon
    assert train_corpus_object is not None

    if not add_unknown and not augmented_lexicon:
        train_corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
            train_corpus_object.corpus_file,
            eow_lexicon,
            all_unknown=False,
        ).out_corpus

    train_dataset_builder = MetaDatasetBuilder()

    if feature_type is not FeatureType.SAMPLES:
        raise NotImplementedError("Currently only support feature types sample")

    bliss_to_pcm_hdf_job = BlissToPcmHDFJob(
        train_corpus_object.corpus_file,
        rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
        returnn_root=returnn_root,
    )
    bliss_to_pcm_hdf_job.rqmt = {"cpu": 2, "mem": 8, "time": 8}
    train_feature_hdf = bliss_to_pcm_hdf_job.out_hdf

    train_dataset_builder.add_dataset(
        name="features",
        key_mapping={"data": "data"},
        dataset_config=hdf_config_dict_for_files(
            [train_feature_hdf],
            {
                "partition_epoch": 6,
                "seq_ordering": "laplace:.1000",
            }),
        control=True,
    )

    train_targets_hdf = BlissCorpusToTargetHdfJob(
        train_corpus_object.corpus_file,
        bliss_lexicon=eow_lexicon,
        returnn_root=returnn_root,
    ).out_hdf
    train_dataset_builder.add_dataset(
        name="targets",
        dataset_config=hdf_config_dict_for_files([train_targets_hdf]),
        key_mapping={"data": "targets"},
        control=False,
    )
    train_data_config = train_dataset_builder.get_dict()

    # ********** CV data **********
    cv_dataset_builder = MetaDatasetBuilder()
    cv_feature_hdf = []

    cv_data_inputs = copy.deepcopy(cv_data_inputs)
    for key in dev_keys:
        if not add_unknown:
            for corpus_object in [cv_data_inputs[key].corpus_object for key in dev_keys]:
                assert corpus_object.corpus_file is not None
                corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
                    corpus_object.corpus_file,
                    eow_lexicon,
                    all_unknown=False,
                ).out_corpus

        bliss_to_pcm_hdf_job = BlissToPcmHDFJob(
            cv_data_inputs[key].corpus_object.corpus_file,
            rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
            returnn_root=returnn_root,
        )
        cv_feature_hdf.append(bliss_to_pcm_hdf_job.out_hdf)

    cv_dataset_builder.add_dataset(
        name="features",
        key_mapping={"data": "data"},
        dataset_config=hdf_config_dict_for_files(
            cv_feature_hdf,
            {
                "partition_epoch": 1,
                "seq_ordering": "laplace:.1000",
            }),
        control=True,
    )

    cv_targets_hdf = []
    for key in dev_keys:
        cv_targets_hdf.append(BlissCorpusToTargetHdfJob(
            cv_data_inputs[key].corpus_object.corpus_file,
            bliss_lexicon=eow_lexicon,
            returnn_root=returnn_root,
        ).out_hdf)

    cv_dataset_builder.add_dataset(
        name="targets",
        dataset_config=hdf_config_dict_for_files(cv_targets_hdf),
        key_mapping={"data": "targets"},
        control=False,
    )
    cv_data_config = cv_dataset_builder.get_dict()

    return PytorchCTCSetupData(
        train_key=train_key,
        dev_keys=list(dev_data_inputs.keys()),
        test_keys=list(test_data_inputs.keys()),
        align_keys=["train", "hub5e00_zoltan_4gram", "hub5e01_zoltan_4gram"],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
        },
    )
