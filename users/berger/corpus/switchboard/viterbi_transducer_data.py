from typing import Dict, List, Optional
import copy

from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.returnn import ReturnnDumpHDFJob
from i6_experiments.users.berger.systems.dataclasses import AlignmentData, FeatureType
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder, hdf_config_dict_for_files
from . import data
from ..general import BasicSetupData, build_feature_alignment_meta_dataset_config
from i6_experiments.users.berger.recipe.returnn.hdf import MatchLengthsJob
from sisyphus import tk


def subsample_by_4(x):
    return -(-x // 4)


def get_switchboard_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    alignments: Dict[str, AlignmentData],
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    train_key: str = "train",
    cv_keys: Optional[List[str]] = None,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    feature_type: FeatureType = FeatureType.SAMPLES,
    dc_detection: bool = False,
    use_wei_data: bool = False,
    **kwargs,
) -> BasicSetupData:
    if cv_keys is None:
        cv_keys = ["hub5e00"]
    if dev_keys is None:
        dev_keys = ["hub5e00"]
    if test_keys is None:
        test_keys = ["hub5e01", "rt03s"]

    # ********** Data inputs **********

    (train_data_inputs, cv_data_inputs, dev_data_inputs, test_data_inputs,) = data.get_data_inputs(
        train_key=train_key,
        cv_keys=cv_keys,
        dev_keys=dev_keys,
        test_keys=test_keys,
        ctc_lexicon=True,
        add_all_allophones=True,
        **kwargs,
    )

    # ********** Train data **********

    if use_wei_data:
        train_feature_hdfs = []
        for idx in range(1, 101):
            dataset_config = {
                "class": "SprintCacheDataset",
                "data": {
                    "data": {
                        "filename": tk.Path(
                            f"/u/zhou/asr-exps/swb1/2021-12-09_phoneme-transducer/work/features/extraction/FeatureExtraction.Gammatone.OKQT9hEV3Zgd/output/gt.cache.{idx}"
                        ),
                        "data_type": "feat",
                    }
                },
            }

            hdf_file = ReturnnDumpHDFJob(
                dataset_config, returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
            ).out_hdf
            train_feature_hdfs.append(hdf_file)

        dataset_builder = MetaDatasetBuilder()
        feature_hdf_config = hdf_config_dict_for_files(train_feature_hdfs)
        dataset_builder.add_dataset(
            name="data", dataset_config=feature_hdf_config, key_mapping={"data": "data"}, control=False
        )

        alignment_hdf_files = [
            MatchLengthsJob(
                alignments[f"{train_key}_align"].get_hdf(
                    returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
                ),
                train_feature_hdfs,
                match_len_transform_func=subsample_by_4,
            ).out_hdf
        ]
        alignment_hdf_config = hdf_config_dict_for_files(
            files=alignment_hdf_files, extra_config={"partition_epoch": 6, "seq_ordering": "laplace:.348"}
        )
        dataset_builder.add_dataset(
            name="classes", dataset_config=alignment_hdf_config, key_mapping={"data": "classes"}, control=True
        )
        train_data_config = dataset_builder.get_dict()
        train_lexicon = tk.Path("/work/asr4/berger/dependencies/switchboard/lexicon/wei_train_ctc.lexicon.v2.xml")
    else:
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
                "partition_epoch": 6,
                "seq_ordering": "laplace:.1000",
            },
        )

        train_lexicon = train_data_inputs[train_key].lexicon.filename

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

    recog_lexicon = AddEowPhonemesToLexiconJob(
        train_lexicon, nonword_phones=["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"]
    ).out_lexicon

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon.filename = recog_lexicon

    # ********** Align data **********

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input) for key, data_input in {**train_data_inputs, **cv_data_inputs}.items()
    }
    align_lexicon = AddEowPhonemesToLexiconJob(
        train_lexicon, nonword_phones=["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"]
    ).out_lexicon

    for data_input in align_data_inputs.values():
        data_input.lexicon.filename = align_lexicon

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
