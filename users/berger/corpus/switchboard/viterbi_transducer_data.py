from typing import Dict, List, Optional
import copy

from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_experiments.users.berger.systems.dataclasses import AlignmentData, FeatureType
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder, hdf_config_dict_for_files
from . import data
from ..general import BasicSetupData, build_feature_alignment_meta_dataset_config
from sisyphus import tk


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
        add_all_allophones=True,
        **kwargs,
    )

    # ********** Train data **********

    if use_wei_data:
        config_file = tk.Path("/work/asr4/berger/dependencies/switchboard/data/wei_train_ctc/sprint.train.config")
        feature_flow_file = tk.Path("/work/asr4/berger/dependencies/switchboard/data/wei_train_ctc/train.feature.flow")
        train_feature_data_config = {
            "class": "ExternSprintDataset",
            "partitionEpoch": 6,
            "sprintConfigStr": f"--config={config_file} --*.LOGFILE=nn-trainer.train.log --*.TASK=1 "
            f"--*.corpus.segment-order-shuffle=true --*.segment-order-sort-by-time-length=true "
            f"--*.segment-order-sort-by-time-length-chunk-size=348 --feature-extraction.file={feature_flow_file}",
            "sprintTrainerExecPath": rasr_binary_path.join_right(f"nn-trainer.{rasr_arch}"),
        }

        dataset_builder = MetaDatasetBuilder()
        dataset_builder.add_dataset(
            name="data", dataset_config=train_feature_data_config, key_mapping={"data": "data"}, control=False
        )

        alignment_hdf_files = [
            alignments[f"{train_key}_align"].get_hdf(returnn_python_exe=returnn_python_exe, returnn_root=returnn_root)
        ]
        alignment_hdf_config = hdf_config_dict_for_files(files=alignment_hdf_files)
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

    if use_wei_data:
        config_file = tk.Path("/work/asr4/berger/dependencies/switchboard/data/wei_train_ctc/sprint.dev.config")
        feature_flow_file = tk.Path("/work/asr4/berger/dependencies/switchboard/data/wei_train_ctc/dev.feature.flow")
        cv_feature_data_config = {
            "class": "ExternSprintDataset",
            "partitionEpoch": 1,
            "sprintConfigStr": f"--config={config_file} --*.LOGFILE=nn-trainer.dev.log --*.TASK=1 "
            f"--*.corpus.segment-order-shuffle=true --*.segment-order-sort-by-time-length=true "
            f"--*.segment-order-sort-by-time-length-chunk-size=50 --feature-extraction.file={feature_flow_file}",
            "sprintTrainerExecPath": rasr_binary_path.join_right(f"nn-trainer.{rasr_arch}"),
        }

        dataset_builder = MetaDatasetBuilder()
        dataset_builder.add_dataset(
            name="data", dataset_config=cv_feature_data_config, key_mapping={"data": "data"}, control=False
        )

        alignment_hdf_files = [
            alignments[f"{key}_align"].get_hdf(returnn_python_exe=returnn_python_exe, returnn_root=returnn_root)
            for key in cv_keys
        ]
        alignment_hdf_config = hdf_config_dict_for_files(files=alignment_hdf_files)
        dataset_builder.add_dataset(
            name="classes", dataset_config=alignment_hdf_config, key_mapping={"data": "classes"}, control=True
        )
        cv_data_config = dataset_builder.get_dict()
    else:
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
