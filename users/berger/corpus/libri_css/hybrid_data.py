from i6_core import corpus
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder, hdf_config_dict_for_files
from sisyphus import tk
from typing import List, Optional
from i6_experiments.users.berger.systems.dataclasses import FeatureType
from i6_experiments.users.berger.corpus.general.experiment_data import (
    HybridSetupData,
)

from .data import get_data_inputs, get_clean_align_hdf
from ..general import build_multi_speaker_feature_hdf_files


def get_hybrid_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    train_key: str = "enhanced_tfgridnet_v1",
    cv_split: float = 0.002,
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    feature_type: FeatureType = FeatureType.CONCAT_MIX_GAMMATONE_16K,
    **kwargs,
) -> HybridSetupData:
    if dev_keys is None:
        dev_keys = ["segmented_libri_css_tfgridnet_dev_v1"]
    if test_keys is None:
        test_keys = ["segmented_libri_css_tfgridnet_eval_v1"]

    train_data_inputs, dev_data_inputs, test_data_inputs = get_data_inputs(
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        **kwargs,
    )

    train_hdfs = build_multi_speaker_feature_hdf_files(
        data_inputs=[train_data_inputs[train_key]],
        feature_type=feature_type,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
        rasr_binary_path=rasr_binary_path,
        rasr_arch=rasr_arch,
        dc_detection=False,
        single_hdf=False,
    )

    align_hdf = get_clean_align_hdf(
        corpus_key=train_key,
        feature_hdfs=train_hdfs["primary"],
        returnn_root=returnn_root,
    )

    all_segments = corpus.SegmentCorpusJob(
        train_data_inputs[train_key].corpus_object.get_primary_corpus_object().corpus_file, 1
    ).out_single_segment_files[1]

    split_segments = corpus.ShuffleAndSplitSegmentsJob(
        all_segments, {"train": 1.0 - cv_split, "cv": cv_split}
    ).out_segments

    dataset_builder = MetaDatasetBuilder()
    for key, hdf_list in train_hdfs.items():
        dataset_builder.add_dataset(
            name=f"features_{key}",
            dataset_config=hdf_config_dict_for_files(files=hdf_list),
            key_mapping={"data": f"features_{key}"},
        )

    dataset_builder.add_dataset(
        name="classes",
        dataset_config=hdf_config_dict_for_files(
            files=[align_hdf],
            extra_config={
                "seq_ordering": "laplace:.1000",
                "partition_epoch": 40,
                "seq_list_filter_file": split_segments["train"],
            },
        ),
        key_mapping={"data": "classes"},
        control=True,
    )

    train_data_config = dataset_builder.get_dict()

    # overwrite for cv
    dataset_builder.add_dataset(
        name="classes",
        dataset_config=hdf_config_dict_for_files(
            files=[align_hdf],
            extra_config={
                "seq_ordering": "sorted",
                "partition_epoch": 1,
                "seq_list_filter_file": split_segments["cv"],
            },
        ),
        key_mapping={"data": "classes"},
        control=True,
    )

    cv_data_config = dataset_builder.get_dict()

    return HybridSetupData(
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        align_keys=[],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        data_inputs={**train_data_inputs, **dev_data_inputs, **test_data_inputs},
    )
