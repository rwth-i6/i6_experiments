from i6_core import corpus
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder
from sisyphus import tk
from typing import List
from i6_experiments.users.berger.corpus.general.experiment_data import (
    HybridSetupData,
)

from .data import get_data_inputs, get_hdf_files


def get_hybrid_data(
    train_key: str,
    gmm_system: GmmSystem,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    cv_split: float = 0.002,
    dev_keys: List[str] = [],
    test_keys: List[str] = [],
    lm_name: str = "4gram",
    use_stress: bool = False,
    add_unknown_phoneme_and_mapping: bool = True,
    ctc_lexicon: bool = False,
    use_augmented_lexicon: bool = True,
    add_all_allophones: bool = False,
    add_sec_audio: bool = False,
    add_mix_audio: bool = True,
) -> HybridSetupData:
    _, dev_data_inputs, test_data_inputs = get_data_inputs(
        dev_keys=dev_keys,
        test_keys=test_keys,
        add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
        use_stress=use_stress,
        ctc_lexicon=ctc_lexicon,
        use_augmented_lexicon=use_augmented_lexicon,
        add_all_allophones=add_all_allophones,
        lm_name=lm_name,
    )

    hdf_files = get_hdf_files(
        gmm_system=gmm_system,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
        rasr_binary_path=rasr_binary_path,
        rasr_arch=rasr_arch,
    )[train_key]

    split_segments = corpus.ShuffleAndSplitSegmentsJob(
        hdf_files.segments, {"train": 1.0 - cv_split, "cv": cv_split}
    ).out_segments

    data_config = {}

    partition = {
        "enhanced_tfgridnet_v1": 40,
        "enhanced_tfgridnet_v2": 40,
        "enhanced_blstm_v2": 40,
        "enhanced_blstm_v1": 40,
    }

    for key in ["train", "cv"]:
        dataset_builder = MetaDatasetBuilder()
        feature_list = [
            ("primary", hdf_files.primary_features_files),
        ]
        if add_sec_audio:
            feature_list.append(("secondary", hdf_files.secondary_features_files))
        if add_mix_audio:
            feature_list.append(("mix", hdf_files.mix_features_files))
        for name, file in feature_list:
            dataset_builder.add_hdf_dataset(
                file,
                name=f"features_{name}",
                key_mapping={"data": f"features_{name}"},
            )
        dataset_builder.add_hdf_dataset(
            hdf_files.alignments_file,
            dataset_config={
                "seq_ordering": "laplace:.1000",
                "partition_epoch": partition[train_key] if key == "train" else 1,
                "seq_list_filter_file": split_segments[key],
            },
            name="classes",
            key_mapping={"data": "classes"},
            control=True,
        )

        data_config[key] = dataset_builder.get_dict()

    return HybridSetupData(
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        align_keys=[],
        train_data_config=data_config["train"],
        cv_data_config=data_config["cv"],
        data_inputs={**dev_data_inputs, **test_data_inputs},
    )
