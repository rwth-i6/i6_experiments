from i6_core import corpus
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder
from sisyphus import tk
from typing import List
import i6_experiments.common.datasets.librispeech as lbs_dataset
from i6_experiments.users.berger.corpus.general.experiment_data import (
    PytorchHybridSetupData,
)

from .data import get_data_inputs, get_hdf_files


def get_hybrid_data(
    gmm_system: GmmSystem,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    train_key: str = "enhanced_tfgridnet_v0",
    cv_split: float = 0.01,
    dev_keys: List[str] = ["libri_css_enhanced_tfgridnet_v0"],
    test_keys: List[str] = [],
    lm_name: str = "4gram",
    use_stress: bool = False,
    add_unknown_phoneme_and_mapping: bool = True,
    ctc_lexicon: bool = False,
    use_augmented_lexicon: bool = True,
    add_all_allophones: bool = False,
) -> PytorchHybridSetupData:
    _, data_inputs, _ = get_data_inputs(
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
        "enhanced_tfgridnet_v0": 1,
        "enhanced_blstm_v0": 20,
    }

    for key in ["train", "cv"]:
        dataset_builder = MetaDatasetBuilder()
        for name, file in [
            ("primary", hdf_files.primary_features_file),
            ("secondary", hdf_files.secondary_features_file),
        ]:
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

    return PytorchHybridSetupData(
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        align_keys=[],
        train_data_config=data_config["train"],
        cv_data_config=data_config["cv"],
        data_inputs=data_inputs,
    )
