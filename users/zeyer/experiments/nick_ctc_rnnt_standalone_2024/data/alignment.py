from dataclasses import dataclass
from sisyphus import tk
from typing import List, Optional

from i6_core.returnn.hdf import RasrAlignmentDumpHDFJob

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict

from .common import build_training_datasets, TrainingDatasets, DatasetSettings, TrainingHDFLabelFiles
from ..default_tools import MINI_RETURNN_ROOT, RETURNN_EXE

@dataclass
class HDFAlignmentData:
    alignment_caches: [tk.Path]
    allophone_file: tk.Path
    state_tying_file: tk.Path


def build_hdf_from_alignment(
    prefix: str,
    alignment_data: HDFAlignmentData,
    returnn_root: tk.Path,
) -> List[tk.Path]:
    """

    :param prefix:
    :param alignment_data:
    :param returnn_root:
    :return: list of HDF files (one for each cache)
    """
    hdf_file_job = RasrAlignmentDumpHDFJob(
        alignment_caches=alignment_data.alignment_caches,
        allophone_file=alignment_data.allophone_file,
        state_tying_file=alignment_data.state_tying_file,
        returnn_root=returnn_root,
        # TODO: unclear if it should be sparse
    )
    hdf_file_job.add_alias(prefix + "/dump_hdf_job")
    return hdf_file_job.out_hdf_files


def build_rasr_alignment_target_training_datasets(
        prefix: str,
        librispeech_key: str,
        settings: DatasetSettings,
        train_alignment_caches: List[tk.Path],
        dev_clean_alignment_caches: List[tk.Path],
        dev_other_alignment_caches: List[tk.Path],
        allophone_file: tk.Path,
        state_tying_file: tk.Path,
) -> TrainingDatasets:
    """
    :param prefix:
    :param librispeech_key: which librispeech corpus to use
    :param settings: configuration object for the dataset pipeline
    """
    ogg_zip_dict = get_ogg_zip_dict(prefix, returnn_root=MINI_RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    train_ogg = ogg_zip_dict[librispeech_key]
    dev_clean_ogg = ogg_zip_dict["dev-clean"]
    dev_other_ogg = ogg_zip_dict["dev-other"]

    train_alignment_data = HDFAlignmentData(
        alignment_caches=train_alignment_caches,
        allophone_file=allophone_file,
        state_tying_file=state_tying_file
    )

    dev_clean_alignment_data = HDFAlignmentData(
        alignment_caches=dev_clean_alignment_caches,
        allophone_file=allophone_file,
        state_tying_file=state_tying_file
    )

    dev_other_alignment_data = HDFAlignmentData(
        alignment_caches=dev_other_alignment_caches,
        allophone_file=allophone_file,
        state_tying_file=state_tying_file
    )

    train_hdf_label_files = TrainingHDFLabelFiles(
        train=build_hdf_from_alignment(
            prefix=prefix + "/train",
            alignment_data=train_alignment_data,
            returnn_root=MINI_RETURNN_ROOT
        ),
        dev_clean=build_hdf_from_alignment(
            prefix=prefix + "/dev_clean",
            alignment_data=dev_clean_alignment_data,
            returnn_root=MINI_RETURNN_ROOT
        ),
        dev_other=build_hdf_from_alignment(
            prefix=prefix + "/dev_other",
            alignment_data=dev_other_alignment_data,
            returnn_root=MINI_RETURNN_ROOT
        ),
    )

    return build_training_datasets(
        train_ogg=train_ogg,
        dev_clean_ogg=dev_clean_ogg,
        dev_other_ogg=dev_other_ogg,
        settings=settings,
        label_datastream=None,
        training_hdf_label_files=train_hdf_label_files,
    )
