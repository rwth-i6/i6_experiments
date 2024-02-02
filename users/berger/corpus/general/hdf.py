from typing import Optional, List

from i6_core.returnn.hdf import BlissToPcmHDFJob
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder, hdf_config_dict_for_files
from i6_experiments.users.berger.systems.dataclasses import AlignmentData, FeatureType
from sisyphus import tk
from i6_experiments.users.berger.args.jobs.rasr_init_args import (
    get_feature_extraction_args_16kHz,
    get_feature_extraction_args_8kHz,
)
from i6_experiments.users.berger.helpers import build_rasr_feature_hdfs, RasrDataInput


def build_feature_hdf_dataset_config(
    data_inputs: List[RasrDataInput],
    feature_type: FeatureType,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    dc_detection: bool = False,
    single_hdf: bool = False,
    extra_config: Optional[dict] = None,
) -> dict:
    feature_hdfs = []

    if feature_type in {
        FeatureType.GAMMATONE_16K,
        FeatureType.GAMMATONE_CACHED_16K,
        FeatureType.GAMMATONE_8K,
        FeatureType.GAMMATONE_CACHED_8K,
    }:
        if feature_type == FeatureType.GAMMATONE_16K or feature_type == FeatureType.GAMMATONE_CACHED_16K:
            gt_args = get_feature_extraction_args_16kHz(dc_detection=dc_detection)["gt"]
        elif feature_type == FeatureType.GAMMATONE_8K or feature_type == FeatureType.GAMMATONE_CACHED_8K:
            gt_args = get_feature_extraction_args_8kHz(dc_detection=dc_detection)["gt"]
        else:
            raise NotImplementedError

        for data_input in data_inputs:
            feature_hdfs += build_rasr_feature_hdfs(
                data_input.corpus_object,
                split=data_input.concurrent,
                feature_type="gt",
                feature_extraction_args=gt_args,
                returnn_python_exe=returnn_python_exe,
                returnn_root=returnn_root,
                rasr_binary_path=rasr_binary_path,
                rasr_arch=rasr_arch,
                single_hdf=single_hdf,
            )

    elif feature_type == FeatureType.SAMPLES:
        for data_input in data_inputs:
            feature_hdf_job = BlissToPcmHDFJob(
                data_input.corpus_object.corpus_file,
                rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
                returnn_root=returnn_root,
            )
            feature_hdf_job.rqmt["mem"] = 8
            feature_hdf_job.rqmt["time"] = 24
            feature_hdfs.append(feature_hdf_job.out_hdf)
    else:
        raise NotImplementedError

    return hdf_config_dict_for_files(files=feature_hdfs, extra_config=extra_config)


def build_feature_alignment_meta_dataset_config(
    data_inputs: List[RasrDataInput],
    feature_type: FeatureType,
    alignments: List[AlignmentData],
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    dc_detection: bool = False,
    single_hdf: bool = False,
    extra_config: Optional[dict] = None,
) -> dict:
    feature_hdf_config = build_feature_hdf_dataset_config(
        data_inputs=data_inputs,
        feature_type=feature_type,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
        rasr_binary_path=rasr_binary_path,
        rasr_arch=rasr_arch,
        dc_detection=dc_detection,
        single_hdf=single_hdf,
    )

    dataset_builder = MetaDatasetBuilder()
    dataset_builder.add_dataset(
        name="data", dataset_config=feature_hdf_config, key_mapping={"data": "data"}, control=False
    )

    alignment_hdf_files = [
        alignment.get_hdf(returnn_python_exe=returnn_python_exe, returnn_root=returnn_root) for alignment in alignments
    ]
    alignment_hdf_config = hdf_config_dict_for_files(files=alignment_hdf_files, extra_config=extra_config)
    dataset_builder.add_dataset(
        name="classes", dataset_config=alignment_hdf_config, key_mapping={"data": "classes"}, control=True
    )
    return dataset_builder.get_dict()
