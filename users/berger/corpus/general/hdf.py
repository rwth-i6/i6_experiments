from typing import Dict, List, Optional

from i6_core.corpus import SegmentCorpusJob
from i6_core.returnn.hdf import BlissToPcmHDFJob
from sisyphus import tk

from i6_experiments.users.berger.args.jobs.rasr_init_args import (
    get_feature_extraction_args_8kHz,
    get_feature_extraction_args_16kHz,
)
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder, hdf_config_dict_for_files
from i6_experiments.users.berger.helpers import RasrDataInput, SeparatedCorpusObject, build_rasr_feature_hdfs
from i6_experiments.users.berger.recipe.corpus.transform import ReplaceUnknownWordsJob
from i6_experiments.users.berger.recipe.returnn.hdf import BlissCorpusToTargetHdfJob, RemoveBlanksFromAlignmentHdfJob
from i6_experiments.users.berger.systems.dataclasses import AlignmentData, FeatureType


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
    elif feature_type == FeatureType.LOGMEL_16K:
        logmel_args = get_feature_extraction_args_16kHz(dc_detection=dc_detection)["filterbank"]
        for data_input in data_inputs:
            feature_hdfs += build_rasr_feature_hdfs(
                data_input.corpus_object,
                split=data_input.concurrent,
                feature_type="fb",
                feature_extraction_args=logmel_args,
                returnn_python_exe=returnn_python_exe,
                returnn_root=returnn_root,
                rasr_binary_path=rasr_binary_path,
                rasr_arch=rasr_arch,
                single_hdf=single_hdf,
            )
    elif feature_type == FeatureType.SAMPLES:
        for data_input in data_inputs:
            if single_hdf:
                segment_files = [None]
            else:
                segment_files = list(
                    SegmentCorpusJob(
                        data_input.corpus_object.corpus_file, data_input.concurrent
                    ).out_single_segment_files.values()
                )

            for segment_file in segment_files:
                feature_hdf_job = BlissToPcmHDFJob(
                    data_input.corpus_object.corpus_file,
                    segment_file=segment_file,
                    rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
                    returnn_root=returnn_root,
                )
                feature_hdf_job.rqmt["mem"] = 8
                feature_hdf_job.rqmt["time"] = 24
                feature_hdfs.append(feature_hdf_job.out_hdf)
    else:
        raise NotImplementedError

    return hdf_config_dict_for_files(files=feature_hdfs, extra_config=extra_config)


def subsample_by_4(x):
    return -(-x // 4)


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
    remove_blank_idx: Optional[int] = None,
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
    if remove_blank_idx is not None:
        alignment_hdf_files = [
            RemoveBlanksFromAlignmentHdfJob(hdf_file, remove_blank_idx).out_hdf for hdf_file in alignment_hdf_files
        ]
    alignment_hdf_config = hdf_config_dict_for_files(files=alignment_hdf_files, extra_config=extra_config)
    dataset_builder.add_dataset(
        name="classes", dataset_config=alignment_hdf_config, key_mapping={"data": "classes"}, control=True
    )
    return dataset_builder.get_dict()


def build_multi_speaker_feature_hdf_files(
    data_inputs: List[RasrDataInput],
    feature_type: FeatureType,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    dc_detection: bool = False,
    single_hdf: bool = False,
) -> dict:
    feature_hdfs = {}

    if feature_type in {
        FeatureType.CONCAT_SEC_GAMMATONE_16K,
        FeatureType.CONCAT_MIX_GAMMATONE_16K,
        FeatureType.CONCAT_SEC_MIX_GAMMATONE_16K,
    }:
        gt_args = get_feature_extraction_args_16kHz(dc_detection=dc_detection)["gt"]

        feature_hdfs_prim = []
        feature_hdfs_sec = []
        feature_hdfs_mix = []

        for data_input in data_inputs:
            assert isinstance(data_input.corpus_object, SeparatedCorpusObject)
            for hdfs_list, subobject in [
                (feature_hdfs_prim, data_input.corpus_object.get_primary_corpus_object()),
                (feature_hdfs_sec, data_input.corpus_object.get_secondary_corpus_object()),
                (feature_hdfs_mix, data_input.corpus_object.get_mix_corpus_object()),
            ]:
                hdfs_list += build_rasr_feature_hdfs(
                    subobject,
                    split=data_input.concurrent,
                    feature_type="gt",
                    feature_extraction_args=gt_args,
                    returnn_python_exe=returnn_python_exe,
                    returnn_root=returnn_root,
                    rasr_binary_path=rasr_binary_path,
                    rasr_arch=rasr_arch,
                    single_hdf=single_hdf,
                )
        feature_hdfs["primary"] = feature_hdfs_prim
        if feature_type in {FeatureType.CONCAT_SEC_GAMMATONE_16K, FeatureType.CONCAT_SEC_MIX_GAMMATONE_16K}:
            feature_hdfs["secondary"] = feature_hdfs_sec
        if feature_type in {FeatureType.CONCAT_MIX_GAMMATONE_16K, FeatureType.CONCAT_SEC_MIX_GAMMATONE_16K}:
            feature_hdfs["mix"] = feature_hdfs_sec
    else:
        raise NotImplementedError

    return feature_hdfs


def build_feature_label_meta_dataset_config(
    data_inputs: List[RasrDataInput],
    feature_type: FeatureType,
    lexicon: tk.Path,
    label_dim: int,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    dc_detection: bool = False,
    single_hdf: bool = False,
    extra_config: Optional[dict] = None,
    segment_files: Optional[Dict[int, tk.Path]] = None,
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
        extra_config=extra_config,
    )

    dataset_builder = MetaDatasetBuilder()
    dataset_builder.add_dataset(
        name="data", dataset_config=feature_hdf_config, key_mapping={"data": "data"}, control=True
    )

    label_hdf_files = [
        BlissCorpusToTargetHdfJob(
            ReplaceUnknownWordsJob(data_input.corpus_object.corpus_file, lexicon_file=lexicon).out_corpus_file,
            bliss_lexicon=lexicon,
            returnn_root=returnn_root,
            dim=label_dim,
        ).out_hdf
        for data_input in data_inputs
    ]
    label_hdf_config = hdf_config_dict_for_files(files=label_hdf_files)
    dataset_builder.add_dataset(
        name="classes", dataset_config=label_hdf_config, key_mapping={"data": "classes"}, control=False
    )
    return dataset_builder.get_dict()
