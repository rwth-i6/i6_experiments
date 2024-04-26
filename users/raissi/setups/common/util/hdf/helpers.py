__all__ = ["hdf_config_dict_for_files", "MetaDatasetBuilder"]

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional


from sisyphus import Path, tk

from i6_experiments.users.raissi.args.rasr.features.init_args import (
    get_feature_extraction_args_16kHz,
    get_feature_extraction_args_8kHz,
)

from i6_experiments.common.setups.rasr.util import (
    RasrDataInput,
    ReturnnRasrDataInput
)

from i6_experiments.users.raissi.setups.common.features.taxonomy import FeatureInfo, FeatureType
from i6_experiments.users.raissi.setups.common.util.hdf.dump import build_rasr_feature_hdfs

@dataclass
class HDFAlignmentData:
    alignment_cache_bundle: tk.Path
    allophone_file: tk.Path
    state_tying_file: tk.Path
    silence_phone: str = "[SILENCE]"

    def get_hdf(self, returnn_python_exe: tk.Path, returnn_root: tk.Path) -> tk.Path:
        return build_hdf_from_alignment(
            alignment_cache=self.alignment_cache_bundle,
            allophone_file=self.allophone_file,
            state_tying_file=self.state_tying_file,
            silence_phone=self.silence_phone,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
        )


def hdf_config_dict_for_files(files: List[tk.Path], extra_config: Optional[Dict] = None) -> dict:
    config = {
        "class": "HDFDataset",
        "use_cache_manager": True,
        "files": files,
    }
    if extra_config:
        config.update(extra_config)
    return config


class MetaDatasetBuilder:
    def __init__(self) -> None:
        self.datasets = {}
        self.data_map = {}
        self.control_dataset = ""

    def add_dataset(
        self,
        name: str,
        dataset_config: Dict[str, Any],
        key_mapping: Dict[str, str],
        control: bool = False,
    ) -> None:
        self.datasets[name] = dataset_config
        for key, val in key_mapping.items():
            self.data_map[val] = (name, key)
        if control:
            self.control_dataset = name

    def get_dict(self) -> Dict[str, Any]:
        return {
            "class": "MetaDataset",
            "datasets": self.datasets,
            "data_map": self.data_map,
            "seq_order_control_dataset": self.control_dataset,
        }

def build_feature_hdf_dataset_config(
    data_inputs: List[ReturnnRasrDataInput],
    feature_info: FeatureInfo,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    dc_detection: bool = False,
    single_hdf: bool = False,
    extra_config: Optional[dict] = None,
) -> dict:
    feature_hdfs = []

    feature_name = feature_info.feature_type.get()

    if feature_info.feature_type is not FeatureType.samples:
        if feature_info.sampling_rate == 16000:
            feat_args = get_feature_extraction_args_16kHz(dc_detection=dc_detection)[feature_name]
        elif feature_info.sampling_rate == 8000:
            feat_args = get_feature_extraction_args_8kHz(dc_detection=dc_detection)[feature_name]
        else:
            raise NotImplementedError

        for data_input in data_inputs:
            feature_hdfs += build_rasr_feature_hdfs(
                data_input=data_input,
                feature_name=feature_name,
                feature_extraction_args=feat_args,
                returnn_python_exe=returnn_python_exe,
                returnn_root=returnn_root,
                single_hdf=single_hdf,
            )

    elif feature_info.feature_type == FeatureType.samples:
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


def build_feature_alignment_meta_dataset_config(
    data_inputs: List[ReturnnRasrDataInput],
    feature_info: FeatureInfo,
    alignments: List[HDFAlignmentData],
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    dc_detection: bool = False,
    single_hdf: bool = False,
    extra_config: Optional[dict] = None,
) -> dict:
    feature_hdf_config = build_feature_hdf_dataset_config(
        data_inputs=data_inputs,
        feature_info=feature_info,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
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


def build_feature_label_meta_dataset_config(
    data_inputs: List[RasrDataInput],
    feature_info: FeatureInfo,
    lexicon: tk.Path,
    label_dim: int,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    dc_detection: bool = False,
    single_hdf: bool = False,
    extra_config: Optional[dict] = None,
) -> dict:
    feature_hdf_config = build_feature_hdf_dataset_config(
        data_inputs=data_inputs,
        feature_info=feature_info,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_python_exe,
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

