from typing import Any, Dict
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.meta.system import CorpusObject
import i6_core.rasr as rasr
import i6_core.features as features
from sisyphus import tk

from i6_core.returnn import ReturnnDumpHDFJob


def build_hdf_from_alignment(
    alignment_cache: tk.Path,
    allophone_file: tk.Path,
    state_tying_file: tk.Path,
    silence_phone: str = "[SILENCE]",
):
    dataset_config = {
        "class": "SprintCacheDataset",
        "data": {
            "data": {
                "filename": alignment_cache,
                "data_type": "align",
                "allophone_labeling": {
                    "silence_phone": silence_phone,
                    "allophone_file": allophone_file,
                    "state_tying_file": state_tying_file,
                },
            }
        },
    }

    hdf_file = ReturnnDumpHDFJob(dataset_config).out_hdf

    return hdf_file


def build_rasr_feature_hdf(
    corpus: CorpusObject,
    split: int,
    feature_type: str,
    feature_extraction_args: Dict[str, Any],
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
):

    # Build CRP
    base_crp = rasr.CommonRasrParameters()
    rasr.crp_add_default_output(base_crp)
    base_crp.set_executables(rasr_binary_path, rasr_arch)

    rasr.crp_set_corpus(base_crp, corpus)
    base_crp.concurrent = split
    base_crp.segment_path = SegmentCorpusJob(corpus.corpus_file, split).out_segment_path

    feature_job = {
        "mfcc": features.MfccJob,
        "gt": features.GammatoneJob,
        "energy": features.EnergyJob,
    }[feature_type](
        crp=base_crp,
        port_name_mapping={"features": feature_type},
        **feature_extraction_args
    )

    dataset_config = {
        "class": "SprintCacheDataset",
        "data": {
            "data": {
                "filename": feature_job.out_feature_bundle[feature_type],
                "data_type": "feat",
            }
        },
    }

    hdf_file = ReturnnDumpHDFJob(dataset_config).out_hdf

    return hdf_file
