import numpy as np
from typing import List

from sisyphus import *


from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo
from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import DecodingTensorMap
from i6_experiments.users.raissi.setups.common.helpers.priors.estimate_povey_like_prior_fh import (
    EstimateFactoredTriphonePriorsJob,
    CombineMeansForTriphoneForward,
    DumpXmlForTriphoneForwardJob,
)

from i6_experiments.users.raissi.setups.common.helpers.priors.util import PartitionDataSetup

Path = setup_path(__package__)
RANDOM_SEED = 42


def get_triphone_priors(
    name: str,
    graph_path: Path,
    model_path: Path,
    data_paths: List[Path],
    label_info: LabelInfo,
    tensor_map: DecodingTensorMap,
    partition_data_setup: PartitionDataSetup,
    tf_library=None,
    n_batch=10000,
    cpu: int = 2,
    gpu: int = 1,
    time: int = 1,
):

    triphone_files = []
    diphone_files = []
    context_files = []
    num_segments = []

    np.random.seed(RANDOM_SEED)
    for i in np.random.choice(range(len(data_paths)//partition_data_setup.data_offset), partition_data_setup.n_data_indices, replace=False):
        start_ind = i * partition_data_setup.data_offset
        end_ind = (i + 1) * partition_data_setup.data_offset
        for j in range(partition_data_setup.n_segment_indices):
            start_ind_seg = j * partition_data_setup.segment_offset
            end_ind_seg = (j + 1) * partition_data_setup.segment_offset
            # if end_ind_seg > 1248: end_ind_seg = 1248
            data_indices = list(range(start_ind, end_ind))
            estimateJob = EstimateFactoredTriphonePriorsJob(
                graph_path=graph_path,
                model_path=model_path,
                tensor_map=tensor_map,
                data_paths=data_paths,
                data_indices=data_indices,
                start_ind_segment=start_ind_seg,
                end_ind_segment=end_ind_seg,
                label_info=label_info,
                tf_library_path=tf_library,
                n_batch=n_batch,
                cpu=cpu,
                gpu=gpu,
                time=time,
            )
            if name is not None:
                estimateJob.add_alias(f"priors/{name}-{data_indices}_{start_ind_seg}")
            triphone_files.extend(estimateJob.triphone_files)
            diphone_files.extend(estimateJob.diphone_files)
            context_files.extend(estimateJob.context_files)
            num_segments.extend(estimateJob.num_segments)

    comb_jobs = []
    for spliter in range(0, len(triphone_files), partition_data_setup.split_step):
        start = spliter
        end = min(spliter + partition_data_setup.split_step, len(triphone_files))
        comb_jobs.append(
            CombineMeansForTriphoneForward(
                triphone_files=triphone_files[start:end],
                diphone_files=diphone_files[start:end],
                context_files=context_files[start:end],
                num_segment_files=num_segments[start:end],
                label_info=label_info,
            )
        )

    comb_triphone_files = [c.triphone_files_out for c in comb_jobs]
    comb_diphone_files = [c.diphone_files_out for c in comb_jobs]
    comb_context_files = [c.context_files_out for c in comb_jobs]
    comb_num_segs = [c.num_segments_out for c in comb_jobs]
    xmlJob = DumpXmlForTriphoneForwardJob(
        triphone_files=comb_triphone_files,
        diphone_files=comb_diphone_files,
        context_files=comb_context_files,
        num_segment_files=comb_num_segs,
        label_info=label_info
    )

    prior_files_triphone = [xmlJob.triphone_xml, xmlJob.diphone_xml, xmlJob.context_xml]
    xml_name = f"priors/{name}"
    tk.register_output(xml_name, prior_files_triphone[0])

    return prior_files_triphone


# needs refactoring
def get_diphone_priors(
    graph_path,
    model_path,
    data_paths,
    data_indices,
    nStateClasses=141,
    nContexts=47,
    gpu=1,
    time=20,
    isSilMapped=True,
    name=None,
    n_batch=10000,
    tf_library=None,
    tensor_map=None,
):

    if tf_library is None:
        tf_library = libraryPath
    if tensor_map is None:
        tensor_map = defaultTfMap

    estimateJob = EstimateSprintDiphoneAndContextPriors(
        graph_path,
        model_path,
        data_paths,
        data_indices,
        tf_library,
        nContexts=nContexts,
        nStateClasses=nStateClasses,
        gpu=gpu,
        time=time,
        tensorMap=tensor_map,
        n_batch=n_batch,
    )
    if name is not None:
        estimateJob.add_alias(f"priors/{name}")

    xmlJob = DumpXmlSprintForDiphone(
        estimateJob.diphone_files,
        estimateJob.context_files,
        estimateJob.num_segments,
        nContexts=nContexts,
        nStateClasses=nStateClasses,
        adjustSilence=isSilMapped,
    )

    priorFiles = [xmlJob.diphoneXml, xmlJob.contextXml]

    xmlName = f"priors/{name}"
    tk.register_output(xmlName, priorFiles[0])

    return priorFiles
