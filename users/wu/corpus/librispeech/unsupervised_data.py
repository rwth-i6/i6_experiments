import copy
from typing import Dict, List, Tuple, Optional
from i6_core.meta.system import CorpusObject
import i6_experiments.common.datasets.librispeech as lbs_dataset
from i6_experiments.common.setups.rasr import util as rasr_util
from i6_experiments.users.berger.corpus.general.helpers import filter_unk_in_corpus_object
from .lm_data import get_lm
from i6_experiments.users.berger import helpers
from sisyphus import tk

from i6_core.returnn.hdf import BlissToPcmHDFJob
from i6_core.corpus import SegmentCorpusJob
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder, hdf_config_dict_for_files

def get_librispeech_unsupervised(
    returnn_root: tk.Path,
    train_key: str = "train-other-960",
    dev_keys: List[str] = ["dev-clean", "dev-other"],
    train_partition_epoch: int = 20,
    audio_format: str = "wav",
):
    corpus_object_dict = copy.deepcopy(
        lbs_dataset.get_corpus_object_dict(
            audio_format=audio_format,
            output_prefix="corpora",
        )
    ) 
    train_corpus_object = corpus_object_dict[train_key]
    
    train_feature_hdfs = []
    train_segment_files = list(
        SegmentCorpusJob(
            train_corpus_object.corpus_file, 100  # 10 hours per hdf
        ).out_single_segment_files.values()
    )
    for segment_file in train_segment_files:
        feature_hdf_job = BlissToPcmHDFJob(
            train_corpus_object.corpus_file,
            segment_file=segment_file,
            rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
            returnn_root=returnn_root,
        )
        feature_hdf_job.rqmt = {"cpu": 2, "mem": 8, "time": 8}
        train_feature_hdfs.append(feature_hdf_job.out_hdf)

    train_data_config = hdf_config_dict_for_files(
        files=train_feature_hdfs,
        extra_config={
            "partition_epoch": train_partition_epoch,
            "seq_ordering": "laplace:.1000",
        }
    )

    cv_feature_hdfs = []
    for key in dev_keys:
        bliss_to_pcm_hdf_job = BlissToPcmHDFJob(
            corpus_object_dict[key].corpus_file,
            rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
            returnn_root=returnn_root,
        )
        cv_feature_hdfs.append(bliss_to_pcm_hdf_job.out_hdf)
    cv_data_config = hdf_config_dict_for_files(
        files=cv_feature_hdfs,
        extra_config={
            "partition_epoch": 1,
            "seq_ordering": "sorted",
        }
    ) 
    
    return train_data_config, cv_data_config
    
