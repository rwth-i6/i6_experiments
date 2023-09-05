from typing import Any, Callable, Dict, List, Optional

from sisyphus import gs, tk

from i6_core.features import gammatone_flow, GammatoneJob
from i6_core import rasr
from i6_core import corpus
from i6_core.lexicon.allophones import DumpStateTyingJob
from i6_core.meta.system import CorpusObject
from i6_core.returnn import ReturnnDumpHDFJob

from i6_experiments.common.setups.rasr import (
    GmmSystem,
    ReturnnRasrDataInput,
)
from i6_experiments.common.datasets import librispeech
from i6_experiments.users.vieting.jobs.converse import (
    EnhancedMeetingDataToBlissCorpusJob,
    EnhancedSegmentedEvalDataToBlissCorpusJob,
    EnhancedMeetingDataRasrAlignmentPadAndDumpHDFJob,
)
from .default_tools import RETURNN_EXE, RETURNN_ROOT, RASR_BINARY_PATH_APPTAINER

feature_extraction_args = {
    "gt_options": {
        "minfreq": 100,
        "maxfreq": 7500,
        "channels": 50,
        "tempint_type": "hanning",
        "tempint_shift": 0.01,
        "tempint_length": 0.025,
        "flush_before_gap": True,
        "do_specint": False,
        "specint_type": "hanning",
        "specint_shift": 4,
        "specint_length": 9,
        "normalize": True,
        "preemphasis": True,
        "legacy_scaling": False,
        "without_samples": False,
        "samples_options": {
            "audio_format": "wav",
            "dc_detection": False,
        },
        "normalization_options": {},
        "add_features_output": True,
    }
}


class HdfDataInput:
    """
    Based on copy from users/luescher. Could be added to common.
    """

    def __init__(
        self,
        features: List[tk.Path],
        alignments: List[tk.Path],
        *,
        meta_args: Optional[Dict[str, Any]] = None,
        align_args: Optional[Dict[str, Any]] = None,
        feat_args: Optional[Dict[str, Any]] = None,
        data_keys: Optional[Dict[str, Any]] = None,
        acoustic_mixtures: Optional[tk.Path] = None,
    ):
        """
        :param features: hdf files which contain raw wve form or features, like GT or MFCC
        :param alignments: hdf files which contain dumped RASR alignments
        :param partition_epoch: if >1, split the full dataset into multiple sub-epochs
        :param seq_ordering: sort the sequences in the dataset, e.g. "random" or "laplace:.100"
        :param meta_args: parameters for the `MetaDataset`
        :param align_args: parameters for the `HDFDataset` for the alignments
        :param feat_args: parameters for the `HDFDataset` for the features
        :param acoustic_mixtures: path to a RASR acoustic mixture file (used in System classes, not RETURNN training)
        """
        self.features = features
        self.alignments = alignments
        self.meta_args = meta_args
        self.align_args = align_args
        self.feat_args = feat_args
        self.data_keys = data_keys or {"feat": "data", "align": "classes"}
        self.acoustic_mixtures = acoustic_mixtures

    def get_data_dict(self):
        return {
            "class": "MetaDataset",
            "data_map": {
                "classes": ("align", self.data_keys["align"]),
                "data": ("feat", self.data_keys["feat"]),
            },
            "datasets": {
                "align": {
                    "class": "HDFDataset",
                    "files": self.alignments,
                    "use_cache_manager": True,
                    **(self.align_args or {}),
                },
                "feat": {
                    "class": "HDFDataset",
                    "files": self.features,
                    "use_cache_manager": True,
                    **(self.feat_args or {}),
                },
            },
            **(self.meta_args or {}),
        }


def _map_audio_paths(audio_path: str) -> str:
    return audio_path.replace(
        "/scratch/hpc-prf-nt2/tvn/data/rwth_train/enhanced_tfgridnet/",
        "/work/asr4/vieting/setups/converse/data/thilo_20230721_enhanced_tfgridnet/enhanced_tfgridnet/",
    )


def get_hdf_datasets(
    name: str,
    database: tk.Path,
    gmm_system: GmmSystem,
    train_corpus_duration: float,
    map_audio_paths: Optional[Callable] = None,
    dev_split: float = 0.01,
    concurrent: int = 20,
) -> Dict[str, HdfDataInput]:
    """
    Prepare data (features and alignments) based on separated data in json
    """
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/libricss/finetune/"
    job = EnhancedMeetingDataToBlissCorpusJob(
        json_database=database,
        audio_path_mapping=map_audio_paths or _map_audio_paths,
        hash_audio_path_mapping=map_audio_paths is None,
        dataset_name="train_960",
    )
    tk.register_output(f"data/{name}.xml.gz", job.out_bliss_corpus)

    train_corpus = CorpusObject()
    train_corpus.corpus_file = job.out_bliss_corpus
    train_corpus.duration = train_corpus_duration
    train_corpus.audio_format = "wav"

    base_crp = rasr.CommonRasrParameters()
    rasr.crp_add_default_output(base_crp)
    base_crp.set_executables(RASR_BINARY_PATH_APPTAINER, "linux-x86_64-standard")

    rasr.crp_set_corpus(base_crp, train_corpus)
    base_crp.concurrent = concurrent
    base_crp.segment_path = corpus.SegmentCorpusJob(train_corpus.corpus_file, base_crp.concurrent).out_segment_path

    feature_job = GammatoneJob(crp=base_crp, **feature_extraction_args)

    dataset_config = {
        "class": "SprintCacheDataset",
        "data": {
            "data": {
                "filename": feature_job.out_feature_bundle["gt"],
                "data_type": "feat",
            }
        },
    }

    feat_hdf = ReturnnDumpHDFJob(dataset_config, returnn_python_exe=RETURNN_EXE, returnn_root=RETURNN_ROOT).out_hdf
    tk.register_output(f"data/{name}_gammatone_v0.hdf", feat_hdf)

    alignments = gmm_system.outputs["train-other-960"]["final"].alignments.alternatives["bundle"]
    allophone_file = gmm_system.outputs["train-other-960"][
        "final"
    ].crp.acoustic_model_post_config.allophones.add_from_file
    state_tying_job = DumpStateTyingJob(gmm_system.outputs["train-other-960"]["final"].crp)
    align_hdf_job = EnhancedMeetingDataRasrAlignmentPadAndDumpHDFJob(
        json_database=database,
        dataset_name="train_960",
        feature_hdfs=[feat_hdf],
        alignment_cache=alignments,
        allophone_file=allophone_file,
        state_tying_file=state_tying_job.out_state_tying,
        returnn_root=RETURNN_ROOT,
    )
    align_hdf_job.rqmt.update({"mem": 10, "time": 24})
    align_hdf = align_hdf_job.out_hdf_file
    tk.register_output(f"data/{name}_align.hdf", align_hdf)

    all_segments = corpus.SegmentCorpusJob(train_corpus.corpus_file, 1).out_single_segment_files[1]
    shuffled_segments = corpus.ShuffleAndSplitSegmentsJob(
        all_segments, {"train": 1 - dev_split, "dev": dev_split}
    ).out_segments
    datasets = {
        key: HdfDataInput(
            features=[feat_hdf],
            alignments=[align_hdf],
            align_args={
                "seq_list_filter_file": shuffled_segments[key],
                "partition_epoch": 1,
                "seq_ordering": "laplace:.1000",
            },
            meta_args={"seq_order_control_dataset": "align"},
            data_keys={"feat": "data", "align": "data"},
            acoustic_mixtures=gmm_system.outputs["train-other-960"]["final"].acoustic_mixtures,
        )
        for key in shuffled_segments
    }
    return datasets


def _map_audio_paths_libricss_tfgridnet_json_database_v0(audio_path: str) -> str:
    return audio_path.replace(
        "/scratch/hpc-prf-nt2/tvn/experiments/rwth/19/evaluation/ckpt_120000_1/",
        "/work/asr4/vieting/setups/converse/data/20230727_libri_css_tvn_rwth_19_120000_1/",
    )


def get_eval_data_input(gmm_system, concurrent: int = 40) -> Dict[str, ReturnnRasrDataInput]:
    """
    Prepare data for recognition.
    """
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/libricss/finetune/"
    database_tfgridnet = tk.Path(
        "/work/asr4/vieting/setups/converse/data/20230727_libri_css_tvn_rwth_19_120000_1.json",
        hash_overwrite="libricss_tfgridnet_json_database_v0",
    )
    sub_corpora = [
        EnhancedSegmentedEvalDataToBlissCorpusJob(
            database_tfgridnet,
            _map_audio_paths_libricss_tfgridnet_json_database_v0,
            dataset_name=dataset_name,
        ).out_bliss_corpus
        for dataset_name in [
            "0S_segments",
            "0L_segments",
            "OV10_segments",
            "OV20_segments",
            "OV30_segments",
            "OV40_segments",
        ]
    ]
    libri_css_bliss = corpus.MergeCorporaJob(sub_corpora, "libricss").out_merged_corpus
    tk.register_output("data/libricss_tfgridnet_v0.xml.gz", libri_css_bliss)

    libri_css_corpus = CorpusObject()
    libri_css_corpus.corpus_file = libri_css_bliss
    libri_css_corpus.duration = 11.0
    libri_css_corpus.audio_format = "wav"

    segment_job = corpus.SegmentCorpusJob(libri_css_corpus.corpus_file, concurrent)
    lexicon = {
        "filename": librispeech.get_bliss_lexicon(),
        "normalize_pronunciation": False,
    }
    lm = {
        "filename": librispeech.get_arpa_lm_dict()["4gram"],
        "type": "ARPA",
        "scale": 10,
    }

    data = ReturnnRasrDataInput(
        name="libricss",
        feature_flow={"gt": gammatone_flow(**feature_extraction_args["gt_options"])},  # TODO: add cached flow
    )
    data.build_crp(
        am_args={},
        corpus_object=libri_css_corpus,
        concurrent=concurrent,
        segment_path=segment_job.out_segment_path,
        lexicon_args=lexicon,
        cart_tree_path=gmm_system.outputs["train-other-960"]["final"].crp.acoustic_model_config.state_tying.file,
        allophone_file=None,
        lm_args=lm,
    )
    return {"libricss": data}
