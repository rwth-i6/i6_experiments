__all__ = ["DecodeConfig", "DecodeRecogResult"]

from dataclasses import dataclass

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

from i6_core.returnn import ReturnnConfig, ReturnnForwardJobV2
from i6_core.serialization.base import Collection, CallImport
from i6_core.corpus import (
    ApplyLexiconToCorpusJob,
    FilterCorpusBySegmentsJob,
    CorpusToTxtJob,
    FilterCorpusRemoveUnknownWordSegmentsJob
)

from .dataset_config import (
    DatasetConfig,
    select_segments,
    get_dataset_config,
    CreateSequenceWhitelistJob
)
from .clustering_config import (
    RETURNN_PYTHON_EXE,
    RETURNN_ROOT,
    get_base_config
)
from .librasr_recognition import RecogConfig, create_rasr_config
from .score import JiwerScoringJob, ScoreResult
from .corpus_setup import setup_corpus
from ..lib.serialization import HashedCode

from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.decode import ClusteringDecodeCallback

@dataclass
class DecodeRecogResult:
    descriptor: str
    corpus_name: str
    per: tk.Variable
    deletion: tk.Variable
    insertion: tk.Variable
    substitution: tk.Variable

def get_callback_config(
    centroids: tk.Path,
    recognition_config: tk.Path,
    distance_scale: float = 1.0,
    subsampling: int | None = None,
    pooling_function: str = "maxpool_time_np",
    verbosity: int = 1,
    exclude_lemmata=["[SILENCE]"],
    rasr_path: str | None = None,
) -> ReturnnConfig:
    serializer_objs = []

    if rasr_path:
        path_insertion_code = DelayedFormat(
            'sys.path.insert(0, "{rp}")\n',
            rp=rasr_path,
        )
        serializer_objs.append(HashedCode(path_insertion_code))

    arguments = {
        "centroids_file": centroids,
        "recognition_config": recognition_config,
        "distance_scale": distance_scale,
        "subsampling": subsampling,
        "pooling_function": pooling_function,
        "verbosity": verbosity,
        "exclude_lemmata": exclude_lemmata,
    }
    clustering_callback = CallImport(
        code_object_path=ClusteringDecodeCallback,
        unhashed_package_root=None,
        hashed_arguments=arguments,
        unhashed_arguments={},
        import_as="forward_callback"
    )
    serializer_objs.append(clustering_callback)
    return ReturnnConfig(
        config={},
        python_epilog=Collection(serializer_objs),
    )


@dataclass
class DecodeConfig:
    centroids: tk.Path
    recog_config: RecogConfig
    distance_scale: float
    subsampling: int | None = None
    pooling_function: str = "maxpool_time_np"
    verbosity: int = 1

@dataclass
class DecodeResult:
    fwd_job: ReturnnForwardJobV2
    hyp: tk.Path

def _decode(
    config: DecodeConfig,
    dataset_config: ReturnnConfig,
    rasr_path: tk.Path | None = None,
    returnn_python_exe: tk.Path | None = None,
    returnn_root: tk.Path | None = None,
):
    if returnn_python_exe is None:
        returnn_python_exe = RETURNN_PYTHON_EXE
    if returnn_root is None:
        returnn_root = RETURNN_ROOT
    
    recog_rasr_config = create_rasr_config(config.recog_config)

    base_config = get_base_config()
    callback_config = get_callback_config(
        centroids=config.centroids,
        recognition_config=recog_rasr_config,
        distance_scale=config.distance_scale,
        subsampling=config.subsampling,
        pooling_function=config.pooling_function,
        verbosity=config.verbosity,
        exclude_lemmata=["[SILENCE]"],
        rasr_path=rasr_path,
    )

    returnn_config = ReturnnConfig({})
    for r_config in [base_config, dataset_config, callback_config]:
        returnn_config.update(r_config)

    returnn_config.black_formatting = False

    hyp_file = "hyp.txt"
    fwd_job = ReturnnForwardJobV2(
        model_checkpoint=None,
        returnn_config=returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=[hyp_file]
    )
    #fwd_job.rqmt["gpu_mem"] = 24

    out_hyp = fwd_job.out_files[hyp_file]

    return DecodeResult(
        fwd_job=fwd_job,
        hyp=out_hyp,
    )

def decode_and_score(
    exp_name: str,
    corpus_name: str,
    config: DecodeConfig,
    dataset_config: DatasetConfig,
    rasr_path: tk.Path | None = None,
    returnn_python_exe: tk.Path | None = None,
    returnn_root: tk.Path | None = None,
) -> DecodeRecogResult:
    # setup corpus
    setup_result = setup_corpus(key="dev-clean")

    filtered_corpus = FilterCorpusRemoveUnknownWordSegmentsJob(setup_result.corpus, setup_result.lexicon, all_unknown=False, delete_empty_recordings=True).out_corpus
    phoneme_corpus = ApplyLexiconToCorpusJob(filtered_corpus, setup_result.lexicon).out_corpus

    sampled_segments = select_segments(dataset_config.sampling_method, setup_result.segments)
    if sampled_segments is not None:
        phoneme_corpus = FilterCorpusBySegmentsJob(phoneme_corpus, sampled_segments).out_corpus

    ref_file = CorpusToTxtJob(phoneme_corpus).out_txt

    # dataset config
    whitelist_job = CreateSequenceWhitelistJob(filtered_corpus)
    whitelist_job.add_alias(f"datasets/LibriSpeech/dev-clean_whitelist")
    whitelist = whitelist_job.out_whitelist

    if sampled_segments is not None:
        dataset_rconfig = get_dataset_config(dataset_config.audio_hdf_path, sampled_segments)
    else:
        dataset_rconfig = get_dataset_config(dataset_config.audio_hdf_path, whitelist)

    decode_res = _decode(
        config,
        dataset_rconfig,
        rasr_path,
        returnn_python_exe,
        returnn_root,
    )

    score_job = JiwerScoringJob(ref_file, decode_res.hyp)

    score_res = ScoreResult.from_job(score_job)

    return DecodeRecogResult(
        exp_name,
        corpus_name,
        score_res.wer,
        score_res.deletions,
        score_res.substitutions,
        score_res.insertions,
    )
