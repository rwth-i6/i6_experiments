__all__ = ["DecodeConfig", "decode"]

from dataclasses import dataclass

from sisyphus import tk

from i6_core.returnn import ReturnnConfig, ReturnnForwardJobV2
from i6_core.serialization.base import Collection, CallImport
from i6_core.corpus import (
    ApplyLexiconToCorpusJob,
    FilterCorpusBySegmentsJob,
    CorpusToTxtJob,
)

from .dataset_config import DatasetConfig, create_config as create_dataset_config, select_segments, get_dataset_config
from .clustering_config import (
    RETURNN_PYTHON_EXE,
    RETURNN_ROOT,
    get_base_config
)
from .librasr_recognition import RecogConfig, create_rasr_config
from .score import JiwerScoringJob, ScoreResult
from config.corpus_setup import py as setup_corpus

from i6_experiments.users.mann.external.unsupervised.lib.pytorch.decode import ClusteringDecodeCallback

def get_callback_config(
    centroids: tk.Path,
    recognition_config: tk.Path,
    distance_scale: float = 1.0,
    subsampling: int | None = None,
    pooling_function: str = "maxpool_time_np",
    verbosity: int = 1,
    exclude_lemmata=["[SILENCE]"]
) -> ReturnnConfig:
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
    return ReturnnConfig(
        config={},
        python_epilog=Collection([clustering_callback]),
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

def decode(
    config: DecodeConfig,
    dataset_config: DatasetConfig,
    returnn_python_exe: tk.Path | None = None,
    returnn_root: tk.Path | None = None,
):
    if returnn_python_exe is None:
        returnn_python_exe = RETURNN_PYTHON_EXE
    if returnn_root is None:
        returnn_root = RETURNN_ROOT
    
    recog_rasr_config = create_rasr_config(config.recog_config)

    base_config = get_base_config()
    dataset_config, sampled_segments = create_dataset_config(dataset_config)
    callback_config = get_callback_config(
        centroids=config.centroids,
        recognition_config=recog_rasr_config,
        distance_scale=config.distance_scale,
        subsampling=config.subsampling,
        pooling_function=config.pooling_function,
        verbosity=config.verbosity,
        exclude_lemmata=["[SILENCE]"]
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
    out_hyp = fwd_job.out_files[hyp_file]

    return DecodeResult(
        fwd_job=fwd_job,
        hyp=out_hyp,
    )


def _decode(
    config: DecodeConfig,
    dataset_config: ReturnnConfig,
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
        exclude_lemmata=["[SILENCE]"]
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
    out_hyp = fwd_job.out_files[hyp_file]

    return DecodeResult(
        fwd_job=fwd_job,
        hyp=out_hyp,
    )

@dataclass
class DecodeScoreResult:
    decode: DecodeResult
    score: ScoreResult


def decode_and_score(
    config: DecodeConfig,
    dataset_config: DatasetConfig,
    returnn_python_exe: tk.Path | None = None,
    returnn_root: tk.Path | None = None,
):
    # setup corpus
    setup_result = setup_corpus()
    sampled_segments = select_segments(dataset_config.sampling_method, setup_result.segments)
    phoneme_corpus = ApplyLexiconToCorpusJob(setup_result.corpus, setup_result.lexicon).out_corpus
    if sampled_segments is not None:
        phoneme_corpus = FilterCorpusBySegmentsJob(phoneme_corpus, sampled_segments).out_corpus
    ref_file = CorpusToTxtJob(phoneme_corpus).out_txt

    # dataset config
    dataset_rconfig = get_dataset_config(dataset_config.audio_hdf_path, sampled_segments)

    decode_res = _decode(
        config,
        dataset_rconfig,
        returnn_python_exe,
        returnn_root,
    )

    score_job = JiwerScoringJob(ref_file, decode_res.hyp)

    score_res = ScoreResult.from_job(score_job)

    return DecodeScoreResult(
        decode_res, score_res
    )
