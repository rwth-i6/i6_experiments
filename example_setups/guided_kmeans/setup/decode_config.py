__all__ = ["DecodeConfig", "DecodeRecogResult"]

from dataclasses import dataclass

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

from i6_core.returnn import ReturnnConfig, ReturnnForwardJobV2, CodeWrapper
from i6_core.serialization.base import Collection, CallImport
from i6_core.corpus import (
    ApplyLexiconToCorpusJob,
    FilterCorpusBySegmentsJob,
    FilterCorpusRemoveUnknownWordSegmentsJob
)

from .dataset_config import (
    DatasetConfig,
    select_segments,
    get_dataset_config,
    CreateSequenceWhitelistJob
)
from .clustering_config import get_base_config
from .librasr_recognition import RecogConfig, create_rasr_config
from .score import JiwerScoringJob, TaggedCorpusToTxtJob, ScoreResult
from .corpus_setup import setup_corpus
from ..lib.serialization import HashedCode
from .. import tools

from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.decode import ClusteringDecodeCallback
from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.model import load_gaussian_model
from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.chunked.models import load_forward_model

@dataclass
class DecodeRecogResult:
    descriptor: str
    corpus_name: str
    per: tk.Variable
    deletion: tk.Variable
    insertion: tk.Variable
    substitution: tk.Variable
    mean_cos_sim: tk.Variable | None = None
    l1_dist: tk.Variable | None = None
    avg_total_score: tk.Variable | None = None
    avg_am_score: tk.Variable | None = None
    avg_transition_score: tk.Variable | None = None
    avg_lm_score: tk.Variable | None = None
    avg_segment_duration: tk.Variable | None = None
    frame_labels: tk.Path | None = None
    confusion_pairs: tk.Path | None = None
    fer: tk.Variable | None = None
    frame_confusion_pairs: tk.Path | None = None

def build_gaussian_model_object(centroids: tk.Path, cov: tk.Path) -> CallImport:
    args = {
        "centroids_path": centroids,
        "covs_path": cov
    }
    load_call = CallImport(
        code_object_path=load_gaussian_model,
        unhashed_package_root=None,
        hashed_arguments=args,
        unhashed_arguments={},
        import_as="gaussian_model"
    )
    return load_call

def build_artifact_model_object(model_dir: tk.Path) -> CallImport:
    """
    Score with whatever model class wrote ``model_dir``, per its manifest.

    The alternative - a ``mixtures=`` argument here beside ``covs=`` - would
    need repeating for every parameter set a model might carry, and the decode
    side has no business knowing them. A directory plus its manifest is what
    the epoch job already produces.
    """
    return CallImport(
        code_object_path=load_forward_model,
        unhashed_package_root=None,
        hashed_arguments={"model_dir": model_dir},
        unhashed_arguments={},
        import_as="gaussian_model",
    )


def get_callback_config(
    centroids: tk.Path,
    recognition_config: tk.Path,
    distance_scale: float = 1.0,
    subsampling: int | None = None,
    pooling_function: str = "maxpool_time_np",
    verbosity: int = 1,
    exclude_lemmata=["[SILENCE]"],
    rasr_path: str | None = None,
    cov_path: tk.Path | None = None,
    num_workers: int = 7,
    write_frame_labels: bool = False,
    legacy_hash_num_workers: bool = False,
    model_dir: tk.Path | None = None,
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
        # Only hashed once it is actually switched on: hyp.txt is unaffected by it, so
        # decodes that don't write frame labels keep the hash they had before the option
        # existed. Same pattern as the scale schedules in clustering_config.py.
        **({"write_frame_labels": True} if write_frame_labels else {}),
    }
    unhashed_args = {}
    if legacy_hash_num_workers:
        arguments["num_workers"] = num_workers
    else:
        unhashed_args["num_workers"] = num_workers

    # model_dir first: it is the general form, and it is only ever set by a
    # caller that opted into it, so nothing that predates it changes hash.
    if model_dir is not None:
        if cov_path is not None:
            raise TypeError(
                "pass either model_dir or centroids/cov_path, not both - the model "
                "directory already carries every artifact its class needs"
            )
        serializer_objs.append(build_artifact_model_object(model_dir))
        arguments.update(
            gaussian_model=CodeWrapper("gaussian_model"),
            centroids_file=None
        )
    elif cov_path is not None:
        serializer_objs.append(build_gaussian_model_object(centroids, cov_path))
        arguments.update(
            gaussian_model=CodeWrapper("gaussian_model"),
            centroids_file=None
        )

    clustering_callback = CallImport(
        code_object_path=ClusteringDecodeCallback,
        unhashed_package_root=None,
        hashed_arguments=arguments,
        unhashed_arguments=unhashed_args,
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
    recog_rasr_config: tk.Path
    # Set instead of centroids/covs to score with a whole model directory, which
    # is the only way to decode a model whose parameters are not (centroids,
    # covs) - a mixture model, say. `centroids` stays required because callers
    # and reports still key on it; it is ignored for scoring when this is set.
    distance_scale: float
    subsampling: int | None = None
    pooling_function: str = "maxpool_time_np"
    covs: tk.Path | None = None
    model_dir: tk.Path | None = None
    verbosity: int = 1
    num_workers: int = 7
    write_frame_labels: bool = False
    # Set this for pre-existing experiments to keep num_workers part of the job hash,
    # reproducing the hash they had before num_workers became unhashed. Do not set it
    # in new configs.
    legacy_hash_num_workers: bool = False

@dataclass
class DecodeResult:
    fwd_job: ReturnnForwardJobV2
    hyp: tk.Path
    frame_labels: tk.Path | None = None

def _decode(
    config: DecodeConfig,
    dataset_config: ReturnnConfig,
    rasr_path: tk.Path | None = None,
    returnn_python_exe: tk.Path | None = None,
    returnn_root: tk.Path | None = None,
    precomputed: bool = False,
    device: str = "gpu",
):
    if returnn_python_exe is None:
        returnn_python_exe = tools.RETURNN_PYTHON_EXE
    if returnn_root is None:
        returnn_root = tools.RETURNN_ROOT

    base_config = get_base_config(precomputed)
    callback_config = get_callback_config(
        centroids=config.centroids,
        recognition_config=config.recog_rasr_config,
        distance_scale=config.distance_scale,
        subsampling=config.subsampling,
        pooling_function=config.pooling_function,
        verbosity=config.verbosity,
        exclude_lemmata=["[SILENCE]"],
        rasr_path=rasr_path,
        cov_path=config.covs,
        model_dir=config.model_dir,
        num_workers=config.num_workers,
        write_frame_labels=config.write_frame_labels,
        legacy_hash_num_workers=config.legacy_hash_num_workers,
    )

    returnn_config = ReturnnConfig({})
    for r_config in [base_config, dataset_config, callback_config]:
        returnn_config.update(r_config)

    returnn_config.black_formatting = False

    output_files = ["hyp.txt"]
    if config.write_frame_labels:
        output_files.append("frame_labels.jsonl")

    num_cpus = config.num_workers + 1

    fwd_job = ReturnnForwardJobV2(
        model_checkpoint=None,
        returnn_config=returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=output_files,
        device=device,
        cpu_rqmt=num_cpus,
    )
    if device == "gpu":
        fwd_job.rqmt["gpu_mem"] = 24
    fwd_job.rqmt["mem"] = num_cpus * 4

    out_hyp = fwd_job.out_files["hyp.txt"]
    out_frame_labels = fwd_job.out_files.get("frame_labels.jsonl")

    return DecodeResult(
        fwd_job=fwd_job,
        hyp=out_hyp,
        frame_labels=out_frame_labels,
    )

def decode_and_score(
    exp_name: str,
    corpus_name: str,
    config: DecodeConfig,
    dataset_config: DatasetConfig,
    rasr_path: tk.Path | None = None,
    returnn_python_exe: tk.Path | None = None,
    returnn_root: tk.Path | None = None,
    device: str = "gpu",
    corpus_key: str | None = None,
) -> DecodeRecogResult:
    # setup corpus
    effective_key = corpus_key if corpus_key is not None else corpus_name
    setup_result = setup_corpus(key=effective_key)

    filtered_corpus = FilterCorpusRemoveUnknownWordSegmentsJob(setup_result.corpus, setup_result.lexicon, all_unknown=False, delete_empty_recordings=True).out_corpus
    phoneme_corpus = ApplyLexiconToCorpusJob(filtered_corpus, setup_result.lexicon).out_corpus

    sampled_segments = select_segments(dataset_config.sampling_method, setup_result.segments)
    if sampled_segments is not None:
        phoneme_corpus = FilterCorpusBySegmentsJob(phoneme_corpus, sampled_segments).out_corpus

    ref_file = TaggedCorpusToTxtJob(phoneme_corpus).out_txt

    # dataset config
    whitelist_job = CreateSequenceWhitelistJob(filtered_corpus)
    whitelist_job.add_alias(f"datasets/LibriSpeech/{effective_key}_whitelist")
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
        precomputed=dataset_config.precomputed,
        device=device,
    )

    score_job = JiwerScoringJob(ref_file, decode_res.hyp)

    score_res = ScoreResult.from_job(score_job)

    return DecodeRecogResult(
        exp_name,
        corpus_name,
        score_res.wer,
        score_res.deletions,
        score_res.insertions,
        score_res.substitutions,
        frame_labels=decode_res.frame_labels,
        confusion_pairs=score_job.out_confusion_pairs,
    )
