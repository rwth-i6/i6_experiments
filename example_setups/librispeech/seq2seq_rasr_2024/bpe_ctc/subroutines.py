from typing import List, Type
import math

import i6_experiments.common.datasets.librispeech as lbs_dataset
from i6_core.corpus.convert import CorpusToStmJob
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob, build_config_from_mapping
from i6_core.rasr.crp import CommonRasrParameters, crp_add_default_output
from i6_core.recognition.scoring import ScliteJob
from i6_core.returnn.config import CodeWrapper, ReturnnConfig, WriteReturnnConfigJob
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.search import SearchBPEtoWordsJob, SearchWordsToCTMJob
from i6_core.returnn.training import ReturnnTrainingJob
from i6_experiments.common.setups.returnn_pytorch.serialization import (
    PyTorchModel,
    build_config_constructor_serializers_v2,
)
from i6_experiments.common.setups.serialization import Collection, ExternalImport, Import, PartialImport
from i6_models.config import ModelConfiguration, ModuleType
from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase

from ..tools import rasr_binary_path, returnn_python_exe, returnn_root, sctk_binary_path
from .configs import (
    FlashlightRecogRoutineConfig,
    RasrBeamRecogRoutineConfig,
    RasrGreedyRecogRoutineConfig,
    PriorRoutineConfig,
    RecogRoutineConfig,
    ResultItem,
    TrainRoutineConfig,
)
from .pytorch_modules import ConformerCTCConfig, ConformerCTCModel, ConformerCTCRecogConfig, ConformerCTCRecogModel
from .returnn_steps import (
    ComputePriorCallback,
    ExtractCTCSearchRTFJob,
    SearchCallback,
    flashlight_recog_step,
    prior_step,
    rasr_recog_step,
    train_step,
)

recipe_imports = [
    "import sys",
    ExternalImport(
        import_path=tk.Path(
            f"{__file__.split('recipe')[0]}/recipe/",
            hash_overwrite="RECIPE_ROOT",
        )
    ),
]


def get_model_serializers(model_class: Type[ModuleType], model_config: ModelConfiguration) -> List[DelayedBase]:
    constructor_call, model_imports = build_config_constructor_serializers_v2(
        cfg=model_config,
        variable_name="cfg",
        unhashed_package_root=__package__,
    )

    model_serializers: List[DelayedBase] = [
        Import(
            f"{model_class.__module__}.{model_class.__name__}",
            unhashed_package_root=__package__,
        ),
    ]
    model_serializers.append(Collection(model_imports))  # type: ignore
    model_serializers.append(constructor_call)
    model_serializers.append(
        PyTorchModel(
            model_class_name=model_class.__name__,
            model_kwargs={"cfg": CodeWrapper("cfg")},
        )
    )

    return model_serializers


def train(config: TrainRoutineConfig, model_config: ConformerCTCConfig) -> ReturnnTrainingJob:
    model_serializers = get_model_serializers(model_class=ConformerCTCModel, model_config=model_config)
    num_epochs = config.save_epochs[-1]

    train_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
                "classes": {"dim": model_config.target_size - 1, "sparse": True, "dtype": "int32"},
            },
            "backend": "torch",
            "batch_size": config.batch_frames * 160,
            "cleanup_old_models": {
                "keep_last_n": 1,
                "keep_best_n": 0,
                "keep": config.save_epochs,
            },
            "gradient_clip": config.gradient_clip,
            "torch_amp": {"dtype": "bfloat16"},
            "stop_on_nonfinite_train_score": True,
            "optimizer": {
                "class": "adamw",
                "epsilon": 1e-16,
                "weight_decay": config.weight_decay,
            },
        },
        python_prolog=recipe_imports,
        python_epilog=[
            *model_serializers,
            Import(
                f"{train_step.__module__}.{train_step.__name__}",
                unhashed_package_root=__package__,
            ),
        ],  # type: ignore
        sort_config=False,
    )

    train_returnn_config.update(config.lr_config.get_returnn_config())
    train_returnn_config.update(config.train_data_config.get_returnn_data())
    train_returnn_config.update(config.cv_data_config.get_returnn_data())

    tk.register_output(
        "train/returnn.config",
        WriteReturnnConfigJob(train_returnn_config).out_returnn_config_file,
    )

    train_job = ReturnnTrainingJob(
        returnn_config=train_returnn_config,
        log_verbosity=5,
        num_epochs=num_epochs,
        time_rqmt=168,
        mem_rqmt=24,
        cpu_rqmt=6,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    )
    train_job.add_alias("training")
    train_job.rqmt["gpu_mem"] = 24

    tk.register_output("train/learning_rates", train_job.out_learning_rates)

    return train_job


def compute_priors(
    config: PriorRoutineConfig, model_config: ConformerCTCConfig, train_job: ReturnnTrainingJob, epoch: int
) -> tk.Path:
    model_serializers = get_model_serializers(model_class=ConformerCTCModel, model_config=model_config)

    prior_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
            },
            "backend": "torch",
            "batch_size": config.batch_frames * 160,
        },
        python_prolog=recipe_imports,
        python_epilog=[
            *model_serializers,
            Import(
                f"{prior_step.__module__}.{prior_step.__name__}",
                import_as="forward_step",
                unhashed_package_root=__package__,
            ),
            Import(
                f"{ComputePriorCallback.__module__}.{ComputePriorCallback.__name__}",
                import_as="forward_callback",
                unhashed_package_root=__package__,
            ),
        ],  # type: ignore
        sort_config=False,
    )

    prior_returnn_config.update(config.prior_data_config.get_returnn_data())

    tk.register_output(
        "prior/returnn.config",
        WriteReturnnConfigJob(prior_returnn_config).out_returnn_config_file,
    )

    prior_job = ReturnnForwardJobV2(
        model_checkpoint=train_job.out_checkpoints[epoch],
        returnn_config=prior_returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=["prior.txt"],
    )
    prior_job.add_alias(f"prior/prior.e-{epoch}")
    tk.register_output(f"prior/prior.e-{epoch}.txt", prior_job.out_files["prior.txt"])

    return prior_job.out_files["prior.txt"]


def rasr_greedy_recog(
    config: RasrGreedyRecogRoutineConfig, model_config: ConformerCTCConfig, train_job: ReturnnTrainingJob
) -> ResultItem:
    prior_file = compute_priors(
        train_job=train_job, epoch=config.epoch, config=config.prior_config, model_config=model_config
    )

    recog_model_config = ConformerCTCRecogConfig(
        logmel_cfg=model_config.logmel_cfg,
        conformer_cfg=model_config.conformer_cfg,
        dim=model_config.dim,
        target_size=model_config.target_size,
        dropout=model_config.dropout,
        specaug_cfg=model_config.specaug_cfg,
        prior_file=prior_file,
        prior_scale=config.prior_scale,
        blank_penalty=config.blank_penalty,
    )

    model_serializers = get_model_serializers(model_class=ConformerCTCRecogModel, model_config=recog_model_config)

    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.set_executables(rasr_binary_path=rasr_binary_path)

    recog_rasr_config, recog_rasr_post_config = build_config_from_mapping(crp=crp, mapping={}, include_log_config=True)

    recog_rasr_config.lib_rasr = RasrConfig()

    recog_rasr_config.lib_rasr.lexicon = RasrConfig()
    recog_rasr_config.lib_rasr.lexicon.type = "vocab-text"
    recog_rasr_config.lib_rasr.lexicon.file = config.vocab_file

    recog_rasr_config.lib_rasr.search_algorithm = RasrConfig()
    recog_rasr_config.lib_rasr.search_algorithm.type = "unconstrained-greedy-search"
    recog_rasr_config.lib_rasr.search_algorithm.use_blank = True
    recog_rasr_config.lib_rasr.search_algorithm.blank_label_index = model_config.target_size - 1
    recog_rasr_config.lib_rasr.search_algorithm.allow_label_loop = True

    recog_rasr_config.lib_rasr.label_scorer = RasrConfig()
    recog_rasr_config.lib_rasr.label_scorer.type = "no-op"

    recog_rasr_config_path = WriteRasrConfigJob(config=recog_rasr_config, post_config=recog_rasr_post_config).out_config
    tk.register_output(f"recognition/{config.corpus_name}/{config.descriptor}/rasr.config", recog_rasr_config_path)

    recog_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
            },
            "model_outputs": {
                "tokens": {
                    "dtype": "string",
                    "feature_dim_axis": None,
                },
                "audio_samples_size": {
                    "dtype": "int32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "am_time": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "search_time": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
            },
            "backend": "torch",
            "batch_size": 36_000 * 160,
        },
        python_prolog=recipe_imports
        + [
            # ExternalImport(rasr_binary_path),
            ExternalImport(tk.Path("/work/asr4/berger/rasr_dev/label_scorer/rasr/lib/linux-x86_64-standard")),
        ],
        python_epilog=[
            *model_serializers,
            Import(
                f"{SearchCallback.__module__}.{SearchCallback.__name__}",
                import_as="forward_callback",
                unhashed_package_root=__package__,
            ),
            PartialImport(
                code_object_path=f"{rasr_recog_step.__module__}.{rasr_recog_step.__name__}",
                unhashed_package_root=__package__ or "",
                hashed_arguments={"config_file": recog_rasr_config_path},
                unhashed_arguments={},
                import_as="forward_step",
            ),
        ],  # type: ignore
        sort_config=False,
    )

    recog_returnn_config.update(config.recog_data_config.get_returnn_data())

    tk.register_output(
        f"recognition/{config.corpus_name}/{config.descriptor}/returnn.config",
        WriteReturnnConfigJob(recog_returnn_config).out_returnn_config_file,
    )

    recog_job = ReturnnForwardJobV2(
        model_checkpoint=train_job.out_checkpoints[config.epoch],
        returnn_config=recog_returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=["search_out.py", "rtf.py"],
        device=config.device,
        mem_rqmt=16,
    )
    recog_job.add_alias(f"recognition/{config.corpus_name}/{config.descriptor}")

    tk.register_output(
        f"recognition/{config.corpus_name}/{config.descriptor}/search_output", recog_job.out_files["search_out.py"]
    )

    extract_rtf_job = ExtractCTCSearchRTFJob(rtf_file=recog_job.out_files["rtf.py"])
    tk.register_output(f"recognition/{config.corpus_name}/{config.descriptor}/rtf", extract_rtf_job.out_total_rtf)

    word_file = SearchBPEtoWordsJob(recog_job.out_files["search_out.py"]).out_word_search_results

    recog_corpus_file: tk.Path = lbs_dataset.get_corpus_object_dict(audio_format="wav", output_prefix="corpora")[
        config.corpus_name
    ].corpus_file
    recog_stm = CorpusToStmJob(recog_corpus_file).out_stm_path

    ctm_file = SearchWordsToCTMJob(word_file, bliss_corpus=recog_corpus_file).out_ctm_file
    score_job = ScliteJob(ref=recog_stm, hyp=ctm_file, sort_files=True, sctk_binary_path=sctk_binary_path)
    tk.register_output(
        f"recognition/{config.corpus_name}/{config.descriptor}/scoring_reports", score_job.out_report_dir
    )

    return ResultItem(
        descriptor=config.descriptor,
        corpus_name=config.corpus_name,
        wer=score_job.out_wer,
        am_rtf=extract_rtf_job.out_am_rtf,
        search_rtf=extract_rtf_job.out_search_rtf,
        total_rtf=extract_rtf_job.out_total_rtf,
    )


def rasr_beam_recog(
    config: RasrBeamRecogRoutineConfig, model_config: ConformerCTCConfig, train_job: ReturnnTrainingJob
) -> ResultItem:
    prior_file = compute_priors(
        train_job=train_job, epoch=config.epoch, config=config.prior_config, model_config=model_config
    )

    recog_model_config = ConformerCTCRecogConfig(
        logmel_cfg=model_config.logmel_cfg,
        conformer_cfg=model_config.conformer_cfg,
        dim=model_config.dim,
        target_size=model_config.target_size,
        dropout=model_config.dropout,
        specaug_cfg=model_config.specaug_cfg,
        prior_file=prior_file,
        prior_scale=config.prior_scale,
        blank_penalty=config.blank_penalty,
    )

    model_serializers = get_model_serializers(model_class=ConformerCTCRecogModel, model_config=recog_model_config)

    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.set_executables(rasr_binary_path=rasr_binary_path)

    recog_rasr_config, recog_rasr_post_config = build_config_from_mapping(crp=crp, mapping={}, include_log_config=True)

    recog_rasr_config.lib_rasr = RasrConfig()

    recog_rasr_config.lib_rasr.lexicon = RasrConfig()
    recog_rasr_config.lib_rasr.lexicon.type = "vocab-text"
    recog_rasr_config.lib_rasr.lexicon.file = config.vocab_file

    recog_rasr_config.lib_rasr.search_algorithm = RasrConfig()
    recog_rasr_config.lib_rasr.search_algorithm.type = "unconstrained-beam-search"
    recog_rasr_config.lib_rasr.search_algorithm.max_beam_size = config.max_beam_size
    if config.top_k_tokens is not None:
        recog_rasr_config.lib_rasr.search_algorithm.top_k_tokens = config.top_k_tokens
    if config.score_threshold is not None:
        recog_rasr_config.lib_rasr.search_algorithm.score_threshold = config.score_threshold

    recog_rasr_config.lib_rasr.search_algorithm.use_blank = True
    recog_rasr_config.lib_rasr.search_algorithm.blank_label_index = model_config.target_size - 1
    recog_rasr_config.lib_rasr.search_algorithm.allow_label_loop = True

    recog_rasr_config.lib_rasr.label_scorer = RasrConfig()
    recog_rasr_config.lib_rasr.label_scorer.type = "no-op"

    recog_rasr_config_path = WriteRasrConfigJob(config=recog_rasr_config, post_config=recog_rasr_post_config).out_config
    tk.register_output(f"recognition/{config.corpus_name}/{config.descriptor}/rasr.config", recog_rasr_config_path)

    recog_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
            },
            "model_outputs": {
                "tokens": {
                    "dtype": "string",
                    "feature_dim_axis": None,
                },
                "audio_samples_size": {
                    "dtype": "int32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "am_time": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "search_time": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
            },
            "backend": "torch",
            "batch_size": 36_000 * 160,
        },
        python_prolog=recipe_imports
        + [
            # ExternalImport(rasr_binary_path),
            ExternalImport(tk.Path("/work/asr4/berger/rasr_dev/label_scorer/rasr/lib/linux-x86_64-standard")),
        ],
        python_epilog=[
            *model_serializers,
            Import(
                f"{SearchCallback.__module__}.{SearchCallback.__name__}",
                import_as="forward_callback",
                unhashed_package_root=__package__,
            ),
            PartialImport(
                code_object_path=f"{rasr_recog_step.__module__}.{rasr_recog_step.__name__}",
                unhashed_package_root=__package__ or "",
                hashed_arguments={"config_file": recog_rasr_config_path},
                unhashed_arguments={},
                import_as="forward_step",
            ),
        ],  # type: ignore
        sort_config=False,
    )

    recog_returnn_config.update(config.recog_data_config.get_returnn_data())

    tk.register_output(
        f"recognition/{config.corpus_name}/{config.descriptor}/returnn.config",
        WriteReturnnConfigJob(recog_returnn_config).out_returnn_config_file,
    )

    recog_job = ReturnnForwardJobV2(
        model_checkpoint=train_job.out_checkpoints[config.epoch],
        returnn_config=recog_returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=["search_out.py", "rtf.py"],
        device=config.device,
        mem_rqmt=16,
    )
    recog_job.add_alias(f"recognition/{config.corpus_name}/{config.descriptor}")

    tk.register_output(
        f"recognition/{config.corpus_name}/{config.descriptor}/search_output", recog_job.out_files["search_out.py"]
    )

    extract_rtf_job = ExtractCTCSearchRTFJob(rtf_file=recog_job.out_files["rtf.py"])
    tk.register_output(f"recognition/{config.corpus_name}/{config.descriptor}/rtf", extract_rtf_job.out_total_rtf)

    word_file = SearchBPEtoWordsJob(recog_job.out_files["search_out.py"]).out_word_search_results

    recog_corpus_file: tk.Path = lbs_dataset.get_corpus_object_dict(audio_format="wav", output_prefix="corpora")[
        config.corpus_name
    ].corpus_file
    recog_stm = CorpusToStmJob(recog_corpus_file).out_stm_path

    ctm_file = SearchWordsToCTMJob(word_file, bliss_corpus=recog_corpus_file).out_ctm_file
    score_job = ScliteJob(ref=recog_stm, hyp=ctm_file, sort_files=True, sctk_binary_path=sctk_binary_path)
    tk.register_output(
        f"recognition/{config.corpus_name}/{config.descriptor}/scoring_reports", score_job.out_report_dir
    )

    return ResultItem(
        descriptor=config.descriptor,
        corpus_name=config.corpus_name,
        wer=score_job.out_wer,
        am_rtf=extract_rtf_job.out_am_rtf,
        search_rtf=extract_rtf_job.out_search_rtf,
        total_rtf=extract_rtf_job.out_total_rtf,
    )


def flashlight_recog(
    config: FlashlightRecogRoutineConfig, model_config: ConformerCTCConfig, train_job: ReturnnTrainingJob
) -> ResultItem:
    prior_file = compute_priors(
        train_job=train_job, epoch=config.epoch, config=config.prior_config, model_config=model_config
    )

    recog_model_config = ConformerCTCRecogConfig(
        logmel_cfg=model_config.logmel_cfg,
        conformer_cfg=model_config.conformer_cfg,
        dim=model_config.dim,
        target_size=model_config.target_size,
        dropout=model_config.dropout,
        specaug_cfg=model_config.specaug_cfg,
        prior_file=prior_file,
        prior_scale=config.prior_scale,
        blank_penalty=config.blank_penalty,
    )

    model_serializers = get_model_serializers(model_class=ConformerCTCRecogModel, model_config=recog_model_config)

    recog_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
            },
            "model_outputs": {
                "tokens": {
                    "dtype": "string",
                    "feature_dim_axis": None,
                },
                "audio_samples_size": {
                    "dtype": "int32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "am_time": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "search_time": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
            },
            "backend": "torch",
            "batch_size": 36_000 * 160,
        },
        python_prolog=recipe_imports
        + [
            # ExternalImport(rasr_binary_path),
            ExternalImport(tk.Path("/work/asr4/berger/rasr_dev/label_scorer/rasr/lib/linux-x86_64-standard")),
        ],
        python_epilog=[
            *model_serializers,
            Import(
                f"{SearchCallback.__module__}.{SearchCallback.__name__}",
                import_as="forward_callback",
                unhashed_package_root=__package__,
            ),
            PartialImport(
                code_object_path=f"{flashlight_recog_step.__module__}.{flashlight_recog_step.__name__}",
                unhashed_package_root=__package__ or "",
                hashed_arguments={
                    "vocab_file": config.vocab_file,
                    "lexicon_file": config.lexicon_file,
                    "lm_file": config.lm_file,
                    "beam_size": config.beam_size,
                    "beam_size_token": config.beam_size_token,
                    "beam_threshold": config.beam_threshold
                    if not math.isinf(config.beam_threshold)
                    else CodeWrapper('float("inf")'),
                    "lm_scale": config.lm_scale,
                },
                unhashed_arguments={},
                import_as="forward_step",
            ),
        ],  # type: ignore
        sort_config=False,
    )

    recog_returnn_config.update(config.recog_data_config.get_returnn_data())

    tk.register_output(
        f"recognition/{config.corpus_name}/{config.descriptor}/returnn.config",
        WriteReturnnConfigJob(recog_returnn_config).out_returnn_config_file,
    )

    recog_job = ReturnnForwardJobV2(
        model_checkpoint=train_job.out_checkpoints[config.epoch],
        returnn_config=recog_returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=["search_out.py", "rtf.py"],
        device=config.device,
        mem_rqmt=16,
    )
    recog_job.add_alias(f"recognition/{config.corpus_name}/{config.descriptor}")

    tk.register_output(
        f"recognition/{config.corpus_name}/{config.descriptor}/search_output", recog_job.out_files["search_out.py"]
    )

    extract_rtf_job = ExtractCTCSearchRTFJob(rtf_file=recog_job.out_files["rtf.py"])
    tk.register_output(f"recognition/{config.corpus_name}/{config.descriptor}/rtf", extract_rtf_job.out_total_rtf)

    word_file = SearchBPEtoWordsJob(recog_job.out_files["search_out.py"]).out_word_search_results

    recog_corpus_file: tk.Path = lbs_dataset.get_corpus_object_dict(audio_format="wav", output_prefix="corpora")[
        config.corpus_name
    ].corpus_file
    recog_stm = CorpusToStmJob(recog_corpus_file).out_stm_path

    ctm_file = SearchWordsToCTMJob(word_file, bliss_corpus=recog_corpus_file).out_ctm_file
    score_job = ScliteJob(ref=recog_stm, hyp=ctm_file, sort_files=True, sctk_binary_path=sctk_binary_path)
    tk.register_output(
        f"recognition/{config.corpus_name}/{config.descriptor}/scoring_reports", score_job.out_report_dir
    )

    return ResultItem(
        descriptor=config.descriptor,
        corpus_name=config.corpus_name,
        wer=score_job.out_wer,
        am_rtf=extract_rtf_job.out_am_rtf,
        search_rtf=extract_rtf_job.out_search_rtf,
        total_rtf=extract_rtf_job.out_total_rtf,
    )


def recog(config: RecogRoutineConfig, model_config: ConformerCTCConfig, train_job: ReturnnTrainingJob) -> ResultItem:
    if isinstance(config, RasrGreedyRecogRoutineConfig):
        return rasr_greedy_recog(config=config, model_config=model_config, train_job=train_job)
    if isinstance(config, RasrBeamRecogRoutineConfig):
        return rasr_beam_recog(config=config, model_config=model_config, train_job=train_job)
    if isinstance(config, FlashlightRecogRoutineConfig):
        return flashlight_recog(config=config, model_config=model_config, train_job=train_job)

    raise TypeError(f"Recognition for config type {type(config)} is not implemented")
