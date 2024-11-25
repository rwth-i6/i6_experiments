import os
from typing import List

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
from i6_experiments.common.setups.serialization import (
    Collection,
    ExternalImport,
    Import,
    PartialImport,
)
from sisyphus import gs, tk
from sisyphus.delayed_ops import DelayedBase

from .data.bpe import get_bpe_vocab_file, get_recog_data, get_train_data
from .pytorch_models.ctc import ConformerCTCModel, ConformerCTCRecogConfig, ConformerCTCRecogModel, get_model_config
from .steps.ctc import ComputePriorCallback, SearchCallback, prior_step, rasr_recog_step, train_step
from .tools import rasr_binary_path, returnn_python_exe, returnn_root, sctk_binary_path


def py() -> None:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    # =================================
    # ==== Training ===================
    # =================================

    train_data = get_train_data(bpe_size=128)

    save_epochs = [10] + list(range(100, 901, 100)) + list(range(900, 1001, 20))

    model_config = get_model_config(target_size=185)

    train_constructor_call, train_model_imports = build_config_constructor_serializers_v2(
        cfg=model_config,
        variable_name="cfg",
        unhashed_package_root=__package__,
    )

    recipe_import = ExternalImport(
        import_path=tk.Path(
            f"{__file__.split('recipe')[0]}/recipe/",
            hash_overwrite="RECIPE_ROOT",
        )
    )

    model_serializers: List[DelayedBase] = [
        Import(
            f"{ConformerCTCModel.__module__}.{ConformerCTCModel.__name__}",
            unhashed_package_root=__package__,
        ),
    ]
    model_serializers.append(Collection(train_model_imports))  # type: ignore
    model_serializers.append(train_constructor_call)
    model_serializers.append(
        PyTorchModel(
            model_class_name=ConformerCTCModel.__name__,
            model_kwargs={"cfg": CodeWrapper("cfg")},
        )
    )

    train_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
                "classes": {"dim": 184, "sparse": True, "dtype": "int32"},
            },
            "train": train_data.train_config_dict,
            "dev": train_data.dev_config_dict,
            "backend": "torch",
            "batch_size": 36_000 * 160,
            "cleanup_old_models": {
                "keep_last_n": 1,
                "keep_best_n": 0,
                "keep": save_epochs,
            },
            "gradient_clip": 1.0,
            "learning_rates": CodeWrapper(
                "list(np.linspace(7e-06, 5e-04, 480))"
                "+ list(np.linspace(5e-04, 5e-05, 480))"
                "+ list(np.linspace(5e-05, 1e-07, 40))"
            ),
            "torch_amp": {"dtype": "bfloat16"},
            "num_workers_per_gpu": 2,
            "stop_on_nonfinite_train_score": True,
            "optimizer": {
                "class": "adamw",
                "epsilon": 1e-16,
                "weight_decay": 0.01,
            },
        },
        python_prolog=[
            "import sys",
            "import numpy as np",
            recipe_import,
            train_data.extra_serializers,
        ],
        python_epilog=[
            *model_serializers,
            Import(
                f"{train_step.__module__}.{train_step.__name__}",
                unhashed_package_root=__package__,
            ),
        ],  # type: ignore
        sort_config=False,
    )
    tk.register_output(
        "train.returnn.config",
        WriteReturnnConfigJob(train_returnn_config).out_returnn_config_file,
    )

    train_job = ReturnnTrainingJob(
        returnn_config=train_returnn_config,
        log_verbosity=5,
        num_epochs=1000,
        time_rqmt=168,
        mem_rqmt=24,
        cpu_rqmt=6,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    )
    train_job.rqmt["gpu_mem"] = 24

    tk.register_output("learning_rates", train_job.out_learning_rates)

    # =================================
    # ==== Prior ======================
    # =================================

    del train_data.train_config_dict["datasets"]["data"]["audio"]["pre_process"]

    for epoch in save_epochs[-1:]:
        prior_returnn_config = ReturnnConfig(
            config={
                "extern_data": {
                    "data": {"dim": 1, "dtype": "float32"},
                    "classes": {"dim": 184, "sparse": True, "dtype": "int32"},
                },
                "forward_data": train_data.train_config_dict,
                "backend": "torch",
                "batch_size": 20_000 * 160,
                "num_workers_per_gpu": 2,
            },
            python_prolog=[
                "import sys",
                recipe_import,
            ],
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
        tk.register_output(
            f"prior.e-{epoch}.returnn.config",
            WriteReturnnConfigJob(prior_returnn_config).out_returnn_config_file,
        )

        prior_job = ReturnnForwardJobV2(
            model_checkpoint=train_job.out_checkpoints[epoch],
            returnn_config=prior_returnn_config,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            output_files=["prior.txt"],
        )

        tk.register_output(f"prior.e-{epoch}.txt", prior_job.out_files["prior.txt"])

        # =================================
        # ==== Recog ======================
        # =================================

        wers = {}

        for prior_scale in [0.0, 0.3, 0.5]:
            for blank_penalty in [0.0, 0.5, 1.0]:
                suffix = f"prior-{prior_scale}_bp-{blank_penalty}"
                recog_model_config = ConformerCTCRecogConfig(
                    logmel_cfg=model_config.logmel_cfg,
                    conformer_cfg=model_config.conformer_cfg,
                    dim=model_config.dim,
                    target_size=model_config.target_size,
                    dropout=model_config.dropout,
                    specaug_cfg=model_config.specaug_cfg,
                    prior_file=prior_job.out_files["prior.txt"],
                    prior_scale=prior_scale,
                    blank_penalty=blank_penalty,
                )

                recog_constructor_call, recog_model_imports = build_config_constructor_serializers_v2(
                    cfg=recog_model_config,
                    variable_name="cfg",
                    unhashed_package_root=__package__,
                )

                recog_model_serializers: List[DelayedBase] = [
                    Import(
                        f"{ConformerCTCRecogModel.__module__}.{ConformerCTCRecogModel.__name__}",
                        unhashed_package_root=__package__,
                    ),
                ]
                recog_model_serializers.append(Collection(recog_model_imports))  # type: ignore
                recog_model_serializers.append(recog_constructor_call)
                recog_model_serializers.append(
                    PyTorchModel(
                        model_class_name=ConformerCTCRecogModel.__name__,
                        model_kwargs={"cfg": CodeWrapper("cfg")},
                    )
                )

                crp = CommonRasrParameters()
                crp_add_default_output(crp)
                crp.set_executables(rasr_binary_path=rasr_binary_path)

                recog_rasr_config, recog_rasr_post_config = build_config_from_mapping(
                    crp=crp, mapping={}, include_log_config=True
                )

                recog_rasr_config.lib_rasr = RasrConfig()

                recog_rasr_config.lib_rasr.lexicon = RasrConfig()
                recog_rasr_config.lib_rasr.lexicon.type = "vocab-text"
                recog_rasr_config.lib_rasr.lexicon.file = get_bpe_vocab_file(bpe_size=128)

                recog_rasr_config.lib_rasr.search_algorithm = RasrConfig()
                recog_rasr_config.lib_rasr.search_algorithm.type = "unconstrained-greedy-search"
                recog_rasr_config.lib_rasr.search_algorithm.use_blank = True
                recog_rasr_config.lib_rasr.search_algorithm.blank_index = 184
                recog_rasr_config.lib_rasr.search_algorithm.allow_label_loop = True

                recog_rasr_config.lib_rasr.label_scorer = RasrConfig()
                recog_rasr_config.lib_rasr.label_scorer.type = "no-op"

                recog_rasr_config_path = WriteRasrConfigJob(
                    config=recog_rasr_config, post_config=recog_rasr_post_config
                ).out_config
                tk.register_output(f"recog.e-{epoch}.{suffix}.rasr.config", recog_rasr_config_path)

                for recog_corpus_name in ["dev-clean", "dev-other"]:
                    recog_returnn_config = ReturnnConfig(
                        config={
                            "forward_data": get_recog_data(corpus_name=recog_corpus_name),
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
                            "num_workers_per_gpu": 2,
                        },
                        python_prolog=[
                            "import sys",
                            recipe_import,
                            # ExternalImport(rasr_binary_path),
                            ExternalImport(
                                tk.Path("/work/asr4/berger/rasr_dev/label_scorer/rasr/lib/linux-x86_64-debug")
                            ),
                        ],
                        python_epilog=[
                            *recog_model_serializers,
                            Import(
                                f"{SearchCallback.__module__}.{SearchCallback.__name__}",
                                import_as="forward_callback",
                                unhashed_package_root=__package__,
                            ),
                            # Collection(
                            #     [
                            #         Import("librasr.Configuration"),
                            #         Import("librasr.SearchAlgorithm"),
                            #     ]
                            # ),
                            # Collection(
                            #     [
                            #         # Call(
                            #         #     callable_name="config.set_from_file",
                            #         #     kwargs=[("filename", DelayedFormat('"{}"', recog_rasr_config_path))],
                            #         # ),
                            #     ]
                            # ),
                            # Call(callable_name="Configuration", return_assign_variables="config"),
                            # DelayedFormat('config.set_from_file("{}")', recog_rasr_config_path),
                            # Call(
                            #     callable_name="SearchAlgorithm",
                            #     kwargs=[("config", CodeWrapper("config"))],
                            #     return_assign_variables="search_algorithm",
                            # ),
                            PartialImport(
                                code_object_path=f"{rasr_recog_step.__module__}.{rasr_recog_step.__name__}",
                                unhashed_package_root=__package__ or "",
                                hashed_arguments={},
                                unhashed_arguments={"config_file": recog_rasr_config_path},
                                # unhashed_arguments={"search_function": CodeWrapper("search_algorithm.recognize_segment")},
                                import_as="forward_step",
                            ),
                        ],  # type: ignore
                        sort_config=False,
                    )

                    tk.register_output(
                        f"{recog_corpus_name}.e-{epoch}.{suffix}.returnn.config",
                        WriteReturnnConfigJob(recog_returnn_config).out_returnn_config_file,
                    )

                    recog_job = ReturnnForwardJobV2(
                        model_checkpoint=train_job.out_checkpoints[epoch],
                        returnn_config=recog_returnn_config,
                        returnn_python_exe=returnn_python_exe,
                        returnn_root=returnn_root,
                        output_files=["search_out.py"],
                    )
                    tk.register_output(
                        f"{recog_corpus_name}.e-{epoch}.{suffix}.search_out.py", recog_job.out_files["search_out.py"]
                    )

                    word_file = SearchBPEtoWordsJob(recog_job.out_files["search_out.py"]).out_word_search_results

                    recog_corpus_file: tk.Path = lbs_dataset.get_corpus_object_dict(
                        audio_format="wav", output_prefix="corpora"
                    )[recog_corpus_name].corpus_file
                    recog_stm = CorpusToStmJob(recog_corpus_file).out_stm_path

                    ctm_file = SearchWordsToCTMJob(word_file, bliss_corpus=recog_corpus_file).out_ctm_file
                    score_job = ScliteJob(
                        ref=recog_stm, hyp=ctm_file, sort_files=True, sctk_binary_path=sctk_binary_path
                    )
                    tk.register_output(f"{recog_corpus_name}.e-{epoch}.{suffix}.reports", score_job.out_report_dir)

                    wers[(recog_corpus_name, epoch, prior_scale, blank_penalty)] = score_job.out_wer

        tk.register_report("results.txt", values=wers, required=True)
