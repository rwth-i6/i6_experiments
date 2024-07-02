import copy

from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
# from i6_experiments.users.berger.pytorch.models import conformer_ctc
# from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks import conformer_ctc_downsample_4 as conformer_ctc
from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks.baseline import \
    conformer_ctc_d_model_512_num_layers_12_new_frontend_raw_wave as conformer_ctc
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.corpus.librispeech.ctc_data import get_librispeech_data_hdf
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.jxu.experiments.ctc.lbs_960.ctc_data import get_librispeech_data_hdf
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.phan.models import multi_model_wrapper, lstm_lm
from i6_experiments.common.setups.serialization import ExplicitHash

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 100

tools = copy.deepcopy(default_tools_v2)
# tools.returnn_root = tk.Path("/u/minh-nghia.phan/tools/simon_returnn") # Sis will ask to run HDF jobs again
# tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/onnx/arch/linux-x86_64-standard")
# tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


# ********** Return Config generators **********


def returnn_config_generator(
    variant: ConfigVariant,
    train_data_config: dict,
    dev_data_config: dict,
    lr: dict,
    batch_size: int,
    conformer_ctc_args: dict,
    lstm_lm_args: dict,
    module_preloads: dict,
    optimizer: Optimizers,
    schedule: LearningRateSchedules,
    mask_ratio,
    mask_audio,
    kwargs: dict,
) -> ReturnnConfig:
    conformer_ctc_args["num_inputs"] = 80
    conformer_ctc_args["num_outputs"] = num_outputs
    conformer_ctc_config = conformer_ctc.get_default_config_v1(**conformer_ctc_args)
    
    lstm_lm_config = lstm_lm.LSTMLMConfig(**lstm_lm_args)

    wrapper_config = multi_model_wrapper.MultiModelWrapperConfig(
        module_class={
            'teacher_ctc': conformer_ctc.ConformerCTCModel,
            'student_lm': lstm_lm.LSTMLM,
        },
        module_config={
            'teacher_ctc': conformer_ctc_config,
            'student_lm': lstm_lm_config,
        },
        module_preload=module_preloads,
    )

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
    }
    if variant == ConfigVariant.RECOG:
        extra_config["model_outputs"] = {"classes": {"dim": num_outputs}}
    module_class_import = {
        'teacher_ctc': "i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks.baseline.conformer_ctc_d_model_512_num_layers_12_new_frontend_raw_wave",
        'student_lm': "i6_experiments.users.phan.models.lstm_lm"
    }
    kwargs.update(lr) # diff LR scheduler will have diff parameters
    # kwargs for serializer
    if variant == ConfigVariant.TRAIN:
        prologue_serializers_kwargs = {
            "train_step_package": "i6_experiments.users.phan.ctc.train_steps.kldiv_ctc_masked_lm",
            "partial_train_step": True,
            "partial_kwargs": {
                "hashed_arguments": {
                    "mask_ratio": mask_ratio,
                    "mask_idx": num_outputs-1,
                    "mask_audio": mask_audio,
                    "sil_index": 0,
                },
                "unhashed_arguments": {},
            }
        }
    elif variant == ConfigVariant.PRIOR:
        prologue_serializers_kwargs = {
            "forward_step_package": "i6_experiments.users.phan.ctc.forward.multi_model_conformer_ctc",
            "prior_package": "i6_experiments.users.phan.ctc.forward.prior_callback",
            "partial_forward_step": True,
            "partial_kwargs": {
                "hashed_arguments": {
                    "conformer_ctc_name": "conformer_ctc",
                },
                "unhashed_arguments": {},
            }
        }
    elif variant == ConfigVariant.RECOG:
        prologue_serializers_kwargs = {
            "export_package": "i6_experiments.users.phan.ctc.exports.conformer_ctc",
            "partial_export": True,
            "partial_kwargs": {
                "hashed_arguments": {
                    "conformer_ctc_name": "conformer_ctc",
                },
                "unhashed_arguments": {},
            }
        }
    extern_data_dict = {"data": {"dim": 1}, "targets": {"dim": 79, "sparse": True}}
    if mask_audio:
        extern_data_dict.update({"align": {"dim": num_outputs, "sparse": True}})
    return get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=1,
        num_outputs=num_outputs,
        target="targets",
        extra_python=[
            multi_model_wrapper.get_serializer(
                model_config=wrapper_config,
                module_class_import=module_class_import,
                variant=variant,
                prologue_serializers_kwargs=prologue_serializers_kwargs,
            )
        ],
        extern_data_config=True,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip_global_norm=2.0,
        optimizer=optimizer,
        schedule=schedule,
        max_seqs=60,
        batch_size=batch_size,
        use_chunking=False,
        extra_config=extra_config,
        extern_data_dict=extern_data_dict, # align should be 79 here
        **kwargs
    )


def get_returnn_config_collection(
        train_data_config: dict,
        dev_data_config: dict,
        lr: dict,
        conformer_ctc_args: dict,
        lstm_lm_args: dict,
        module_preloads: dict,
        batch_size: int,
        optimizer: Optimizers,
        schedule: LearningRateSchedules,
        mask_ratio: float,
        mask_audio: bool,
        kwargs: dict
) -> ReturnnConfigs[ReturnnConfig]:
    generator_kwargs = {
        "train_data_config": train_data_config,
        "dev_data_config": dev_data_config,
        "lr": lr,
        "batch_size": batch_size,
        "conformer_ctc_args": conformer_ctc_args,
        "lstm_lm_args": lstm_lm_args,
        "module_preloads": module_preloads,
        "optimizer": optimizer,
        "schedule": schedule,
        "mask_ratio": mask_ratio,
        "mask_audio": mask_audio,
        "kwargs":kwargs
    }
    return ReturnnConfigs(
        train_config=returnn_config_generator(variant=ConfigVariant.TRAIN, **generator_kwargs),
        prior_config=returnn_config_generator(variant=ConfigVariant.PRIOR, **generator_kwargs),
        recog_configs={"recog": returnn_config_generator(variant=ConfigVariant.RECOG, **generator_kwargs)},
    )


def lbs_960_run_kldiv_ctc_masked_lstm() -> SummaryReport:
    prefix = "experiments/ctc/kldiv_ctc_masked_lstm"
    gs.ALIAS_AND_OUTPUT_SUBDIR = (
        prefix
    )

    data = get_librispeech_data_hdf(
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        augmented_lexicon=True,
        feature_type=FeatureType.SAMPLES,
        blank_index_last=False,
        use_alignments_in_train="gmm",
    )

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=num_subepochs,
        gpu_mem_rqmt=11,
    )
    # recog_args = exp_args.get_ctc_recog_step_args(
    #     num_classes=num_outputs,
    #     epochs=[num_subepochs],
    #     prior_scales=[0.45, 0.5, 0.55],
    #     lm_scales=[0.9,1.0,1.1],
    #     feature_type=FeatureType.SAMPLES,
    #     flow_args={"scale_input": 1}
    # )
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        epochs=[num_subepochs],
        prior_scales=[0.45],
        lm_scales=[0.9],
        feature_type=FeatureType.SAMPLES,
        flow_args={"scale_input": 1}
    )

    # ********** System **********

    # tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
    tools.rasr_binary_path = tk.Path(
        "/u/berger/repositories/rasr_versions/nour_seq2seq_onnx_apptainer/arch/linux-x86_64-standard"
    )
    tools.returnn_root = tk.Path("/u/minh-nghia.phan/tools/simon_returnn")
    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.ctc_recog_am_args,
    )
    system.setup_scoring(score_kwargs={"sctk_binary_path": SCTK_BINARY_PATH})

    # ********** Returnn Configs **********
    for n_lstm_layers in [1]:
        for mask_audio in [True, False]:
            for init_learning_rate in [1e-3, 1e-4]:
                for mask_ratio in [0.2, 0.4]:
                    conformer_ctc_args = {
                        "time_max_mask_per_n_frames": 25,
                        "freq_max_num_masks": 5,
                        "vgg_act": "relu",
                        "dropout": 0.1,
                        "num_layers": 12,
                    }
                    lstm_lm_args = {
                        "vocab_dim": num_outputs, # 79
                        "output_dim": num_outputs-1,
                        "embed_dim": 128,
                        "hidden_dim": 640,
                        "n_lstm_layers": n_lstm_layers,
                        "dropout": 0.1,
                        "bidirectional": True,
                    }
                    module_preloads = {
                        "teacher_ctc": "/work/asr4/zyang/torch/librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.nuHCdB8qL7NJ/output/models/epoch.600.pt"
                    }
                    newbob_lr = {
                        "learning_rate": init_learning_rate,
                        "decay": 0.9 ,
                        "multi_num_epochs": 20,
                        "relative_error_threshold": -0.005,
                    }
                    mask_audio_str = "maskAudio" if mask_audio else "noMaskAudio"
                    system.add_experiment_configs(
                        f"kldiv_ctc_masked_lstm_ctc{12}_lstm{n_lstm_layers}_adamw_newbobrel_lr{init_learning_rate}_maskRatio{mask_ratio}_{mask_audio_str}_eps{num_subepochs}",
                        get_returnn_config_collection(
                            data.train_data_config,
                            data.cv_data_config,
                            lr=newbob_lr,
                            batch_size=15000 * 160,
                            conformer_ctc_args=conformer_ctc_args,
                            lstm_lm_args=lstm_lm_args,
                            module_preloads=module_preloads,
                            optimizer=Optimizers.AdamW,
                            schedule=LearningRateSchedules.NewbobRel,
                            mask_ratio=mask_ratio,
                            mask_audio=mask_audio,
                            kwargs={"weight_decay": 0.001},
                        )
                    )

    system.run_train_step(**train_args)

    assert system.summary_report
    return system.summary_report

