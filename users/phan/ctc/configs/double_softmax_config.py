import copy
import inspect

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
from i6_experiments.users.phan.ctc.ctc_pref_scores_loss import (
    ctc_double_softmax_loss, log_ctc_pref_beam_scores,
)

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
    am_scale: float,
    lm_scale: float,
    kwargs: dict,
) -> ReturnnConfig:
    conformer_ctc_args["num_inputs"] = 80
    conformer_ctc_args["num_outputs"] = num_outputs
    conformer_ctc_config = conformer_ctc.get_default_config_v1(**conformer_ctc_args)
    
    lstm_lm_config = lstm_lm.LSTMLMConfig(**lstm_lm_args)

    # Preload when init is only needed in train
    wrapper_config_kwargs = {
        'module_class': {
            'conformer_ctc': conformer_ctc.ConformerCTCModel,
            'train_lm': lstm_lm.LSTMLM,
        },
        'module_config': {
            'conformer_ctc': conformer_ctc_config,
            'train_lm': lstm_lm_config,
        },
        'module_preload': None,
    }
    if variant == ConfigVariant.TRAIN:
        wrapper_config_kwargs['module_preload'] = module_preloads
    wrapper_config = multi_model_wrapper.MultiModelWrapperConfig(**wrapper_config_kwargs)

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
    }
    if variant == ConfigVariant.RECOG:
        extra_config["model_outputs"] = {"classes": {"dim": num_outputs}}
    module_class_import = {
        'conformer_ctc': "i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks.baseline.conformer_ctc_d_model_512_num_layers_12_new_frontend_raw_wave",
        'train_lm': "i6_experiments.users.phan.models.lstm_lm"
    }
    kwargs.update(lr) # diff LR scheduler will have diff parameters
    
    # kwargs for serializer
    if variant == ConfigVariant.TRAIN:
        prologue_serializers_kwargs = {
            "train_step_package": "i6_experiments.users.phan.ctc.train_steps.double_softmax",
            "partial_train_step": True,
            "partial_kwargs": {
                "hashed_arguments": {
                    "am_scale": am_scale,
                    "lm_scale": lm_scale,
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
            ),
            ExplicitHash(inspect.getsource(ctc_double_softmax_loss)),
            ExplicitHash(inspect.getsource(log_ctc_pref_beam_scores)),
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
        am_scale: float,
        lm_scale: float,
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
        "am_scale": am_scale,
        "lm_scale": lm_scale,
        "kwargs":kwargs
    }
    return ReturnnConfigs(
        train_config=returnn_config_generator(variant=ConfigVariant.TRAIN, **generator_kwargs),
        prior_config=returnn_config_generator(variant=ConfigVariant.PRIOR, **generator_kwargs),
        recog_configs={"recog": returnn_config_generator(variant=ConfigVariant.RECOG, **generator_kwargs)},
    )


def lbs_960_double_softmax() -> SummaryReport:
    prefix = "experiments/ctc/double_softmax"
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
    )

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=num_subepochs,
        gpu_mem_rqmt=24, # now this, but optimize ctc pref score later
    )
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        epochs=list(range(20, num_subepochs+1, 20)),
        prior_scales=[0.45, 0.5, 0.55],
        lm_scales=[0.9,1.0,1.1],
        feature_type=FeatureType.SAMPLES,
        flow_args={"scale_input": 1}
    )

    # ********** System **********

    # tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
    tools.rasr_binary_path = tk.Path(
        "/u/minh-nghia.phan/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard"
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
    conformer_ctc_args = {
        "time_max_mask_per_n_frames": 25,
        "freq_max_num_masks": 5,
        "vgg_act": "relu",
        "dropout": 0.1,
        "num_layers": 12
    }
    lstm_lm_args = {
        "vocab_dim": num_outputs, # 79
        "embed_dim": 128,
        "hidden_dim": 640,
        "n_lstm_layers": 1,
        "dropout": 0.1,
    }
    module_preloads = {
        "conformer_ctc": "/work/asr4/zyang/torch/librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.nuHCdB8qL7NJ/output/models/epoch.600.pt",
        "train_lm": "/work/asr3/zyang/share/mnphan/work_torch_ctc_libri/i6_core/returnn/training/ReturnnTrainingJob.7bqxeOpBaeEk/output/models/epoch.120.pt"
    }
    for am_scale in [0.7, 1.0, 1.3]:
        for lm_scale in [0.3]:
            for learning_rate in [1e-4, 1e-5, 1e-6]:
                lr_dict = {
                    "learning_rate": learning_rate,
                }
                system.add_experiment_configs(
                    f"double_softmax_ctc{12}_lstm{1}_am{am_scale}_lm{lm_scale}_adamw_const_lr{learning_rate}_eps{num_subepochs}",
                    get_returnn_config_collection(
                        data.train_data_config,
                        data.cv_data_config,
                        lr=lr_dict,
                        batch_size=15000 * 160,
                        conformer_ctc_args=conformer_ctc_args,
                        lstm_lm_args=lstm_lm_args,
                        module_preloads=module_preloads,
                        optimizer=Optimizers.AdamW,
                        schedule=LearningRateSchedules.CONST_LR,
                        am_scale=am_scale,
                        lm_scale=lm_scale,
                        kwargs={"weight_decay": 0.001},
                    )
                )

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)
    system.run_test_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report

