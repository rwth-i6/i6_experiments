import copy

from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
from i6_experiments.users.wu.experiments.modality_matching.networks import maestro as maestro
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.wu.corpus.librispeech.maestro_data import get_librispeech_data
from i6_experiments.users.wu.corpus.librispeech.unsupervised_data import get_librispeech_unsupervised
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 200  # count based on lbs-960: 20 full ep with partition ep 10(only less than 1/3 full ep on LM data)

tools = copy.deepcopy(default_tools_v2)
tools.returnn_root = tk.Path("/u/hwu/repositories/returnn/", hash_overwrite="/u/berger/repositories/returnn/")
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


# ********** Return Config generators **********


def returnn_config_generator(variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, extra_config:dict, lr: dict,
                             batch_size: int, network_args:dict, python_prolog=None) -> ReturnnConfig:
    model_config = maestro.get_default_config_v1(80, num_outputs, network_args)

    extra_config.update({
        "train": train_data_config,
        "dev": dev_data_config,
        "torch_dataloader_opts": {"num_workers": 0},
        "torch_log_memory_usage": True,
        "log_grad_norm": True,
        "gradient_clip_global_norm": 5.0,  # better than simple gradient clipping!
    })
    if variant == ConfigVariant.RECOG:
        extra_config["model_outputs"] = {"log_probs": {"dim": num_outputs}}
    if variant == ConfigVariant.TRAIN:
        extra_config["max_seq_length"] = {"text": 400}
        # text has no blank but align has it
        extra_config["extern_data"] = {"audio": {"dim": 1}, "text": {"dim": num_outputs-1, "sparse": True}, "paired_audio": {"dim": 1}, "paired_align": {"dim": num_outputs, "sparse": True}}
    else:
        # only forward audio data is enough
        extra_config["extern_data"] = {"data": {"dim": 1}}
        batch_size = 20000*160

    returnn_config = get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=1,
        num_outputs=num_outputs,
        target="classes",
        extra_python=[maestro.get_serializer(model_config, variant=variant)],
        extern_data_config=True,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        optimizer=Optimizers.AdamW,
        schedule=LearningRateSchedules.OCLR,
        initial_lr=lr["initial_lr"],
        peak_lr=lr["peak_lr"],
        final_lr=lr["final_lr"],
        batch_size=batch_size,
        use_chunking=False,
        python_prolog=[python_prolog],  # dataset config needs to be passed here!
        extra_config=extra_config,
        keep=[10, 20, 50, 100, 150, num_subepochs],
    )
    return returnn_config


def get_returnn_config_collection(
        train_data_config: dict,
        dev_data_config: dict,
        extra_config: dict,
        lr: dict,
        network_args: dict,
        python_prolog=None,
        batch_size: int = 36000
) -> ReturnnConfigs[ReturnnConfig]:
    generator_kwargs = {"train_data_config": train_data_config, "dev_data_config": dev_data_config, "lr": lr,
                        "batch_size": batch_size, "network_args": network_args, "extra_config":extra_config}
    train_generator_kwargs = copy.deepcopy(generator_kwargs)
    train_generator_kwargs['network_args']["recog_num_layer"] = 12
    if python_prolog:
        train_generator_kwargs["python_prolog"] = python_prolog
    train_config = returnn_config_generator(variant=ConfigVariant.TRAIN, **train_generator_kwargs)
    forward_generator_kwargs = copy.deepcopy(generator_kwargs)
    forward_generator_kwargs["train_data_config"] = get_librispeech_unsupervised(
        returnn_root=tk.Path("/u/jxu/setups/pretraining/2025-02-28--best-rq-pretraining/tools/20241021_returnn/returnn"),
    )[0] 
    return ReturnnConfigs(
        train_config=train_config,
        prior_config=returnn_config_generator(variant=ConfigVariant.PRIOR, **forward_generator_kwargs),
        recog_configs={"recog": returnn_config_generator(variant=ConfigVariant.RECOG, **forward_generator_kwargs)},
    )


def run_lbs_maestro(alignments) -> SummaryReport:
    prefix = "experiments/maestro"
    gs.ALIAS_AND_OUTPUT_SUBDIR = (
        prefix
    )

    data, train_data_func = get_librispeech_data(
        num_classes=num_outputs,
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        alignments=alignments,
        rasr_binary_path=tools.rasr_binary_path,
        feature_type=FeatureType.SAMPLES,
        add_unknown_phoneme_and_mapping=False,
        use_augmented_lexicon=True,
    )

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=num_subepochs,
    )
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        epochs=[10, 20, 50, 100, 150, num_subepochs],
        prior_scales=[0.3],
        lm_scales=[1.0],
        feature_type=FeatureType.SAMPLES,
        flow_args={"scale_input": 1}
    )

    # ********** System **********

    tools.rasr_binary_path = tk.Path(
        "/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard",
        hash_overwrite="/u/berger/repositories/rasr_versions/gen_seq2seq_onnx_apptainer/arch/linux-x86_64-standard"
    )
    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=[],
        corpus_data=data.data_inputs,
    )
    system.setup_scoring(score_kwargs={"sctk_binary_path": SCTK_BINARY_PATH})

    # ********** Returnn Configs **********

    for text_ctc_loss_scale in [0.1, 0.3, 0.5]:
        peak_lr = 1e-3
        # same as in preliminary exp
        best_rq_args = {
            "num_outputs": 8192,
            "num_layers": 4,
            "aux_losses": {"4": 1},
            "input_codebook_dim": 16,
            "input_codebook_num_vars": 8192,
            "mask_replace_val": "zero",
            "mask_prob": 0.6,
            "mask_length": 16,
            "cb_distance_measure": "L2_norm",
            "internal_subsampling_rate": 4,
            "normalise_after_PCA": True,
            "d_model": 768,
        }
        network_args = {
            "best_rq_args": best_rq_args,
            "loss_scales": {"ctc_paired": 1.0, "ctc_text": text_ctc_loss_scale, "match": 1.0, "best_rq": 0.5},
            "text_encoder_args": {"frame_drop_p": 0.5},  # harder target needed
        }

        peak_lr_dict = {
            "initial_lr": 4e-5,
            "peak_lr": peak_lr,
            "final_lr": 1e-8,
        }
        str_peak_lr = str(peak_lr).replace("-", "_").replace(".", "_")

        extra_config = {
            "preload_from_files":{
                "best_rq": {
                    "filename": "/work/asr4/hwu/setups/librispeech-960/2025-09-16-nar-asr-text/i6_core/returnn/training/ReturnnTrainingJob.founbOZMDyM5/output/models/epoch.500.pt",
                    "checkpoint_key": "model",
                    "init_for_train": True,
                    "ignore_missing": True,
                    "prefix": "best_rq.",
                },
                "duration_model": {
                    "filename": "/work/asr4/hwu/setups/librispeech-960/2025-09-16-nar-asr-text/i6_core/returnn/training/ReturnnTrainingJob.8iRx85GoHKDo/output/models/epoch.400.pt",
                    "checkpoint_key": "model",
                    "init_for_train": True,
                    "ignore_missing": False,
                    "prefix": "duration.",
                }
            }
        }
        system.add_experiment_configs(
            f"audio_lbs_960_paired_lbs_100_lr_{str_peak_lr}_textDrop0.5_textScale{text_ctc_loss_scale}",
            get_returnn_config_collection(data.train_data_config, data.cv_data_config,
                                            extra_config=extra_config, lr=peak_lr_dict,
                                            batch_size={"audio":30000 * 160, "paired_audio": 30000 * 160, "text": 12000}, 
                                            python_prolog=train_data_func,
                                            network_args=network_args)
        )

        system.run_train_step(**train_args)
        for job in system._train_jobs.values():
            job.rqmt.update({"gpu_mem": 48})
        system.run_dev_recog_step(**recog_args)
        system.run_test_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report

