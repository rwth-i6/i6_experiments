import copy

from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
from i6_experiments.users.wu.experiments.label_aware_ctc.networks import conformer_rel_pos_ctc_blank_bias as conformer_ctc
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.wu.corpus.librispeech.ctc_data import get_librispeech_data_dumped_labels
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 500

tools = copy.deepcopy(default_tools_v2)
tools.returnn_root = tk.Path("/u/hwu/repositories/returnn/", hash_overwrite="/u/berger/repositories/returnn/")
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


# ********** Return Config generators **********


def returnn_config_generator(variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, extra_config:dict, lr: dict,
                             batch_size: int, network_args:dict) -> ReturnnConfig:
    model_config = conformer_ctc.get_default_config_v1(80, num_outputs, network_args)

    train_data_config["datasets"]["data"]["partition_epoch"] = 10
    extra_config.update({
        "train": train_data_config,
        "dev": dev_data_config,
        "extern_data": {"data": {"dim": 1}, "classes": {"dim": 79, "sparse": True}},
        "log_grad_norm": True,
    })
    if variant != ConfigVariant.PRIOR:
        # 11gb gpu for prior does not support bfloat
        extra_config["torch_amp"] = {"dtype": "bfloat16", "grad_scaler": None}
    if variant == ConfigVariant.RECOG:
        extra_config["model_outputs"] = {"log_probs": {"dim": num_outputs}}
    if variant == ConfigVariant.PRIOR:
        extra_config["torch_dataloader_opts"] = {"num_workers": 0}

    returnn_config = get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=1,
        num_outputs=num_outputs,
        target="classes",
        extra_python=[conformer_ctc.get_serializer(model_config, variant=variant)],
        extern_data_config=True,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        gradient_clip_global_norm=5.0,
        optimizer=Optimizers.AdamW,
        schedule=LearningRateSchedules.OCLR,
        initial_lr=lr["initial_lr"],
        peak_lr=lr["peak_lr"],
        final_lr=lr["final_lr"],
        batch_size=batch_size,
        use_chunking=False,
        extra_config=extra_config,
        keep=[50, 100, 200, 300, 400, 450, num_subepochs],
    )
    return returnn_config


def get_returnn_config_collection(
        train_data_config: dict,
        dev_data_config: dict,
        extra_config: dict,
        lr: dict,
        network_args: dict,
        batch_size: int = 36000
) -> ReturnnConfigs[ReturnnConfig]:
    generator_kwargs = {"train_data_config": train_data_config, "dev_data_config": dev_data_config, "lr": lr,
                        "batch_size": batch_size, "network_args": network_args, "extra_config":extra_config}
    train_generator_kwargs = copy.deepcopy(generator_kwargs)
    train_generator_kwargs['network_args']["recog_num_layer"] = 12
    train_config = returnn_config_generator(variant=ConfigVariant.TRAIN, **train_generator_kwargs)
    return ReturnnConfigs(
        train_config=train_config,
        prior_config=returnn_config_generator(variant=ConfigVariant.PRIOR, **generator_kwargs),
        recog_configs={"recog": returnn_config_generator(variant=ConfigVariant.RECOG, **generator_kwargs)},
    )


def run_lbs_960_conformer_rel_pos() -> SummaryReport:
    prefix = "experiments/ctc/conformer_attention_biasing_small"
    gs.ALIAS_AND_OUTPUT_SUBDIR = (
        prefix
    )

    data = get_librispeech_data_dumped_labels(
        num_classes=num_outputs,
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        add_unknown_phoneme_and_mapping=False,
        use_augmented_lexicon=True,
        feature_type=FeatureType.SAMPLES,
    )

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs)
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        epochs=[50, 100, 200, 300, 400, 450, num_subepochs],
        prior_scales=[0.3],
        lm_scales=[1.0],
        feature_type=FeatureType.SAMPLES,
        flow_args={"scale_input": 1}
    )

    # ********** System **********

    # tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
    tools.rasr_binary_path = tk.Path(
        "/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard",
        hash_overwrite="/u/berger/repositories/rasr_versions/gen_seq2seq_onnx_apptainer/arch/linux-x86_64-standard"
    )
    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=data.align_keys,
        corpus_data=data.data_inputs,
    )
    system.setup_scoring(score_kwargs={"sctk_binary_path": SCTK_BINARY_PATH})

    # ********** Returnn Configs **********
    for bias_layer_index in [6, 8, 10]:
    #for method, scale in [("tanh", 2), ("tanh", 1)]:
        start_step = 100 * 370 # start after 100 subepochs(20% steps) for now 
        peak_lr = 1e-3
        aux_losses = {"12": 1, str(bias_layer_index): 0.3}
        network_args = {
            "aux_losses": aux_losses,
            "d_model": 512,
            "bias_layer_index": bias_layer_index,
            "compute_bias_type": "learnable_embedding",
            #"compute_bias_args": {"method": method, "scale": scale, "start_step": start_step},
            "compute_bias_args": {"start_step": start_step, "embed_dim": 512//8}
        }
        peak_lr_dict = {
            "initial_lr": peak_lr / 100,
            "peak_lr": peak_lr,
            "final_lr": 1e-8,
        }
        str_peak_lr = str(peak_lr).replace("-", "_").replace(".", "_")

        extra_config = {}
        system.add_experiment_configs(
            f"from_scratch_lbs_960_lr{str_peak_lr}_biasLayer{bias_layer_index}_learnableEmbed",
            get_returnn_config_collection(data.train_data_config, data.cv_data_config,
                                            extra_config=extra_config, lr=peak_lr_dict,
                                            batch_size=100000 * 160, network_args=network_args)
        )

    system.run_train_step(**train_args)
    for job in system._train_jobs.values():
        job.rqmt.update({"gpu_mem": 48})
    system.run_dev_recog_step(**recog_args)
    #system.run_test_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report

