import copy
from dataclasses import asdict,  dataclass
import numpy as np
from typing import cast, List, Dict, Any, Optional

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset, TrainingDatasets
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel, quantize_static
from ...report import generate_report
from ...config import get_static_quant_config

@dataclass
class QuantArgs:
    sample_ls: List[int]
    quant_config_dict: Dict[str, Any]
    decoder: str
    num_iterations: int
    datasets: TrainingDatasets
    network_module: str

QUANT_RETURNN = CloneGitRepositoryJob(
    "https://github.com/JackTemaki/MiniReturnn", commit="f31614f2a071aa75588eff6f2231b54751fb962c"
).out_repository.copy()


def eow_phon_ted_1023_base():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    dev_dataset_tuples = {}
    for testset in ["dev"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig

    def tune_and_evaluate_helper(
        training_name: str,
        asr_model: ASRModel,
        base_decoder_config: DecoderConfig,
        lm_scales: List[float],
        prior_scales: List[float],
        quant_str: Optional[str] = None,
        eval_test: bool = False,
        quant_args: Optional[QuantArgs] =  None,
    ):
        """
        Example helper to execute tuning over lm_scales and prior scales.
        With the best values runs test-clean and test-other.

        This is just a reference helper and can (should) be freely changed, copied, modified etc...

        :param training_name: for alias and output names
        :param asr_model: ASR model to use
        :param base_decoder_config: any decoder config dataclass
        :param lm_scales: lm scales for tuning
        :param prior_scales: prior scales for tuning, same length as lm scales
        """
        tune_parameters = []
        tune_values = []
        results = {}
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                decoder_config.lm_weight = lm_weight
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config={},
                    asr_model=asr_model,
                    decoder_module="ctc.decoder.flashlight_ctc_v1",
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    **default_returnn,
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values.append((wers[search_name + "/dev"]))
                results.update(wers)
        if quant_args is not None:
            assert quant_str is not None, "You want your quant to have a name"
            for num_samples in quant_args.sample_ls:
                for seed in range(quant_args.num_iterations):
                    it_name = training_name + quant_str + f"/quantize_static/samples_{num_samples}/seed_{seed}"
                    quant_config = get_static_quant_config(
                        training_datasets=quant_args.datasets,
                        network_module=quant_args.network_module,
                        net_args=asr_model.net_args,
                        quant_args=quant_args.quant_config_dict,
                        config={},
                        num_samples=num_samples,
                        dataset_seed=seed,
                        debug=False,
                    )
                    quant_chkpt = quantize_static(
                        prefix_name=it_name,
                        returnn_config=quant_config,
                        checkpoint=asr_model.checkpoint,
                        returnn_exe=RETURNN_EXE,
                        returnn_root=QUANT_RETURNN,
                    )
                    quant_model = ASRModel(
                        checkpoint=quant_chkpt,
                        net_args=asr_model.net_args | quant_args.quant_config_dict,
                        network_module=quant_args.network_module,
                        prior_file=asr_model.prior_file,
                        prefix_name=it_name
                    )
                    for lm_weight in lm_scales:
                        for prior_scale in prior_scales:
                            decoder_config = copy.deepcopy(base_decoder_config)
                            decoder_config.lm_weight = lm_weight
                            decoder_config.prior_scale = prior_scale
                            search_name = it_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                            search_jobs, wers = search(
                                search_name,
                                forward_config={},
                                asr_model=quant_model,
                                decoder_module=quant_args.decoder,
                                decoder_args={"config": asdict(decoder_config)},
                                test_dataset_tuples=dev_dataset_tuples,
                                **default_returnn,
                            )
                            results.update(wers)
        pick_optimal_params_job = GetOptimalParametersAsVariableJob(
            parameters=tune_parameters, values=tune_values, mode="minimize"
        )
        pick_optimal_params_job.add_alias(training_name + f"/pick_best_dev")
        if eval_test:
            for key, tune_values in [("test", tune_values)]:
                decoder_config = copy.deepcopy(base_decoder_config)
                decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
                decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
                search_jobs, wers = search(
                    training_name,
                    forward_config={},
                    asr_model=asr_model,
                    decoder_module="ctc.decoder.flashlight_ctc_v1",
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples={key: test_dataset_tuples[key]},
                    **default_returnn,
                )
                results.update(wers)
        return results, pick_optimal_params_job

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
    )

    fe_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=False,
    )
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,  # Jingjing style
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=1536,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=1,
    )

    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
        + list(np.linspace(5e-4, 5e-5, 110))
        + list(np.linspace(5e-5, 1e-7, 30)),
        #############
        "batch_size": 360 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6"
    train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }
    results = {}
    training_name = prefix_name + "/" + network_module + "_384dim_sub4_24gbgpu_50eps_amp"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    res, _ = tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4], prior_scales=[0.0, 0.3, 0.5, 0.7, 1.0]
    )
    results.update(res)
    generate_report(results=results, exp_name=training_name)
    del results

    train_config = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                          + list(np.linspace(5e-4, 5e-5, 110))
                          + list(np.linspace(5e-5, 1e-7, 30)),
        #############
        "batch_size": 180 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
    }
    train_args = {
        "config": train_config,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }
    results = {}
    training_name = prefix_name + "/" + network_module + "_384dim_sub4_50eps"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    lm_scales = [2.0, 2.2, 2.4, 2.6, 2.8]
    prior_scales = [0.7, 0.9]
    res, _ = tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=lm_scales,
        prior_scales=prior_scales
    )
    results.update(res)
    asr_model_best4 = prepare_asr_model(
        training_name + "/best4", train_job, train_args, with_prior=True, datasets=train_data,
        get_best_averaged_checkpoint=(4, "dev_loss_ctc")
    )
    res, _ = tune_and_evaluate_helper(training_name + "/best4", asr_model_best4, default_decoder_config,
                                   lm_scales=lm_scales, prior_scales=prior_scales)
    results.update(res)
    asr_model_best = prepare_asr_model(
        training_name + "/best", train_job, train_args, with_prior=True, datasets=train_data,
        get_best_averaged_checkpoint=(1, "dev_loss_ctc")
    )
    res, _ = tune_and_evaluate_helper(training_name + "/best", asr_model_best, default_decoder_config,
                                   lm_scales=lm_scales, prior_scales=prior_scales)
    results.update(res)
    generate_report(results=results, exp_name=training_name)  # TODO current best with 7.083
    del results
    from ...pytorch_networks.ctc.conformer_1023.quant.baseline_quant_v1_cfg import QuantModelConfigV1
    num_iterations = 100
    # what if we give more information to the activation instead?
    for activation_bit in [8, 7, 6, 5, 4, 3, 2, 1]:
        for weight_bit in [8, 7, 6, 5, 4, 3, 2, 1]:
            results = {}
            model_config_quant_v1 = QuantModelConfigV1(
                weight_quant_dtype="qint8",
                weight_quant_method="per_tensor",
                activation_quant_dtype="qint8",
                activation_quant_method="per_tensor",
                dot_quant_dtype="qint8",
                dot_quant_method="per_tensor",
                Av_quant_dtype="qint8",
                Av_quant_method="per_tensor",
                moving_average=0.01,
                weight_bit_prec=weight_bit,
                activation_bit_prec=activation_bit,
                linear_quant_output=False,
            )
            quant_args = QuantArgs(
                sample_ls=[10] if weight_bit < 8 or activation_bit < 8 else [10, 100, 1000, 10000],
                quant_config_dict={"quant_config_dict": asdict(model_config_quant_v1)},
                decoder="ctc.decoder.flashlight_quant_stat_phoneme_ctc",
                num_iterations=num_iterations,
                datasets=train_data,
                network_module="ctc.conformer_1023.quant.baseline_quant_v1",
            )
            quant_str = f"_weight_{weight_bit}_act_{activation_bit}"
            asr_model = prepare_asr_model(
                training_name+quant_str,
                train_job,
                train_args,
                with_prior=True,
                datasets=train_data,
                get_specific_checkpoint=250,
            )
            res, _ = tune_and_evaluate_helper(  # only take best for now, since otherwise too many searches
                training_name, asr_model, default_decoder_config, lm_scales=[2.8],
                prior_scales=[0.7], quant_args=quant_args, quant_str=quant_str,
            )
            results.update(res)
            generate_report(results=results, exp_name=training_name + quant_str)
            del results

    num_iterations = 100
    for activation_bit in [8]:
        for weight_bit in [8, 7, 6, 5, 4, 3, 2, 1]:
            results = {}
            model_config_quant_v1 = QuantModelConfigV1(
                weight_quant_dtype="qint8",
                weight_quant_method="per_tensor",
                activation_quant_dtype="qint8",
                activation_quant_method="per_tensor",
                dot_quant_dtype="qint8",
                dot_quant_method="per_tensor",
                Av_quant_dtype="qint8",
                Av_quant_method="per_tensor",
                moving_average=0.01,
                weight_bit_prec=weight_bit,
                activation_bit_prec=activation_bit,
                linear_quant_output=True,
            )
            quant_args = QuantArgs(
                sample_ls=[10] if weight_bit < 8 or activation_bit < 8 else [10, 100, 1000, 10000],
                quant_config_dict={"quant_config_dict": asdict(model_config_quant_v1)},
                decoder="ctc.decoder.flashlight_quant_stat_phoneme_ctc",
                num_iterations=num_iterations,
                datasets=train_data,
                network_module="ctc.conformer_1023.quant.baseline_quant_v1",
            )
            quant_str = f"_weight_{weight_bit}_act_{activation_bit}_qlin"
            asr_model = prepare_asr_model(
                training_name+quant_str, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
            )
            res, _ = tune_and_evaluate_helper(  # only take best for now, since otherwise too many searches
                training_name, asr_model, default_decoder_config, lm_scales=[2.8],
                prior_scales=[0.7], quant_args=quant_args, quant_str=quant_str
            )
            results.update(res)
            generate_report(results=results, exp_name=training_name+quant_str)
            del results

    model_config_drop_03 = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=1536,
        att_weights_dropout=0.3,
        conv_dropout=0.3,
        ff_dropout=0.3,
        mhsa_dropout=0.3,
        conv_kernel_size=31,
        final_dropout=0.3,
        specauc_start_epoch=1,
    )

    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 210))
        + list(np.linspace(5e-4, 5e-5, 210))
        + list(np.linspace(5e-5, 1e-7, 30)),
        #############
        "batch_size": 360 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "torch_amp_options": {"dtype": "bfloat16"},
    }
    train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config_drop_03)},
        "debug": False,
        "use_speed_perturbation": True
    }
    results = {}
    training_name = prefix_name + "/" + network_module + "_384dim_sub4_24gbgpu_50eps_amp_longer"
    train_job = training(training_name, train_data, train_args, num_epochs=450, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=450
    )
    res, _ = tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8], prior_scales=[0.5, 0.7]
    )
    results.update(res)
    asr_model_best4 = prepare_asr_model(
        training_name + "/best4", train_job, train_args, with_prior=True, datasets=train_data, get_best_averaged_checkpoint=(4, "dev_loss_ctc")
    )
    res, _ = tune_and_evaluate_helper(training_name + "/best4", asr_model_best4, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.5, 0.7])
    results.update(res)
    asr_model_best = prepare_asr_model(
        training_name + "/best", train_job, train_args, with_prior=True, datasets=train_data,
        get_best_averaged_checkpoint=(1, "dev_loss_ctc")
    )
    res, _ = tune_and_evaluate_helper(training_name + "/best", asr_model_best, default_decoder_config,
                             lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.5, 0.7])
    results.update(res)
    generate_report(results=results, exp_name=training_name)
    del results

    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_conv_first"
    train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }
    results = {}
    training_name = prefix_name + "/" + network_module + "_384dim_sub4_24gbgpu_50eps_conv_first_amp"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    res, _ = tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        prior_scales=[0.5, 0.7]
    )
    results.update(res)
    generate_report(results=results, exp_name=training_name)
    del results
