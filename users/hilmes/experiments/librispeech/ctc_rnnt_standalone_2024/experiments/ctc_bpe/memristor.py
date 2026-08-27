from dataclasses import asdict
from typing import cast
import os


from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm, get_arpa_lm_config
from ...pipeline import training
from ...report import generate_report, build_qat_report_v2, multi_scale_cycle_report_format

from ..ctc_phon.tune_eval import eval_model
from functools import partial
from sisyphus import tk
import numpy as np
import copy
from ...data.bpe import build_bpe_training_datasets, get_bpe_bliss_lexicon

def get_observer_excludes(num_layers: int = 12):
    excludes = []
    checkpoint_prefix = "conformer.module_list"
    for layer in range(num_layers):
        excludes.extend(
            [
                f'{checkpoint_prefix}.{layer}.module_list.0.linear_ff.weight_quantizer',
                f'{checkpoint_prefix}.{layer}.module_list.0.linear_out.weight_quantizer',
                f'{checkpoint_prefix}.{layer}.module_list.0.lin_1_out_quant',
                f'{checkpoint_prefix}.{layer}.module_list.0.lin_1_in_quant',
                f'{checkpoint_prefix}.{layer}.module_list.0.lin_2_out_quant',
                f'{checkpoint_prefix}.{layer}.module_list.0.lin_2_in_quant',
                f'{checkpoint_prefix}.{layer}.module_list.1.pconv_2_out_quant',
                f'{checkpoint_prefix}.{layer}.module_list.1.dconv_1_out_quant',
                f'{checkpoint_prefix}.{layer}.module_list.1.pconv_2_in_quant',
                f'{checkpoint_prefix}.{layer}.module_list.1.pointwise_conv2.weight_quantizer',
                f'{checkpoint_prefix}.{layer}.module_list.1.depthwise_conv.weight_quantizer',
                f'{checkpoint_prefix}.{layer}.module_list.1.dconv_1_in_quant',
                f'{checkpoint_prefix}.{layer}.module_list.1.pointwise_conv1.weight_quantizer',
                f'{checkpoint_prefix}.{layer}.module_list.1.pconv_1_out_quant',
                f'{checkpoint_prefix}.{layer}.module_list.1.pconv_1_in_quant',
                f'{checkpoint_prefix}.{layer}.module_list.2.mhsa.out_proj_in_quant',
                f'{checkpoint_prefix}.{layer}.module_list.2.mhsa.linear_pos.weight_quantizer',
                f'{checkpoint_prefix}.{layer}.module_list.2.mhsa.qkv_proj.weight_quantizer',
                f'{checkpoint_prefix}.{layer}.module_list.2.mhsa.out_proj_out_quant',
                f'{checkpoint_prefix}.{layer}.module_list.2.mhsa.out_proj.weight_quantizer',
                f'{checkpoint_prefix}.{layer}.module_list.2.mhsa.learn_emb_out_quant',
                f'{checkpoint_prefix}.{layer}.module_list.2.mhsa.in_proj_out_quant',
                f'{checkpoint_prefix}.{layer}.module_list.2.mhsa.learn_emb_in_quant',
                f'{checkpoint_prefix}.{layer}.module_list.2.mhsa.in_proj_in_quant',
                f'{checkpoint_prefix}.{layer}.module_list.2.mhsa.q_quantizer',
                f'{checkpoint_prefix}.{layer}.module_list.2.mhsa.k_quantizer',
                f'{checkpoint_prefix}.{layer}.module_list.3.lin_1_in_quant',
                f'{checkpoint_prefix}.{layer}.module_list.3.linear_ff.weight_quantizer',
                f'{checkpoint_prefix}.{layer}.module_list.3.lin_1_out_quant',
                f'{checkpoint_prefix}.{layer}.module_list.3.lin_2_out_quant',
                f'{checkpoint_prefix}.{layer}.module_list.3.linear_out.weight_quantizer',
                f'{checkpoint_prefix}.{layer}.module_list.3.lin_2_in_quant',
            ]
        )
    return excludes

def run_memristor_cycle_eval(
    train_job, train_data, train_config, model_config,
    recog_name_prefix, rasr_config, dev_dataset_tuples,
    prior_scales, lm_scales, batch_size, max_runs, report_dict,
    prior_network_module, recog_network_module,
    greedy_config=None,
    recog_model_config_class=None,
    final_name=None,
    recog_dac_settings=None,
    posenc_dac_settings=None,
    search_gpu=11,
    search_gpu_type=None,
    prune_weights=False,
    run_rasr_multi=False,
    num_search_workers=8,
    energy_recog_network_module=None,
    energy_cycles=None,
    energy_lm_scale=None,
    energy_prior_scale=None,
    energy_measure_args=None,
    energy_dataset_tuples=None,
    energy_report_dict=None,
    eval_recog_network_module=None,
    extra_forward_config=None,
    fast_inference=False,
    run_search_on_hpc=False,
    noise_free_inference=False,
    no_outq=False,
):
    """
    eval_recog_network_module: overrides the network module for the recognition jobs only
        (e.g. "ctc.qat_0711.memristor_v11_no_outq"); "_mem_inited" is appended as usual.
        The conversion and prior jobs keep recog_network_module/prior_network_module and
        therefore stay cached, so this is for variants that only change recognition
        behavior on the same converted checkpoint.
    fast_inference: enables SynaptogenML's opt-in fast inference path by routing the
        recognition forward through a "_fast" decoder module (e.g.
        "ctc.decoder.rasr_ctc_v1_batched_fast") instead of a "_fast" network-module file
        copy. Since recog_network_module/prior_network_module/net_args are unaffected,
        the conversion and prior jobs are automatically shared with a non-fast call using
        the same arguments -- no eval_recog_network_module needed for this. The SynaptogenML
        pin is passed explicitly (new_v3, device-adaptive fast path), so recog_name_prefix
        needs no "_newsynap_*" marker; prefix-based routing in pipeline.py still overrides
        the pin for the legacy tagged runs (e.g. seeded stays v2, noisefix/noisefree v4).
    run_search_on_hpc: holds the recognition search jobs locally and flags them
        move_to_hpc, so `sis_itc_helper.py <config> sync` transfers and submits them
        on the CLAIX cluster (walltime = rqmt time; keep it < 12h for c25g).
    noise_free_inference: disables the memristor readout noise (Johnson + shot) in
        recognition via the "_fast_nonoise" decoder wrapper, which also enables the
        fast path (so fast_inference need not be set additionally). Requires a
        SynaptogenML pin with the set_readout_noise API (v4+); used to quantify the
        WER contribution of readout noise.
    energy_recog_network_module: network module to use for the energy recognition (the
        plain module, e.g. "ctc.qat_0711.claude.memristor_v16_dynmic_prec" -- fast path
        needs no dedicated per-cell-capture variant); when set, cycles listed in
        energy_cycles additionally run one recognition at (energy_lm_scale,
        energy_prior_scale) with the fast energy decoder ("new_v3" pin, column currents +
        analytic row sums), reusing the cached conversion/prior jobs, and emit
        energy_report.pkl / energy_summary.json as job outputs. Validated against the
        per-cell-capture reference (eager "energy" pin) on two configs
        (taper8664_qk8v4, taper8664_rotb_qk8v4): column-level scalars agreed to <0.03%,
        per-cell analytic estimate to ~0.34%, device counts exact -- fast path is used
        exclusively now, no eager fallback.
    energy_measure_args: COMPLETE kwargs dict for claude.energy_measure.EnergyMeasureConfig
        (no defaults on purpose - validated here at graph time and hashed verbatim, so the
        measurement behavior is always explicit in the graph)
    energy_dataset_tuples: measurement datasets, defaults to dev_dataset_tuples
    energy_report_dict: if given, filled with {<recog_name>_<cycle>_energy:
        tk.Path to energy_summary.json} for the report's energy table
    no_outq: recognise with the activation output quantizers (*_out_quant) replaced by
        nn.Identity, like the input quantizers already are in the _mem_inited modules. Routes
        the recognition through the generic shim in pytorch_networks/ctc/qat_0711/no_outq/
        (one ~12 line file per model version, see no_outq/_strip.py) by setting
        eval_recog_network_module, so conversion and prior jobs are shared with the
        corresponding plain run. "_no_outq" is appended to recog_name_prefix/final_name,
        otherwise the new jobs would land on the plain run's alias paths and report keys.
        Improves the hardware WER (v11 w4/a8 512dim dev-other, lm 0.8 / prior 0.5:
        6.48-6.65 over 5 cycles vs 6.80-6.91 with out-quants).
    """
    from torch_memristor.memristor_modules import DacAdcHardwareSettings
    if no_outq:
        if eval_recog_network_module is None:
            eval_recog_network_module = "ctc.qat_0711.no_outq." + recog_network_module.split(".")[-1]
        recog_name_prefix = recog_name_prefix + "_no_outq"
        if final_name is not None:
            final_name = final_name + "_no_outq"
    if recog_dac_settings is None:
        recog_dac_settings = DacAdcHardwareSettings(
            input_bits=8,
            output_precision_bits=4,
            output_range_bits=4,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )
    if posenc_dac_settings is None:
        posenc_dac_settings = DacAdcHardwareSettings(
            input_bits=8,
            output_precision_bits=1,
            output_range_bits=7,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )
    prior_args = {
        "config": train_config,
        "network_module": prior_network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }
    if noise_free_inference:
        decoder_suffix = "_fast_nonoise"
    else:
        decoder_suffix = "_fast" if fast_inference else ""
    # nonoise needs the set_readout_noise API (v4+), the fast path set_fast_inference
    # (v3+, device-adaptive); a "_newsynap_*" marker in recog_name_prefix still wins
    # in pipeline.py's routing, so legacy tagged runs keep their pins/hashes.
    import_memristor = "new_v4" if noise_free_inference else ("new_v3" if fast_inference else True)
    if "_newsynap_progfast_" in recog_name_prefix:
        # programming-speedup A/B: the v5 pin must reach the CONVERSION and
        # prior configs too (their import_memristor comes from this value, not
        # from pipeline.py's prefix routing, which only covers the searches).
        # With plain new_v3 the conversion config hashes identically to the
        # baseline and re-links to it -- the fast programming would never run.
        import_memristor = "new_v5"
    res_fixed, res_greedy = {}, {}
    for num_cycles in range(1, max_runs + 1):
        if recog_model_config_class is not None:
            d = dict(model_config.__dict__)
            d.pop('pos_enc_converter_hardware_settings', None)
            model_config_recog = recog_model_config_class(**d, pos_enc_converter_hardware_settings=None)
        else:
            model_config_recog = copy.deepcopy(model_config)
        model_config_recog.converter_hardware_settings = recog_dac_settings
        model_config_recog.num_cycles = num_cycles
        model_config_recog.pos_enc_converter_hardware_settings = posenc_dac_settings
        train_args_recog = {
            "config": train_config,
            "network_module": recog_network_module,
            "net_args": {"model_config_dict": asdict(model_config_recog)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }
        recog_name = recog_name_prefix + f"/cycle_{num_cycles // 11}"
        res_fixed = eval_model(
            training_name=recog_name + f"_{num_cycles}",
            train_job=train_job,
            train_args=train_args_recog,
            train_data=train_data,
            decoder_config=rasr_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=res_fixed,
            decoder_module=("ctc.decoder.rasr_ctc_v1_batched_multi" if run_rasr_multi else "ctc.decoder.rasr_ctc_v1_batched") + decoder_suffix,
            prior_scales=prior_scales,
            lm_scales=lm_scales,
            use_gpu=True,
            import_memristor=import_memristor,
            extra_forward_config={"batch_size": batch_size, **(extra_forward_config or {})},
            run_best_4=False,
            run_best=False,
            prior_args=prior_args,
            run_search_on_hpc=run_search_on_hpc,
            run_rasr=True,
            split_mem_init=True,
            search_gpu=search_gpu,
            search_gpu_type=search_gpu_type,
            prune_weights=prune_weights,
            run_rasr_multi=run_rasr_multi,
            num_search_workers=num_search_workers,
            recog_network_module=eval_recog_network_module,
        )
        if greedy_config is not None:
            res_greedy = eval_model(
                training_name=recog_name + f"_{num_cycles}",
                train_job=train_job,
                train_args=train_args_recog,
                train_data=train_data,
                decoder_config=greedy_config,
                dev_dataset_tuples=dev_dataset_tuples,
                result_dict=res_greedy,
                decoder_module="ctc.decoder.greedy_bpe_ctc_quant_v1" + decoder_suffix,
                prior_scales=[0.0],
                lm_scales=[0.0],
                use_gpu=True,
                import_memristor=import_memristor,
                extra_forward_config={"batch_size": batch_size, **(extra_forward_config or {})},
                run_best_4=False,
                run_best=False,
                prior_args=None,
                with_prior=False,
                run_search_on_hpc=run_search_on_hpc,
                run_rasr=False,
                split_mem_init=True,
                search_gpu=search_gpu,
                search_gpu_type=search_gpu_type,
                prune_weights=prune_weights,
                recog_network_module=eval_recog_network_module,
            )
        if energy_recog_network_module is not None and num_cycles in (energy_cycles or []):
            # one extra recognition with the energy model + decoder at fixed scales
            # (currents do not depend on lm/prior); conversion and prior jobs are shared
            # with the eval above via identical train_args_recog/prior_args
            assert energy_lm_scale is not None and energy_prior_scale is not None
            from ...pytorch_networks.ctc.qat_0711.claude.energy_measure import EnergyMeasureConfig

            # graph-time validation: the dict must be complete and valid, and is hashed as-is
            energy_args = asdict(EnergyMeasureConfig(**energy_measure_args))
            energy_outputs = {}
            eval_model(
                training_name=recog_name + f"_{num_cycles}_energy",
                train_job=train_job,
                train_args=train_args_recog,
                train_data=train_data,
                decoder_config=rasr_config,
                dev_dataset_tuples=energy_dataset_tuples or dev_dataset_tuples,
                result_dict={},
                decoder_module="ctc.decoder.rasr_ctc_v1_batched_fast_energy",
                prior_scales=[energy_prior_scale],
                lm_scales=[energy_lm_scale],
                use_gpu=True,
                import_memristor="new_v3",  # fast-path pin; column currents + analytic row sums
                extra_forward_config={"batch_size": batch_size},
                run_best_4=False,
                run_best=False,
                prior_args=prior_args,
                run_search_on_hpc=run_search_on_hpc,
                run_rasr=True,
                split_mem_init=True,
                search_gpu=search_gpu,
                prune_weights=prune_weights,
                recog_network_module=energy_recog_network_module,
                extra_decoder_args={"energy": energy_args},
                additional_search_output_files=["energy_report.pkl", "energy_summary.json"],
                search_output_collector=energy_outputs,
            )
            if energy_report_dict is not None and "energy_summary.json" in energy_outputs:
                energy_report_dict[recog_name + f"_{num_cycles}_energy"] = energy_outputs["energy_summary.json"]
        if num_cycles == max_runs:
            report_name = final_name if final_name is not None else recog_name
            if run_rasr_multi:
                # per-(lm, prior) mean/std/min/max across cycles, sorted best-WER-first (dev-other)
                generate_report(
                    results=res_fixed, exp_name=report_name,
                    report_template=multi_scale_cycle_report_format,
                )
            else:
                generate_report(results=res_fixed, exp_name=report_name)
            report_dict[report_name] = copy.deepcopy(res_fixed)
            if greedy_config is not None:
                # generate_report(results=res_greedy, exp_name=report_name + "_greedy")
                report_dict[report_name + "_greedy"] = copy.deepcopy(res_greedy)


def run_non_memristor_eval(
    training_name, train_job, train_args, train_data,
    rasr_config, greedy_config, dev_dataset_tuples,
    rasr_prior_scales, rasr_lm_scales, report_dict,
    prune_weights: bool = False,
):
    results, best_params_job = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        decoder_config=rasr_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict={},
        decoder_module="ctc.decoder.rasr_ctc_v1",
        prior_scales=rasr_prior_scales,
        lm_scales=rasr_lm_scales,
        import_memristor=True,
        get_best_params=True,
        run_rasr=True,
        run_best_4=False,
        run_best=False,
        prune_weights=prune_weights,
    )
    generate_report(results=results, exp_name=training_name + "/non_memristor")
    report_dict[training_name] = results

    results, _ = eval_model(
        training_name=training_name + "/greedy",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        decoder_config=greedy_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict={},
        decoder_module="ctc.decoder.greedy_bpe_ctc_quant_v1",
        prior_scales=[0.0],
        lm_scales=[0.0],
        import_memristor=True,
        get_best_params=True,
        run_rasr=False,
        run_best_4=False,
        run_best=False,
        with_prior=False,
        prune_weights=prune_weights,
    )
     # generate_report(results=results, exp_name=training_name + "/greedy/non_memristor")
    report_dict[training_name + "_greedy"] = results
    return best_params_job


def bpe_ls960_1225_memristor():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/bpe_ls960_memristor"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe128 = build_bpe_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=128,
        settings=train_settings,
        use_postfix=False,
    )
    label_datastream_bpe128 = cast(LabelDatastream, train_data_bpe128.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe128.vocab_size

    dev_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.rasr_ctc_v1 import DecoderConfig as RasrDecoderConfig
    from ...rasr_recog_config import get_tree_timesync_recog_config, get_no_op_label_scorer_config

    recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
        lexicon_file=get_bpe_bliss_lexicon(bpe_size=128, add_blank=True, librispeech_key="train-other-960"),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=vocab_size_without_blank,
        max_beam_size=2048,
        score_threshold=18.0,
        logfile_suffix="recog",
        lm_config=get_arpa_lm_config("4gram", lexicon_file=get_bpe_bliss_lexicon(bpe_size=128, add_blank=True, librispeech_key="train-other-960"), scale=0.0),
    )

    as_training_rasr_config = RasrDecoderConfig(
        rasr_config_file=recog_rasr_config,
        rasr_post_config=recog_rasr_post_config,
        blank_log_penalty=None,
        prior_scale=0.0,  # this will be overwritten internally
        prior_file=None,
        turn_off_quant="leave_as_is",  # this does not have memristor
    )
    rasr_config_memristor = copy.deepcopy(as_training_rasr_config)
    rasr_config_memristor.turn_off_quant = False

    rasr_prior_scales = [0.2, 0.3, 0.4, 0.5]
    rasr_lm_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_quant_v1 import DecoderConfig as GreedyDecoderConfig
    as_training_greedy_decoder_config = GreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe128.vocab,
        turn_off_quant="leave_as_is",
    )
    greedy_decoder_memristor = copy.deepcopy(as_training_greedy_decoder_config)
    greedy_decoder_memristor.turn_off_quant = False

    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config

    FINETUNE_MODELS = {}

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

    specaug_config_full = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # Normal Style
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
        out_features=512,
        activation=None,
    )

    from ...pytorch_networks.ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
        ModelConfig as RelPosModelConfigV1,
        ConformerPosEmbConfig,
    )

    pos_emb_cfg = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    memristor_report = {}
    baseline_network_module = "ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"

    model_config = RelPosModelConfigV1(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=11,
        pos_emb_config=pos_emb_cfg,
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=None,
        aux_ctc_loss_scales=None,
        dropout_broadcast_axes=None,
        mhsa_with_bias=True,
    )

    for epochs in [500, 1000, 2000, 2500, 3000]:
        train_config_24gbgpu = {
            "optimizer": {
                "class": "radam",
                "epsilon": 1e-12,
                "weight_decay": 1e-2,
                "decoupled_weight_decay": True,
            },
            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs // 2 - 20)))
                              + list(np.linspace(5e-4, 5e-5, (epochs // 2 - 20)))
                              + list(np.linspace(5e-5, 1e-7, 40)),
            #############
            "batch_size": 360 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "gradient_clip_norm": 1.0,
        }
        train_args_base = {
            "config": train_config_24gbgpu,
            "network_module": baseline_network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "use_speed_perturbation": True,
            "post_config": {"num_workers_per_gpu": 8}
        }
        name = f".baseline_512dim_sub4_48gbgpu_{epochs//10}eps_radam_bs360_sp"
        training_name = prefix_name + "/" + baseline_network_module + name
        train_job = training(training_name, train_data_bpe128, train_args_base, num_epochs=epochs, **default_returnn)
        FINETUNE_MODELS[training_name] = train_job.out_checkpoints[epochs]
        if not os.path.exists(
            f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
            train_job.hold()
            train_job.move_to_hpc = True
        results = {}
        results, best_params_job = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args_base,
            train_data=train_data_bpe128,
            decoder_config=as_training_rasr_config,
            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
            result_dict=results,
            decoder_module="ctc.decoder.rasr_ctc_v1",
            prior_scales=rasr_prior_scales,
            lm_scales=rasr_lm_scales,
            import_memristor=True,
            get_best_params=True,
            run_rasr=True,
            run_best_4=False,
            run_best=False,
        )
        generate_report(results=results, exp_name=training_name)
        memristor_report[training_name] = results

        # --- fast end-to-end validation of the multi-scale search (non-memristor) ---
        # The baseline conformer forward is cheap (no memristor simulation), so this
        # exercises the multi decoder + parallel ctx.Process search + report in minutes.
        # Tiny grid keeps search RTF minimal. Reuses the proven baseline train_job /
        # as_training_rasr_config, only swapping in the multi decoder.
        if epochs == 1000:
            multi_results = {}
            # eval_model returns just result_dict unless get_best_params=True (tune_eval.py:197-200)
            multi_results = eval_model(
                training_name=training_name + "_multi_sweep_test",
                train_job=train_job,
                train_args=train_args_base,
                train_data=train_data_bpe128,
                decoder_config=copy.deepcopy(as_training_rasr_config),
                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                result_dict=multi_results,
                decoder_module="ctc.decoder.rasr_ctc_v1_batched_multi",
                prior_scales=[0.0, 0.3],
                lm_scales=[0.7, 0.9],
                use_gpu=False,   # baseline forward is cheap; run on CPU to dodge the GPU
                                 # quota + flaky-GPU-node crashes (search phase is CPU anyway)
                import_memristor=True,   # mirrors the proven baseline recog above
                run_rasr=True,
                run_rasr_multi=True,
                num_search_workers=2,
                run_best_4=False,
                run_best=False,
            )
            generate_report(
                results=multi_results,
                exp_name=training_name + "_multi_sweep_test",
                report_template=multi_scale_cycle_report_format,
            )
            memristor_report[training_name + "_multi_sweep_test"] = multi_results

        results = {}
        results, _ = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args_base,
            train_data=train_data_bpe128,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_quant_v1",
            prior_scales=[0.0],
            lm_scales=[0.0],
            import_memristor=True,
            get_best_params=True,
            run_rasr=False,
            run_best_4=False,
            run_best=False,
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        memristor_report[training_name + '_greedy'] = results

    def _make_frontend_config(out_features):
        return VGG4LayerActFrontendV1Config_mod(
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
            out_features=out_features,
            activation=None,
        )

    def _make_model_config_kwargs(frontend_config, conformer_size, activation_bit, converter_hardware_settings):
        return dict(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            specaug_config=specaug_config_full,
            label_target_size=vocab_size_without_blank,
            conformer_size=conformer_size,
            num_layers=12,
            num_heads=8,
            ff_dim=conformer_size * 4,
            att_weights_dropout=0.1,
            conv_dropout=0.1,
            ff_dropout=0.1,
            mhsa_dropout=0.1,
            conv_kernel_size=31,
            final_dropout=0.1,
            specauc_start_epoch=11,
            weight_quant_dtype="qint8",
            weight_quant_method="per_tensor_symmetric",
            activation_quant_dtype="qint8",
            activation_quant_method="per_tensor_symmetric",
            dot_quant_dtype="qint8",
            dot_quant_method="per_tensor_symmetric",
            Av_quant_dtype="qint8",
            Av_quant_method="per_tensor_symmetric",
            moving_average=None,
            activation_bit_prec=activation_bit,
            quantize_output=False,
            converter_hardware_settings=converter_hardware_settings,
            quant_in_linear=True,
            num_cycles=0,
            correction_settings=None,
            weight_noise_func=None,
            weight_noise_values=None,
            weight_noise_start_epoch=None,
            pos_emb_config=pos_emb_cfg,
            module_list=["ff", "conv", "mhsa", "ff"],
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=None,
            aux_ctc_loss_scales=None,
            dropout_broadcast_axes=None,
        )

    network_module_mem_v7 = "ctc.qat_0711.memristor_v7"
    network_module_mem_v10 = "ctc.qat_0711.memristor_v10"
    network_module_mem_v11 = "ctc.qat_0711.memristor_v11"
    network_module_mem_v12 = "ctc.qat_0711.memristor_v12"
    network_module_mem_v13 = "ctc.qat_0711.memristor_v13"
    network_module_mem_v14 = "ctc.qat_0711.memristor_v14"

    from ...pytorch_networks.ctc.qat_0711.memristor_v7_cfg import QuantModelTrainConfigV7 as MemristorModelTrainConfigV7
    from ...pytorch_networks.ctc.qat_0711.memristor_v8_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV8

    from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings

    for activation_bit in [8]:
        for epochs in [1000]:
            for dim in [512, 1024]:
                for weight_bit in [4, 8]:
                    frontend_config_dim = _make_frontend_config(dim)
                    prior_train_dac_settings = DacAdcHardwareSettings(
                        input_bits=0,
                        output_precision_bits=0,
                        output_range_bits=0,
                        hardware_input_vmax=0.6,
                        hardware_output_current_scaling=8020.0,
                    )
                    model_config = MemristorModelTrainConfigV8(
                        **_make_model_config_kwargs(frontend_config_dim, dim, activation_bit, prior_train_dac_settings),
                        weight_bit_prec=weight_bit,
                    )
                    seeds = 3
                    if dim not in [512]:
                        seeds = 2
                        if dim not in [1024]:
                            seeds = 1
                    elif epochs not in [1000]:
                        seeds = 2
                    if weight_bit in [6, 7]:
                        seeds = 1
                    for seed in range(seeds):
                        train_config_24gbgpu = {
                            "optimizer": {
                                "class": "radam",
                                "epsilon": 1e-12,
                                "weight_decay": 1e-2,
                                "decoupled_weight_decay": True,
                            },
                            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs // 2 - 20)))
                                              + list(np.linspace(5e-4, 5e-5, (epochs // 2 - 20)))
                                              + list(np.linspace(5e-5, 1e-7, 40)),
                            #############
                            "batch_size": 360 * 16000,
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1,
                            "gradient_clip_norm": 1.0,
                            "seed": seed,
                            "torch_amp_options": {"dtype": "bfloat16"},
                        }
                        train_args = {
                            "config": train_config_24gbgpu,
                            "network_module": network_module_mem_v10,
                            "net_args": {"model_config_dict": asdict(model_config)},
                            "debug": False,
                            "post_config": {"num_workers_per_gpu": 8},
                            "use_speed_perturbation": True,
                        }
                        training_name = prefix_name + "/" + network_module_mem_v10 + f"_{epochs//10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}"
                        train_job = training(training_name, train_data_bpe128, train_args, num_epochs=epochs, **default_returnn)
                        FINETUNE_MODELS[training_name] = train_job.out_checkpoints[epochs]
                        if not os.path.exists(
                            f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                            train_job.rqmt['cpu'] = 8
                            train_job.hold()
                            train_job.move_to_hpc = True

                        _ = run_non_memristor_eval(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args,
                            train_data=train_data_bpe128,
                            rasr_config=as_training_rasr_config,
                            greedy_config=as_training_greedy_decoder_config,
                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            rasr_prior_scales=rasr_prior_scales,
                            rasr_lm_scales=rasr_lm_scales,
                            report_dict=memristor_report,
                        )

                        max_runs = 10 if dim <= 512 else 3
                        from ...pytorch_networks.ctc.qat_0711.memristor_v11_cfg import \
                            QuantModelTrainConfigV11 as MemristorModelTrainConfigV11
                        run_memristor_cycle_eval(
                            train_job=train_job,
                            train_data=train_data_bpe128,
                            train_config=train_config_24gbgpu,
                            model_config=model_config,
                            recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}",
                            rasr_config=rasr_config_memristor,
                            greedy_config=greedy_decoder_memristor,
                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            prior_scales=[0.5],
                            lm_scales=[0.8],
                            batch_size=3500000 if weight_bit not in [8] else 2500000,
                            max_runs=max_runs,
                            report_dict=memristor_report,
                            prior_network_module=network_module_mem_v10,
                            recog_network_module=network_module_mem_v11,
                            recog_model_config_class=MemristorModelTrainConfigV11,
                            search_gpu=11 if dim < 1024 else 24,
                            # validated fast path under the original alias
                            # (replaces the former _newsynap_fast_ mirror runs)
                            fast_inference=True,
                        )
                        if dim == 512 and seed == 0:
                            seeded_batch_size = 3500000 if weight_bit not in [8] else 2500000
                            # Shot-noise-constant A/B on top of the seeded setup:
                            # both arms use the SAME seed-42 _seeded_fast model file
                            # and share the cached conversion/prior jobs, but route
                            # to the v4 SynaptogenML pin (corrected e/BW readout
                            # constants, "_newsynap_noisefix_"/"_newsynap_noisefree_"
                            # matches in pipeline.py). Baseline arm = the finished
                            # seeded_fast results (v2, old constants).
                            #   noisefix:  corrected constants, noise on
                            #   noisefree: corrected constants + readout noise OFF
                            #              (via the _fast_nonoise decoder wrapper)
                            # rasr-only: greedy adds nothing to this question.
                            for tag, arm_noise_free in [
                                ("noisefix_seeded", False),
                                ("noisefree_seeded", True),
                            ]:
                                run_memristor_cycle_eval(
                                    train_job=train_job,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_24gbgpu,
                                    model_config=model_config,
                                    recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_newsynap_{tag}_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}",
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=None,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[0.5],
                                    lm_scales=[0.8],
                                    batch_size=seeded_batch_size,
                                    max_runs=3,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v10,
                                    recog_network_module=network_module_mem_v11,
                                    recog_model_config_class=MemristorModelTrainConfigV11,
                                    search_gpu=24,
                                    eval_recog_network_module="ctc.qat_0711.memristor_v11_seeded_fast",
                                    noise_free_inference=arm_noise_free,
                                )
                        if weight_bit == 8 and seed == 2 and dim == 512:
                            for lm, prior in [(0.8, 0.3), (0.8, 0.4), (0.8, 0.5), (0.8, 0.6), (0.8, 0.7), (0.9, 0.5), (1.0, 0.5), (0.7, 0.5)]:
                                run_memristor_cycle_eval(
                                    train_job=train_job,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_24gbgpu,
                                    model_config=model_config,
                                    recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_fixed_lm{lm}_prior{prior}_seed_{seed}",
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=greedy_decoder_memristor,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[prior],
                                    lm_scales=[lm],
                                    batch_size=3500000 if weight_bit not in [8] else 2500000,
                                    max_runs=max_runs,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v10,
                                    recog_network_module=network_module_mem_v11,
                                    recog_model_config_class=MemristorModelTrainConfigV11,
                                    search_gpu=11,
                                    fast_inference=True,
                                )

                            # single multi-scale sweep: one forward per (cycle, dataset), all
                            # (lm, prior) combinations applied to the same posteriors
                            run_memristor_cycle_eval(
                                train_job=train_job,
                                train_data=train_data_bpe128,
                                train_config=train_config_24gbgpu,
                                model_config=model_config,
                                recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_fixed_multi_sweep_seed_{seed}",
                                rasr_config=rasr_config_memristor,
                                greedy_config=greedy_decoder_memristor,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                prior_scales=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                lm_scales=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
                                batch_size=3500000,
                                max_runs=5,
                                report_dict=memristor_report,
                                prior_network_module=network_module_mem_v10,
                                recog_network_module=network_module_mem_v11,
                                recog_model_config_class=MemristorModelTrainConfigV11,
                                search_gpu=48,
                                run_rasr_multi=True,
                                fast_inference=True,
                            )

                            for prec, ran in [(4, 8), (8, 8)]:
                                recog_dac_settings_test = DacAdcHardwareSettings(
                                    input_bits=8,
                                    output_precision_bits=prec,
                                    output_range_bits=ran,
                                    hardware_input_vmax=0.6,
                                    hardware_output_current_scaling=8020.0,
                                )
                                run_memristor_cycle_eval(
                                    train_job=train_job,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_24gbgpu,
                                    model_config=model_config,
                                    recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_adc_prec{prec}_range{ran}_seed_{seed}",
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=greedy_decoder_memristor,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[prior],
                                    lm_scales=[lm],
                                    batch_size=3500000 if weight_bit not in [8] else 2500000,
                                    max_runs=max_runs,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v10,
                                    recog_network_module=network_module_mem_v11,
                                    recog_model_config_class=MemristorModelTrainConfigV11,
                                    search_gpu=11,
                                    recog_dac_settings=recog_dac_settings_test,
                                    fast_inference=True,
                                )


                        model_config_ideal = copy.deepcopy(model_config)
                        train_dac_settings_ideal = DacAdcHardwareSettings(
                            input_bits=8,
                            output_precision_bits=4,
                            output_range_bits=4,
                            hardware_input_vmax=0.6,
                            hardware_output_current_scaling=5476.0,
                        )

                        posenc_dac_settings_ideal = DacAdcHardwareSettings(
                            input_bits=8,
                            output_precision_bits=1,
                            output_range_bits=7,
                            hardware_input_vmax=0.6,
                            hardware_output_current_scaling=5476.0,
                        )
                        from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings
                        ideal = CycleCorrectionSettings(
                            num_cycles=None,
                            test_input_value=None,
                            relative_deviation=None,
                            ideal_programming=True
                        )
                        model_config_ideal.correction_settings = ideal

                        if dim == 512:
                            run_memristor_cycle_eval(
                                train_job=train_job,
                                train_data=train_data_bpe128,
                                train_config=train_config_24gbgpu,
                                model_config=model_config_ideal,
                                recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_ideal_seed_{seed}",
                                rasr_config=rasr_config_memristor,
                                greedy_config=greedy_decoder_memristor,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                prior_scales=[0.5],
                                lm_scales=[0.8],
                                batch_size=3500000 if weight_bit not in [8] else 2500000,
                                max_runs=3,
                                report_dict=memristor_report,
                                prior_network_module=network_module_mem_v10,
                                recog_network_module=network_module_mem_v11,
                                recog_model_config_class=MemristorModelTrainConfigV11,
                                recog_dac_settings=train_dac_settings_ideal,
                                posenc_dac_settings=posenc_dac_settings_ideal,
                                fast_inference=True,
                            )
                            if seed == 0 and weight_bit in [4, 8]:
                                # Shot-noise-constant A/B under IDEAL programming, i.e. the
                                # same v2->v4 question as the _newsynap_noisefix_/_noisefree_
                                # seeded arms above, but with the conductance states set
                                # exactly (correction_settings.ideal_programming=True). With
                                # programming noise removed, the inference-time readout noise
                                # (Johnson + shot in MemristorArray.compute_noise) is the ONLY
                                # remaining stochastic term, so this isolates it cleanly:
                                #   baseline:  the ideal run above (v3 pin, fast path, old
                                #              constants e=exp(1)/BW=1e-8 -> Johnson zeroed,
                                #              shot sigma ~40x)
                                #   noisefix:  corrected constants, readout noise on
                                #   noisefree: corrected constants + readout noise OFF
                                # Question: how much of the residual ideal->QAT gap (w8 5.8 vs
                                # 5.3, w4 6.1 vs 5.8) is the readout-noise bug rather than
                                # ADC/DAC quantization. Both arms take the v4 pin via the
                                # "_newsynap_noisefix_"/"_newsynap_noisefree_" match in
                                # pipeline.py and share the ideal conversion/prior jobs
                                # (same model_config_ideal + dac settings). All arms run
                                # on the fast path. rasr-only, no greedy.
                                # noisefree is fully deterministic under ideal programming
                                # (no RNG left anywhere), so one cycle instead of three.
                                for tag, arm_noise_free, arm_runs in [
                                    ("noisefix", False, 3),
                                    ("noisefree", True, 1),
                                ]:
                                    run_memristor_cycle_eval(
                                        train_job=train_job,
                                        train_data=train_data_bpe128,
                                        train_config=train_config_24gbgpu,
                                        model_config=model_config_ideal,
                                        recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_newsynap_{tag}_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_ideal_seed_{seed}",
                                        rasr_config=rasr_config_memristor,
                                        greedy_config=None,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        prior_scales=[0.5],
                                        lm_scales=[0.8],
                                        batch_size=3500000 if weight_bit not in [8] else 2500000,
                                        max_runs=arm_runs,
                                        report_dict=memristor_report,
                                        prior_network_module=network_module_mem_v10,
                                        recog_network_module=network_module_mem_v11,
                                        recog_model_config_class=MemristorModelTrainConfigV11,
                                        recog_dac_settings=train_dac_settings_ideal,
                                        posenc_dac_settings=posenc_dac_settings_ideal,
                                        search_gpu=24,
                                        eval_recog_network_module="ctc.qat_0711.memristor_v11_fast",
                                        noise_free_inference=arm_noise_free,
                                    )
                            if weight_bit == 4 and seed == 0:
                                # no_outq: recognition with the activation output quantizers
                                # (*_out_quant) set to nn.Identity as well; shares the
                                # conversion/prior jobs with the ideal run above
                                run_memristor_cycle_eval(
                                    train_job=train_job,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_24gbgpu,
                                    model_config=model_config,
                                    recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_no_outq_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_ideal_seed_{seed}",
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=greedy_decoder_memristor,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[0.5],
                                    lm_scales=[0.8],
                                    batch_size=3500000,
                                    max_runs=5,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v10,
                                    recog_network_module=network_module_mem_v11,
                                    recog_model_config_class=MemristorModelTrainConfigV11,
                                    eval_recog_network_module="ctc.qat_0711.memristor_v11_no_outq",
                                    fast_inference=True,
                                    search_gpu=48,
                                )
                                # correctness gate for the generic no_outq shim
                                # (ctc.qat_0711.no_outq.memristor_v11 + no_outq/_strip.py):
                                # same arguments as the hand-copied run above, one cycle only.
                                # Must reproduce its cycle 1: 6.48 (lm 0.8 / prior 0.5) and
                                # 11.59 greedy on dev-other. Conversion/prior jobs are cached.
                                run_memristor_cycle_eval(
                                    train_job=train_job,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_24gbgpu,
                                    model_config=model_config,
                                    recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_ideal_seed_{seed}",
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=greedy_decoder_memristor,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[0.5],
                                    lm_scales=[0.8],
                                    batch_size=3500000,
                                    max_runs=1,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v10,
                                    recog_network_module=network_module_mem_v11,
                                    recog_model_config_class=MemristorModelTrainConfigV11,
                                    fast_inference=True,
                                    search_gpu=48,
                                    no_outq=True,
                                )



    from ...pytorch_networks.ctc.qat_0711.memristor_v12_cfg import QuantModelTrainConfigV12 as MemristorModelTrainConfigV12
    weight_bit = 4
    activation_bit = 8
    dim = 512
    epochs = 1000
    frontend_config_dim = _make_frontend_config(dim)
    prior_train_dac_settings = DacAdcHardwareSettings(
        input_bits=0,
        output_precision_bits=0,
        output_range_bits=0,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )

    model_config = MemristorModelTrainConfigV12(
        **_make_model_config_kwargs(frontend_config_dim, dim, activation_bit, prior_train_dac_settings),
        weight_bit_prec=[4] * 12,
        pos_enc_converter_hardware_settings=prior_train_dac_settings,
    )
    for seed in range(3):
        train_config_24gbgpu = {
            "optimizer": {
                "class": "radam",
                "epsilon": 1e-12,
                "weight_decay": 1e-2,
                "decoupled_weight_decay": True,
            },
            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs // 2 - 20)))
                              + list(np.linspace(5e-4, 5e-5, (epochs // 2 - 20)))
                              + list(np.linspace(5e-5, 1e-7, 40)),
            #############
            "batch_size": 360 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "gradient_clip_norm": 1.0,
            "seed": seed,
            "torch_amp_options": {"dtype": "bfloat16"},
        }
        train_args = {
            "config": train_config_24gbgpu,
            "network_module": network_module_mem_v12,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }
        training_name = prefix_name + "/" + network_module_mem_v12 + f"_{epochs // 10}eps_test_conv_order_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}"
        train_job = training(training_name, train_data_bpe128, train_args, num_epochs=epochs, **default_returnn)

        train_job.rqmt['gpu_mem'] = 48
        train_job.rqmt['cpu'] = 8
        train_job.rqmt['mem'] = 36

        _ = run_non_memristor_eval(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe128,
            rasr_config=as_training_rasr_config,
            greedy_config=as_training_greedy_decoder_config,
            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
            rasr_prior_scales=rasr_prior_scales,
            rasr_lm_scales=rasr_lm_scales,
            report_dict=memristor_report,
        )

    for weight_precisions in [
        [8] * 3 + [4] * 6 + [8] * 3,
        [8] * 6 + [4] * 6,
        [8] * 3 + [4] * 2 + [3] * 2 + [4] * 2 + [8] * 3,
        [4] * 3 + [3] * 6 + [4] * 3,
        [8] * 2 + [4] * 8 + [8] * 2,
    ]:
        model_config = MemristorModelTrainConfigV12(
            **_make_model_config_kwargs(frontend_config_dim, dim, activation_bit, prior_train_dac_settings),
            weight_bit_prec=weight_precisions,
            pos_enc_converter_hardware_settings=prior_train_dac_settings,
        )
        for seed in range(2 if weight_precisions == [8] * 3 + [4] * 6 + [8] * 3 else 1):
            train_config_24gbgpu = {
                "optimizer": {
                    "class": "radam",
                    "epsilon": 1e-12,
                    "weight_decay": 1e-2,
                    "decoupled_weight_decay": True,
                },
                "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs // 2 - 20)))
                                  + list(np.linspace(5e-4, 5e-5, (epochs // 2 - 20)))
                                  + list(np.linspace(5e-5, 1e-7, 40)),
                #############
                "batch_size": 360 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "accum_grad_multiple_step": 1,
                "gradient_clip_norm": 1.0,
                "seed": seed,
                "torch_amp_options": {"dtype": "bfloat16"},
            }
            train_args = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v12,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }
            from itertools import groupby as _groupby
            w_str = "_".join(f"{v}x{sum(1 for _ in g)}" for v, g in _groupby(weight_precisions))
            training_name = prefix_name + "/" + network_module_mem_v12 + f"_{epochs // 10}eps_{dim}dim_w{w_str}_a{activation_bit}_seed_{seed}"
            train_job = training(training_name, train_data_bpe128, train_args, num_epochs=epochs, **default_returnn)


            if weight_precisions == [8] * 3 + [4] * 6 + [8] * 3:
                train_job.rqmt['gpu_mem'] = 48
                train_job.rqmt['cpu'] = 8
                train_job.rqmt['mem'] = 36
            elif not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                train_job.rqmt['cpu'] = 8
                train_job.hold()
                train_job.move_to_hpc = True
                if weight_precisions == [4] * 3 + [3] * 6 + [4] * 3:
                    train_job.rqmt['time'] = 24

            _ = run_non_memristor_eval(
                training_name=training_name,
                train_job=train_job,
                train_args=train_args,
                train_data=train_data_bpe128,
                rasr_config=as_training_rasr_config,
                greedy_config=as_training_greedy_decoder_config,
                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                rasr_prior_scales=rasr_prior_scales,
                rasr_lm_scales=rasr_lm_scales,
                report_dict=memristor_report,
            )

            run_memristor_cycle_eval(
                train_job=train_job,
                train_data=train_data_bpe128,
                train_config=train_config_24gbgpu,
                model_config=model_config,
                recog_name_prefix=prefix_name + "/" + network_module_mem_v12 +  f"_{epochs // 10}eps_{dim}dim_w{w_str}_a{activation_bit}_seed_{seed}",
                rasr_config=rasr_config_memristor,
                greedy_config=greedy_decoder_memristor,
                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                prior_scales=[0.5],
                lm_scales=[0.8],
                batch_size=3500000 if weight_bit not in [8] else 2500000,
                max_runs=5,
                report_dict=memristor_report,
                prior_network_module=network_module_mem_v12,
                recog_network_module=network_module_mem_v12,
                fast_inference=True,
            )

    # finetuning experiments
    for weight_precisions in [
        [3] * 12,
        [4] * 12,
        [8] * 12,
        [8] * 2 + [4] * 8 + [8] * 2,
        [4] * 3 + [3] * 6 + [4] * 3,
        [4] * 4 + [3] * 4 + [4] * 4,
        [4] * 5 + [3] * 2 + [4] * 5,
        [4] * 4 + [2] * 4 + [4] * 4,
        [8] * 2 + [3] * 8 + [8] * 2,
        [4] * 3 + [2] * 6 + [4] * 3,
        [8] * 2 + [2] * 8 + [8] * 2,
    ]:
        # TODO: define the baseline. Either 8 bit or 4 bit
        # TODO: finetune
        for num_finetune_epochs in [10, 50, 100, 250, 500, 750, 1000]:
            if weight_precisions not in [[3] * 12, [4] * 12, [8] * 12]:
                if num_finetune_epochs > 250:
                    continue
                if weight_precisions not in [
        [3] * 12,
        [4] * 12,
        [8] * 12,
        [8] * 2 + [4] * 8 + [8] * 2,
    ] and not num_finetune_epochs == 100:
                    continue
            model_config = MemristorModelTrainConfigV12(
                **_make_model_config_kwargs(frontend_config_dim, dim, activation_bit, prior_train_dac_settings),
                weight_bit_prec=weight_precisions,
                pos_enc_converter_hardware_settings=prior_train_dac_settings,
            )
            checkpoint_prefix = "conformer.module_list"
            train_config_24gbgpu = {
                "optimizer": {
                    "class": "radam",
                    "epsilon": 1e-12,
                    "weight_decay": 1e-2,
                    "decoupled_weight_decay": True,
                },
                "learning_rates": list(np.linspace(7e-6, 5e-4, (num_finetune_epochs // 2)))
                                  + list(np.linspace(5e-4, 1e-7, (num_finetune_epochs // 2))),
                #############
                "batch_size": 360 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "accum_grad_multiple_step": 1,
                "gradient_clip_norm": 1.0,
                "torch_amp_options": {"dtype": "bfloat16"},
                "preload_from_files": {
                    "conformer": {
                        "filename": FINETUNE_MODELS['experiments/librispeech/ctc_rnnt_standalone_2024/bpe_ls960_memristor/ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1.baseline_512dim_sub4_48gbgpu_100eps_radam_bs360_sp'],
                        "init_for_train": True,
                        "ignore_missing": False,
                        "var_name_mapping": {
                            **{
                                new: old
                                for layer in range(12)
                                for old, new in [
                                    (f"{checkpoint_prefix}.{layer}.module_list.2.layernorm.weight",
                                     f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.layernorm.weight"),
                                    (f"{checkpoint_prefix}.{layer}.module_list.2.layernorm.bias",
                                     f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.layernorm.bias"),
                                    (f"{checkpoint_prefix}.{layer}.module_list.2.pos_bias_u",
                                     f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.pos_bias_u"),
                                    (f"{checkpoint_prefix}.{layer}.module_list.2.pos_bias_v",
                                     f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.pos_bias_v"),
                                    (f"{checkpoint_prefix}.{layer}.module_list.2.qkv_proj.weight",
                                     f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.qkv_proj.weight"),
                                    (f"{checkpoint_prefix}.{layer}.module_list.2.qkv_proj.bias",
                                     f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.qkv_proj.bias"),
                                    (f"{checkpoint_prefix}.{layer}.module_list.2.out_proj.weight",
                                     f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.out_proj.weight"),
                                    (f"{checkpoint_prefix}.{layer}.module_list.2.out_proj.bias",
                                     f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.out_proj.bias"),
                                    (f"{checkpoint_prefix}.{layer}.module_list.2.linear_pos.weight",
                                     f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.linear_pos.weight"),
                                    ("output_linears.0.weight", "final_linear.0.weight"),
                                    ("output_linears.0.bias", "final_linear.0.bias"),
                                ]
                            },
                        },
                        'allowed_missing_prefix': get_observer_excludes(),
                    }
                }
            }
            train_args = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v12,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }
            from itertools import groupby as _groupby
            w_str = "_".join(f"{v}x{sum(1 for _ in g)}" for v, g in _groupby(weight_precisions))
            training_name = prefix_name + "/" + network_module_mem_v12 + f"_finetune_{num_finetune_epochs // 10}eps_{512}dim_w{w_str}_a{activation_bit}"
            train_job = training(training_name, train_data_bpe128, train_args, num_epochs=num_finetune_epochs, **default_returnn)
            if weight_precisions not in [
                    [3] * 12,
                    [4] * 12,
                    [8] * 12,
                    [8] * 2 + [4] * 8 + [8] * 2,
            ] and not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):
                train_job.rqmt['cpu'] = 8
                train_job.hold()
                train_job.move_to_hpc = True
            else:
              train_job.rqmt['gpu_mem'] = 24
              train_job.rqmt['cpu'] = 8
              train_job.rqmt['mem'] = 36
              train_job.has_priority = True

            _ = run_non_memristor_eval(
                training_name=training_name,
                train_job=train_job,
                train_args=train_args,
                train_data=train_data_bpe128,
                rasr_config=as_training_rasr_config,
                greedy_config=as_training_greedy_decoder_config,
                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                rasr_prior_scales=rasr_prior_scales,
                rasr_lm_scales=rasr_lm_scales,
                report_dict=memristor_report,
            )
            if weight_precisions == [4] * 12:
                model_config = MemristorModelTrainConfigV12(
                    **_make_model_config_kwargs(frontend_config_dim, dim, activation_bit, prior_train_dac_settings),
                    weight_bit_prec=weight_precisions,
                    pos_enc_converter_hardware_settings=prior_train_dac_settings,
                )
                model_config.module_list =["ff", "mhsa","conv", "ff"]

                train_config_24gbgpu = {
                    "optimizer": {
                        "class": "radam",
                        "epsilon": 1e-12,
                        "weight_decay": 1e-2,
                        "decoupled_weight_decay": True,
                    },
                    "learning_rates": list(np.linspace(7e-6, 5e-4, (num_finetune_epochs // 2)))
                                      + list(np.linspace(5e-4, 1e-7, (num_finetune_epochs // 2))),
                    #############
                    "batch_size": 360 * 16000,
                    "max_seq_length": {"audio_features": 35 * 16000},
                    "accum_grad_multiple_step": 1,
                    "gradient_clip_norm": 1.0,
                    "torch_amp_options": {"dtype": "bfloat16"},
                    "preload_from_files": {
                        "conformer": {
                            "filename": FINETUNE_MODELS[prefix_name + "/" + network_module_mem_v10 + f"_{100}eps_{dim}dim_w{8}_a{8}_seed_{0}"],
                            "init_for_train": True,
                            "ignore_missing": False,
                            'ignore_params_prefixes': get_observer_excludes(),
                            # "var_name_mapping": {
                            #     **{
                            #         new: old
                            #         for layer in range(12)
                            #         for old, new in [
                            #             (f"{checkpoint_prefix}.{layer}.module_list.2.layernorm.weight",
                            #                 f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.layernorm.weight"),
                            #             (f"{checkpoint_prefix}.{layer}.module_list.2.layernorm.bias",
                            #                 f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.layernorm.bias"),
                            #             (f"{checkpoint_prefix}.{layer}.module_list.2.pos_bias_u",
                            #                 f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.pos_bias_u"),
                            #             (f"{checkpoint_prefix}.{layer}.module_list.2.pos_bias_v",
                            #                 f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.pos_bias_v"),
                            #             (f"{checkpoint_prefix}.{layer}.module_list.2.qkv_proj.weight",
                            #                 f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.qkv_proj.weight"),
                            #             (f"{checkpoint_prefix}.{layer}.module_list.2.qkv_proj.bias",
                            #                 f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.qkv_proj.bias"),
                            #             (f"{checkpoint_prefix}.{layer}.module_list.2.out_proj.weight",
                            #                 f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.out_proj.weight"),
                            #             (f"{checkpoint_prefix}.{layer}.module_list.2.out_proj.bias",
                            #                 f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.out_proj.bias"),
                            #             (f"{checkpoint_prefix}.{layer}.module_list.2.linear_pos.weight",
                            #                 f"{checkpoint_prefix}.{layer}.module_list.2.mhsa.linear_pos.weight"),
                            #             ("output_linears.0.weight", "final_linear.0.weight"),
                            #             ("output_linears.0.bias", "final_linear.0.bias"),
                            #         ]
                            #     },
                            # },
                            # 'allowed_missing_prefix': get_observer_excludes(),
                        }
                    }
                }
                train_args = {
                    "config": train_config_24gbgpu,
                    "network_module": network_module_mem_v12,
                    "net_args": {"model_config_dict": asdict(model_config)},
                    "debug": False,
                    "post_config": {"num_workers_per_gpu": 8},
                    "use_speed_perturbation": True,
                }
                from itertools import groupby as _groupby
                w_str = "_".join(f"{v}x{sum(1 for _ in g)}" for v, g in _groupby(weight_precisions))
                # this already uses new observers
                training_name = prefix_name + "/" + network_module_mem_v12 + f"_finetune_fromw8_{num_finetune_epochs // 10}eps_{512}dim_w{w_str}_a{activation_bit}"
                train_job = training(training_name, train_data_bpe128, train_args, num_epochs=num_finetune_epochs,
                    **default_returnn)
                if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                    train_job.rqmt['cpu'] = 8
                    train_job.hold()
                    train_job.move_to_hpc = True
                    train_job.rqmt['time'] = 24 if num_finetune_epochs < 250 else 48

                _ = run_non_memristor_eval(
                    training_name=training_name,
                    train_job=train_job,
                    train_args=train_args,
                    train_data=train_data_bpe128,
                    rasr_config=as_training_rasr_config,
                    greedy_config=as_training_greedy_decoder_config,
                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                    rasr_prior_scales=rasr_prior_scales,
                    rasr_lm_scales=rasr_lm_scales,
                    report_dict=memristor_report,
                )




    from ...pytorch_networks.ctc.qat_0711.memristor_v13_cfg import QuantModelTrainConfigV13 as MemristorModelTrainConfigV13
    # TODO: finetune
    for weight_bit in [4, 8]:
        for activation_bit in [8]:
            for epochs in [1000]:
                for dim in [512]:
                    for weight_dropout in [0.0, 0.1, 0.2]:
                        frontend_config_dim = _make_frontend_config(dim)
                        prior_train_dac_settings = DacAdcHardwareSettings(
                            input_bits=0,
                            output_precision_bits=0,
                            output_range_bits=0,
                            hardware_input_vmax=0.6,
                            hardware_output_current_scaling=8020.0,
                        )

                        model_config = MemristorModelTrainConfigV13(
                            **_make_model_config_kwargs(frontend_config_dim, dim, activation_bit, prior_train_dac_settings),
                            weight_bit_prec=weight_bit,
                            pos_enc_converter_hardware_settings=prior_train_dac_settings,
                            weight_dropout=weight_dropout,
                        )

                        for seed in range(2):
                            train_config_24gbgpu = {
                                "optimizer": {
                                    "class": "radam",
                                    "epsilon": 1e-12,
                                    "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True,
                                },
                                "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs // 2 - 20)))
                                                  + list(np.linspace(5e-4, 5e-5, (epochs // 2 - 20)))
                                                  + list(np.linspace(5e-5, 1e-7, 40)),
                                #############
                                "batch_size": 360 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                                "gradient_clip_norm": 1.0,
                                "seed": seed,
                                "torch_amp_options": {"dtype": "bfloat16"},
                            }
                            train_args = {
                                "config": train_config_24gbgpu,
                                "network_module": network_module_mem_v13,
                                "net_args": {"model_config_dict": asdict(model_config)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }
                            training_name = prefix_name + "/" + network_module_mem_v13 + f"_{epochs // 10}eps_wdrop{weight_dropout}_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}"
                            train_job = training(training_name, train_data_bpe128, train_args, num_epochs=epochs,
                                **default_returnn)

                            if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                train_job.rqmt['cpu'] = 8
                                train_job.hold()
                                train_job.move_to_hpc = True

                            _ = run_non_memristor_eval(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data_bpe128,
                                rasr_config=as_training_rasr_config,
                                greedy_config=as_training_greedy_decoder_config,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                rasr_prior_scales=rasr_prior_scales,
                                rasr_lm_scales=rasr_lm_scales,
                                report_dict=memristor_report,
                            )

                            # memristor cycle eval of these wdrop trainings lives in
                            # noise.py (fast path) -- the eager base-graph copy was
                            # removed 2026-08-03


    tk.register_report("reports/lbs/v2/memristor_bpe", partial(build_qat_report_v2, memristor_report),
                       required=memristor_report, update_frequency=400)
    return FINETUNE_MODELS, memristor_report


def bpe_ls960_1225_memristor_mixed_prec():
    finetune_models, pretrain_report = bpe_ls960_1225_memristor()

    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/bpe_ls960_memristor/mixed_prec"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    train_data_bpe128 = build_bpe_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=128,
        settings=train_settings,
        use_postfix=False,
    )
    label_datastream_bpe128 = cast(LabelDatastream, train_data_bpe128.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe128.vocab_size

    dev_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.rasr_ctc_v1 import DecoderConfig as RasrDecoderConfig
    from ...rasr_recog_config import get_tree_timesync_recog_config, get_no_op_label_scorer_config

    recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
        lexicon_file=get_bpe_bliss_lexicon(bpe_size=128, add_blank=True, librispeech_key="train-other-960"),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=vocab_size_without_blank,
        max_beam_size=2048,
        score_threshold=18.0,
        logfile_suffix="recog",
        lm_config=get_arpa_lm_config(
            "4gram",
            lexicon_file=get_bpe_bliss_lexicon(bpe_size=128, add_blank=True, librispeech_key="train-other-960"),
            scale=0.0,
        ),
    )
    as_training_rasr_config = RasrDecoderConfig(
        rasr_config_file=recog_rasr_config,
        rasr_post_config=recog_rasr_post_config,
        blank_log_penalty=None,
        prior_scale=0.0,
        prior_file=None,
        turn_off_quant="leave_as_is",
    )
    rasr_config_memristor = copy.deepcopy(as_training_rasr_config)
    rasr_config_memristor.turn_off_quant = False

    rasr_prior_scales = [0.2, 0.3, 0.4, 0.5]
    rasr_lm_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_quant_v1 import DecoderConfig as GreedyDecoderConfig
    as_training_greedy_decoder_config = GreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe128.vocab,
        turn_off_quant="leave_as_is",
    )
    greedy_decoder_memristor = copy.deepcopy(as_training_greedy_decoder_config)
    greedy_decoder_memristor.turn_off_quant = False

    from ...pytorch_networks.ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
        ConformerPosEmbConfig,
    )
    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, LogMelFeatureExtractionV1Config,
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
    specaug_config_full = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    pos_emb_cfg = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    def _make_frontend_config(out_features):
        return VGG4LayerActFrontendV1Config_mod(
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
            out_features=out_features,
            activation=None,
        )

    def _make_model_config_kwargs(frontend_config, conformer_size, activation_bit, converter_hardware_settings):
        return dict(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            specaug_config=specaug_config_full,
            label_target_size=vocab_size_without_blank,
            conformer_size=conformer_size,
            num_layers=12,
            num_heads=8,
            ff_dim=conformer_size * 4,
            att_weights_dropout=0.1,
            conv_dropout=0.1,
            ff_dropout=0.1,
            mhsa_dropout=0.1,
            conv_kernel_size=31,
            final_dropout=0.1,
            specauc_start_epoch=11,
            weight_quant_dtype="qint8",
            weight_quant_method="per_tensor_symmetric",
            activation_quant_dtype="qint8",
            activation_quant_method="per_tensor_symmetric",
            dot_quant_dtype="qint8",
            dot_quant_method="per_tensor_symmetric",
            Av_quant_dtype="qint8",
            Av_quant_method="per_tensor_symmetric",
            moving_average=None,
            activation_bit_prec=activation_bit,
            quantize_output=False,
            converter_hardware_settings=converter_hardware_settings,
            quant_in_linear=True,
            num_cycles=0,
            correction_settings=None,
            weight_noise_func=None,
            weight_noise_values=None,
            weight_noise_start_epoch=None,
            pos_emb_config=pos_emb_cfg,
            module_list=["ff", "mhsa", "conv", "ff"], # this was handled wrong internally, so this was the correct ordering
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=None,
            aux_ctc_loss_scales=None,
            dropout_broadcast_axes=None,
        )

    from ...pytorch_networks.ctc.qat_0711.memristor_v14_cfg import (
        QuantModelTrainConfigV14 as MemristorModelTrainConfigV14,
    )
    from torch_memristor.memristor_modules import DacAdcHardwareSettings

    network_module_mem_v10 = "ctc.qat_0711.memristor_v10"
    network_module_mem_v14 = "ctc.qat_0711.memristor_v14"
    baseline_network_module = "ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"

    # seed report with float32 baseline (1000 epoch, 512dim)
    baseline_prefix = "experiments/librispeech/ctc_rnnt_standalone_2024/bpe_ls960_memristor"
    fp32_baseline_name = (
        baseline_prefix + "/" + baseline_network_module
        + ".baseline_512dim_sub4_48gbgpu_100eps_radam_bs360_sp"
    )
    memristor_report = {
        k: pretrain_report[k]
        for k in [fp32_baseline_name, fp32_baseline_name + "_greedy"]
        if k in pretrain_report
    }

    # device usage (crossbars / individual resistances) of the QAT baseline, shown as an
    # extra report column; keyed like memristor_report so the report can match the rows
    device_stats = {}
    # fast-path energy summaries per energy eval, own table in the report
    energy_stats = {}

    def _device_stats_spec(layer_cfg):
        """Convert one layer's weight_bit_prec cfg (v14 module-type scalars or v16
        per-matrix sub-specs) into the {regex: spec} format of the device stats job."""
        ff, mhsa, conv = layer_cfg["ff"], layer_cfg["mhsa"], layer_cfg["conv"]
        if isinstance(ff, dict):  # v16 per-matrix (sub-matrix precision) specs
            return {
                r"linear_ff\.weight$": ff["lin_1"],
                r"linear_out\.weight$": ff["lin_2"],
                r"qkv_proj\.weight$": mhsa["W_i"],
                r"out_proj\.weight$": mhsa["W_o"],
                r"linear_pos\.weight$": mhsa["learn_emb"],
                r"pointwise_conv1\.weight$": conv["pconv_1"],
                r"pointwise_conv2\.weight$": conv["pconv_2"],
                r"depthwise_conv\.weight$": conv["dconv"],
            }
        return {
            r"linear_ff\.weight$|linear_out\.weight$": ff,
            r"qkv_proj\.weight$|out_proj\.weight$|linear_pos\.weight$": mhsa,
            r"pointwise_conv1\.weight$|pointwise_conv2\.weight$|depthwise_conv\.weight$": conv,
        }

    def _register_device_stats(exp_key, checkpoint=None, layer_cfgs=None):
        """checkpoint defaults to the pretrained model of exp_key; layer_cfgs is the
        per-layer weight_bit_prec list of the model config (uniform 8 bit if None)."""
        if exp_key in device_stats:
            return
        if checkpoint is None:
            if exp_key not in finetune_models:
                return
            checkpoint = finetune_models[exp_key]
        if layer_cfgs is None:
            weight_bit_prec = 8
        else:
            # the job addresses matrices by name across all layers, so the per-layer
            # cfgs must be identical (holds for all current sweeps)
            assert all(cfg == layer_cfgs[0] for cfg in layer_cfgs), exp_key
            weight_bit_prec = _device_stats_spec(layer_cfgs[0])
        from i6_experiments.users.hilmes.tools.device_stats import CalculateMemristorDeviceStatsJob

        stats_job = CalculateMemristorDeviceStatsJob(
            checkpoint=checkpoint,
            weight_bit_prec=weight_bit_prec,
        )
        stats_job.add_alias(exp_key + "/device_stats")
        tk.register_output(exp_key + "/device_stats.txt", stats_job.out_report)
        device_stats[exp_key] = (stats_job.out_num_crossbars, stats_job.out_num_resistances)

    base_checkpoint_name = (
        baseline_prefix + "/" + network_module_mem_v10
        + f"_{1000 // 10}eps_{512}dim_w4_a{8}_seed_0"
    )
    # seed pretrain QAT baseline for this finetune configuration
    for key in [base_checkpoint_name, base_checkpoint_name + "_greedy"]:
        if key in pretrain_report:
            memristor_report[key] = pretrain_report[key]

    for activation_bit in [8]:
        for base_epochs in [1000]:
            for num_finetune_epochs in [50, 100, 250, 500, 750, 1000]:
                for dim in [512]:
                    frontend_config_dim = _make_frontend_config(dim)
                    prior_train_dac_settings = DacAdcHardwareSettings(
                        input_bits=0,
                        output_precision_bits=0,
                        output_range_bits=0,
                        hardware_input_vmax=0.6,
                        hardware_output_current_scaling=8020.0,
                    )
                    base_checkpoint_name = (
                        baseline_prefix + "/" + network_module_mem_v10
                        + f"_{base_epochs // 10}eps_{dim}dim_w8_a{activation_bit}_seed_0"
                    )
                    # seed pretrain QAT baseline for this finetune configuration
                    for key in [base_checkpoint_name, base_checkpoint_name + "_greedy"]:
                        if key in pretrain_report:
                            memristor_report[key] = pretrain_report[key]
                    _register_device_stats(base_checkpoint_name)
                    for mixed_prec_cfg, prec_name in [
                        ([{"ff": 4, "mhsa": 8, "conv": 4}] * 12, "ff4mhsa8conv4"),
                        ([{"ff": 4, "mhsa": 6, "conv": 4}] * 12, "ff4mhsa6conv4"),
                        ([{"ff": 6, "mhsa": 6, "conv": 6}] * 12, "ff6mhsa6conv6"),
                        ([{"ff": 8, "mhsa": 4, "conv": 4}] * 12, "ff8mhsa4conv4"),
                        ([{"ff": 8, "mhsa": 3, "conv": 3}] * 12, "ff8mhsa3conv3"),
                        ([{"ff": 8, "mhsa": 3, "conv": 2}] * 12, "ff8mhsa3conv2"),
                        ([{"ff": 4, "mhsa": 4, "conv": 8}] * 12, "ff4mhsa4conv8"),
                        ([{"ff": 4, "mhsa": 4, "conv": 4}] * 12, "ff4mhsa4conv4"),
                        ([{"ff": 8, "mhsa": 8, "conv": 8}] * 12, "ff8mhsa8conv8"),
                    ]:
                        if prec_name != "ff4mhsa8conv4" and num_finetune_epochs not in [50, 100, 250]:
                            continue

                        model_config = MemristorModelTrainConfigV14(
                            **_make_model_config_kwargs(
                                frontend_config_dim, dim, activation_bit, prior_train_dac_settings
                            ),
                            weight_bit_prec=mixed_prec_cfg,
                            pos_enc_converter_hardware_settings=prior_train_dac_settings,
                            weight_dropout=0.0,
                            weight_pruning=None,
                        )
                        train_config_finetune = {
                            "optimizer": {
                                "class": "radam",
                                "epsilon": 1e-12,
                                "weight_decay": 1e-2,
                                "decoupled_weight_decay": True,
                            },
                            "learning_rates": (
                                list(np.linspace(7e-6, 1e-4, num_finetune_epochs // 2))
                                + list(np.linspace(1e-4, 1e-7, num_finetune_epochs // 2))
                            ),
                            "batch_size": 360 * 16000,
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1,
                            "gradient_clip_norm": 1.0,
                            "seed": 0,
                            "torch_amp_options": {"dtype": "bfloat16"},
                            "preload_from_files": {
                                "model": {
                                    "filename": finetune_models[base_checkpoint_name],
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                }
                            },
                        }
                        train_args = {
                            "config": train_config_finetune,
                            "network_module": network_module_mem_v14,
                            "net_args": {"model_config_dict": asdict(model_config)},
                            "debug": False,
                            "post_config": {"num_workers_per_gpu": 8},
                            "use_speed_perturbation": True,
                        }
                        train_config_finetune = {
                            "optimizer": {
                                "class": "radam",
                                "epsilon": 1e-12,
                                "weight_decay": 1e-2,
                                "decoupled_weight_decay": True,
                            },
                            "learning_rates": (
                                list(np.linspace(7e-6, 1e-4, num_finetune_epochs // 2))
                                + list(np.linspace(1e-4, 1e-7, num_finetune_epochs // 2))
                            ),
                            "batch_size": 360 * 16000,
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1,
                            "gradient_clip_norm": 1.0,
                            "seed": 0,
                            "torch_amp_options": {"dtype": "bfloat16"},
                            "preload_from_files": {
                                "model": {
                                    "filename": finetune_models[base_checkpoint_name],
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    'ignore_params_prefixes': get_observer_excludes(),
                                }
                            },
                        }
                        train_args = {
                            "config": train_config_finetune,
                            "network_module": network_module_mem_v14,
                            "net_args": {"model_config_dict": asdict(model_config)},
                            "debug": False,
                            "post_config": {"num_workers_per_gpu": 8},
                            "use_speed_perturbation": True,
                        }
                        training_name = (
                            prefix_name + "/" + network_module_mem_v14
                            + f"_ft{num_finetune_epochs//10}eps_from{base_epochs // 10}eps"
                              f"_mixedprec_newobs_{prec_name}_{dim}dim_a{activation_bit}_seed_0"
                        )
                        train_job = training(
                            training_name, train_data_bpe128, train_args,
                            num_epochs=num_finetune_epochs, **default_returnn,
                        )
                        _register_device_stats(
                            training_name, train_job.out_checkpoints[num_finetune_epochs], mixed_prec_cfg
                        )

                        if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):
                            train_job.rqmt['cpu'] = 8
                            train_job.hold()
                            train_job.move_to_hpc = True
                            train_job.rqmt['time'] = 24

                        _ = run_non_memristor_eval(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args,
                            train_data=train_data_bpe128,
                            rasr_config=as_training_rasr_config,
                            greedy_config=as_training_greedy_decoder_config,
                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            rasr_prior_scales=rasr_prior_scales,
                            rasr_lm_scales=rasr_lm_scales,
                            report_dict=memristor_report,
                        )

                        # fast_inference replaced the slow eval under the same name after
                        # being validated against it (36 rasr + 6 greedy pairs within
                        # cycle noise; same switch as in noise.py). The finished fast
                        # results (previously "_newsynap_fast_hpctest_" rows) carry over
                        # unchanged since the name is not hashed; the slow searches were
                        # removed from the graph and cancelled.
                        run_memristor_cycle_eval(
                            train_job=train_job,
                            train_data=train_data_bpe128,
                            train_config=train_config_finetune,
                            model_config=model_config,
                            recog_name_prefix=training_name,
                            rasr_config=rasr_config_memristor,
                            greedy_config=greedy_decoder_memristor,
                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            prior_scales=[0.5],
                            lm_scales=[0.8],
                            batch_size=3500000,
                            max_runs=5,
                            report_dict=memristor_report,
                            prior_network_module=network_module_mem_v14,
                            recog_network_module=network_module_mem_v14,
                            fast_inference=True,
                        )

    # sub-matrix mixed precision: different 128x128 memristor subarrays of one weight matrix
    # are quantized to different bit precisions (claude v16 network)
    from ...pytorch_networks.ctc.qat_0711.claude.memristor_v16_dynmic_prec_cfg import (
        QuantModelTrainConfigV16 as MemristorModelTrainConfigV16SubMatrix,
    )

    network_module_mem_v16_submatrix = "ctc.qat_0711.claude.memristor_v16_dynmic_prec"

    def _stretch_prec_pattern(pattern, out_features, tile_size=128):
        """Stretch a precision pattern (one entry per equally sized band of output tile rows of
        W [out, in]) to one precision per 128-row tile,
        e.g. [8, 6, 6, 4] for out_features=2048 (16 tiles) -> [8]*4 + [6]*8 + [4]*4."""
        num_tiles = out_features // tile_size
        assert num_tiles % len(pattern) == 0, (num_tiles, pattern)
        return [prec for prec in pattern for _ in range(num_tiles // len(pattern))]

    def _rotated_prec_grid(base_pattern, out_features, in_features, tile_size=128, shift=1):
        """Build a balanced ("rotated") 2D precision grid for W [out_features, in_features].

        ``base_pattern`` is stretched (same banding rule as _stretch_prec_pattern) to one
        precision per input (column) tile, giving the layout of output-row-tile 0. Each
        subsequent row tile r is that pattern cyclically rotated by ``r * shift`` positions, so
        every row is a permutation of the same multiset (equal row sums, i.e. equal total
        precision per output channel) and -- when num_row_tiles is a multiple of num_col_tiles --
        every column is balanced too. Unlike the row-band _stretch_prec_pattern, no output
        channel runs entirely at the lowest precision. Cost-neutral: same multiset of tile
        precisions as the row-band layout. Returns a 2D list[list[int]] usable as a
        SubMatrixPrecision 2D spec, e.g. [8, 6, 4] on a 3x3 grid -> [[8,6,4],[6,4,8],[4,8,6]]."""
        num_row_tiles = out_features // tile_size
        num_col_tiles = in_features // tile_size
        assert num_col_tiles > 1, (
            f"rotated layout needs >1 input tile column (got in_features={in_features}); "
            f"use a row-band spec instead"
        )
        base_col = _stretch_prec_pattern(base_pattern, in_features, tile_size)
        return [
            [base_col[(c + r * shift) % num_col_tiles] for c in range(num_col_tiles)]
            for r in range(num_row_tiles)
        ]

    for activation_bit in [8]:
        for base_epochs in [1000]:
            for num_finetune_epochs in [10, 50, 100]:
                for dim in [512]:
                    frontend_config_dim = _make_frontend_config(dim)
                    prior_train_dac_settings = DacAdcHardwareSettings(
                        input_bits=0,
                        output_precision_bits=0,
                        output_range_bits=0,
                        hardware_input_vmax=0.6,
                        hardware_output_current_scaling=8020.0,
                    )
                    base_checkpoint_name = (
                        baseline_prefix + "/" + network_module_mem_v10
                        + f"_{base_epochs // 10}eps_{dim}dim_w8_a{activation_bit}_seed_0"
                    )
                    for key in [base_checkpoint_name, base_checkpoint_name + "_greedy"]:
                        if key in pretrain_report:
                            memristor_report[key] = pretrain_report[key]
                    _register_device_stats(base_checkpoint_name)

                    # example: row-tapered precision over the output tile rows of the large
                    # linears; in the qkv projection the query/key parts (attention logits,
                    # sensitive) stay at 8 bit while the value part runs at 4 bit
                    taper_pattern = [8, 6, 6, 4]
                    sub_prec_layer = {
                        "ff": {
                            "lin_1": _stretch_prec_pattern(taper_pattern, dim * 4),
                            "lin_2": _stretch_prec_pattern(taper_pattern, dim),
                        },
                        "mhsa": {
                            "W_i": _stretch_prec_pattern([8, 8, 4], 3 * dim),  # query/key/value bands
                            "W_o": _stretch_prec_pattern(taper_pattern, dim),
                            "learn_emb": 8,
                        },
                        "conv": {
                            "pconv_1": _stretch_prec_pattern(taper_pattern, 2 * dim),
                            "pconv_2": _stretch_prec_pattern(taper_pattern, dim),
                            "dconv": 8,
                        },
                    }
                    # balanced ("rotated") variant: same taper multiset, but distributed across
                    # the input tiles of each output-row tile and cyclically rotated down the
                    # rows, so every output channel gets a mix of precisions (Latin-square layout)
                    # instead of whole channels running at a single precision. qkv keeps its
                    # semantic row-band assignment (query/key 8 bit, value 4 bit).
                    # per-column balance is only approximate here: lin_2 is 4x16 (num_row_tiles
                    # not a multiple of num_col_tiles), so only 4 of the 16 rotations are sampled
                    # and its input tile columns range from all-8-bit to all-4-bit (col sums
                    # 32..16). The "rotb" variant below fixes this with shift=4 for lin_2.
                    sub_prec_layer_rot = {
                        "ff": {
                            "lin_1": _rotated_prec_grid(taper_pattern, dim * 4, dim),
                            "lin_2": _rotated_prec_grid(taper_pattern, dim, dim * 4),
                        },
                        "mhsa": {
                            "W_i": _stretch_prec_pattern([8, 8, 4], 3 * dim),  # keep semantic row-band
                            "W_o": _rotated_prec_grid(taper_pattern, dim, dim),
                            "learn_emb": 8,
                        },
                        "conv": {
                            "pconv_1": _rotated_prec_grid(taper_pattern, 2 * dim, dim),
                            "pconv_2": _rotated_prec_grid(taper_pattern, dim, dim),
                            "dconv": 8,
                        },
                    }
                    # fully balanced rotated variant: identical to sub_prec_layer_rot except that
                    # lin_2 rotates by one whole precision band per row tile (shift = band width
                    # = num_col_tiles // len(taper_pattern) = 4) instead of one tile, so the 4
                    # sampled rotations hit each band exactly once and every input tile column
                    # sums to the same total precision (all 24 instead of 32..16). Cost-neutral:
                    # same tile-precision multiset as the other two layouts.
                    sub_prec_layer_rot_bal = {
                        "ff": {
                            "lin_1": _rotated_prec_grid(taper_pattern, dim * 4, dim),
                            "lin_2": _rotated_prec_grid(taper_pattern, dim, dim * 4, shift=4),
                        },
                        "mhsa": {
                            "W_i": _stretch_prec_pattern([8, 8, 4], 3 * dim),  # keep semantic row-band
                            "W_o": _rotated_prec_grid(taper_pattern, dim, dim),
                            "learn_emb": 8,
                        },
                        "conv": {
                            "pconv_1": _rotated_prec_grid(taper_pattern, 2 * dim, dim),
                            "pconv_2": _rotated_prec_grid(taper_pattern, dim, dim),
                            "dconv": 8,
                        },
                    }
                    # ablation: every tensor at 6 bit like v14 ff6mhsa6conv6 (same device count),
                    # but through the v16 per-tile quantizers ([6] is a 1D spec -> per-tile
                    # observers/scales; a scalar would fall back to the legacy per-tensor
                    # quantizer). Isolates the per-tile-scale machinery from the taper layout.
                    # learn_emb and dconv stay scalar (per-tensor) as in the taper variants;
                    # conv weights do not support per-tile specs.
                    sub_prec_layer_uniform6 = {
                        "ff": {
                            "lin_1": _stretch_prec_pattern([6], dim * 4),
                            "lin_2": _stretch_prec_pattern([6], dim),
                        },
                        "mhsa": {
                            "W_i": _stretch_prec_pattern([6], 3 * dim),
                            "W_o": _stretch_prec_pattern([6], dim),
                            "learn_emb": 6,
                        },
                        "conv": {
                            "pconv_1": _stretch_prec_pattern([6], 2 * dim),
                            "pconv_2": _stretch_prec_pattern([6], dim),
                            "dconv": 6,
                        },
                    }
                    # prec_name scheme: "taper<pattern>" = output-row precision pattern of the
                    # large linears (taper8664 = [8, 6, 6, 4]), "qk8v4" = qkv projection with
                    # query/key at 8 bit and value at 4 bit, "rot" = balanced/rotated tile layout,
                    # "rotb" = rot with per-column balance recovered for lin_2 (shift=4),
                    # "uniform6" = all-6-bit per-tile ablation (device budget of ff6mhsa6conv6)
                    for sub_prec_cfg, prec_name in [
                        ([sub_prec_layer] * 12, "taper8664_qk8v4"),
                        ([sub_prec_layer_rot] * 12, "taper8664_rot_qk8v4"),
                        ([sub_prec_layer_rot_bal] * 12, "taper8664_rotb_qk8v4"),
                        ([sub_prec_layer_uniform6] * 12, "uniform6"),
                    ]:
                        v16_kwargs = _make_model_config_kwargs(
                            frontend_config_dim, dim, activation_bit, prior_train_dac_settings
                        )
                        # the claude v16 config replaces the v14 weight noise args
                        for noise_key in ["weight_noise_func", "weight_noise_values", "weight_noise_start_epoch"]:
                            v16_kwargs.pop(noise_key)
                        model_config = MemristorModelTrainConfigV16SubMatrix(
                            **v16_kwargs,
                            weight_bit_prec=sub_prec_cfg,
                            weight_noise=None,
                            pos_enc_converter_hardware_settings=prior_train_dac_settings,
                            weight_dropout=0.0,
                            weight_pruning=None,
                        )
                        train_config_finetune = {
                            "optimizer": {
                                "class": "radam",
                                "epsilon": 1e-12,
                                "weight_decay": 1e-2,
                                "decoupled_weight_decay": True,
                            },
                            "learning_rates": (
                                list(np.linspace(7e-6, 1e-4, num_finetune_epochs // 2))
                                + list(np.linspace(1e-4, 1e-7, num_finetune_epochs // 2))
                            ),
                            "batch_size": 360 * 16000,
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1,
                            "gradient_clip_norm": 1.0,
                            "seed": 0,
                            "torch_amp_options": {"dtype": "bfloat16"},
                            "preload_from_files": {
                                "model": {
                                    "filename": finetune_models[base_checkpoint_name],
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    "ignore_params_prefixes": get_observer_excludes(),
                                    # The uniform base checkpoint has a single weight observer per
                                    # matrix (...weight_quantizer.observer.*), but the mixed-precision
                                    # model expects per-tile observers
                                    # (...weight_quantizer.quantizers.R.C.observer.*). Since
                                    # get_observer_excludes() drops the checkpoint observers, the
                                    # per-tile ones would be reported missing; allow that and let the
                                    # MinMax observers recalibrate from data during finetuning.
                                    "allowed_missing_suffix": [
                                        ".observer.min_val",
                                        ".observer.max_val",
                                        ".observer.eps",
                                    ],
                                }
                            },
                        }
                        train_args = {
                            "config": train_config_finetune,
                            "network_module": network_module_mem_v16_submatrix,
                            "net_args": {"model_config_dict": asdict(model_config)},
                            "debug": False,
                            "post_config": {"num_workers_per_gpu": 8},
                            "use_speed_perturbation": True,
                        }
                        training_name = (
                            prefix_name + "/" + network_module_mem_v16_submatrix
                            + f"_ft{num_finetune_epochs//10}eps_from{base_epochs // 10}eps"
                              f"_submatrix_{prec_name}_{dim}dim_a{activation_bit}_seed_0"
                        )
                        train_job = training(
                            training_name, train_data_bpe128, train_args,
                            num_epochs=num_finetune_epochs, **default_returnn,
                        )
                        _register_device_stats(
                            training_name, train_job.out_checkpoints[num_finetune_epochs], sub_prec_cfg
                        )

                        if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):
                            train_job.rqmt['cpu'] = 8
                            # train_job.rqmt['gpu_mem'] = 24
                            # train_job.has_priority = True
                            train_job.hold()
                            train_job.move_to_hpc = True
                            train_job.rqmt['time'] = 12

                        _ = run_non_memristor_eval(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args,
                            train_data=train_data_bpe128,
                            rasr_config=as_training_rasr_config,
                            greedy_config=as_training_greedy_decoder_config,
                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            rasr_prior_scales=rasr_prior_scales,
                            rasr_lm_scales=rasr_lm_scales,
                            report_dict=memristor_report,
                        )
                        # energy for all finetune lengths (conversions all cached): cycle 1
                        # everywhere, plus cycle 5 on ft1eps as the cycle-invariance check
                        energy_kwargs = {
                            "energy_recog_network_module" : network_module_mem_v16_submatrix,
                            "energy_cycles" : [1, 5] if num_finetune_epochs == 10 else [1],
                            "energy_lm_scale" : 0.8,
                            "energy_prior_scale" : 0.5,
                            "energy_measure_args" : {
                                "detail": "summary",
                                "raw_batches": 0,
                                "raw_layers": [],
                                "raw_max_rows": 0,
                                "raw_dtype": "float16",
                                "percell_map_layers": [],
                                "enabled": True,
                            },
                            "energy_report_dict": energy_stats,
                        }
                        # fast_inference replaced the slow eval under the same name (see
                        # the v14 block above / noise.py); energy evals ride along
                        # unchanged and keep running locally.
                        run_memristor_cycle_eval(
                            train_job=train_job,
                            train_data=train_data_bpe128,
                            train_config=train_config_finetune,
                            model_config=model_config,
                            recog_name_prefix=training_name,
                            rasr_config=rasr_config_memristor,
                            greedy_config=greedy_decoder_memristor,
                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            prior_scales=[0.5],
                            lm_scales=[0.8],
                            batch_size=3500000,
                            max_runs=5,
                            report_dict=memristor_report,
                            prior_network_module=network_module_mem_v16_submatrix,
                            recog_network_module=network_module_mem_v16_submatrix,
                            fast_inference=True,
                            **energy_kwargs,
                        )

                        # v17 comparison run: identical setting to the v16 taper ft1eps run
                        # above (same base checkpoint, config and HPC hardware), only the
                        # network module changes to the vectorized/sync-free QAT
                        # implementation (bit-exact vs v16, see claude/test_v17_speed_equiv.py).
                        # Single run to compare training wall-clock and, as a sanity check,
                        # QAT + cycle WER; no energy evals.
                        if prec_name == "taper8664_qk8v4" and num_finetune_epochs == 10:
                            for network_module_mem_v17_submatrix in ["ctc.qat_0711.claude.memristor_v17_dynmic_prec"]:
                                train_args_v17 = copy.deepcopy(train_args)
                                train_args_v17["network_module"] = network_module_mem_v17_submatrix
                                training_name_v17 = (
                                    prefix_name + "/" + network_module_mem_v17_submatrix
                                    + f"_ft{num_finetune_epochs//10}eps_from{base_epochs // 10}eps"
                                      f"_submatrix_{prec_name}_{dim}dim_a{activation_bit}_seed_0"
                                )
                                train_job_v17 = training(
                                    training_name_v17, train_data_bpe128, train_args_v17,
                                    num_epochs=num_finetune_epochs, **default_returnn,
                                )
                                _register_device_stats(
                                    training_name_v17,
                                    train_job_v17.out_checkpoints[num_finetune_epochs],
                                    sub_prec_cfg,
                                )
                                if not os.path.exists(f"{train_job_v17._sis_path()}/finished.run.1"):
                                    train_job_v17.rqmt['cpu'] = 8
                                    train_job_v17.hold()
                                    train_job_v17.move_to_hpc = True
                                    train_job_v17.rqmt['time'] = 12
                                _ = run_non_memristor_eval(
                                    training_name=training_name_v17,
                                    train_job=train_job_v17,
                                    train_args=train_args_v17,
                                    train_data=train_data_bpe128,
                                    rasr_config=as_training_rasr_config,
                                    greedy_config=as_training_greedy_decoder_config,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    rasr_prior_scales=rasr_prior_scales,
                                    rasr_lm_scales=rasr_lm_scales,
                                    report_dict=memristor_report,
                                )
                                run_memristor_cycle_eval(
                                    train_job=train_job_v17,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_finetune,
                                    model_config=model_config,
                                    recog_name_prefix=training_name_v17,
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=greedy_decoder_memristor,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[0.5],
                                    lm_scales=[0.8],
                                    batch_size=3500000,
                                    max_runs=5,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v17_submatrix,
                                    recog_network_module=network_module_mem_v17_submatrix,
                                    fast_inference=True,
                                )

    tk.register_report(
        "reports/lbs/v2/memristor_bpe_mixed_prec",
        partial(build_qat_report_v2, memristor_report, device_info=device_stats, energy_info=energy_stats),
        required=memristor_report,
        update_frequency=400,
    )


def bpe_ls960_1225_memristor_pruning():
    finetune_models, pretrain_report = bpe_ls960_1225_memristor()

    baseline_prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/bpe_ls960_memristor"
    prefix_name = baseline_prefix_name + "/" + "pruning"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    train_data_bpe128 = build_bpe_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=128,
        settings=train_settings,
        use_postfix=False,
    )
    label_datastream_bpe128 = cast(LabelDatastream, train_data_bpe128.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe128.vocab_size

    dev_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.rasr_ctc_v1 import DecoderConfig as RasrDecoderConfig
    from ...rasr_recog_config import get_tree_timesync_recog_config, get_no_op_label_scorer_config

    recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
        lexicon_file=get_bpe_bliss_lexicon(bpe_size=128, add_blank=True, librispeech_key="train-other-960"),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=vocab_size_without_blank,
        max_beam_size=2048,
        score_threshold=18.0,
        logfile_suffix="recog",
        lm_config=get_arpa_lm_config(
            "4gram",
            lexicon_file=get_bpe_bliss_lexicon(bpe_size=128, add_blank=True, librispeech_key="train-other-960"),
            scale=0.0,
        ),
    )
    as_training_rasr_config = RasrDecoderConfig(
        rasr_config_file=recog_rasr_config,
        rasr_post_config=recog_rasr_post_config,
        blank_log_penalty=None,
        prior_scale=0.0,
        prior_file=None,
        turn_off_quant="leave_as_is",
    )
    rasr_config_memristor = copy.deepcopy(as_training_rasr_config)
    rasr_config_memristor.turn_off_quant = False

    rasr_prior_scales = [0.2, 0.3, 0.4, 0.5]
    rasr_lm_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_quant_v1 import DecoderConfig as GreedyDecoderConfig
    as_training_greedy_decoder_config = GreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe128.vocab,
        turn_off_quant="leave_as_is",
    )
    greedy_decoder_memristor = copy.deepcopy(as_training_greedy_decoder_config)
    greedy_decoder_memristor.turn_off_quant = False

    from ...pytorch_networks.ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
        ConformerPosEmbConfig,
    )
    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, LogMelFeatureExtractionV1Config,
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
    specaug_config_full = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    pos_emb_cfg = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    def _make_frontend_config(out_features):
        return VGG4LayerActFrontendV1Config_mod(
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
            out_features=out_features,
            activation=None,
        )

    def _make_model_config_kwargs(frontend_config, conformer_size, activation_bit, converter_hardware_settings):
        return dict(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            specaug_config=specaug_config_full,
            label_target_size=vocab_size_without_blank,
            conformer_size=conformer_size,
            num_layers=12,
            num_heads=8,
            ff_dim=conformer_size * 4,
            att_weights_dropout=0.1,
            conv_dropout=0.1,
            ff_dropout=0.1,
            mhsa_dropout=0.1,
            conv_kernel_size=31,
            final_dropout=0.1,
            specauc_start_epoch=11,
            weight_quant_dtype="qint8",
            weight_quant_method="per_tensor_symmetric",
            activation_quant_dtype="qint8",
            activation_quant_method="per_tensor_symmetric",
            dot_quant_dtype="qint8",
            dot_quant_method="per_tensor_symmetric",
            Av_quant_dtype="qint8",
            Av_quant_method="per_tensor_symmetric",
            moving_average=None,
            activation_bit_prec=activation_bit,
            quantize_output=False,
            converter_hardware_settings=converter_hardware_settings,
            quant_in_linear=True,
            num_cycles=0,
            correction_settings=None,
            weight_noise_func=None,
            weight_noise_values=None,
            weight_noise_start_epoch=None,
            pos_emb_config=pos_emb_cfg,
            module_list=["ff", "mhsa", "conv", "ff"], # this was handled wrong internally, so this was the correct ordering
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=None,
            aux_ctc_loss_scales=None,
            dropout_broadcast_axes=None,
        )

    from ...pytorch_networks.ctc.qat_0711.memristor_v14_cfg import (
        QuantModelTrainConfigV14 as MemristorModelTrainConfigV14,
        ThresholdPruningConfig,
        PercentilePruningConfig,
    )
    from ...pytorch_networks.ctc.qat_0711.memristor_v15_cfg import (
        QuantModelTrainConfigV15 as MemristorModelTrainConfigV15,
    )
    from ...pytorch_networks.ctc.qat_0711.memristor_v16_cfg import (
        QuantModelTrainConfigV16 as MemristorModelTrainConfigV16,
    )
    from torch_memristor.memristor_modules import DacAdcHardwareSettings

    network_module_mem_v10 = "ctc.qat_0711.memristor_v10"
    network_module_mem_v14 = "ctc.qat_0711.memristor_v14"
    network_module_mem_v15 = "ctc.qat_0711.memristor_v15"
    network_module_mem_v16 = "ctc.qat_0711.memristor_v16"
    baseline_network_module = "ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"

    fp32_baseline_name = (
        baseline_prefix_name + "/" + baseline_network_module
        + ".baseline_512dim_sub4_48gbgpu_100eps_radam_bs360_sp"
    )
    memristor_report = {
        k: pretrain_report[k]
        for k in [fp32_baseline_name, fp32_baseline_name + "_greedy"]
        if k in pretrain_report
    }

    pruning_stats = {}
    # fast-path energy summaries per energy eval, own table in the report
    energy_stats = {}

    def _register_pruning_stats(training_name, train_job, num_epochs, pruning_cfg):
        from i6_experiments.users.hilmes.tools.pruning_stats import CalculatePruningStatsJob

        stats_job = CalculatePruningStatsJob(
            checkpoint=train_job.out_checkpoints[num_epochs],
            threshold=getattr(pruning_cfg, "threshold", None),
            percentile=getattr(pruning_cfg, "percentile", None),
            # exactly the LinearQuant weights that get a pruning_config in memristor_v14/v15/v16
            # (quantize_output=False, so the final linear is not pruned; depthwise_conv neither)
            weight_name_patterns=[
                r"linear_ff\.weight$",
                r"linear_out\.weight$",
                r"pointwise_conv1\.weight$",
                r"pointwise_conv2\.weight$",
                r"qkv_proj\.weight$",
                r"out_proj\.weight$",
                r"linear_pos\.weight$",
            ],
        )
        stats_job.add_alias(training_name + "/pruning_stats")
        tk.register_output(training_name + "/pruning_stats.txt", stats_job.out_report)
        tk.register_output(training_name + "/pruning_histogram.png", stats_job.out_plot)
        pruning_stats[training_name] = stats_job.out_pruning_percent

    # from-scratch pruning
    for weight_bit in [4, 8]:
        for activation_bit in [8]:
            for epochs in [1000]:
                for dim in [512]:
                    frontend_config_dim = _make_frontend_config(dim)
                    prior_train_dac_settings = DacAdcHardwareSettings(
                        input_bits=0,
                        output_precision_bits=0,
                        output_range_bits=0,
                        hardware_input_vmax=0.6,
                        hardware_output_current_scaling=8020.0,
                    )
                    # seed non-pruned QAT baseline for this (weight_bit, dim, epochs)
                    base_name = (
                        baseline_prefix_name + "/" + network_module_mem_v10
                        + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_0"
                    )
                    for key in [base_name, base_name + "_greedy"]:
                        if key in pretrain_report:
                            memristor_report[key] = pretrain_report[key]
                    # Thresholding in comb. with weight decay causes a lot of weights to be pruned over training
                    for pruning_cfg, pruning_name in [
                        (ThresholdPruningConfig(start_epoch=1, threshold=0.02), "thresh0.02"),
                        (PercentilePruningConfig(start_epoch=1, percentile=0.1), "pctile0.1"),
                    ]:
                        model_config = MemristorModelTrainConfigV14(
                            **_make_model_config_kwargs(frontend_config_dim, dim, activation_bit, prior_train_dac_settings),
                            weight_bit_prec=weight_bit,
                            pos_enc_converter_hardware_settings=prior_train_dac_settings,
                            weight_dropout=0.0,
                            weight_pruning=pruning_cfg,
                        )
                        for seed in range(1):
                            train_config_24gbgpu = {
                                "optimizer": {
                                    "class": "radam",
                                    "epsilon": 1e-12,
                                    "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True,
                                },
                                "learning_rates": (
                                    list(np.linspace(7e-6, 5e-4, (epochs // 2 - 20)))
                                    + list(np.linspace(5e-4, 5e-5, (epochs // 2 - 20)))
                                    + list(np.linspace(5e-5, 1e-7, 40))
                                ),
                                "batch_size": 360 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                                "gradient_clip_norm": 1.0,
                                "seed": seed,
                                "torch_amp_options": {"dtype": "bfloat16"},
                            }
                            train_args = {
                                "config": train_config_24gbgpu,
                                "network_module": network_module_mem_v14,
                                "net_args": {"model_config_dict": asdict(model_config)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }
                            training_name = (
                                prefix_name + "/" + network_module_mem_v14
                                + f"_{epochs // 10}eps_{pruning_name}_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}"
                            )
                            train_job = training(
                                training_name, train_data_bpe128, train_args,
                                num_epochs=epochs, **default_returnn,
                            )
                            _register_pruning_stats(training_name, train_job, epochs, pruning_cfg)

                            if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):
                                train_job.rqmt['cpu'] = 8
                                train_job.hold()
                                train_job.move_to_hpc = True

                            _ = run_non_memristor_eval(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data_bpe128,
                                rasr_config=as_training_rasr_config,
                                greedy_config=as_training_greedy_decoder_config,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                rasr_prior_scales=rasr_prior_scales,
                                rasr_lm_scales=rasr_lm_scales,
                                report_dict=memristor_report,
                            )
                            if "pctile" in pruning_name:
                                run_memristor_cycle_eval(
                                    train_job=train_job,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_24gbgpu,
                                    model_config=model_config,
                                    recog_name_prefix=training_name,
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=None,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[0.5],
                                    lm_scales=[0.8],
                                    batch_size=3500000 if weight_bit not in [8] else 2500000,
                                    max_runs=5,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v14,
                                    recog_network_module=network_module_mem_v14,
                                    fast_inference=True,
                                )

    def _make_v15_model_config_kwargs(frontend_config, conformer_size, activation_bit, converter_hardware_settings):
        return dict(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            specaug_config=specaug_config_full,
            label_target_size=vocab_size_without_blank,
            conformer_size=conformer_size,
            num_layers=12,
            num_heads=8,
            ff_dim=conformer_size * 4,
            att_weights_dropout=0.1,
            conv_dropout=0.1,
            ff_dropout=0.1,
            mhsa_dropout=0.1,
            conv_kernel_size=31,
            final_dropout=0.1,
            specauc_start_epoch=11,
            weight_quant_dtype="qint8",
            weight_quant_method="per_tensor_symmetric",
            activation_quant_dtype="qint8",
            activation_quant_method="per_tensor_symmetric",
            dot_quant_dtype="qint8",
            dot_quant_method="per_tensor_symmetric",
            Av_quant_dtype="qint8",
            Av_quant_method="per_tensor_symmetric",
            moving_average=None,
            activation_bit_prec=activation_bit,
            quantize_output=False,
            converter_hardware_settings=converter_hardware_settings,
            quant_in_linear=True,
            num_cycles=0,
            correction_settings=None,
            weight_noise=None,
            pos_emb_config=pos_emb_cfg,
            module_list=["ff", "mhsa", "conv", "ff"],
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=None,
            aux_ctc_loss_scales=None,
            dropout_broadcast_axes=None,
        )

    # finetuning from non-pruned v10 checkpoints with v15
    for weight_bit in [4, 8]:
        for activation_bit in [8]:
            for base_epochs in [1000]:
                for num_finetune_epochs in [50]:
                    for dim in [512]:
                        frontend_config_dim = _make_frontend_config(dim)
                        prior_train_dac_settings = DacAdcHardwareSettings(
                            input_bits=0,
                            output_precision_bits=0,
                            output_range_bits=0,
                            hardware_input_vmax=0.6,
                            hardware_output_current_scaling=8020.0,
                        )
                        base_checkpoint_name = (
                            baseline_prefix_name + "/" + network_module_mem_v10
                            + f"_{base_epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_0"
                        )
                        for key in [base_checkpoint_name, base_checkpoint_name + "_greedy"]:
                            if key in pretrain_report:
                                memristor_report[key] = pretrain_report[key]
                        for pruning_cfg, pruning_name in [
                            (ThresholdPruningConfig(start_epoch=1, threshold=0.02), "thresh0.02"),
                            (ThresholdPruningConfig(start_epoch=1, threshold=0.1), "thresh0.1"),
                            (ThresholdPruningConfig(start_epoch=1, threshold=1.0), "thresh1.0"),
                            (PercentilePruningConfig(start_epoch=1, percentile=0.1), "pctile0.1"),
                            (PercentilePruningConfig(start_epoch=1, percentile=0.2), "pctile0.2"),
                            (PercentilePruningConfig(start_epoch=1, percentile=0.3), "pctile0.3"),
                            (PercentilePruningConfig(start_epoch=1, percentile=0.4), "pctile0.4"),
                            (PercentilePruningConfig(start_epoch=1, percentile=0.8), "pctile0.8"),
                        ]:
                            model_config = MemristorModelTrainConfigV15(
                                **_make_v15_model_config_kwargs(frontend_config_dim, dim, activation_bit, prior_train_dac_settings),
                                weight_bit_prec=weight_bit,
                                pos_enc_converter_hardware_settings=prior_train_dac_settings,
                                weight_dropout=0.0,
                                weight_pruning=pruning_cfg,
                            )
                            train_config_finetune = {
                                "optimizer": {
                                    "class": "radam",
                                    "epsilon": 1e-12,
                                    "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True,
                                },
                                "learning_rates": (
                                    list(np.linspace(7e-6, 1e-4, num_finetune_epochs // 2))
                                    + list(np.linspace(1e-4, 1e-7, num_finetune_epochs // 2))
                                ),
                                "batch_size": 360 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                                "gradient_clip_norm": 1.0,
                                "seed": 0,
                                "torch_amp_options": {"dtype": "bfloat16"},
                                "preload_from_files": {
                                    "model": {
                                        "filename": finetune_models[base_checkpoint_name],
                                        "init_for_train": True,
                                        "ignore_missing": False,
                                    }
                                },
                            }
                            train_args = {
                                "config": train_config_finetune,
                                "network_module": network_module_mem_v15,
                                "net_args": {"model_config_dict": asdict(model_config)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }
                            training_name = (
                                prefix_name + "/" + network_module_mem_v15
                                + f"_ft{num_finetune_epochs}eps_from{base_epochs // 10}eps"
                                f"_{pruning_name}_{dim}dim_w{weight_bit}_a{activation_bit}_seed_0"
                            )
                            train_job = training(
                                training_name, train_data_bpe128, train_args,
                                num_epochs=num_finetune_epochs, **default_returnn,
                            )
                            _register_pruning_stats(training_name, train_job, num_finetune_epochs, pruning_cfg)

                            if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):
                                train_job.rqmt['cpu'] = 8
                                train_job.hold()
                                train_job.move_to_hpc = True

                            _ = run_non_memristor_eval(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data_bpe128,
                                rasr_config=as_training_rasr_config,
                                greedy_config=as_training_greedy_decoder_config,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                rasr_prior_scales=rasr_prior_scales,
                                rasr_lm_scales=rasr_lm_scales,
                                report_dict=memristor_report,
                            )

                            run_memristor_cycle_eval(
                                train_job=train_job,
                                train_data=train_data_bpe128,
                                train_config=train_config_finetune,
                                model_config=model_config,
                                recog_name_prefix=training_name,
                                rasr_config=rasr_config_memristor,
                                greedy_config=None,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                prior_scales=[0.5],
                                lm_scales=[0.8],
                                batch_size=3500000 if weight_bit not in [8] else 2500000,
                                max_runs=5,
                                report_dict=memristor_report,
                                prior_network_module=network_module_mem_v15,
                                recog_network_module=network_module_mem_v15,
                                fast_inference=True,
                            )

                            model_config_no_prune = copy.deepcopy(model_config)
                            model_config_no_prune.weight_pruning = None
                            train_args_no_prune = copy.deepcopy(train_args)
                            train_args_no_prune["net_args"] = {"model_config_dict": asdict(model_config_no_prune)}

                            # _ = run_non_memristor_eval(
                            #     training_name=training_name + "/no_pruning",
                            #     train_job=train_job,
                            #     train_args=train_args_no_prune,
                            #     train_data=train_data_bpe128,
                            #     rasr_config=as_training_rasr_config,
                            #     greedy_config=as_training_greedy_decoder_config,
                            #     dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            #     rasr_prior_scales=rasr_prior_scales,
                            #     rasr_lm_scales=rasr_lm_scales,
                            #     report_dict=memristor_report,
                            # )
                            #
                            # run_memristor_cycle_eval(
                            #     train_job=train_job,
                            #     train_data=train_data_bpe128,
                            #     train_config=train_config_finetune,
                            #     model_config=model_config_no_prune,
                            #     recog_name_prefix=training_name + "/no_pruning",
                            #     rasr_config=rasr_config_memristor,
                            #     greedy_config=greedy_decoder_memristor,
                            #     dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            #     prior_scales=[0.5],
                            #     lm_scales=[0.8],
                            #     batch_size=3500000 if weight_bit not in [8] else 2500000,
                            #     max_runs=5,
                            #     report_dict=memristor_report,
                            #     prior_network_module=network_module_mem_v15,
                            #     recog_network_module=network_module_mem_v15,
                            # )

                            # explicit pruning comparison: pre-prune checkpoint via job, then eval with same
                            # pruning config (runtime masking becomes a no-op on already-zero weights)
                            # _ = run_non_memristor_eval(
                            #     training_name=training_name + "/explicit_pruning",
                            #     train_job=train_job,
                            #     train_args=train_args,
                            #     train_data=train_data_bpe128,
                            #     rasr_config=as_training_rasr_config,
                            #     greedy_config=as_training_greedy_decoder_config,
                            #     dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            #     rasr_prior_scales=rasr_prior_scales,
                            #     rasr_lm_scales=rasr_lm_scales,
                            #     report_dict=memristor_report,
                            #     prune_weights=True,
                            # )
    from ...pytorch_networks.ctc.qat_0711.memristor_v16_cfg import (
        ThresholdPruningConfig as ThreholdPruningConfigV2,
        PercentilePruningConfig as PercentilePruningConfigV2,
        PercentileThresholdPruningConfig,
    )
    for weight_bit in [4, 8]:
        for activation_bit in [8]:
            for base_epochs in [1000]:
                for num_finetune_epochs in [50]:
                    for dim in [512]:
                        frontend_config_dim = _make_frontend_config(dim)
                        prior_train_dac_settings = DacAdcHardwareSettings(
                            input_bits=0,
                            output_precision_bits=0,
                            output_range_bits=0,
                            hardware_input_vmax=0.6,
                            hardware_output_current_scaling=8020.0,
                        )
                        base_checkpoint_name = (
                            baseline_prefix_name + "/" + network_module_mem_v10
                            + f"_{base_epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_0"
                        )
                        for key in [base_checkpoint_name, base_checkpoint_name + "_greedy"]:
                            if key in pretrain_report:
                                memristor_report[key] = pretrain_report[key]
                        # v15 is identical to v16 when prune_before_quant is False
                        for pruning_cfg, pruning_name in [
                            (ThreholdPruningConfigV2(start_epoch=1, threshold=0.02, prune_before_quant=True), "thresh0.02_beforequant"),
                            (ThreholdPruningConfigV2(start_epoch=1, threshold=0.1, prune_before_quant=True), "thresh0.1_beforequant"),
                            (ThreholdPruningConfigV2(start_epoch=1, threshold=1.0, prune_before_quant=True), "thresh1.0_beforequant"),
                            (PercentilePruningConfigV2(start_epoch=1, percentile=0.1, prune_before_quant=True), "pctile0.1_beforequant"),
                            (PercentilePruningConfigV2(start_epoch=1, percentile=0.2, prune_before_quant=True), "pctile0.2_beforequant"),
                            (PercentilePruningConfigV2(start_epoch=1, percentile=0.3, prune_before_quant=True), "pctile0.3_beforequant"),
                            (PercentilePruningConfigV2(start_epoch=1, percentile=0.4, prune_before_quant=True), "pctile0.4_beforequant"),
                            (PercentilePruningConfigV2(start_epoch=1, percentile=0.8, prune_before_quant=True), "pctile0.8_beforequant"),
                        ]:
                            model_config = MemristorModelTrainConfigV16(
                                **_make_v15_model_config_kwargs(frontend_config_dim, dim, activation_bit, prior_train_dac_settings),
                                weight_bit_prec=weight_bit,
                                pos_enc_converter_hardware_settings=prior_train_dac_settings,
                                weight_dropout=0.0,
                                weight_pruning=pruning_cfg,
                            )
                            train_config_finetune = {
                                "optimizer": {
                                    "class": "radam",
                                    "epsilon": 1e-12,
                                    "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True,
                                },
                                "learning_rates": (
                                    list(np.linspace(7e-6, 1e-4, num_finetune_epochs // 2))
                                    + list(np.linspace(1e-4, 1e-7, num_finetune_epochs // 2))
                                ),
                                "batch_size": 360 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                                "gradient_clip_norm": 1.0,
                                "seed": 0,
                                "torch_amp_options": {"dtype": "bfloat16"},
                                "preload_from_files": {
                                    "model": {
                                        "filename": finetune_models[base_checkpoint_name],
                                        "init_for_train": True,
                                        "ignore_missing": False,
                                    }
                                },
                            }
                            train_args = {
                                "config": train_config_finetune,
                                "network_module": network_module_mem_v16,
                                "net_args": {"model_config_dict": asdict(model_config)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }
                            training_name = (
                                prefix_name + "/" + network_module_mem_v16
                                + f"_ft{num_finetune_epochs}eps_from{base_epochs // 10}eps"
                                f"_{pruning_name}_{dim}dim_w{weight_bit}_a{activation_bit}_seed_0"
                            )
                            train_job = training(
                                training_name, train_data_bpe128, train_args,
                                num_epochs=num_finetune_epochs, **default_returnn,
                            )
                            _register_pruning_stats(training_name, train_job, num_finetune_epochs, pruning_cfg)

                            if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):
                                train_job.rqmt['cpu'] = 8
                                train_job.hold()
                                train_job.move_to_hpc = True

                            _ = run_non_memristor_eval(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data_bpe128,
                                rasr_config=as_training_rasr_config,
                                greedy_config=as_training_greedy_decoder_config,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                rasr_prior_scales=rasr_prior_scales,
                                rasr_lm_scales=rasr_lm_scales,
                                report_dict=memristor_report,
                            )

                            # fast-path energy for the percentile sparsity axis (cycle 1 only,
                            # energy is cycle-invariant, see mixed_prec); thresh rows are
                            # WER-broken or redundant with pctile
                            if "pctile" in pruning_name:
                                energy_kwargs = {
                                    "energy_recog_network_module": network_module_mem_v16,
                                    "energy_cycles": [1],
                                    "energy_lm_scale": 0.8,
                                    "energy_prior_scale": 0.5,
                                    "energy_measure_args": {
                                        "detail": "summary",
                                        "raw_batches": 0,
                                        "raw_layers": [],
                                        "raw_max_rows": 0,
                                        "raw_dtype": "float16",
                                        "percell_map_layers": [],
                                        "enabled": True,
                                    },
                                    "energy_report_dict": energy_stats,
                                }
                            else:
                                energy_kwargs = {}

                            run_memristor_cycle_eval(
                                train_job=train_job,
                                train_data=train_data_bpe128,
                                train_config=train_config_finetune,
                                model_config=model_config,
                                recog_name_prefix=training_name,
                                rasr_config=rasr_config_memristor,
                                greedy_config=None,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                prior_scales=[0.5],
                                lm_scales=[0.8],
                                batch_size=3500000 if weight_bit not in [8] else 2500000,
                                max_runs=5,
                                report_dict=memristor_report,
                                prior_network_module=network_module_mem_v16,
                                recog_network_module=network_module_mem_v16,
                                fast_inference=True,
                                **energy_kwargs,
                            )

                            if weight_bit == 4 and pruning_name in [
                                "thresh0.02_beforequant", "pctile0.2_beforequant",
                            ]:
                                # parallel no_outq eval (out-quants -> nn.Identity) for two
                                # pruning settings; conversion/prior jobs shared with the run
                                # above, results land under <training_name>_no_outq
                                run_memristor_cycle_eval(
                                    train_job=train_job,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_finetune,
                                    model_config=model_config,
                                    recog_name_prefix=training_name,
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=None,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[0.5],
                                    lm_scales=[0.8],
                                    batch_size=3500000,
                                    max_runs=5,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v16,
                                    recog_network_module=network_module_mem_v16,
                                    fast_inference=True,
                                    no_outq=True,
                                )

                            # model_config_no_prune = copy.deepcopy(model_config)
                            # model_config_no_prune.weight_pruning = None
                            # train_args_no_prune = copy.deepcopy(train_args)
                            # train_args_no_prune["net_args"] = {"model_config_dict": asdict(model_config_no_prune)}
                            #
                            # _ = run_non_memristor_eval(
                            #     training_name=training_name + "/no_pruning",
                            #     train_job=train_job,
                            #     train_args=train_args_no_prune,
                            #     train_data=train_data_bpe128,
                            #     rasr_config=as_training_rasr_config,
                            #     greedy_config=as_training_greedy_decoder_config,
                            #     dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            #     rasr_prior_scales=rasr_prior_scales,
                            #     rasr_lm_scales=rasr_lm_scales,
                            #     report_dict=memristor_report,
                            # )
                            #
                            # run_memristor_cycle_eval(
                            #     train_job=train_job,
                            #     train_data=train_data_bpe128,
                            #     train_config=train_config_finetune,
                            #     model_config=model_config_no_prune,
                            #     recog_name_prefix=training_name + "/no_pruning",
                            #     rasr_config=rasr_config_memristor,
                            #     greedy_config=greedy_decoder_memristor,
                            #     dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            #     prior_scales=[0.5],
                            #     lm_scales=[0.8],
                            #     batch_size=3500000 if weight_bit not in [8] else 2500000,
                            #     max_runs=5,
                            #     report_dict=memristor_report,
                            #     prior_network_module=network_module_mem_v16,
                            #     recog_network_module=network_module_mem_v16,
                            # )

                            # unpruned anchor for the energy axis: same checkpoint, pruning mask
                            # dropped at conversion (isolates the mask's device/current savings);
                            # once per weight_bit, new conversion/prior jobs
                            if pruning_name == "pctile0.1_beforequant":
                                model_config_no_prune = copy.deepcopy(model_config)
                                model_config_no_prune.weight_pruning = None
                                run_memristor_cycle_eval(
                                    train_job=train_job,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_finetune,
                                    model_config=model_config_no_prune,
                                    recog_name_prefix=training_name + "/no_pruning",
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=None,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[0.5],
                                    lm_scales=[0.8],
                                    batch_size=3500000 if weight_bit not in [8] else 2500000,
                                    max_runs=5,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v16,
                                    recog_network_module=network_module_mem_v16,
                                    fast_inference=True,
                                    **energy_kwargs,
                                )

                            # # explicit pruning comparison: pre-prune checkpoint via job, then eval with same
                            # # pruning config (runtime masking becomes a no-op on already-zero weights)
                            # _ = run_non_memristor_eval(
                            #     training_name=training_name + "/explicit_pruning",
                            #     train_job=train_job,
                            #     train_args=train_args,
                            #     train_data=train_data_bpe128,
                            #     rasr_config=as_training_rasr_config,
                            #     greedy_config=as_training_greedy_decoder_config,
                            #     dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            #     rasr_prior_scales=rasr_prior_scales,
                            #     rasr_lm_scales=rasr_lm_scales,
                            #     report_dict=memristor_report,
                            #     prune_weights=True,
                            # )

    # 25-eps percentile finetune with thresholds applied on top at recognition only
    # (union mask via PercentileThresholdPruningConfig, no retraining per threshold)
    for weight_bit in [4, 8]:
        for activation_bit in [8]:
            for base_epochs in [1000]:
                for num_finetune_epochs in [25]:
                    for dim in [512]:
                        frontend_config_dim = _make_frontend_config(dim)
                        prior_train_dac_settings = DacAdcHardwareSettings(
                            input_bits=0,
                            output_precision_bits=0,
                            output_range_bits=0,
                            hardware_input_vmax=0.6,
                            hardware_output_current_scaling=8020.0,
                        )
                        base_checkpoint_name = (
                            baseline_prefix_name + "/" + network_module_mem_v10
                            + f"_{base_epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_0"
                        )
                        for key in [base_checkpoint_name, base_checkpoint_name + "_greedy"]:
                            if key in pretrain_report:
                                memristor_report[key] = pretrain_report[key]
                        for percentile in [0.1, 0.2, 0.4]:
                            pruning_cfg = PercentilePruningConfigV2(
                                start_epoch=1, percentile=percentile, prune_before_quant=True
                            )
                            pruning_name = f"pctile{percentile}_beforequant"
                            model_config = MemristorModelTrainConfigV16(
                                **_make_v15_model_config_kwargs(frontend_config_dim, dim, activation_bit, prior_train_dac_settings),
                                weight_bit_prec=weight_bit,
                                pos_enc_converter_hardware_settings=prior_train_dac_settings,
                                weight_dropout=0.0,
                                weight_pruning=pruning_cfg,
                            )
                            train_config_finetune = {
                                "optimizer": {
                                    "class": "radam",
                                    "epsilon": 1e-12,
                                    "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True,
                                },
                                "learning_rates": (
                                    list(np.linspace(7e-6, 1e-4, num_finetune_epochs // 2))
                                    + list(np.linspace(1e-4, 1e-7, num_finetune_epochs // 2))
                                ),
                                "batch_size": 360 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                                "gradient_clip_norm": 1.0,
                                "seed": 0,
                                "torch_amp_options": {"dtype": "bfloat16"},
                                "preload_from_files": {
                                    "model": {
                                        "filename": finetune_models[base_checkpoint_name],
                                        "init_for_train": True,
                                        "ignore_missing": False,
                                    }
                                },
                            }
                            train_args = {
                                "config": train_config_finetune,
                                "network_module": network_module_mem_v16,
                                "net_args": {"model_config_dict": asdict(model_config)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }
                            training_name = (
                                prefix_name + "/pctile_thresh/" + network_module_mem_v16
                                + f"_ft{num_finetune_epochs}eps_from{base_epochs // 10}eps"
                                f"_{pruning_name}_{dim}dim_w{weight_bit}_a{activation_bit}_seed_0"
                            )
                            train_job = training(
                                training_name, train_data_bpe128, train_args,
                                num_epochs=num_finetune_epochs, **default_returnn,
                            )
                            _register_pruning_stats(training_name, train_job, num_finetune_epochs, pruning_cfg)

                            if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):
                                train_job.rqmt['cpu'] = 8
                                train_job.hold()
                                train_job.move_to_hpc = True
                                train_job.rqmt['time'] = 12

                            _ = run_non_memristor_eval(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data_bpe128,
                                rasr_config=as_training_rasr_config,
                                greedy_config=as_training_greedy_decoder_config,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                rasr_prior_scales=rasr_prior_scales,
                                rasr_lm_scales=rasr_lm_scales,
                                report_dict=memristor_report,
                            )

                            if os.path.exists(f"{train_job._sis_path()}/finished.run.1") and False:
                                run_memristor_cycle_eval(
                                    train_job=train_job,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_finetune,
                                    model_config=model_config,
                                    recog_name_prefix=training_name,
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=None,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[0.5],
                                    lm_scales=[0.8],
                                    batch_size=3500000 if weight_bit not in [8] else 2500000,
                                    max_runs=5,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v16,
                                    recog_network_module=network_module_mem_v16,
                                )

                            for threshold in [0.02, 0.05, 0.1]:
                                combined_cfg = PercentileThresholdPruningConfig(
                                    start_epoch=1,
                                    percentile=percentile,
                                    threshold=threshold,
                                    prune_before_quant=True,
                                )
                                model_config_thresh = copy.deepcopy(model_config)
                                model_config_thresh.weight_pruning = combined_cfg
                                train_args_thresh = copy.deepcopy(train_args)
                                train_args_thresh["net_args"] = {"model_config_dict": asdict(model_config_thresh)}
                                variant_name = training_name + f"/thresh{threshold}_on_top"

                                _ = run_non_memristor_eval(
                                    training_name=variant_name,
                                    train_job=train_job,
                                    train_args=train_args_thresh,
                                    train_data=train_data_bpe128,
                                    rasr_config=as_training_rasr_config,
                                    greedy_config=as_training_greedy_decoder_config,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    rasr_prior_scales=rasr_prior_scales,
                                    rasr_lm_scales=rasr_lm_scales,
                                    report_dict=memristor_report,
                                )
                                _register_pruning_stats(variant_name, train_job, num_finetune_epochs, combined_cfg)

                                if os.path.exists(f"{train_job._sis_path()}/finished.run.1") and False:
                                    run_memristor_cycle_eval(
                                        train_job=train_job,
                                        train_data=train_data_bpe128,
                                        train_config=train_config_finetune,
                                        model_config=model_config_thresh,
                                        recog_name_prefix=variant_name,
                                        rasr_config=rasr_config_memristor,
                                        greedy_config=None,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        prior_scales=[0.5],
                                        lm_scales=[0.8],
                                        batch_size=3500000 if weight_bit not in [8] else 2500000,
                                        max_runs=5,
                                        report_dict=memristor_report,
                                        prior_network_module=network_module_mem_v16,
                                        recog_network_module=network_module_mem_v16,
                                    )

    tk.register_report(
        "reports/lbs/v2/memristor_bpe_pruning",
        partial(build_qat_report_v2, memristor_report, pruning_stats, energy_info=energy_stats),
        required=memristor_report,
        update_frequency=400,
    )
