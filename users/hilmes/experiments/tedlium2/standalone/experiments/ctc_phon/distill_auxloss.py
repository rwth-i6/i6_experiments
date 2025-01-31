from dataclasses import asdict
import numpy as np
from typing import cast
import copy

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon, get_eow_bliss
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT, SCTK_BINARY_PATH
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, get_forward_config, generate_kd_hypothesis
from ...report import generate_report
from functools import partial
from sisyphus import tk
from .tune_eval import (
    build_report,
    eval_model,
    build_distill_report,
    tune_and_evaluate_helper,
    search,
    build_base_report,
)
from i6_core.returnn.forward import ReturnnForwardJobV2

from i6_core.corpus.convert import CorpusToStmJob
from i6_core.recognition.scoring import ScliteJob
from i6_core.returnn.search import SearchWordsToCTMJob


def eow_phon_ted_auxloss_distill(get_report=False):
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/distill_auxloss"

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
    train_dataset_tuples = {}
    for testset in ["train"]:
        train_dataset_tuples[testset] = build_test_dataset(
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

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )
    no_lm_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=None,
        beam_threshold=14,
    )

    from ...pytorch_networks.ctc.conformer_0106.i6modelsV2_VGG4LayerActFrontendV1_auxloss_v1_cfg import (
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
        max_dim_feat=8,
        num_repeat_feat=5,  # Jingjing style
    )
    report = {}
    chkpts = {}
    # for dim in [64, 128, 256, 384, 512, 768, 1024]:
    for dim in [384]:  # keep 128 for baseline
        # for layer_count in [4, 6, 9, 12, 16, 20]:
        for layer_count in [12]:
            loss_mapping = {
                4: [1, 2, 3],
                6: [1, 3, 5],
                9: [2, 5, 8],
                12: [3, 7, 11],
                16: [3, 7, 11, 15],
                20: [3, 7, 11, 15, 19],
            }
            """
                 64     128     256     384     512     768    1024
            4   17.068  12.383  9.321   8.273   8.032   7.862   7.703
            6   15.142  10.786  8.559   7.955   7.324   7.11    7.401
            9   13.315  9.705   7.862   7.204   6.968   6.968   6.973
            12  12.745  9.332   7.379   7.198   7.044   6.918   6.913
            16  12.136  8.893   7.478   7.313   7.138   6.902   6.743
            20  11.982  9.535   8.092   7.516   7.165   7.154   7.099
            """
            for drop in [0.0, 0.1, 0.2]:
                if drop < 0.2 and not dim == 384:
                    continue
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
                    out_features=dim,
                    activation=None,
                )
                model_config = ModelConfig(
                    feature_extraction_config=fe_config,
                    frontend_config=frontend_config,
                    specaug_config=specaug_config,
                    label_target_size=vocab_size_without_blank,
                    conformer_size=dim,
                    num_layers=layer_count,
                    num_heads=4,
                    ff_dim=4 * dim,
                    att_weights_dropout=drop,
                    conv_dropout=drop,
                    ff_dropout=drop,
                    mhsa_dropout=drop,
                    conv_kernel_size=31,
                    final_dropout=drop,
                    specauc_start_epoch=1,
                    module_list=["ff", "conv", "mhsa", "ff"],
                    module_scales=[0.5, 1.0, 1.0, 0.5],
                    aux_ctc_loss_layers=loss_mapping[layer_count],  # 4, 8, 12 when counting from 1
                    aux_ctc_loss_scales=(len(loss_mapping[layer_count]) - 1) * [0.3] + [1.0],
                )
                model_config_decoding = copy.deepcopy(model_config)
                model_config_decoding.aux_ctc_loss_scales = [
                    0.0,
                    0.0,
                    1.0,
                ]  # for decoding use result only of last layer

                network_module = "ctc.conformer_0106.i6modelsV2_VGG4LayerActFrontendV1_auxloss_v1"
                small = layer_count > 16 and dim > 768
                train_config = {
                    "optimizer": {
                        "class": "radam",
                        "epsilon": 1e-16,
                        "weight_decay": 1e-2,
                        "decoupled_weight_decay": True,
                    },
                    "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                    + list(np.linspace(5e-4, 5e-5, 110))
                    + list(np.linspace(5e-5, 1e-7, 30)),
                    #############
                    "batch_size": 180 * 16000 if not small else 90 * 16000,
                    "max_seq_length": {"audio_features": 35 * 16000},
                    "accum_grad_multiple_step": 1 if not small else 2,
                }
                train_args = {
                    "config": train_config,
                    "network_module": network_module,
                    "net_args": {"model_config_dict": asdict(model_config)},
                    "debug": False,
                }
                train_args_decoding = copy.deepcopy(train_args)
                train_args_decoding["net_args"] = {"model_config_dict": asdict(model_config_decoding)}

                results = {}
                training_name = (
                    prefix_name + "/" + network_module + f"_{layer_count}_{dim}"
                    if drop == 0.2
                    else prefix_name + "/" + network_module + f"_{layer_count}_{dim}_drop{drop}"
                )
                train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
                if dim >= 768 or layer_count > 12:
                    train_job.rqmt["gpu_mem"] = 24
                if dim == 384 and layer_count == 12:
                    PRETRAIN_CHECKPOINT_DISTILL_V1 = train_job.out_checkpoints[250]

                results = eval_model(
                    training_name=training_name,
                    train_job=train_job,
                    train_args=train_args_decoding,
                    train_data=train_data,
                    decoder_config=default_decoder_config,
                    dev_dataset_tuples=dev_dataset_tuples,
                    result_dict=results,
                    loss_name=f"ctc_loss_layer{layer_count}",
                    run_test=drop == 0.2,
                    test_dataset_tuples=test_dataset_tuples,
                    lm_scales=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                    prior_scales=[0.5, 0.7, 0.9],
                )
                generate_report(results=results, exp_name=training_name)
                report[training_name] = results
                del results
                for feat, time in [(2, 12)]:
                    if drop < 0.2 or not dim == 384:
                        continue
                    specaug_config_less = SpecaugConfig(
                        repeat_per_n_frames=time,
                        max_dim_time=20,
                        max_dim_feat=8,
                        num_repeat_feat=feat,  # Jingjing style
                    )
                    model_config = ModelConfig(
                        feature_extraction_config=fe_config,
                        frontend_config=frontend_config,
                        specaug_config=specaug_config_less,
                        label_target_size=vocab_size_without_blank,
                        conformer_size=dim,
                        num_layers=layer_count,
                        num_heads=4,
                        ff_dim=4 * dim,
                        att_weights_dropout=drop,
                        conv_dropout=drop,
                        ff_dropout=drop,
                        mhsa_dropout=drop,
                        conv_kernel_size=31,
                        final_dropout=drop,
                        specauc_start_epoch=1,
                        module_list=["ff", "conv", "mhsa", "ff"],
                        module_scales=[0.5, 1.0, 1.0, 0.5],
                        aux_ctc_loss_layers=loss_mapping[layer_count],  # 4, 8, 12 when counting from 1
                        aux_ctc_loss_scales=(len(loss_mapping[layer_count]) - 1) * [0.3] + [1.0],
                    )
                    model_config_decoding = copy.deepcopy(model_config)
                    model_config_decoding.aux_ctc_loss_scales = [
                        0.0,
                        0.0,
                        1.0,
                    ]  # for decoding use result only of last layer

                    network_module = "ctc.conformer_0106.i6modelsV2_VGG4LayerActFrontendV1_auxloss_v1"
                    small = layer_count > 16 and dim > 768
                    train_config = {
                        "optimizer": {
                            "class": "radam",
                            "epsilon": 1e-16,
                            "weight_decay": 1e-2,
                            "decoupled_weight_decay": True,
                        },
                        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                        + list(np.linspace(5e-4, 5e-5, 110))
                        + list(np.linspace(5e-5, 1e-7, 30)),
                        #############
                        "batch_size": 180 * 16000 if not small else 90 * 16000,
                        "max_seq_length": {"audio_features": 35 * 16000},
                        "accum_grad_multiple_step": 1 if not small else 2,
                    }
                    train_args = {
                        "config": train_config,
                        "network_module": network_module,
                        "net_args": {"model_config_dict": asdict(model_config)},
                        "debug": False,
                    }
                    train_args_decoding = copy.deepcopy(train_args)
                    train_args_decoding["net_args"] = {"model_config_dict": asdict(model_config_decoding)}

                    results = {}
                    training_name = (
                        prefix_name
                        + "/"
                        + network_module
                        + f"_{layer_count}_{dim}_drop{drop}_less_spec_f{feat}_t{time}"
                    )
                    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)

                    results = eval_model(
                        training_name=training_name,
                        train_job=train_job,
                        train_args=train_args_decoding,
                        train_data=train_data,
                        decoder_config=default_decoder_config,
                        dev_dataset_tuples=dev_dataset_tuples,
                        result_dict=results,
                        loss_name=f"ctc_loss_layer{layer_count}",
                    )
                    generate_report(results=results, exp_name=training_name)
                    report[training_name] = results
                    del results

    tk.register_report("reports/aux_size_report", partial(build_base_report, report), required=report)

    from ...pytorch_networks.ctc.conformer_0106.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
        ModelConfig as RelPosModelConfig,
        ConformerPosEmbConfig,
    )

    new_rep = {}
    # Best: 384, 1, 500, 16, 8
    for dim in [384]:
        for spec_start in [1]:
            for epochs in [250, 500]:
                for spec in [8, 16]:
                    for num_heads in [4, 8]:
                        for drop in [0.2, 0.1, 0.0]:
                            if drop < 0.2 and (not spec == 16 or not num_heads == 8 or not spec_start == 1):
                                continue
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
                                out_features=dim,
                                activation=None,
                            )
                            specaug_config_test = SpecaugConfig(
                                repeat_per_n_frames=25,
                                max_dim_time=20,
                                max_dim_feat=spec,
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
                            model_config_pos_enc = RelPosModelConfig(
                                feature_extraction_config=fe_config,
                                frontend_config=frontend_config,
                                specaug_config=specaug_config_test,
                                label_target_size=vocab_size_without_blank,
                                pos_emb_config=pos_emb_cfg,
                                conformer_size=dim,
                                num_layers=12,
                                num_heads=num_heads,
                                ff_dim=4 * dim,
                                att_weights_dropout=drop,
                                conv_dropout=drop,
                                ff_dropout=drop,
                                mhsa_dropout=drop,
                                mhsa_with_bias=True,
                                conv_kernel_size=31,
                                final_dropout=drop,
                                dropout_broadcast_axes=None,
                                specauc_start_epoch=spec_start,
                                module_list=["ff", "conv", "mhsa", "ff"],
                                module_scales=[0.5, 1.0, 1.0, 0.5],
                                aux_ctc_loss_layers=None,
                                aux_ctc_loss_scales=None,
                            )
                            network_module_pos_enc = "ctc.conformer_0106.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"
                            train_config = {
                                "optimizer": {
                                    "class": "radam",
                                    "epsilon": 1e-16,
                                    "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True,
                                },
                                "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                                + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                                + list(np.linspace(5e-5, 1e-7, 30)),
                                #############
                                "batch_size": 180 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                            }
                            train_args = {
                                "config": train_config,
                                "network_module": network_module_pos_enc,
                                "net_args": {"model_config_dict": asdict(model_config_pos_enc)},
                                "debug": True,
                            }
                            results = {}
                            if drop == 0.2:
                                training_name = (
                                    prefix_name
                                    + "/"
                                    + network_module_pos_enc
                                    + f"_{epochs}_{dim}_{num_heads}_{spec}_{spec_start}"
                                )
                            else:
                                training_name = (
                                    prefix_name
                                    + "/"
                                    + network_module_pos_enc
                                    + f"_{epochs}_{dim}_{num_heads}_{spec}_{spec_start}_drop{drop}"
                                )
                            train_job = training(
                                training_name, train_data, train_args, num_epochs=epochs, **default_returnn
                            )

                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                result_dict=results,
                                loss_name=f"ctc_loss_layer12",
                                specific_epoch=epochs,
                                prior_scales=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            new_rep[training_name] = results
                            chkpts[training_name] = train_job.out_checkpoints[250]
                            del results
                            if dim == 384 and spec_start == 1 and spec == 16 and num_heads == 8 and drop == 0.2:
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
                                    out_features=dim,
                                    activation=None,
                                )
                                specaug_config_test = SpecaugConfig(
                                    repeat_per_n_frames=25,
                                    max_dim_time=20,
                                    max_dim_feat=spec,
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
                                model_config_pos_enc = RelPosModelConfig(
                                    feature_extraction_config=fe_config,
                                    frontend_config=frontend_config,
                                    specaug_config=specaug_config_test,
                                    label_target_size=vocab_size_without_blank,
                                    pos_emb_config=pos_emb_cfg,
                                    conformer_size=dim,
                                    num_layers=12,
                                    num_heads=num_heads,
                                    ff_dim=4 * dim,
                                    att_weights_dropout=drop,
                                    conv_dropout=drop,
                                    ff_dropout=drop,
                                    mhsa_dropout=drop,
                                    mhsa_with_bias=True,
                                    conv_kernel_size=31,
                                    final_dropout=drop,
                                    dropout_broadcast_axes=None,
                                    specauc_start_epoch=spec_start,
                                    module_list=["ff", "conv", "mhsa", "ff"],
                                    module_scales=[0.5, 1.0, 1.0, 0.5],
                                    aux_ctc_loss_layers=None,
                                    aux_ctc_loss_scales=None,
                                )
                                network_module_pos_enc = (
                                    "ctc.conformer_0106.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"
                                )
                                train_config_no_wd = {
                                    "optimizer": {
                                        "class": "radam",
                                        "epsilon": 1e-16,
                                        # "weight_decay": 1e-2,
                                        # "decoupled_weight_decay": True,
                                    },
                                    "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                                    + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                                    + list(np.linspace(5e-5, 1e-7, 30)),
                                    #############
                                    "batch_size": 180 * 16000,
                                    "max_seq_length": {"audio_features": 35 * 16000},
                                    "accum_grad_multiple_step": 1,
                                }
                                train_args = {
                                    "config": train_config_no_wd,
                                    "network_module": network_module_pos_enc,
                                    "net_args": {"model_config_dict": asdict(model_config_pos_enc)},
                                    "debug": True,
                                }
                                results = {}
                                training_name = (
                                    prefix_name
                                    + "/"
                                    + network_module_pos_enc
                                    + f"_{epochs}_{dim}_{num_heads}_{spec}_{spec_start}_drop{drop}_no_wdecay"
                                )
                                train_job = training(
                                    training_name, train_data, train_args, num_epochs=epochs, **default_returnn
                                )
                                results = eval_model(
                                    training_name=training_name,
                                    train_job=train_job,
                                    train_args=train_args,
                                    train_data=train_data,
                                    decoder_config=default_decoder_config,
                                    dev_dataset_tuples=dev_dataset_tuples,
                                    result_dict=results,
                                    loss_name=f"ctc_loss_layer12",
                                    specific_epoch=epochs,
                                    prior_scales=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                                    lm_scales=[1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                    run_test=True,
                                    test_dataset_tuples=test_dataset_tuples,
                                )
                                generate_report(results=results, exp_name=training_name)
                                new_rep[training_name] = results
                                chkpts[training_name] = train_job.out_checkpoints[250]
                                del results

                            if dim == 384 and spec_start == 1 and spec == 16 and num_heads == 8 and drop == 0.2:
                                for feat, time in [(2, 12)]:
                                    specaug_config_less = SpecaugConfig(
                                        repeat_per_n_frames=time,
                                        max_dim_time=20,
                                        max_dim_feat=spec,
                                        num_repeat_feat=feat,
                                    )
                                    model_config_pos_enc_spec = RelPosModelConfig(
                                        feature_extraction_config=fe_config,
                                        frontend_config=frontend_config,
                                        specaug_config=specaug_config_less,
                                        label_target_size=vocab_size_without_blank,
                                        pos_emb_config=pos_emb_cfg,
                                        conformer_size=dim,
                                        num_layers=12,
                                        num_heads=num_heads,
                                        ff_dim=4 * dim,
                                        att_weights_dropout=drop,
                                        conv_dropout=drop,
                                        ff_dropout=drop,
                                        mhsa_dropout=drop,
                                        mhsa_with_bias=True,
                                        conv_kernel_size=31,
                                        final_dropout=drop,
                                        dropout_broadcast_axes=None,
                                        specauc_start_epoch=spec_start,
                                        module_list=["ff", "conv", "mhsa", "ff"],
                                        module_scales=[0.5, 1.0, 1.0, 0.5],
                                        aux_ctc_loss_layers=None,
                                        aux_ctc_loss_scales=None,
                                    )
                                    network_module_pos_enc = (
                                        "ctc.conformer_0106.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"
                                    )
                                    train_config = {
                                        "optimizer": {
                                            "class": "radam",
                                            "epsilon": 1e-16,
                                            "weight_decay": 1e-2,
                                            "decoupled_weight_decay": True,
                                        },
                                        "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                                        + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                                        + list(np.linspace(5e-5, 1e-7, 30)),
                                        #############
                                        "batch_size": 180 * 16000,
                                        "max_seq_length": {"audio_features": 35 * 16000},
                                        "accum_grad_multiple_step": 1,
                                    }
                                    train_args = {
                                        "config": train_config,
                                        "network_module": network_module_pos_enc,
                                        "net_args": {"model_config_dict": asdict(model_config_pos_enc_spec)},
                                        "debug": True,
                                    }
                                    results = {}
                                    training_name = (
                                        prefix_name
                                        + "/"
                                        + network_module_pos_enc
                                        + f"_{epochs}_{dim}_{num_heads}_{spec}_{spec_start}_drop{drop}_less_spec_f{feat}_t{time}"
                                    )
                                    train_job = training(
                                        training_name, train_data, train_args, num_epochs=epochs, **default_returnn
                                    )
                                    results = eval_model(
                                        training_name=training_name,
                                        train_job=train_job,
                                        train_args=train_args,
                                        train_data=train_data,
                                        decoder_config=default_decoder_config,
                                        dev_dataset_tuples=dev_dataset_tuples,
                                        result_dict=results,
                                        loss_name=f"ctc_loss_layer12",
                                        specific_epoch=epochs,
                                        prior_scales=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                                        lm_scales=[1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                        run_test=True,
                                        test_dataset_tuples=test_dataset_tuples,
                                    )
                                    generate_report(results=results, exp_name=training_name)
                                    new_rep[training_name] = results
                                    chkpts[training_name] = train_job.out_checkpoints[250]
                                    del results

                            if (
                                dim == 384
                                and spec_start == 1
                                and spec == 16
                                and num_heads == 8
                                and drop == 0.2
                                and False
                            ):
                                train_config = {
                                    "optimizer": {
                                        "class": "adamw",
                                        "epsilon": 1e-16,
                                        "weight_decay": 1e-2,
                                    },
                                    "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                                    + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                                    + list(np.linspace(5e-5, 1e-7, 30)),
                                    #############
                                    "batch_size": 180 * 16000,
                                    "max_seq_length": {"audio_features": 35 * 16000},
                                    "accum_grad_multiple_step": 1,
                                    "gradient_clip_norm": 1.0,
                                }
                                train_args = {
                                    "config": train_config,
                                    "network_module": network_module_pos_enc,
                                    "net_args": {"model_config_dict": asdict(model_config_pos_enc)},
                                    "debug": True,
                                    "use_speed_perturbation": True,
                                }
                                results = {}
                                training_name = (
                                    prefix_name
                                    + "/"
                                    + network_module_pos_enc
                                    + f"_better_params_{epochs}_{dim}_{num_heads}_{spec}_{spec_start}"
                                )
                                train_job = training(
                                    training_name, train_data, train_args, num_epochs=epochs, **default_returnn
                                )

                                results = eval_model(
                                    training_name=training_name,
                                    train_job=train_job,
                                    train_args=train_args,
                                    train_data=train_data,
                                    decoder_config=default_decoder_config,
                                    dev_dataset_tuples=dev_dataset_tuples,
                                    result_dict=results,
                                    loss_name=f"ctc_loss_layer12",
                                    specific_epoch=epochs,
                                    prior_scales=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                                    lm_scales=[1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                    run_test=True,
                                    test_dataset_tuples=test_dataset_tuples,
                                )
                                generate_report(results=results, exp_name=training_name)
                                new_rep[training_name] = results
                                del results

    tk.register_report("reports/pos_enc_report", partial(build_base_report, new_rep), required=new_rep)
    if get_report is True:
        rep = {}
        rep.update(new_rep)
        rep.update(report)
        return rep, chkpts
    from ...pytorch_networks.ctc.conformer_distill_1206.self_distill_conformer_auxloss_v2_cfg import (
        ModelConfig as StudentConfigV2,
        DistillConfig as TeacherConfigV2,
    )

    distill_module_v2 = "ctc.conformer_distill_1206.self_distill_conformer_auxloss_v2"
    distill_report = {}
    distill_report["baselines"] = {}
    for dim in []:
        for layer_count in [12]:
            for distill_scale in [0.25]:
                for T in [2]:
                    distill_report["baselines"][prefix_name + "/" + network_module + f"_{layer_count}_{dim}"] = report[
                        prefix_name + "/" + network_module + f"_{layer_count}_{dim}"
                    ]
                    frontend_config_teacher = VGG4LayerActFrontendV1Config_mod(
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
                    teacher_config = TeacherConfigV2(
                        frontend_config=frontend_config_teacher,
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
                        distill_scale=distill_scale,
                        ctc_scale=1 - distill_scale,
                        t=T,
                        spec_aug=False,  # TODO
                        module_list=["ff", "conv", "mhsa", "ff"],
                        module_scales=[0.5, 1.0, 1.0, 0.5],
                        aux_kd_loss_layers=[3, 7, 11],  # 4, 8, 12 when counting from 1
                        aux_kd_loss_scales=[0.0, 0.0, 1.0],
                        final_dropout=0.0,  # TODO
                        exp_targets=False,  # TODO
                        eliminate_blanks=False,
                    )
                    frontend_config_student = VGG4LayerActFrontendV1Config_mod(
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
                        out_features=128,
                        activation=None,
                    )

                    student_config = StudentConfigV2(
                        feature_extraction_config=fe_config,
                        frontend_config=frontend_config_student,
                        specaug_config=specaug_config,
                        label_target_size=vocab_size_without_blank,
                        conformer_size=dim,
                        num_layers=layer_count,
                        num_heads=4,
                        ff_dim=4 * dim,
                        att_weights_dropout=0.2,
                        conv_dropout=0.2,
                        ff_dropout=0.2,
                        mhsa_dropout=0.2,
                        conv_kernel_size=31,
                        final_dropout=0.2,
                        specauc_start_epoch=1,
                        module_list=["ff", "conv", "mhsa", "ff"],
                        module_scales=[0.5, 1.0, 1.0, 0.5],
                        aux_ctc_loss_layers=[3, 7, 11],  # 4, 8, 12 when counting from 1
                        aux_ctc_loss_scales=[0.3, 0.3, 1.0],
                    )

                    train_config_distill = {
                        "optimizer": {
                            "class": "radam",
                            "epsilon": 1e-16,
                            "weight_decay": 1e-2,
                            "decoupled_weight_decay": True,
                        },
                        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                        + list(np.linspace(5e-4, 5e-5, 110))
                        + list(np.linspace(5e-5, 1e-7, 30)),
                        #############
                        "batch_size": 180 * 16000,
                        "max_seq_length": {"audio_features": 35 * 16000},
                        "accum_grad_multiple_step": 1,
                    }
                    train_args_distill = {
                        "config": train_config_distill,
                        "network_module": distill_module_v2,
                        "net_args": {
                            "model_config_dict": asdict(student_config),
                            "distill_config_dict": asdict(teacher_config),
                        },
                        "debug": False,
                    }
                    train_args_distill["config"]["preload_from_files"] = {
                        "teacher": {
                            "filename": PRETRAIN_CHECKPOINT_DISTILL_V1,
                            "init_for_train": True,
                            "ignore_missing": False,
                            "prefix": "teacher.",
                            "ignore_params_prefixes": ["teacher.feature_extraction"],
                        }
                    }
                    model_config_decoding = copy.deepcopy(student_config)
                    model_config_decoding.aux_ctc_loss_scales = [
                        0.0,
                        0.0,
                        1.0,
                    ]  # for decoding use result only of last layer
                    train_args_distill_decoding = copy.deepcopy(train_args_distill)
                    train_args_distill_decoding["net_args"] = {
                        "model_config_dict": asdict(model_config_decoding),
                        "distill_config_dict": None,
                    }
                    del train_args_distill_decoding["config"]["preload_from_files"]

                    decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                    training_name = prefix_name + "/" + distill_module_v2 + f"_{layer_count}_{dim}_{distill_scale}_{T}"
                    train_job = training(
                        training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                    )
                    results = eval_model(
                        training_name=training_name,
                        train_job=train_job,
                        train_args=train_args_distill_decoding,
                        train_data=train_data,
                        decoder_config=default_decoder_config,
                        dev_dataset_tuples=dev_dataset_tuples,
                        specific_epoch=250,
                        decoder_module=decoder_module,
                        loss_name=f"ctc_loss_layer{layer_count}",
                    )
                    generate_report(results=results, exp_name=training_name)
                    distill_report[training_name] = results
                    del results

                    teacher_config_v2 = TeacherConfigV2(
                        frontend_config=frontend_config_teacher,
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
                        distill_scale=distill_scale,
                        ctc_scale=1 - distill_scale,
                        t=T,
                        spec_aug=False,  # TODO
                        module_list=["ff", "conv", "mhsa", "ff"],
                        module_scales=[0.5, 1.0, 1.0, 0.5],
                        aux_kd_loss_layers=[3, 7, 11],  # 4, 8, 12 when counting from 1
                        aux_kd_loss_scales=[0.3, 0.3, 1.0],
                        final_dropout=0.0,  # TODO
                        exp_targets=False,  # TODO
                        eliminate_blanks=True,
                    )
                    train_config_distill_v2 = {
                        "optimizer": {
                            "class": "radam",
                            "epsilon": 1e-16,
                            "weight_decay": 1e-2,
                            "decoupled_weight_decay": True,
                        },
                        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                        + list(np.linspace(5e-4, 5e-5, 110))
                        + list(np.linspace(5e-5, 1e-7, 30)),
                        #############
                        "batch_size": 180 * 16000,
                        "max_seq_length": {"audio_features": 35 * 16000},
                        "accum_grad_multiple_step": 1,
                    }
                    train_args_distill_v2 = {
                        "config": train_config_distill_v2,
                        "network_module": distill_module_v2,
                        "net_args": {
                            "model_config_dict": asdict(student_config),
                            "distill_config_dict": asdict(teacher_config_v2),
                        },
                        "debug": True,
                    }
                    train_args_distill_v2["config"]["preload_from_files"] = {
                        "teacher": {
                            "filename": PRETRAIN_CHECKPOINT_DISTILL_V1,
                            "init_for_train": True,
                            "ignore_missing": False,
                            "prefix": "teacher.",
                            "ignore_params_prefixes": ["teacher.feature_extraction"],
                        }
                    }
                    model_config_decoding_v2 = copy.deepcopy(student_config)
                    model_config_decoding_v2.aux_ctc_loss_scales = [
                        0.0,
                        0.0,
                        1.0,
                    ]  # for decoding use result only of last layer
                    train_args_distill_v2_decoding = copy.deepcopy(train_args_distill_v2)
                    train_args_distill_v2_decoding["net_args"] = {
                        "model_config_dict": asdict(model_config_decoding_v2),
                        "distill_config_dict": None,
                    }
                    del train_args_distill_v2_decoding["config"]["preload_from_files"]

                    training_name = (
                        prefix_name + "/" + distill_module_v2 + f"_{layer_count}_{dim}_{distill_scale}_{T}_elim_blanks"
                    )
                    train_job = training(
                        training_name, train_data, train_args_distill_v2, num_epochs=250, **default_returnn
                    )
                    results = eval_model(
                        training_name=training_name,
                        train_job=train_job,
                        train_args=train_args_distill_v2_decoding,
                        train_data=train_data,
                        decoder_config=default_decoder_config,
                        dev_dataset_tuples=dev_dataset_tuples,
                        specific_epoch=250,
                        decoder_module=decoder_module,
                        loss_name=f"ctc_loss_layer{layer_count}",
                    )
                    generate_report(results=results, exp_name=training_name)
                    # distill_report[training_name] = results
                    del results

    tk.register_report(
        "reports/aux_distill_report", partial(build_distill_report, distill_report), required=distill_report
    )
