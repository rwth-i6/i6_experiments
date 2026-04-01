from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...tune_eval import tune_and_evaluate_helper
from ...storage import add_ctc_model, get_lm_model, NeuralLM
from ...report import tune_and_evalue_report
from ... import PACKAGE


def bpe_ls960_2603_synthcompare():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_bpe_synthcompare"

    train_settings_laplace4 = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.4000",
    )

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    
    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyDecoderConfig

    def greedy_search_helper(
            training_name: str,
            asr_model: ASRModel,
            decoder_config: GreedyDecoderConfig
        ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_greedy"
        search_jobs, wers = search(
            search_name,
            forward_config={},
            asr_model=asr_model,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            **default_returnn,
        )

    from ...pytorch_networks.ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config, ConformerPosEmbConfig

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
        max_dim_feat=16,  # classic style
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

    # Try to do like returnn frontend
    posemb_config = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    for BPE_SIZE in [512]:

        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe_laplace4 = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            settings=train_settings_laplace4,
            use_postfix=False,
        )
        label_datastream_bpe = cast(LabelDatastream, train_data_bpe_laplace4.datastreams["labels"])
        vocab_size_without_blank = label_datastream_bpe.vocab_size

        dev_dataset_tuples = {}
        for testset in ["dev-clean", "dev-other"]:
            dev_dataset_tuples[testset] = build_test_dataset(
                dataset_key=testset,
                settings=train_settings_laplace4,
            )

        test_dataset_tuples = {}
        for testset in ["test-clean", "test-other"]:
            test_dataset_tuples[testset] = build_test_dataset(
                dataset_key=testset,
                settings=train_settings_laplace4,
            )

        model_config = ModelConfig(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            pos_emb_config=posemb_config,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=8,
            ff_dim=2048,
            att_weights_dropout=0.1,
            conv_dropout=0.1,
            ff_dropout=0.1,
            mhsa_dropout=0.1,
            mhsa_with_bias=True,
            conv_kernel_size=31,
            final_dropout=0.1,
            specauc_start_epoch=11,
            dropout_broadcast_axes=None, # No dropout broadcast yet to properly compare
            module_list=["ff", "conv", "mhsa", "ff"],
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=None,
            aux_ctc_loss_scales=None,
        )

        # New test more closer to other setup
        train_config_24gbgpu_amp_radam = {
            "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            "learning_rates":list(np.linspace(5e-5, 5e-4, 480)) + list(
            np.linspace(5e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40)),
            #############
            "batch_size": 480 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "gradient_clip_norm": 1.0,
            "torch_amp_options": {"dtype": "bfloat16"},
        }

        network_module = "ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
        train_args_radam = {
            "config": train_config_24gbgpu_amp_radam,
            "post_config": {"num_workers_per_gpu": 8},
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "use_speed_perturbation": True,
            "debug": False,
        }

        training_name = prefix_name + "/" + str(
            BPE_SIZE) + "/" + network_module + ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_radam_lr5e-4"
        train_job = training(training_name, train_data_bpe_laplace4, train_args_radam, num_epochs=1000,
                             **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        #train_job.move_to_hpc = True
        #train_job.hold()

        asr_model = prepare_asr_model(
            training_name, train_job, train_args_radam, with_prior=True, datasets=train_data_bpe_laplace4,
            get_specific_checkpoint=1000
        )

        greedy_decoder_config = GreedyDecoderConfig(
            returnn_vocab=label_datastream_bpe.vocab,
        )
        greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

        decoder_config = DecoderConfig(
            lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=BPE_SIZE),
            returnn_vocab=label_datastream_bpe.vocab,
            beam_size=1024,
            beam_size_token=16,  # makes it much faster
            arpa_lm=arpa_4gram_lm,
            beam_threshold=14,
        )

        tune_and_evaluate_helper(
            training_name=training_name + "/first_shot_tuning",
            asr_model=asr_model,
            base_decoder_config=decoder_config,
            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
            test_dataset_tuples=test_dataset_tuples,
            default_returnn=default_returnn,
            lm_scales=[0.5, 1.0, 1.5, 2.0, 2.5],
            prior_scales=[0.0, 0.2, 0.4, 0.6],
            use_gpu=False,
            extra_rqmt=None,
        )

        from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v5 import DecoderConfig as BeamSearchDecoderConfigv5
        from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v5 import DecoderExtraConfig
        trafo_24x768 : NeuralLM = get_lm_model("bpe%i_trafo24x768_5ep" % BPE_SIZE)
        beam_search_decoder_config_v5_24lm = BeamSearchDecoderConfigv5(
            returnn_vocab=label_datastream_bpe.vocab,
            beam_size=12,
            lm_model_args = trafo_24x768.net_args,
            lm_checkpoint = trafo_24x768.checkpoint,
            lm_module = "pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v2.Model",
            lm_states_need_label_axis=True,
        )
        decoder_unhashed_config_v5 = DecoderExtraConfig(
            lm_package=PACKAGE,
        )

        tune_and_evaluate_helper(
            training_name + "/trafolm_24x768_tune1",
            asr_model=asr_model,
            base_decoder_config=beam_search_decoder_config_v5_24lm,
            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
            test_dataset_tuples={},
            unhashed_decoder_config=decoder_unhashed_config_v5,
            extra_forward_config={"batch_size": 100 * 16000},
            lm_scales=[0.5,1.0,1.5], prior_scales=[0.0,0.1,0.2], use_gpu=True,
            default_returnn=default_returnn
        )

        tune_and_evaluate_helper(
            training_name + "/trafolm_24x768_tune2",
            asr_model=asr_model,
            base_decoder_config=beam_search_decoder_config_v5_24lm,
            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
            test_dataset_tuples={},
            unhashed_decoder_config=decoder_unhashed_config_v5,
            extra_forward_config={"batch_size": 100 * 16000},
            lm_scales=[0.8,1.0,1.2], prior_scales=[0.0,0.1,0.2,0.3], use_gpu=True,
            default_returnn=default_returnn
        )

        # try expansion trainings
        from ...storage import get_synthetic_data

        for num_datasets in [100]:
            synth_oggs = []
            for i in range(num_datasets):
                _, synth_ogg = get_synthetic_data("glowtts460_lm_data_%i" % i)
                synth_oggs.append(synth_ogg)

            train_settings_ = copy.deepcopy(train_settings_laplace4)
            train_settings_.train_partition_epoch = num_datasets
            train_settings_.train_seq_ordering = "laplace:.2000"

            train_data_bpe_synth_mix = build_bpe_training_datasets(
                prefix=prefix_name,
                librispeech_key="train-other-960",
                bpe_size=BPE_SIZE,
                settings=train_settings_,
                use_postfix=False,
                extra_train_ogg_zips=synth_oggs,
                data_repetition_factors=[num_datasets // 10] + [1] * num_datasets
            )

            from i6_core.tools.git import CloneGitRepositoryJob
            MINI_RETURNN_ROOT_LOW_MEM = CloneGitRepositoryJob(
                "https://github.com/JackTemaki/MiniReturnn", commit="86e785017cd9393f43238931ae1c96d873e1f0c2"
            ).out_repository.copy()
            MINI_RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"
            low_mem_returnn = {
                "returnn_exe": RETURNN_EXE,
                "returnn_root": MINI_RETURNN_ROOT_LOW_MEM,
            }

            train_args_radam_nodevtrain = copy.deepcopy(train_args_radam)
            train_args_radam_nodevtrain["exclude_devtrain"] = True
            train_args_radam_nodevtrain["post_config"] = {"num_workers_per_gpu": 4}

            training_name = prefix_name + "/" + str(
                BPE_SIZE) + "/" + network_module + ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_radam_lr5e-4/synth_mix_%i" % num_datasets
            train_job = training(training_name, train_data_bpe_synth_mix, train_args_radam_nodevtrain, num_epochs=1000,
                                 **low_mem_returnn)
            train_job.rqmt["gpu_mem"] = 48
            train_job.rqmt["mem"] = 120
            train_job.rqmt["cpu"] = 10
            train_job.move_to_hpc = True
            train_job.hold()

            asr_model = prepare_asr_model(
                training_name, train_job, train_args_radam_nodevtrain, with_prior=True, datasets=train_data_bpe_laplace4,
                get_specific_checkpoint=1000
            )
            greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

            tune_and_evaluate_helper(
                training_name=training_name + "/first_shot_tuning",
                asr_model=asr_model,
                base_decoder_config=decoder_config,
                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                test_dataset_tuples=test_dataset_tuples,
                default_returnn=default_returnn,
                lm_scales=[0.5, 1.0, 1.5, 2.0, 2.5],
                prior_scales=[0.0, 0.2, 0.4, 0.6],
                use_gpu=False,
                extra_rqmt=None,
            )
            tune_and_evaluate_helper(
                training_name + "/trafolm_24x768_tune2",
                asr_model=asr_model,
                base_decoder_config=beam_search_decoder_config_v5_24lm,
                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                test_dataset_tuples={},
                unhashed_decoder_config=decoder_unhashed_config_v5,
                extra_forward_config={"batch_size": 100 * 16000},
                lm_scales=[0.8, 1.0, 1.2], prior_scales=[0.0, 0.1, 0.2, 0.3], use_gpu=True,
                default_returnn=default_returnn
            )


