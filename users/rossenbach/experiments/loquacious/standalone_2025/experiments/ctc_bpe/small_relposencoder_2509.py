from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset, build_short_dev_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT, KENLM_BINARY_PATH
from ...pipeline import training, prepare_asr_model, search, ASRModel, evaluate_all
from ...storage import get_lm_model
from ... import PACKAGE
from ...storage import get_synthetic_data


def bpe_small_trial():
    prefix_name = "experiments/loquacious/standalone_2025/ctc_bpe_small_trial"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )
    
    train_settings_laplace4 = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.4000",
    )

    short_dev_dataset_tuples = {
        "dev.short": build_short_dev_dataset(train_settings)
    }

    dev_dataset_tuples = {}
    for testset in ["dev.commonvoice", "dev.librispeech", "dev.voxpopuli", "dev.yodas"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test.commonvoice", "test.librispeech", "test.voxpopuli", "test.yodas"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v2 import DecoderConfig
    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyDecoderConfig



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

    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 240)) + list(
                np.linspace(5e-4, 5e-5, 240)) + list(np.linspace(5e-5, 1e-7, 20)),
        #############
        "batch_size": 240 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    network_module = "ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
    global_train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "use_speed_perturbation": True,
        "debug": True,
    }

    def greedy_search_helper(
            training_name: str,
            asr_model: ASRModel,
            dev_dataset_tuples,
            test_dataset_tuples,
            decoder_config: GreedyDecoderConfig
        ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_greedy"
        dev_search_jobs, dev_wers, dev_ctms = search(
            search_name,
            forward_config={},
            asr_model=asr_model,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples},
            **default_returnn,
        )
        test_search_jobs, test_wers, test_ctms = search(
            search_name,
            forward_config={},
            asr_model=asr_model,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**test_dataset_tuples},
            **default_returnn,
        )
        evaluate_all(search_name, dev_ctms, test_ctms)




    from i6_core.lm.kenlm import CreateBinaryLMJob
    # THIS LM IS WRONG
    arpa_4gram_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=tk.Path("/work/asr4/rossenbach/corpora/loquacious/LoquaciousAdditionalResources/4gram-pruned.arpa.gz"),
        kenlm_binary_folder=KENLM_BINARY_PATH
    )
    arpa_4gram_lm = arpa_4gram_binary_lm_job.out_lm

    arpa_4gram_binary_lm_v2_job = CreateBinaryLMJob(
        arpa_lm=tk.Path("/work/asr4/rossenbach/corpora/loquacious/LoquaciousAdditionalResources/4gram-pruned-test2.arpa.gz"),
        kenlm_binary_folder=KENLM_BINARY_PATH
    )
    arpa_4gram_lm_v2 = arpa_4gram_binary_lm_v2_job.out_lm

    arpa_4gram_ls_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=tk.Path("/work/asr4/rossenbach/corpora/loquacious/LoquaciousAdditionalResources/4gram-0-0-1-1-pluslibrispeech.gz"),
        kenlm_binary_folder=KENLM_BINARY_PATH
    )
    arpa_4gram_ls_lm = arpa_4gram_ls_binary_lm_job.out_lm

    # CMUDict only
    arpa_4gram_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=tk.Path("/work/asr4/rossenbach/corpora/loquacious/LoquaciousAdditionalResources/4gram-pruned.cmuonly.arpa.gz"),
        kenlm_binary_folder=KENLM_BINARY_PATH
    )
    arpa_4gram_cmuonly_lm = arpa_4gram_binary_lm_job.out_lm

    # unpruned
    arpa_4gram_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=tk.Path("/work/asr4/rossenbach/corpora/loquacious/LoquaciousAdditionalResources/4gram-unpruned.arpa.gz"),
        kenlm_binary_folder=KENLM_BINARY_PATH
    )
    arpa_4gram_unpruned_lm = arpa_4gram_binary_lm_job.out_lm

    # for BPE_SIZE in [0, 128, 512]:
    for subsampling in [4, 6]:
        BPE_SIZES = [128, 512, 1024] if subsampling == 6 else [128, 256, 512, 1000, 1024]
        for BPE_SIZE in BPE_SIZES:

            # build the training datasets object containing train, cv, dev-train and the extern_data dict
            train_data_bpe = build_bpe_training_datasets(
                prefix=prefix_name,
                loquacious_key="train.small",
                bpe_size=BPE_SIZE,
                settings=train_settings,
                use_postfix=False,
            )
            train_data_bpe_laplace4 = build_bpe_training_datasets(
                prefix=prefix_name,
                loquacious_key="train.small",
                bpe_size=BPE_SIZE,
                settings=train_settings_laplace4,
                use_postfix=False,
            )
            label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
            vocab_size_without_blank = label_datastream_bpe.vocab_size



            frontend_config_sub = copy.deepcopy(frontend_config)
            frontend_config_sub.pool1_kernel_size = (subsampling//2, 1)
            frontend_config_sub.pool1_stride= (subsampling//2, 1)

            model_config = ModelConfig(
                feature_extraction_config=fe_config,
                frontend_config=frontend_config_sub,
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

            greedy_decoder_config = GreedyDecoderConfig(
                returnn_vocab=label_datastream_bpe.vocab,
            )



            default_decoder_config_bpe = DecoderConfig(
                lexicon=get_text_lexicon(prefix=prefix_name, loquacious_key="train.small", bpe_size=BPE_SIZE),
                returnn_vocab=label_datastream_bpe.vocab,
                beam_size=1024,
                beam_size_token=16,  # makes it much faster
                arpa_lm=arpa_4gram_lm,
                beam_threshold=14,
            )

            # with v2 LM
            default_decoder_config_bpe_v2lm = DecoderConfig(
                lexicon=get_text_lexicon(prefix=prefix_name, loquacious_key="train.small", bpe_size=BPE_SIZE),
                returnn_vocab=label_datastream_bpe.vocab,
                beam_size=1024,
                beam_size_token=16,  # makes it much faster
                arpa_lm=arpa_4gram_lm_v2,
                beam_threshold=14,
            )

            default_decoder_config_bpe_unpruned = DecoderConfig(
                lexicon=get_text_lexicon(prefix=prefix_name, loquacious_key="train.small", bpe_size=BPE_SIZE),
                returnn_vocab=label_datastream_bpe.vocab,
                beam_size=1024,
                beam_size_token=16,  # makes it much faster
                arpa_lm=arpa_4gram_unpruned_lm,
                beam_threshold=14,
            )

            decoder_config_bpe_large_search = DecoderConfig(
                lexicon=get_text_lexicon(prefix=prefix_name, loquacious_key="train.small", bpe_size=BPE_SIZE),
                returnn_vocab=label_datastream_bpe.vocab,
                beam_size=2048,
                beam_size_token=32,  # makes it much faster
                arpa_lm=arpa_4gram_lm,
                beam_threshold=16,
            )

            decoder_config_bpe_ls4gram = DecoderConfig(
                lexicon=get_text_lexicon(prefix=prefix_name, loquacious_key="train.small", bpe_size=BPE_SIZE),
                returnn_vocab=label_datastream_bpe.vocab,
                beam_size=1024,
                beam_size_token=16,  # makes it much faster
                arpa_lm=arpa_4gram_ls_lm,
                beam_threshold=14,
            )

            decoder_config_bpe_cmudict_only = DecoderConfig(
                lexicon=get_text_lexicon(prefix=prefix_name, loquacious_key="train.small", bpe_size=BPE_SIZE, variant=10),
                returnn_vocab=label_datastream_bpe.vocab,
                beam_size=1024,
                beam_size_token=16,  # makes it much faster
                arpa_lm=arpa_4gram_cmuonly_lm,
                beam_threshold=14,
            )

            train_args = copy.deepcopy(global_train_args)
            train_args["net_args"] = {"model_config_dict": asdict(model_config)}

            training_name = prefix_name + "/" + str(BPE_SIZE) + "/" + network_module + f".512dim_sub{subsampling}_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch"
            train_job = training(training_name, train_data_bpe, train_args, num_epochs=500, **default_returnn)
            train_job.rqmt["gpu_mem"] = 48
            #if BPE_SIZE not in [128, 512, 1024]:
            #    train_job.hold()
            #    train_job.move_to_hpc = True

            asr_model = prepare_asr_model(
                training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe, get_specific_checkpoint=500
            )
            base_asr_model = asr_model

            greedy_search_helper(training_name, dev_dataset_tuples=dev_dataset_tuples, test_dataset_tuples=test_dataset_tuples, asr_model=asr_model, decoder_config=greedy_decoder_config)
            from ...tune_eval import tune_and_evaluate_helper
            default_returnn = {
                "returnn_exe": RETURNN_EXE,
                "returnn_root": MINI_RETURNN_ROOT,
            }
            tune_and_evaluate_helper(
                training_name=training_name + "/first_shot_tuning",
                asr_model=asr_model,
                base_decoder_config=default_decoder_config_bpe,
                dev_dataset_tuples=short_dev_dataset_tuples,
                test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                default_returnn=default_returnn,
                lm_scales=[0.5,1.0,1.5,2.0,2.5],
                prior_scales=[0.0, 0.1, 0.2],
            )
            tune_and_evaluate_helper(
                training_name=training_name + "/second_shot_tuning_v2lm",
                asr_model=asr_model,
                base_decoder_config=default_decoder_config_bpe_v2lm,
                dev_dataset_tuples=short_dev_dataset_tuples,
                test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                default_returnn=default_returnn,
                lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                prior_scales=[0.1, 0.2, 0.3],
            )
            if BPE_SIZE == 128:
                # 4-gram DECODING
                tune_and_evaluate_helper(
                    training_name=training_name + "/first_shot_tuning_loq+lslm",
                    asr_model=asr_model,
                    base_decoder_config=decoder_config_bpe_ls4gram,
                    dev_dataset_tuples=short_dev_dataset_tuples,
                    test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                    default_returnn=default_returnn,
                    lm_scales=[0.5,1.0,1.5,2.0,2.5],
                    prior_scales=[0.0, 0.1, 0.2],
                )
                tune_and_evaluate_helper(
                    training_name=training_name + "/second_shot_tuning",
                    asr_model=asr_model,
                    base_decoder_config=default_decoder_config_bpe,
                    dev_dataset_tuples=short_dev_dataset_tuples,
                    test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                    default_returnn=default_returnn,
                    lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                    prior_scales=[0.1, 0.2, 0.3],
                )
                tune_and_evaluate_helper(
                    training_name=training_name + "/second_shot_tuning_large_search",
                    asr_model=asr_model,
                    base_decoder_config=decoder_config_bpe_large_search,
                    dev_dataset_tuples=short_dev_dataset_tuples,
                    test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                    default_returnn=default_returnn,
                    lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                    prior_scales=[0.1, 0.2, 0.3],
                )

                # unpruned
                tune_and_evaluate_helper(
                    training_name=training_name + "/second_shot_tuning_unpruned_lm",
                    asr_model=asr_model,
                    base_decoder_config=default_decoder_config_bpe_unpruned,
                    dev_dataset_tuples=short_dev_dataset_tuples,
                    test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                    default_returnn=default_returnn,
                    lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                    prior_scales=[0.1, 0.2, 0.3],
                )

                # cmudict only
                tune_and_evaluate_helper(
                    training_name=training_name + "/second_shot_tuning_cmudict_only",
                    asr_model=asr_model,
                    base_decoder_config=decoder_config_bpe_cmudict_only,
                    dev_dataset_tuples=short_dev_dataset_tuples,
                    test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                    default_returnn=default_returnn,
                    lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                    prior_scales=[0.1, 0.2, 0.3],
                )

                tune_and_evaluate_helper(
                    training_name=training_name + "/lexicon_no_lm",
                    asr_model=asr_model,
                    base_decoder_config=default_decoder_config_bpe,
                    dev_dataset_tuples=short_dev_dataset_tuples,
                    test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                    default_returnn=default_returnn,
                    lm_scales=[0.0],
                    prior_scales=[0.0],
                )

                # NeuralLM Decoding
                from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v5 import \
                    DecoderConfig as BeamSearchDecoderConfigv5
                from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v5 import DecoderExtraConfig
                trafo_24x768: NeuralLM = get_lm_model("bpe%i_trafo24x768_5ep" % BPE_SIZE)

                beam_search_decoder_config_v5_24lm = BeamSearchDecoderConfigv5(
                    returnn_vocab=label_datastream_bpe.vocab,
                    beam_size=10,
                    lm_model_args = trafo_24x768.net_args,
                    lm_checkpoint = trafo_24x768.checkpoint,
                    lm_module = "pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v2.Model",
                    lm_states_need_label_axis=True,
                )
                decoder_unhashed_config_v5 = DecoderExtraConfig(
                    lm_package=PACKAGE,
                )
                tune_and_evaluate_helper(
                    training_name + "/trafolm_32x768",
                    asr_model=asr_model,
                    base_decoder_config=beam_search_decoder_config_v5_24lm,
                    dev_dataset_tuples=short_dev_dataset_tuples,
                    test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                    unhashed_decoder_config=decoder_unhashed_config_v5,
                    extra_forward_config={"batch_size": 200 * 16000},
                    lm_scales=[0.5, 0.75, 1.0, 1.25, 1.5], prior_scales=[0.2, 0.3, 0.4],
                    # lm_scales=[1.0], prior_scales=[0.3],
                    use_gpu=True,
                    default_returnn=default_returnn,
                )

            if BPE_SIZE == 128 and subsampling == 4:

                # extra evaluations
                for dev_key in dev_dataset_tuples.keys():
                    name = dev_key.split(".")[1]
                    test_key = f"test.{name}"
                    tune_and_evaluate_helper(
                        training_name=training_name + "/second_shot_tuning_on_%s" % name,
                        asr_model=asr_model,
                        base_decoder_config=default_decoder_config_bpe,
                        dev_dataset_tuples={dev_key: dev_dataset_tuples[dev_key]},
                        test_dataset_tuples={test_key: test_dataset_tuples[test_key]},
                        default_returnn=default_returnn,
                        lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                        prior_scales=[0.1, 0.2, 0.3],
                        evaluate_independent=True,
                    )


                train_args_nospeed = copy.deepcopy(train_args)
                train_args_nospeed.pop("use_speed_perturbation")
                training_name = prefix_name + "/" + str(
                    BPE_SIZE) + "/" + network_module + f".512dim_sub{subsampling}_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch_nospeedpert"
                train_job = training(training_name, train_data_bpe, train_args_nospeed, num_epochs=500, **default_returnn)
                train_job.rqmt["gpu_mem"] = 48
                asr_model = prepare_asr_model(
                    training_name, train_job, train_args_nospeed, with_prior=True, datasets=train_data_bpe,
                    get_specific_checkpoint=500
                )
                greedy_search_helper(training_name, dev_dataset_tuples=dev_dataset_tuples,
                                     test_dataset_tuples=test_dataset_tuples, asr_model=asr_model,
                                     decoder_config=greedy_decoder_config)
                tune_and_evaluate_helper(
                    training_name=training_name + "/second_shot_tuning",
                    asr_model=asr_model,
                    base_decoder_config=default_decoder_config_bpe,
                    dev_dataset_tuples=short_dev_dataset_tuples,
                    test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                    default_returnn=default_returnn,
                    lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                    prior_scales=[0.1, 0.2, 0.3],
                )

                tune_and_evaluate_helper(
                    training_name=training_name + "/lexicon_no_lm",
                    asr_model=asr_model,
                    base_decoder_config=default_decoder_config_bpe,
                    dev_dataset_tuples=short_dev_dataset_tuples,
                    test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                    default_returnn=default_returnn,
                    lm_scales=[0.0],
                    prior_scales=[0.0],
                )


                # NO specaug
                train_args_nospecaug = copy.deepcopy(train_args_nospeed)
                model_config_no_specaug = copy.deepcopy(model_config)
                model_config_no_specaug.specauc_start_epoch = 99999  # ugly hack
                train_args_nospecaug["net_args"] = {"model_config_dict": asdict(model_config_no_specaug)}

                training_name = prefix_name + "/" + str(
                    BPE_SIZE) + "/" + network_module + f".512dim_sub{subsampling}_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch_nospeedpert_nospecaug"
                train_job = training(training_name, train_data_bpe, train_args_nospecaug, num_epochs=500, **default_returnn)
                train_job.rqmt["gpu_mem"] = 48
                asr_model = prepare_asr_model(
                    training_name, train_job, train_args_nospecaug, with_prior=True, datasets=train_data_bpe,
                    get_specific_checkpoint=500
                )
                greedy_search_helper(training_name, dev_dataset_tuples=dev_dataset_tuples,
                                     test_dataset_tuples=test_dataset_tuples, asr_model=asr_model,
                                     decoder_config=greedy_decoder_config)
                tune_and_evaluate_helper(
                    training_name=training_name + "/second_shot_tuning",
                    asr_model=asr_model,
                    base_decoder_config=default_decoder_config_bpe,
                    dev_dataset_tuples=short_dev_dataset_tuples,
                    test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                    default_returnn=default_returnn,
                    lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                    prior_scales=[0.1, 0.2, 0.3],
                )



            if BPE_SIZE == 128 and subsampling == 4:
                # combined training
                synthetic_data_keys_with_segments = [
                    ("train.small_grad_tts.grad_tts_v3_ext_dur_glowttsv2align_400eps_noise_0.80_train-small-shuffled", None, 1),
                    ("train.small_ar_tts.tacotron2_decoding.tacotron2_decoding_v3_glowttsv2align_400eps_train-small-shuffled", None, 1),
                    ("train.small_grad_tts.grad_tts_v3_ext_dur_glowttsv2align_400eps_noise_0.80_train-medium-wo-small-shuffled", None, 9),
                    ("train.small_ar_tts.tacotron2_decoding.tacotron2_decoding_v3_glowttsv2align_400eps_train-medium-wo-small-shuffled", None, 9),
                ]

                for syn_data_key, segments, factor in synthetic_data_keys_with_segments:
                    syn_bliss, zip = get_synthetic_data(syn_data_key)
                    train_settings_ = copy.deepcopy(train_settings)
                    train_settings_.train_partition_epoch = 5 * (factor*2)

                    train_data_syn = build_bpe_training_datasets(
                        prefix=prefix_name + "/synthetic/" + syn_data_key,
                        loquacious_key="train.small",
                        bpe_size=BPE_SIZE,
                        settings=train_settings_,
                        use_postfix=False,
                        extra_train_ogg_zips=[zip],
                        data_repetition_factors=[factor, 1],
                        explicit_devtrain=True if ("medium-wo-small" in syn_data_key and segments is None) else False,
                        extra_train_bliss=[syn_bliss] if ("medium-wo-small" in syn_data_key and segments is None) else None,
                    )

                    sub_name = "full"

                    train_args_ = copy.deepcopy(global_train_args)
                    train_args_["net_args"] = {"model_config_dict": asdict(model_config)}
                    train_args_["config"]["import_model_train_epoch1"] = base_asr_model.checkpoint  # only get checkpoint, rest should be identical

                    training_name = prefix_name + "/" + str(
                        BPE_SIZE) + "/" + network_module + f".512dim_sub{subsampling}_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch_resume" + f"/synthetic_combined/{syn_data_key}_{sub_name}"
                    train_job = training(training_name, train_data_syn, train_args_, num_epochs=500, **default_returnn)
                    train_job.rqmt["gpu_mem"] = 48
                    train_job.rqmt["mem"] = 120
                    train_job.hold()
                    train_job.move_to_hpc = True

                    asr_model = prepare_asr_model(
                        training_name, train_job, train_args_, with_prior=True, datasets=train_data_syn,
                        get_specific_checkpoint=500
                    )

                    tune_and_evaluate_helper(
                        training_name=training_name + "/second_shot_tuning",
                        asr_model=asr_model,
                        base_decoder_config=default_decoder_config_bpe,
                        dev_dataset_tuples=short_dev_dataset_tuples,
                        test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                        default_returnn=default_returnn,
                        lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                        prior_scales=[0.1, 0.2, 0.3],
                    )
