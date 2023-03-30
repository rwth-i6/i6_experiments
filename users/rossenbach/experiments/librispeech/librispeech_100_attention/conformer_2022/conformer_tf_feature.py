import copy

import numpy

from sisyphus import tk
from typing import List

from i6_core.tools import CloneGitRepositoryJob
from i6_core.returnn import ReturnnConfig

from .pipeline import \
    build_training_datasets, build_test_dataset, training, search, get_best_checkpoint, search_single, get_average_checkpoint, get_average_checkpoint_v2

from .attention_asr_config import create_config, ConformerEncoderArgs, TransformerDecoderArgs, RNNDecoderArgs
from .base_config import get_lm_opts, apply_fairseq_init_to_conformer_encoder, get_lm_opts_new
from .feature_extraction_net import log10_net, log10_net_10ms, get_roll_augment_net, get_roll_augment_net_exponential


class LocalReport():
    def __init__(self, header):
        self.format_str = header
        self.values = {}


def freeze_all_train_only(returnn_config: ReturnnConfig, train_only: List[str]):
    """
    freeze everything except decoder "s" , inplace edit

    :param returnn_config:
    :return:
    """
    for layer_name, layer_content in returnn_config.config["network"].items():
        if layer_name.startswith("output"):
            for sub_layer_name, sub_layer_content in layer_content["unit"].items():
                if sub_layer_name not in train_only:
                    sub_layer_content["trainable"] = False
        else:
            layer_content["trainable"] = False


def conformer_tf_features():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root_datasets = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                                  commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443").out_repository
    prefix_name = "experiments/librispeech/librispeech_100_attention/conformer_2022"

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    training_datasets_speedperturbed = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True)
    training_datasets_speedperturbed_460 = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, ls_corpus_key="train-clean-460", bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, partition_epoch=9)
    # retrain dataset has curriculum options disabled
    training_datasets_speedperturbed_retrain = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=False)
    training_datasets_speedperturbed_460_retrain = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, ls_corpus_key="train-clean-460", bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, partition_epoch=9, use_curicculum=False)

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, returnn_python_exe=returnn_exe, returnn_root=returnn_root_datasets, output_path=prefix_name, use_raw_features=True)

    # ------------------------------------------------------------------------------------------------------------------- #

    conformer_enc_args = ConformerEncoderArgs(
        num_blocks=12, input_layer='lstm-6', att_num_heads=8, ff_dim=2048, enc_key_dim=512, conv_kernel_size=32,
        pos_enc='rel', dropout=0.1, att_dropout=0.1, l2=0.0001)

    conformer_enc_fixed_bn_args = ConformerEncoderArgs(
        num_blocks=12, input_layer='lstm-6', att_num_heads=8, ff_dim=2048, enc_key_dim=512, conv_kernel_size=32,
        pos_enc='rel', dropout=0.1, att_dropout=0.1, l2=0.0001)

    apply_fairseq_init_to_conformer_encoder(conformer_enc_args)
    conformer_enc_args.ctc_loss_scale = 1.0

    apply_fairseq_init_to_conformer_encoder(conformer_enc_fixed_bn_args)
    conformer_enc_fixed_bn_args.ctc_loss_scale = 1.0


    rnn_dec_args = RNNDecoderArgs()
    training_args = {}

    # LR scheduling
    training_args['const_lr'] = [50, 20]  # use const LR during pretraining
    training_args['wup_start_lr'] = 0.0002
    training_args['wup'] = 30
    training_args['with_staged_network'] = True
    training_args['speed_pert'] = True

    # overwrite BN params
    conformer_enc_args.batch_norm_opts = {
        'momentum': 0.1,
        'epsilon': 1e-3,
        'update_sample_only_in_training': False,
        'delay_sample_update': False
    }

    conformer_enc_fixed_bn_args.batch_norm_opts = {
        'momentum': 0.1,
        'epsilon': 1e-3,
        'update_sample_only_in_training': True,
        'delay_sample_update': True
    }

    transf_lm_opts = get_lm_opts()
    new_transf_lm_opts = get_lm_opts()

    # conformer round 2
    name = 'tf_feature_conformer_12l_lstm_1l_normal_v2'
    local_conformer_enc_args = copy.deepcopy(conformer_enc_args)
    local_training_args = copy.deepcopy(training_args)

    # pretraining
    local_training_args['pretrain_opts'] = {'variant': 3, "initial_batch_size": 18000*200}
    local_training_args['pretrain_reps'] = 5
    local_training_args['batch_size'] = 12000*200  # frames * samples per frame

    exp_prefix = prefix_name + "/" + name
    args = copy.deepcopy({**local_training_args, "encoder_args": local_conformer_enc_args, "decoder_args": rnn_dec_args})
    args['name'] = name
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="3f62155a08722310f51276792819b3c7c64ad356").out_repository

    header = "Name,best,,,,,last,,,,,average_4,,,\n,dev-clean,dev-other,test-clean,test-other,,dev-clean,dev-other,test-clean,test-other,,dev-clean,dev-other,test-clean,test-other\n"
    report = LocalReport(header=header)

    def run_exp_v2(ft_name, feature_extraction_net, datasets, train_args, search_args=None, search_extraction_net=None, report=None, num_epochs=250, extra_beam=None, full_lm=None):
        train_args = copy.deepcopy(train_args)
        search_args = search_args if search_args is not None else train_args
        search_extraction_net = search_extraction_net if search_extraction_net is not None else feature_extraction_net
        returnn_config = create_config(training_datasets=datasets, **train_args, feature_extraction_net=feature_extraction_net)

        returnn_search_config = create_config(training_datasets=datasets, **search_args, feature_extraction_net=search_extraction_net, is_recog=True)
        train_job = training(ft_name, returnn_config, returnn_exe, returnn_root, num_epochs=num_epochs)
        # average = get_average_checkpoint(train_job, returnn_exe=returnn_exe, returnn_root=returnn_root, num_average=4)
        average = get_average_checkpoint_v2(train_job, returnn_exe=returnn_exe, returnn_root=returnn_root, num_average=4)
        from i6_core.returnn.training import GetBestTFCheckpointJob
        best_checkpoint_job = GetBestTFCheckpointJob(
            train_job.out_model_dir,
            train_job.out_learning_rates,
            key="dev_score_output/output_prob",
            index=0)

        format_str_1, values_1 = search(ft_name + "/default_best", returnn_search_config, best_checkpoint_job.out_checkpoint, test_dataset_tuples, returnn_exe, returnn_root)
        format_str_2, values_2 = search(ft_name + "/default_last", returnn_search_config, train_job.out_checkpoints[num_epochs], test_dataset_tuples, returnn_exe, returnn_root)
        format_str_3, values_3 = search(ft_name + "/average_4", returnn_search_config, average, test_dataset_tuples, returnn_exe, returnn_root)

        if report:
            report.format_str = report.format_str + ft_name + "," + format_str_1 + ",," + format_str_2 + ",," + format_str_3 + "\n"
            report.values.update(values_1)
            report.values.update(values_2)
            report.values.update(values_3)

        ext_lm_search_args = copy.deepcopy(args)
        ext_lm_search_args["ext_lm_opts"] = transf_lm_opts

        beam_sizes = [12]
        if extra_beam:
            beam_sizes += extra_beam

        for beam_size in beam_sizes:
            for lm_scale in [0.36, 0.38, 0.4, 0.42, 0.44]:
                search_args = copy.deepcopy(ext_lm_search_args)
                search_args['ext_lm_opts']['lm_scale'] = lm_scale
                search_args["beam_size"] = beam_size
                returnn_config = create_config(training_datasets=datasets, **search_args, feature_extraction_net=search_extraction_net, is_recog=True)
                returnn_config.config["batch_size"] = 3000*200 if beam_size > 12 else 10000*200  # smaller size for recognition
                if beam_size > 12:
                    returnn_config.config["max_seqs"] = 10
                search_single(ft_name + "/default_last_ext_lm_%.2f_bs%i" % (lm_scale, beam_size),
                              returnn_config,
                              train_job.out_checkpoints[num_epochs],
                              test_dataset_tuples["dev-other"][0],
                              test_dataset_tuples["dev-other"][1],
                              returnn_exe,
                              returnn_root)
            if full_lm:
                for lm_scale in full_lm:
                    search_args = copy.deepcopy(ext_lm_search_args)
                    search_args['ext_lm_opts']['lm_scale'] = lm_scale
                    search_args["beam_size"] = beam_size
                    returnn_config = create_config(training_datasets=datasets, **search_args,
                                                   feature_extraction_net=search_extraction_net, is_recog=True)
                    returnn_config.config[
                        "batch_size"] = 3000 * 200 if beam_size > 12 else 10000 * 200  # smaller size for recognition
                    if beam_size > 12:
                        returnn_config.config["max_seqs"] = 10
                    name = ft_name + "/average_ext_lm_%.2f_bs%i" % (lm_scale, beam_size)
                    search(name.replace(".", "_"), returnn_config, average, test_dataset_tuples, returnn_exe,
                           returnn_root)

        return train_job

    def run_extra_lm(ft_name, train_job, extraction_net, datasets, search_args, filename, lm_scale=0.4):
        search_args = copy.deepcopy(search_args)
        search_args["ext_lm_opts"] = copy.deepcopy(transf_lm_opts)
        search_args["ext_lm_opts"]["preload_from_files"]["lm_model"]["filename"] = filename
        search_args['ext_lm_opts']['lm_scale'] = lm_scale
        returnn_config = create_config(training_datasets=datasets, **search_args, feature_extraction_net=extraction_net, is_recog=True)
        returnn_config.config["batch_size"] = 8000*200  # smaller size for recognition
        search_single(ft_name + "/default_last_ext_lm_%.2f" % lm_scale,
                      returnn_config,
                      train_job.out_checkpoints[250],
                      test_dataset_tuples["dev-other"][0],
                      test_dataset_tuples["dev-other"][1],
                      returnn_exe,
                      returnn_root)

    train_job_base = run_exp_v2(exp_prefix + "/" + "raw_log10", log10_net_10ms, datasets=training_datasets_speedperturbed, train_args=args, report=report)
    run_extra_lm(ft_name=exp_prefix + "/" + "raw_log10" + "/" + "new_lm", train_job=train_job_base, extraction_net=log10_net_10ms,
                 datasets=training_datasets_speedperturbed,
                 search_args=args,
                 filename="/work/asr4/rossenbach/sisyphus_work_folders/tts_asr_2021_work/i6_core/returnn/training/ReturnnTrainingJob.5ay7wpGaxNPg/output/models/epoch.017")
    from i6_experiments.users.rossenbach.experiments.librispeech.kazuki_lm.experiment import get_lm
    lm = get_lm("ls100_trafo24_bs3000_5ep_2kbpe")
    run_extra_lm(ft_name=exp_prefix + "/" + "raw_log10" + "/" + "new_lm_final", train_job=train_job_base, extraction_net=log10_net_10ms,
                 datasets=training_datasets_speedperturbed,
                 search_args=args,
                 filename=lm.train_job.out_checkpoints[100])

    args_bn_fix = copy.deepcopy(args)
    args_bn_fix["encoder_args"] = conformer_enc_fixed_bn_args
    train_job = run_exp_v2(exp_prefix + "/" + "raw_log10_bn_fix", log10_net_10ms, datasets=training_datasets_speedperturbed, train_args=args_bn_fix, report=report)


    #local_args = copy.deepcopy(args)
    #local_args["pretrain_opts"]["variant"] = 7
    #train_job = run_exp_v2(exp_prefix + "/" + "raw_log10_var7", log10_net_10ms, **local_args)

    args_retrain = copy.deepcopy(args)
    args_retrain["retrain_checkpoint"] = train_job_base.out_checkpoints[250]
    train_job = run_exp_v2(exp_prefix + "/" + "raw_log10_retrain", log10_net_10ms, datasets=training_datasets_speedperturbed_retrain, train_args=args_retrain, report=report, full_lm=[0.42, 0.44, 0.46])

    # args_retrain_newbob = copy.deepcopy(args)
    # args_retrain_newbob["retrain_checkpoint"] = train_job_base.out_checkpoints[250]
    # args_retrain_newbob["config_override"] = {"newbob_error_threshold": 0.0}
    # train_job = run_exp_v2(exp_prefix + "/" + "raw_log10_retrain_newbob_threshold_0", log10_net_10ms, datasets=training_datasets_speedperturbed_retrain, train_args=args_retrain_newbob)
 
    # # args_retrain_newbob_v2 = copy.deepcopy(args)
    # # args_retrain_newbob_v2["retrain_checkpoint"] = train_job_base.out_checkpoints[250]
    # # args_retrain_newbob_v2["config_override"] = {"newbob_relative_error_threshold": 0.0}
    # # train_job = run_exp_v2(exp_prefix + "/" + "raw_log10_retrain_newbob_v2_threshold_0", log10_net_10ms, datasets=training_datasets_speedperturbed_retrain, train_args=args_retrain_newbob_v2)

    # from i6_experiments.users.rossenbach.experiments.alignment_analysis_tts.synthetic_storage import synthetic_ogg_zip_data




    #### interspeech experiments ####

    train_job_base_460 = run_exp_v2(exp_prefix + "/" + "raw_log10_ls460", log10_net_10ms, datasets=training_datasets_speedperturbed_460, train_args=args, report=report)
    args_retrain = copy.deepcopy(args)
    args_retrain["retrain_checkpoint"] = train_job_base_460.out_checkpoints[250]
    train_job_base_460_retrain = run_exp_v2(exp_prefix + "/" + "raw_log10_ls460_retrain", log10_net_10ms, datasets=training_datasets_speedperturbed_460, train_args=args_retrain, report=report, full_lm=[0.4, 0.42, 0.44, 0.46])

    sat_gauss_pred = tk.Path("/u/hilmes/experiments/tts_new_sis/output/paper_nick/sat_gauss_pred/real_tags_corpus.xml.gz")
    sat_gauss_pred_scale11 = tk.Path("/u/hilmes/experiments/tts_new_sis/output/paper_nick/sat_gauss_scale_1.1/real_tags_corpus.xml.gz")
    sat_gauss_pred_random10_0025 = tk.Path("/u/hilmes/experiments/tts_new_sis/output/paper_nick/sat_gauss_random_1.0_0.025/real_tags_corpus.xml.gz")
    from i6_core.returnn.oggzip import BlissToOggZipJob

    def create_datasets(name, bliss_corpus, original_scale=3, partition_epoch=9):
        sat_gauss_pred_zip = BlissToOggZipJob(
            bliss_corpus=bliss_corpus,
            no_conversion=True,
            returnn_python_exe=returnn_exe,
            returnn_root=returnn_root
        )
        sat_gauss_pred_zip.add_alias(exp_prefix + "/" + name + "/make_zip")
        training_datasets_sat_gauss_pred = build_training_datasets(
            returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=True,
            synthetic_ogg_zip=sat_gauss_pred_zip.out_ogg_zip,
            partition_epoch=partition_epoch,
            original_scale=original_scale,
            synthetic_scale=1,
        )
        training_datasets_sat_gauss_pred_retrain = build_training_datasets(
            returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=False,
            synthetic_ogg_zip=sat_gauss_pred_zip.out_ogg_zip,
            partition_epoch=partition_epoch,
            original_scale=original_scale,
            synthetic_scale=1,
        )
        return training_datasets_sat_gauss_pred, training_datasets_sat_gauss_pred_retrain

    name = "sat_gauss_pred_broken"
    sat_gauss_pred_data = create_datasets(name, sat_gauss_pred)
    sat_gauss_pred_base = run_exp_v2(exp_prefix + "/" + "raw_log10_" + name, log10_net_10ms, datasets=sat_gauss_pred_data[0], train_args=args, report=report)

    name = "sat_gauss_pred"
    sat_gauss_pred_data = create_datasets(name, sat_gauss_pred, original_scale=1, partition_epoch=6)
    sat_gauss_pred_base = run_exp_v2(exp_prefix + "/" + "raw_log10_" + name, log10_net_10ms, datasets=sat_gauss_pred_data[0], train_args=args, report=report)

    name = "sat_gauss_pred_scale11"
    sat_gauss_pred_scale11_data = create_datasets(name, sat_gauss_pred_scale11, original_scale=1, partition_epoch=6)
    sat_gauss_pred_scale11_base = run_exp_v2(exp_prefix + "/" + "raw_log10_" + name, log10_net_10ms, datasets=sat_gauss_pred_scale11_data[0], train_args=args, report=report)

    name = "sat_gauss_pred_random10_0025"
    sat_gauss_pred_random10_0025_data = create_datasets(name, sat_gauss_pred_random10_0025, original_scale=1, partition_epoch=6)
    sat_gauss_pred_random10_0025_base = run_exp_v2(exp_prefix + "/" + "raw_log10_" + name, log10_net_10ms, datasets=sat_gauss_pred_random10_0025_data[0], train_args=args, report=report)

    for name, path in [
        ("sat_gauss_pred", "sat_gauss_pred"),
        #("sat_gauss_scale_11", "sat_gauss_scale_1.1"),
        #("sat_gauss_walk_10_0025", "sat_gauss_1.0_0.025"),
    ]:
        synth_path = tk.Path(f"/u/hilmes/experiments/tts_new_sis/output/paper_nick/ls360/{path}/real_tags_corpus.xml.gz")
        synth_data = create_datasets(name, synth_path)
        synth_base_train = run_exp_v2(exp_prefix + "/" + "combined_bn_" + name, log10_net_10ms,
                                         datasets=synth_data[0], train_args=args_bn_fix, report=report)
        args_retrain = copy.deepcopy(args_bn_fix)
        args_retrain["retrain_checkpoint"] = synth_base_train.out_checkpoints[250]
        synth_base_retrain = run_exp_v2(exp_prefix + "/" + "combined_bn_retrain_" + name, log10_net_10ms, datasets=synth_data[0], train_args=args_retrain, report=report, full_lm=[0.42, 0.44, 0.46])

        synth_path = tk.Path(f"/u/hilmes/experiments/tts_new_sis/output/paper_nick/ls360/{path}/real_tags_corpus.xml.gz")
        synth_data = create_datasets(name, synth_path)
        synth_base_train = run_exp_v2(exp_prefix + "/" + "combined_" + name, log10_net_10ms,
                                         datasets=synth_data[0], train_args=args, report=report)
        args_retrain = copy.deepcopy(args)
        args_retrain["retrain_checkpoint"] = synth_base_train.out_checkpoints[250]
        synth_base_retrain = run_exp_v2(exp_prefix + "/" + "combined_retrain_" + name, log10_net_10ms, datasets=synth_data[0], train_args=args_retrain, report=report)

    for name, path in [
        ("new_sat_gauss_pred", "pred"),
        ("new_sat_gauss_scale_11", "scale_1.1"),
        ("new_sat_gauss_walk_105_00125", "random_1.05_0.0125"),
        ("new_sat_gauss_walk_10_00125", "random_1.0_0.0125"),
        ("new_sat_gauss_walk_105_0025", "random_1.05_0.025"),
        ("new_sat_gauss_walk_10_0025", "random_1.0_0.025"),
        ("new_sat_gauss_walk_105_00375", "random_1.05_0.0375"),
        ("new_sat_gauss_walk_10_00375", "random_1.0_0.0375"),
        ("new_sat_gauss_walk_105_005", "random_1.05_0.05"),
        ("new_sat_gauss_walk_10_005", "random_1.0_0.05"),
    ]:
        synth_path = tk.Path(f"/u/hilmes/experiments/tts_new_sis/output/paper_nick/21_02_23_ls360/{path}_real_tags_corpus.xml.gz")
        synth_data = create_datasets(name, synth_path)
        synth_base_train = run_exp_v2(exp_prefix + "/" + "combined_bn_" + name, log10_net_10ms,
                                         datasets=synth_data[0], train_args=args_bn_fix, report=report)
        args_retrain = copy.deepcopy(args_bn_fix)
        args_retrain["retrain_checkpoint"] = synth_base_train.out_checkpoints[250]
        if "105" in name:
            synth_base_retrain = run_exp_v2(exp_prefix + "/" + "combined_bn_retrain_" + name, log10_net_10ms, datasets=synth_data[0], train_args=args_retrain, report=report)
        else:
            synth_base_retrain = run_exp_v2(exp_prefix + "/" + "combined_bn_retrain_" + name, log10_net_10ms, datasets=synth_data[0], train_args=args_retrain, report=report, full_lm=[0.42, 0.44, 0.46])


    # CTC 0.0
    name = "ctc_gauss_pred"
    synth_path = tk.Path(
        f"/u/hilmes/experiments/tts_new_sis/output/paper_nick/27_02_23_ls360/ctc_pred_real_tags_corpus.xml.gz")
    synth_data = create_datasets(name, synth_path)
    synth_base_train = run_exp_v2(exp_prefix + "/" + "combined_bn_" + name, log10_net_10ms,
                                  datasets=synth_data[0], train_args=args_bn_fix, report=report)
    args_retrain = copy.deepcopy(args_bn_fix)
    args_retrain["retrain_checkpoint"] = synth_base_train.out_checkpoints[250]
    synth_base_retrain = run_exp_v2(exp_prefix + "/" + "combined_bn_retrain_" + name, log10_net_10ms,
                                    datasets=synth_data[0], train_args=args_retrain, report=report)

    # Tacotron 2
    name = "tacotron_2"
    synth_path = tk.Path("/u/rossenbach/experiments/librispeech_tts/output/input_tts_experiments/threshold_fixed_phonemes/tts_outputs/librispeech-360_corpus.xml.gz")
    synth_data = create_datasets(name, synth_path)
    synth_base_train = run_exp_v2(exp_prefix + "/" + "combined_bn_" + name, log10_net_10ms,
                                  datasets=synth_data[0], train_args=args_bn_fix, report=report)
    args_retrain = copy.deepcopy(args_bn_fix)
    args_retrain["retrain_checkpoint"] = synth_base_train.out_checkpoints[250]
    synth_base_retrain = run_exp_v2(exp_prefix + "/" + "combined_bn_retrain_" + name, log10_net_10ms,
                                    datasets=synth_data[0], train_args=args_retrain, report=report)

    from i6_experiments.users.rossenbach.experiments.alignment_analysis_tts.storage import synthetic_ogg_zip_data

    training_datasets_speedperturbed_synth2 = build_training_datasets(
        returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=True,
        synthetic_ogg_zip=synthetic_ogg_zip_data["default_ctc_tts"],
        partition_epoch=9,
        original_scale=3,
        synthetic_scale=1,
    )

    training_datasets_speedperturbed_retrain_synth = build_training_datasets(
        returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=False,
        synthetic_ogg_zip=synthetic_ogg_zip_data["default_ctc_tts"],
        partition_epoch=18,
        original_scale=3,
        synthetic_scale=1,
    )

    training_datasets_speedperturbed_retrain_synth2 = build_training_datasets(
        returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=False,
        synthetic_ogg_zip=synthetic_ogg_zip_data["default_ctc_tts"],
        partition_epoch=9,
        original_scale=3,
        synthetic_scale=1,
    )
    
    # Tacotron 2
    name = "ctc_05"
    synth_base_train = run_exp_v2(exp_prefix + "/" + "combined_bn_" + name, log10_net_10ms,
                                  datasets=training_datasets_speedperturbed_synth2, train_args=args_bn_fix, report=report)
    args_retrain = copy.deepcopy(args_bn_fix)
    args_retrain["retrain_checkpoint"] = synth_base_train.out_checkpoints[250]
    synth_base_retrain = run_exp_v2(exp_prefix + "/" + "combined_bn_retrain_" + name, log10_net_10ms,
                                    datasets=training_datasets_speedperturbed_retrain_synth2, train_args=args_retrain, report=report)

    def run_freeze_evaluation(ft_name, feature_extraction_net, datasets, train_args, num_epochs, train_only, extra_beam=None, extra_scales=[]):
        train_args = copy.deepcopy(train_args)
        returnn_config = create_config(training_datasets=datasets, **train_args, feature_extraction_net=feature_extraction_net)

        freeze_all_train_only(returnn_config, train_only)



        train_job = training(ft_name, returnn_config, returnn_exe, returnn_root, num_epochs=num_epochs)

        search_single(ft_name + "/default_last_nolm",
                      returnn_config,
                      train_job.out_checkpoints[num_epochs],
                      test_dataset_tuples["dev-other"][0],
                      test_dataset_tuples["dev-other"][1],
                      returnn_exe,
                      returnn_root)


        ext_lm_search_args = copy.deepcopy(train_args)
        ext_lm_search_args["ext_lm_opts"] = transf_lm_opts
        beam_sizes = [12]
        if extra_beam:
            beam_sizes += extra_beam

        for beam_size in beam_sizes:
            for lm_scale in [0.4, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54] + extra_scales:
                search_args = copy.deepcopy(ext_lm_search_args)
                search_args['ext_lm_opts']['lm_scale'] = lm_scale
                search_args["beam_size"] = beam_size
                returnn_config = create_config(training_datasets=datasets, **search_args, feature_extraction_net=feature_extraction_net, is_recog=True)
                returnn_config.config["batch_size"] = 3000*200 if beam_size > 12 else 8000*200  # smaller size for recognition
                if beam_size > 12:
                    returnn_config.config["max_seqs"] = 10
                #returnn_config.config["beam_size"] = beam_size
                search_single(ft_name + "/default_last_ext_lm_%.2f_bs%i" % (lm_scale, beam_size),
                              returnn_config,
                              train_job.out_checkpoints[num_epochs],
                              test_dataset_tuples["dev-other"][0],
                              test_dataset_tuples["dev-other"][1],
                              returnn_exe,
                              returnn_root)



    # new ILM tests
    import numpy as np
    args_frozen = copy.deepcopy(args)
    args_frozen["lr"] = list(np.linspace(1e-4, 1e-5, 10))
    args_frozen["retrain_checkpoint"] = train_job_base.out_checkpoints[250]
    training_datasets_speedperturbed_random_synth = build_training_datasets(
        returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=False,
        synthetic_ogg_zip=synthetic_ogg_zip_data["default_ctc_tts_random460"],
        partition_epoch=10,
        original_scale=1,
        synthetic_scale=1,
    )

    training_datasets_speedperturbed_random_synth_only = build_training_datasets(
        returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=False,
        synthetic_ogg_zip=synthetic_ogg_zip_data["default_ctc_tts_random460"],
        partition_epoch=10,
        original_scale=0,
        synthetic_scale=1,
    )

    training_datasets_speedperturbed_random_lex = build_training_datasets(
        returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=False,
        synthetic_ogg_zip=synthetic_ogg_zip_data["default_ctc_tts_random_lex10k"],
        partition_epoch=10,
        original_scale=1,
        synthetic_scale=1,
    )
    
    training_datasets_speedperturbed_random_lexf3 = build_training_datasets(
        returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=False,
        synthetic_ogg_zip=synthetic_ogg_zip_data["default_ctc_tts_random_lex10k"],
        partition_epoch=10,
        original_scale=1,
        synthetic_scale=3,
    )

    training_datasets_speedperturbed_random_lexf10 = build_training_datasets(
        returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=False,
        synthetic_ogg_zip=synthetic_ogg_zip_data["default_ctc_tts_random_lex10k"],
        partition_epoch=10,
        original_scale=1,
        synthetic_scale=10,
    )
    
    training_datasets_speedperturbed_non_random_synth = build_training_datasets(
        returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=False,
        synthetic_ogg_zip=synthetic_ogg_zip_data["default_ctc_tts"],
        partition_epoch=10,
        original_scale=1,
        synthetic_scale=1,
    )

    train_job = run_freeze_evaluation(
        exp_prefix + "/" + "raw_log10_finetune_synth-sequencerandom_ep10_onlyS",
        log10_net_10ms, datasets=training_datasets_speedperturbed_random_synth, train_args=args_frozen, num_epochs=10, train_only=["s"])

    train_job = run_freeze_evaluation(
        exp_prefix + "/" + "raw_log10_finetune_synth-sequencerandom_ep10_only_S+readout_in",
        log10_net_10ms, datasets=training_datasets_speedperturbed_random_synth, train_args=args_frozen, num_epochs=10, train_only=["s", "readout_in"])

    train_job = run_freeze_evaluation(
        exp_prefix + "/" + "raw_log10_finetune_synth-sequencerandom_ep10_only_S+readout_in+softmax",
        log10_net_10ms, datasets=training_datasets_speedperturbed_random_synth, train_args=args_frozen, num_epochs=10, train_only=["s", "readout_in", "output_prob"])

    train_job = run_freeze_evaluation(
        exp_prefix + "/" + "raw_log10_finetune_synth-non-random_ep10_only_S+readout_in+softmax",
        log10_net_10ms, datasets=training_datasets_speedperturbed_non_random_synth, train_args=args_frozen, num_epochs=10,
        train_only=["s", "readout_in", "output_prob"])

    train_job = run_freeze_evaluation(
        exp_prefix + "/" + "raw_log10_finetune_onlysynth-sequencerandom_ep10_only_S+readout_in",
        log10_net_10ms, datasets=training_datasets_speedperturbed_random_synth_only, train_args=args_frozen, num_epochs=10, train_only=["s", "readout_in"])


    train_job = run_freeze_evaluation(
        exp_prefix + "/" + "raw_log10_finetune_synth-randomlex_ep10_only_S+readout_in+softmax",
        log10_net_10ms, datasets=training_datasets_speedperturbed_random_lex, train_args=args_frozen, num_epochs=10, train_only=["s", "readout_in", "output_prob"])

    train_job = run_freeze_evaluation(
        exp_prefix + "/" + "raw_log10_finetune_synth-randomlexf3_ep10_only_S+readout_in+softmax",
        log10_net_10ms, datasets=training_datasets_speedperturbed_random_lexf3, train_args=args_frozen, num_epochs=10, train_only=["s", "readout_in", "output_prob"], extra_scales=[0.34, 0.36, 0.38])

    train_job = run_freeze_evaluation(
        exp_prefix + "/" + "raw_log10_finetune_synth-randomlexf10_ep10_only_S+readout_in+softmax",
        log10_net_10ms, datasets=training_datasets_speedperturbed_random_lexf10, train_args=args_frozen, num_epochs=10, train_only=["s", "readout_in", "output_prob"])
    #train_job = run_exp_v2(
    #    exp_prefix + "/" + "raw_log10_finetune_synth-sequencerandom_ep10_onlyS",
    #    log10_net_10ms, datasets=training_datasets_speedperturbed_random_synth, train_args=args_frozen, num_epochs=10)
    #train_job = run_exp_v2(
    #    exp_prefix + "/" + "raw_log10_finetune_synth-sequencerandom_ep10_onlyS_large_batch_XYZ",
    #    log10_net_10ms, datasets=training_datasets_speedperturbed_random_synth, train_args=args_frozen)

    #train_job = run_exp_v2(
    #    exp_prefix + "/" + "raw_log10_finetune_synth-sequencerandom_ep10_onlyS_large_batch",
    #    log10_net_10ms, datasets=training_datasets_speedperturbed_random_synth, train_args=args_frozen, num_epochs=20)

    #train_job = run_freeze_evaluation(
    #    exp_prefix + "/" + "raw_log10_finetune_synth-sequencerandom_ep10_onlyS_large_batch",
    #    log10_net_10ms, datasets=training_datasets_speedperturbed_random_synth, train_args=args_frozen, num_epochs=20)

    # train_job_synth_base = run_exp_v2(exp_prefix + "/" + "raw_log10_synthtest", log10_net_10ms, datasets=training_datasets_speedperturbed_synth, train_args=args)
    train_job_synretrain = run_exp_v2(exp_prefix + "/" + "raw_log10_retrain_synthtest", log10_net_10ms, datasets=training_datasets_speedperturbed_retrain_synth, train_args=args_retrain, extra_beam=[1, 36])

    args_frozen_retrain = copy.deepcopy(args_frozen)
    args_frozen_retrain["retrain_checkpoint"] = train_job_synretrain.out_checkpoints[250]

    train_job = run_freeze_evaluation(
        exp_prefix + "/" + "raw_log10_finetune_synth-retrain-sequencerandom_ep10_only_S+readout_in+softmax",
        log10_net_10ms, datasets=training_datasets_speedperturbed_random_synth, train_args=args_frozen_retrain, num_epochs=10, train_only=["s", "readout_in", "output_prob"], extra_beam=[1, 36])
    
    train_job = run_freeze_evaluation(
        exp_prefix + "/" + "raw_log10_finetune_synth-retrain-sequencerandom_only_ep10_only_S+readout_in+softmax",
        log10_net_10ms, datasets=training_datasets_speedperturbed_random_synth_only, train_args=args_frozen_retrain, num_epochs=10, train_only=["s", "readout_in", "output_prob"], extra_beam=[1, 36])

    # train_job = run_exp_v2(exp_prefix + "/" + "raw_log10_retrain_synthtest_newbob_threshold_0", log10_net_10ms, datasets=training_datasets_speedperturbed_retrain_synth, train_args=args_retrain_newbob)
    # # Had issues
    # # train_job = run_exp_v2(exp_prefix + "/" + "raw_log10_retrain_synthtest_newbob_v2_threshold_0", log10_net_10ms, datasets=training_datasets_speedperturbed_retrain_synth, train_args=args_retrain_newbob_v2)

    # args_retrain = copy.deepcopy(args)
    # args_retrain["retrain_checkpoint"] = train_job_synth_base.out_checkpoints[250]
    # run_exp_v2(exp_prefix + "/" + "raw_log10_synthtest_retrain", log10_net_10ms, datasets=training_datasets_speedperturbed_synth, train_args=args_retrain)

    # args_retrain = copy.deepcopy(args)
    # args_retrain["retrain_checkpoint"] = train_job_synth_base.out_checkpoints[250]
    # args_retrain["config_override"] = {"newbob_error_threshold": 0.0}
    # run_exp_v2(exp_prefix + "/" + "raw_log10_synthtest_retrain_newbob_threshold_0", log10_net_10ms, datasets=training_datasets_speedperturbed_synth, train_args=args_retrain)

    # local_args = copy.deepcopy(args)
    # local_args["config_override"] = {"newbob_error_threshold": 0.0}
    # run_exp_v2(exp_prefix + "/" + "raw_log10_newbob_threshold_0", log10_net_10ms, datasets=training_datasets_speedperturbed, train_args=local_args)

    # # Dataaug experiment
    # local_args = copy.deepcopy(args)

    # log10_net_10ms_aug = get_roll_augment_net(min_val=0.125, max_val=0.25)
    # base = run_exp_v2(exp_prefix + "/" + "raw_log10_roll_aug_18_12", log10_net_10ms_aug, datasets=training_datasets_speedperturbed, train_args=local_args, search_extraction_net=log10_net_10ms)
    # args_retrain = copy.deepcopy(args)
    # args_retrain["retrain_checkpoint"] = base.out_checkpoints[250]
    # run_exp_v2(exp_prefix + "/" + "raw_log10_roll_aug_18_12_retrain", log10_net_10ms_aug, datasets=training_datasets_speedperturbed, train_args=args_retrain, search_extraction_net=log10_net_10ms)

    # log10_net_10ms_aug = get_roll_augment_net(min_val=0.0625, max_val=0.25)
    # base = run_exp_v2(exp_prefix + "/" + "raw_log10_roll_aug_24_12", log10_net_10ms_aug, datasets=training_datasets_speedperturbed, train_args=local_args, search_extraction_net=log10_net_10ms)
    # args_retrain = copy.deepcopy(args)
    # args_retrain["retrain_checkpoint"] = base.out_checkpoints[250]
    # run_exp_v2(exp_prefix + "/" + "raw_log10_roll_aug_24_12_retrain", log10_net_10ms_aug, datasets=training_datasets_speedperturbed, train_args=args_retrain, search_extraction_net=log10_net_10ms)

    # log10_net_10ms_aug = get_roll_augment_net(min_val=0.0625, max_val=0.25, broadcast_scale=False)
    # base = run_exp_v2(exp_prefix + "/" + "raw_log10_roll_aug_24_12_no_broadcast", log10_net_10ms_aug, datasets=training_datasets_speedperturbed, train_args=local_args, search_extraction_net=log10_net_10ms)
    # args_retrain = copy.deepcopy(args)
    # args_retrain["retrain_checkpoint"] = base.out_checkpoints[250]
    # run_exp_v2(exp_prefix + "/" + "raw_log10_roll_aug_24_12_no_broadcast_retrain", log10_net_10ms_aug, datasets=training_datasets_speedperturbed, train_args=args_retrain, search_extraction_net=log10_net_10ms)

    # Did not converge
    # log10_net_10ms_aug = get_roll_augment_net_exponential(min_val=-120, max_val=-12)
    # base = run_exp_v2(exp_prefix + "/" + "raw_log10_roll_aug_120_12_expo", log10_net_10ms_aug, datasets=training_datasets_speedperturbed, train_args=local_args, search_extraction_net=log10_net_10ms)
    # args_retrain = copy.deepcopy(args)
    # args_retrain["retrain_checkpoint"] = base.out_checkpoints[250]
    # run_exp_v2(exp_prefix + "/" + "raw_log10_roll_aug_120_12_expo_retrain", log10_net_10ms_aug, datasets=training_datasets_speedperturbed, train_args=args_retrain, search_extraction_net=log10_net_10ms)


    tk.register_report(prefix_name + "/report.csv", values=report.values, template=report.format_str)
