import copy

from sisyphus import tk
from typing import List

from i6_core.tools import CloneGitRepositoryJob
from i6_core.returnn import ReturnnConfig

from .pipeline import \
    build_training_datasets, build_test_dataset, training, search, search_single, get_average_checkpoint_v2

from .attention_asr_config import create_config, ConformerEncoderArgs, RNNDecoderArgs
from .base_config import get_lm_opts, apply_fairseq_init_to_conformer_encoder
from .feature_extraction_net import log10_net_10ms


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


def conformer_duration_variability():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root_datasets = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                                  commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443").out_repository
    prefix_name = "experiments/librispeech/librispeech_100_attention/conformer_2022"

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    training_datasets_speedperturbed = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True)
    training_datasets_speedperturbed_460 = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, ls_corpus_key="train-clean-460", bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, partition_epoch=9)
    # retrain dataset has curriculum options disabled
    training_datasets_speedperturbed_retrain = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=False)

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, returnn_python_exe=returnn_exe, returnn_root=returnn_root_datasets, output_path=prefix_name, use_raw_features=True)

    # ------------------------------------------------------------------------------------------------------------------- #

    conformer_enc_args = ConformerEncoderArgs(
        num_blocks=12, input_layer='lstm-6', att_num_heads=8, ff_dim=2048, enc_key_dim=512, conv_kernel_size=32,
        pos_enc='rel', dropout=0.1, att_dropout=0.1, l2=0.0001)

    apply_fairseq_init_to_conformer_encoder(conformer_enc_args)
    conformer_enc_args.ctc_loss_scale = 1.0

    rnn_dec_args = RNNDecoderArgs()
    training_args = {}

    # LR scheduling
    training_args['const_lr'] = [50, 20]  # use const LR during pretraining
    training_args['wup_start_lr'] = 0.0002
    training_args['wup'] = 30
    training_args['with_staged_network'] = True
    training_args['speed_pert'] = True

    # overwrite BN params
    # those are not exactly correct as both should be True,
    # but make compatible to older setups, also performance does not seem to change
    conformer_enc_args.batch_norm_opts = {
        'momentum': 0.1,
        'epsilon': 1e-3,
        'update_sample_only_in_training': False,
        'delay_sample_update': False
    }

    transf_lm_opts = get_lm_opts()

    name = 'tf_feature_conformer_12l_lstm_1l_normal_v2'
    local_conformer_enc_args = copy.deepcopy(conformer_enc_args)
    local_training_args = copy.deepcopy(training_args)

    # pretraining
    local_training_args['pretrain_opts'] = {'variant': 3, "initial_batch_size": 225 * 16000}
    local_training_args['pretrain_reps'] = 5
    local_training_args['batch_size'] = 150 * 16000  # batch size in seconds


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


    ##### Baseline experiment, Table 4 Row 1 #####
    train_job_base = run_exp_v2(exp_prefix + "/" + "raw_log10", log10_net_10ms, datasets=training_datasets_speedperturbed, train_args=args, report=report)

    args_retrain = copy.deepcopy(args)
    args_retrain["retrain_checkpoint"] = train_job_base.out_checkpoints[250]
    run_exp_v2(exp_prefix + "/" + "raw_log10_retrain", log10_net_10ms, datasets=training_datasets_speedperturbed_retrain, train_args=args_retrain, report=report, full_lm=[0.42, 0.44, 0.46])


    ##### Oracle experiment, Table 4 Row 8 #####
    train_job_base_460 = run_exp_v2(exp_prefix + "/" + "raw_log10_ls460", log10_net_10ms, datasets=training_datasets_speedperturbed_460, train_args=args, report=report)
    args_retrain = copy.deepcopy(args)
    args_retrain["retrain_checkpoint"] = train_job_base_460.out_checkpoints[250]
    run_exp_v2(exp_prefix + "/" + "raw_log10_ls460_retrain", log10_net_10ms, datasets=training_datasets_speedperturbed_460, train_args=args_retrain, report=report, full_lm=[0.4, 0.42, 0.44, 0.46])


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


    ##### Synthetic data experiments, Table 4 Rows 2-7 in the same order #####
    for name, path in [
        ("new_sat_gauss_pred", "pred"),
        ("new_sat_gauss_scale_11", "scale_1.1"),
        ("new_sat_gauss_walk_10_00125", "random_1.0_0.0125"),
        ("new_sat_gauss_walk_10_0025", "random_1.0_0.025"),
        ("new_sat_gauss_walk_10_00375", "random_1.0_0.0375"),
        ("new_sat_gauss_walk_10_005", "random_1.0_0.05"),
    ]:
        synth_path = tk.Path(f"/u/hilmes/experiments/tts_new_sis/output/paper_nick/21_02_23_ls360/{path}_real_tags_corpus.xml.gz")
        synth_data = create_datasets(name, synth_path)
        synth_base_train = run_exp_v2(exp_prefix + "/" + "combined_" + name, log10_net_10ms,
                                         datasets=synth_data[0], train_args=args, report=report)
        args_retrain = copy.deepcopy(args)
        args_retrain["retrain_checkpoint"] = synth_base_train.out_checkpoints[250]
        run_exp_v2(exp_prefix + "/" + "combined_retrain_" + name, log10_net_10ms, datasets=synth_data[0], train_args=args_retrain, report=report, full_lm=[0.42, 0.44, 0.46])

    tk.register_report(prefix_name + "/report.csv", values=report.values, template=report.format_str)
