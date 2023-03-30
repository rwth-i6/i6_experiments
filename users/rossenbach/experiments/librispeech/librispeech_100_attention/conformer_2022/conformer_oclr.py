import copy
import numpy as np
from sisyphus import tk

from .pipeline import \
    build_training_datasets, build_test_dataset, training, search, get_best_checkpoint, search_single, get_average_checkpoint_v2

from .attention_asr_config import create_config, ConformerEncoderArgs, TransformerDecoderArgs, RNNDecoderArgs
from .base_config import apply_fairseq_init_to_conformer_encoder, get_lm_opts_new
from .feature_extraction_net import log10_net_10ms

from .default_tools import RETURNN_EXE, RETURNN_ROOT


class LocalReport():
    def __init__(self, header):
        self.format_str = header
        self.values = {}


def conformer_oclr():
    prefix_name = "experiments/librispeech/librispeech_100_attention/conformer_2022"

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    training_datasets_speedperturbed = build_training_datasets(RETURNN_EXE, RETURNN_ROOT, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True)

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, returnn_python_exe=RETURNN_EXE, returnn_root=RETURNN_ROOT, output_path=prefix_name, use_raw_features=True)

    # ------------------------------------------------------------------------------------------------------------------- #

    conformer_enc_args = ConformerEncoderArgs(
        num_blocks=12, input_layer='conv-6-fixed', att_num_heads=8, ff_dim=2048, enc_key_dim=512, conv_kernel_size=32,
        pos_enc='rel', dropout=0.1, att_dropout=0.1, l2=0.0001)

    apply_fairseq_init_to_conformer_encoder(conformer_enc_args)
    conformer_enc_args.ctc_loss_scale = 1.0

    conformer_enc_args.batch_norm_opts = {
        'momentum': 0.1,
        'epsilon': 1e-3,
        'update_sample_only_in_training': True,
        'delay_sample_update': True
    }

    rnn_dec_args = RNNDecoderArgs()
    training_args = {}

    training_args["encoder_args"] = conformer_enc_args
    training_args["decoder_args"] = rnn_dec_args
    training_args["pretrain_opts"] = {"variant": 3, "initial_batch_size": 22500 * 160}
    training_args["pretrain_reps"] = 5
    training_args["batch_size"] = 15000 * 160  # frames * samples per frame
    training_args['with_staged_network'] = True
    training_args['speed_pert'] = True
    transf_lm_opts = get_lm_opts_new()
    
    header = "Name,best,,,,,last,,,,,average_4,,,\n,dev-clean,dev-other,test-clean,test-other,,dev-clean,dev-other,test-clean,test-other,,dev-clean,dev-other,test-clean,test-other\n"
    report = LocalReport(header=header)


    def run_exp_v2(ft_name, datasets, train_args, search_args=None, search_extraction_net=None, report=None, num_epochs=250, extra_beam=None, full_lm=None):
        train_args = copy.deepcopy(train_args)
        search_args = search_args if search_args is not None else train_args
        search_extraction_net = search_extraction_net if search_extraction_net is not None else log10_net_10ms
        returnn_config = create_config(training_datasets=datasets, **train_args, feature_extraction_net=log10_net_10ms)

        returnn_search_config = create_config(training_datasets=datasets, **search_args, feature_extraction_net=search_extraction_net, is_recog=True)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, RETURNN_ROOT, num_epochs=num_epochs)
        average = get_average_checkpoint_v2(train_job, returnn_exe=RETURNN_EXE, returnn_root=RETURNN_ROOT, num_average=4)

        #format_str_1, values_1 = search(ft_name + "/default_best", returnn_search_config, best_checkpoint_job.out_checkpoint, test_dataset_tuples, returnn_exe, returnn_root)
        #format_str_2, values_2 = search(ft_name + "/default_last", returnn_search_config, train_job.out_checkpoints[num_epochs], test_dataset_tuples, returnn_exe, returnn_root)
        format_str_3, values_3 = search(ft_name + "/average_4", returnn_search_config, average, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        if report:
            # report.format_str = report.format_str + ft_name + "," + format_str_1 + ",," + format_str_2 + ",," + format_str_3 + "\n"
            report.format_str = report.format_str + ft_name + "," + format_str_3 + "\n"
            #report.values.update(values_1)
            #report.values.update(values_2)
            report.values.update(values_3)

        ext_lm_search_args = copy.deepcopy(search_args)
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
                              returnn_exe=RETURNN_EXE,
                              returnn_root=RETURNN_ROOT)
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
                    search(name.replace(".", "_"), returnn_config, average, test_dataset_tuples, RETURNN_EXE,
                           RETURNN_ROOT)

        return train_job

    prefix_group = prefix_name + "/" + "tf_feature_conformer_12l_lstm_1l_oclrv1"
    exp_args = copy.deepcopy(training_args)
    exp_args["learning_rates"] = list(np.linspace(8e-5, 8e-4, 90)) + list(np.linspace(8e-4, 8e-5, 90)) + list(np.linspace(8e-5, 1e-6, 20))
    train_job_base = run_exp_v2(prefix_group + "/" + "epoch_based",
                                datasets=training_datasets_speedperturbed, train_args=exp_args, report=report, num_epochs=200)

