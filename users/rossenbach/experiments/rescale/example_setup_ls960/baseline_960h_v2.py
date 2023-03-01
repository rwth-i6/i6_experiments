"""
This is the minimal baseline ls960 Conformer Attention setup from February 2023

All external LM related code was removed to keep the setup as simple as possible
"""
import copy, os

from .attention_asr_config import \
    create_config, ConformerEncoderArgs, TransformerDecoderArgs, RNNDecoderArgs, ConformerDecoderArgs
from .additional_config import apply_fairseq_init_to_conformer, apply_fairseq_init_to_transformer_decoder
from .data import build_training_datasets, build_test_dataset
from .default_tools import RETURNN_ROOT, RETURNN_CPU_EXE
from .feature_extraction_net import log10_net_10ms
from .pipeline import training, search, get_average_checkpoint, get_best_checkpoint, search_single

train_jobs_map = {}  # dict[str, ReturnnTrainJob]
train_job_avg_ckpt = {}
train_job_best_epoch = {}

BPE_10K = 10000


def run_conformer_baseline():
    prefix_name = "experiments/rescale/example_setup_ls960/"

    def get_test_dataset_tuples(bpe_size):
        test_dataset_tuples = {}
        for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
            test_dataset_tuples[testset] = build_test_dataset(
                testset, bpe_size=bpe_size,
            )
        return test_dataset_tuples

    def run_train(exp_name, train_args, train_data, feature_extraction_net, num_epochs, recog_epochs, **kwargs):
        exp_prefix = os.path.join(prefix_name, exp_name)
        returnn_config = create_config(
            training_datasets=train_data,
            **train_args,
            feature_extraction_net=feature_extraction_net,
            recog_epochs=recog_epochs,
        )
        train_job = training(exp_prefix, returnn_config, RETURNN_CPU_EXE, RETURNN_ROOT, num_epochs=num_epochs)
        return train_job

    def run_single_search(
            exp_name, train_data, search_args, checkpoint, feature_extraction_net, recog_dataset, recog_ref,
            mem_rqmt=8, time_rqmt=4, **kwargs):

        exp_prefix = os.path.join(prefix_name, exp_name)
        returnn_search_config = create_config(
            training_datasets=train_data,
            **search_args,
            feature_extraction_net=feature_extraction_net,
            is_recog=True,
        )
        search_single(
            exp_prefix,
            returnn_search_config,
            checkpoint,
            recognition_dataset=recog_dataset,
            recognition_reference=recog_ref,
            returnn_exe=RETURNN_CPU_EXE,
            returnn_root=RETURNN_ROOT,
            mem_rqmt=mem_rqmt,
            time_rqmt=time_rqmt,
        )

    def run_search(
            exp_name, train_args, train_data, train_job,  num_epochs, search_args, recog_epochs,
            bpe_size, feature_extraction_net=log10_net_10ms, **kwargs):

        exp_prefix = os.path.join(prefix_name, exp_name)

        search_args = search_args if search_args is not None else train_args

        returnn_search_config = create_config(
            training_datasets=train_data,
            **search_args,
            feature_extraction_net=feature_extraction_net,
            is_recog=True,
        )

        num_avg = kwargs.get('num_avg', 4)
        averaged_checkpoint = get_average_checkpoint(
            train_job, returnn_exe=RETURNN_CPU_EXE, returnn_root=RETURNN_ROOT, num_average=num_avg,
        )
        if num_avg == 4:  # TODO: just for now to not break hashes
            train_job_avg_ckpt[exp_name] = averaged_checkpoint

        best_checkpoint = get_best_checkpoint(train_job)
        train_job_best_epoch[exp_name] = best_checkpoint

        if recog_epochs is None:
            default_recog_epochs = [80 * i for i in range(1, int(num_epochs / 80) + 1)]
            if num_epochs % 80 != 0:
                default_recog_epochs += [num_epochs]
        else:
            default_recog_epochs = recog_epochs

        test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)

        for ep in default_recog_epochs:
            search(
                exp_prefix + f"/recogs/ep-{ep}", returnn_search_config, train_job.out_checkpoints[ep],
                test_dataset_tuples, RETURNN_CPU_EXE, RETURNN_ROOT)

        search(
            exp_prefix + "/default_last", returnn_search_config, train_job.out_checkpoints[num_epochs],
            test_dataset_tuples, RETURNN_CPU_EXE, RETURNN_ROOT)

        search(
            exp_prefix + "/default_best", returnn_search_config, best_checkpoint,
            test_dataset_tuples, RETURNN_CPU_EXE, RETURNN_ROOT)

        search(
            exp_prefix + f"/average_{num_avg}", returnn_search_config, averaged_checkpoint,
            test_dataset_tuples, RETURNN_CPU_EXE, RETURNN_ROOT)

    def run_exp(
            exp_name, train_args, feature_extraction_net=log10_net_10ms, num_epochs=300, search_args=None,
            recog_epochs=None, bpe_size=10000, **kwargs
    ):
        if train_args.get('retrain_checkpoint', None):
            assert kwargs.get('epoch_wise_filter', None) is None, 'epoch_wise_filter should be disabled for retraining.'
        train_data = build_training_datasets(
            bpe_size=bpe_size,
            epoch_wise_filter=kwargs.get("epoch_wise_filter", [(1, 5, 1000)]),
            link_speed_perturbation=train_args.get("speed_pert", True),
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )
        train_job = run_train(exp_name, train_args, train_data, feature_extraction_net, num_epochs, recog_epochs, **kwargs)
        train_jobs_map[exp_name] = train_job

        run_search(
            exp_name, train_args, train_data, train_job, num_epochs, search_args, recog_epochs,
            bpe_size=bpe_size, feature_extraction_net=feature_extraction_net, **kwargs
        )
        return train_job, train_data

    # --------------------------- General Settings --------------------------- #

    conformer_enc_args = ConformerEncoderArgs(
        num_blocks=12,
        input_layer="conv-6",
        att_num_heads=8,
        ff_dim=2048,
        enc_key_dim=512,
        conv_kernel_size=32,
        pos_enc="rel",
        dropout=0.1,
        att_dropout=0.1,
        l2=0.0001,
    )
    apply_fairseq_init_to_conformer(conformer_enc_args)
    conformer_enc_args.ctc_loss_scale = 1.0

    rnn_dec_args = RNNDecoderArgs()

    trafo_dec_args = TransformerDecoderArgs(
        num_layers=6,
        embed_dropout=0.1,
        label_smoothing=0.1,
        apply_embed_weight=True,
        pos_enc="rel",
    )
    apply_fairseq_init_to_transformer_decoder(trafo_dec_args)

    conformer_dec_args = ConformerDecoderArgs()
    apply_fairseq_init_to_conformer(conformer_dec_args)

    training_args = dict()

    # LR scheduling
    training_args["const_lr"] = [42, 100]  # use const LR during pretraining
    training_args["wup_start_lr"] = 0.0002
    training_args["wup"] = 20
    training_args["with_staged_network"] = True
    training_args["speed_pert"] = True

    trafo_training_args = copy.deepcopy(training_args)
    trafo_training_args["pretrain_opts"] = {"variant": 3, "initial_batch_size": 20000 * 160}
    trafo_training_args["pretrain_reps"] = 5
    trafo_training_args["batch_size"] = 12000 * 160  # frames * samples per frame

    trafo_dec_exp_args = copy.deepcopy(
        {
            **trafo_training_args,
            "encoder_args": conformer_enc_args,
            "decoder_args": trafo_dec_args,
        }
    )

    conformer_dec_exp_args = copy.deepcopy(trafo_dec_exp_args)
    conformer_dec_exp_args['decoder_args'] = conformer_dec_args

    lstm_training_args = copy.deepcopy(training_args)
    lstm_training_args["pretrain_opts"] = {"variant": 3, "initial_batch_size": 22500 * 160}
    lstm_training_args["pretrain_reps"] = 5
    lstm_training_args["batch_size"] = 15000 * 160  # frames * samples per frame
    lstm_training_args["accum_grad"] = 1  # frames * samples per frame

    lstm_dec_exp_args = copy.deepcopy(
        {
            **lstm_training_args,
            "encoder_args": conformer_enc_args,
            "decoder_args": rnn_dec_args,
        }
    )

    # --------------------------- Experiments --------------------------- #

    oclr_args = copy.deepcopy(lstm_dec_exp_args)
    oclr_args["oclr_opts"] = {
        "peak_lr": 9e-4,
        "final_lr": 1e-6,
        "cycle_ep": 915,
        "total_ep": 2035,  # 20 epochs
        "n_step": 1350,
    }
    oclr_args["encoder_args"].input_layer = "conv-6"
    oclr_args['encoder_args'].use_sqrd_relu = True


    name = "base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009"
    train_j, train_data = run_exp(name, train_args=oclr_args, num_epochs=2035)

    # TODO: LM + ILM
    #for beam_size in [12]:
    #    run_lm_fusion(
    #        lm_type='trafo', exp_name=name, epoch='avg',
    #        test_set_names=['dev-clean', 'dev-other'],
    #        lm_scales=[0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52],
    #        train_job=train_j, train_data=train_data, feature_net=log10_net_10ms, args=oclr_args,
    #        beam_size=beam_size, bpe_size=BPE_10K,
    #   )

