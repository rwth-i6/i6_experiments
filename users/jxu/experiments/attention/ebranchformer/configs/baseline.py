import copy, os

from i6_experiments.users.jxu.experiments.attention.ebranchformer.attention_asr_config import (
    create_config,
    ConformerEncoderArgs,
    TransformerDecoderArgs,
    EbranchformerEncoderArgs,
    RNNDecoderArgs,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.additional_config import (
    apply_fairseq_init_to_conformer,
    apply_fairseq_init_to_transformer_decoder,
    reset_params_init,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.data import (
    build_training_datasets,
    build_test_dataset,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.default_tools import (
    RETURNN_EXE,
    RETURNN_ROOT,
    RETURNN_CPU_EXE,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.feature_extraction_net import (
    log10_net_10ms,
    log10_net_10ms_long_bn,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.pipeline import (
    training,
    search,
    get_average_checkpoint,
    get_best_checkpoint,
)


train_jobs_map = {}  # dict[str, ReturnnTrainJob]
train_job_avg_ckpt = {}
train_job_best_epoch = {}

def ebranchformer_baseline():

    abs_name = os.path.abspath(__file__)
    # prefix_name = abs_name[abs_name.find('/experiments') + len('/experiments') + 1:][:-len('.py')]
    prefix_name = os.path.basename(abs_name)[: -len(".py")]

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            testset, use_raw_features=True
        )

    def run_exp(
        exp_name,
        train_args,
        feature_extraction_net=log10_net_10ms,
        num_epochs=300,
        search_args=None,
        recog_epochs=None,
        **kwargs,
    ):
        exp_prefix = os.path.join(prefix_name, exp_name)

        train_data = build_training_datasets(
            bpe_size=10000,
            use_raw_features=True,
            epoch_wise_filter=kwargs.get("epoch_wise_filter", [(1, 5, 1000)]),
            link_speed_perturbation=train_args.get("speed_pert", True),
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )

        returnn_config = create_config(
            training_datasets=train_data,
            **train_args,
            feature_extraction_net=feature_extraction_net,
            recog_epochs=recog_epochs,
        )

        train_job = training(
            exp_prefix, returnn_config, RETURNN_EXE, RETURNN_ROOT, num_epochs=num_epochs
        )

        train_jobs_map[exp_name] = train_job

        search_args = search_args if search_args is not None else train_args

        returnn_search_config = create_config(
            training_datasets=train_data,
            **search_args,
            feature_extraction_net=feature_extraction_net,
            is_recog=True,
        )

        averaged_checkpoint = get_average_checkpoint(
            train_job, returnn_exe=RETURNN_CPU_EXE, returnn_root=RETURNN_ROOT, num_average=4
        )
        train_job_avg_ckpt[exp_name] = averaged_checkpoint

        best_checkpoint = get_best_checkpoint(train_job)
        train_job_best_epoch[exp_name] = best_checkpoint

        if recog_epochs is None:
            default_recog_epochs = [80 * i for i in range(1, int(num_epochs / 80) + 1)]
            if num_epochs % 80 != 0:
                default_recog_epochs += [num_epochs]
        else:
            default_recog_epochs = recog_epochs

        for ep in default_recog_epochs:
            search(
                exp_prefix + f"/recogs/ep-{ep}",
                returnn_search_config,
                train_job.out_checkpoints[ep],
                test_dataset_tuples,
                RETURNN_CPU_EXE,
                RETURNN_ROOT,
            )

        search(
            exp_prefix + "/default_last",
            returnn_search_config,
            train_job.out_checkpoints[num_epochs],
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
        )
        search(
            exp_prefix + "/default_best",
            returnn_search_config,
            best_checkpoint,
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
        )
        search(
            exp_prefix + "/average_4",
            returnn_search_config,
            averaged_checkpoint,
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
        )

        return train_job

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

    training_args = dict()

    training_args["with_staged_network"] = True
    training_args["speed_pert"] = True

    trafo_training_args = copy.deepcopy(training_args)
    trafo_training_args["pretrain_opts"] = {
        "variant": 3,
        "initial_batch_size": 20000 * 160,
    }
    trafo_training_args["pretrain_reps"] = 5
    trafo_training_args["batch_size"] = 10000 * 160  # frames * samples per frame

    trafo_dec_exp_args = copy.deepcopy(
        {
            **trafo_training_args,
            "encoder_args": conformer_enc_args,
            "decoder_args": trafo_dec_args,
        }
    )

    lstm_training_args = copy.deepcopy(training_args)
    lstm_training_args["pretrain_opts"] = {
        "variant": 3,
        "initial_batch_size": 22500 * 160,
    }
    lstm_training_args["pretrain_reps"] = 5
    lstm_training_args["batch_size"] = 10000 * 160  # frames * samples per frame

    ebranchformer_enc_args = EbranchformerEncoderArgs(
        input_layer="conv-6",
        pos_enc="rel",
        l2=0.0001,
    )

    apply_fairseq_init_to_conformer(ebranchformer_enc_args)
    ebranchformer_enc_args.ctc_loss_scale = 1.0

    lstm_dec_exp_args = copy.deepcopy(
        {
            **lstm_training_args,
            "encoder_args": ebranchformer_enc_args,
            "decoder_args": rnn_dec_args,
        }
    )

    # --------------------------- Experiments --------------------------- #

    for peak_lr in [7e-4,8e-4,9e-4]:
        pretrain_reps = 6
        best_args = copy.deepcopy(lstm_dec_exp_args)
        best_args["oclr_opts"] = {
            "peak_lr": peak_lr,
            "final_lr": 1e-6,
            "cycle_ep": 195,
            "total_ep": 435,  # 20 epochs
            "n_step": 1550,
        }
        # run_exp("base_ebranchformer_17l_lstm_1l_OCLR_conv6_relu", train_args=best_args, num_epochs=435)

        smaller_model_args = copy.deepcopy(best_args)
        smaller_model_args['encoder_args'].num_blocks=12
        smaller_model_args['pretrain_reps'] = pretrain_reps
        smaller_model_args["batch_size"] = 13000 * 160
        run_exp("base_ebranchformer_12l_lstm_1l_OCLR_conv6_relu_pretrain_reprs_{}".format(('%.0E' % peak_lr).replace('-','_').replace('E','e')), train_args=smaller_model_args, num_epochs=435)




