# Not finished -> use alberts setup instead
import os
from i6_core.returnn.forward import ReturnnForwardJob
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.pipeline import search_single
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.convert_checkpoint import convert_checkpoint

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.serializers.serializer import get_serializer

_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"

BPE_10K = 10000

def rf_ctc_att_search():
    abs_name = os.path.abspath(__file__)
    prefix_name = os.path.basename(abs_name)[: -len(".py")]

    def run_decoding(
        exp_name,
        # train_data,
        checkpoint,
        search_args,
        # feature_extraction_net,
        bpe_size,
        test_sets: list,
        time_rqmt: float = 1.0,
        remove_label=None,
        two_pass_rescore=False,
        **kwargs,
    ):
        test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)
        for test_set in test_sets:
            run_single_search(
                exp_name=exp_name + f"/recogs/{test_set}",
                # train_data=train_data,
                search_args=search_args,
                checkpoint=checkpoint,
                # feature_extraction_net=feature_extraction_net,
                recog_dataset=test_dataset_tuples[test_set][0],
                recog_ref=test_dataset_tuples[test_set][1],
                recog_bliss=test_dataset_tuples[test_set][2],
                time_rqmt=time_rqmt,
                remove_label=remove_label,
                two_pass_rescore=two_pass_rescore,
                **kwargs,
            )

    def run_single_search(
        exp_name,
        # train_data,
        search_args,
        checkpoint,
        # feature_extraction_net,
        recog_dataset,
        recog_ref,
        recog_bliss,
        mem_rqmt: float = 8,
        time_rqmt: float = 4,
        **kwargs,
    ):
        exp_prefix = os.path.join(prefix_name, exp_name)
        # returnn_search_config = create_config(
        #     training_datasets=train_data,
        #     **search_args,
        #     # feature_extraction_net=feature_extraction_net,
        #     is_recog=True,
        # )
        # TODO: Get serialized config

        recog_serializer = get_serializer(
            model_config=model_config,
            model_import_path="models.torchaudio_conformer_ctc",
            train=False,
            import_kwargs={
                "text_lexicon": get_text_lexicon()
            },
            debug=debug,
        )

        search_config = get_pt_search_config(
            forward_dataset=0, # TODO
            serializer=recog_serializer,
        )
        search_single(
            exp_prefix,
            search_config,
            checkpoint,
            recognition_dataset=recog_dataset,
            recognition_reference=recog_ref,
            recognition_bliss_corpus=recog_bliss,
            returnn_exe=RETURNN_CPU_EXE,
            returnn_root=RETURNN_ROOT,
            mem_rqmt=mem_rqmt,
            time_rqmt=time_rqmt,
            **kwargs,
        )

    new_checkpoint = convert_checkpoint(_returnn_tf_ckpt_filename)

    search_args = {}

    run_decoding(
        exp_name="test_ctc_greedy",
        # train_data=train_data,
        # checkpoint=train_job_avg_ckpt["base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009"],
        checkpoint=new_checkpoint,
        search_args={**search_args},
        # feature_extraction_net=log10_net_10ms,
        bpe_size=BPE_10K,
        test_sets=["dev-other"],
        remove_label={"<s>", "<blank>"},  # blanks are removed in the network
        use_sclite=True,
    )

    # partial_recognition = ctc_search_segmented(
    #     checkpoint=train_job.out_checkpoints[num_epochs],
    #     config=search_config,
    #     returnn_exe=RETURNN_PYTORCH_EXE,
    #     returnn_root=MINI_RETURNN_ROOT,
    #     prefix=test_set_prefix,
    #     segment_num=i,
    # )


def get_pt_search_config(
        forward_dataset: GenericDataset,
        serializer: Collection,
):
    config = {
        "batch_size": 50_000 * 160,
        "max_seqs": 200,
        #############
        "forward": forward_dataset.as_returnn_opts()
    }

    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config

def ctc_search_segmented(checkpoint, config, returnn_exe, returnn_root, prefix, segment_num):
    search_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=config,
        hdf_outputs=["recognition.txt"],
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        device="cpu",
        cpu_rqmt=8,
        mem_rqmt=8,
        time_rqmt=2,
    )

    search_job.add_alias(prefix + f"/search_job.{segment_num}")
    recognition = search_job.out_hdf_files[f"recognition.txt"]
    tk.register_output(prefix + f"/recognition.{segment_num}", recognition)
    return recognition


