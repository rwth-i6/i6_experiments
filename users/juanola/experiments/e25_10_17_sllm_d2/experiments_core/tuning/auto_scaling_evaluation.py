from typing import Optional, Dict, Any, Union


from i6_core.returnn.search import SearchOutputRawReplaceJob, SearchTakeBestJob
from i6_core.returnn.training import ReturnnTrainingJob, ReturnnConfig
from i6_core.text.processing import PipelineJob
from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
from i6_experiments.users.zeyer.datasets.utils.vocab import ExtractVocabLabelsJob
from i6_experiments.users.zeyer.decoding.rescoring import SearchCombineScoresJob
from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob
from sisyphus import tk

default_returnn = { # TODO: use other?
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}


# TODO: call this from main exps?

# TODO: check if something needs  to be changed to librispeech!!
def ctc_label_sync_eval_auto_scale(
        training_name: str,
        recog_name: str,
        train_job: ReturnnTrainingJob,
        train_args: Dict[str, Any],
        train_data: TrainingDatasets,
        tune_dataset,
        checkpoint_name: Union[int, str],
        loss_name: str = "dev_loss_ce",
        extra_forward_config: Optional[ReturnnConfig] = None,
        label_datastream_key="text",
        lowercase_ref: bool = False,
) -> Dict[str, tk.Variable]:
    """
    Robins code (SLLM repo)
    """

    # GET CHECKPOINT & PATH # TODO: PROBABLY REPLACE

    if isinstance(checkpoint_name, int):
        get_best_averaged_checkpoint = None
    elif checkpoint_name == "best":
        get_best_averaged_checkpoint = (1, loss_name)
    else:
        assert checkpoint_name == "best4"
        get_best_averaged_checkpoint = (4, loss_name)
    checkpoint = get_checkpoint(
        training_name,
        train_job,
        get_specific_checkpoint=checkpoint_name if isinstance(checkpoint_name, int) else None,
        get_best_averaged_checkpoint=get_best_averaged_checkpoint,
    )
    recog_path = f"{training_name}/{recog_name}/{str(checkpoint_name)}"

    # CTC N-BEST

    returnn_ctc_n_best_config = get_forward_config(
        network_module=train_args["network_module"],
        extra_config=extra_forward_config if extra_forward_config else ReturnnConfig({}),
        net_args=train_args["net_args"],
        decoder_args={
            "beam_size": 64,
            "max_tokens_per_sec": 20,
            "sample_rate": 16000,
        },
        decoder="recognition.ctc.forward_step.forward_step_v1",
        callback_module="recognition.callback.RecognitionToTextDictCallback",
        label_datastream=train_data.datastreams[label_datastream_key],
        callback_opts={"include_beam": True, "merge_labels": False,}
    )
    ctc_n_best = forward_single(
        f"{recog_path}/ctc-n-best",
        returnn_config=returnn_ctc_n_best_config,
        checkpoint=checkpoint,
        dataset_dict=tune_dataset.as_returnn_opts(),
        **default_returnn,
        rqmt={},
    )

    # RESCORING

    vocab_file = ExtractVocabLabelsJob(train_data.datastreams[label_datastream_key].as_returnn_targets_opts()).out_vocab
    rescore_data = {
        "class": "TextDictDataset",
        "filename":ctc_n_best,
        "vocab": {
            "class": "Vocabulary",
            "vocab_file": vocab_file,
            "unknown_label": None,
        }
    }
    rescore_data = { # TODO: add prior and SLLM to be tuned here?
        "class": "MetaDataset",
        "datasets": {
            "orig_data": tune_dataset.as_returnn_opts(),
            "hyps": rescore_data
        },
        "data_map": {
            "audio": ("orig_data", "audio"),
            "hyps_flat": ("hyps", "data_flat"),
            "hyps_seq_lens": ("hyps", "data_seq_lens"),
        },
        "seq_order_control_dataset": "hyps",
    }
    returnn_rescoring_config = get_forward_config(
        network_module=train_args["network_module"],
        extra_config=extra_forward_config if extra_forward_config else ReturnnConfig({}),
        net_args=train_args["net_args"],
        decoder_args={
            "beam_size": 64,
            "max_tokens_per_sec": 20,
            "sample_rate": 16000,
        },
        decoder="rescoring.forward_step.forward_step_v1",
        callback_module="recognition.callback.RecognitionToTextDictCallback",
        label_datastream=train_data.datastreams[label_datastream_key],
        callback_opts={"include_beam": True},
        extern_data={
            "audio": {"shape": (None,)},
            "hyps_flat": {
                "shape": [None],
                "dtype": "int32",
                "vocab": rescore_data["datasets"]["hyps"]["vocab"],
            },
            "hyps_seq_lens": {
                "shape": [64],
                "dtype": "int32"
            },
        },
        # because of large beam dim, we need to lower the batch size
        base_config={"batch_size": 2_000 * 160}
    )
    sllm_rescore_results = forward_single( # TODO: change to LM?
        f"{recog_path}/sllm_rescoring",
        returnn_config=returnn_rescoring_config,
        checkpoint=checkpoint,
        dataset_dict=rescore_data,
        **default_returnn,
        rqmt={},
    )


    ctc_n_best_merged = SearchOutputRawReplaceJob(
        ctc_n_best, replacement_list=[(" ", ""), ("Ġ", " ")]
    ).out_search_results

    ref = ReturnnDatasetToTextDictJob(
        returnn_dataset=tune_dataset.as_returnn_opts(),
        data_key="text",
        vocab=train_data.datastreams[label_datastream_key].as_returnn_targets_opts(),
    ).out_txt
    if lowercase_ref:
        ref = PipelineJob(
            ref,
            pipeline=[r"""perl -pi -e "s/(:\\s*)(['\"])(.*?)\\2/\$1 . \$2 . lc(\$3) . \$2/e" """]
        ).out

    opt_scales_job = ScaleTuningJob(
        scores={"ctc": ctc_n_best_merged, "slm": sllm_rescore_results},
        ref=ref,
        fixed_scales={"ctc": 1.0},
        evaluation="edit_distance",
    )
    opt_scales_job.rqmt["engine"] = "short"  # should be fine
    tk.register_output(f"{recog_path}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{recog_path}/opt-rel-scales", opt_scales_job.out_scales)

    combined_scores = SearchCombineScoresJob(
        [
            (opt_scales_job.out_real_scale_per_name["ctc"], ctc_n_best_merged),
            (opt_scales_job.out_real_scale_per_name["slm"], sllm_rescore_results),
        ]
    ).out_search_results
    best_rescore = SearchTakeBestJob(combined_scores).out_best_search_results
    tune_score_result = generic_sclite_score_recog_out(
        dataset=tune_dataset.as_returnn_opts(),
        recog_output=best_rescore,
        corpus_name="dev",
        use_lowercase=lowercase_ref,
        apply_text_norm=False,
    )
    tk.register_output(f"{recog_path}/tune_rescore/wer", tune_score_result.main_measure_value)
    tk.register_output(f"{recog_path}/tune_rescore/report", tune_score_result.report)

    # We use the real scales.
    return opt_scales_job.out_real_scale_per_name