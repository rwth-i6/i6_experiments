import copy
from enum import Enum
from typing import Dict, Any

from i6_core.returnn import ReturnnForwardJobV2
from i6_core.returnn.search import SearchOutputRawReplaceJob, SearchTakeBestJob
from i6_core.returnn.training import ReturnnConfig
from i6_core.text.processing import PipelineJob
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from i6_experiments.users.juanola.experiments.e25_11_10_llm.constants import DATA_PARAM_NAME
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput, ScoreResult
from i6_experiments.users.zeyer.datasets.utils import sclite_generic_score
from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
from i6_experiments.users.zeyer.datasets.utils.vocab import ExtractVocabLabelsJob
from i6_experiments.users.zeyer.decoding.rescoring import SearchCombineScoresJob
from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob
from sisyphus import tk
from .asr_model import ASRModel
from ..model_creation.returnn_config_helpers import get_forward_config_v2
from ...configurations.pipeline.search_config import SearchConfig
from ...default_tools import RETURNN_EXE, RETURNN_ROOT

default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}


class Scales(Enum):
    CTC = "ctc"
    LLM = "llm"
    SLLM = "sllm"
    PRIOR = "prior"

def ctc_label_sync_eval_auto_scale(
    asr_model: ASRModel,
    search_config: SearchConfig,
    train_data: TrainingDatasets,
    tune_datasets: Dict[str, Any],
    evaluation_name: str,
    label_datastream_key="labels",
    ogg_zip_dataset_target_key="classes",
    lowercase_ref: bool = False,

) -> Dict[str, tk.Variable]:
    """
    Robins code (SLLM repo)
    """

    use_ctc = search_config.ctc_scales is not None and len(search_config.ctc_scales) == 1 and search_config.ctc_scales[0] == 1.0
    use_llm = search_config.ctc_scales is not None and len(search_config.lm_scales) == 1 and search_config.lm_scales[0] == 1.0
    use_sllm = search_config.ctc_scales is not None and len(search_config.sllm_scales) == 1 and search_config.sllm_scales[0] == 1.0
    use_prior = search_config.ctc_scales is not None and len(search_config.prior_scales) == 1 and search_config.prior_scales[0] == 1.0

    print(f"Using: ", end="")
    if use_ctc:
        print("CTC ", end="")
    if use_llm:
        print("LLM ", end="")
    if use_sllm:
        print("SLLM ", end="")
    if use_prior:
        print("PRIOR ", end="")
    print("scales!")

    # DATASET FOR TUNNING
    assert len(tune_datasets.items()) == 1, "Only one dataset is supported for now!"
    tune_dataset_name, (tune_dataset, tune_dataset_ref) = list(tune_datasets.items())[0]

    # Add target params
    tune_dataset_dict = tune_dataset.as_returnn_opts()
    tune_dataset_dict["dataset"]["targets"] = train_data.datastreams[label_datastream_key].as_returnn_targets_opts()

    # GET CHECKPOINT & PATH
    recog_path = (
        f"{evaluation_name}/auto_scaling_test1/{tune_dataset_name}"  # TODO: add some params of what is being tuned!
    )

    # TEXT REFERENCES
    ref = ReturnnDatasetToTextDictJob(
        returnn_dataset=tune_dataset_dict,
        data_key=ogg_zip_dataset_target_key,
        vocab=train_data.datastreams[label_datastream_key].as_returnn_targets_opts(),
    ).out_txt
    if lowercase_ref:
        ref = PipelineJob(
            ref, pipeline=[r"""perl -pi -e "s/(:\\s*)(['\"])(.*?)\\2/\$1 . \$2 . lc(\$3) . \$2/e" """]
        ).out

    # Net args # todo! use model with extra networks

    # BASE RECOG CONFIG
    beam_size = search_config.beam_search.beam_sizes[0]
    forward_config = {
        "batch_size": search_config.batch_size * search_config.batch_size_factor,
        "max_seqs": search_config.max_seqs,
    }
    forward_config_low_batch_size = {
        "batch_size": 2_000 * search_config.batch_size_factor,
        "max_seqs": search_config.max_seqs,
    }
    forward_params = {  # fix params from here
        "beam_size": beam_size,
        "max_tokens_per_sec": 20,
        "sample_rate": 16000,
    }

    # ACCUMULATED SCORES
    scores = {}

    if use_ctc:
        # CTC N-BEST RECOGNITION
        returnn_ctc_n_best_config = get_forward_config_v2(
            network_import_path=asr_model.network_import_path,
            base_config=forward_config,
            net_args=asr_model.net_args,
            decoder_args=forward_params,
            forward_module="recognition",
            forward_method="forward_step_v2",
            callback_name="RecognitionToTextDictCallbackV2",
            label_datastream=train_data.datastreams[label_datastream_key],
            callback_opts={
                "include_beam": True,
                "merge_labels": False,
            },
            debug=search_config.debug_returnn_param,
        )
        ctc_n_best_original = forward_single(
            f"{recog_path}/ctc-n-best",
            returnn_config=returnn_ctc_n_best_config,
            checkpoint=(asr_model.checkpoint),
            dataset_dict=tune_dataset_dict,
            **default_returnn,
            rqmt={},
        )
        ctc_n_best = SearchOutputRawReplaceJob(
            ctc_n_best_original, replacement_list=[(" ", ""), ("▁", " ")]
        ).out_search_results
        scores[Scales.CTC.value] = ctc_n_best
    else:
        assert False, "NEEDS CTC"

    if use_sllm:
        # SLLM RESCORING
        vocab_file = ExtractVocabLabelsJob(train_data.datastreams[label_datastream_key].as_returnn_targets_opts()).out_vocab
        rescore_data = {
            "class": "TextDictDataset",
            "filename": ctc_n_best_original,
            "vocab": {
                "class": "Vocabulary",
                "vocab_file": vocab_file,
                "unknown_label": None,
            },
        }
        rescore_data = {  # TODO: add prior and SLLM to be tuned here?
            "class": "MetaDataset",
            "datasets": {"orig_data": tune_dataset_dict["dataset"], "hyps": rescore_data},
            "data_map": {
                "audio": ("orig_data", "data"),
                "hyps_flat": ("hyps", "data_flat"),
                "hyps_seq_lens": ("hyps", "data_seq_lens"),
            },
            "seq_order_control_dataset": "hyps",
        }
        returnn_rescoring_config = get_forward_config_v2(
            network_import_path=asr_model.network_import_path,
            base_config=forward_config_low_batch_size,  # because of large beam dim, we need to lower the batch size
            net_args=asr_model.net_args,
            decoder_args=forward_params, # TODO: add param with which module to use for decoding (SLLM or ext LLM)
            forward_module="rescoring",
            forward_method="forward_step_v1",
            callback_name="RecognitionToTextDictCallbackV2",
            label_datastream=train_data.datastreams[label_datastream_key],
            callback_opts={"include_beam": True},
            extern_data={
                "audio": {"dim": 1},
                "hyps_flat": {
                    "shape": [None],
                    "dtype": "int32",
                    "vocab": rescore_data["datasets"]["hyps"]["vocab"],
                },
                "hyps_seq_lens": {"shape": [beam_size], "dtype": "int32"},
            },
            debug=search_config.debug_returnn_param,
        )
        sllm_rescore_results = forward_single(  # TODO: change to LM? and proper network...
            f"{recog_path}/sllm_rescoring",
            returnn_config=returnn_rescoring_config,
            checkpoint=(asr_model.checkpoint),
            dataset_dict=rescore_data,
            **default_returnn,
            rqmt={},
        )
        sllm_rescore_results = SearchOutputRawReplaceJob(
            sllm_rescore_results, replacement_list=[(" ", ""), ("▁", " ")]
        ).out_search_results
        scores[Scales.SLLM.value] = sllm_rescore_results

    if use_llm:
        pass

    if use_prior:
        pass

        # TODO: add prior here !!
        # Broadcast prior for batch and beam.. ?
        # Use trivial forward job rescoring/../broadcast_prior


    # AUTO SCALING
    opt_scales_job = ScaleTuningJob(
        scores=scores,
        ref=ref,
        fixed_scales={Scales.CTC.value: 1.0},
        evaluation="edit_distance",
    )
    opt_scales_job.rqmt["engine"] = "short"  # should be fine
    tk.register_output(f"{recog_path}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{recog_path}/opt-rel-scales", opt_scales_job.out_scales)
    combined_scores = SearchCombineScoresJob(
        [
            (opt_scales_job.out_real_scale_per_name[Scales.CTC.value], scores[Scales.CTC.value]),
            (opt_scales_job.out_real_scale_per_name[Scales.SLLM.value], scores[Scales.SLLM.value]),
        ]
    ).out_search_results
    best_rescore = SearchTakeBestJob(combined_scores).out_best_search_results

    # GET SCORES
    tune_score_result = generic_sclite_score_recog_out(
        dataset=tune_dataset_dict,
        recog_output=best_rescore,
        corpus_name="dev",  # TODO?
        vocab=train_data.datastreams[label_datastream_key].as_returnn_targets_opts(),
        data_key=ogg_zip_dataset_target_key,
        use_lowercase=lowercase_ref,
        # apply_text_norm=False,
    )
    tk.register_output(f"{recog_path}/tune_rescore/wer", tune_score_result.main_measure_value)
    tk.register_output(f"{recog_path}/tune_rescore/report", tune_score_result.report)

    return opt_scales_job.out_real_scale_per_name


def generic_sclite_score_recog_out(
    dataset: Dict[str, Any],
    recog_output: tk.Path,
    vocab: Dict[str, Any],
    data_key: str,
    corpus_name: str,
    use_lowercase: bool = False,
) -> ScoreResult:
    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset["dataset"] if dataset["class"] == "MultiProcDataset" else dataset,
            data_key=data_key,
            vocab=vocab,
        ).out_txt
    )

    if use_lowercase:
        ref = RecogOutput(
            output=PipelineJob(
                ref.output, pipeline=[r"""perl -pi -e "s/(:\\s*)(['\"])(.*?)\\2/\$1 . \$2 . lc(\$3) . \$2/e" """]
            ).out
        )

    return sclite_generic_score.sclite_score_recog_out_to_ref(
        RecogOutput(recog_output), ref=ref, corpus_name=corpus_name
    )


def forward_single(
    prefix_name: str,
    returnn_config: ReturnnConfig,
    checkpoint: tk.Path,
    dataset_dict: Dict,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    rqmt: Dict,
):
    """
    Run search for a specific test dataset

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for forwarding
    :param Checkpoint checkpoint: path to RETURNN PyTorch model checkpoint
    :param recognition_dataset: Dataset to perform recognition on
    :param recognition_bliss_corpus: path to bliss file used as Sclite evaluation reference
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param mem_rqmt: some search jobs might need more memory
    :param use_gpu: if to do GPU decoding
    """
    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["forward_data"] = dataset_dict
    search_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=rqmt.get("mem", 20),
        time_rqmt=1,
        device="gpu",
        cpu_rqmt=rqmt.get("cpu", 8),
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["search_out.py.gz"],
    )
    search_job.add_alias(prefix_name + "/search_job")
    tk.register_output(prefix_name + "/search_out.py.gz", search_job.out_files["search_out.py.gz"])

    return search_job.out_files["search_out.py.gz"]
