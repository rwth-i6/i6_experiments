import copy
from enum import Enum
from typing import Dict, Any

from i6_core.corpus import CorpusToStmJob
from i6_core.recognition import ScliteJob
from i6_core.returnn import ReturnnForwardJobV2
from i6_core.returnn.search import SearchOutputRawReplaceJob, SearchTakeBestJob, SearchWordsToCTMJob
from i6_core.returnn.training import ReturnnConfig
from i6_core.text.processing import PipelineJob
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput, ScoreResult
from i6_experiments.users.zeyer.datasets.utils import sclite_generic_score
from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
from i6_experiments.users.zeyer.datasets.utils.vocab import ExtractVocabLabelsJob
from i6_experiments.users.zeyer.decoding.prior_rescoring import SearchPriorRescoreJob
from i6_experiments.users.zeyer.decoding.rescoring import SearchCombineScoresJob
from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob
from sisyphus import tk
from .asr_model import ASRModel
from ..model_creation.returnn_config_helpers import get_forward_config_v2
from ...configurations.pipeline.search_config import SearchConfig
from ...configurations.pretrained_models import get_encoder_checkpoint_from_str, get_decoder_checkpoint_from_str
from ...default_tools import RETURNN_EXE, RETURNN_ROOT, SCTK_BINARY_PATH
from ...utils_network_args import get_network_args

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
) -> tuple[Dict[str, tk.Variable], str]:
    """
    Robins code (SLLM repo)
    """
    autoscale_id = "autoscale"

    use_ctc = search_config.ctc_scales is not None and len(search_config.ctc_scales) == 1 and search_config.ctc_scales[0] == 1.0
    use_llm = search_config.ctc_scales is not None and len(search_config.lm_scales) == 1 and search_config.lm_scales[0] == 1.0
    use_sllm = search_config.ctc_scales is not None and len(search_config.sllm_scales) == 1 and search_config.sllm_scales[0] == 1.0
    use_prior = search_config.ctc_scales is not None and len(search_config.prior_scales) == 1 and search_config.prior_scales[0] == 1.0

    if use_ctc:
        autoscale_id += "_CTC"
        if search_config.auto_scaling_use_ctc_sum_scores:
            autoscale_id += "_sum_scores"
    if use_llm:
        autoscale_id += "_LLM"
    if use_sllm:
        autoscale_id += "_SLLM"
    if use_prior:
        autoscale_id += "_PRIOR"

    # DATASET FOR TUNNING
    assert len(tune_datasets.items()) == 1, "Only one dataset is supported for now!"
    tune_dataset_name, (tune_dataset, tune_dataset_ref) = list(tune_datasets.items())[0]

    # Add target params
    tune_dataset_dict = tune_dataset.as_returnn_opts()
    tune_dataset_dict["dataset"]["targets"] = train_data.datastreams[label_datastream_key].as_returnn_targets_opts()

    # GET CHECKPOINT & PATH
    recog_path = (
        f"{evaluation_name}/{autoscale_id}/{tune_dataset_name}"  # TODO: add some params of what is being tuned!
    )

    # TEXT REFERENCES & VOCAB
    ref = ReturnnDatasetToTextDictJob(
        returnn_dataset=tune_dataset_dict,
        data_key=ogg_zip_dataset_target_key,
        vocab=train_data.datastreams[label_datastream_key].as_returnn_targets_opts(),
    ).out_txt
    ref = SearchOutputRawReplaceJob(
        ref, replacement_list=[(" ", ""), ("▁", " ")]
    ).out_search_results
    if lowercase_ref:
        ref = PipelineJob(
            ref, pipeline=[r"""perl -pi -e "s/(:\\s*)(['\"])(.*?)\\2/\$1 . \$2 . lc(\$3) . \$2/e" """]
        ).out

    # Net args
    extra_returnn_configs = [] #TODO: apply this
    net_args = asr_model.net_args
    preloading = {}
    python_prolog = None

    if search_config.ext_encoder is not None:
        net_args["external_ctc_args"] = get_network_args(search_config.ext_encoder["network_config"], search_config.ext_encoder["label_config"])
        preloading[f"EXT_ENCODER-{search_config.ext_encoder['checkpoint_key']}"] = {
            "filename": get_encoder_checkpoint_from_str(search_config.ext_encoder["checkpoint_key"]),
            "prefix": "external_ctc.",
            "init_for_train": False,
            "ignore_missing": True,
            "ignore_params_prefixes": ["external_lm.", "decoder_embed_func", "decoder"]
        }
    else:
        assert "external_ctc_args" not in net_args, "Search is not using external ctc but net arguments are provided!"

    if search_config.ext_decoder is not None:
        net_args["external_lm_args"] = get_network_args(search_config.ext_decoder["network_config"], search_config.ext_decoder["label_config"])
        if not search_config.ext_decoder_no_preloading:
            preloading[f"EXT_DECODER-{search_config.ext_decoder['checkpoint_key']}"] = {
                "filename": get_decoder_checkpoint_from_str(search_config.ext_decoder["checkpoint_key"]),
                "prefix": "external_lm.",
                "init_for_train": False,
                "ignore_missing": True,
                "ignore_params_prefixes": ["external_ctc.", "encoder", "mel_frontend"], # , "external_lm.decoder.model.embed_tokens"
                #"var_name_mapping": {"external_lm.decoder.model.embed_tokens.weight": "external_lm.decoder_embed_func.weight"}
                #"custom_missing_load_func": CodeWrapper("adapt_extern_decoder_embedding"),
            }
    else:
        assert "external_lm_args" not in net_args, "Search is not using external lm but net arguments are provided!"

    if preloading:
        preloading_config = {
            "preload_from_files": preloading,
        }
        extra_returnn_configs.append(ReturnnConfig(config=preloading_config, python_prolog=python_prolog))

    # BASE RECOG CONFIG
    beam_size = 64
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

    rqmt = {
        "mem": search_config.gpu_memory,
        "cpu": search_config.cpu_memory,
    }

    # ACCUMULATED SCORES
    scores = {}

    # CTC N-BEST RECOGNITION [base for any combination]
    returnn_ctc_n_best_config = get_forward_config_v2(
        network_import_path=asr_model.network_import_path,
        base_config=forward_config,
        net_args=asr_model.net_args,
        decoder_args=forward_params,
        forward_module="recognition.ctc",
        forward_method="ctc_forward_step_v1",
        callback_name="RecognitionToTextDictCallbackV2",
        label_datastream=train_data.datastreams[label_datastream_key],
        callback_opts={
            "include_beam": True,
            "merge_labels": False,
        },
        debug=search_config.debug_returnn_param,
        extra_configs= extra_returnn_configs,
    )
    ctc_n_best_original = forward_single(
        f"{recog_path}/ctc-n-best",
        returnn_config=returnn_ctc_n_best_config,
        checkpoint=asr_model.checkpoint,
        dataset_dict=tune_dataset_dict,
        **default_returnn,
        rqmt=rqmt,
    )

    # RESCORINGS
    vocab_file = ExtractVocabLabelsJob(train_data.datastreams[label_datastream_key].as_returnn_targets_opts()).out_vocab # Has "_word" form
    rescore_data = {
        "class": "TextDictDataset",
        "filename": ctc_n_best_original, # Has "_word" form
        "vocab": {
            "class": "Vocabulary",
            "vocab_file": vocab_file, # Has "_word" form
            "unknown_label": None,
        },
    }
    rescore_data = {
        "class": "MetaDataset",
        "datasets": {"orig_data": tune_dataset_dict["dataset"], "hyps": copy.deepcopy(rescore_data)},
        "data_map": {
            "audio": ("orig_data", "data"),
            "hyps_flat": ("hyps", "data_flat"),
            "hyps_seq_lens": ("hyps", "data_seq_lens"),
        },
        "seq_order_control_dataset": "hyps",
    }

    if use_ctc:
        if search_config.auto_scaling_use_ctc_sum_scores:
            returnn_ctc_rescoring_config = get_forward_config_v2(
                network_import_path=asr_model.network_import_path,
                base_config=forward_config_low_batch_size,
                net_args=asr_model.net_args,
                decoder_args=forward_params,
                forward_module="recognition.ctc",
                forward_method="ctc_forward_step_v1",
                callback_name="RecognitionToTextDictCallbackV2",
                label_datastream=train_data.datastreams[label_datastream_key],
                callback_opts={
                    "include_beam": True,
                },
                debug=search_config.debug_returnn_param,
                extra_configs= extra_returnn_configs,
                extern_data={
                    "audio": {"dim": 1},
                    "hyps_flat": {
                        "shape": [None],
                        "dtype": "int32",
                        "vocab": copy.deepcopy(rescore_data["datasets"]["hyps"]["vocab"]),
                    },
                    "hyps_seq_lens": {"shape": [beam_size], "dtype": "int32"},
                },
            )
            ctc_rescore_results= forward_single(
                f"{recog_path}/ctc_sum_rescoring",
                returnn_config=returnn_ctc_rescoring_config,
                checkpoint=asr_model.checkpoint,
                dataset_dict=rescore_data,
                **default_returnn,
                rqmt=rqmt,
            )
            ctc_rescore_results = SearchOutputRawReplaceJob(
                ctc_rescore_results, replacement_list=[(" ", ""), ("▁", " ")]
            ).out_search_results
        else:
            ctc_rescore_results = SearchOutputRawReplaceJob(
                ctc_n_best_original, replacement_list=[(" ", ""), ("▁", " ")]
            ).out_search_results
        scores[Scales.CTC.value] = ctc_rescore_results

    if use_sllm:
        # SLLM RESCORING
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
                    "vocab": copy.deepcopy(rescore_data["datasets"]["hyps"]["vocab"]),
                },
                "hyps_seq_lens": {"shape": [beam_size], "dtype": "int32"},
            },
            debug=search_config.debug_returnn_param,
            extra_configs= extra_returnn_configs,
        )
        sllm_rescore_results = forward_single(  # TODO: change to LM? and proper network...
            f"{recog_path}/sllm_rescoring",
            returnn_config=returnn_rescoring_config,
            checkpoint=asr_model.checkpoint,
            dataset_dict=rescore_data,
            **default_returnn,
            rqmt=rqmt,
        )
        sllm_rescore_results = SearchOutputRawReplaceJob(
            sllm_rescore_results, replacement_list=[(" ", ""), ("▁", " ")]
        ).out_search_results
        scores[Scales.SLLM.value] = sllm_rescore_results

    if use_llm:
        # LLM RESCORING
        lm_forward_params = copy.deepcopy(forward_params)
        lm_forward_params["use_ext_lm"] = True
        if search_config.sllm_as_llm:
            lm_forward_params["sllm_as_llm"] = True

        returnn_rescoring_config = get_forward_config_v2(
            network_import_path=asr_model.network_import_path,
            base_config=forward_config_low_batch_size,  # because of large beam dim, we need to lower the batch size
            net_args=asr_model.net_args,
            decoder_args=lm_forward_params,
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
                    "vocab": copy.deepcopy(rescore_data["datasets"]["hyps"]["vocab"]),
                },
                "hyps_seq_lens": {"shape": [beam_size], "dtype": "int32"},
            },
            debug=search_config.debug_returnn_param,
            extra_configs= extra_returnn_configs,
        )
        llm_rescore_results = forward_single(  # TODO: change to LM? and proper network...
            f"{recog_path}/llm_rescoring",
            returnn_config=returnn_rescoring_config,
            checkpoint=asr_model.checkpoint,
            dataset_dict=rescore_data,
            **default_returnn,
            rqmt=rqmt,
        )
        llm_rescore_results = SearchOutputRawReplaceJob(
            llm_rescore_results, replacement_list=[(" ", ""), ("▁", " ")]
        ).out_search_results
        scores[Scales.LLM.value] = llm_rescore_results

    if use_prior:
        assert asr_model.prior_text_file is not None, "Prior text file is needed"
        prior_rescore_job = SearchPriorRescoreJob(
            ctc_n_best_original, # Has "_word" form
            prior=asr_model.prior_text_file,
            prior_type="prob",
            vocab=vocab_file, # Has "_word" form
            vocab_is_chars=False,
        )
        prior_rescore_job.add_alias(f"{recog_path}/prior_rescoring")
        prior_rescore_results = prior_rescore_job.out_search_results
        prior_rescore_results = SearchOutputRawReplaceJob(
            prior_rescore_results, replacement_list=[(" ", ""), ("▁", " ")]
        ).out_search_results
        scores["prior"] = prior_rescore_results

    # AUTO SCALING
    fixed_scales = None
    if use_ctc:
        fixed_scales = {Scales.CTC.value: 1.0}
    elif use_sllm:
        fixed_scales = {Scales.SLLM.value: 1.0}
    else:
        fixed_scales = None

    opt_scales_job = ScaleTuningJob(
        scores=scores,
        ref=ref,
        fixed_scales=fixed_scales,
        negative_scales={Scales.PRIOR.value} if use_prior else None,
        evaluation="edit_distance",
    )
    opt_scales_job.rqmt["engine"] = "short"  # should be fine
    tk.register_output(f"{recog_path}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{recog_path}/opt-rel-scales", opt_scales_job.out_scales)

    # CTC BEST
    # best_ctc = SearchTakeBestJob(ctc_rescore_results).out_best_search_results
    # ctc_search_ctm = SearchWordsToCTMJob(
    #     recog_words_file=best_ctc,
    #     bliss_corpus=tune_dataset_ref,
    # ).out_ctm_file
    # ctc_stm_file = CorpusToStmJob(bliss_corpus=tune_dataset_ref).out_stm_path
    # ctc_sclite_job = ScliteJob(ref=ctc_stm_file, hyp=ctc_search_ctm, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=1)
    # tk.register_output(f"{recog_path}/ctc_tune_rescore/wer", ctc_sclite_job.out_wer)
    # tk.register_output(f"{recog_path}/ctc_tune_rescore/report", ctc_sclite_job.out_report_dir)

    # EVALUATE BEST RESCORE (MULTIPASS (fusion) SCORE)
    files_to_merge = []
    if use_ctc:
        files_to_merge.append((opt_scales_job.out_real_scale_per_name[Scales.CTC.value], scores[Scales.CTC.value]))
    if use_prior:
        files_to_merge.append((opt_scales_job.out_real_scale_per_name[Scales.PRIOR.value], scores[Scales.PRIOR.value]))
    if use_sllm:
        files_to_merge.append((opt_scales_job.out_real_scale_per_name[Scales.SLLM.value], scores[Scales.SLLM.value]))
    if use_llm:
        files_to_merge.append((opt_scales_job.out_real_scale_per_name[Scales.LLM.value], scores[Scales.LLM.value]))
    combined_scores = SearchCombineScoresJob(files_to_merge).out_search_results
    best_rescore = SearchTakeBestJob(combined_scores).out_best_search_results
    search_ctm = SearchWordsToCTMJob(
        recog_words_file=best_rescore,
        bliss_corpus=tune_dataset_ref,
    ).out_ctm_file
    stm_file = CorpusToStmJob(bliss_corpus=tune_dataset_ref).out_stm_path
    sclite_job = ScliteJob(ref=stm_file, hyp=search_ctm, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=1)
    tk.register_output(f"{recog_path}/tune_rescore/wer", sclite_job.out_wer)
    tk.register_output(f"{recog_path}/tune_rescore/report", sclite_job.out_report_dir)

    # OUTPUT
    scales = copy.deepcopy(opt_scales_job.out_real_scale_per_name)
    if use_prior:
        scales[Scales.PRIOR.value] *= -1.0 # negative sign is applyed ad forward step (V4)
    return scales, autoscale_id


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
        mem_rqmt=rqmt["mem"],
        time_rqmt=1,
        device="gpu",
        cpu_rqmt=8 if rqmt["mem"] < 30 else 16,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["search_out.py.gz"],
    )
    search_job.add_alias(prefix_name + "/search_job")
    tk.register_output(prefix_name + "/search_out.py.gz", search_job.out_files["search_out.py.gz"])

    return search_job.out_files["search_out.py.gz"]
