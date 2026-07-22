import ast

from i6_experiments.users.schmitt.experiments.exp2026_04_09_unsupervised_asr.sis_recipe.default_tools import (
    RETURNN_EXE,
    RETURNN_ONNX_EXE,
    RETURNN_ROOT,
    RETURNN_ONNX_ROOT,
)
from .pipeline import (
    search_single,
    get_checkpoint,
)
from typing import List, Optional, Dict, Any, List, Union, Iterator, Tuple, Callable
from dataclasses import dataclass, asdict
import copy
from enum import Enum

from sisyphus import tk, Task

from i6_core.returnn.compile import TorchOnnxExportJob
from i6_core.returnn.training import ReturnnTrainingJob, PtCheckpoint, ReturnnConfig
from i6_core.returnn.config import CodeWrapper
from i6_core.am.config import TdpValues, acoustic_model_config
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob, build_config_from_mapping
from i6_core.rasr.crp import CommonRasrParameters
from i6_core.lexicon.conversion import LexiconFromTextFileJob
from i6_core.text.processing import PipelineJob
from i6_core.serialization.base import CallImport

from i6_experiments.common.setups.serialization import PartialImport
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms.recognition.aed.beam_search import (
    DecoderConfig,
)
from i6_experiments.users.zeyer.datasets.score_results import (
    ScoreResultCollection,
    join_score_results,
    ScoreResult,
)
from i6_experiments.users.schmitt.lexicon.modification import (
    ReorderPhonemeInventoryByReturnnVocabJob,
    AddPhonemesAndLemmasToLexiconJob,
)
from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import NonhashedCode

from returnn.tensor import Tensor
from returnn.tensor.dim import batch_dim, Dim

from .data.common import TrainingDatasets
from .config import get_forward_config, get_export_onnx_config

default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}


class SummarizeScoreResultsJob(tk.Job):
    def __init__(self, score_results: Dict[str, ScoreResultCollection]):
        super().__init__()
        self.score_results = score_results
        self.out_results_all_epochs_json = self.output_path("results_all_epoch.json")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        import json

        with open(self.out_results_all_epochs_json.get_path(), "w") as f:
            f.write("{\n")
            count = 0
            for epoch, score in sorted(self.score_results.items()):
                assert isinstance(score, ScoreResultCollection)
                if count > 0:
                    f.write(",\n")
                res = json.load(open(score.output.get_path()))
                f.write(f'  "{epoch}": {json.dumps(res)}')
                count += 1
            f.write("\n}\n")


class SummarizeScoreResultsJobV2(tk.Job):
    def __init__(self, score_results: Dict[str, ScoreResultCollection]):
        super().__init__()
        self.score_results = score_results
        self.out_results_all_epochs_json = self.output_path("results_all_epoch.json")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        import json
        import ast

        with open(self.out_results_all_epochs_json.get_path(), "w") as f:
            f.write("{\n")
            count = 0
            for epoch, score in sorted(self.score_results.items()):
                assert isinstance(score, ScoreResultCollection)
                if count > 0:
                    f.write(",\n")
                # res = json.load(open(score.output.get_path()))
                res = ast.literal_eval(open(score.output.get_path()).read())
                f.write(f"  {epoch}: {{\n")
                max_key_len = max(map(lambda x: len(x), res.keys()))
                for key, value in res.items():
                    key_str = key.ljust(max_key_len + 2)
                    f.write(f"    {key_str}| {value},\n")
                    f.write("    " + "-" * max_key_len + "--------------\n")
                f.write("  }")
                count += 1
            f.write("\n}\n")


class JoinScoreResultsJobV2(tk.Job):
    """
    Joins the score results of multiple jobs into one ScoreResultCollection.
    """

    def __init__(self, score_results: Dict[str, ScoreResult]):
        self.score_results = score_results
        self.out_score_results = self.output_path("score_results.json")
        self.out_score_results_table = self.output_path("score_results.txt")

    def tasks(self) -> Iterator[Task]:
        """tasks"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        import ast
        import json

        res = {}
        max_key_len = max(map(lambda x: len(x), self.score_results.keys()))
        f = open(self.out_score_results_table.get_path(), "w")
        f.write("{\n")
        for key, score_result in self.score_results.items():
            # value_str = open(score_result.main_measure_value.get_path(), "r").read()
            value_str = score_result.main_measure_value.get()
            if isinstance(value_str, str):
                value_str = open(value_str).read()
                value = ast.literal_eval(value_str)
            else:
                assert isinstance(value_str, float), f"Unexpected type for main_measure_value: {type(value_str)}"
                value = f"{value_str:.2f}"

            res[key] = value
            key_str = key.ljust(max_key_len + 2)
            f.write(f"  {key_str}| {value},\n")
            f.write("  " + "-" * max_key_len + "--------------\n")
        f.write("}\n")
        f.close()

        with open(self.out_score_results.get_path(), "w") as f:
            f.write(json.dumps(res))
            f.write("\n")


def join_score_results_v2(
    score_results: Dict[str, ScoreResult], main_measure_key: Optional[str] = None
) -> ScoreResultCollection:
    """join score results"""
    return ScoreResultCollection(
        main_measure_value=score_results[main_measure_key].main_measure_value if main_measure_key else None,
        output=JoinScoreResultsJobV2(score_results).out_score_results,
        individual_results=score_results,
    )


def eval_model(
    config: Dict,
    training_name: str,
    recog_name: str,
    train_job: ReturnnTrainingJob,
    train_args: Dict[str, Any],
    train_data: TrainingDatasets,
    base_decoder_config: DecoderConfig,
    test_data_dict: Dict[str, Any],
    checkpoints: List[Union[int, str]],
    decoder_module: str,
    callback_module: str,
    loss_name: str = "dev_loss_ce",
    extra_forward_config: Optional[ReturnnConfig] = None,
    main_eval_measure_key: str = "dev",
    rqmt: Optional[Dict[str, int]] = None,
    recog_post_proc_funcs: Optional[List[Callable[[tk.Path], tk.Path]]] = None,
    input_modality: str = "audio",
    output_modality: str = "text",
    mask_input: bool = False,
    masking_opts: Optional[Dict[str, Any]] = None,
    expansion_opts: Optional[Dict[str, Any]] = None,
):
    """
    :param input_modality: "audio" or "text" -- which modality is fed to the (shared) encoder.
    :param output_modality: "audio" or "text" -- which decoder/vocab is used to produce + score
        hypotheses. The default (input "audio", output "text") is standard ASR; setting both to the
        same modality is a same-modality reconstruction probing the denoising model.
    :param mask_input: if True, mask the encoder input like in training (requires ``masking_opts``).
    :param masking_opts: masking config (``mask_prob``/``min_span``/``max_span``), as in training.
    :param expansion_opts: if given (``{"min_dup", "max_dup"}``), additionally upsample the (masked)
        encoder input like the train-time text upsampling (applied after masking). The decode/score
        length stays at the original length.
    """
    assert input_modality in ("audio", "text"), input_modality
    assert output_modality in ("audio", "text"), output_modality

    default_data_key = config.get("default_data_key", "data")
    default_target_key = config.get("default_target_key", "text")
    # map modality -> datastream / extern_data key
    modality_to_key = {"audio": default_data_key, "text": default_target_key}
    input_data_key = modality_to_key[input_modality]
    output_data_key = modality_to_key[output_modality]

    # forward_step init args (hashed). Only add the reconstruction args when they differ from the
    # forward_step defaults (audio->text, no masking), so the hash of standard ASR recog jobs that
    # ran before these args existed stays unchanged.
    forward_step_args = asdict(base_decoder_config)
    if input_modality != "audio":
        forward_step_args["input_modality"] = input_modality
        forward_step_args["input_data_key"] = input_data_key
    if output_modality != "text":
        forward_step_args["output_modality"] = output_modality
    if mask_input:
        assert masking_opts is not None, "mask_input=True requires masking_opts"
        forward_step_args["masking_opts"] = masking_opts
    # only serialize expansion_opts when set, so existing recog job hashes stay unchanged.
    if expansion_opts is not None:
        forward_step_args["expansion_opts"] = expansion_opts

    # the encoder input key must be present in extern_data. The default extern_data only declares
    # the audio key ("data"); declare the text key as well when the input is text.
    add_text_to_extern_data = input_data_key != default_data_key

    out_search_files = {}
    result_collections = {}
    for checkpoint_name in checkpoints:
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

        returnn_search_config = get_forward_config(
            config=config,
            network_module=train_args["network_module"],
            extra_config=extra_forward_config if extra_forward_config else ReturnnConfig({}),
            net_args=train_args["net_args"],
            decoder_args=forward_step_args,
            decoder=decoder_module,
            callback_module=callback_module,
            datastreams=train_data.datastreams,
            callback_opts={
                "include_beam": True,
            },
            vocab_key=output_data_key,
            add_text_to_extern_data=add_text_to_extern_data,
        )

        outputs = {}
        search_ctms = {}
        out_search_files[checkpoint_name] = {}
        recog_path = f"{training_name}/{recog_name}/{str(checkpoint_name)}"
        for key, dataset in test_data_dict.items():
            recog_dataset_path = f"{recog_path}/{key}"
            score_result, out_search_files[checkpoint_name][key], search_ctm = search_single(
                recog_dataset_path,
                returnn_config=returnn_search_config,
                checkpoint=checkpoint,
                recognition_dataset=dataset,
                dataset_name=key,
                **default_returnn,
                rqmt=rqmt,
                vocab_opts=train_data.datastreams[output_data_key].as_returnn_targets_opts(),
                recog_post_proc_funcs=recog_post_proc_funcs,
                score_target_key=output_data_key,
            )
            search_ctms[key] = search_ctm
            outputs[key] = score_result
            tk.register_output(f"{recog_dataset_path}/wer", score_result.main_measure_value)
            tk.register_output(f"{recog_dataset_path}/report", score_result.report)

        score_collection_job = JoinScoreResultsJobV2(outputs)
        tk.register_output(f"{recog_path}/score_results", score_collection_job.out_score_results_table)
        result_collections[checkpoint_name] = join_score_results(outputs, main_measure_key=main_eval_measure_key)

    summarize_job = SummarizeScoreResultsJobV2(result_collections)
    tk.register_output(
        f"{training_name}/{recog_name}/results_all_epochs",
        summarize_job.out_results_all_epochs_json,
    )

    return out_search_files


def eval_model_rasr(
    recog_config: Dict,
    onnx_config: Dict,
    training_name: str,
    recog_name: str,
    train_job: ReturnnTrainingJob,
    train_args: Dict[str, Any],
    train_data: TrainingDatasets,
    test_data_dict: Dict[str, Any],
    checkpoints: List[Union[int, str]],
    decoder_module: str,
    export_forward_step: str,
    callback_module: str,
    recog_opts: Dict,
    base_decoder_config: Optional[DecoderConfig] = None,
    loss_name: str = "dev_loss_ce",
    extra_forward_config: Optional[ReturnnConfig] = None,
    main_eval_measure_key: str = "dev",
    rqmt: Optional[Dict[str, int]] = None,
    recog_post_proc_funcs: Optional[List[Callable[[tk.Path], tk.Path]]] = None,
    input_modality: str = "audio",
    output_modality: str = "text",
    mask_input: bool = False,
    masking_opts: Optional[Dict[str, Any]] = None,
):
    """
    :param input_modality: "audio" or "text" -- which modality is fed to the (shared) encoder.
    :param output_modality: "audio" or "text" -- which decoder/vocab is used to produce + score
        hypotheses. The default (input "audio", output "text") is standard ASR; setting both to the
        same modality is a same-modality reconstruction probing the denoising model.
    :param mask_input: if True, mask the encoder input like in training (requires ``masking_opts``).
    :param masking_opts: masking config (``mask_prob``/``min_span``/``max_span``), as in training.
    """
    assert input_modality in ("audio", "text"), input_modality
    assert output_modality in ("audio", "text"), output_modality

    default_data_key = recog_config.get("default_data_key", "data")
    default_target_key = recog_config.get("default_target_key", "text")
    # map modality -> datastream / extern_data key
    modality_to_key = {"audio": default_data_key, "text": default_target_key}
    input_data_key = modality_to_key[input_modality]
    output_data_key = modality_to_key[output_modality]

    # forward_step init args (hashed). Only add the reconstruction args when they differ from the
    # forward_step defaults (audio->text, no masking), so the hash of standard ASR recog jobs that
    # ran before these args existed stays unchanged.
    forward_step_args = asdict(base_decoder_config) if base_decoder_config else {}
    if input_modality != "audio":
        forward_step_args["input_modality"] = input_modality
        forward_step_args["input_data_key"] = input_data_key
    if output_modality != "text":
        forward_step_args["output_modality"] = output_modality
    if mask_input:
        assert masking_opts is not None, "mask_input=True requires masking_opts"
        forward_step_args["masking_opts"] = masking_opts

    # the encoder input key must be present in extern_data. The default extern_data only declares
    # the audio key ("data"); declare the text key as well when the input is text.
    add_text_to_extern_data = input_data_key != default_data_key

    out_search_files = {}
    result_collections = {}
    for checkpoint_name in checkpoints:
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

        tensor_tokens_length = Tensor(name="tokens_length", dims=[batch_dim], dtype="int32")
        dim_tokens_length = Dim(dimension=tensor_tokens_length, name="tokens_length")
        tensor_enc_length = Tensor(name="enc_length", dims=[batch_dim], dtype="int32")
        dim_enc_length = Dim(dimension=tensor_enc_length, name="enc_length")
        # +3 for BOS, EOS, and MASK tokens
        dim_vocab = Dim(dimension=train_data.datastreams[default_target_key].vocab_size + 3, name="vocab")
        dim_enc_out = Dim(dimension=512, name="enc_feat")
        extern_data = {
            "encoder_output": {"available_for_inference": True, "dim_tags": (batch_dim, dim_enc_length, dim_enc_out)},
            "tokens": {
                "available_for_inference": True,
                "dim_tags": (batch_dim, dim_tokens_length),
                "sparse_dim": dim_vocab,
            },
        }
        returnn_onnx_config = get_export_onnx_config(
            config=onnx_config,
            network_module=train_args["network_module"],
            extra_config=extra_forward_config if extra_forward_config else ReturnnConfig({}),
            net_args=train_args["net_args"],
            decoder_args=forward_step_args,
            decoder=export_forward_step,
            base_config={"model_outputs": {"scores": {"dim_tags": (batch_dim, dim_vocab), "dtype": "float32"}}},
            extern_data=extern_data,
        )

        outputs = {}
        search_ctms = {}
        out_search_files[checkpoint_name] = {}
        recog_path = f"{training_name}/{recog_name}/{str(checkpoint_name)}"

        onnx_export_job = TorchOnnxExportJob(
            returnn_config=copy.deepcopy(returnn_onnx_config),
            checkpoint=checkpoint,
            input_names=["encoder_output", "encoder_output:size1", "tokens", "tokens:size1"],
            output_names=["scores"],
            returnn_python_exe=RETURNN_ONNX_EXE,
            returnn_root=RETURNN_ONNX_ROOT,
            verify_model=True,
        )
        tk.register_output(f"{recog_path}/onnx_model", onnx_export_job.out_onnx_model)
        onnx_export_job.add_alias(f"{recog_path}/onnx_export")

        lexicon = get_lexicon(
            vocab_file=train_data.datastreams[output_data_key].vocab,
            line_based_lexicon_file=recog_opts.pop("line_based_lexicon_file"),
        )
        label_scorer_config = get_stateless_label_scorer_config(
            exported_model=onnx_export_job.out_onnx_model,
            execution_provider_type="cuda",
        )
        lm_config = get_lm_config(get_arpa_lm_dict()["3gram"], scale=0.1)
        rasr_config = get_tree_labelsync_recog_config(
            lexicon_file=lexicon,
            label_scorer_configs=[label_scorer_config],
            lm_config=lm_config,
            length_norm_scale=1.0,
        )
        search_function = CallImport(
            code_object_path="i6_experiments.users.schmitt.experiments.exp2026_04_09_unsupervised_asr.models.recognition.discrete_audio_aed.rasr.forward_step._get_rasr_search_function",
            import_as="search_function",
            hashed_arguments={
                "config_file": rasr_config,
            },
            unhashed_arguments={},
            unhashed_package_root=None,
        )
        forward_step_args["search_function"] = CodeWrapper("search_function")
        python_prolog = [
            Collection(
                [
                    # needed for librasr import
                    NonhashedCode(
                        'sys.path.insert(0, "/work/asr4/lkleppel/rasr_dev/tree-labelsync-search/rasr/arch/linux-x86_64-standard")\n'
                    ),
                    search_function,
                ],
                make_local_package_copy=False,
                # packages={
                #     PACKAGE,
                # },
            ),
        ]
        returnn_search_config = get_forward_config(
            config=recog_config,
            network_module=train_args["network_module"],
            extra_config=extra_forward_config if extra_forward_config else ReturnnConfig({}),
            net_args=train_args["net_args"],
            decoder_args=forward_step_args,
            decoder=decoder_module,
            callback_module=callback_module,
            datastreams=train_data.datastreams,
            callback_opts={
                "include_beam": True,
            },
            vocab_key=output_data_key,
            add_text_to_extern_data=add_text_to_extern_data,
            # base_config=forward_base_config,
            python_prolog=python_prolog,
        )

        for key, dataset in test_data_dict.items():
            recog_dataset_path = f"{recog_path}/{key}"
            score_result, out_search_files[checkpoint_name][key], search_ctm = search_single(
                recog_dataset_path,
                returnn_config=returnn_search_config,
                checkpoint=checkpoint,
                recognition_dataset=dataset,
                dataset_name=key,
                **default_returnn,
                rqmt=rqmt,
                vocab_opts=train_data.datastreams[output_data_key].as_returnn_targets_opts(),
                recog_post_proc_funcs=recog_post_proc_funcs,
                score_target_key=output_data_key,
            )
            search_ctms[key] = search_ctm
            outputs[key] = score_result
            tk.register_output(f"{recog_dataset_path}/wer", score_result.main_measure_value)
            tk.register_output(f"{recog_dataset_path}/report", score_result.report)

        score_collection_job = JoinScoreResultsJobV2(outputs)
        tk.register_output(f"{recog_path}/score_results", score_collection_job.out_score_results_table)
        result_collections[checkpoint_name] = join_score_results(outputs, main_measure_key=main_eval_measure_key)

    summarize_job = SummarizeScoreResultsJobV2(result_collections)
    tk.register_output(
        f"{training_name}/{recog_name}/results_all_epochs",
        summarize_job.out_results_all_epochs_json,
    )

    return out_search_files


class LabelsyncGlobalPruningStrategy(Enum):
    NONE = "none"
    ACTIVE_AGAINST_TERMINATED = "active-against-terminated"
    ALL = "all"


def _add_label_scorers_to_rasr_config(label_scorer_configs: List[RasrConfig], rasr_config: RasrConfig) -> None:
    if len(label_scorer_configs) == 1:
        rasr_config.label_scorer = label_scorer_configs[0]
    else:
        rasr_config.num_label_scorers = len(label_scorer_configs)
        for i, scorer_config in enumerate(label_scorer_configs, start=1):
            rasr_config[f"label-scorer-{i}"] = scorer_config


def get_stateless_label_scorer_config(
    exported_model: tk.Path,
    execution_provider_type: Optional[str] = None,
) -> RasrConfig:

    rasr_config = RasrConfig()
    rasr_config.type = "full-context-onnx"
    rasr_config.max_batch_size = 1
    rasr_config.start_label_index = 42

    rasr_config.onnx_model = RasrConfig()
    rasr_config.onnx_model.session = RasrConfig()
    rasr_config.onnx_model.session.file = exported_model
    rasr_config.onnx_model.session.inter_op_num_threads = 2
    rasr_config.onnx_model.session.intra_op_num_threads = 2

    rasr_config.onnx_model.io_map = RasrConfig()
    rasr_config.onnx_model.io_map.history = "tokens"
    rasr_config.onnx_model.io_map.history_size = "tokens:size1"
    rasr_config.onnx_model.io_map.encoder_states = "encoder_output"
    rasr_config.onnx_model.io_map.encoder_states_size = "encoder_output:size1"
    rasr_config.onnx_model.io_map.scores = "scores"

    if execution_provider_type:
        rasr_config.onnx_model.session.execution_provider_type = execution_provider_type

    return rasr_config


def get_tree_labelsync_recog_config(
    lexicon_file: tk.Path,
    label_scorer_configs: List[RasrConfig],
    am_config: Optional[RasrConfig] = None,
    lm_config: Optional[RasrConfig] = None,
    max_beam_size: int = 12,
    max_word_end_beam_size: Optional[int] = 12,
    score_threshold: Optional[float] = None,
    word_end_score_threshold: Optional[float] = None,
    length_norm_scale: Optional[float] = None,
    max_labels_per_time_step: int = 1,
    sentence_end_fallback: bool = True,
    log_stepwise_statistics: bool = False,
    logfile_suffix: str = "recog",
) -> tk.Path:
    crp = CommonRasrParameters()

    # LibRASR does not have a channel manager so the settings from `crp_add_default_output` don't work
    logfile_name = f"rasr.{logfile_suffix}.log"
    log_config = RasrConfig()
    log_config["*.log.channel"] = logfile_name
    log_config["*.warning.channel"] = logfile_name
    log_config["*.error.channel"] = logfile_name
    log_config["*.statistics.channel"] = logfile_name
    log_config["*.unbuffered"] = False

    log_post_config = RasrConfig()
    log_post_config["*.encoding"] = "UTF-8"
    crp.log_config = log_config  # type: ignore
    crp.log_post_config = log_post_config  # type: ignore
    crp.default_log_channel = logfile_name

    rasr_config, rasr_post_config = build_config_from_mapping(crp=crp, mapping={}, include_log_config=True)

    rasr_config.lib_rasr = RasrConfig()

    rasr_config.lib_rasr.lexicon = RasrConfig()

    rasr_config.lib_rasr.search_algorithm = RasrConfig()
    rasr_config.lib_rasr.search_algorithm.type = "tree-labelsync-beam-search"

    rasr_config.lib_rasr.lexicon.file = lexicon_file

    if lm_config is not None:
        rasr_config.lib_rasr.lm = lm_config
    else:
        rasr_config.lib_rasr.lm = RasrConfig()
        rasr_config.lib_rasr.lm.scale = 0.0

    if am_config is None:
        am_config = acoustic_model_config(
            states_per_phone=1,
            tdp_transition=TdpValues(loop=0.0, forward=0.0, skip="infinity", exit=0.0),
            tdp_silence=TdpValues(loop=0.0, forward=0.0, skip="infinity", exit=0.0),
            phon_history_length=0,
            phon_future_length=0,
        )
    rasr_config.lib_rasr.acoustic_model = am_config

    rasr_config.lib_rasr.search_algorithm.tree_builder_type = "aed"

    rasr_config.lib_rasr.search_algorithm.max_beam_size = max_beam_size
    if max_word_end_beam_size is not None:
        rasr_config.lib_rasr.search_algorithm.max_word_end_beam_size = max_word_end_beam_size
    if score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.score_threshold = score_threshold
    if word_end_score_threshold is not None:
        rasr_config.lib_rasr.search_algorithm.word_end_score_threshold = word_end_score_threshold
    if length_norm_scale is not None:
        rasr_config.lib_rasr.search_algorithm.length_norm_scale = length_norm_scale
    if max_labels_per_time_step is not None:
        rasr_config.lib_rasr.search_algorithm.max_labels_per_time_step = max_labels_per_time_step
    rasr_config.lib_rasr.search_algorithm.sentence_end_fall_back = sentence_end_fallback
    rasr_config.lib_rasr.search_algorithm.log_stepwise_statistics = log_stepwise_statistics

    _add_label_scorers_to_rasr_config(label_scorer_configs, rasr_config.lib_rasr)

    recog_rasr_config_path = WriteRasrConfigJob(rasr_config, rasr_post_config).out_config
    return recog_rasr_config_path


def get_lm_config(lm_path: tk.Path, scale=1.0) -> RasrConfig:
    config = RasrConfig()
    config.scale = scale
    if scale > 0.0:
        config.type = "ARPA"
        config.file = lm_path

    return config


def get_lexicon(vocab_file: tk.Path, line_based_lexicon_file: tk.Path):
    lowercase_lexicon = PipelineJob(line_based_lexicon_file, pipeline=["tr a-z A-Z"]).out
    lexicon = LexiconFromTextFileJob(
        text_file=lowercase_lexicon,
        variation="none",
    ).out_bliss_lexicon

    lexicon = AddPhonemesAndLemmasToLexiconJob(
        lexicon,
        phonemes=["<SIL>"],
        lemmas=[
            {
                "orth": ["<SIL>"],
                "phon": ["<SIL>"],
                "synt": [],  # empty, so it's not scored by LM
                "special": "silence",
            }
        ],
    ).out_lexicon

    lexicon = ReorderPhonemeInventoryByReturnnVocabJob(lex_to_modify=lexicon, vocab=vocab_file).out_lexicon

    lexicon = AddPhonemesAndLemmasToLexiconJob(
        lexicon,
        phonemes=["<MASK>", "<BOS>", "<EOS>"],
        lemmas=[
            {
                "orth": ["<EOS>"],
                "phon": ["<EOS>"],
                "synt": ["</s>"],
                "special": "sentence-boundary",
            }
        ],
    ).out_lexicon

    return lexicon
