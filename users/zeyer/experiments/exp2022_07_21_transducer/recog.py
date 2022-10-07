"""
recog helpers
"""


# Python stdlib imports
from __future__ import annotations
from typing import Dict, Any, Protocol, Tuple, Optional
# sisyphus imports
from sisyphus import Job, Task
# i6_core imports
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.search import ReturnnSearchJobV2, SearchBPEtoWordsJob
from i6_core.returnn.search import SearchRemoveLabelJob, SearchBeamJoinScoresJob, SearchTakeBestJob
# returnn_common imports
from returnn_common.datasets.interface import DatasetConfig
from returnn_common import nn
# i6_experiments.common imports
from i6_experiments.common.setups.returnn_common import serialization

# i6_experiments.users imports
from .task import Task, ScoreResultCollection
from .model import ModelWithCheckpoint, ModelWithCheckpoints, ModelDef, RecogDef
from i6_experiments.users.zeyer.datasets.base import RecogOutput
from i6_experiments.users.zeyer.returnn.training import GetRelevantEpochsFromTrainingJob
from i6_experiments.users.zeyer import tools_paths


def recog_training_exp(task: Task, model: ModelWithCheckpoints, recog_def: RecogDef):
    """recog on all relevant epochs"""
    # TODO ...
    GetRelevantEpochsFromTrainingJob(

    )


def recog_model(task: Task, model: ModelWithCheckpoint, recog_def: RecogDef) -> ScoreResultCollection:
    """recog"""
    outputs = {}
    for name, dataset in task.eval_datasets.items():
        recog_out = search_dataset(dataset=dataset, model=model, recog_def=recog_def)
        for f in task.recog_post_proc_funcs:
            recog_out = f(recog_out)
        score_out = task.score_recog_output_func(dataset, recog_out)
        outputs[name] = score_out
    return task.collect_score_results_func(outputs)


def search_dataset(dataset: DatasetConfig, model: ModelWithCheckpoint, recog_def: RecogDef) -> RecogOutput:
    """
    recog on the specific dataset
    """
    search_job = ReturnnSearchJobV2(
        search_data=dataset.get_main_dataset(),
        model_checkpoint=model.checkpoint,
        returnn_config=search_config(dataset, model.definition, recog_def),
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    res = search_job.out_search_file
    if recog_def.output_blank_label:
        res = SearchRemoveLabelJob(res, remove_label=recog_def.output_blank_label).out_search_results
        res = SearchBeamJoinScoresJob(res).out_search_results
    if recog_def.output_with_beam:
        res = SearchTakeBestJob(res).out_best_search_results
    return RecogOutput(output=res)


def search_config(dataset: DatasetConfig, model_def: ModelDef, recog_def: RecogDef) -> ReturnnConfig:
    """
    config for search
    """

    returnn_recog_config_dict = dict(
        use_tensorflow=True,
        behavior_version=12,

        # dataset
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),
        dev=dataset.get_main_dataset(),
    )

    returnn_recog_config = ReturnnConfig(
        returnn_recog_config_dict,
        python_epilog=[serialization.Collection(
            [
                serialization.NonhashedCode(
                    nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(
                        dataset.get_extern_data())),
                serialization.Import(model_def, "_model_def", ignore_import_as_for_hash=True),
                serialization.Import(recog_def, "_recog_def", ignore_import_as_for_hash=True),
                serialization.Import(_returnn_get_network, "get_network", use_for_hash=False),
                serialization.ExplicitHash({
                    # Increase the version whenever some incompatible change is made in this recog() function,
                    # which influences the outcome, but would otherwise not influence the hash.
                    "version": 1,
                }),
                serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                serialization.PythonCacheManagerFunctionNonhashedCode,
                serialization.PythonModelineNonhashedCode,
            ]
        )],
        post_config=dict(  # not hashed
            log_batch_size=True,
            tf_log_memory_usage=True,
            tf_session_opts={"gpu_options": {"allow_growth": True}},
            # debug_add_check_numerics_ops = True
            # debug_add_check_numerics_on_output = True
            # flat_net_construction=True,

            # With num seqs > 1 in a batch, the search result might depend on the batch for RNN-T or AED,
            # when the output seq length (amount of align labels) is variable.
            # For RNA, this is always fixed (same as encoder seq len), so there it should not have an effect.
            # In any case, the effect should be low.
            batching="sorted",
            batch_size=20000,
            max_seqs=200,
        ),
        sort_config=False,
    )

    return returnn_recog_config


def bpe_to_words(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    words = SearchBPEtoWordsJob(bpe.output).out_word_search_results
    return RecogOutput(output=words)


class IDecoder(Protocol):
    """
    Decoder interface.
    If there is any encoder output, this would already be computed and stored in here,
    as well as other necessary information, such as batch dims.
    """
    target_spatial_dim: Optional[nn.Dim] = None  # must be given when loop.unstack is used
    include_eos: bool  # eg true for transducer, false for AED

    def max_seq_len(self) -> nn.Tensor:
        """max seq len"""
        raise NotImplementedError

    def initial_state(self) -> nn.LayerState:
        """initial state"""
        raise NotImplementedError

    def bos_label(self) -> nn.Tensor:
        """begin-of-sequence (BOS) label"""
        raise NotImplementedError

    def __call__(self, prev_target: nn.Tensor, *, state: nn.LayerState) -> Tuple[nn.Tensor, nn.LayerState]:
        """return log_prob, new_state"""
        raise NotImplementedError

    def end(self, target: nn.Tensor, *, state: nn.LayerState) -> nn.Tensor:
        """
        Checks whether the (new) target, or (new) state reached the end-of-sequence (EOS).
        Only called when target_spatial_dim is not defined, i.e. when the length is dynamic.

        :return: bool tensor, True if EOS reached
        """
        raise nn.OptionalNotImplementedError


def beam_search(decoder: IDecoder, *, beam_size: int = 12) -> nn.Tensor:
    """search"""
    loop = nn.Loop(axis=decoder.target_spatial_dim)
    loop.max_seq_len = decoder.max_seq_len()
    loop.state.decoder = decoder.initial_state()
    loop.state.target = decoder.bos_label()
    with loop:
        log_prob, loop.state.decoder = decoder(loop.state.target, state=loop.state.decoder)
        loop.state.target = nn.choice(
            log_prob, input_type="log_prob",
            target=None, search=True, beam_size=beam_size,
            length_normalization=False)
        if not decoder.target_spatial_dim:
            loop.end(decoder.end(loop.state.target, state=loop.state.decoder), include_eos=decoder.include_eos)
        found = loop.stack(loop.state.target)
    return found


def _returnn_get_network(*, epoch: int, **_kwargs_unused) -> Dict[str, Any]:
    """called from the RETURNN config"""
    from returnn_common import nn
    from returnn.config import get_global_config
    from returnn.tf.util.data import Data
    nn.reset_default_root_name_ctx()
    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    extern_data_dict = config.typed_value("extern_data")
    data = Data(name=default_input_key, **extern_data_dict[default_input_key])
    targets = Data(name=default_target_key, **extern_data_dict[default_target_key])
    data_spatial_dim = data.get_time_dim_tag()
    data = nn.get_extern_data(data)
    targets = nn.get_extern_data(targets)
    model_def = config.typed_value("_model_def")
    model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.feature_dim)
    recog_def = config.typed_value("_recog_def")
    recog_out = recog_def(
        model=model,
        data=data, data_spatial_dim=data_spatial_dim, targets_dim=targets.feature_dim)
    assert isinstance(recog_out, nn.Tensor)
    recog_out.mark_as_default_output()
    net_dict = nn.get_returnn_config().get_net_dict_raw_dict(root_module=model)
    return net_dict


class SummarizeRecogTrainExp(Job):
    """collect all info from recogs"""

    def __init__(self):
        super(SummarizeRecogTrainExp, self).__init__()
        # TODO ...
        pass
