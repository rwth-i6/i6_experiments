"""
recog helpers
"""


from __future__ import annotations
from typing import Dict, Any
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.search import ReturnnSearchJobV2, SearchBPEtoWordsJob
from returnn_common.datasets.interface import DatasetConfig
from returnn_common import nn
from i6_experiments.common.setups.returnn_common import serialization
from .task import Task, ScoreResultCollection
from .model import ModelWithCheckpoint
from i6_experiments.users.zeyer.datasets.base import RecogOutput
from i6_experiments.users.zeyer import tools_paths


def recog(task: Task, model: ModelWithCheckpoint) -> ScoreResultCollection:
    """recog"""
    outputs = {}
    for name, dataset in task.eval_datasets.items():
        recog_out = search_dataset(dataset=dataset, model=model)
        for f in task.recog_post_proc_funcs:
            recog_out = f(recog_out)
        score_out = task.score_recog_output_func(dataset, recog_out)
        outputs[name] = score_out
    return task.collect_score_results_func(outputs)


def search_dataset(dataset: DatasetConfig, model: ModelWithCheckpoint) -> RecogOutput:
    """
    recog on the specific dataset
    """

    returnn_recog_config_dict = dict(
        use_tensorflow=True,
        behavior_version=12,

        # dataset
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),
        dev=dataset.get_eval_datasets(),

        batching="sorted",
        batch_size=20000,
        max_seqs=200,
    )

    returnn_recog_config = ReturnnConfig(
        returnn_recog_config_dict,
        python_epilog=[serialization.Collection(
            [
                serialization.NonhashedCode(
                    nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(
                        dataset.get_extern_data())),
                serialization.Import(model.definition, "_model_def", ignore_import_as_for_hash=True),
                serialization.Import(model_search, "_search", ignore_import_as_for_hash=True),  # TODO...
                serialization.Import(_returnn_get_network, "get_network", use_for_hash=False),
                serialization.ExplicitHash({
                    # Increase the version whenever some incompatible change is made in this recog() function,
                    # which influences the outcome, but would otherwise not influence the hash.
                    "version": 1,
                }),
                serialization.PythonEnlargeStackWorkaroundNonhashedCode,
            ]
        )],
        post_config=dict(  # not hashed
            log_batch_size=True,
            tf_log_memory_usage=True,
            tf_session_opts={"gpu_options": {"allow_growth": True}},
            # debug_add_check_numerics_ops = True
            # debug_add_check_numerics_on_output = True
            # flat_net_construction=True,
        ),
        sort_config=False,
    )

    search_job = ReturnnSearchJobV2(
        search_data=dataset.get_main_dataset(),
        model_checkpoint=model.checkpoint,
        returnn_config=returnn_recog_config,
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    return RecogOutput(output=search_job.out_search_file)


def bpe_to_words(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    words = SearchBPEtoWordsJob(bpe.output).out_word_search_results
    return RecogOutput(output=words)


def model_search(decoder, *, beam_size: int = 12) -> nn.Tensor:
    """search"""
    loop = nn.Loop(axis=decoder.align_spatial_dim)
    loop.max_seq_len = decoder.max_seq_len()
    loop.state = decoder.initial_state()
    with loop:
        log_prob, loop.state.decoder = decoder(loop.state.target, state=loop.state.decoder)
        loop.state.target = nn.choice(log_prob, input_type="log_prob", target=None, search=True, beam_size=beam_size)
        loop.end(decoder.end(loop.state.target, loop.state.decoder), include_eos=False)
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
    model = model_def(epoch=epoch)
    search_def = config.typed_value("_search_def")
    search_def(
        model=model,
        data=data, data_spatial_dim=data_spatial_dim, targets_dim=targets.feature_dim)
    net_dict = nn.get_returnn_config().get_net_dict_raw_dict(root_module=model)
    return net_dict
