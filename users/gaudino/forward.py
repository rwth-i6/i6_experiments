"""
Generic recog, for the model interfaces defined in model_interfaces.py
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Union, Any, Dict, Sequence, Collection, Iterator, Callable

import sisyphus
from sisyphus import tk
from i6_core.util import instanciate_delayed

from i6_core.returnn import ReturnnConfig
from i6_core.returnn.search import ReturnnSearchJobV2, SearchRemoveLabelJob, SearchTakeBestJob
from i6_core.returnn.forward import ReturnnForwardJobV2
from returnn_common import nn
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.common.setups.returnn_common import serialization

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.datasets.task import Task
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput, ScoreResultCollection
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, ModelWithCheckpoint, ModelWithCheckpoints
from i6_experiments.users.zeyer.returnn.training import get_relevant_epochs_from_training_learning_rate_scores

if TYPE_CHECKING:
    from returnn.tensor import TensorDict

def forward_model(
    task: Task,
    model: ModelWithCheckpoint,
    recog_def: RecogDef,
    *,
    search_post_config: Optional[Dict[str, Any]] = None,
    search_mem_rqmt: Union[int, float] = 6,
    dev_sets: Optional[Collection[str]] = None,
    model_args: Optional[Dict[str, Any]] = None,
    search_args: Optional[Dict[str, Any]] = None,
    prefix_name: str,
    forward_lm: bool = False,
) -> ScoreResultCollection:
    """recog"""
    if dev_sets is not None:
        assert all(k in task.eval_datasets for k in dev_sets)
    outputs = {}
    for name, dataset in task.eval_datasets.items():
        if dev_sets is not None:
            if name not in dev_sets:
                continue
        recog_out = forward_dataset(dataset=dataset, model=model, recog_def=recog_def,
                                    search_post_config=search_post_config, search_mem_rqmt=search_mem_rqmt,
                                    recog_post_proc_funcs=task.recog_post_proc_funcs, model_args=model_args,
                                    search_args=search_args, prefix_name=prefix_name + f"/{name}", forward_lm=forward_lm)
        # score_out = task.score_recog_output_func(dataset, recog_out)
        outputs[name] = recog_out
    return outputs

def forward_custom_dataset(
    *,
    dataset: DatasetConfig,
    model: ModelWithCheckpoint,
    recog_def: RecogDef,
    search_post_config: Optional[Dict[str, Any]] = None,
    search_mem_rqmt: Union[int, float] = 6,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    model_args: Optional[Dict[str, Any]] = None,
    search_args: Optional[Dict[str, Any]] = None,
    prefix_name: str,
    forward_lm: bool = False,
) -> RecogOutput:
    """
    recog on the specific dataset
    """
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=model.checkpoint,
        returnn_config=forward_config(dataset, model.definition, recog_def, post_config=search_post_config,
                                      search_args=search_args, model_args=model_args, forward_lm=forward_lm),
        output_files=[_v2_forward_out_filename],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        mem_rqmt=search_mem_rqmt,
    )
    forward_job.add_alias(prefix_name + "/forward_custom_job")
    res = forward_job.out_files[_v2_forward_out_filename]

    # if recog_def.output_blank_label:
    #     res = SearchRemoveLabelJob(res, remove_label=recog_def.output_blank_label, output_gzip=True).out_search_results
    # for f in recog_post_proc_funcs:  # for example BPE to words
    #     res = f(RecogOutput(output=res)).output
    # if recog_def.output_with_beam:
    #     # Don't join scores here (SearchBeamJoinScoresJob).
    #     #   It's not clear whether this is helpful in general.
    #     #   As our beam sizes are very small, this might boost some hyps too much.
    #     res = SearchTakeBestJob(res, output_gzip=True).out_best_search_results
    return RecogOutput(output=res)


def forward_dataset(
    *,
    dataset: DatasetConfig,
    model: ModelWithCheckpoint,
    recog_def: RecogDef,
    search_post_config: Optional[Dict[str, Any]] = None,
    search_mem_rqmt: Union[int, float] = 6,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    model_args: Optional[Dict[str, Any]] = None,
    search_args: Optional[Dict[str, Any]] = None,
    prefix_name: str,
    forward_lm: bool = False,
) -> RecogOutput:
    """
    recog on the specific dataset
    """
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=model.checkpoint,
        returnn_config=forward_config(dataset, model.definition, recog_def, post_config=search_post_config,
                                      search_args=search_args, model_args=model_args, forward_lm=forward_lm),
        output_files=[_v2_forward_out_filename],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        mem_rqmt=search_mem_rqmt,
    )
    forward_job.add_alias(prefix_name + "/forward_gt_job")
    res = forward_job.out_files[_v2_forward_out_filename]

    # if recog_def.output_blank_label:
    #     res = SearchRemoveLabelJob(res, remove_label=recog_def.output_blank_label, output_gzip=True).out_search_results
    # for f in recog_post_proc_funcs:  # for example BPE to words
    #     res = f(RecogOutput(output=res)).output
    # if recog_def.output_with_beam:
    #     # Don't join scores here (SearchBeamJoinScoresJob).
    #     #   It's not clear whether this is helpful in general.
    #     #   As our beam sizes are very small, this might boost some hyps too much.
    #     res = SearchTakeBestJob(res, output_gzip=True).out_best_search_results
    return RecogOutput(output=res)


# Those are applied for both training, recog and potential others.
# The values are only used if they are neither set in config nor post_config already.
# They should also not infer with other things from the epilog.
SharedPostConfig = {
    # In case pretraining overwrites some of these, they need a default.
    "accum_grad_multiple_step": None,
    "use_last_best_model": None,
}


def forward_config(
    dataset: DatasetConfig,
    model_def: ModelDef,
    recog_def: RecogDef,
    *,
    post_config: Optional[Dict[str, Any]] = None,
    search_args: Optional[Dict[str, Any]] = None,
    model_args: Optional[Dict[str, Any]] = None,
    forward_lm: bool = False,
) -> ReturnnConfig:
    returnn_recog_config_dict = dict(
        backend=model_def.backend,
        behavior_version=model_def.behavior_version,
        # dataset
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),
        forward_data=dataset.get_main_dataset(),
    )

    extern_data_raw = dataset.get_extern_data()
    # The extern_data is anyway not hashed, so we can also instanciate any delayed objects here.
    # It's not hashed because we assume that all aspects of the dataset are already covered
    # by the datasets itself as part in the config above.
    extern_data_raw = instanciate_delayed(extern_data_raw)

    returnn_recog_config_dict.update({
        "search_args": search_args,
        "model_args": model_args,
    })

    returnn_recog_config = ReturnnConfig(
        config=returnn_recog_config_dict,
        python_epilog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(
                        nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
                    ),
                    serialization.Import(model_def, import_as="_model_def", ignore_import_as_for_hash=True),
                    serialization.Import(_returnn_v2_get_model, import_as="get_model"),
                    serialization.Import(recog_def, import_as="_recog_def", ignore_import_as_for_hash=True),
                    serialization.Import(_returnn_v2_forward_lm_step if forward_lm else _returnn_v2_forward_step, import_as="forward_step"),
                    serialization.Import(_returnn_v2_get_forward_lm_callback if forward_lm else _returnn_v2_get_forward_callback, import_as="forward_callback"),
                    serialization.ExplicitHash(
                        {
                            # Increase the version whenever some incompatible change is made in this recog() function,
                            # which influences the outcome, but would otherwise not influence the hash.
                            "version": 2,
                        }
                    ),
                    serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                    serialization.PythonCacheManagerFunctionNonhashedCode,
                    serialization.PythonModelineNonhashedCode,
                ]
            )
        ],
        post_config=dict(  # not hashed
            log_batch_size=True,
            # debug_add_check_numerics_ops = True
            # debug_add_check_numerics_on_output = True
            # flat_net_construction=True,
        ),
        sort_config=False,
    )

    (returnn_recog_config.config if recog_def.batch_size_dependent else returnn_recog_config.post_config).update(
        dict(
            batching="sorted",
            batch_size=20000 * model_def.batch_size_factor,
            max_seqs=model_def.max_seqs,
        )
    )

    if post_config:
        returnn_recog_config.post_config.update(post_config)

    for k, v in SharedPostConfig.items():
        if k in returnn_recog_config.config or k in returnn_recog_config.post_config:
            continue
        returnn_recog_config.post_config[k] = v

    return returnn_recog_config


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
    model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
    recog_def = config.typed_value("_recog_def")
    recog_out = recog_def(model=model, data=data, data_spatial_dim=data_spatial_dim, targets_dim=targets.sparse_dim)
    assert isinstance(recog_out, nn.Tensor)
    recog_out.mark_as_default_output()
    net_dict = nn.get_returnn_config().get_net_dict_raw_dict(root_module=model)
    return net_dict


def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
    from returnn.tensor import Tensor
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    extern_data_dict = config.typed_value("extern_data")
    data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    assert targets.sparse_dim and targets.sparse_dim.vocab, f"no vocab for {targets}"

    model_def = config.typed_value("_model_def")
    model_args = config.typed_value("model_args")
    search_args = config.typed_value("search_args")
    model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim, model_args=model_args, search_args=search_args)
    return model


def _returnn_v2_forward_step(*, model, extern_data: TensorDict, **_kwargs_unused):
    import returnn.frontend as rf
    from returnn.tensor import batch_dim
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    data = extern_data[default_input_key]
    target = extern_data[default_target_key]
    data_spatial_dim = data.get_time_dim_tag()
    recog_def = config.typed_value("_recog_def")
    recog_out = recog_def(model=model, data=data, data_spatial_dim=data_spatial_dim, ground_truth=target)
    # recog results including beam {batch, beam, out_spatial},
    # log probs {batch, beam},
    # out_spatial_dim,
    # final beam_dim
    hyps, scores, out_spatial_dim, beam_dim = recog_out
    rf.get_run_ctx().mark_as_output(hyps, "hyps", dims=[batch_dim, beam_dim, out_spatial_dim])
    rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim, beam_dim])


def _returnn_v2_forward_lm_step(*, model, extern_data: TensorDict, **_kwargs_unused):
    import returnn.frontend as rf
    from returnn.tensor import batch_dim
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    data = extern_data[default_input_key]
    target = extern_data[default_target_key]
    data_spatial_dim = data.get_time_dim_tag()
    recog_def = config.typed_value("_recog_def")
    recog_out = recog_def(model=model, data=data, data_spatial_dim=data_spatial_dim, ground_truth=target)

    hyps, scores = recog_out
    rf.get_run_ctx().mark_as_output(hyps, "hyps", dims=[batch_dim, hyps.dims[1]])
    rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim])


_v2_forward_out_filename = "output.py.gz"


def _returnn_v2_get_forward_callback():
    from returnn.tensor import Tensor, TensorDict
    from returnn.forward_iface import ForwardCallbackIface

    class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.out_file = None

        def init(self, *, model):
            import gzip

            self.out_file = gzip.open(_v2_forward_out_filename, "wt")
            self.out_file.write("{\n")

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            import numpy as np
            hyps: Tensor = outputs["hyps"]  # [beam, out_spatial]
            scores: Tensor = outputs["scores"]  # [beam]
            assert hyps.sparse_dim and hyps.sparse_dim.vocab  # should come from the model
            assert hyps.dims[1].dyn_size_ext, f"hyps {hyps} do not define seq lengths"
            hyps_len_raw = hyps.dims[1].dyn_size_ext.raw_tensor # [beam]
            if len(hyps_len_raw.shape) == 0:
                hyps_len_raw = hyps_len_raw[np.newaxis]
            assert hyps.raw_tensor.shape[:1] == hyps_len_raw.shape == scores.raw_tensor.shape  # (beam,)
            num_beam = hyps.raw_tensor.shape[0]
            # Consistent to old search task, list[(float,str)].
            self.out_file.write(f"{seq_tag!r}: [\n")
            for i in range(num_beam):
                score = float(scores.raw_tensor[i])
                hyp_ids = hyps.raw_tensor[i, : hyps_len_raw[i]]
                hyp_serialized = hyps.sparse_dim.vocab.get_seq_labels(hyp_ids)
                self.out_file.write(f"  ({score!r}, {hyp_serialized!r}),\n")
            self.out_file.write("],\n")

        def finish(self):
            self.out_file.write("}\n")
            self.out_file.close()

    return _ReturnnRecogV2ForwardCallbackIface()


def _returnn_v2_get_forward_lm_callback():
    from returnn.tensor import Tensor, TensorDict
    from returnn.forward_iface import ForwardCallbackIface

    class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.out_file = None

        def init(self, *, model):
            import gzip

            self.out_file = gzip.open(_v2_forward_out_filename, "wt")
            self.out_file.write("{\n")

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            import numpy as np
            hyps: Tensor = outputs["hyps"]  # [beam, out_spatial]
            scores: Tensor = outputs["scores"]  # [beam]
            assert hyps.sparse_dim and hyps.sparse_dim.vocab  # should come from the model
            assert hyps.dims[0].dyn_size_ext, f"hyps {hyps} do not define seq lengths"
            hyps_len_raw = hyps.dims[0].dyn_size_ext.raw_tensor # [beam]
            # Consistent to old search task, list[(float,str)].
            self.out_file.write(f"{seq_tag!r}: [\n")
            score = float(scores.raw_tensor)
            hyp_ids = hyps.raw_tensor[: hyps_len_raw]
            hyp_serialized = hyps.sparse_dim.vocab.get_seq_labels(hyp_ids)
            self.out_file.write(f"  ({score!r}, {int(hyps_len_raw)!r}, {hyp_serialized!r}),\n")
            self.out_file.write("],\n")

        def finish(self):
            self.out_file.write("}\n")
            self.out_file.close()

    return _ReturnnRecogV2ForwardCallbackIface()

