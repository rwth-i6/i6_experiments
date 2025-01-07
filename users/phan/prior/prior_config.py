"""
Generic recog, for the model interfaces defined in model_interfaces.py
https://github.com/rwth-i6/i6_experiments/blob/main/users/gaudino/recog_2.py#L674
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Union, Any, Dict, Sequence, Collection, Iterator, Callable

import sisyphus
from sisyphus import tk
from i6_core.util import instanciate_delayed

from i6_core.returnn import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob, PtCheckpoint, AverageTorchCheckpointsJob
from i6_core.returnn.search import ReturnnSearchJobV2, SearchRemoveLabelJob, SearchTakeBestJob
from i6_core.returnn.forward import ReturnnForwardJobV2
from returnn_common import nn
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.common.setups import serialization
from i6_experiments.users.zeyer.utils.serialization import get_import_py_code

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.gaudino.datasets.task import Task
from i6_experiments.users.gaudino.datasets.score_results import RecogOutput, ScoreResultCollection
from i6_experiments.users.gaudino.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, serialize_model_def
from i6_experiments.users.gaudino.model_with_checkpoints import ModelWithCheckpoint, ModelWithCheckpoints

if TYPE_CHECKING:
    from returnn.tensor import TensorDict

def compute_prior_job(
    task: Task,
    model: ModelWithCheckpoint,
    # epoch: int,
    recog_def: RecogDef,
    *,
    config: Optional[Dict[str, Any]] = None,
    search_rqmt: Optional[Dict[str, Any]] = None,
    search_mem_rqmt: Union[int, float] = 6,
    search_post_config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = "gpu",
):
    """compute prior"""
    train_set = task.train_dataset
    out_files = [_prior_out_filename]
    search_job = ReturnnForwardJobV2(
        model_checkpoint=model.checkpoint,
        returnn_config=search_config_v2(
            train_set,
            model.definition,
            recog_def,
            config=config,
            get_forward_callback=_returnn_v2_get_prior_callback,
            forward_step=_returnn_v2_forward_step_prior,
            post_config=search_post_config
        ),
        output_files=out_files,
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        mem_rqmt=search_mem_rqmt,
        device=device,
    )
    if search_rqmt:
        search_job.rqmt.update(search_rqmt)
    # res = search_job.out_files[_prior_out_filename]
    return search_job


# Those are applied for both training, recog and potential others.
# The values are only used if they are neither set in config nor post_config already.
# They should also not infer with other things from the epilog.
SharedPostConfig = {
    # In case pretraining overwrites some of these, they need a default.
    "accum_grad_multiple_step": None,
    "use_last_best_model": None,
}

def search_config_v2(
    dataset: DatasetConfig,
    model_def: Union[ModelDef, ModelDefWithCfg],
    recog_def: RecogDef,
    *,
    config: Optional[Dict[str, Any]] = None,
    get_forward_callback = None,
    forward_step = None,
    post_config: Optional[Dict[str, Any]] = None,
) -> ReturnnConfig:
    """
    Create config for search.

    v2: Use any backend (usually PyTorch) and the new API (get_model, forward_step).

    TODO should use sth like unhashed_package_root (https://github.com/rwth-i6/i6_experiments/pull/157)
    """
    from i6_experiments.common.setups.returnn.serialization import get_serializable_config

    returnn_recog_config_dict = dict(
        backend=model_def.backend,
        behavior_version=model_def.behavior_version,
        # dataset
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),
        forward_data=dataset.get_main_dataset(),
        # forward_data=dataset.get_main_dataset(),
    )
    if config:
        returnn_recog_config_dict.update(config)
    if isinstance(model_def, ModelDefWithCfg):
        returnn_recog_config_dict.update(model_def.config)

    extern_data_raw = dataset.get_extern_data()
    # The extern_data is anyway not hashed, so we can also instanciate any delayed objects here.
    # It's not hashed because we assume that all aspects of the dataset are already covered
    # by the datasets itself as part in the config above.
    extern_data_raw = instanciate_delayed(extern_data_raw)

    returnn_recog_config = ReturnnConfig(
        config=returnn_recog_config_dict,
        python_epilog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(get_import_py_code()),
                    serialization.NonhashedCode(
                        nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
                    ),
                    *serialize_model_def(model_def),
                    serialization.Import(_returnn_v2_get_model, import_as="get_model"),
                    serialization.Import(recog_def, import_as="_recog_def", ignore_import_as_for_hash=True),
                    serialization.Import(forward_step if forward_step else _returnn_v2_forward_step, import_as="forward_step"),
                    serialization.Import(get_forward_callback if get_forward_callback else _returnn_v2_get_forward_callback, import_as="forward_callback"),
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
            torch_log_memory_usage=True,
            watch_memory=True,
            use_lovely_tensors=True,
        ),
        sort_config=False,
    )

    # There might be some further functions in the config, e.g. some dataset postprocessing.
    returnn_recog_config = get_serializable_config(
        returnn_recog_config,
        # The only dim tags we directly have in the config are via extern_data, maybe also model_outputs.
        # All other dim tags are inside functions such as get_model or train_step,
        # so we do not need to care about them here, only about the serialization of those functions.
        # Those dim tags and those functions are already handled above.
        serialize_dim_tags=False,
    )

    batch_size_dependent = recog_def.batch_size_dependent
    if "__batch_size_dependent" in returnn_recog_config.config:
        batch_size_dependent = returnn_recog_config.config.pop("__batch_size_dependent")
    if "__batch_size_dependent" in returnn_recog_config.post_config:
        batch_size_dependent = returnn_recog_config.post_config.pop("__batch_size_dependent")
    for k, v in dict(
        batching="sorted",
        batch_size=20000 * model_def.batch_size_factor,
        max_seqs=200,
    ).items():
        if k in returnn_recog_config.config:
            v = returnn_recog_config.config.pop(k)
        if k in returnn_recog_config.post_config:
            v = returnn_recog_config.post_config.pop(k)
        (returnn_recog_config.config if batch_size_dependent else returnn_recog_config.post_config)[k] = v

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
    if model_args:
        model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim, model_args=model_args, search_args=search_args)
    else:
        model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
    return model


def _returnn_v2_forward_step(*, model, extern_data: TensorDict, **_kwargs_unused):
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim, batch_dim
    from returnn.config import get_global_config

    if rf.is_executing_eagerly():
        batch_size = int(batch_dim.get_dim_value())
        for batch_idx in range(batch_size):
            seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
            print(f"batch {batch_idx+1}/{batch_size} seq_tag: {seq_tag!r}")

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()
    recog_def = config.typed_value("_recog_def")
    extra = {}

    search_args = config.typed_value("search_args")
    if search_args:
        extra["search_args"] = search_args

    if config.bool("cheating", False):
        default_target_key = config.typed_value("target")
        targets = extern_data[default_target_key]
        extra.update(dict(targets=targets, targets_spatial_dim=targets.get_time_dim_tag()))
    recog_out = recog_def(model=model, data=data, data_spatial_dim=data_spatial_dim, **extra)
    if len(recog_out) == 5:
        # recog results including beam {batch, beam, out_spatial},
        # log probs {batch, beam},
        # extra outputs {batch, beam, ...},
        # out_spatial_dim,
        # final beam_dim
        assert len(recog_out) == 5, f"mismatch, got {len(recog_out)} outputs with recog_def_ext=True"
        hyps, scores, extra, out_spatial_dim, beam_dim = recog_out
    elif len(recog_out) == 4:
        # same without extra outputs
        assert len(recog_out) == 4, f"mismatch, got {len(recog_out)} outputs recog_def_ext=False"
        hyps, scores, out_spatial_dim, beam_dim = recog_out
        extra = {}
    else:
        raise ValueError(f"unexpected num outputs {len(recog_out)} from recog_def")
    assert isinstance(hyps, Tensor) and isinstance(scores, Tensor)
    assert isinstance(out_spatial_dim, Dim) and isinstance(beam_dim, Dim)
    rf.get_run_ctx().mark_as_output(hyps, "hyps", dims=[batch_dim, beam_dim, out_spatial_dim])
    rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim, beam_dim])
    assert isinstance(extra, dict)
    for k, v in extra.items():
        assert isinstance(k, str) and isinstance(v, Tensor)
        assert v.dims[:2] == (batch_dim, beam_dim)
        rf.get_run_ctx().mark_as_output(v, k, dims=v.dims)


def _returnn_v2_forward_step_prior(*, model, extern_data: TensorDict, **_kwargs_unused):
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim, batch_dim
    from returnn.config import get_global_config

    if rf.is_executing_eagerly():
        batch_size = int(batch_dim.get_dim_value())
        for batch_idx in range(batch_size):
            seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
            print(f"batch {batch_idx+1}/{batch_size} seq_tag: {seq_tag!r}")

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()
    recog_def = config.typed_value("_recog_def")

    ctc_out, enc_spatial_dim = recog_def(model=model, data=data, data_spatial_dim=data_spatial_dim)

    assert isinstance(ctc_out, Tensor)
    rf.get_run_ctx().mark_as_output(ctc_out, "ctc_out", dims=[batch_dim, enc_spatial_dim, model.target_dim_w_blank])


_v2_forward_out_filename = "output.py.gz"
_v2_forward_ext_out_filename = "output_ext.py.gz"

_prior_out_filename = "prior.txt"

def _returnn_v2_get_prior_callback():
    from typing import TextIO
    from returnn.tensor import Tensor, TensorDict
    from returnn.forward_iface import ForwardCallbackIface
    from returnn.config import get_global_config

    config = get_global_config()

    class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.out_file: Optional[TextIO] = None
            self.count = 0
            self.accum = None

        def init(self, *, model):

            self.out_file = open(_prior_out_filename, "wt")

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            ctc_out = outputs["ctc_out"].raw_tensor  # [beam, out_spatial]
            len = int(outputs["ctc_out"].dims[0].dyn_size_ext.raw_tensor)
            probs = ctc_out[:len].sum(0)
            if self.accum is None:
                self.accum = probs
            else:
                self.accum = self.accum + probs

            self.count += len

        def finish(self):
            self.accum = self.accum / self.count
            print(f"Sum should be close to 1.0: {self.accum.sum()}")
            for i in range(self.accum.shape[0]):
                self.out_file.write(f"{self.accum[i]}\n")
            self.out_file.close()

    return _ReturnnRecogV2ForwardCallbackIface()

