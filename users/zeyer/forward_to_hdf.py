"""
Forward model outputs (or anything) to HDF
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Any, Dict, Tuple

from sisyphus import tk
from i6_core.util import instanciate_delayed

from i6_core.returnn import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from returnn_common import nn
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.common.setups import serialization
from i6_experiments.users.zeyer.utils.serialization import get_import_py_code

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, ForwardDef, serialize_model_def
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim, TensorDict


def forward_to_hdf(
    *,
    dataset: DatasetConfig,
    model: Optional[ModelWithCheckpoint] = None,
    forward_def: ForwardDef,
    config: Optional[Dict[str, Any]] = None,
    forward_post_config: Optional[Dict[str, Any]] = None,
    forward_mem_rqmt: Union[int, float] = 6,
    forward_rqmt: Optional[Dict[str, Any]] = None,
    forward_alias_name: Optional[str] = None,
) -> tk.Path:
    """
    forward on the specific dataset

    :return: HDF file path
    """
    env_updates = None
    if (config and config.get("__env_updates")) or (forward_post_config and forward_post_config.get("__env_updates")):
        env_updates = (config and config.pop("__env_updates", None)) or (
            forward_post_config and forward_post_config.pop("__env_updates", None)
        )
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=model.checkpoint if model else None,
        returnn_config=_returnn_forward_config(
            dataset, model.definition if model else None, forward_def, config=config, post_config=forward_post_config
        ),
        output_files=[_hdf_out_filename],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        mem_rqmt=forward_mem_rqmt,
    )
    if forward_rqmt:
        forward_job.rqmt.update(forward_rqmt)
    if env_updates:
        for k, v in env_updates.items():
            forward_job.set_env(k, v)
    if forward_alias_name:
        forward_job.add_alias(forward_alias_name)
    return forward_job.out_files[_hdf_out_filename]


def forward_posteriors_to_hdf(
    *,
    model: ModelWithCheckpoint,
    model_output_kind: str = "logits",
    output_kind: str,
    dataset: DatasetConfig,
    backend: str = "torch",
    behavior_version: int = 21,
    **kwargs,
) -> tk.Path:
    """
    Forward model outputs

    :param model: after construction, the model will be called as:
        ``out, out_spatial_dim = model(input, in_spatial_dim=in_spatial_dim)``
        (This is the RETURNN ISeqDownsamplingEncoder interface.)
        ``out.feature_dim`` is expected to be set.
        Use ``model_output_kind`` to specify what kind of output you have in the model output ``out``.
        (If you have a model with a different interface, just call :collect_statistics` directly
         with your custom ``forward_def`` function.)
    :param model_output_kind: "logits", "log_prob" or "prob": what your model(...) returns
    :param output_kind: "logits", "log_prob" or "prob": what you want to have in the HDF
    :param dataset:
    :param backend:
    :param behavior_version:
    :param kwargs: passed to :func:`forward_to_hdf`
    :return: path to the HDF file
    """
    assert model_output_kind in {"logits", "log_prob", "prob"}
    return forward_to_hdf(
        model=model,
        dataset=dataset,
        forward_def=_model_returnn_forward,
        config={
            "backend": backend,
            "behavior_version": behavior_version,
            "_model_output_kind": model_output_kind,
            "_output_kind": output_kind,
        },
        **kwargs,
    )


def _model_returnn_forward(source: Tensor, /, in_spatial_dim: Dim, model: Any) -> Tuple[Tensor, Dim]:
    """ForwardDef API"""
    from returnn.config import get_global_config
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim

    out, out_spatial_dim = model(source, in_spatial_dim=in_spatial_dim)
    assert isinstance(out, Tensor) and isinstance(out_spatial_dim, Dim)
    assert out.feature_dim  # we expect a feature dim
    assert out_spatial_dim in out.dims

    config = get_global_config()
    model_output_kind = config.typed_value("_model_output_kind", None)
    assert model_output_kind in {"logits", "log_prob", "prob"}
    output_kind = config.typed_value("_output_kind", None)
    assert output_kind in {"logits", "log_prob", "prob"}
    if model_output_kind == "logits":
        if output_kind == "logits":
            pass
        elif output_kind == "log_prob":
            out = rf.log_softmax(out, axis=out.feature_dim)
        elif output_kind == "prob":
            out = rf.softmax(out, axis=out.feature_dim)
        else:
            raise ValueError(f"invalid output_kind {output_kind!r}")
    elif model_output_kind == "log_prob":
        if output_kind in {"logits", "log_prob"}:
            pass
        elif output_kind == "prob":
            out = rf.exp(out)
        else:
            raise ValueError(f"invalid output_kind {output_kind!r}")
    elif model_output_kind == "prob":
        if output_kind in {"logits", "log_prob"}:
            out = rf.log(out)
        elif output_kind == "prob":
            pass
        else:
            raise ValueError(f"invalid output_kind {output_kind!r}")
    else:
        raise ValueError(f"invalid model_output_kind {model_output_kind!r}")

    return out, out_spatial_dim


_hdf_out_filename = "out.hdf"


def _returnn_get_forward_callback():
    from returnn.tensor import Tensor, TensorDict
    from returnn.forward_iface import ForwardCallbackIface
    from returnn.datasets.hdf import SimpleHDFWriter
    from returnn.config import get_global_config

    class _ReturnnForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.hdf_writer: Optional[SimpleHDFWriter] = None

        def init(self, *, model):
            config = get_global_config()
            expected_outputs_ = config.typed_value("model_outputs")
            assert isinstance(expected_outputs_, dict)
            expected_outputs = TensorDict()
            expected_outputs.update(expected_outputs_, auto_convert=True)
            output = expected_outputs["output"]
            self.hdf_writer = SimpleHDFWriter(
                filename=_hdf_out_filename,
                dim=output.dim,
                ndim=output.ndim,
                labels=output.vocab and output.vocab.labels,
                extra_type={k: (v.dim, v.ndim, v.dtype) for k, v in expected_outputs.data.items() if k != "output"},
            )

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            # see _returnn_forward_step
            out: Tensor = outputs["output"]
            self.hdf_writer.insert_batch(
                out.raw_tensor[None, :],
                seq_len={i + 1: out.raw_tensor.shape[i] for i, dim in enumerate(out.dims) if dim.dyn_size_ext},
                seq_tag=[seq_tag],
                extra={k: v.raw_tensor[None, :] for k, v in outputs.data.items() if k != "output"},
            )

        def finish(self):
            self.hdf_writer.close()

    return _ReturnnForwardCallbackIface()


# Those are applied for both training, recog and potential others.
# The values are only used if they are neither set in config nor post_config already.
# They should also not infer with other things from the epilog.
SharedPostConfig = {
    # In case pretraining overwrites some of these, they need a default.
    "accum_grad_multiple_step": None,
    "use_last_best_model": None,
}


def _returnn_forward_config(
    dataset: DatasetConfig,
    model_def: Union[None, ModelDef, ModelDefWithCfg],
    forward_def: ForwardDef,
    *,
    config: Optional[Dict[str, Any]] = None,
    post_config: Optional[Dict[str, Any]] = None,
) -> ReturnnConfig:
    """
    Create config for collecting stats.

    TODO should use sth like unhashed_package_root (https://github.com/rwth-i6/i6_experiments/pull/157)
    """
    import tree
    from i6_experiments.common.setups.returnn.serialization import get_serializable_config
    from returnn.tensor import Dim

    # Need to move out any parts which have Dim in it,
    # because the common ReturnnConfig serialization cannot handle that.
    # Also, unify the serialization with extern_data,
    # such that we reuse the same dim tags.
    # E.g. if there is model_outputs here.
    config = config.copy()
    config_dim_items = {}
    config_dim_items_extra_hash = {}
    for k, v in list(config.items()):
        if any(isinstance(v_, Dim) for v_ in tree.flatten(v)):
            config.pop(k)
            config_dim_items[k] = v
            config_dim_items_extra_hash[k] = tree.map_structure(
                lambda v_: {"dim": v_.dimension} if isinstance(v_, Dim) else v_, v
            )

    returnn_recog_config_dict = dict(
        # dataset
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),  # only for get_model with model_def
        forward_data=dataset.get_main_dataset(),
    )
    if model_def:
        returnn_recog_config_dict.update(
            dict(
                backend=model_def.backend,
                behavior_version=model_def.behavior_version,
            )
        )
    else:
        assert config and config.get("backend") and config.get("behavior_version")
    if config:
        returnn_recog_config_dict.update(config)
    if isinstance(model_def, ModelDefWithCfg):
        returnn_recog_config_dict.update(model_def.config)

    extern_data_raw = dataset.get_extern_data()
    # TODO why is the instanciate_delayed needed?
    # The extern_data is anyway not hashed, so we can also instanciate any delayed objects here.
    # It's not hashed because we assume that all aspects of the dataset are already covered
    # by the datasets itself as part in the config above.
    extern_data_raw = instanciate_delayed(extern_data_raw)

    returnn_forward_config = ReturnnConfig(
        config=returnn_recog_config_dict,
        python_epilog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(get_import_py_code()),
                    serialization.NonhashedCode(
                        nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(
                            extern_data_raw, other=config_dim_items
                        )
                    ),
                    serialization.ExplicitHash(config_dim_items_extra_hash),
                    *(
                        serialize_model_def(model_def)
                        if model_def
                        else [serialization.NonhashedCode("_model_def = None\n")]
                    ),
                    serialization.Import(_returnn_get_model, import_as="get_model"),
                    serialization.Import(forward_def, import_as="_forward_def", ignore_import_as_for_hash=True),
                    serialization.Import(_returnn_forward_step, import_as="forward_step"),
                    serialization.Import(_returnn_get_forward_callback, import_as="forward_callback"),
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
    returnn_forward_config = get_serializable_config(
        returnn_forward_config,
        # The only dim tags we directly have in the config are via extern_data, maybe also model_outputs.
        # All other dim tags are inside functions such as get_model or train_step,
        # so we do not need to care about them here, only about the serialization of those functions.
        # Those dim tags and those functions are already handled above.
        serialize_dim_tags=False,
    )

    batch_size_dependent = False
    if "__batch_size_dependent" in returnn_forward_config.config:
        batch_size_dependent = returnn_forward_config.config.pop("__batch_size_dependent")
    if "__batch_size_dependent" in returnn_forward_config.post_config:
        batch_size_dependent = returnn_forward_config.post_config.pop("__batch_size_dependent")
    for k, v in dict(
        batching="sorted",
        batch_size=(20000 * model_def.batch_size_factor) if model_def else (20000 * 160),
        max_seqs=200,
    ).items():
        if k in returnn_forward_config.config:
            v = returnn_forward_config.config.pop(k)
        if k in returnn_forward_config.post_config:
            v = returnn_forward_config.post_config.pop(k)
        (returnn_forward_config.config if batch_size_dependent else returnn_forward_config.post_config)[k] = v

    if post_config:
        returnn_forward_config.post_config.update(post_config)

    for k, v in SharedPostConfig.items():
        if k in returnn_forward_config.config or k in returnn_forward_config.post_config:
            continue
        returnn_forward_config.post_config[k] = v

    return returnn_forward_config


def _returnn_get_model(*, epoch: int, **_kwargs_unused):
    from returnn.tensor import Tensor
    from returnn.config import get_global_config
    import returnn.frontend as rf

    config = get_global_config()
    model_def = config.typed_value("_model_def")
    if model_def is None:
        return rf.Module()  # empty dummy module

    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    extern_data_dict = config.typed_value("extern_data")
    data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    assert targets.sparse_dim and targets.sparse_dim.vocab, f"no vocab for {targets}"

    model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
    return model


def _returnn_forward_step(*, model, extern_data: TensorDict, **_kwargs_unused):
    # This whole function doesn't really do much. It just wraps the forward_def.
    # We might consider to remove this function and just use forward_def directly.
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
    forward_def: ForwardDef = config.typed_value("_forward_def")
    # Note: forward_def uses a well-defined interface, which we also used elsewhere already.
    # It's a bit restricted though in that it only supports a single output...
    # The other code here does not have this restriction,
    # and we might want to have a more general interface in the future.
    out, out_spatial_dim = forward_def(data, in_spatial_dim=data_spatial_dim, model=model)
    assert isinstance(out, Tensor) and isinstance(out_spatial_dim, Dim)
    # rely on model_outputs being set in the config for the dim order of the output
    rf.get_run_ctx().mark_as_output(out, "output")
