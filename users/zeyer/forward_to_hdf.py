"""
Forward model outputs (or anything) to HDF

Related:
Also see :mod:`collect_model_dataset_stats` for collecting stats on the dataset and/or model,
e.g. like getting the average model softmax output, i.e. the model softmax prior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Any, Callable, Dict, Tuple

from sisyphus import tk

from i6_core.returnn import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from returnn_common import nn
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.common.setups import serialization
from i6_experiments.users.zeyer.utils.serialization import get_import_py_code

from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_inplace_with_warning
from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, ForwardRFDef, serialize_model_def
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim, TensorDict


def forward_to_hdf(
    *,
    dataset: DatasetConfig,
    model: Optional[ModelWithCheckpoint] = None,
    forward_def: Optional[ForwardRFDef] = None,
    forward_step: Optional[Callable] = None,
    config: Optional[Dict[str, Any]] = None,
    forward_post_config: Optional[Dict[str, Any]] = None,
    forward_mem_rqmt: Union[int, float] = 6,
    forward_rqmt: Optional[Dict[str, Any]] = None,
    forward_device: Optional[str] = None,
    forward_alias_name: Optional[str] = None,
    _config_v2: bool = True,  # testing...
) -> tk.Path:
    """
    Forward on the specific dataset,
    via the forward_def/forward_step (or a no-op copy),
    maybe optionally using a model,
    into an HDF file (using :class:`SimpleHDFWriter`).

    The default output ("output") is saved as "data" key in the HDF file.
    All other outputs keep their name.

    Note that HDF currently has some limitations:
    For sparse data, we expect the shape [time], for dense data, we expect the shape [time] or [time, feat].
    The :class:`SimpleHDFWriter` will automatically convert it as necessary, e.g. flattening the data.
    When the data was flattened, there will be an additional key "sizes" which contains the original sizes
    of e.g. some tensor [time1,time2].
    This flattening logic is however only supported for the main key "data", not for any other keys.

    :param dataset: dataset to forward, using its get_main_dataset(),
        and also get_default_input() to define the default output,
        and get_extern_data().
    :param model: optional some model to be used in the ``forward_def`` or ``forward_step``
    :param forward_def: function (source: Tensor, /, in_spatial_dim: Dim, model: ModelT) -> None,
        will get called with the default input, is supposed to call ``rf.get_run_ctx().mark_as_output(...)``.
    :param forward_step: function (extern_data: TensorDict, **_kwargs_unused) -> None,
        will get called with all inputs, is supposed to call ``rf.get_run_ctx().mark_as_output(...)``.
        Use either ``forward_def`` or ``forward_step``.
        If none is given, a default no-op step will be used
        which just forwards the input to the output as-is,
        translating the default input key (e.g. "data") to the default output key ("output").
    :param config: additional RETURNN config opts for the forward job
    :param forward_post_config: additional RETURNN post config (non-hashed) opts for the forward job
    :param forward_mem_rqmt: memory requirement for the forward job (in GB)
    :param forward_rqmt: additional rqmt opts for the forward job (e.g. "time" (in hours))
    :param forward_device: "cpu" or "gpu". if not given, will be "gpu" if model is given, else "cpu"
    :param forward_alias_name: optional alias name for the forward job
    :param _config_v2: new RETURNN config serialization
    :return: HDF file path
    """
    assert not (forward_def and forward_step), "either forward_def or forward_step, not both"
    env_updates = None
    if (config and config.get("__env_updates")) or (forward_post_config and forward_post_config.get("__env_updates")):
        env_updates = (config and config.pop("__env_updates", None)) or (
            forward_post_config and forward_post_config.pop("__env_updates", None)
        )
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=model.checkpoint.path if model else None,
        returnn_config=(_returnn_forward_config_v2 if _config_v2 else _returnn_forward_config)(
            dataset=dataset,
            model_def=model.definition if model else None,
            forward_def=forward_def,
            forward_step=forward_step,
            config=config,
            post_config=forward_post_config,
        ),
        output_files=[_hdf_out_filename],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        mem_rqmt=forward_mem_rqmt,
        device=forward_device or ("gpu" if model else "cpu"),
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
                extra_labels={k: v.vocab.labels for k, v in expected_outputs.data.items() if k != "output" and v.vocab},
            )

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            # see _returnn_forward_step
            out: Tensor = outputs["output"]
            self.hdf_writer.insert_batch(
                out.raw_tensor[None, :],
                seq_len={
                    i: [out.raw_tensor.shape[i]]
                    for i, dim in enumerate(out.dims)
                    if dim.dyn_size_ext is not None or out.sparse
                },
                seq_tag=[seq_tag],
                extra={k: v.raw_tensor[None] for k, v in outputs.data.items() if k != "output"},
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
    *,
    dataset: DatasetConfig,
    model_def: Union[None, ModelDef, ModelDefWithCfg],
    forward_def: Optional[ForwardRFDef] = None,
    forward_step: Optional[Callable] = None,
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

    assert not (forward_def and forward_step), "either forward_def or forward_step, not both"
    if not forward_def and not forward_step:
        forward_step = _returnn_forward_noop_step

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

    # Need to move out any parts which have Dim in it,
    # because the common ReturnnConfig serialization cannot handle that.
    # Also, unify the serialization with extern_data,
    # such that we reuse the same dim tags.
    # E.g. if there is model_outputs here.
    config_dim_items = {}
    config_dim_items_extra_hash = {}
    for k, v in list(returnn_recog_config_dict.items()):
        if any(isinstance(v_, Dim) for v_ in tree.flatten(v)):
            returnn_recog_config_dict.pop(k)
            config_dim_items[k] = v
            config_dim_items_extra_hash[k] = tree.map_structure(
                lambda v_: {"dim": v_.dimension} if isinstance(v_, Dim) else v_, v
            )

    # TODO why is the instanciate_delayed needed?
    # The extern_data is anyway not hashed, so we can also instanciate any delayed objects here.
    # It's not hashed because we assume that all aspects of the dataset are already covered
    # by the datasets itself as part in the config above.
    extern_data_raw = instanciate_delayed_inplace_with_warning(dataset.get_extern_data)

    if (
        forward_step is _returnn_forward_noop_step
        and "model_outputs" not in config_dim_items
        and "model_outputs" not in returnn_recog_config_dict
    ):
        # Copy the extern_data to model_outputs.
        # This only works if all dim tags are already defined.
        # (Note, if we don't have this, we could create the dim tags now, by calling Tensor(**v),
        #  and then using the automatically created dim tags.
        #  We don't do this for now.)
        model_outputs = extern_data_raw.copy()
        assert all(v.get("dims") is not None or v.get("dim_tags") is not None for v in model_outputs.values())
        # Map the default input key (e.g. "data") to the default RF output key (which is "output").
        input_key = dataset.get_default_input()
        if input_key:
            assert input_key in model_outputs
            assert "output" not in model_outputs
            model_outputs["output"] = model_outputs.pop(input_key)
        config_dim_items["model_outputs"] = model_outputs

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
                    *(
                        [
                            serialization.Import(forward_def, import_as="_forward_def", ignore_import_as_for_hash=True),
                            serialization.Import(_returnn_forward_step, import_as="forward_step"),
                        ]
                        if forward_def
                        else [serialization.Import(forward_step, import_as="forward_step")]
                    ),
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


def _returnn_forward_config_v2(
    *,
    dataset: DatasetConfig,
    model_def: Union[None, ModelDef, ModelDefWithCfg],
    forward_def: Optional[ForwardRFDef] = None,
    forward_step: Optional[Callable] = None,
    config: Optional[Dict[str, Any]] = None,
    post_config: Optional[Dict[str, Any]] = None,
) -> ReturnnConfig:
    """
    Create config for collecting stats.
    """
    from i6_experiments.users.zeyer.serialization_v2 import ReturnnConfigWithNewSerialization

    assert not (forward_def and forward_step), "either forward_def or forward_step, not both"
    if not forward_def and not forward_step:
        forward_step = _returnn_forward_noop_step

    config = dict(
        **(config or {}),
        # dataset
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),  # only for get_model with model_def
        extern_data=dataset.get_extern_data(),
        forward_data=dataset.get_main_dataset(),
    )

    if forward_step is _returnn_forward_noop_step and "model_outputs" not in config:
        # Copy the extern_data to model_outputs.
        model_outputs = config["extern_data"].copy()
        assert all(v.get("dims") is not None or v.get("dim_tags") is not None for v in model_outputs.values())
        # Map the default input key (e.g. "data") to the default RF output key (which is "output").
        input_key = dataset.get_default_input()
        if input_key:
            assert input_key in model_outputs
            assert "output" not in model_outputs
            model_outputs["output"] = model_outputs.pop(input_key)
        config["model_outputs"] = model_outputs

    if model_def:
        if "backend" not in config:
            config["backend"] = model_def.backend
        config["behavior_version"] = max(model_def.behavior_version, config.get("behavior_version", 0))
    else:
        assert (
            config and config.get("backend") and config.get("behavior_version")
        ), f"config: {config}\nbackend: {config.get('backend')}, behavior_version: {config.get('behavior_version')}"

    if isinstance(model_def, ModelDefWithCfg):
        config["_model_def"] = model_def.model_def
        config.update(model_def.config)
    else:
        config["_model_def"] = model_def
    config["get_model"] = _returnn_get_model

    if forward_def:
        assert not forward_step
        config["_forward_def"] = forward_def
        config["forward_step"] = _returnn_forward_step
    if forward_step:
        assert not forward_def
        config["forward_step"] = forward_step

    config["forward_callback"] = _returnn_get_forward_callback

    # post_config is not hashed
    post_config_ = dict(
        log_batch_size=True,
        # debug_add_check_numerics_ops = True
        # debug_add_check_numerics_on_output = True
        torch_log_memory_usage=True,
        watch_memory=True,
        use_lovely_tensors=True,
    )
    if post_config:
        post_config_.update(post_config)
    post_config = post_config_

    batch_size_dependent = False
    if "__batch_size_dependent" in config:
        batch_size_dependent = config.pop("__batch_size_dependent")
    if "__batch_size_dependent" in post_config:
        batch_size_dependent = post_config.pop("__batch_size_dependent")
    for k, v in dict(
        batching="sorted",
        batch_size=(20000 * model_def.batch_size_factor) if model_def else (20000 * 160),
        max_seqs=200,
    ).items():
        if k in config:
            v = config.pop(k)
        if k in post_config:
            v = post_config.pop(k)
        (config if batch_size_dependent else post_config)[k] = v

    for k, v in SharedPostConfig.items():
        if k in config or k in post_config:
            continue
        post_config[k] = v

    # Trigger new hash because of a serious bug.
    config["__forward_config_v2_extra_version"] = 2

    return ReturnnConfigWithNewSerialization(config, post_config)


def _returnn_get_model(*, epoch: int, **_kwargs_unused):
    from returnn.tensor import Tensor
    from returnn.config import get_global_config
    import returnn.frontend as rf

    config = get_global_config()
    model_def = config.typed_value("_model_def")
    if model_def is None:
        return rf.Module()  # empty dummy module

    extern_data_dict = config.typed_value("extern_data")
    model_outputs_dict = config.typed_value("model_outputs")

    default_input_key = config.typed_value("default_input")
    data_templ_dict = {"name": default_input_key, **extern_data_dict[default_input_key]}
    default_target_key = config.typed_value("target")
    if default_target_key:
        targets_templ_dict = {"name": default_target_key, **extern_data_dict[default_target_key]}
    elif model_outputs_dict and "output" in model_outputs_dict:
        targets_templ_dict = {"name": "output", **model_outputs_dict["output"]}
    else:
        raise ValueError(f"default_target_key {default_target_key} and model_outputs {model_outputs_dict}")

    data = Tensor(**data_templ_dict)
    targets = Tensor(**targets_templ_dict)
    assert targets.sparse_dim and targets.sparse_dim.vocab, f"no vocab for {targets}"

    model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
    return model


def _returnn_forward_step(*, model, extern_data: TensorDict, **_kwargs_unused):
    # This whole function doesn't really do much. It just wraps the forward_def.
    # We might consider to remove this function and just use forward_def directly.
    import returnn.frontend as rf
    from returnn.tensor import batch_dim
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
    forward_def: ForwardRFDef = config.typed_value("_forward_def")
    # Note: forward_def uses a well-defined interface, which we also used elsewhere already.
    # It's a bit restricted though in that it only supports a single output...
    # The other code here does not have this restriction,
    # and we might want to have a more general interface in the future.
    forward_def(data, in_spatial_dim=data_spatial_dim, model=model)


def _returnn_forward_noop_step(*, extern_data: TensorDict, **_kwargs_unused):
    import returnn.frontend as rf
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")

    for k, v in extern_data.data.items():
        if k == default_input_key:
            k = "output"
        rf.get_run_ctx().mark_as_output(v, k, dims=v.dims)
