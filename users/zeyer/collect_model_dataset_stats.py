"""
Calculate statistics over any dataset, e.g. feature statistics.
Can also perform any computation on the dataset, e.g. forward pass through a model.
Thus, this can also be used to calculate prior statistics for a model,
such as the average softmax output, the model softmax prior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Any, Dict, Tuple
from dataclasses import dataclass

from sisyphus import tk

from i6_core.returnn import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from returnn_common import nn
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.common.setups import serialization
from i6_experiments.users.zeyer.utils.serialization import get_import_py_code

from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy
from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, ForwardDef, serialize_model_def
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim, TensorDict


@dataclass
class StatisticsOutput:
    """statistics, as txt files. numpy.loadtxt can be used to read them."""

    mean: tk.Path
    std_dev: tk.Path
    min: tk.Path
    max: tk.Path
    info: tk.Path


def collect_log_mel_feature_statistics(
    *, dataset: DatasetConfig, dim: int, backend: str = "torch", behavior_version: int = 21, **kwargs
) -> StatisticsOutput:
    """
    Get feature stats

    :param dataset:
    :param dim: log mel feature dim
    :param backend:
    :param behavior_version:
    :param kwargs: all passed to rf.audio.log_mel_filterbank_from_raw.
        Default sampling_rate is 16_000, which is also what we have for Librispeech usually.
        Note on log_base: Default is 10.0.
            Note that in some earlier setups, and also Mohammads original AED setup,
            we used log_base=math.exp(2.3026), which is almost 10.0 but not exactly...
    """
    return collect_statistics(
        dataset=dataset,
        forward_def=_log_mel_stats_returnn_forward,
        config={
            "backend": backend,
            "behavior_version": behavior_version,
            "_audio_feature_dim": dim,
            "_audio_feature_opts": kwargs,
        },
    )


def compute_model_softmax_prior_statistics(
    *,
    model: ModelWithCheckpoint,
    model_output_kind: str = "logits",
    dataset: DatasetConfig,
    backend: str = "torch",
    behavior_version: int = 21,
    **kwargs,
) -> StatisticsOutput:
    """
    Calculate model softmax prior average.

    :param model: after construction, the model will be called as:
        ``out, out_spatial_dim = model(input, in_spatial_dim=in_spatial_dim)``
        (This is the RETURNN ISeqDownsamplingEncoder interface.)
        ``out.feature_dim`` is expected to be set.
        Use ``model_output_kind`` to specify what kind of output you have in the model output ``out``.
        (If you have a model with a different interface, just call :collect_statistics` directly
         with your custom ``forward_def`` function.)
    :param model_output_kind: "logits", "log_prob" or "prob"
    :param dataset:
    :param backend:
    :param behavior_version:
    :param kwargs: passed to :func:`collect_statistics`
    """
    assert model_output_kind in {"logits", "log_prob", "prob"}
    return collect_statistics(
        model=model,
        dataset=dataset,
        forward_def=_model_softmax_prior_returnn_forward,
        config={
            "backend": backend,
            "behavior_version": behavior_version,
            "_model_output_kind": model_output_kind,
        },
        **kwargs,
    )


def collect_statistics(
    *,
    dataset: DatasetConfig,
    model: Optional[ModelWithCheckpoint] = None,
    forward_def: ForwardDef,
    config: Optional[Dict[str, Any]] = None,
    forward_post_config: Optional[Dict[str, Any]] = None,
    forward_mem_rqmt: Union[int, float] = 6,
    forward_rqmt: Optional[Dict[str, Any]] = None,
    forward_alias_name: Optional[str] = None,
) -> StatisticsOutput:
    """
    recog on the specific dataset
    """
    env_updates = None
    config = config.copy() if config else {}
    forward_post_config = forward_post_config.copy() if forward_post_config else {}
    if (config and config.get("__env_updates")) or (forward_post_config and forward_post_config.get("__env_updates")):
        env_updates = (config and config.pop("__env_updates", None)) or (
            forward_post_config and forward_post_config.pop("__env_updates", None)
        )
    out_files = {
        "mean": _prior_mean_out_filename,
        "std_dev": _prior_std_dev_out_filename,
        "min": _prior_min_out_filename,
        "max": _prior_max_out_filename,
        "info": _prior_info_out_filename,
    }
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=model.checkpoint if model else None,
        returnn_config=_collect_stats_returnn_forward_config(
            dataset, model.definition if model else None, forward_def, config=config, post_config=forward_post_config
        ),
        output_files=list(out_files.values()),
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
    return StatisticsOutput(**{k: forward_job.out_files[v] for k, v in out_files.items()})


def _log_mel_stats_returnn_forward(source: Tensor, /, in_spatial_dim: Dim, model: Any) -> Tuple[Tensor, Dim]:
    """ForwardDef API"""
    from returnn.config import get_global_config
    import returnn.frontend as rf
    from returnn.tensor import Dim

    model  # noqa # unused
    config = get_global_config()
    feat_dim = config.int("_audio_feature_dim", -1)
    assert feat_dim > 0
    feat_dim = Dim(feat_dim, name="audio", kind=Dim.Types.Feature)
    opts = config.typed_value("_audio_feature_opts", None)
    assert isinstance(opts, dict)

    source, out_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
        source, in_spatial_dim=in_spatial_dim, out_dim=feat_dim, **opts
    )
    return source, out_spatial_dim


def _model_softmax_prior_returnn_forward(source: Tensor, /, in_spatial_dim: Dim, model: Any) -> Tuple[Tensor, Dim]:
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
    if model_output_kind == "logits":
        out = rf.softmax(out, axis=out.feature_dim)
    elif model_output_kind == "log_prob":
        out = rf.exp(out)
    elif model_output_kind == "prob":
        pass
    else:
        raise ValueError(f"invalid model_output_kind {model_output_kind!r}")

    return out, out_spatial_dim


_prior_mean_out_filename = "stats.mean.txt"
_prior_std_dev_out_filename = "stats.std_dev.txt"
_prior_min_out_filename = "stats.min.txt"
_prior_max_out_filename = "stats.max.txt"
_prior_info_out_filename = "stats.info.txt"


def _returnn_get_forward_callback():
    from returnn.tensor import Tensor, TensorDict
    from returnn.forward_iface import ForwardCallbackIface
    from returnn.util.basic import Stats

    class _ReturnnCollectStatsForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.stats: Optional[Stats] = None

        def init(self, *, model):
            self.stats = Stats()

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            # see _returnn_forward_step
            out: Tensor = outputs["output"].copy_with_feature_last()
            assert out.batch_ndim == 2  # (time,feature)
            self.stats.collect(out.raw_tensor)

        def finish(self):
            self.stats.dump("stats")

    return _ReturnnCollectStatsForwardCallbackIface()


# Those are applied for both training, recog and potential others.
# The values are only used if they are neither set in config nor post_config already.
# They should also not infer with other things from the epilog.
SharedPostConfig = {
    # In case pretraining overwrites some of these, they need a default.
    "accum_grad_multiple_step": None,
    "use_last_best_model": None,
}


def _collect_stats_returnn_forward_config(
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
    from i6_experiments.common.setups.returnn.serialization import get_serializable_config

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
    # The extern_data is anyway not hashed, so we can also instanciate any delayed objects here.
    # It's not hashed because we assume that all aspects of the dataset are already covered
    # by the datasets itself as part in the config above.
    extern_data_raw = instanciate_delayed_copy(extern_data_raw)

    returnn_forward_config = ReturnnConfig(
        config=returnn_recog_config_dict,
        python_epilog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(get_import_py_code()),
                    serialization.NonhashedCode(
                        nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
                    ),
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
    out, out_spatial_dim = forward_def(data, in_spatial_dim=data_spatial_dim, model=model)
    assert isinstance(out, Tensor) and isinstance(out_spatial_dim, Dim)
    assert out.feature_dim  # we expect a feature dim
    rf.get_run_ctx().mark_as_output(out, "output", dims=[batch_dim, out_spatial_dim, out.feature_dim])
