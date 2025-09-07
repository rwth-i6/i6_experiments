"""
helpers for training.

- Uses ``unhashed_package_root`` now, via :func:`i6_experiments.users.zeyer.utils.sis_setup.get_base_module`.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Protocol, Union, Dict, Any, Sequence, Tuple, TypeVar
import torch
import copy
from i6_experiments.users.zeyer.model_interfaces import ModelT, ModelDef, ModelDefWithCfg, TrainDef, serialize_model_def
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.utils.sis_setup import get_base_module
from sisyphus import tk

if TYPE_CHECKING:
    from returnn.tensor import TensorDict, Tensor, Dim
    from returnn.forward_iface import ForwardCallbackIface
    from i6_experiments.common.setups import serialization
    from i6_experiments.users.zeyer.datasets.task import Task, DatasetConfig
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, Checkpoint


from i6_experiments.users.enrique.jobs.initializations.k_means.returnn_kmeans.k_means_guided_by_LM import (
    NnOutputClusteringCallback,
)


def _returnn_get_k_means_forward_callback():
    return NnOutputClusteringCallback()


# ModelT = TypeVar("ModelT", contravariant=True)


def k_means_guided_by_LM_train(
    prefix_name: str,
    *,
    task: Optional[Task] = None,
    train_dataset: Optional[DatasetConfig] = None,
    train_epoch_split: Optional[int] = None,
    config: Dict[str, Any],
    post_config: Optional[Dict[str, Any]] = None,
    env_updates: Optional[Dict[str, str]] = None,
    epilog: Sequence[serialization.SerializerObject] = (),
    model_def: Union[ModelDefWithCfg, ModelDef[ModelT]],
    train_def: TrainDef,
    init_params: Optional[Checkpoint] = None,
    reset_steps: bool = True,
    finish_all: bool = False,
    extra_hash: Any = None,
    gpu_mem: Optional[int] = None,
    num_processes: Optional[int] = None,
    init_hdf_writer: bool = False,
    keep_train_lm_def: bool = True,
    executables: Optional[Dict[str, tk.Path]] = None,
    **kwargs,
) -> tk.Path:
    """
    train

    Note on hash:
    - model_def/train_def: just the module name + function name goes into the hash, not the content!
    - extra_hash: explicitly goes into the hash
    - others just as one would expect

    We extract the unhashed_package_root automatically via
    :func:`i6_experiments.users.zeyer.utils.sis_setup.get_base_module`
    from train_def.
    """
    from sisyphus import tk
    from i6_core.util import instanciate_delayed
    from i6_core.returnn.training import ReturnnTrainingJob
    from i6_core.returnn.forward import ReturnnForwardJobV2
    from i6_core.returnn.config import ReturnnConfig
    from i6_experiments.common.setups import serialization
    from i6_experiments.common.setups.returnn.serialization import get_serializable_config
    from i6_experiments.users.zeyer.utils.serialization import get_import_py_code
    from i6_experiments.users.zeyer.datasets.utils import multi_proc as mp_ds_utils
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints
    from i6_experiments.users.zeyer.recog import SharedPostConfig
    from returnn_common import nn

    unhashed_package_root_train_def, setup_base_name_train_def = get_base_module(train_def)
    unhashed_package_root_model_def, setup_base_name_model_def = get_base_module(
        model_def.model_def if isinstance(model_def, ModelDefWithCfg) else model_def
    )

    if train_dataset is None:
        assert task
        train_dataset = task.train_dataset
    train_dataset_dict = train_dataset.get_train_dataset()
    if train_epoch_split is None:
        if task:
            train_epoch_split = task.train_epoch_split
        elif "partition_epoch" in train_dataset_dict:
            train_epoch_split = train_dataset_dict["partition_epoch"]
    # Usually always apply MultiProcDataset. But some exceptions for now:
    apply_multi_proc = train_dataset_dict["class"] != "LmDataset"
    del train_dataset_dict
    del task

    config = config.copy()
    kwargs = kwargs.copy()
    if "__num_epochs" in config:
        kwargs["num_epochs"] = config.pop("__num_epochs")
    if "__gpu_mem" in config:
        gpu_mem = config.pop("__gpu_mem")
    if "__num_processes" in config:
        num_processes = config.pop("__num_processes")
    if "__mem_rqmt" in config:
        kwargs["mem_rqmt"] = config.pop("__mem_rqmt")
    if "__cpu_rqmt" in config:
        kwargs["cpu_rqmt"] = config.pop("__cpu_rqmt")
    if not kwargs.get("distributed_launch_cmd"):
        kwargs["distributed_launch_cmd"] = "torchrun" if num_processes else "mpirun"
    if "__train_audio_preprocess" in config:
        train_dataset = copy.copy(train_dataset)
        assert hasattr(train_dataset, "train_audio_preprocess")
        train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")
    multi_proc_opts = (post_config.pop("__multi_proc_dataset_opts", None) or {}) if post_config else {}

    returnn_train_config_dict: Dict[str, Any] = dict(
        backend=model_def.backend,
        behavior_version=model_def.behavior_version,
        # dataset
        default_input=train_dataset.get_default_input(),
        target=train_dataset.get_default_target(),
        train=(
            mp_ds_utils.multi_proc_dataset_opts(train_dataset.get_train_dataset(), **multi_proc_opts)
            if apply_multi_proc
            else train_dataset.get_train_dataset()
        ),
        eval_datasets=(
            mp_ds_utils.multi_proc_eval_datasets_opts(train_dataset.get_eval_datasets(), **multi_proc_opts)
            if apply_multi_proc
            else train_dataset.get_eval_datasets()
        ),
        forward_data=(
            mp_ds_utils.multi_proc_dataset_opts(train_dataset.get_train_dataset(), **multi_proc_opts)
            if apply_multi_proc
            else train_dataset.get_train_dataset()
        ),
        learning_rate_control_error_measure=train_def.learning_rate_control_error_measure,
        newbob_multi_num_epochs=train_epoch_split or 1,
    )
    returnn_train_config_dict = dict_update_deep(returnn_train_config_dict, config)
    if isinstance(model_def, ModelDefWithCfg):
        model_conf = model_def.config.copy()
        model_conf.pop("recog_language_model", None)
        if not keep_train_lm_def:
            model_conf.pop("train_language_model", None)
        returnn_train_config_dict = dict_update_deep(returnn_train_config_dict, model_conf)

    max_seq_length_default_target = returnn_train_config_dict.pop("max_seq_length_default_target", None)

    if max_seq_length_default_target is not None:
        max_seq_length = returnn_train_config_dict.setdefault("max_seq_length", {})
        assert isinstance(max_seq_length, dict)
        max_seq_length[train_dataset.get_default_target()] = max_seq_length_default_target
    max_seq_length_default_input = returnn_train_config_dict.pop("max_seq_length_default_input", None)
    if max_seq_length_default_input is not None:
        max_seq_length = returnn_train_config_dict.setdefault("max_seq_length", {})
        assert isinstance(max_seq_length, dict)
        max_seq_length[train_dataset.get_default_input()] = max_seq_length_default_input

    if init_params:
        returnn_train_config_dict["import_model_train_epoch1"] = init_params
        if not reset_steps:
            returnn_train_config_dict["reset_steps"] = False
    if finish_all:
        returnn_train_config_dict["_horovod_finish_all"] = True

    extern_data_raw = train_dataset.get_extern_data()
    # The extern_data is anyway not hashed, so we can also instanciate any delayed objects here.
    # It's not hashed because we assume that all aspects of the dataset are already covered
    # by the datasets itself as part in the config above.
    extern_data_raw = instanciate_delayed(extern_data_raw)

    # from .lib.pytorch.clustering import NnOutputClusteringCallback, EMAUpdater, BatchwiseUpdater
    # forward_callback = serialization.CallImport(
    #     NnOutputClusteringCallback,

    # )

    returnn_train_config = ReturnnConfig(
        returnn_train_config_dict,
        python_epilog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(get_import_py_code()),
                    serialization.NonhashedCode(
                        nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
                    ),
                    *serialize_model_def(model_def, unhashed_package_root=unhashed_package_root_model_def),
                    serialization.Import(_enc_forward, import_as="_forward_def", use_for_hash=False),
                    serialization.Import(_returnn_get_k_means_forward_callback, import_as="forward_callback"),
                    # Consider the imports as non-hashed. We handle any logic changes via the explicit hash below.
                    serialization.Import(_returnn_v2_get_model, import_as="get_model", use_for_hash=False),
                    serialization.Import(_returnn_v2_forward_step, import_as="forward_step", use_for_hash=False),
                ]
                + (
                    [
                        serialization.Import(
                            _returnn_epoch_start_hdf_writer, import_as="epoch_start", use_for_hash=False
                        ),
                        serialization.Import(_returnn_epoch_end_hdf_writer, import_as="epoch_end", use_for_hash=False),
                    ]
                    if init_hdf_writer
                    else []
                )
                + [
                    serialization.ExplicitHash(
                        {
                            # Increase the version whenever some incompatible change is made in this train() function,
                            # which influences the outcome, but would otherwise not influence the hash.
                            "version": 3,
                            # Whatever the caller provides. This could also include another version,
                            # but this is up to the caller.
                            "extra": extra_hash,
                            **(
                                {"setup_base_name": setup_base_name_train_def}
                                if setup_base_name_train_def == setup_base_name_model_def
                                else {
                                    "setup_base_name_train_def": setup_base_name_train_def,
                                    "setup_base_name_model_def": setup_base_name_model_def,
                                }
                            ),
                        }
                    ),
                    serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                    serialization.PythonCacheManagerFunctionNonhashedCode,
                    serialization.PythonModelineNonhashedCode,
                ]
                + list(epilog)
            )
        ],  # pyright: ignore[reportArgumentType]
        post_config=dict(  # not hashed
            log_batch_size=True,
            cleanup_old_models=True,
            # debug_add_check_numerics_ops = True
            # debug_add_check_numerics_on_output = True
            # stop_on_nonfinite_train_score = False,
            torch_log_memory_usage=True,
            # watch_memory=True,
            # use_lovely_tensors=True,
            # use_train_proc_manager=True,
            alias=prefix_name,
        ),
        sort_config=False,
    )
    if post_config:
        returnn_train_config.post_config = dict_update_deep(returnn_train_config.post_config, post_config)

    for k, v in SharedPostConfig.items():
        if k in returnn_train_config.config or k in returnn_train_config.post_config:
            continue
        returnn_train_config.post_config[k] = v

    # There might be some further functions in the config, e.g. some dataset postprocessing.
    returnn_train_config = get_serializable_config(
        returnn_train_config,
        # The only dim tags we directly have in the config are via extern_data, maybe also model_outputs.
        # All other dim tags are inside functions such as get_model or train_step,
        # so we do not need to care about them here, only about the serialization of those functions.
        # Those dim tags and those functions are already handled above.
        serialize_dim_tags=False,
    )

    for k, v in dict(
        log_verbosity=5,
        num_epochs=10,
        time_rqmt=80,
        mem_rqmt=30 if gpu_mem and gpu_mem > 11 else 15,  # 15, 30
        cpu_rqmt=4 if (not num_processes or num_processes <= 4) else 3,
        horovod_num_processes=num_processes,  # legacy name but also applies for Torch
    ).items():
        if k not in kwargs or kwargs[k] is None:
            kwargs[k] = v

    #del kwargs["num_epochs"]
    kwargs = {
        "time_rqmt": 4,
        "mem_rqmt": 30,
        "cpu_rqmt": 4,
    }
    # returnn_train_job = ReturnnTrainingJob(returnn_train_config, **kwargs)
    experiments = ["ema"]
    centroids_and_scores_file = "centroids_and_scores.json"
    best_centroids = "best_centroids.json"
    val_loss_graph_path = "validation_loss_graph.png"
    for i, exp in enumerate(experiments):
        returnn_train_config = copy.deepcopy(returnn_train_config)
        returnn_train_config.config["experiment"] = exp
        returnn_fwd_job = ReturnnForwardJobV2(
            model_checkpoint=None,
            returnn_config=returnn_train_config,
            output_files=[centroids_and_scores_file, best_centroids, val_loss_graph_path],
            **kwargs,
            **(executables or {}),
        )
        returnn_fwd_job.add_alias(f"clustering-{exp}")
        returnn_fwd_job.rqmt["time"] = 48.0
        if gpu_mem:
            returnn_fwd_job.rqmt["gpu_mem"] = gpu_mem
        tk.register_output(f"clusters-{exp}.pt", returnn_fwd_job.out_files[centroids_and_scores_file, best_centroids, val_loss_graph_path])
    return returnn_fwd_job.out_files[centroids_and_scores_file, best_centroids]


def _enc_forward(*, model: torch.nn.Module, data: Tensor, data_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
    from returnn.config import get_global_config

    config = get_global_config()
    assert config is not None

    if config.typed_value("use_w2v_model", False):
        enc, _, enc_spatial_dim = model.wav2vec_model(data, in_spatial_dim=data_spatial_dim)
        import logging

        logging.info("W2V model output:")
        logging.info(enc)
        logging.info("Spatial dimension:")
        logging.info(enc, enc_spatial_dim)
        return enc, enc_spatial_dim
    else:
        raise NotImplementedError("Only W2V model is supported for now.")


class ForwardDef(Protocol[ModelT]):
    """
    Defines the losses (mark_as_loss).
    """

    def __call__(
        self,
        *,
        model: ModelT,
        data: Tensor,
        data_spatial_dim: Dim,
    ):
        raise NotImplementedError


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
    model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
    return model


def _returnn_v2_forward_step(*, model, extern_data: TensorDict, **_kwargs_unused):
    from returnn.config import get_global_config
    import returnn.frontend as rf

    config = get_global_config()
    assert config is not None
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()
    targets = extern_data[default_target_key]
    targets_spatial_dim = targets.get_time_dim_tag()
    forward_def: ForwardDef = config.typed_value("_forward_def")
    output, spatial_dim = forward_def(
        model=model,
        data=data,
        data_spatial_dim=data_spatial_dim,
    )
    # dims = list(rf.get_run_ctx().expected_outputs["output"].dims)
    # dims[1] = spatial_dim
    # dims[2] =
    # rf.get_run_ctx().expected_outputs["output"]._dims = tuple(dims)
    # rf.get_run_ctx().expected_outputs["output"]._dims = output.dims
    # rf.get_run_ctx().expected_outputs["output"].dims[1].dyn_size_ext.raw_tensor = spatial_dim.dyn_size_ext.raw_tensor
    # rf.get_run_ctx().expected_outputs["output"].dims[1].dyn_size_ext.raw_tensor = spatial_dim.dyn_size_ext.raw_tensor
    rf.get_run_ctx().mark_as_default_output(output)


class _ReturnnForwardCallbackIface:
    """
    Generic returnn forward callback interface which wraps around custom callback interfaces
    depending on the given parameters.
    """

    def __init__(
        self,
    ):
        pass


def _returnn_epoch_start_hdf_writer(*, epoch: int, step: int, model: ModelT, dataset_name: str, **_kwargs_unused):
    if dataset_name == "train":
        device_id = torch.cuda.current_device()
        if device_id == 0:
            import os
            from returnn.config import get_global_config
            from i6_core.lib.hdf import get_returnn_simple_hdf_writer

            config = get_global_config()

            SimpleHDFWriter = get_returnn_simple_hdf_writer(None)
            train_dataset_dict = config.typed_value("train")
            partition_epoch = train_dataset_dict["dataset"]["datasets"]["zip_dataset"]["partition_epoch"]

            hdf_filename = f"targets-epoch-{((epoch - 1) // partition_epoch) + 1}-{device_id}.hdf"
            hdf_writer = SimpleHDFWriter(
                filename=hdf_filename,
                dim=None,
                ndim=1,
                extend_existing_file=os.path.exists(hdf_filename),
            )

            print("TMP:", hdf_writer.tmp_filename, "Filename:", hdf_writer.filename)

            config.set("train_hdf_writer", hdf_writer)


def _returnn_v2_train_step(*, model, extern_data: TensorDict, **_kwargs_unused):
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()
    targets = extern_data[default_target_key]
    targets_spatial_dim = targets.get_time_dim_tag()
    train_def: TrainDef = config.typed_value("_train_def")
    if train_def.__name__ == "ctc_sum_training":
        seq_tags = extern_data["seq_tag"]
        train_def(
            model=model,
            data=data,
            data_spatial_dim=data_spatial_dim,
            seq_tags=seq_tags,
            targets=targets,
            targets_spatial_dim=targets_spatial_dim,
        )
    elif train_def.__name__ == "ce_training":
        targets_indices = None
        if "targets_indices" in extern_data:
            targets_indices = extern_data["targets_indices"]

        train_def(
            model=model,
            data=data,
            data_spatial_dim=data_spatial_dim,
            targets=targets,
            targets_spatial_dim=targets_spatial_dim,
            targets_indices=targets_indices,
        )
    elif train_def.__name__ == "ctc_training" or train_def.__name__ == "seq_gamma_training":
        nbest_lengths = None
        scores = None
        if "nbest_lengths" in extern_data:
            nbest_lengths = extern_data["nbest_lengths"]
        if "scores" in extern_data:
            scores = extern_data["scores"]
        seq_tags = extern_data["seq_tag"]
        train_def(
            model=model,
            data=data,
            data_spatial_dim=data_spatial_dim,
            targets=targets,
            targets_spatial_dim=targets_spatial_dim,
            nbest_lengths=nbest_lengths,
            scores=scores,
            seq_tags=seq_tags,
        )
    else:
        train_def(
            model=model,
            data=data,
            data_spatial_dim=data_spatial_dim,
            targets=targets,
            targets_spatial_dim=targets_spatial_dim,
        )


def _returnn_epoch_start_hdf_writer(*, epoch: int, step: int, model: ModelT, dataset_name: str, **_kwargs_unused):
    if dataset_name == "train":
        device_id = torch.cuda.current_device()
        if device_id == 0:
            import os
            from returnn.config import get_global_config
            from i6_core.lib.hdf import get_returnn_simple_hdf_writer

            config = get_global_config()

            SimpleHDFWriter = get_returnn_simple_hdf_writer(None)
            train_dataset_dict = config.typed_value("train")
            partition_epoch = train_dataset_dict["dataset"]["datasets"]["zip_dataset"]["partition_epoch"]

            hdf_filename = f"targets-epoch-{((epoch - 1) // partition_epoch) + 1}-{device_id}.hdf"
            hdf_writer = SimpleHDFWriter(
                filename=hdf_filename,
                dim=None,
                ndim=1,
                extend_existing_file=os.path.exists(hdf_filename),
            )

            print("TMP:", hdf_writer.tmp_filename, "Filename:", hdf_writer.filename)

            config.set("train_hdf_writer", hdf_writer)


def _returnn_epoch_end_hdf_writer(*, epoch: int, step: int, model: ModelT, dataset_name: str, **_kwargs_unused):
    if dataset_name == "train":
        device_id = torch.cuda.current_device()
        if device_id == 0:
            from returnn.config import get_global_config

            config = get_global_config()

            hdf_writer = config.typed_value("train_hdf_writer")
            hdf_writer.close()
            del config.typed_dict["train_hdf_writer"]


class SumTrainDef(TrainDef):
    """
    Extended version of TrainDef, which also allows to return some additional values.
    """

    def __call__(
        self,
        *,
        model: ModelT,
        data: Tensor,
        data_spatial_dim: Tensor,
        seq_tags: Tensor = None,
        targets: Tensor,
        targets_spatial_dim: Dim,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError


class CETrainDef(TrainDef):
    """
    Extended version of TrainDef, which also allows to return some additional values.
    """

    def __call__(
        self,
        *,
        model: ModelT,
        data: Tensor,
        data_spatial_dim: Tensor,
        targets: Tensor,
        targets_spatial_dim: Dim,
        targets_indices: Tensor = None,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError
