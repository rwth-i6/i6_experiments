"""
Convert RF checkpoints between backend formats, by parameter name.

Both engines store the parameters under the RF names
(:func:`rf.Module.named_parameters`; the PT engine via ``rf_module_to_pt_module``,
the TF RF engine saves its variables under exactly those names plus ``global_step``),
so the conversion is a pure rename-free repack -- no model build needed.
See ``returnn.frontend.checkpoint`` / ``returnn.tf.checkpoint_rf`` for the same mapping
used in the other direction.

Both jobs VERIFY the written checkpoint by reading it back and comparing every tensor
bit-exactly against the source. That checks the conversion and the save/restore fidelity;
that equal parameters give equal model outputs ACROSS backends (up to float reassociation)
is covered by the RETURNN cross-backend tests
(``test_rf_tf_backend.py::test_full_model_torch_checkpoint_parity`` and the packed smokes).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from sisyphus import Job, Task

from i6_core.returnn.training import Checkpoint as TfCheckpoint, PtCheckpoint

from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg


class ConvertTfCheckpointToPtJob(Job):
    """
    Convert a TF-format RF checkpoint (written by the RF TF engine)
    into a PyTorch-engine checkpoint (``{"model": state_dict, "epoch": ..., "step": ...}``),
    so torch-based forward/recog jobs can consume a TF training.
    """

    def __init__(self, checkpoint: TfCheckpoint, *, epoch: Optional[int] = None):
        """
        :param checkpoint: TF checkpoint (index/data pair, RF variable names)
        :param epoch: stored in the PT checkpoint. Default: the ``epoch`` tensor of the
            TF checkpoint (the RF TF engine stores it, like the PT engine's dict entry);
            an old checkpoint without it and without this param gets NO epoch entry
            (the PT engine treats it as optional).
        """
        self.checkpoint = checkpoint
        self.epoch = epoch
        self.out_checkpoint = PtCheckpoint(self.output_path("checkpoint.pt"))
        self.rqmt = {"time": 1, "cpu": 2, "mem": 8}

    def tasks(self):
        """tasks"""
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        """run"""
        import numpy
        import tensorflow as tf
        import torch

        index_path = self.checkpoint.index_path.get_path()
        assert index_path.endswith(".index")
        prefix = index_path[: -len(".index")]
        reader = tf.train.load_checkpoint(prefix)
        step = 0
        epoch = self.epoch
        params: Dict[str, Any] = {}
        for name in reader.get_variable_to_shape_map():
            value = reader.get_tensor(name)
            if name == "global_step":
                step = int(value)
                continue
            if name == "epoch":
                if epoch is None:
                    epoch = int(value)
                continue
            params[name] = torch.from_numpy(numpy.asarray(value))
        out = self.out_checkpoint.path.get_path()
        torch.save(
            {"model": params, **({"epoch": int(epoch)} if epoch is not None else {}), "step": step},
            out,
        )
        # read-back verification: every tensor bit-exact vs the source
        data = torch.load(out, map_location="cpu", weights_only=False)
        assert set(data["model"]) == set(params) and data["step"] == step
        for name, value in params.items():
            assert numpy.array_equal(data["model"][name].numpy(), value.numpy()), f"mismatch on read-back: {name}"
        print(f"Converted {prefix} -> {out} ({len(params)} params, epoch {epoch}, step {step}), read-back verified")


class ConvertPtCheckpointToTfJob(Job):
    """
    Convert a PyTorch-engine RF checkpoint into a TF-format checkpoint (index/data pair),
    so TF-based forward/recog jobs can consume a torch training --
    e.g. the torch-trained LM preloaded into a combined recog model running on the TF engine.
    No model build: one TF variable per parameter, named by the RF name, plus ``global_step``.
    """

    def __init__(self, checkpoint: PtCheckpoint):
        """
        :param checkpoint: ``.pt`` file written by the PyTorch engine
            (``{"model": state_dict, ...}``, RF parameter names), or a bare state dict
        """
        self.checkpoint = checkpoint
        self.out_checkpoint = TfCheckpoint(index_path=self.output_path("checkpoint.index"))
        self.rqmt = {"time": 1, "cpu": 2, "mem": 8}

    def tasks(self):
        """tasks"""
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        """run"""
        import numpy
        import tensorflow as tf
        import torch

        data = torch.load(self.checkpoint.path.get_path(), map_location="cpu", weights_only=False)
        state_dict = data["model"] if isinstance(data, dict) and "model" in data else data
        step = int(data.get("step", 0)) if isinstance(data, dict) else 0
        epoch = data.get("epoch") if isinstance(data, dict) else None
        params = {name: value.detach().cpu().numpy() for name, value in state_dict.items()}
        prefix = self.out_checkpoint.index_path.get_path()[: -len(".index")]
        meta = {"global_step": numpy.int64(step)}
        if epoch is not None:
            meta["epoch"] = numpy.int64(epoch)  # same tensor the RF TF engine stores
        graph = tf.Graph()
        with graph.as_default():
            variables, feeds, assigns = {}, {}, []
            for name, value in {**params, **meta}.items():
                var = tf.compat.v1.get_variable(
                    name,
                    shape=value.shape,
                    dtype=value.dtype,
                    use_resource=True,
                    initializer=tf.compat.v1.zeros_initializer(),
                )
                placeholder = tf.compat.v1.placeholder(value.dtype, value.shape)
                variables[name] = var
                feeds[placeholder] = value
                assigns.append(var.assign(placeholder))
            saver = tf.compat.v1.train.Saver(var_list=variables, max_to_keep=1)
            with tf.compat.v1.Session(graph=graph) as session:
                session.run(assigns, feed_dict=feeds)
                # the meta graph is unused (we restore by variable name),
                # but the engine checks that it is there, as for any TF-engine checkpoint.
                # the state file would be 'checkpoint', which is exactly our prefix here,
                # and Saver rejects that collision even when not writing it.
                saver.save(
                    session, prefix, write_meta_graph=True, write_state=False, latest_filename="checkpoint.state"
                )
        # read-back verification: every tensor bit-exact vs the source
        reader = tf.train.load_checkpoint(prefix)
        names = set(reader.get_variable_to_shape_map())
        assert names == set(params) | set(meta), f"read-back names mismatch: {names ^ (set(params) | set(meta))}"
        assert int(reader.get_tensor("global_step")) == step
        for name, value in params.items():
            assert numpy.array_equal(reader.get_tensor(name), value), f"mismatch on read-back: {name}"
        print(
            f"Converted {self.checkpoint} -> {prefix}"
            f" ({len(params)} params, epoch {epoch}, step {step}), read-back verified"
        )


def checkpoint_as_backend(
    checkpoint: Union[TfCheckpoint, PtCheckpoint, None], backend: Optional[str]
) -> Union[TfCheckpoint, PtCheckpoint, None]:
    """
    :param checkpoint:
    :param backend: the backend that will load the checkpoint
    :return: the checkpoint in the format that backend can load:
        unchanged when they already agree (the standard case -- nothing rehashes),
        else converted via the jobs above.
    """
    if backend == "torch" and isinstance(checkpoint, TfCheckpoint):
        return ConvertTfCheckpointToPtJob(checkpoint=checkpoint).out_checkpoint
    if backend == "tensorflow" and isinstance(checkpoint, PtCheckpoint):
        return ConvertPtCheckpointToTfJob(checkpoint=checkpoint).out_checkpoint
    return checkpoint


def backend_of(definition: Union[ModelDefWithCfg, Any]) -> Optional[str]:
    """
    :param definition: the (possibly wrapped) model def
    :return: the backend it will run on, resolved as the config builders do it:
        a config ``backend`` entry wins over the def attribute.
        ``getattr(definition, "backend")`` alone does not do that,
        since ModelDefWithCfg proxies it to the wrapped def,
        so a torch def in a TF config reports "torch".
    """
    backend = getattr(definition, "backend", None)
    if isinstance(definition, ModelDefWithCfg):
        backend = definition.config.get("backend", backend)
    return backend


def checkpoint_for_backend(
    definition: Union[ModelDefWithCfg, Any], checkpoint: Union[TfCheckpoint, PtCheckpoint, None]
) -> Union[TfCheckpoint, PtCheckpoint, None]:
    """
    :param definition: the (possibly wrapped) model def the checkpoint is loaded into.
        Its resolved backend decides the required checkpoint format.
    :param checkpoint:
    :return: see :func:`checkpoint_as_backend`
    """
    return checkpoint_as_backend(checkpoint, backend_of(definition))
