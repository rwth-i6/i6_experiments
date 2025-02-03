from typing import Union, Optional

from sisyphus import *

from i6_core.returnn.config import ReturnnConfig
from i6_core.util import instanciate_delayed

from i6_experiments.common.setups import serialization
from i6_experiments.users.zeyer.model_interfaces.model import ModelDef, ModelDefWithCfg, serialize_model_def
from i6_experiments.users.zeyer.datasets.librispeech import LibrispeechLmDataset

from i6_experiments.users.zeyer.train_v3 import _returnn_v2_get_model
from i6_experiments.users.zeyer.recog import _v2_forward_out_filename
from i6_experiments.users.zeyer.utils.serialization import get_import_py_code
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep

from returnn_common import nn

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict


def lm_forward_def(*, model: rf.Module, targets: Tensor, targets_spatial_dim: Dim, **_other) -> Tensor:
    # noinspection PyTypeChecker
    vocab = model.vocab_dim.vocab
    assert vocab.bos_label_id is not None and vocab.eos_label_id is not None

    _, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=vocab.bos_label_id
    )
    targets_w_eos, _ = rf.pad(
        targets,
        axes=[targets_spatial_dim],
        padding=[(0, 1)],
        value=vocab.eos_label_id,
        out_dims=[targets_w_eos_spatial_dim],
    )

    batch_dims = targets.remaining_dims(targets_spatial_dim)
    logits, _ = model(
        targets,
        spatial_dim=targets_spatial_dim,
        out_spatial_dim=targets_w_eos_spatial_dim,
        state=model.default_initial_state(batch_dims=batch_dims),
    )

    log_prob = rf.log_softmax(logits, axis=model.vocab_dim)
    log_prob_targets = rf.gather(log_prob, indices=targets_w_eos, axis=model.vocab_dim)
    log_prob_targets_seq = rf.reduce_sum(log_prob_targets, axis=targets_w_eos_spatial_dim)  # [batch,beam]
    assert log_prob_targets_seq.dims_set == set(batch_dims)

    return targets_w_eos, log_prob_targets_seq


def _returnn_forward_step(*, model: rf.Module, extern_data: TensorDict, **kwargs_unused):
    import returnn.frontend as rf
    from returnn.tensor import batch_dim
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    data = extern_data[default_input_key]
    target = extern_data[default_target_key]
    data_spatial_dim = data.get_time_dim_tag()
    forward_def = config.typed_value("_forward_def")
    forward_out = forward_def(model=model, targets=target, targets_spatial_dim=data_spatial_dim)

    hyps, scores = forward_out
    rf.get_run_ctx().mark_as_output(hyps, "hyps", dims=[batch_dim, hyps.dims[1]])
    rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim])


def _returnn_forward_callback():
    from typing import TextIO
    from returnn.tensor import Tensor, TensorDict
    from returnn.forward_iface import ForwardCallbackIface

    class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.data = {}

        def init(self, *, model):
            pass

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            hyps: Tensor = outputs["hyps"]  # [out_spatial]
            scores: Tensor = outputs["scores"]  # []
            assert hyps.sparse_dim and hyps.sparse_dim.vocab  # should come from the model
            assert hyps.dims[0].dyn_size_ext is not None, f"hyps {hyps} do not define seq lengths"
            hyps_len = hyps.dims[0].dyn_size_ext
            self.data[seq_tag] = [hyps_len.raw_tensor.item(), scores.raw_tensor.item()]

        def finish(self):
            import json
            import gzip

            out_file_fp = gzip.open(_v2_forward_out_filename, "wt", encoding="utf-8")
            json.dump(self.data, out_file_fp)

    return _ReturnnRecogV2ForwardCallbackIface()


def _returnn_ppl_config(model_def: ModelDef, dataset: LibrispeechLmDataset, dataset_key: str) -> ReturnnConfig:
    from i6_experiments.users.zeyer.utils.sis_setup import get_base_module

    unhashed_package_root_model_def, setup_base_name_model_def = get_base_module(
        model_def.model_def if isinstance(model_def, ModelDefWithCfg) else model_def
    )

    returnn_config = dict(
        backend=model_def.backend,
        behavior_version=model_def.behavior_version,
        default_input=dataset.get_default_input(),
        target=dataset.get_default_target(),
        forward_data=dataset.get_dataset(dataset_key),
        batch_size=100_000,
    )

    if isinstance(model_def, ModelDefWithCfg):
        returnn_config = dict_update_deep(returnn_config, model_def.config)

    extern_data_raw = instanciate_delayed(dataset.get_extern_data())

    serial_collection = [
        serialization.NonhashedCode(get_import_py_code()),
        serialization.NonhashedCode(
            nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
        ),
        *serialize_model_def(model_def, unhashed_package_root=unhashed_package_root_model_def),
        serialization.Import(_returnn_v2_get_model, import_as="get_model"),
        serialization.Import(lm_forward_def, import_as="_forward_def", ignore_import_as_for_hash=True),
        serialization.Import(_returnn_forward_step, import_as="forward_step"),
        serialization.Import(_returnn_forward_callback, import_as="forward_callback"),
        serialization.ExplicitHash({"version": "4"}),
        serialization.PythonEnlargeStackWorkaroundNonhashedCode,
        serialization.PythonCacheManagerFunctionNonhashedCode,
        serialization.PythonModelineNonhashedCode,
    ]

    returnn_config = ReturnnConfig(config=returnn_config, python_epilog=serialization.Collection(serial_collection))

    return returnn_config


def compute_ppl(*, prefix_name, model_with_checkpoints, dataset, dataset_key):
    from i6_core.returnn.forward import ReturnnForwardJobV2
    from i6_experiments.users.zeyer import tools_paths

    returnn_config = _returnn_ppl_config(model_with_checkpoints.definition, dataset, dataset_key)

    for epoch in model_with_checkpoints.fixed_epochs:
        res = ReturnnForwardJobV2(
            model_checkpoint=model_with_checkpoints.get_epoch(epoch).checkpoint,
            returnn_config=returnn_config,
            output_files=[_v2_forward_out_filename],
            returnn_python_exe=tools_paths.get_returnn_python_exe(),
            returnn_root=tools_paths.get_returnn_root(),
        )
        ppl_job = ComputePerplexityJob(scores_and_lens_file=res.out_files[_v2_forward_out_filename])
        tk.register_output(f"ppl/{prefix_name}", ppl_job.out_ppl)


class ComputePerplexityJob(Job):
    def __init__(self, scores_and_lens_file: Optional[Path]):
        self.scores_and_lens_file = scores_and_lens_file

        self.out_ppl = self.output_path("ppl")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

    def run(self):
        import math
        import json

        fpath = self.scores_and_lens_file.get_path()
        if fpath.endswith(".gz"):
            import gzip

            open_func = gzip.open
        else:
            open_func = open

        with open_func(fpath, "rt") as f:
            d = json.load(f)

        scores = 0.0
        lens = 0
        for v in d.values():
            scores += v[1]
            lens += v[0]

        ppl = math.exp(-1.0 * scores / lens)

        with open(self.out_ppl.get_path(), "w+") as f:
            f.write("Perplexity: %f" % ppl)
