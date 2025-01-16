"""
Rescoring multiple text dicts / search outputs.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Any, Dict, List, Tuple, Set
from sisyphus import Job, Task, tk

import i6_core.util as util
from i6_core.returnn import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2

from returnn_common.datasets_old_2022_10.interface import DatasetConfig

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RescoreDef
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput

from i6_experiments.users.zeyer.recog import (
    _returnn_v2_get_model,
    _returnn_v2_get_forward_callback,
    _v2_forward_out_filename,
)

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim, TensorDict


def combine_scores(scores: Dict[RecogOutput, float]) -> RecogOutput:
    """
    Combine scores from multiple sources, linearly weighted by the given weights.

    :param scores: dict: recog output -> weight. We assume they have the same hyp txt.
    :param same_txt: if all the hyps have the same txt across scores.
    :param out_txt: used to define the hyp txt if not same_txt
    :return: combined scores
    """
    assert scores
    job = SearchCombineScoresJob({recog_output.output: weight for recog_output, weight in scores.items()})
    return RecogOutput(output=job.out_search_results)


class SearchCombineScoresJob(Job):
    """
    Takes a number of files, each with the same N-best list, including scores,
    and combines the scores with some weights.
    """

    def __init__(self, search_py_output: Dict[tk.Path, float], *, output_gzip: bool = True):
        """
        :param search_py_output: dict: search output file from RETURNN in python format (n-best list) -> weight
        :param output_gzip: gzip the output
        """
        assert len(search_py_output) > 0
        self.search_py_output = search_py_output
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        data: List[Tuple[float, Dict[str, List[Tuple[float, str]]]]] = [
            (weight, eval(util.uopen(fn, "rt").read(), {"nan": float("nan"), "inf": float("inf")}))
            for fn, weight in self.search_py_output.items()
        ]
        weights: List[float] = [weight for weight, _ in data]
        seq_tags: List[str] = list(data[0][1].keys())
        seq_tags_set: Set[str] = set(seq_tags)
        assert len(seq_tags) == len(seq_tags_set), "duplicate seq tags"
        for _, d in data:
            assert set(d.keys()) == seq_tags_set, "inconsistent seq tags"

        with util.uopen(self.out_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag in seq_tags:
                data_: List[List[Tuple[float, str]]] = [d[seq_tag] for _, d in data]
                hyps_: List[List[str]] = [[h for _, h in entry] for entry in data_]
                hyps0: List[str] = hyps_[0]
                assert isinstance(hyps0, list) and all(isinstance(h, str) for h in hyps0)
                assert all(hyps0 == hyps for hyps in hyps_)
                scores_per_hyp: List[List[float]] = [[score for score, _ in entry] for entry in data_]
                # n-best list as [(score, text), ...]
                out.write(f"{seq_tag!r}: [\n")
                for hyp_idx, hyp in enumerate(hyps0):
                    score = sum(scores_per_hyp[file_idx][hyp_idx] * weights[file_idx] for file_idx in range(len(data)))
                    out.write(f"({score!r}, {hyp!r}),\n")
                out.write("],\n")
            out.write("}\n")


def rescore(
    *,
    recog_output: RecogOutput,
    dataset: Optional[DatasetConfig] = None,
    vocab: tk.Path,
    model: ModelWithCheckpoint,
    rescore_def: RescoreDef,
    config: Optional[Dict[str, Any]] = None,
    forward_post_config: Optional[Dict[str, Any]] = None,
    forward_rqmt: Optional[Dict[str, Any]] = None,
    forward_device: str = "gpu",
    forward_alias_name: Optional[str] = None,
) -> RecogOutput:
    """
    Rescore on the specific dataset, given some hypotheses, using the :class:`RescoreDef` interface.

    :param recog_output: output from previous recog, with the hyps to rescore
    :param dataset: dataset to forward, using its get_main_dataset(),
        and also get_default_input() to define the default output,
        and get_extern_data().
    :param vocab:
    :param model:
    :param rescore_def:
    :param config: additional RETURNN config opts for the forward job
    :param forward_post_config: additional RETURNN post config (non-hashed) opts for the forward job
    :param forward_rqmt: additional rqmt opts for the forward job (e.g. "time" (in hours), "mem" (in GB))
    :param forward_device: "cpu" or "gpu". if not given, will be "gpu" if model is given, else "cpu"
    :param forward_alias_name: optional alias name for the forward job
    :return: new scores
    """
    env_updates = None
    if (config and config.get("__env_updates")) or (forward_post_config and forward_post_config.get("__env_updates")):
        env_updates = (config and config.pop("__env_updates", None)) or (
            forward_post_config and forward_post_config.pop("__env_updates", None)
        )
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=model.checkpoint.path,
        returnn_config=_returnn_rescore_config(
            recog_output=recog_output,
            vocab=vocab,
            dataset=dataset,
            model_def=model.definition,
            rescore_def=rescore_def,
            config=config,
            post_config=forward_post_config,
        ),
        output_files=[_v2_forward_out_filename],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        device=forward_device,
    )
    if forward_rqmt:
        forward_job.rqmt.update(forward_rqmt)
    if env_updates:
        for k, v in env_updates.items():
            forward_job.set_env(k, v)
    if forward_alias_name:
        forward_job.add_alias(forward_alias_name)
    return RecogOutput(output=forward_job.out_files[_v2_forward_out_filename])


# Those are applied for both training, recog and potential others.
# The values are only used if they are neither set in config nor post_config already.
# They should also not infer with other things from the epilog.
SharedPostConfig = {
    # In case pretraining overwrites some of these, they need a default.
    "accum_grad_multiple_step": None,
    "use_last_best_model": None,
}


def _returnn_rescore_config(
    *,
    recog_output: RecogOutput,
    vocab: tk.Path,
    dataset: Optional[DatasetConfig] = None,
    model_def: Union[ModelDef, ModelDefWithCfg],
    rescore_def: RescoreDef,
    config: Optional[Dict[str, Any]] = None,
    post_config: Optional[Dict[str, Any]] = None,
) -> ReturnnConfig:
    """
    Create config for rescoring.
    """
    from i6_experiments.users.zeyer.serialization_v2 import ReturnnConfigWithNewSerialization

    config = config.copy() if config else {}

    if dataset:
        # TODO make MetaDataset, putting this inside, alongside with the dataset for recog output
        raise NotImplementedError(f"dataset {dataset} ...")

    # Beam dim size unknown. Usually static size, but it's ok to leave this unknown here (right?).
    beam_dim = Dim(Tensor("beam_size", dims=[], dtype="int32"), name="beam")

    # Note: we should not put SPM/BPE directly here,
    # because the recog output still has individual labels,
    # so no SPM/BPE encoding on the text.
    vocab_opts = {"class": "Vocabulary", "vocab_file": vocab, "unknown_label": None}

    config.update(
        {
            "forward_data": {"class": "TextDictDataset", "filename": recog_output.output, "vocab": vocab_opts},
            # default input/target should not matter
            "default_input": None,
            "target": None,
            "_beam_dim": beam_dim,
            "extern_data": {
                # data_flat dyn dim is the flattened dim, no need to define dim tags now
                "data_flat": {"shape": [None], "dtype": "int32", "vocab": vocab_opts},
                "data_seq_lens": {"dims": [beam_dim], "dtype": "int32"},
            },
        }
    )

    if "backend" not in config:
        config["backend"] = model_def.backend
    config["behavior_version"] = max(model_def.behavior_version, config.get("behavior_version", 0))

    if isinstance(model_def, ModelDefWithCfg):
        config["_model_def"] = model_def.model_def
        config.update(model_def.config)
    else:
        config["_model_def"] = model_def
    config["get_model"] = _returnn_v2_get_model
    config["_rescore_def"] = rescore_def
    config["forward_step"] = _returnn_score_step
    config["forward_callback"] = _returnn_v2_get_forward_callback

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
        batch_size=(20_000 * model_def.batch_size_factor) if model_def else (20_000 * 160),
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

    return ReturnnConfigWithNewSerialization(config, post_config)


def _returnn_score_step(*, model, extern_data: TensorDict, **_kwargs_unused):
    # Similar to i6_experiments.users.zeyer.recog._returnn_v2_forward_step,
    # but using score_def instead of recog_def.
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
    if default_input_key:
        data = extern_data[default_input_key]
        data_spatial_dim = data.get_time_dim_tag()
    else:
        data, data_spatial_dim = None, None

    targets_beam_dim = config.typed_value("_beam_dim")
    targets_flat = extern_data["data_flat"]
    targets_flat_time_dim = targets_flat.get_time_dim_tag()
    targets_seq_lens = extern_data["data_seq_lens"]  # [B, beam]
    targets_spatial_dim = Dim(targets_seq_lens, name="targets_spatial")
    targets = rf.pad_packed(targets_flat, in_dim=targets_flat_time_dim, dims=[targets_beam_dim, targets_spatial_dim])

    rescore_def: RescoreDef = config.typed_value("_rescore_def")
    scores = rescore_def(
        model=model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        targets=targets,
        targets_spatial_dim=targets_spatial_dim,
        targets_beam_dim=targets_beam_dim,
    )
    assert isinstance(scores, Tensor)
    rf.get_run_ctx().mark_as_output(targets, "hyps", dims=[batch_dim, targets_beam_dim, targets_spatial_dim])
    rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim, targets_beam_dim])
