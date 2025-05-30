"""
Interfaces/configs to read hypotheses from the HDF from the first pass
and then do rescoring
"""

from __future__ import annotations

import os
import dataclasses
from typing import TYPE_CHECKING, Optional, Union, Any, Dict, Sequence, Collection, Iterator, Callable, List
import copy

import sisyphus
from sisyphus import tk
from sisyphus import tools as sis_tools
from i6_core.util import instanciate_delayed

import returnn.frontend as rf
from torch.nn.utils.rnn import pad_sequence
from returnn.tensor import Tensor, Dim, batch_dim
from returnn.tensor.dim import DimTypes
from returnn.config import get_global_config
import torch

from i6_core.returnn import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob, PtCheckpoint, AverageTorchCheckpointsJob
from i6_core.returnn.search import (
    ReturnnSearchJobV2,
    SearchRemoveLabelJob,
    SearchCollapseRepeatedLabelsJob,
    SearchTakeBestJob,
)
from i6_core.returnn.forward import ReturnnForwardJobV2
from returnn_common import nn
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.common.setups import serialization
from i6_experiments.users.zeyer.utils.serialization import get_import_py_code

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.datasets.task import Task
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput, ScoreResultCollection
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, serialize_model_def
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint, ModelWithCheckpoints
from i6_experiments.users.zeyer.returnn.training import get_relevant_epochs_from_training_learning_rate_scores

from i6_experiments.users.phan.utils.masking import get_seq_mask
from i6_experiments.users.phan.utils.pseudo_ppl import compute_log_pseudo_ppl_loop_s_rf_models
from i6_experiments.users.phan.rf_models.lstm_lm import LSTMLMRF

if TYPE_CHECKING:
    from returnn.tensor import TensorDict
from i6_experiments.users.phan.rescoring.first_pass import HdfRescoringOutput

_v2_forward_out_filename = "output.py.gz"
_v2_forward_ext_out_filename = "output_ext.py.gz"

SharedPostConfig = {
    # In case pretraining overwrites some of these, they need a default.
    "accum_grad_multiple_step": None,
    "use_last_best_model": None,
}

def dummy_get_model(*, epoch: int, **_kwargs_unused):
    from returnn.tensor import Tensor
    from returnn.tensor import Dim, batch_dim
    from returnn.config import get_global_config

    config = get_global_config()
    feature_dim = Dim(description='feature', dimension=80)
    vocab_dim = Dim(description='vocab', dimension=10026)


    model_def = config.typed_value("_model_def")
    model = model_def(epoch=epoch, in_dim=feature_dim, target_dim=vocab_dim)
    return model

def rescoring_bi_ilm_forward(*, model, extern_data: TensorDict, **_kwargs_unused):
    """
    Forward step to read the hypotheses from the HDF and then do rescoring
    """
    

    if rf.is_executing_eagerly():
        batch_size = int(batch_dim.get_dim_value())
        for batch_idx in range(batch_size):
            seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
            print(f"batch {batch_idx+1}/{batch_size} seq_tag: {seq_tag!r}")

    config = get_global_config()
    rescoring_args = config.typed_value("rescoring_args", {})

    # Read and unpack hypotheses from HDF
    hyps = extern_data['hyps']
    lens = extern_data['hyp_lens']
    am_score = extern_data['am_scores']
    lm_score = extern_data['lm_scores']
    hyps_raw = hyps.raw_tensor # (batch, beam * T)
    lens_raw = lens.raw_tensor # (batch, beam)
    batch_size = hyps_raw.shape[0]
    beam_size = lens_raw.shape[1]
    max_length = torch.max(lens_raw)
    hyps_unpack = []
    for b in range(batch_size):
        length_b =lens_raw[b]
        length_b_total = length_b.sum()
        flat_hyp = hyps_raw[b,:length_b_total]
        split_hyp = torch.split(flat_hyp, length_b.tolist())
        pad_hyp = pad_sequence(split_hyp,batch_first=True)
        hyps_unpack.append(pad_hyp)
    max_T = max([x.size(1) for x in hyps_unpack])
    for i in range(len(hyps_unpack)):
        pad_len = max_T - hyps_unpack[i].size(1)
        if pad_len > 0:
            pad_tensor = torch.zeros((beam_size, pad_len), dtype=hyps_raw.dtype, device=hyps_raw.device)
            hyps_unpack[i] = torch.cat([hyps_unpack[i], pad_tensor], dim=1)
    hyps_unpack = torch.stack(hyps_unpack) # (batch, beam, T)
    hyps_unpack = hyps_unpack[:,:,1:]
    lens = lens-1

    len_is_zero = lens.raw_tensor == 0
    lens.raw_tensor = torch.where(
        len_is_zero,
        torch.ones_like(lens.raw_tensor, device=lens.raw_tensor.device),
        lens.raw_tensor
        )

    # # --------- debug print ----------
    # for b1 in range(batch_size):
    #     for b2 in range(beam_size):
    #         if lens.raw_tensor[b1][b2] == 0:
    #             print(f"Hyps len 0 found, batch {b1}, beam {b2}")
    #             print(f"Hyp: {hyps_unpack[b1][b2]}")

    # print(hyps_unpack)
    # print(lens)
    # print(am_score.raw_tensor)
    # print(lm_score.raw_tensor)

    # Construct RF tensor to feed to the ILM
    # lens_rf = rf.Tensor(name="hyps_len", raw_tensor=lens, dtype="int64")
    batch_dim_flatten = Dim(name="batch_beam_merge", dimension=batch_size*beam_size)
    lens_rf_flatten = rf.Tensor(
        name="lens_flatten",
        dims=[batch_dim_flatten],
        raw_tensor=lens.raw_tensor.long().flatten().cpu(),
        dtype="int64",
    )
    spatial_dim = Dim(name="out-spatial", dyn_size_ext=lens, dimension=lens, kind=DimTypes.Spatial)
    beam_dim = Dim(name="beam", dimension=lens.raw_tensor.shape[1])
    spatial_dim_flatten = Dim(
        name="out-spatial-flatten",
        dyn_size_ext=lens_rf_flatten,
        dimension=lens_rf_flatten,
        kind=DimTypes.Spatial,
        )
    hyps_unpack_rf = rf.Tensor(
        name="hyps_unpack",
        dims=[batch_dim, beam_dim, spatial_dim],
        raw_tensor=hyps_unpack.long(),
        dtype="int64",
        sparse_dim=model.target_dim,
        )
    hyps_merge_batch_beam = rf.Tensor(
        name="hyps_unpack_merge_batch_beam",
        dims=[batch_dim_flatten, spatial_dim_flatten],
        raw_tensor=hyps_unpack.flatten(0, 1).long(),
        dtype="int64",
        sparse_dim=model.ilm.input_dim,
        )

    # Feed the sequences to the ILM and get the pseudo PPL
    # then do the rescoring
    MASK_TOKEN = model.ilm.input_dim.capacity - 1
    mlm_metric = rescoring_args.get("mlm_metric", "pseudoPpl")

    raw_rescored_scores = am_score.raw_tensor \
        + lm_score.raw_tensor * rescoring_args.get("lm_scale", 0.0)

    if rescoring_args.get("ilm_scale", 0.0) > 0.0:
        if isinstance(model.ilm, LSTMLMRF): # rescoring for autoregressive ILM
            hyps_merge_batch_beam_w_bos, out_pad_dims = rf.pad(
                hyps_merge_batch_beam,
                padding=[(1, 0)],
                axes=[spatial_dim_flatten],
                value=model.eos_idx
            )
            ilm_out = model.ilm_forward(hyps_merge_batch_beam_w_bos, out_pad_dims[0])
            ilm_out_raw = ilm_out["output"].raw_tensor # (T, B, V)
            log_lm_score = ilm_out_raw.transpose(0, 1).log_softmax(-1) # (B, T, V)
            targets_eos = torch.cat(
                [hyps_merge_batch_beam.raw_tensor,
                 torch.full((batch_size*beam_size, 1),fill_value=model.eos_idx,device=hyps_merge_batch_beam.raw_tensor.device)],
                dim=1,
            ).long()
            ce = torch.nn.functional.cross_entropy(log_lm_score.transpose(1, 2), targets_eos, reduction='none')
            seq_mask = get_seq_mask(
                lens_rf_flatten.raw_tensor + 1,
                lens_rf_flatten.raw_tensor.max() + 1,
                device=ce.device
                )
            raw_ilm_score = (ce*seq_mask).sum(-1)

        elif mlm_metric == "pseudoPpl":
            raw_ilm_score = compute_log_pseudo_ppl_loop_s_rf_models(
                model.ilm,
                hyps_merge_batch_beam,
                targets_spatial_dim=spatial_dim_flatten,
                mask_idx=MASK_TOKEN,
                model_kwargs={"batch_dims": batch_dim_flatten},
                )
        elif mlm_metric == "pseudoPplL2R":
            from i6_experiments.users.phan.utils.pseudo_ppl import compute_log_pseudo_ppl_variants_loop_s_rf_models
            out_ce, out_seq_mask = compute_log_pseudo_ppl_variants_loop_s_rf_models(
                model.ilm,
                hyps_merge_batch_beam,
                targets_spatial_dim=spatial_dim_flatten,
                mask_idx=MASK_TOKEN,
                model_kwargs={"batch_dims": batch_dim_flatten},
                variant="l2r",
            )
            raw_ilm_score = (out_ce * out_seq_mask).sum(-1)
        else:
            raise ValueError(f"The MLM metric {mlm_metric} is not supported")
        raw_rescored_scores -= raw_ilm_score.unflatten(0, (batch_size, beam_size)) * rescoring_args.get("ilm_scale", 0.0)

    raw_rescored_scores = torch.where(
        len_is_zero,
        torch.full_like(raw_rescored_scores, fill_value=-1e25, device=raw_rescored_scores.device),
        raw_rescored_scores
    )

    rescored_scores_rf = rf.Tensor(
        name="rescores-scores",
        dims=[batch_dim, beam_dim],
        raw_tensor=raw_rescored_scores,
        dtype="float32",
    )

    # # --------- debug print ----------
    # print("Flatten sanity check")
    # for b in range(hyps_merge_batch_beam.raw_tensor.shape[0]):
    #     if lens_rf_flatten.raw_tensor[b] == 0:
    #         print(f"Hyps len 0 found, flatten idx {b}")
    #         print(f"Flattened hyp: {hyps_merge_batch_beam.raw_tensor[b]}")

    rf.get_run_ctx().mark_as_output(hyps_unpack_rf, "hyps")
    # print("hyps", hyps_unpack_rf.raw_tensor)
    lens.feature_dim_axis = None
    rf.get_run_ctx().mark_as_output(lens, "hyps_len")
    rf.get_run_ctx().mark_as_output(rescored_scores_rf, "rescored_scores")
    # print("Rescored", rescored_scores_rf.raw_tensor)

def _hdf_rescoring_get_model(*, epoch: int, **_kwargs_unused):
    from returnn.tensor import Tensor
    from returnn.config import get_global_config

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    extern_data_dict = config.typed_value("model_def_extern_data")
    data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    assert targets.sparse_dim

    model_def = config.typed_value("_model_def")
    model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
    return model

# all these things to import the extern data for the model def
# without having to touch returnn_common.nn
from returnn_common.nn.naming import ReturnnConfigSerializer, ReturnnDimTagsProxy
def serialize_extern_data(
    hdf_extern_data,
    model_def_extern_data,
    extern_data_name="extern_data",
    model_def_extern_data_name="model_def_extern_data",
    cls=ReturnnConfigSerializer):
    """
    serialize two extern datas: one for the HDF, one for the model
    """
    dim_tags_proxy = ReturnnDimTagsProxy()
    from returnn.util.pprint import pformat

    hdf_extern_data = dim_tags_proxy.collect_dim_tags_and_transform_config(hdf_extern_data)
    model_def_extern_data = dim_tags_proxy.collect_dim_tags_and_transform_config(model_def_extern_data)

    code_lines = [
        cls.ImportPyCodeStr,
        f"{dim_tags_proxy.py_code_str()}\n",
        f"{extern_data_name} = {pformat(hdf_extern_data)}\n",
        f"{model_def_extern_data_name} = {pformat(model_def_extern_data)}\n",
    ]
    return "".join(code_lines)

def bi_ilm_rescoring_hdf_job(
    path: tk.Path,
    rescoring_config: Dict,
    model: ModelWithCheckpoint,
    model_def,
    beam_size=32,
):
    from i6_experiments.common.setups.returnn.serialization import get_serializable_config
    from returnn.tensor import Dim, batch_dim
    data_dict={
        "class": "MetaDataset",
        "data_map": {
            "hyps": ("hyps", "data"),
            "am_scores": ("hyps", "am_scores"),
            "lm_scores": ("hyps", "lm_scores"),
            "hyp_lens": ("hyps", "hyp_lens"),
        },
        "datasets": {
            "hyps":{
                "class": "HDFDataset",
                "files": [path],
                "use_cache_manager": True,

            }
        },
        "seq_order_control_dataset": "hyps"
    }
    out_spatial_dim = Dim(description="out_spatial", dimension=None, kind=Dim.Types.Spatial)
    score_dim = Dim(description='score', dimension=beam_size, kind = None)
    vocab_dim = Dim(description="vocab", dimension=10025, kind=Dim.Types.Feature)
    beam_dim = Dim(description='beam_size', dimension=beam_size, kind=None)
    time_dim = Dim(description="time", dimension=None, kind=Dim.Types.Spatial)
    audio_dim = Dim(description="audio", dimension=1, kind=Dim.Types.Feature)

    # this is used for model def
    model_def_extern_data = {
        "data": {"dim_tags": [batch_dim, time_dim, audio_dim]},
        "classes": {
            "dim_tags": [batch_dim, out_spatial_dim],
            "sparse_dim": vocab_dim,
            # "vocab": vocab,
        },
    }

    extern_data_raw = {
        "hyps": {"dim_tags": [batch_dim, out_spatial_dim], "sparse_dim": vocab_dim},
        "hyp_lens": {"dim_tags":[batch_dim, beam_dim]},
        "am_scores": {"dim_tags": [batch_dim, beam_dim]},
        "lm_scores": {"dim_tags": [batch_dim, beam_dim]},
    }
    extern_data_raw = instanciate_delayed(extern_data_raw)
    model_def_extern_data = instanciate_delayed(model_def_extern_data)
    returnn_recog_config_dict = dict(
        backend="torch",
        behavior_version=21,
        # dataset
        default_input="data",
        target="classes",
        forward_data=data_dict,
    )
    returnn_recog_config_dict.update(rescoring_config)
    returnn_recog_config = ReturnnConfig(
        config=returnn_recog_config_dict,
        python_epilog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(get_import_py_code()),
                    serialization.NonhashedCode(
                        serialize_extern_data(extern_data_raw, model_def_extern_data)
                    ),
                    *serialize_model_def(model_def),
                    serialization.Import(_hdf_rescoring_get_model, import_as="get_model"),
                    serialization.Import(rescoring_bi_ilm_forward, import_as="forward_step"),
                    serialization.Import(bi_ilm_rescoring_callback, import_as="forward_callback"),
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

    for k, v in dict(
            batching="sorted",
            batch_size=20000,
            max_seqs=20,
    ).items():
        if k in returnn_recog_config.config:
            v = returnn_recog_config.config.pop(k)
        if k in returnn_recog_config.post_config:
            v = returnn_recog_config.post_config.pop(k)
        returnn_recog_config.config[k] = v


    for k, v in SharedPostConfig.items():
        if k in returnn_recog_config.config or k in returnn_recog_config.post_config:
            continue
        returnn_recog_config.post_config[k] = v

    search_job = ReturnnForwardJobV2(
        model_checkpoint=model.checkpoint,
        returnn_config=returnn_recog_config,
        output_files=["output.py.gz"],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        mem_rqmt=6,
        time_rqmt=7,
    )
    return search_job

def bi_ilm_rescoring_callback():
    from typing import TextIO
    from returnn.tensor import Tensor, TensorDict
    from returnn.forward_iface import ForwardCallbackIface
    from returnn.config import get_global_config

    config = get_global_config()
    recog_def_ext = config.bool("__recog_def_ext", False)

    class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.out_file: Optional[TextIO] = None
            self.out_ext_file: Optional[TextIO] = None

        def init(self, *, model):
            import gzip

            self.out_file = gzip.open(_v2_forward_out_filename, "wt")
            self.out_file.write("{\n")

            config = get_global_config()
            self.bpe_idx_to_label = model.bpe_idx_to_label
            if recog_def_ext:
                self.out_ext_file = gzip.open(_v2_forward_ext_out_filename, "wt")
                self.out_ext_file.write("{\n")

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            hyps: Tensor = outputs["hyps"]  # [T, beam], not sure why returnn trasnposed it
            scores: Tensor = outputs["rescored_scores"]  # [beam]
            hyps_len: Tensor = outputs["hyps_len"]  # [beam]
            # print("callback hyps", hyps.raw_tensor)
            # print("callback_hyps shape", hyps.shape)
            # print("scores", scores.raw_tensor)
            # AED/Transducer etc will have hyps len depending on beam -- however, CTC will not.
            raw_hyps = hyps.raw_tensor.transpose() # [beam, T]
            num_beam = raw_hyps.shape[0]
            assert raw_hyps.shape[0] == scores.raw_tensor.shape[0]
            # Consistent to old search task, list[(float,str)].
            self.out_file.write(f"{seq_tag!r}: [\n")
            for i in range(num_beam):
                score = float(scores.raw_tensor[i])
                hyp_ids = raw_hyps[
                    i, : hyps_len.raw_tensor[i] if hyps_len.raw_tensor.shape else hyps_len.raw_tensor
                ]
                hyp_serialized = " ".join([self.bpe_idx_to_label[bpe_idx] for bpe_idx in hyp_ids])
                self.out_file.write(f"  ({score!r}, {hyp_serialized!r}),\n")
            self.out_file.write("],\n")


        def finish(self):
            self.out_file.write("}\n")
            self.out_file.close()
            if self.out_ext_file:
                self.out_ext_file.write("}\n")
                self.out_ext_file.close()

    return _ReturnnRecogV2ForwardCallbackIface()

def second_pass_search_dataset(
    *,
    first_pass_hdf: tk.Path,
    model: ModelWithCheckpoint,
    recog_def: RecogDef,
    rescoring_config: Optional[Dict[str, Any]] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    search_mem_rqmt: Union[int, float] = 6,
    merge_contraction: bool = False,
    search_rqmt: Optional[Dict[str, Any]] = None,
    search_alias_name: Optional[str] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    dataset_name: Optional[str] = None,
    train_exp_name: Optional[str] = None,
) -> RecogOutput:
    """
    recog on the specific dataset
    """
    env_updates = None
    if (rescoring_config and rescoring_config.get("__env_updates")) or (search_post_config and search_post_config.get("__env_updates")):
        env_updates = (rescoring_config and rescoring_config.pop("__env_updates", None)) or (
            search_post_config and search_post_config.pop("__env_updates", None)
        )
    assert getattr(model.definition, "backend", None) == "torch"
    out_files = [_v2_forward_out_filename]
    if rescoring_config and rescoring_config.get("__recog_def_ext", False):
        out_files.append(_v2_forward_ext_out_filename)
    second_pass_search_job = bi_ilm_rescoring_hdf_job(
        path=first_pass_hdf,
        rescoring_config=rescoring_config,
        model=model,
        model_def=model.definition,
        beam_size=rescoring_config["rescoring_args"]["beam_size"],
    )
    second_pass_search_job.set_vis_name(f"Rescoring job {train_exp_name}, {os.path.split(model.checkpoint.__repr__())[-1][:-1]}, {dataset_name}, {rescoring_config.get('rescoring_args', '') if rescoring_config else ''}")
    res = second_pass_search_job.out_files[_v2_forward_out_filename]

    if search_rqmt:
        second_pass_search_job.rqmt.update(search_rqmt)
    if env_updates:
        for k, v in env_updates.items():
            second_pass_search_job.set_env(k, v)
    if search_alias_name:
        second_pass_search_job.add_alias(search_alias_name)
    if recog_def.output_blank_label:
        # collapes of the repeated label is handled in the rec function
        #res = SearchCollapseRepeatedLabelsJob(res, output_gzip=True).out_search_results
        res = SearchRemoveLabelJob(res, remove_label=recog_def.output_blank_label, output_gzip=True).out_search_results
    for f in recog_post_proc_funcs:  # for example BPE to words
        res = f(RecogOutput(output=res)).output
    if recog_def.output_with_beam:
        # Don't join scores here (SearchBeamJoinScoresJob).
        #   It's not clear whether this is helpful in general.
        #   As our beam sizes are very small, this might boost some hyps too much.
        res = SearchTakeBestJob(res, output_gzip=True).out_best_search_results
    if merge_contraction:
        from i6_experiments.users.phan.datasets.librispeech_tedlium2 import MergeContractionsJob
        res = MergeContractionsJob(res).out_dict
    return RecogOutput(output=res)
