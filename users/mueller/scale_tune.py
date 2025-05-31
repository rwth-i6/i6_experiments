from __future__ import annotations

import torch
from typing import Optional, Any, Dict

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.datasets.util.vocabulary import Vocabulary

from sisyphus import tk, Task
from sisyphus.job_path import Variable

from i6_core.returnn.search import SearchOutputRawReplaceJob
from i6_core.returnn import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2

from i6_experiments.users.mueller.datasets.task import Task
from i6_experiments.users.mueller.recog import search_dataset
from i6_experiments.users.mueller.experiments.language_models.ffnn import FeedForwardLm
from i6_experiments.users.mueller.experiments.ctc_baseline.ctc import model_recog as model_recog_ctc_only, model_recog_lm_albert as model_recog_recomb

from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint, RescoreDef
from i6_experiments.users.zeyer.recog import ctc_alignment_to_label_seq, _returnn_v2_get_forward_callback, _v2_forward_out_filename
from i6_experiments.users.zeyer.decoding.prior_rescoring import Prior
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob
from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
from i6_experiments.users.zeyer.decoding.rescoring import rescore, SharedPostConfig, _returnn_score_step
from i6_experiments.users.zeyer.decoding.lm_rescoring import prior_score
from i6_experiments.users.zeyer import tools_paths

def ctc_recog_framewise_prior_auto_scale(
    *,
    prefix: str,
    task: Task,
    ctc_model: ModelWithCheckpoint,
    prior_file: tk.Path,
    lm: ModelWithCheckpoint | tk.Path,
    vocab_file: tk.Path,
    vocab_opts_file: tk.Path,
    vocab_w_blank_file: tk.Path,
    num_shards: int,
    search_config: dict,
    recomb_config: dict | None = None,
) -> tuple[Variable, Variable]:
    """
    Recog with ``model_recog_ctc_only`` to get N-best list on ``task.dev_dataset``,
    then calc scores with framewise prior and LM on N-best list,
    then tune optimal scales on N-best list,
    then rescore on all ``task.eval_datasets`` using those scales,
    and also do first-pass recog (``model_recog``) with those scales.
    """
    dataset = task.dev_dataset
    if recomb_config is not None:
        mr = model_recog_recomb
    else:
        mr = model_recog_ctc_only
    _, asr_scores, _ = search_dataset(
        decoder_hyperparameters={} if recomb_config is None else recomb_config,
        dataset=dataset,
        model=ctc_model,
        recog_def=mr,
        prior_path=None,
        config=search_config,
        num_shards=num_shards,
        pseudo_label_alignment=True,
    )
    asr_scores = RecogOutput(output=asr_scores)
    
    framewise_prior = Prior(file=prior_file, type="log_prob", vocab=vocab_w_blank_file)
    prior_scores = prior_score(asr_scores, prior=framewise_prior)
    if model_recog_ctc_only.output_blank_label:
        asr_scores = ctc_alignment_to_label_seq(asr_scores, blank_label=model_recog_ctc_only.output_blank_label)
        prior_scores = ctc_alignment_to_label_seq(prior_scores, blank_label=model_recog_ctc_only.output_blank_label)
    if isinstance(lm, ModelWithCheckpoint):
        lm_scores = rescore(
            recog_output=asr_scores,
            model=lm,
            vocab=vocab_file,
            vocab_opts_file=vocab_opts_file,
            rescore_def=lm_rescore_def,
            forward_device="cpu",
        )
    else:
        assert isinstance(lm, tk.Path)
        lm_scores = ngram_rescore(
            recog_output=asr_scores,
            model=lm,
            vocab=vocab_file,
            vocab_opts_file=vocab_opts_file,
            rescore_def=ngram_lm_rescore_def,
            config={"backend": ctc_model.definition.backend, "behavior_version": ctc_model.definition.behavior_version},
            forward_device="cpu",
        )

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )

    for f in task.recog_post_proc_funcs:  # BPE to words or so
        asr_scores = f(asr_scores)
        prior_scores = f(prior_scores)
        lm_scores = f(lm_scores)
        ref = f(ref)
        
    asr_scores = SearchOutputRawReplaceJob(asr_scores.output, [("@@", "")], output_gzip=True).out_search_results
    prior_scores = SearchOutputRawReplaceJob(prior_scores.output, [("@@", "")], output_gzip=True).out_search_results
    lm_scores = SearchOutputRawReplaceJob(lm_scores.output, [("@@", "")], output_gzip=True).out_search_results
    ref = SearchOutputRawReplaceJob(ref.output, [("@@", "")], output_gzip=True).out_search_results

    opt_scales_job = ScaleTuningJob(
        scores={"am": asr_scores, "prior": prior_scores, "lm": lm_scores},
        ref=ref,
        fixed_scales={"am": 1.0},
        negative_scales={"prior"},
        scale_relative_to={"prior": "lm"},
        evaluation="edit_distance",
    )
    opt_scales_job.add_alias(prefix)
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    # We use the real scales.
    prior_scale = opt_scales_job.out_real_scale_per_name["prior"] * (-1)
    lm_scale = opt_scales_job.out_real_scale_per_name["lm"]

    return prior_scale, lm_scale

def ctc_recog_labelwise_prior_auto_scale(
    *,
    prefix: str,
    task: Task,
    ctc_model: ModelWithCheckpoint,
    prior_file: tk.Path,
    lm: ModelWithCheckpoint | tk.Path,
    vocab_file: tk.Path,
    vocab_opts_file: tk.Path,
    num_shards: int,
    search_config: dict,
    recomb_config: dict | None = None,
) -> tuple[Variable, Variable]:
    """
    Recog with ``model_recog_ctc_only`` to get N-best list on ``task.dev_dataset``,
    then calc scores with labelwise prior and LM on N-best list,
    then tune optimal scales on N-best list,
    then rescore on all ``task.eval_datasets`` using those scales,
    and also do first-pass recog (``model_recog``) with those scales.
    """
    dataset = task.dev_dataset
    if recomb_config is not None:
        mr = model_recog_recomb
    else:
        mr = model_recog_ctc_only
    _, asr_scores, _ = search_dataset(
        decoder_hyperparameters={} if recomb_config is None else recomb_config,
        dataset=dataset,
        model=ctc_model,
        recog_def=mr,
        prior_path=None,
        config=search_config,
        num_shards=num_shards,
    )
    asr_scores = RecogOutput(output=asr_scores)
    
    labelwise_prior = Prior(file=prior_file, type="log_prob", vocab=vocab_file)
    prior_scores = prior_score(asr_scores, prior=labelwise_prior)
    if isinstance(lm, ModelWithCheckpoint):
        lm_scores = rescore(
            recog_output=asr_scores,
            model=lm,
            vocab=vocab_file,
            vocab_opts_file=vocab_opts_file,
            rescore_def=lm_rescore_def,
            forward_device="cpu",
        )
    else:
        assert isinstance(lm, tk.Path)
        lm_scores = ngram_rescore(
            recog_output=asr_scores,
            model=lm,
            vocab=vocab_file,
            vocab_opts_file=vocab_opts_file,
            rescore_def=ngram_lm_rescore_def,
            config={"backend": ctc_model.definition.backend, "behavior_version": ctc_model.definition.behavior_version},
            forward_device="cpu",
        )

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )

    for f in task.recog_post_proc_funcs:  # BPE to words or so
        asr_scores = f(asr_scores)
        prior_scores = f(prior_scores)
        lm_scores = f(lm_scores)
        ref = f(ref)
        
    asr_scores = SearchOutputRawReplaceJob(asr_scores.output, [("@@", "")], output_gzip=True).out_search_results
    prior_scores = SearchOutputRawReplaceJob(prior_scores.output, [("@@", "")], output_gzip=True).out_search_results
    lm_scores = SearchOutputRawReplaceJob(lm_scores.output, [("@@", "")], output_gzip=True).out_search_results
    ref = SearchOutputRawReplaceJob(ref.output, [("@@", "")], output_gzip=True).out_search_results

    opt_scales_job = ScaleTuningJob(
        scores={"am": asr_scores, "prior": prior_scores, "lm": lm_scores},
        ref=ref,
        fixed_scales={"am": 1.0},
        negative_scales={"prior"},
        scale_relative_to={"prior": "lm"},
        evaluation="edit_distance",
    )
    opt_scales_job.add_alias(prefix)
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    # We use the real scales.
    prior_scale = opt_scales_job.out_real_scale_per_name["prior"] * (-1)
    lm_scale = opt_scales_job.out_real_scale_per_name["lm"]

    return prior_scale, lm_scale

def lm_rescore_def(*, model: rf.Module, targets: Tensor, targets_beam_dim: Dim, targets_spatial_dim: Dim, **_other):
    import returnn.frontend as rf

    targets_beam_dim  # noqa  # unused here

    # noinspection PyTypeChecker
    model: FeedForwardLm
    vocab = model.vocab_dim.vocab
    assert vocab.bos_label_id is not None and vocab.eos_label_id is not None

    targets_w_eos, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=vocab.eos_label_id
    )

    batch_dims = targets.remaining_dims(targets_spatial_dim)
    lm_state = model.default_initial_state(batch_dims=batch_dims)
    logits, _ = model(
        targets,
        spatial_dim=targets_spatial_dim,
        out_spatial_dim=targets_w_eos_spatial_dim,
        state=lm_state,
    )
    assert logits.dims == (*batch_dims, targets_w_eos_spatial_dim, model.vocab_dim)
    log_prob = rf.log_softmax(logits, axis=model.vocab_dim)
    log_prob_targets = rf.gather(
        log_prob, indices=targets_w_eos, axis=model.vocab_dim
    )  # [batch,beam,targets_spatial_w_eos]
    log_prob_targets_seq = rf.reduce_sum(log_prob_targets, axis=targets_w_eos_spatial_dim)  # [batch,beam]
    assert log_prob_targets_seq.dims_set == set(batch_dims)
    return log_prob_targets_seq

lm_rescore_def: RescoreDef

def ngram_lm_rescore_def(*, model: torch.Module, targets: Tensor, targets_beam_dim: Dim, targets_spatial_dim: Dim, **_other):
    import returnn.frontend as rf

    targets_beam_dim  # noqa  # unused here

    batch_dims = targets.remaining_dims(targets_spatial_dim)
    
    log_prob_targets_seq = model(
        targets.raw_tensor,
        targets_spatial_dim.get_size_tensor().raw_tensor,
        targets.vocab
    )
    assert log_prob_targets_seq.ndim == 2 and log_prob_targets_seq.size(0) == targets.raw_tensor.size(0) and log_prob_targets_seq.size(1) == targets.raw_tensor.size(1)
    log_prob_targets_seq = rf.convert_to_tensor(log_prob_targets_seq, dims=batch_dims, device=targets.device, dtype="float32")  # [batch,beam]
    assert log_prob_targets_seq.dims_set == set(batch_dims)
    return log_prob_targets_seq

ngram_lm_rescore_def: RescoreDef

# ------------------------------------------------

def ngram_rescore(
    *,
    recog_output: RecogOutput,
    vocab: tk.Path,
    vocab_opts_file: Optional[tk.Path] = None,
    model: tk.Path,
    rescore_def: RescoreDef,
    config: Optional[Dict[str, Any]] = None,
    forward_post_config: Optional[Dict[str, Any]] = None,
    forward_rqmt: Optional[Dict[str, Any]] = None,
    forward_device: str = "gpu",
    forward_alias_name: Optional[str] = None,
) -> RecogOutput:
    env_updates = None
    if (config and config.get("__env_updates")) or (forward_post_config and forward_post_config.get("__env_updates")):
        env_updates = (config and config.pop("__env_updates", None)) or (
            forward_post_config and forward_post_config.pop("__env_updates", None)
        )
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=None,
        returnn_config=_returnn_ngram_rescore_config(
            recog_output=recog_output,
            vocab=vocab,
            vocab_opts_file=vocab_opts_file,
            model=model,
            rescore_def=rescore_def,
            config=config,
            post_config=forward_post_config,
            device=forward_device
        ),
        output_files=[_v2_forward_out_filename],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        device=forward_device,
    )
    forward_job.rqmt["mem"] = 32  # often needs more mem
    if forward_rqmt:
        forward_job.rqmt.update(forward_rqmt)
    if env_updates:
        for k, v in env_updates.items():
            forward_job.set_env(k, v)
    if forward_alias_name:
        forward_job.add_alias(forward_alias_name)
    return RecogOutput(output=forward_job.out_files[_v2_forward_out_filename])


def _returnn_ngram_rescore_config(
    *,
    recog_output: RecogOutput,
    vocab: tk.Path,
    vocab_opts_file: Optional[tk.Path] = None,
    model: tk.Path,
    rescore_def: RescoreDef,
    config: Optional[Dict[str, Any]] = None,
    post_config: Optional[Dict[str, Any]] = None,
    device: str
) -> ReturnnConfig:
    """
    Create config for rescoring.
    """
    from returnn.tensor import Tensor, Dim, batch_dim
    from i6_experiments.users.zeyer.serialization_v2 import ReturnnConfigWithNewSerialization
    from i6_experiments.users.zeyer.returnn.config import config_dict_update_

    config_ = config
    config = {}

    # Note: we should not put SPM/BPE directly here,
    # because the recog output still has individual labels,
    # so no SPM/BPE encoding on the text.
    vocab_opts = {"class": "Vocabulary", "vocab_file": vocab}
    if vocab_opts_file:
        vocab_opts["special_symbols_via_file"] = vocab_opts_file
    else:
        vocab_opts["unknown_label"] = None

    # Beam dim size unknown. Usually static size, but it's ok to leave this unknown here (right?).
    beam_dim = Dim(Tensor("beam_size", dims=[], dtype="int32"), name="beam")

    data_flat_spatial_dim = Dim(None, name="data_flat_spatial")

    forward_data = {"class": "TextDictDataset", "filename": recog_output.output, "vocab": vocab_opts}
    extern_data = {
        # data_flat dyn dim is the flattened dim, no need to define dim tags now
        "data_flat": {"dims": [batch_dim, data_flat_spatial_dim], "dtype": "int32", "vocab": vocab_opts},
        "data_seq_lens": {"dims": [batch_dim, beam_dim], "dtype": "int32"},
    }

    config.update(
        {
            "forward_data": forward_data,
            "default_input": None,
            "target": "data_flat",  # needed for get_model to know the target dim
            "_beam_dim": beam_dim,
            "_data_flat_spatial_dim": data_flat_spatial_dim,
            "extern_data": extern_data,
        }
    )
    
    assert "backend" in config_
    assert "behavior_version" in config_
    
    config["_model_path"] = model
    config["_dev"] = device
    config["get_model"] = _get_ngram_model
    config["_rescore_def"] = rescore_def
    config["forward_step"] = _returnn_score_step
    config["forward_callback"] = _returnn_v2_get_forward_callback
    
    config["_version"] = 2

    if config_:
        config_dict_update_(config, config_)

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
        batch_size=(20_000 * 160),
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

def _get_ngram_model(*, epoch: int, **_kwargs_unused):
    from returnn.tensor import Tensor
    from returnn.config import get_global_config
    from i6_core.util import uopen
    import kenlm

    config = get_global_config()
    model_path = config.typed_value("_model_path")
    
    lm = kenlm.Model(model_path)
    
    # with uopen(model_path, "rb") as f:
    #     lm = torch.load(f, map_location=config.typed_value("_dev"))
    #     assert isinstance(lm, torch.Tensor), "Loaded LM is not a tensor"
    # lm = torch.log_softmax(lm, dim=-1) # NOTE do not use
    
    class Model(torch.nn.Module):
        def __init__(self, lm: torch.Tensor):
            super().__init__()
            self.lm = lm

        # def forward(self, x: torch.Tensor, length: torch.Tensor):
        #     assert x.ndim == 3
        #     ndim = self.lm.ndim
        #     context_size = ndim - 1
        #     x = torch.nn.functional.pad(x, (context_size, 1), value=0)
        #     scores = torch.zeros((x.size(0), x.size(1)), dtype=torch.float32, device=x.device)
        #     for t in range(length.max() + 1):
        #         indices = []
        #         for i in range(ndim):
        #             indices.append(x[..., t + i])
        #         new_score = self.lm[*indices]
        #         new_score[new_score.isneginf()] = -1e30
        #         # assert logits.isnan().sum() == 0, f"Failed at {t} with {log_lm_probs.isnan().sum()}"
        #         # log_lm_probs = torch.log_softmax(logits, dim=-1)
        #         # assert log_lm_probs.isnan().sum() == 0, f"Failed at {t} 2 with {log_lm_probs.isnan().sum()}"
        #         # new_score = log_lm_probs.gather(-1, x[..., t + context_size].unsqueeze(-1).long()).squeeze(-1)
        #         scores = torch.where(
        #             t < length + 1,
        #             scores + new_score,
        #             scores,
        #         )
        #         if scores.isneginf().any():
        #             print(f"Failed at {t} with {scores.isneginf().sum()}, {scores}, {new_score}")
        #     assert scores.ndim == 2 and scores.size(0) == x.size(0) and scores.size(1) == x.size(1)
        #     return scores
        
        def forward(self, x: torch.Tensor, length: torch.Tensor, vocab: Vocabulary):
            assert x.ndim == 3
            scores = torch.zeros((x.size(0), x.size(1)), dtype=torch.float32, device=x.device)
            for i in range(x.size(0)):
                for j in range(x.size(1)):
                    sentence = x[i, j, :].tolist()
                    sentence = [vocab.id_to_label(c) for c in sentence]
                    sentence = " ".join(sentence).replace("@@ ", "")
                    if sentence.endswith("@@"):
                        sentence = sentence[:-2]
                    scores[i, j] = self.lm.score(sentence)
            assert scores.ndim == 2 and scores.size(0) == x.size(0) and scores.size(1) == x.size(1)
            return scores
        
    model = Model(lm)

    return model