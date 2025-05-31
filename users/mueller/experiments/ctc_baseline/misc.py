"""
helpers for training.

- Uses ``unhashed_package_root`` now, via :func:`i6_experiments.users.zeyer.utils.sis_setup.get_base_module`.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Dict, Any, Sequence
import torch
import numpy as np
import copy

from sisyphus import tk
from returnn.tensor import batch_dim
import returnn.frontend as rf

from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn import ReturnnConfig
from i6_core.util import instanciate_delayed

from i6_experiments.users.mueller.experiments.ctc_baseline.training import _LM_score, _prior_score
from i6_experiments.users.mueller.experiments.ctc_baseline.decoding import recog_flashlight_ngram, recog_ffnn
from i6_experiments.users.mueller.experiments.ctc_baseline.utils import convert_to_output_hyps
from i6_experiments.users.mueller.experiments.ctc_baseline.model import Model
from i6_experiments.users.mueller.experiments.language_models.n_gram import get_binary_lm
from i6_experiments.users.mueller.recog import _returnn_v2_get_model
from i6_experiments.users.mueller.utils import ReturnnConfigCustom
from i6_experiments.users.mueller.datasets.librispeech import LibrispeechOggZip, _raw_audio_opts, get_bpe_lexicon
from i6_experiments.users.zeyer.model_interfaces import ModelT, ModelDef, ModelDefWithCfg, TrainDef, serialize_model_def
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.utils.serialization import get_import_py_code
from i6_experiments.users.zeyer.nn_rf.torch_ctc_fixed_grad import ctc_loss_fixed_grad
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
from i6_experiments.users.zeyer import tools_paths
from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict

if TYPE_CHECKING:
    from returnn.tensor import TensorDict, Tensor, Dim
    from i6_experiments.common.setups import serialization
    from i6_experiments.users.zeyer.datasets.task import Task, DatasetConfig, VocabConfig
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, Checkpoint, ModelWithCheckpoint

# Scores for Max and GT -----------------------------------------------

def subset_scoring(
    *,
    model: ModelWithCheckpoint,
    lm_name: str,
    lm_checkpoint_path: Optional[tk.Path],
    vocab: VocabConfig,
    prior_file: tk.Path,
    forward_alias_name: str,
    forward_rqmt: Optional[Dict[str, Any]] = None,
    forward_device: str = "gpu",
):
    assert lm_name
    USE_CTC_LOSS = False
    USE_WORD_4GRAM = True
    LENGTH_NORM = False
    
    if USE_WORD_4GRAM:
        lm_name = "word4gram"
        arpa_file = get_binary_lm(get_arpa_lm_dict()["4gram"])
    else:
        assert lm_name.startswith("ffnn")
    
    config = {
        "prior_file": prior_file,
        "hyperparameters": {
            "lm_weight": 1.25,
            "prior_weight": 0.3,
            "lm_order": lm_name,
            "beam_size": 80,
        },
        "use_ctc_loss": USE_CTC_LOSS,
        "use_word_4gram": USE_WORD_4GRAM,
    }
    if USE_WORD_4GRAM:
        config["arpa_file"] = arpa_file
        config["hyperparameters"]["use_logsoftmax"] = True
        config["lexicon"] = get_bpe_lexicon(vocab)
    else:
        config["hyperparameters"]["use_recombination"] = True
        config["hyperparameters"]["recomb_blank"] = True
        config["hyperparameters"]["recomb_after_topk"] = True
        config["hyperparameters"]["recomb_with_sum"] = False
    if lm_checkpoint_path is not None:
        config["preload_from_files"] = {
            "recog_lm": {
                "prefix": "recog_language_model.",
                "filename": lm_checkpoint_path,
            },
        }
    if LENGTH_NORM:
        config["length_norm"] = True
        
    audio_opts_ = _raw_audio_opts.copy()
    dataset_common_opts = dict(audio=audio_opts_, audio_dim=1, vocab=vocab)
    dataset = LibrispeechOggZip(**dataset_common_opts, main_key="train-other-860", forward_subset=18_000)
    """
    Seq-length 'data' Stats:
        18000 seqs
        Mean: 196247.3239444452
        Std dev: 61465.12494455205
        Min/max: 14720 / 317040
    Seq-length 'classes' Stats:
        18000 seqs
        Mean: 86.22505555555539
        Std dev: 30.463069900696127
        Min/max: 2 / 191
    Seq-length 'orth' Stats:
        18000 seqs
        Mean: 176.2277777777769
        Std dev: 62.58316516227298
        Min/max: 3 / 391
    Seq-length 'raw' Stats:
        18000 seqs
        Mean: 1.0
        Std dev: 0.0
        Min/max: 1 / 1
    Data 'data' Stats:
        18000 seqs, 3532451831 total frames, 196247.323944 average frames
        Mean: [-3.24545363e-05]
        Std dev: [0.11606654]
        Min/max: [-1.] / [1.]
        
        ==> ca 61h
    """
    
    """
    FFNN:
    gt_score_am: 0.01599056667751736
    gt_score_lm: 7.356120982472526e-10
    gt_score_prior: 1.0682799089778428e-13
    gt_score_combined: 4.602839714312129e-15
    max_score_am: 0.019778676350911458
    max_score_lm: 1.1483641881366363e-09
    max_score_prior: 1.511515034103973e-08
    max_score_combined: 4.55754377138974e-15
    Log Space:
    gt_score_am: -4.135756492614746
    gt_score_lm: -21.030319213867188
    gt_score_prior: -29.867557525634766
    gt_score_combined: -33.01210403442383
    max_score_am: -3.9231510162353516
    max_score_lm: -20.58492660522461
    max_score_prior: -18.007568359375
    max_score_combined: -33.02199172973633

    4gram:
    gt_score_am: 0.01599056667751736
    gt_score_lm: 1.1164877220279676e-11
    gt_score_prior: 1.0682799089778428e-13
    gt_score_combined: 5.041833367544149e-31
    max_score_am: 0.021518756442599825
    max_score_lm: 3.277289629800685e-11
    max_score_prior: 3.1707826487882407e-13
    max_score_combined: 5.297055463852185e-31
    Log Space:
    gt_score_am: -4.135756492614746
    gt_score_lm: -25.21824836730957
    gt_score_prior: -29.867557525634766
    gt_score_combined: -69.76236724853516
    max_score_am: -3.8388304710388184
    max_score_lm: -24.14141845703125
    max_score_prior: -28.77962875366211
    max_score_combined: -69.7129898071289

    """
        
    output_file = "scores.txt"
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=model.checkpoint,
        returnn_config=_returnn_scoring_config(
            dataset=dataset,
            model_def=model.definition,
            config=config,
        ),
        output_files=[output_file],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        device=forward_device,
        cpu_rqmt=4,
        mem_rqmt=16,
        time_rqmt=16,
    )
    if forward_rqmt:
        forward_job.rqmt.update(forward_rqmt)
    assert forward_alias_name
    forward_job.add_alias(forward_alias_name)

    tk.register_output(forward_alias_name, forward_job.out_files[output_file])
    
SharedPostConfig = {
    # In case pretraining overwrites some of these, they need a default.
    "accum_grad_multiple_step": None,
    "use_last_best_model": None,
}
  
def _returnn_scoring_config(
    *,
    dataset: DatasetConfig,
    model_def: Union[ModelDef, ModelDefWithCfg],
    config: Optional[Dict[str, Any]] = None,
    post_config: Optional[Dict[str, Any]] = None
) -> ReturnnConfig:
    """
    Create config for scoring.
    """
    from i6_experiments.users.zeyer.returnn.config import config_dict_update_
    from i6_experiments.common.setups.returnn.serialization import get_serializable_config
    from i6_experiments.common.setups import serialization
    from returnn_common import nn
    
    assert dataset

    config_ = config
    config = {}

    extern_data = dataset.get_extern_data()
    extern_data = instanciate_delayed(extern_data)
    forward_data = dataset.get_main_dataset()

    config.update(
        {
            "forward_data": forward_data,
            "default_input": dataset.get_default_input(),
            "target": dataset.get_default_target(),
        }
    )
    
    if "backend" not in config:
        config["backend"] = model_def.backend
    config["behavior_version"] = max(model_def.behavior_version, config.get("behavior_version", 0))
    
    if config_:
        config_dict_update_(config, config_)
    if isinstance(model_def, ModelDefWithCfg):
        config.update(model_def.config)

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
        
    returnn_config = ReturnnConfigCustom(
        config=config,
        python_prolog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(get_import_py_code()),
                    serialization.PythonCacheManagerFunctionNonhashedCode,
                ]
            )
        ],
        python_epilog=[
            serialization.Collection(
                [
                    serialization.NonhashedCode(
                        nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data)
                    ),
                    *serialize_model_def(model_def),
                    serialization.Import(_returnn_v2_get_model, import_as="get_model"),
                    serialization.Import(score_max_and_gt, import_as="_score_def"),
                    serialization.Import(_returnn_score_step, import_as="forward_step"),
                    serialization.Import(_returnn_get_forward_callback, import_as="forward_callback"),
                    serialization.PythonEnlargeStackWorkaroundNonhashedCode,
                    serialization.PythonModelineNonhashedCode,
                ]
            )
        ],
        post_config=post_config,
        sort_config=False,
    )
    
    returnn_config = get_serializable_config(
        returnn_config,
        serialize_dim_tags=False,
    )

    return returnn_config

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

    default_target_key = config.typed_value("target")
    targets = extern_data[default_target_key]
    targets_spatial_dim = targets.get_time_dim_tag()

    score_def = config.typed_value("_score_def")
    scores_dict = score_def(
        model=model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        targets=targets,
        targets_spatial_dim=targets_spatial_dim,
        hyperparameters=config.typed_value("hyperparameters"),
        prior_file=config.typed_value("prior_file"),
        use_ctc_loss=config.typed_value("use_ctc_loss"),
        use_word_4gram=config.typed_value("use_word_4gram"),
        arpa_file=config.typed_value("arpa_file", default=None),
        lexicon=config.typed_value("lexicon", default=None),
        length_norm=config.typed_value("length_norm", default=False),
    )
    assert isinstance(scores_dict, dict)
    rf.get_run_ctx().mark_as_output(scores_dict["gt_score"]["am"], "gt_am", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["gt_score"]["lm"], "gt_lm", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["gt_score"]["prior"], "gt_prior", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["gt_score"]["combined"], "gt_comb", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["max_score"]["am"], "max_am", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["max_score"]["lm"], "max_lm", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["max_score"]["prior"], "max_prior", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["max_score"]["combined"], "max_comb", dims=[batch_dim])
    
def _returnn_get_forward_callback():
    from typing import TextIO
    from returnn.forward_iface import ForwardCallbackIface

    class _ReturnnScoringForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.out_file: Optional[TextIO] = None

        def init(self, *, model):
            self.out_file = open("scores.txt", "wt", encoding="utf-8")
            init_prob = np.array([float("-inf")], dtype=np.float32)
            self.scores_sum = {
                "gt_score": {
                    "am": init_prob,
                    "lm": init_prob,
                    "prior": init_prob,
                    "combined": init_prob,
                },
                "max_score": {
                    "am": init_prob,
                    "lm": init_prob,
                    "prior": init_prob,
                    "combined": init_prob,
                },
            }
            self.n_seq = 0

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            self.n_seq += 1
            
            self.scores_sum["gt_score"]["am"] = np.logaddexp(self.scores_sum["gt_score"]["am"], outputs["gt_am"].raw_tensor)
            self.scores_sum["gt_score"]["lm"] = np.logaddexp(self.scores_sum["gt_score"]["lm"], outputs["gt_lm"].raw_tensor)
            self.scores_sum["gt_score"]["prior"] = np.logaddexp(self.scores_sum["gt_score"]["prior"], outputs["gt_prior"].raw_tensor)
            self.scores_sum["gt_score"]["combined"] = np.logaddexp(self.scores_sum["gt_score"]["combined"], outputs["gt_comb"].raw_tensor)
            self.scores_sum["max_score"]["am"] = np.logaddexp(self.scores_sum["max_score"]["am"], outputs["max_am"].raw_tensor)
            self.scores_sum["max_score"]["lm"] = np.logaddexp(self.scores_sum["max_score"]["lm"], outputs["max_lm"].raw_tensor)
            self.scores_sum["max_score"]["prior"] = np.logaddexp(self.scores_sum["max_score"]["prior"], outputs["max_prior"].raw_tensor)
            self.scores_sum["max_score"]["combined"] = np.logaddexp(self.scores_sum["max_score"]["combined"], outputs["max_comb"].raw_tensor)

        def finish(self):
            for key, scores in self.scores_sum.items():
                for score_name, score_value in scores.items():
                    assert isinstance(score_value, np.ndarray)
                    self.out_file.write(f"{key}_{score_name}: {np.exp(score_value).item() / self.n_seq}\n")
            self.out_file.write("Log Space:\n")
            for key, scores in self.scores_sum.items():
                for score_name, score_value in scores.items():
                    assert isinstance(score_value, np.ndarray)
                    self.out_file.write(f"{key}_{score_name}: {(score_value - np.log(self.n_seq)).item()}\n")
            self.out_file.close()

    return _ReturnnScoringForwardCallbackIface()


def score_max_and_gt(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    targets: Tensor,
    targets_spatial_dim: Dim,
    hyperparameters: dict,
    prior_file: tk.Path,
    use_ctc_loss: bool,
    use_word_4gram: bool,
    arpa_file: Optional[tk.Path],
    lexicon: Optional[tk.Path],
    length_norm: bool,
):
    import torchaudio
    
    if use_word_4gram:
        assert lexicon and arpa_file
    
    # Get log_probs
    assert data.dims_set == {batch_dim, data_spatial_dim, data.feature_dim}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    assert logits.dims_set == {batch_dim, enc_spatial_dim, model.wb_target_dim}
    
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    log_probs = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    
    # Score ground truth
    if use_ctc_loss:
        ctc_loss_gt = ctc_loss_fixed_grad(
            logits=log_probs,
            logits_normalized=True,
            targets=targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=targets_spatial_dim,
            blank_index=model.blank_idx,
        )
        am_gt_score = -ctc_loss_gt
        # NOTE: using label prior here
        prior_gt_score = _prior_score(
            targets=targets,
            targets_spatial_dim=targets_spatial_dim,
            model=model,
            prior_file=prior_file,
            force_label_prior=True,
        )
    else:
        forced_targets_align = None
        am_gt_score = None
        max_enc_len = int(enc_spatial_dim.get_dim_value())
        for b in range(log_probs.raw_tensor.shape[0]):
            target_len = targets_spatial_dim.dyn_size_ext.raw_tensor[b]
            enc_len = enc_spatial_dim.dyn_size_ext.raw_tensor[b]
            if target_len == 0:
                am_gt_score_b = log_probs.raw_tensor[b, :enc_len, model.blank_idx].unsqueeze(0)
                forced_align_b = torch.full((1, enc_len), model.blank_idx, dtype=torch.int32, device=log_probs.raw_tensor.device)
            else:
                forced_align_b, am_gt_score_b = torchaudio.functional.forced_align(
                    log_probs=log_probs.raw_tensor[b, :enc_len].unsqueeze(0),
                    targets=targets.raw_tensor[b, :target_len].unsqueeze(0),
                    input_lengths=enc_len.unsqueeze(0),
                    target_lengths=target_len.unsqueeze(0),
                    blank=model.blank_idx,
                )
            forced_align_b = torch.nn.functional.pad(forced_align_b, (0, max_enc_len - forced_align_b.shape[1]), value=model.eos_idx)
            if am_gt_score is None:
                am_gt_score = am_gt_score_b.sum(dim=-1) # sum over time
                forced_targets_align = forced_align_b
            else:
                am_gt_score = torch.cat((am_gt_score, am_gt_score_b.sum(dim=-1)), dim=0)
                forced_targets_align = torch.cat((forced_targets_align, forced_align_b), dim=0)
        am_gt_score = rf.convert_to_tensor(am_gt_score, dims=(batch_dim,), dtype="float32", name="am_gt_score")
        forced_targets_align = rf.convert_to_tensor(forced_targets_align, dims=[batch_dim, enc_spatial_dim], dtype="int32", name="forced_targets_align")
        # NOTE: using frame prior here
        prior_gt_score = _prior_score(
            targets=forced_targets_align,
            targets_spatial_dim=enc_spatial_dim,
            model=model,
            prior_file=prior_file,
            force_label_prior=False,
        )
    lm_gt_score = _LM_score(targets, targets_spatial_dim, model, hyperparameters.get("lm_order", None), train_lm=False, arpa_file=arpa_file)
    
    lm_scale = hyperparameters["lm_weight"]
    prior_scale = hyperparameters["prior_weight"]
    
    combined_gt_score = am_gt_score + lm_scale * lm_gt_score + prior_scale * prior_gt_score
    
    # Extract max score
    if use_word_4gram:
        max_hyp, _, _, beam_dim = recog_flashlight_ngram(model=model, data=data, data_spatial_dim=data_spatial_dim, arpa_4gram_lm = arpa_file, lexicon=lexicon, hyperparameters=hyperparameters, prior_file=prior_file)
        assert beam_dim.dimension == 1, "Beam dimension should be 1 for max score extraction"
        max_hyp = max_hyp.raw_tensor
        max_hyp = max_hyp.squeeze(axis=1)
        max_hyp = max_hyp.tolist()
        max_hyp = [model.target_dim.vocab.get_seq(seq[0]) for seq in max_hyp]
    else:
        max_hyp, _, _, beam_dim = recog_ffnn(model=model, label_log_prob=log_probs, enc_spatial_dim=enc_spatial_dim, hyperparameters=hyperparameters, batch_dims=batch_dims, prior_file=prior_file, train_lm=False)
        assert beam_dim.dimension == 1, "Beam dimension should be 1 for max score extraction"
        max_hyp = rf.squeeze(max_hyp, axis=beam_dim)
        max_hyp = [convert_to_output_hyps(model, h) for h in max_hyp.raw_tensor.transpose(0, 1).tolist()]
    lengths = [len(h) for h in max_hyp]
    max_hyp_spatial_dim = torch.tensor(lengths, dtype=torch.int32, device=data.raw_tensor.device)
    max_length = max_hyp_spatial_dim.max().item()
    max_hyp_spatial_dim = rf.convert_to_tensor(max_hyp_spatial_dim, dims=(batch_dim,))
    max_hyp_spatial_dim = rf.Dim(max_hyp_spatial_dim, name="out_spatial", dyn_size_ext=max_hyp_spatial_dim)
    max_hyp = [h + [model.eos_idx] * (max_length - len(h)) for h in max_hyp]
    max_hyp = torch.tensor(max_hyp, dtype=torch.int32, device=data.raw_tensor.device)
    max_hyp = rf.convert_to_tensor(max_hyp, dims=[batch_dim, max_hyp_spatial_dim], sparse_dim=model.target_dim, dtype="int32", name="max_hyp")
    
    # Score max
    if use_ctc_loss:
        ctc_loss_max = ctc_loss_fixed_grad(
            logits=log_probs,
            logits_normalized=True,
            targets=max_hyp,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=max_hyp_spatial_dim,
            blank_index=model.blank_idx,
        )
        am_max_score = -ctc_loss_max
        prior_max_score = _prior_score(
            targets=max_hyp,
            targets_spatial_dim=max_hyp_spatial_dim,
            model=model,
            prior_file=prior_file,
            force_label_prior=True,
        )
    else:
        forced_max_align = None
        am_max_score = None
        max_enc_len = int(enc_spatial_dim.get_dim_value())
        for b in range(log_probs.raw_tensor.shape[0]):
            max_hyp_len = max_hyp_spatial_dim.dyn_size_ext.raw_tensor[b]
            enc_len = enc_spatial_dim.dyn_size_ext.raw_tensor[b]
            if max_hyp_len == 0:
                am_max_score_b = log_probs.raw_tensor[b, :enc_len, model.blank_idx].unsqueeze(0)
                forced_align_b = torch.full((1, enc_len), model.blank_idx, dtype=torch.int32, device=log_probs.raw_tensor.device)
            else:
                forced_align_b, am_max_score_b = torchaudio.functional.forced_align(
                    log_probs=log_probs.raw_tensor[b, :enc_len].unsqueeze(0),
                    targets=max_hyp.raw_tensor[b, :max_hyp_len].unsqueeze(0),
                    input_lengths=enc_len.unsqueeze(0),
                    target_lengths=max_hyp_len.unsqueeze(0),
                    blank=model.blank_idx,
                )
            forced_align_b = torch.nn.functional.pad(forced_align_b, (0, max_enc_len - forced_align_b.shape[1]), value=model.eos_idx)
            if am_max_score is None:
                am_max_score = am_max_score_b.sum(dim=-1) # sum over time
                forced_max_align = forced_align_b
            else:
                am_max_score = torch.cat((am_max_score, am_max_score_b.sum(dim=-1)), dim=0)
                forced_max_align = torch.cat((forced_max_align, forced_align_b), dim=0)
        am_max_score = rf.convert_to_tensor(am_max_score, dims=(batch_dim,), dtype="float32", name="am_max_score")
        forced_max_align = rf.convert_to_tensor(forced_max_align, dims=[batch_dim, enc_spatial_dim], dtype="int32", name="forced_max_align")
        prior_max_score = _prior_score(
            targets=forced_max_align,
            targets_spatial_dim=enc_spatial_dim,
            model=model,
            prior_file=prior_file,
            force_label_prior=False,
        )
    lm_max_score = _LM_score(max_hyp, max_hyp_spatial_dim, model, hyperparameters.get("lm_order", None), train_lm=False, arpa_file=arpa_file)
    
    combined_max_score = am_max_score + lm_scale * lm_max_score + prior_scale * prior_max_score
    
    if length_norm:
        norm = rf.log(enc_spatial_dim.dyn_size_ext)
        norm = rf.copy_to_device(norm, data.device)
        ret = {
            "gt_score": {
                "am": am_gt_score - norm,
                "lm": lm_gt_score - norm,
                "prior": prior_gt_score - norm,
                "combined": combined_gt_score - norm,
            },
            "max_score": {
                "am": am_max_score - norm,
                "lm": lm_max_score - norm,
                "prior": prior_max_score - norm,
                "combined": combined_max_score - norm,
            },
        }
    else:
        ret = {
            "gt_score": {
                "am": am_gt_score,
                "lm": lm_gt_score,
                "prior": prior_gt_score,
                "combined": combined_gt_score,
            },
            "max_score": {
                "am": am_max_score,
                "lm": lm_max_score,
                "prior": prior_max_score,
                "combined": combined_max_score,
            },
        }
    
    return ret
    

# Histogram for N-Best scores------------------------------------------

# def histogram_forward(
#     prefix_name: str,
#     *,
#     task: Optional[Task] = None,
#     train_dataset: Optional[DatasetConfig] = None,
#     train_epoch_split: Optional[int] = None,
#     config: Dict[str, Any],
#     post_config: Optional[Dict[str, Any]] = None,
#     env_updates: Optional[Dict[str, str]] = None,
#     epilog: Sequence[serialization.SerializerObject] = (),
#     model_def: Union[ModelDefWithCfg, ModelDef[ModelT]],
#     train_def: TrainDef[ModelT],
#     init_params: Optional[Checkpoint] = None,
#     reset_steps: bool = True,
#     finish_all: bool = False,
#     extra_hash: Any = None,
#     gpu_mem: Optional[int] = None,
#     num_processes: Optional[int] = None,
#     init_hdf_writer: bool = False,
#     keep_train_lm_def: bool = True,
#     **kwargs,
# ) -> ModelWithCheckpoints:
#     from sisyphus import tk
#     from i6_core.util import instanciate_delayed
#     from i6_core.returnn.training import ReturnnTrainingJob
#     from i6_core.returnn.config import ReturnnConfig
#     from i6_experiments.common.setups import serialization
#     from i6_experiments.common.setups.returnn.serialization import get_serializable_config
#     from i6_experiments.users.zeyer.utils.serialization import get_import_py_code
#     from i6_experiments.users.zeyer.datasets.utils import multi_proc as mp_ds_utils
#     from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints
#     from i6_experiments.users.zeyer.recog import SharedPostConfig
#     from i6_experiments.users.zeyer.utils.sis_setup import get_base_module
#     from returnn_common import nn

#     unhashed_package_root_train_def, setup_base_name_train_def = get_base_module(train_def)
#     unhashed_package_root_model_def, setup_base_name_model_def = get_base_module(
#         model_def.model_def if isinstance(model_def, ModelDefWithCfg) else model_def
#     )

#     if train_dataset is None:
#         assert task
#         train_dataset = task.train_dataset
#     train_dataset_dict = train_dataset.get_train_dataset()
#     if train_epoch_split is None:
#         if task:
#             train_epoch_split = task.train_epoch_split
#         elif "partition_epoch" in train_dataset_dict:
#             train_epoch_split = train_dataset_dict["partition_epoch"]
#     # Usually always apply MultiProcDataset. But some exceptions for now:
#     apply_multi_proc = train_dataset_dict["class"] != "LmDataset"
#     del train_dataset_dict
#     del task

#     config = config.copy()
#     kwargs = kwargs.copy()
#     if "__num_epochs" in config:
#         kwargs["num_epochs"] = config.pop("__num_epochs")
#     if "__gpu_mem" in config:
#         gpu_mem = config.pop("__gpu_mem")
#     if "__num_processes" in config:
#         num_processes = config.pop("__num_processes")
#     if "__mem_rqmt" in config:
#         kwargs["mem_rqmt"] = config.pop("__mem_rqmt")
#     if "__cpu_rqmt" in config:
#         kwargs["cpu_rqmt"] = config.pop("__cpu_rqmt")
#     if not kwargs.get("distributed_launch_cmd"):
#         kwargs["distributed_launch_cmd"] = "torchrun" if num_processes else "mpirun"
#     if "__train_audio_preprocess" in config:
#         train_dataset = copy.copy(train_dataset)
#         assert hasattr(train_dataset, "train_audio_preprocess")
#         train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")
#     multi_proc_opts = (post_config.pop("__multi_proc_dataset_opts", None) or {}) if post_config else {}

#     returnn_train_config_dict: Dict[str, Any] = dict(
#         backend=model_def.backend,
#         behavior_version=model_def.behavior_version,
#         # dataset
#         default_input=train_dataset.get_default_input(),
#         target=train_dataset.get_default_target(),
#         train=(
#             mp_ds_utils.multi_proc_dataset_opts(train_dataset.get_train_dataset(), **multi_proc_opts)
#             if apply_multi_proc
#             else train_dataset.get_train_dataset()
#         ),
#         eval_datasets=(
#             mp_ds_utils.multi_proc_eval_datasets_opts(train_dataset.get_eval_datasets(), **multi_proc_opts)
#             if apply_multi_proc
#             else train_dataset.get_eval_datasets()
#         ),
#         learning_rate_control_error_measure=train_def.learning_rate_control_error_measure,
#         newbob_multi_num_epochs=train_epoch_split or 1,
#     )
#     returnn_train_config_dict = dict_update_deep(returnn_train_config_dict, config)
#     if isinstance(model_def, ModelDefWithCfg):
#         model_conf = model_def.config.copy()
#         model_conf.pop("recog_language_model", None)
#         if not keep_train_lm_def:
#             model_conf.pop("train_language_model", None)
#         returnn_train_config_dict = dict_update_deep(returnn_train_config_dict, model_conf)

#     max_seq_length_default_target = returnn_train_config_dict.pop("max_seq_length_default_target", None)
#     if max_seq_length_default_target is not None:
#         max_seq_length = returnn_train_config_dict.setdefault("max_seq_length", {})
#         assert isinstance(max_seq_length, dict)
#         max_seq_length[train_dataset.get_default_target()] = max_seq_length_default_target
#     max_seq_length_default_input = returnn_train_config_dict.pop("max_seq_length_default_input", None)
#     if max_seq_length_default_input is not None:
#         max_seq_length = returnn_train_config_dict.setdefault("max_seq_length", {})
#         assert isinstance(max_seq_length, dict)
#         max_seq_length[train_dataset.get_default_input()] = max_seq_length_default_input

#     if init_params:
#         returnn_train_config_dict["import_model_train_epoch1"] = init_params
#         if not reset_steps:
#             returnn_train_config_dict["reset_steps"] = False
#     if finish_all:
#         returnn_train_config_dict["_horovod_finish_all"] = True

#     extern_data_raw = train_dataset.get_extern_data()
#     # The extern_data is anyway not hashed, so we can also instanciate any delayed objects here.
#     # It's not hashed because we assume that all aspects of the dataset are already covered
#     # by the datasets itself as part in the config above.
#     extern_data_raw = instanciate_delayed(extern_data_raw)

#     returnn_train_config = ReturnnConfig(
#         returnn_train_config_dict,
#         python_epilog=[
#             serialization.Collection(
#                 [
#                     serialization.NonhashedCode(get_import_py_code()),
#                     serialization.NonhashedCode(
#                         nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
#                     ),
#                     *serialize_model_def(model_def, unhashed_package_root=unhashed_package_root_model_def),
#                     serialization.Import(
#                         train_def, import_as="_train_def", unhashed_package_root=unhashed_package_root_train_def
#                     ),
#                     # Consider the imports as non-hashed. We handle any logic changes via the explicit hash below.
#                     serialization.Import(_returnn_v2_get_model, import_as="get_model", use_for_hash=False),
#                     serialization.Import(_returnn_v2_train_step, import_as="train_step", use_for_hash=False),
#                 ]
#                 + ([
#                     serialization.Import(_returnn_epoch_start_hdf_writer, import_as="epoch_start", use_for_hash=False),
#                     serialization.Import(_returnn_epoch_end_hdf_writer, import_as="epoch_end", use_for_hash=False),
#                 ] if init_hdf_writer else [])
#                 + [
#                     serialization.ExplicitHash(
#                         {
#                             # Increase the version whenever some incompatible change is made in this train() function,
#                             # which influences the outcome, but would otherwise not influence the hash.
#                             "version": 3,
#                             # Whatever the caller provides. This could also include another version,
#                             # but this is up to the caller.
#                             "extra": extra_hash,
#                             **(
#                                 {"setup_base_name": setup_base_name_train_def}
#                                 if setup_base_name_train_def == setup_base_name_model_def
#                                 else {
#                                     "setup_base_name_train_def": setup_base_name_train_def,
#                                     "setup_base_name_model_def": setup_base_name_model_def,
#                                 }
#                             ),
#                         }
#                     ),
#                     serialization.PythonEnlargeStackWorkaroundNonhashedCode,
#                     serialization.PythonCacheManagerFunctionNonhashedCode,
#                     serialization.PythonModelineNonhashedCode,
#                 ]
#                 + list(epilog)
#             )
#         ],
#         post_config=dict(  # not hashed
#             log_batch_size=True,
#             cleanup_old_models=True,
#             # debug_add_check_numerics_ops = True
#             # debug_add_check_numerics_on_output = True
#             # stop_on_nonfinite_train_score = False,
#             torch_log_memory_usage=True,
#             watch_memory=True,
#             use_lovely_tensors=True,
#             use_train_proc_manager=True,
#             alias=prefix_name
#         ),
#         sort_config=False,
#     )
#     if post_config:
#         returnn_train_config.post_config = dict_update_deep(returnn_train_config.post_config, post_config)

#     for k, v in SharedPostConfig.items():
#         if k in returnn_train_config.config or k in returnn_train_config.post_config:
#             continue
#         returnn_train_config.post_config[k] = v

#     # There might be some further functions in the config, e.g. some dataset postprocessing.
#     returnn_train_config = get_serializable_config(
#         returnn_train_config,
#         # The only dim tags we directly have in the config are via extern_data, maybe also model_outputs.
#         # All other dim tags are inside functions such as get_model or train_step,
#         # so we do not need to care about them here, only about the serialization of those functions.
#         # Those dim tags and those functions are already handled above.
#         serialize_dim_tags=False,
#     )

#     for k, v in dict(
#         log_verbosity=5,
#         num_epochs=150,
#         time_rqmt=80,
#         mem_rqmt=30 if gpu_mem and gpu_mem > 11 else 15, # 15, 30
#         cpu_rqmt=4 if (not num_processes or num_processes <= 4) else 3,
#         horovod_num_processes=num_processes,  # legacy name but also applies for Torch
#     ).items():
#         if k not in kwargs or kwargs[k] is None:
#             kwargs[k] = v
#     returnn_train_job = ReturnnTrainingJob(returnn_train_config, **kwargs)
#     returnn_train_job.add_alias(prefix_name + "/train")
#     if gpu_mem:
#         returnn_train_job.rqmt["gpu_mem"] = gpu_mem
#     if env_updates:
#         for k, v in env_updates.items():
#             returnn_train_job.set_env(k, v)
#     tk.register_output(prefix_name + "/train_scores", returnn_train_job.out_learning_rates)

#     return ModelWithCheckpoints.from_training_job(definition=model_def, training_job=returnn_train_job)


# def _returnn_custom_get_model(*, epoch: int, **_kwargs_unused):
#     from returnn.tensor import Tensor
#     from returnn.config import get_global_config

#     config = get_global_config()
#     default_input_key = config.typed_value("default_input")
#     default_target_key = config.typed_value("target")
#     extern_data_dict = config.typed_value("extern_data")
#     data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
#     targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
#     assert targets.sparse_dim and targets.sparse_dim.vocab, f"no vocab for {targets}"

#     model_def = config.typed_value("_model_def")
#     model = model_def(epoch=epoch, in_dim=data.feature_dim, target_dim=targets.sparse_dim)
#     return model


# def _returnn_custom_train_step(*, model, extern_data: TensorDict, **_kwargs_unused):
#     from returnn.config import get_global_config

#     config = get_global_config()
#     default_input_key = config.typed_value("default_input")
#     default_target_key = config.typed_value("target")
#     data = extern_data[default_input_key]
#     data_spatial_dim = data.get_time_dim_tag()
#     targets = extern_data[default_target_key]
#     targets_spatial_dim = targets.get_time_dim_tag()
#     train_def: TrainDef = config.typed_value("_train_def")
#     if train_def.__name__ == "ctc_sum_training":
#         seq_tags = extern_data["seq_tag"]
#         train_def(
#             model=model,
#             data=data,
#             data_spatial_dim=data_spatial_dim,
#             seq_tags=seq_tags,
#             targets=targets,
#             targets_spatial_dim=targets_spatial_dim,
#         )
#     elif train_def.__name__ == "ce_training":
#         targets_indices = None
#         if "targets_indices" in extern_data:
#             targets_indices = extern_data["targets_indices"]
            
#         train_def(
#             model=model,
#             data=data,
#             data_spatial_dim=data_spatial_dim,
#             targets=targets,
#             targets_spatial_dim=targets_spatial_dim,
#             targets_indices=targets_indices
#         )
#     elif train_def.__name__ == "ctc_training" or train_def.__name__ == "seq_gamma_training":
#         nbest_lengths = None
#         scores = None
#         if "nbest_lengths" in extern_data:
#             nbest_lengths = extern_data["nbest_lengths"]
#         if "scores" in extern_data:
#             scores = extern_data["scores"]
#         seq_tags = extern_data["seq_tag"]
#         train_def(
#             model=model,
#             data=data,
#             data_spatial_dim=data_spatial_dim,
#             targets=targets,
#             targets_spatial_dim=targets_spatial_dim,
#             nbest_lengths=nbest_lengths,
#             scores=scores,
#             seq_tags=seq_tags,
#         )
#     else:
#         train_def(
#             model=model,
#             data=data,
#             data_spatial_dim=data_spatial_dim,
#             targets=targets,
#             targets_spatial_dim=targets_spatial_dim,
#         )


# def score_histogram(
#     *,
#     model: Model,
#     label_log_prob: Tensor,
#     enc_spatial_dim: Dim,
#     hyperparameters: dict,
#     prior_file: tk.Path = None,
#     train_lm: bool = False,
#     arpa_lm: Optional[str] = None,
#     targets: Tensor,
#     targets_spatial_dim: Dim,
#     nbest_lengths: Tensor = None
# ) -> Tensor:
#     from returnn.config import get_global_config
#     from i6_experiments.users.mueller.experiments.ctc_baseline.training import _rescore

#     config = get_global_config()  # noqa
#     nbest = config.int("ps_nbest", 1)

#     if data.feature_dim and data.feature_dim.dimension == 1:
#         data = rf.squeeze(data, axis=data.feature_dim)
#     assert not data.feature_dim  # raw audio
    
    
#     assert nbest_lengths is not None
    
#     prior_file = config.typed_value("empirical_prior")
#     rescore_alignment_prior = config.bool("rescore_alignment_prior", False)
#     assert hyperparameters and prior_file
    
#     new_spatial_dim = targets_spatial_dim.div_left(nbest)
#     new_spatial_dim_raw = new_spatial_dim.dyn_size_ext.raw_tensor
#     targets_raw = targets.raw_tensor
#     lengths_raw = nbest_lengths.raw_tensor
    
#     # Split targets into nbest connsidering the nbest lengths
#     tensor_ls = []
#     sizes_ls = []
#     for i in range(nbest):
#         max_len = lengths_raw[:, i].max()
#         # rf.pad_packed
#         targets_i = []
#         for b in range(targets_raw.shape[0]):
#             if lengths_raw[b][i] > 0:
#                 s = new_spatial_dim_raw[b] * i
#                 t_i = targets_raw[b][s:s+lengths_raw[b][i]]
#                 t_i = torch.nn.functional.pad(t_i, (0, max_len - lengths_raw[b][i]), value=model.eos_idx)
#                 targets_i.append(t_i)
#             else:
#                 t_i = torch.full((max_len,), model.eos_idx, dtype=torch.int32, device=data.raw_tensor.device)
#                 targets_i.append(t_i)
#         targets_i = torch.stack(targets_i, dim=0)
#         new_s = rf.convert_to_tensor(lengths_raw[:, i], dims=(batch_dim,))
#         new_s = Dim(new_s, name=f"out_spatial_{i}", dyn_size_ext=new_s)
#         targets_i = rf.convert_to_tensor(targets_i, dims=(batch_dim, new_s), sparse_dim=targets.sparse_dim)
#         tensor_ls.append(targets_i)
#         sizes_ls.append(new_s)
    
#     loss_ls = []
            
#     if rescore_alignment_prior:
#         prior_weight = hyperparameters.get("prior_weight", 0.0)
#         if prior_file and prior_weight > 0.0:
#             prior = np.loadtxt(prior_file, dtype="float32")
#             prior *= prior_weight
#             prior = torch.tensor(prior, dtype=torch.float32, device=log_probs.raw_tensor.device)
#             assert prior.size(0) == log_probs.raw_tensor.size(2), "Prior size does not match!"
#             prior = rtf.TorchBackend.convert_to_tensor(prior, dims=[model.wb_target_dim], dtype="float32")
#             log_probs = log_probs - prior
    
#     for j in range(nbest):
#         targets_s = tensor_ls[j]
#         targets_spatial_dim_s = sizes_ls[j]
        
#         with torch.no_grad():
#             lm_prior_score = _rescore(targets_s, targets_spatial_dim_s, model, hyperparameters, prior_file if not rescore_alignment_prior else None).raw_tensor
        
#         if config.bool("use_eos_postfix", False):
#             targets_s, (targets_spatial_dim_s,) = rf.pad(
#                 targets_s, axes=[targets_spatial_dim_s], padding=[(0, 1)], value=model.eos_idx
#             )
                

#         loss = ctc_loss_fixed_grad(
#             logits=log_probs,
#             logits_normalized=True,
#             targets=targets_s,
#             input_spatial_dim=enc_spatial_dim,
#             targets_spatial_dim=targets_spatial_dim_s,
#             blank_index=model.blank_idx,
#         )
#         loss_rescored = (-loss).raw_tensor + lm_prior_score
#         if j > 0:
#             # Set loss to -inf if target length is 0
#             loss_rescored = torch.where(targets_spatial_dim_s.dyn_size_ext.raw_tensor == 0, float("-inf"), loss_rescored)
#         loss_ls.append(loss_rescored)

#     loss = torch.stack(loss_ls, dim=-1)
#     loss = torch.log_softmax(loss, dim=-1)
#     nbest_dim = Dim(nbest, name="nbest")
#     loss = rf.convert_to_tensor(loss, dims=[batch_dim, nbest_dim], dtype="float32", name=f"rescored_ctc")
    
#     return loss