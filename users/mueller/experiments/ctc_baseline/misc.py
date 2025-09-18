"""
helpers for training.

- Uses ``unhashed_package_root`` now, via :func:`i6_experiments.users.zeyer.utils.sis_setup.get_base_module`.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Dict, Any, Sequence
import torch
import numpy as np
import copy
from torcheval.metrics.functional import word_error_rate

from sisyphus import tk
from returnn.tensor import batch_dim
import returnn.frontend as rf
import returnn.torch.frontend as rtf
from returnn.tensor import TensorDict, Tensor, Dim

from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn import ReturnnConfig
from i6_core.util import instanciate_delayed

from i6_experiments.users.mueller.experiments.ctc_baseline.training import _LM_score, _prior_score, _rescore
from i6_experiments.users.mueller.experiments.ctc_baseline.decoding import recog_flashlight_ngram, recog_ffnn
from i6_experiments.users.mueller.experiments.ctc_baseline.utils import convert_to_output_hyps, hyps_ids_to_label
from i6_experiments.users.mueller.experiments.ctc_baseline.model import Model, Wav2VecModel, _log_mel_feature_dim
from i6_experiments.users.mueller.experiments.ctc_baseline.sum_criterion import get_lm_logits, sum_loss_ngram
from i6_experiments.users.mueller.experiments.ctc_baseline.configs import _batch_size_factor
from i6_experiments.users.mueller.experiments.language_models.ffnn import FeedForwardLm, get_ffnn_lm
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
    from i6_experiments.common.setups import serialization
    from i6_experiments.users.zeyer.datasets.task import Task, DatasetConfig, VocabConfig
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, Checkpoint, ModelWithCheckpoint

# Scores for Max and GT -----------------------------------------------

def subset_scoring(
    *,
    model_def: Union[ModelDef, ModelDefWithCfg],
    checkpoint: Optional[Checkpoint],
    lm_name: str,
    lm_checkpoint_path: Optional[tk.Path],
    vocab: VocabConfig,
    prior_file: tk.Path,
    forward_alias_name: str,
    forward_rqmt: Optional[Dict[str, Any]] = None,
    forward_device: str = "gpu",
):
    USE_WORD_4GRAM = True
    LENGTH_NORM = False
    
    if USE_WORD_4GRAM:
        lm_name = "word4gram"
        arpa_file = get_binary_lm(get_arpa_lm_dict()["4gram"])
    else:
        assert lm_name
        assert lm_name.startswith("ffnn")
    
    config = {
        "prior_file": prior_file,
        "hyperparameters": {
            "lm_weight": 1.25,
            "prior_weight": 0.3,
            "lm_order": lm_name,
            "beam_size": 80,
        },
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
        
    
    config["preload_from_files"] = {}
    if checkpoint:
        config["preload_from_files"] = {
            "chkpt": {
                "filename": checkpoint,
                "ignore_params_prefixes": ["train_language_model."],
            },
        }
    if lm_checkpoint_path is not None:
        config["preload_from_files"]["recog_lm"] = {
            "prefix": "recog_language_model.",
            "filename": lm_checkpoint_path,
        }
        
    mc = model_def.config.copy()
    mc["train_language_model"] = {"class": "FeedForwardLm", "context_size": 2}
    model_def = ModelDefWithCfg(
        model_def=model_def.model_def,
        config=mc
    )
    full_sum_lm_checkpoint_path = get_ffnn_lm(vocab, context_size=2).checkpoint
    if full_sum_lm_checkpoint_path is not None:
        config["preload_from_files"]["train_lm"] = {
            "prefix": "train_language_model.",
            "filename": full_sum_lm_checkpoint_path,
        }
    if LENGTH_NORM:
        config["length_norm"] = True
    config["__version"] = 4
        
    audio_opts_ = _raw_audio_opts.copy()
    dataset_common_opts = dict(audio=audio_opts_, audio_dim=1, vocab=vocab)
    # dataset = LibrispeechOggZip(**dataset_common_opts, main_key="train-other-860")# , forward_subset=18_000)
    dataset = LibrispeechOggZip(**dataset_common_opts, main_key="dev-other")
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
        
        ==> 16_000: ca 61h
        
        
    Seq-length 'data' Stats:
        36000 seqs
        Mean: 196029.9418888878
        Std dev: 61898.81456420404
        Min/max: 14720 / 400560
    Seq-length 'classes' Stats:
        36000 seqs
        Mean: 86.06625000000001
        Std dev: 30.52811954837234
        Min/max: 1 / 195
    Seq-length 'orth' Stats:
        36000 seqs
        Mean: 175.96691666666712
        Std dev: 62.70739757834533
        Min/max: 3 / 403
    Seq-length 'raw' Stats:
        36000 seqs
        Mean: 1.0
        Std dev: 0.0
        Min/max: 1 / 1
    Data 'data' Stats:
        36000 seqs, 7057077908 total frames, 196029.941889 average frames
        Mean: [0.0001587]
        Std dev: [0.11555858]
        Min/max: [-1.] / [1.]
        
        ==> 32_000: ca 122h
    """
        
    output_file = "scores.txt"
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=None,
        returnn_config=_returnn_scoring_config(
            dataset=dataset,
            model_def=model_def,
            config=config,
        ),
        output_files=[output_file],
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        device=forward_device,
        cpu_rqmt=4,
        mem_rqmt=16,
        time_rqmt=172,
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
    from returnn.tensor import batch_dim
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
        use_word_4gram=config.typed_value("use_word_4gram"),
        arpa_file=config.typed_value("arpa_file", default=None),
        lexicon=config.typed_value("lexicon", default=None),
        length_norm=config.typed_value("length_norm", default=False),
    )
    assert isinstance(scores_dict, dict)
    rf.get_run_ctx().mark_as_output(scores_dict["gt_score"]["am_ctc"], "gt_am_ctc", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["gt_score"]["am_fa"], "gt_am_fa", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["gt_score"]["lm"], "gt_lm", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["gt_score"]["prior_la"], "gt_prior_la", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["gt_score"]["prior_fr"], "gt_prior_fr", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["gt_score"]["combined_ctc"], "gt_comb_ctc", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["gt_score"]["combined_fa"], "gt_comb_fa", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["max_score"]["am_ctc"], "max_am_ctc", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["max_score"]["am_fa"], "max_am_fa", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["max_score"]["lm"], "max_lm", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["max_score"]["prior_la"], "max_prior_la", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["max_score"]["prior_fr"], "max_prior_fr", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["max_score"]["combined_ctc"], "max_comb_ctc", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["max_score"]["combined_fa"], "max_comb_fa", dims=[batch_dim])
    rf.get_run_ctx().mark_as_output(scores_dict["fs_score"]["combined"], "fs_comb", dims=[batch_dim])
    
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
                    "am_ctc": init_prob,
                    "am_fa": init_prob,
                    "lm": init_prob,
                    "prior_la": init_prob,
                    "prior_fr": init_prob,
                    "combined_ctc": init_prob,
                    "combined_fa": init_prob,
                },
                "max_score": {
                    "am_ctc": init_prob,
                    "am_fa": init_prob,
                    "lm": init_prob,
                    "prior_la": init_prob,
                    "prior_fr": init_prob,
                    "combined_ctc": init_prob,
                    "combined_fa": init_prob,
                },
                "fs_score": {
                    "combined": init_prob,
                }
            }
            self.n_seq = 0

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            self.n_seq += 1
            
            self.scores_sum["gt_score"]["am_ctc"] = np.logaddexp(self.scores_sum["gt_score"]["am_ctc"], outputs["gt_am_ctc"].raw_tensor)
            self.scores_sum["gt_score"]["am_fa"] = np.logaddexp(self.scores_sum["gt_score"]["am_fa"], outputs["gt_am_fa"].raw_tensor)
            self.scores_sum["gt_score"]["lm"] = np.logaddexp(self.scores_sum["gt_score"]["lm"], outputs["gt_lm"].raw_tensor)
            self.scores_sum["gt_score"]["prior_la"] = np.logaddexp(self.scores_sum["gt_score"]["prior_la"], outputs["gt_prior_la"].raw_tensor)
            self.scores_sum["gt_score"]["prior_fr"] = np.logaddexp(self.scores_sum["gt_score"]["prior_fr"], outputs["gt_prior_fr"].raw_tensor)
            self.scores_sum["gt_score"]["combined_ctc"] = np.logaddexp(self.scores_sum["gt_score"]["combined_ctc"], outputs["gt_comb_ctc"].raw_tensor)
            self.scores_sum["gt_score"]["combined_fa"] = np.logaddexp(self.scores_sum["gt_score"]["combined_fa"], outputs["gt_comb_fa"].raw_tensor)
            self.scores_sum["max_score"]["am_ctc"] = np.logaddexp(self.scores_sum["max_score"]["am_ctc"], outputs["max_am_ctc"].raw_tensor)
            self.scores_sum["max_score"]["am_fa"] = np.logaddexp(self.scores_sum["max_score"]["am_fa"], outputs["max_am_fa"].raw_tensor)
            self.scores_sum["max_score"]["lm"] = np.logaddexp(self.scores_sum["max_score"]["lm"], outputs["max_lm"].raw_tensor)
            self.scores_sum["max_score"]["prior_la"] = np.logaddexp(self.scores_sum["max_score"]["prior_la"], outputs["max_prior_la"].raw_tensor)
            self.scores_sum["max_score"]["prior_fr"] = np.logaddexp(self.scores_sum["max_score"]["prior_fr"], outputs["max_prior_fr"].raw_tensor)
            self.scores_sum["max_score"]["combined_ctc"] = np.logaddexp(self.scores_sum["max_score"]["combined_ctc"], outputs["max_comb_ctc"].raw_tensor)
            self.scores_sum["max_score"]["combined_fa"] = np.logaddexp(self.scores_sum["max_score"]["combined_fa"], outputs["max_comb_fa"].raw_tensor)
            self.scores_sum["fs_score"]["combined"] = np.logaddexp(self.scores_sum["fs_score"]["combined"], outputs["fs_comb"].raw_tensor)

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
    
    # Score ground truth with CTC
    ctc_loss_gt = ctc_loss_fixed_grad(
        logits=log_probs,
        logits_normalized=True,
        targets=targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx,
    )
    am_gt_score_ctc = -ctc_loss_gt
    # NOTE: using label prior here
    prior_gt_score_la = _prior_score(
        targets=targets,
        targets_spatial_dim=targets_spatial_dim,
        model=model,
        prior_file=prior_file,
        force_label_prior=True,
    )
    
    # Score ground truth with forced alignment
    forced_targets_align = None
    am_gt_score_fa = None
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
        if am_gt_score_fa is None:
            am_gt_score_fa = am_gt_score_b.sum(dim=-1) # sum over time
            forced_targets_align = forced_align_b
        else:
            am_gt_score_fa = torch.cat((am_gt_score_fa, am_gt_score_b.sum(dim=-1)), dim=0)
            forced_targets_align = torch.cat((forced_targets_align, forced_align_b), dim=0)
    am_gt_score_fa = rf.convert_to_tensor(am_gt_score_fa, dims=(batch_dim,), dtype="float32", name="am_gt_score")
    forced_targets_align = rf.convert_to_tensor(forced_targets_align, dims=[batch_dim, enc_spatial_dim], dtype="int32", name="forced_targets_align")
    # NOTE: using frame prior here
    prior_gt_score_fr = _prior_score(
        targets=forced_targets_align,
        targets_spatial_dim=enc_spatial_dim,
        model=model,
        prior_file=prior_file,
        force_label_prior=False,
    )
    
    lm_gt_score = _LM_score(targets, targets_spatial_dim, model, hyperparameters.get("lm_order", None), train_lm=False, arpa_file=arpa_file)
    
    lm_scale = hyperparameters["lm_weight"]
    prior_scale = hyperparameters["prior_weight"]
    
    combined_gt_score_ctc = am_gt_score_ctc + lm_scale * lm_gt_score - prior_scale * prior_gt_score_la
    combined_gt_score_fa = am_gt_score_fa + lm_scale * lm_gt_score - prior_scale * prior_gt_score_fr
    
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
    
    # Score max with CTC
    ctc_loss_max = ctc_loss_fixed_grad(
        logits=log_probs,
        logits_normalized=True,
        targets=max_hyp,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=max_hyp_spatial_dim,
        blank_index=model.blank_idx,
    )
    am_max_score_ctc = -ctc_loss_max
    prior_max_score_la = _prior_score(
        targets=max_hyp,
        targets_spatial_dim=max_hyp_spatial_dim,
        model=model,
        prior_file=prior_file,
        force_label_prior=True,
    )
    
    # Score max with forced alignment
    forced_max_align = None
    am_max_score_fa = None
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
        if am_max_score_fa is None:
            am_max_score_fa = am_max_score_b.sum(dim=-1) # sum over time
            forced_max_align = forced_align_b
        else:
            am_max_score_fa = torch.cat((am_max_score_fa, am_max_score_b.sum(dim=-1)), dim=0)
            forced_max_align = torch.cat((forced_max_align, forced_align_b), dim=0)
    am_max_score_fa = rf.convert_to_tensor(am_max_score_fa, dims=(batch_dim,), dtype="float32", name="am_max_score")
    forced_max_align = rf.convert_to_tensor(forced_max_align, dims=[batch_dim, enc_spatial_dim], dtype="int32", name="forced_max_align")
    prior_max_score_fr = _prior_score(
        targets=forced_max_align,
        targets_spatial_dim=enc_spatial_dim,
        model=model,
        prior_file=prior_file,
        force_label_prior=False,
    )
    
    lm_max_score = _LM_score(max_hyp, max_hyp_spatial_dim, model, hyperparameters.get("lm_order", None), train_lm=False, arpa_file=arpa_file)
    
    combined_max_score_ctc = am_max_score_ctc + lm_scale * lm_max_score - prior_scale * prior_max_score_la
    combined_max_score_fa = am_max_score_fa + lm_scale * lm_max_score - prior_scale * prior_max_score_fr
    
    # Calculate full-sum score with 3-gram LM
    assert model.train_language_model
    assert model.train_language_model.vocab_dim == model.target_dim
    lm: FeedForwardLm = model.train_language_model
    context_size = model.train_language_model.conv_filter_size_dim.dimension
    context_dim = rf.Dim(context_size, name="context")
    lm_out_dim = rf.Dim(context_size + 1, name="context+1")
    target = torch.arange(model.target_dim.dimension, device=log_probs.device)
    if context_size == 2:
        target1 = target.unsqueeze(1).expand(model.target_dim.dimension, model.target_dim.dimension)
        target2 = target.unsqueeze(0).expand(model.target_dim.dimension, model.target_dim.dimension)
        target = torch.stack([target1, target2], dim=-1)
        batch_dims = [Dim(model.target_dim.dimension, name="v1"), Dim(model.target_dim.dimension, name="v2")]
    elif context_size == 1:
        target = target.unsqueeze(1)
        batch_dims = [Dim(model.target_dim.dimension, name="v1")]
    else:
        raise NotImplementedError(f"Full-sum on context size {context_size} not implemented")
    target = rf.convert_to_tensor(target, dims=batch_dims + [context_dim], sparse_dim=model.target_dim)
    lm_state = lm.default_initial_state(batch_dims=[])
    lm_logits, lm_state = get_lm_logits(batch_dims, target, lm, context_dim, lm_out_dim, lm_state)
    lm_logits = rf.gather(lm_logits, axis=lm_out_dim, indices=rf.last_frame_position_of_dim(lm_out_dim))
    assert lm_logits.dims == (*batch_dims, model.target_dim)
    lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
    lm_log_probs = lm_log_probs.raw_tensor
    
    # Read out prior
    prior = None
    prior = np.loadtxt(prior_file, dtype="float32")
    prior = torch.tensor(prior, dtype=torch.float32, device=log_probs.device)
    assert prior.shape[0] == log_probs.raw_tensor.shape[-1]
    
    log_prob_raw = log_probs.raw_tensor.permute(1, 0, 2)
    fs_loss = sum_loss_ngram(
        log_probs=log_prob_raw,
        log_lm_probs=lm_log_probs,
        log_prior=prior,
        input_lengths=enc_spatial_dim.dyn_size_ext.raw_tensor,
        top_k=0,
        LM_order=lm_log_probs.ndim,
        am_scale=1.0,
        lm_scale=0.8,
        prior_scale=prior_scale,
        horizontal_prior=True,
        blank_prior=True,
        blank_idx=model.blank_idx,
        eos_idx=model.eos_idx,
        device=log_prob_raw.device,
    )
    fs_score = rtf.TorchBackend.convert_to_tensor(-fs_loss, dims = [batch_dim], dtype = "float32", name=f"full_sum")
    
    if length_norm:
        norm = rf.log(enc_spatial_dim.dyn_size_ext)
        norm = rf.copy_to_device(norm, data.device)
        ret = {
            "gt_score": {
                "am_ctc": am_gt_score_ctc - norm,
                "am_fa": am_gt_score_fa - norm,
                "lm": lm_gt_score - norm,
                "prior_la": prior_gt_score_la - norm,
                "prior_fr": prior_gt_score_fr - norm,
                "combined_ctc": combined_gt_score_ctc - norm,
                "combined_fa": combined_gt_score_fa - norm,
            },
            "max_score": {
                "am_ctc": am_max_score_ctc - norm,
                "am_fa": am_max_score_fa - norm,
                "lm": lm_max_score - norm,
                "prior_la": prior_max_score_la - norm,
                "prior_fr": prior_max_score_fr - norm,
                "combined_ctc": combined_max_score_ctc - norm,
                "combined_fa": combined_max_score_fa - norm,
            },
            "fs_score": {
                "combined": fs_score - norm,
            }
        }
    else:
        ret = {
            "gt_score": {
                "am_ctc": am_gt_score_ctc,
                "am_fa": am_gt_score_fa,
                "lm": lm_gt_score,
                "prior_la": prior_gt_score_la,
                "prior_fr": prior_gt_score_fr,
                "combined_ctc": combined_gt_score_ctc,
                "combined_fa": combined_gt_score_fa,
            },
            "max_score": {
                "am_ctc": am_max_score_ctc,
                "am_fa": am_max_score_fa,
                "lm": lm_max_score,
                "prior_la": prior_max_score_la,
                "prior_fr": prior_max_score_fr,
                "combined_ctc": combined_max_score_ctc,
                "combined_fa": combined_max_score_fa,
            },
            "fs_score": {
                "combined": fs_score,
            }
        }
    
    return ret
    

# Histogram for N-Best scores------------------------------------------

def histogram_scoring(
    *,
    model_def: Union[ModelDef, ModelDefWithCfg],
    checkpoint: Optional[Checkpoint],
    lm_name: str,
    lm_checkpoint_path: Optional[tk.Path],
    vocab: VocabConfig,
    prior_file: tk.Path,
    forward_alias_name: str,
    forward_rqmt: Optional[Dict[str, Any]] = None,
    forward_device: str = "gpu",
):
    USE_WORD_4GRAM = True
    LENGTH_NORM = False
    NBEST = 5
    
    if USE_WORD_4GRAM:
        lm_name = "word4gram"
        arpa_file = get_binary_lm(get_arpa_lm_dict()["4gram"])
    else:
        assert lm_name
        assert lm_name.startswith("ffnn")
    
    config = {
        "prior_file": prior_file,
        "hyperparameters": {
            "lm_weight": 1.25,
            "prior_weight": 0.3,
            "lm_order": lm_name,
            "beam_size": 80,
            "ps_nbest": NBEST,
        },
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
        
    config["_version"] = 3
        
    audio_opts_ = _raw_audio_opts.copy()
    dataset_common_opts = dict(audio=audio_opts_, audio_dim=1, vocab=vocab)
    dataset = LibrispeechOggZip(**dataset_common_opts, main_key="train-other-860", forward_subset=36_000)
        
    output_files = ["hist_scores.npy", "hist_scores_stats.txt"]
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=_returnn_hist_scoring_config(
            dataset=dataset,
            model_def=model_def,
            config=config,
        ),
        output_files=output_files,
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
        device=forward_device,
        cpu_rqmt=4,
        mem_rqmt=16,
        time_rqmt=64,
    )
    if forward_rqmt:
        forward_job.rqmt.update(forward_rqmt)
    assert forward_alias_name
    forward_job.add_alias(forward_alias_name)

    tk.register_output(forward_alias_name, forward_job.out_files[output_files[1]])
    
SharedPostConfig = {
    # In case pretraining overwrites some of these, they need a default.
    "accum_grad_multiple_step": None,
    "use_last_best_model": None,
}
  
def _returnn_hist_scoring_config(
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
                    serialization.Import(score_histogram, import_as="_score_def"),
                    serialization.Import(_returnn_hist_score_step, import_as="forward_step"),
                    serialization.Import(_returnn_hist_get_forward_callback, import_as="forward_callback"),
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

def _returnn_hist_score_step(*, model, extern_data: TensorDict, **_kwargs_unused):
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
    if default_input_key:
        data = extern_data[default_input_key]
        data_spatial_dim = data.get_time_dim_tag()
    else:
        data, data_spatial_dim = None, None

    default_target_key = config.typed_value("target")
    targets = extern_data[default_target_key]
    targets_spatial_dim = targets.get_time_dim_tag()

    score_def = config.typed_value("_score_def")
    nbest_scores_dict = score_def(
        model=model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        targets=targets,
        targets_spatial_dim=targets_spatial_dim,
        hyperparameters=config.typed_value("hyperparameters"),
        prior_file=config.typed_value("prior_file"),
        use_word_4gram=config.typed_value("use_word_4gram"),
        arpa_file=config.typed_value("arpa_file", default=None),
        lexicon=config.typed_value("lexicon", default=None),
        length_norm=config.typed_value("length_norm", default=False),
    )
    assert isinstance(nbest_scores_dict, dict)
    dims = nbest_scores_dict["combined"].dims
    rf.get_run_ctx().mark_as_output(nbest_scores_dict["combined"], "combined", dims=dims)
    rf.get_run_ctx().mark_as_output(nbest_scores_dict["am"], "am", dims=dims)
    rf.get_run_ctx().mark_as_output(nbest_scores_dict["lm"], "lm", dims=dims)
    rf.get_run_ctx().mark_as_output(nbest_scores_dict["prior"], "prior", dims=dims)
    rf.get_run_ctx().mark_as_output(nbest_scores_dict["wer"], "wer", dims=dims)
        
def _returnn_hist_get_forward_callback():
    from typing import TextIO
    from returnn.forward_iface import ForwardCallbackIface
    import gzip
    from scipy.special import logsumexp

    class _ReturnnHistScoringForwardCallbackIface(ForwardCallbackIface):
        def __init__(self):
            self.out_file: Optional[TextIO] = None

        def init(self, *, model):
            self.scores_dict = {
                "combined": [],
                "am": [],
                "lm": [],
                "prior": [],
                "wer": [],
            }

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            self.scores_dict["combined"].append(outputs["combined"].raw_tensor)
            self.scores_dict["am"].append(outputs["am"].raw_tensor)
            self.scores_dict["lm"].append(outputs["lm"].raw_tensor)
            self.scores_dict["prior"].append(outputs["prior"].raw_tensor)
            self.scores_dict["wer"].append(outputs["wer"].raw_tensor)

        def finish(self):
            self.out_file = open("hist_scores_stats.txt", "wt", encoding="utf-8")
            for key, sc in self.scores_dict.items():
                scores = np.stack(sc, axis=0)
                if key == "combined":
                    np.save("hist_scores.npy", scores)
                
                if key == "wer":
                    mean_nbest = scores.mean(axis=0)
                    std_nbest = scores.std(axis=0, ddof=1)
                else:
                    n = scores.shape[0]
                    log_sum = logsumexp(scores, axis=0)
                    mean_nbest = log_sum - np.log(n)
                    std_nbest = scores.std(axis=0, ddof=1)
                
                self.out_file.write(f"{key}: Mean per nbest: {mean_nbest.tolist()}\n")
                self.out_file.write(f"{key}: Std per nbest: {std_nbest.tolist()}")
            self.out_file.close()

    return _ReturnnHistScoringForwardCallbackIface()

def score_histogram(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    targets: Tensor,
    targets_spatial_dim: Dim,
    hyperparameters: dict,
    prior_file: tk.Path,
    use_word_4gram: bool,
    arpa_file: Optional[tk.Path],
    lexicon: Optional[tk.Path],
    length_norm: bool,
) -> dict:
    if use_word_4gram:
        assert lexicon and arpa_file
    
    # Get log_probs
    assert data.dims_set == {batch_dim, data_spatial_dim, data.feature_dim}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    assert logits.dims_set == {batch_dim, enc_spatial_dim, model.wb_target_dim}
    
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    log_probs = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    
    nbest = hyperparameters["ps_nbest"]
    
    # Extract nbest scores
    if use_word_4gram:
        nbest_hyps, _, _, beam_dim = recog_flashlight_ngram(model=model, data=data, data_spatial_dim=data_spatial_dim, arpa_4gram_lm = arpa_file, lexicon=lexicon, hyperparameters=hyperparameters, prior_file=prior_file)
        assert beam_dim.dimension == nbest, f"Beam dimension should be {nbest} for nbest score extraction but is {beam_dim.dimension}"
        nbest_hyps = nbest_hyps.raw_tensor
        nbest_hyps = nbest_hyps.tolist()
        nbest_hyps = [[model.target_dim.vocab.get_seq(seq[0]) for seq in seq_batch] for seq_batch in nbest_hyps]
    else:
        nbest_hyps, _, _, beam_dim = recog_ffnn(model=model, label_log_prob=log_probs, enc_spatial_dim=enc_spatial_dim, hyperparameters=hyperparameters, batch_dims=batch_dims, prior_file=prior_file, train_lm=False)
        assert beam_dim.dimension == nbest, f"Beam dimension should be {nbest} for nbest score extraction but is {beam_dim.dimension}"
        nbest_hyps = [[convert_to_output_hyps(model, h)for h in h_batch] for h_batch in nbest_hyps.raw_tensor.transpose(0, 1).transpose(1,2).tolist()]
    lengths = [[len(h)for h in h_b] for h_b in nbest_hyps]
    nbest_hyps_spatial_dim = torch.tensor(lengths, dtype=torch.int32, device=data.raw_tensor.device)
    max_length = nbest_hyps_spatial_dim.max().item()
    nbest_hyps_spatial_dim = rf.convert_to_tensor(nbest_hyps_spatial_dim, dims=[batch_dim, beam_dim])
    nbest_hyps_spatial_dim = rf.Dim(nbest_hyps_spatial_dim, name="out_spatial", dyn_size_ext=nbest_hyps_spatial_dim)
    nbest_hyps = [[h + [model.eos_idx] * (max_length - len(h)) for h in h_b] for h_b in nbest_hyps]
    nbest_hyps = torch.tensor(nbest_hyps, dtype=torch.int32, device=data.raw_tensor.device)
    nbest_hyps = rf.convert_to_tensor(nbest_hyps, dims=[batch_dim, beam_dim, nbest_hyps_spatial_dim], sparse_dim=model.target_dim, dtype="int32", name="nbest_hyps")
    
    log_probs = rf.expand_dim(log_probs, dim=beam_dim)
    
    ctc_loss = ctc_loss_fixed_grad(
        logits=log_probs,
        logits_normalized=True,
        targets=nbest_hyps,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=nbest_hyps_spatial_dim,
        blank_index=model.blank_idx,
    )
    ctc_loss = -ctc_loss
    
    prior_weight = hyperparameters.get("prior_weight", 0.0)
    prior_score = None
    if prior_file and prior_weight > 0.0:
        prior_score = _prior_score(
            targets=nbest_hyps,
            targets_spatial_dim=nbest_hyps_spatial_dim,
            model=model,
            prior_file=prior_file,
            force_label_prior=True,
        )
    
    lm_weight = hyperparameters["lm_weight"]
    lm_score = _LM_score(
        nbest_hyps,
        nbest_hyps_spatial_dim,
        model,
        hyperparameters.get("lm_order", None),
        train_lm=False,
        arpa_file=arpa_file,
    )
    
    combined_score = ctc_loss + lm_score * lm_weight
    if prior_score is not None:
        combined_score -= (prior_score * prior_weight)
    
    # Compute WER
    hyps = nbest_hyps.raw_tensor
    wer = []
    for i in range(hyps.size(0)):
        wer_i = []
        t = targets.raw_tensor[i, :targets_spatial_dim.dyn_size_ext.raw_tensor[i]].tolist()
        target_hyp = hyps_ids_to_label(model, t)
        for j in range(hyps.size(1)):
            h = hyps[i, j, :lengths[i][j]].tolist()
            word_hyp = hyps_ids_to_label(model, h)
            wer_i.append(word_error_rate(word_hyp, target_hyp))
        wer.append(wer_i)
    wer = torch.tensor(wer, dtype=torch.float32, device=data.raw_tensor.device)
    wer = rf.convert_to_tensor(wer, dims=[batch_dim, beam_dim], name="word_error_rate")
                
    ret = {
        "combined": combined_score,
        "am": ctc_loss,
        "lm": lm_score,
        "prior": prior_score,
        "wer": wer,
    }
    
    return ret