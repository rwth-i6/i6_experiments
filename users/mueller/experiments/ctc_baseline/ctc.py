"""
CTC experiments.
"""

from __future__ import annotations

import copy
import functools
import torch
from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, Callable, Dict, Any, List
import numpy as np

import returnn.frontend as rf
import returnn.torch.frontend as rtf
from returnn.tensor import Tensor, Dim, batch_dim
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerEncoderLayer, ConformerConvSubsample
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.tensor_array import TensorArray

from sisyphus import tk

from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, TrainDef, RecogDef
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.zeyer.returnn.models.rf_layerdrop import SequentialLayerDrop
from i6_experiments.users.mueller.train import ExtendedTrainDef
from i6_experiments.users.mueller.experiments.language_models.n_gram import get_count_based_n_gram

from i6_core.util import uopen

from .configs import _get_cfg_lrlin_oclr_by_bs_nep, _batch_size_factor, config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4, dict_update_deep, post_config

if TYPE_CHECKING:
    from i6_experiments.common.setups import serialization
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints
    from i6_experiments.users.mueller.datasets.task import Task
    from i6_experiments.users.zeyer.datasets.score_results import RecogOutput


OUT_BLANK_LABEL = "<blank>"
CHECK_DECODER_CONSISTENCY = False
_raw_sample_rate = _batch_size_factor * 100  # bs factor is from 10ms frames to raw samples


# 'train-clean-360/4837-302000-0048/4837-302000-0048'


def py():
    """Sisyphus entry point"""
    # relPosAttDef: Use the default RelPosSelfAttention instead of the Shawn et al 2018 style, old RETURNN way.
    enc_conformer_layer_default = rf.build_dict(
        rf.encoder.conformer.ConformerEncoderLayer,
        ff_activation=rf.build_dict(rf.relu_square),
        num_heads=8,
    )

    # Supervised CTC-baseline
    # vocab = "spm20k"
    # vocab = "char"
    vocab = "bpe128"
    # vocab = "bpe10k"
    use_flashlight = True
    use_greedy = False
    epochs = 500
    self_training_rounds = 1
    test_self_training_on_small_dataset = 0 # TODO remove this parameter
    train_small = True
    with_prior = True
    use_sum_criterion = False
    aux_loss = True
    alt_decoder = True
    horizontal_prior = True
    
    if train_small:
        epochs = 50
    if self_training_rounds > 0:
        self_epochs = 450
    
    decoder_hyperparameters = None
    if use_greedy:
        decoder_hyperparameters = {
            "greedy": True
        }
        greedy_str = "-recog_greedy"
        if with_prior:
            decoder_hyperparameters["prior_weight"] = 0.2
            greedy_str += f"_p{str(decoder_hyperparameters['prior_weight']).replace('.', '')}"
    elif use_flashlight:
        decoder_hyperparameters = {
            "log_add": False,
            "nbest": 1,
            "beam_size": 80,
            "lm_weight": 1.9,
            "use_logsoftmax": True,
            "use_lm": True,
            "use_lexicon": True,
        }
        if with_prior:
            decoder_hyperparameters["prior_weight"] = 0.2
            
        p0 = f"_p{str(decoder_hyperparameters['prior_weight']).replace('.', '')}" if with_prior else ""
        p1 = "sum" if decoder_hyperparameters['log_add'] else "max"
        p2 = f"n{decoder_hyperparameters['nbest']}"
        p3 = f"b{decoder_hyperparameters['beam_size']}"
        p4 = f"w{str(decoder_hyperparameters['lm_weight']).replace('.', '')}"
        p5 = "_logsoftmax" if decoder_hyperparameters['use_logsoftmax'] else ""
        p6 = "_noLM" if not decoder_hyperparameters['use_lm'] else ""
        p7 = "_noLEX" if not decoder_hyperparameters['use_lexicon'] else ""
        lm_hyperparamters_str = f"{p0}_{p1}_{p2}_{p3}_{p4}{p5}{p6}{p7}"
        
        if alt_decoder:
            alt_decoder_hyperparameters = decoder_hyperparameters.copy()
            alt_decoder_hyperparameters["lm_weight"] = 1.15
            alt_decoder_hyperparameters["beam_size"] = 70
            if with_prior:
                alt_decoder_hyperparameters["prior_weight"] = 0.3
                
            a0 = f"_p{str(alt_decoder_hyperparameters['prior_weight']).replace('.', '')}" if with_prior else ""
            a1 = f"b{alt_decoder_hyperparameters['beam_size']}"
            a2 = f"w{str(alt_decoder_hyperparameters['lm_weight']).replace('.', '')}"
            lm_hyperparamters_str += f"_ALT{a0}_{a1}_{a2}"
    
    alias_name = f"ctc-baseline" + \
        (f"-full_sum" + ("_wo_hor_pr" if not horizontal_prior else "") if use_sum_criterion else "") + \
        (f"-self_training_{self_training_rounds}" if self_training_rounds > 0 else "") + \
        (f"-wo_aux_loss" if not aux_loss else "") + \
        (f"-dataset_size_{test_self_training_on_small_dataset}" if test_self_training_on_small_dataset > 0 else "") + \
        (f"-ds100h" if train_small else "") + \
        f"-{vocab}" + \
        (greedy_str if use_greedy else (("-recog_lm" + lm_hyperparamters_str) if use_flashlight else "-recog_albert"))
    
    config_updates = {
        **_get_cfg_lrlin_oclr_by_bs_nep(15_000, epochs),
        "optimizer.weight_decay": 1e-2,
        "__train_audio_preprocess": speed_pert_librosa_config,
        "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        "max_seq_length_default_target": None,
        "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    }
    config_updates_self_training = {
        **_get_cfg_lrlin_oclr_by_bs_nep(15_000, self_epochs),
        "optimizer.weight_decay": 1e-2,
        "__train_audio_preprocess": speed_pert_librosa_config,
        "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        "max_seq_length_default_target": None,
        "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    } if self_training_rounds > 0 else None

    train_exp(
        name = alias_name,
        config = config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
        decoder_def = model_recog_lm if (use_flashlight or use_greedy) else model_recog,
        decoder_hyperparameters = decoder_hyperparameters,
        hyperparamters_self_training = alt_decoder_hyperparameters if alt_decoder else None,
        model_config = {"enc_conformer_layer": enc_conformer_layer_default, "feature_batch_norm": True},
        config_updates = config_updates,
        config_updates_self_training = config_updates_self_training,
        vocab = vocab,
        self_training_rounds = self_training_rounds,
        train_small = train_small,
        test_self_training_on_small_dataset = test_self_training_on_small_dataset,
        with_prior = with_prior,
        use_sum_criterion=use_sum_criterion,
        aux_loss=aux_loss,
        horizontal_prior=horizontal_prior
    )
    

_train_experiments: Dict[str, ModelWithCheckpoints] = {}


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    decoder_def: Callable,
    *,
    decoder_hyperparameters: dict = None,
    hyperparamters_self_training: dict = None,
    model_def: Optional[Union[ModelDefWithCfg, ModelDef[Model]]] = None,
    vocab: str = "bpe10k",
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    train_def: Optional[TrainDef[Model]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    config_updates: Optional[Dict[str, Any]] = None,
    config_updates_self_training: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    epilog: Sequence[serialization.SerializerObject] = (),
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    time_rqmt: Optional[int] = None,  # set this to 1 or below to get the fast test queue
    env_updates: Optional[Dict[str, str]] = None,
    enabled: bool = True,
    self_training_rounds: int = 0,
    train_small: bool = False,
    test_self_training_on_small_dataset: int = 0,
    with_prior: bool = False,
    use_sum_criterion: bool = False,
    aux_loss: bool = False,
    horizontal_prior: bool = True,
) -> Optional[ModelWithCheckpoints]:
    """
    Train experiment
    """
    from i6_experiments.users.mueller.train import train
    from i6_experiments.users.mueller.recog import recog_training_exp
    from i6_experiments.users.mueller.datasets.librispeech import get_librispeech_task_raw_v2, TrainDatasetSel

    print("Job Name:", name)
    if not enabled:
        return None

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    
    task, pseudo_labels_ds = get_librispeech_task_raw_v2(
        vocab=vocab,
        train_vocab_opts=train_vocab_opts,
        save_pseudo_labels = self_training_rounds > 0,
        test_self_training_on_small_dataset = test_self_training_on_small_dataset,
        ds_sel = TrainDatasetSel.train_100h if train_small else TrainDatasetSel.train_960h,
        with_prior=with_prior
    )
    config = config.copy()
    config = dict_update_deep(config, config_updates, config_deletes)
    # This logic is also in train(), but keep it here because it would break the hash because of _RecogAndScoreFunc...
    if "__train_audio_preprocess" in config:
        task: Task = copy.copy(task)
        task.train_dataset = copy.copy(task.train_dataset)
        task.train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")

    if not model_def:
        model_def = ctc_model_def
    if model_config:
        model_def = ModelDefWithCfg(model_def, model_config)
    if not train_def:
        train_def = ctc_training
    model_with_checkpoint = []
    model_with_checkpoint.append(train(
        prefix,
        task=task,
        config=config,
        post_config=dict_update_deep(post_config, post_config_updates),
        epilog=epilog,
        model_def=model_def,
        train_def=train_def,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        time_rqmt=time_rqmt if time_rqmt else (36 if train_small else 132),
    ))
    train_job = model_with_checkpoint[0].get_training_job()
    if env_updates:
        for k, v in env_updates.items():
            train_job.set_env(k, v)

    recog_post_proc_funcs = []
    if config.get("use_eos_postfix", False):
        recog_post_proc_funcs.append(_remove_eos_label_v2)
    pseudo_label_path_dict = recog_training_exp(
        prefix,
        task,
        model_with_checkpoint[0],
        recog_def=decoder_def,
        decoder_hyperparameters=decoder_hyperparameters,
        save_pseudo_labels=pseudo_labels_ds,
        calculate_pseudo_label_scores=False,
        recog_post_proc_funcs=recog_post_proc_funcs
    )
    
    # Do self training on pseudo labels
    for i in range(self_training_rounds):
        assert pseudo_label_path_dict is not None, "Pseudo label path is not set"
        prefix_self_training = prefix + f"/self-training-{i+1}"
        task, _ = get_librispeech_task_raw_v2(
            vocab=vocab,
            train_vocab_opts=train_vocab_opts,
            ds_sel = TrainDatasetSel.train_860h if train_small else TrainDatasetSel.train_960h,
            with_prior=with_prior,
            pseudo_label_path = pseudo_label_path_dict
        )
        
        config_self = config.copy()
        config_self = dict_update_deep(config_self, config_updates_self_training)
        # This logic is also in train(), but keep it here because it would break the hash because of _RecogAndScoreFunc...
        if "__train_audio_preprocess" in config_self:
            task: Task = copy.copy(task)
            task.train_dataset = copy.copy(task.train_dataset)
            task.train_dataset.train_audio_preprocess = config_self.pop("__train_audio_preprocess")
        
        if use_sum_criterion:
            train_def = ctc_sum_training
            lm_path = get_count_based_n_gram(task.train_dataset.vocab, 2)
            config_self["lm_path"] = lm_path
            
        init_checkpoint = model_with_checkpoint[i].get_last_fixed_epoch().checkpoint
        # config_self.pop("__num_processes")
        if not aux_loss:
            config_self.pop("aux_loss_layers")
        if not horizontal_prior:
            config_self["horizontal_prior"] = horizontal_prior
        model_with_checkpoint.append(train(
            prefix_self_training,
            task=task,
            config=config_self,
            post_config=dict_update_deep(post_config, post_config_updates),
            epilog=epilog,
            model_def=model_def,
            train_def=train_def,
            init_params=init_checkpoint,
            num_epochs=num_epochs,
            gpu_mem=gpu_mem,
            num_processes=num_processes,
            time_rqmt=time_rqmt if time_rqmt else (24 if use_sum_criterion else 156),
        ))
        train_job = model_with_checkpoint[i + 1].get_training_job()
        if env_updates:
            for k, v in env_updates.items():
                train_job.set_env(k, v)
        
        pseudo_label_path_dict = recog_training_exp(
            prefix_self_training,
            task,
            model_with_checkpoint[i + 1],
            recog_def=decoder_def,
            decoder_hyperparameters=hyperparamters_self_training if hyperparamters_self_training else decoder_hyperparameters,
            save_pseudo_labels=None if i+1 == self_training_rounds else pseudo_labels_ds,
            recog_post_proc_funcs=recog_post_proc_funcs,
            num_shards=64
        )

    _train_experiments[name] = model_with_checkpoint[-1]
    return model_with_checkpoint[-1]


def _remove_eos_label_v2(res: RecogOutput) -> RecogOutput:
    from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
    from i6_core.returnn.search import SearchRemoveLabelJob

    return RecogOutput(SearchRemoveLabelJob(res.output, remove_label="</s>", output_gzip=True).out_search_results)


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name


#---------------------------------------------------------------------------------------------------------------------------------------
# MODEL DEFINITION

# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80

def ctc_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    num_enc_layers = config.int("num_enc_layers", 12)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)

    conv_norm = config.typed_value("conv_norm", None)
    enc_conformer_layer = config.typed_value("enc_conformer_layer", None)
    if enc_conformer_layer:
        assert not conv_norm, "set only enc_conformer_layer or conv_norm, not both"
        assert isinstance(enc_conformer_layer, dict) and "class" in enc_conformer_layer
    else:
        enc_conformer_layer = rf.build_dict(
            rf.encoder.conformer.ConformerEncoderLayer,
            conv_norm=conv_norm or {"class": "rf.BatchNorm", "use_mask": True},
            self_att=rf.build_dict(
                rf.RelPosSelfAttention,
                # Shawn et al 2018 style, old RETURNN way.
                with_bias=False,
                with_linear_pos=False,
                with_pos_bias=False,
                learnable_pos_emb=True,
                separate_pos_emb_per_head=False,
            ),
            ff_activation=rf.build_dict(rf.relu_square),
            num_heads=8,
        )
    enc_other_opts = config.typed_value("enc_other_opts", None)

    return Model(
        in_dim,
        num_enc_layers=num_enc_layers,
        enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
        enc_conformer_layer=enc_conformer_layer,
        enc_other_opts=enc_other_opts,
        target_dim=target_dim,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        eos_idx=_get_eos_idx(target_dim),
        enc_aux_logits=enc_aux_logits or (),
    )


ctc_model_def: ModelDef[Model]
ctc_model_def.behavior_version = 21
ctc_model_def.backend = "torch"
ctc_model_def.batch_size_factor = _batch_size_factor


def _get_bos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def _get_eos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.eos_label_id is not None:
        eos_idx = target_dim.vocab.eos_label_id
    else:
        raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
    return eos_idx


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        target_dim: Dim,
        wb_target_dim: Optional[Dim] = None,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_aux_logits: Sequence[int] = (),  # layers
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_conformer_layer: Optional[Dict[str, Any]] = None,
        enc_other_opts: Optional[Dict[str, Any]] = None,
    ):
        super(Model, self).__init__()

        import numpy
        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        enc_layer_drop = config.float("enc_layer_drop", 0.0)
        if enc_layer_drop:
            enc_sequential = functools.partial(SequentialLayerDrop, layer_drop=enc_layer_drop)
        else:
            enc_sequential = rf.Sequential

        self.in_dim = in_dim
        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer=enc_conformer_layer,
            num_layers=num_enc_layers,
            sequential=enc_sequential,
            **(enc_other_opts or {}),
        )

        # Experiments without final layer norm. (We might clean this up when this is not successful.)
        # Just patch the encoder here.
        enc_conformer_final_layer_norm = config.typed_value("enc_conformer_final_layer_norm", None)
        if enc_conformer_final_layer_norm is None:
            pass
        elif enc_conformer_final_layer_norm == "last":  # only in the last, i.e. remove everywhere else
            for layer in self.encoder.layers[:-1]:
                layer: ConformerEncoderLayer
                layer.final_layer_norm = rf.identity
        else:
            raise ValueError(f"invalid enc_conformer_final_layer_norm {enc_conformer_final_layer_norm!r}")

        disable_encoder_self_attention = config.typed_value("disable_encoder_self_attention", None)
        if disable_encoder_self_attention is not None:
            # Disable self-attention in encoder.
            from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.model_ext.disable_self_att import apply_disable_self_attention_

            apply_disable_self_attention_(self.encoder, disable_encoder_self_attention)

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        if not wb_target_dim:
            wb_target_dim = target_dim + 1
        for i in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))
        self.enc_logits = rf.Linear(self.encoder.out_dim, wb_target_dim)
        self.wb_target_dim = wb_target_dim
        self.out_blank_separated = config.bool("out_blank_separated", False)

        if target_dim.vocab and not wb_target_dim.vocab:
            from returnn.datasets.util.vocabulary import Vocabulary

            # Just assumption for code now, might extend this later.
            assert wb_target_dim.dimension == target_dim.dimension + 1 and blank_idx == target_dim.dimension
            vocab_labels = list(target_dim.vocab.labels) + [OUT_BLANK_LABEL]
            wb_target_dim.vocab = Vocabulary.create_vocab_from_labels(
                vocab_labels, user_defined_symbols={OUT_BLANK_LABEL: blank_idx}
            )

        ctc_label_smoothing = config.float("ctc_label_smoothing", 0.0)
        ctc_label_smoothing_exclude_blank = config.bool("ctc_label_smoothing_exclude_blank", self.out_blank_separated)
        self.ctc_label_smoothing_exclude_blank = ctc_label_smoothing_exclude_blank
        if not self.out_blank_separated:
            self.ctc_label_smoothing_opts = {
                "smoothing": ctc_label_smoothing,
                "axis": self.wb_target_dim,
                "exclude_labels": [self.blank_idx] if ctc_label_smoothing_exclude_blank else None,
            }
        else:  # separate blank
            self.ctc_label_smoothing_opts = {
                "smoothing": ctc_label_smoothing,
                "axis": self.target_dim if ctc_label_smoothing_exclude_blank else self.wb_target_dim,
            }
        self.log_prob_normed_grad_opts = config.typed_value("log_prob_normed_grad", None)
        self.log_prob_normed_grad_exclude_blank = config.bool(
            "log_prob_normed_grad_exclude_blank", self.out_blank_separated
        )

        self.feature_batch_norm = None
        if config.bool("feature_batch_norm", False):
            self.feature_batch_norm = rf.BatchNorm(self.in_dim, affine=False, use_mask=True)
        self.feature_norm = config.bool("feature_norm", False)
        self.feature_stats = None
        feature_stats = config.typed_value("feature_stats")
        if feature_stats:
            assert isinstance(feature_stats, dict)
            self.feature_stats = rf.ParameterList(
                {
                    k: rf.Parameter(
                        rf.convert_to_tensor(numpy.loadtxt(v), dims=[self.in_dim], dtype=rf.get_default_float_dtype()),
                        auxiliary=True,
                    )
                    for k, v in feature_stats.items()
                }
            )

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
            "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
            or (_log_mel_feature_dim // 5),
            "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
        }

        self._mixup = None
        if config.typed_value("mixup", None) is not None:
            from i6_experiments.users.zeyer.returnn.models.rf_mixup import Mixup, MixupOpts

            self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))

        self.decoder = None
        aux_attention_decoder = config.typed_value("aux_attention_decoder", None)
        if aux_attention_decoder:
            assert isinstance(aux_attention_decoder, dict)
            aux_attention_decoder = aux_attention_decoder.copy()
            aux_attention_decoder.setdefault("class", "returnn.frontend.decoder.transformer.TransformerDecoder")
            if isinstance(aux_attention_decoder.get("model_dim", None), int):
                aux_attention_decoder["model_dim"] = Dim(aux_attention_decoder["model_dim"], name="dec_model")
            self.decoder = rf.build_from_dict(aux_attention_decoder, encoder_dim=enc_model_dim, vocab_dim=target_dim)

        vn = config.typed_value("variational_noise", None)
        if vn:
            # Use some blacklist. I think the same blacklist as for weight decay is reasonable.
            # Usually sth like: ["rf.Embedding", "rf.LearnedRelativePositionalEncoding"]
            blacklist = config.typed_value("optimizer")["weight_decay_modules_blacklist"]
            blacklist = tuple(eval(name, {"rf": rf}) for name in blacklist)
            for mod in self.modules():
                if isinstance(mod, blacklist):
                    continue
                for param_name, param in mod.named_parameters(recurse=False):
                    if param_name.endswith("bias"):  # no bias
                        continue
                    if param.auxiliary:
                        continue
                    rf.weight_noise(mod, param_name, std=vn)

        weight_dropout = config.typed_value("weight_dropout", None)
        if weight_dropout:
            # Use some blacklist. I think the same blacklist as for weight decay is reasonable.
            # Usually sth like: ["rf.Embedding", "rf.LearnedRelativePositionalEncoding"]
            blacklist = config.typed_value("optimizer")["weight_decay_modules_blacklist"]
            blacklist = tuple(eval(name, {"rf": rf}) for name in blacklist)
            for mod in self.modules():
                if isinstance(mod, blacklist):
                    continue
                for param_name, param in mod.named_parameters(recurse=False):
                    if param_name.endswith("bias"):  # no bias
                        continue
                    if param.auxiliary:
                        continue
                    rf.weight_dropout(mod, param_name, drop_prob=weight_dropout)

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Dim]:
        """encode, get logits"""
        # log mel filterbank features
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source,
            in_spatial_dim=in_spatial_dim,
            out_dim=self.in_dim,
            sampling_rate=16_000,
        )
        if self.feature_batch_norm:
            source = self.feature_batch_norm(source)
        if self.feature_norm:
            source = rf.normalize(source, axis=in_spatial_dim)
        if self.feature_stats:
            source = (source - self.feature_stats.mean) / self.feature_stats.std_dev
        if self._mixup:
            source = self._mixup(source, spatial_dim=in_spatial_dim)
        # SpecAugment
        source = rf.audio.specaugment(
            source,
            spatial_dim=in_spatial_dim,
            feature_dim=self.in_dim,
            **self._specaugment_opts,
        )
        # Encoder including convolutional frontend
        enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
        logits = self.enc_logits(enc)
        return logits, enc, enc_spatial_dim

    def log_probs_wb_from_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: incl blank
        :return: log probs with blank from logits (wb_target_dim)
            If out_blank_separated, we use a separate sigmoid for the blank.
        """
        if not self.out_blank_separated:  # standard case, joint distrib incl blank
            log_probs = rf.log_softmax(logits, axis=self.wb_target_dim)
        else:  # separate blank
            assert self.blank_idx == self.target_dim.dimension  # not implemented otherwise
            dummy_blank_feat_dim = Dim(1, name="blank_feat")
            logits_wo_blank, logits_blank = rf.split(
                logits, axis=self.wb_target_dim, out_dims=[self.target_dim, dummy_blank_feat_dim]
            )
            log_probs_wo_blank = rf.log_softmax(logits_wo_blank, axis=self.target_dim)
            log_probs_wo_blank = self._maybe_apply_on_log_probs(log_probs_wo_blank)
            log_probs_blank = rf.log_sigmoid(logits_blank)
            log_probs_emit = rf.squeeze(rf.log_sigmoid(-logits_blank), axis=dummy_blank_feat_dim)
            log_probs, _ = rf.concat(
                (log_probs_wo_blank + log_probs_emit, self.target_dim),
                (log_probs_blank, dummy_blank_feat_dim),
                out_dim=self.wb_target_dim,
            )
            log_probs.feature_dim = self.wb_target_dim
        log_probs = self._maybe_apply_on_log_probs(log_probs)
        return log_probs

    def _maybe_apply_on_log_probs(self, log_probs: Tensor) -> Tensor:
        """
        :param log_probs: either with blank or without blank
        :return: log probs, maybe some smoothing applied (all on gradients so far, not on log probs itself)
        """
        assert log_probs.feature_dim in (self.wb_target_dim, self.target_dim)
        if not self.out_blank_separated:
            assert log_probs.feature_dim == self.wb_target_dim

        log_probs = self._maybe_apply_log_probs_normed_grad(log_probs)

        if self.ctc_label_smoothing_exclude_blank:
            if self.out_blank_separated:
                if log_probs.feature_dim == self.target_dim:
                    log_probs = rf.label_smoothed_log_prob_gradient(log_probs, **self.ctc_label_smoothing_opts)
            else:
                assert log_probs.feature_dim == self.wb_target_dim
                assert self.ctc_label_smoothing_opts["exclude_labels"] == [self.blank_idx]
                log_probs = rf.label_smoothed_log_prob_gradient(log_probs, **self.ctc_label_smoothing_opts)
        else:
            if log_probs.feature_dim == self.wb_target_dim:
                log_probs = rf.label_smoothed_log_prob_gradient(log_probs, **self.ctc_label_smoothing_opts)

        return log_probs

    def _maybe_apply_log_probs_normed_grad(self, log_probs: Tensor) -> Tensor:
        if not self.log_prob_normed_grad_opts:
            return log_probs

        assert log_probs.feature_dim in (self.wb_target_dim, self.target_dim)
        if not self.out_blank_separated:
            assert log_probs.feature_dim == self.wb_target_dim
        if self.log_prob_normed_grad_exclude_blank:
            assert self.out_blank_separated
            if log_probs.feature_dim == self.wb_target_dim:
                return log_probs
        else:  # not excluded blank
            if log_probs.feature_dim == self.target_dim:
                return log_probs

        from alignments.util import normed_gradient, NormedGradientFuncInvPrior

        opts: Dict[str, Any] = self.log_prob_normed_grad_opts.copy()
        func_opts = opts.pop("func")
        assert isinstance(func_opts, dict)
        func_opts = func_opts.copy()
        assert func_opts.get("class", "inv_prior") == "inv_prior"  # only case for now
        func_opts.pop("class", None)
        func = NormedGradientFuncInvPrior(**func_opts)

        assert log_probs.batch_dim_axis is not None and log_probs.feature_dim_axis is not None
        log_probs_ = log_probs.copy_template()
        log_probs_.raw_tensor = normed_gradient(
            log_probs.raw_tensor,
            batch_axis=log_probs.batch_dim_axis,
            feat_axis=log_probs.feature_dim_axis,
            **opts,
            func=func,
        )
        return log_probs_
    
    
#---------------------------------------------------------------------------------------------------------------------------------------
# TRAINING DEFINITION

def ctc_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    if config.bool("use_eos_postfix", False):
        targets, (targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )

    collected_outputs = {}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
            aux_loss = rf.ctc_loss(
                logits=aux_log_probs,
                logits_normalized=True,
                targets=targets,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim,
                blank_index=model.blank_idx,
            )
            aux_loss.mark_as_loss(
                f"ctc_{layer_idx}",
                scale=aux_loss_scales[i],
                custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )

    log_probs = model.log_probs_wb_from_logits(logits)
    loss = rf.ctc_loss(
        logits=log_probs,
        logits_normalized=True,
        targets=targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx,
    )
    loss.mark_as_loss(
        "ctc",
        custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
        use_normalized_loss=use_normalized_loss,
    )

    if model.decoder:
        # potentially also other types but just assume
        # noinspection PyTypeChecker
        decoder: TransformerDecoder = model.decoder

        input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
        )
        targets_w_eos, _ = rf.pad(
            targets,
            axes=[targets_spatial_dim],
            padding=[(0, 1)],
            value=model.eos_idx,
            out_dims=[targets_w_eos_spatial_dim],
        )

        batch_dims = data.remaining_dims(data_spatial_dim)
        logits, _ = model.decoder(
            input_labels,
            spatial_dim=targets_w_eos_spatial_dim,
            encoder=decoder.transform_encoder(enc, axis=enc_spatial_dim),
            state=model.decoder.default_initial_state(batch_dims=batch_dims),
        )

        logits_packed, pack_dim = rf.pack_padded(
            logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
        )
        targets_packed, _ = rf.pack_padded(
            targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
        )

        log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
        log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
        loss = rf.cross_entropy(
            target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
        )
        loss.mark_as_loss("aed_ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

        best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
        frame_error = best != targets_packed
        frame_error.mark_as_loss(name="aed_fer", as_error=True)


ctc_training: TrainDef[Model]
ctc_training.learning_rate_control_error_measure = "ctc"

def ctc_sum_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, lm_path: tk.Path):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config
    from .sum_criterion import sum_loss, safe_logsumexp
    
    # torch.autograd.set_detect_anomaly(True)
    
    def _calc_log_prior(log_probs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.to(log_probs.device)
        assert lengths.size(0) == log_probs.size(0), "Prior calculation batch lengths are not the same (full_sum)!"
        
        mask = torch.arange(log_probs.size(1), device=log_probs.device).expand(log_probs.size(0), -1) < lengths.unsqueeze(1)
        mask = torch.where(mask, 0.0, float("-inf"))
        mask = mask.unsqueeze(-1).expand(-1, -1, log_probs.size(2))
        log_probs = log_probs + mask
        
        sum_frames = lengths.sum()
        log_sum_probs = torch.full([log_probs.size(2) + 1,], float("-inf"), device=log_probs.device)
        log_sum_probs[1:-1] = safe_logsumexp(safe_logsumexp(log_probs[:,:,1:], dim=0), dim=0) # Sum over batch and time
        log_sum_probs[0] = safe_logsumexp(log_probs[:,0,0], dim=0) # BOS prob
        log_sum_probs[-1] = safe_logsumexp(safe_logsumexp(log_probs[:,1:,0], dim=0), dim=0) # EOS prob
        
        log_mean_probs = log_sum_probs - sum_frames.log()
        
        with torch.no_grad():
            assert log_mean_probs.exp().sum().allclose(torch.tensor(1.0, device=log_mean_probs.device)), f"Prior probs do not sum to 1.0, but to {log_mean_probs.exp().sum()}"
            if log_mean_probs.isclose(torch.tensor([0.0], device=log_probs.device)).any() or log_mean_probs.isinf().any() or log_mean_probs.isnan().any():
                print("Prior probs contain inf or nan or 0 values!", log_mean_probs, log_mean_probs.exp())
        
        return log_mean_probs

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    
    am_scale = config.float("am_scale", 1.0)
    lm_scale = config.float("lm_scale", 1.0)
    
    horizontal_prior = config.float("horizontal_prior", True)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio
    
    with uopen(lm_path, "rb") as f:
        lm = torch.load(f, map_location=data.device)
        assert isinstance(lm, torch.Tensor), "Loaded LM is not a tensor"

    collected_outputs = {}
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
            aux_log_probs = aux_log_probs.raw_tensor
            aux_log_prior = _calc_log_prior(aux_log_probs, enc_spatial_dim.dyn_size_ext.raw_tensor)
            # (B, T, F) -> (T, B, F)
            aux_log_probs = aux_log_probs.permute(1, 0, 2)
            aux_loss = sum_loss(
                log_probs=aux_log_probs,
                log_lm_probs=lm,
                log_prior=aux_log_prior,
                input_lengths=enc_spatial_dim.dyn_size_ext.raw_tensor,
                LM_order=2,
                am_scale=am_scale,
                lm_scale=lm_scale,
                horizontal_prior=horizontal_prior,
                blank_idx=model.blank_idx,
                eos_idx=model.eos_idx,
                unk_idx=1
            )
            aux_loss = rtf.TorchBackend.convert_to_tensor(aux_loss, dims = [batch_dim], dtype = "float32", name=f"aux_full_sum_{layer_idx}")
            aux_loss.mark_as_loss(
                f"aux_full_sum_{layer_idx}",
                scale=aux_loss_scales[i],
                custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )

    log_probs = model.log_probs_wb_from_logits(logits)
    log_probs = log_probs.raw_tensor
    log_prior = _calc_log_prior(log_probs, enc_spatial_dim.dyn_size_ext.raw_tensor)
    # (B, T, F) -> (T, B, F)
    log_probs = log_probs.permute(1, 0, 2)
    loss = sum_loss(
        log_probs=log_probs,
        log_lm_probs=lm,
        log_prior=log_prior,
        input_lengths=enc_spatial_dim.dyn_size_ext.raw_tensor,
        LM_order=2,
        am_scale=am_scale,
        lm_scale=lm_scale,
        horizontal_prior=horizontal_prior,
        blank_idx=model.blank_idx,
        eos_idx=model.eos_idx,
        unk_idx=1,
        device=log_probs.device
    )
    loss = rtf.TorchBackend.convert_to_tensor(loss, dims = [batch_dim], dtype = "float32", name=f"full_sum")
    loss.mark_as_loss(
        f"full_sum",
        custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(),
        use_normalized_loss=use_normalized_loss,
    )

ctc_sum_training: ExtendedTrainDef[Model]
ctc_sum_training.learning_rate_control_error_measure = "full_sum"


#---------------------------------------------------------------------------------------------------------------------------------------
# RECOG DEFINITION

def model_recog(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, Vocab
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob_pre_filter, (backrefs_pre_filter,), pre_filter_beam_dim = rf.top_k(
        label_log_prob, k_dim=Dim(beam_size, name=f"pre-filter-beam"), axis=[model.wb_target_dim]
    )  # seq_log_prob, backrefs_global: Batch, Spatial, PreFilterBeam. backrefs_pre_filter -> Vocab
    label_log_prob_pre_filter_ta = TensorArray.unstack(
        label_log_prob_pre_filter, axis=enc_spatial_dim
    )  # t -> Batch, PreFilterBeam
    backrefs_pre_filter_ta = TensorArray.unstack(backrefs_pre_filter, axis=enc_spatial_dim)  # t -> Batch, PreFilterBeam

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets = []
    seq_backrefs = []
    for t in range(max_seq_len):
        # Filter out finished beams
        seq_log_prob = seq_log_prob + label_log_prob_pre_filter_ta[t]  # Batch, InBeam, PreFilterBeam
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, pre_filter_beam_dim]
        )  # seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> PreFilterBeam.
        target = rf.gather(backrefs_pre_filter_ta[t], indices=target)  # Batch, Beam -> Vocab
        seq_targets.append(target)
        seq_backrefs.append(backrefs)

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = enc_spatial_dim
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = OUT_BLANK_LABEL
model_recog.batch_size_dependent = False  # not totally correct, but we treat it as such...

def model_recog_lm(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    arpa_4gram_lm: str,
    lexicon: str,
    hyperparameters: dict,
    prior_file: tk.Path = None
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.
    
    Uses a 4gram LM and beam search.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from torchaudio.models.decoder import ctc_decoder
    import torch
    from returnn.util.basic import cf
    
    # Get the logits from the model
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    
    hyp_params = copy.copy(hyperparameters)
    greedy = hyp_params.pop("greedy", False)
    prior_weight = hyp_params.pop("prior_weight", 0.0)
    use_logsoftmax = hyp_params.pop("use_logsoftmax", False)
    
    if greedy:
        use_logsoftmax = True
    
    if use_logsoftmax:
        label_log_prob = model.log_probs_wb_from_logits(logits)
        label_log_prob = label_log_prob.raw_tensor.cpu()
    
        # Subtract prior of labels if available
        if prior_file and prior_weight > 0.0:
            prior = np.loadtxt(prior_file, dtype="float32")
            label_log_prob -= prior_weight * prior
            print("We subtracted the prior!")
    elif prior_file and prior_weight > 0.0:
        print("Cannot subtract prior without running log softmax")
        return None
    
    if greedy:
        probs, greedy_res = torch.max(label_log_prob, dim=-1)
        greedy_res = greedy_res.unsqueeze(1)
        
        scores = torch.sum(probs, dim=-1)
        scores = scores.unsqueeze(1)
        
        beam_dim = rtf.TorchBackend.get_new_dim_raw(greedy_res, 1, name="beam_dim")
        dims = [batch_dim, beam_dim, enc_spatial_dim]
        hyps = rtf.TorchBackend.convert_to_tensor(greedy_res, dims = dims, sparse_dim=model.wb_target_dim, dtype = "int64", name="hyps")
        
        dims = [batch_dim, beam_dim]
        scores = Tensor("scores", dims = dims, dtype = "float32", raw_tensor = scores)
        
        return hyps, scores, enc_spatial_dim, beam_dim
    
    arpa_4gram_lm = str(cf(arpa_4gram_lm))
    
    use_lm = hyp_params.pop("use_lm", True)
    use_lexicon = hyp_params.pop("use_lexicon", True)
    
    configs = {
        "tokens": list(model.wb_target_dim.vocab.labels),
        "blank_token": OUT_BLANK_LABEL,
        "sil_token": OUT_BLANK_LABEL,
        "unk_word": "<unk>",
        "beam_size_token": None, # 16
        "beam_threshold": 1000000, # 14
    }
    configs["lexicon"] = lexicon if use_lexicon else None
    configs["lm"] = arpa_4gram_lm if use_lm else None
    
    configs.update(hyp_params)
    
    decoder = ctc_decoder(**configs)
    enc_spatial_dim_torch = enc_spatial_dim.dyn_size_ext.raw_tensor.cpu()
    if use_logsoftmax:
        decoder_results = decoder(label_log_prob, enc_spatial_dim_torch)
    else:
        decoder_results = decoder(logits.raw_tensor.cpu(), enc_spatial_dim_torch)
    
    if use_lexicon:
        print("Use words directly!")
        if CHECK_DECODER_CONSISTENCY:
            for l1 in decoder_results:
                for l2 in l1:
                    lexicon_words = " ".join(l2.words)
                    token_words = " ".join([configs["tokens"][t] for t in l2.tokens])
                    assert not token_words.endswith("@@"), f"Token words ends with @@: {token_words}, Lexicon words: {lexicon_words}"
                    token_words = token_words.replace("@@ ", "")
                    assert lexicon_words == token_words, f"Words don't match: Lexicon words: {lexicon_words}, Token words: {token_words}"
        
        words = [[" ".join(l2.words) for l2 in l1] for l1 in decoder_results]
        words = np.array(words)
        words = np.expand_dims(words, axis=2)
        scores = [[l2.score for l2 in l1] for l1 in decoder_results]
        scores = torch.tensor(scores)
        
        beam_dim = Dim(words.shape[1], name="beam_dim")
        enc_spatial_dim = Dim(1, name="spatial_dim")
        words = rf._numpy_backend.NumpyBackend.convert_to_tensor(words, dims = [batch_dim, beam_dim, enc_spatial_dim], dtype = "string", name="hyps")
        scores = Tensor("scores", dims = [batch_dim, beam_dim], dtype = "float32", raw_tensor = scores)
        
        return words, scores, enc_spatial_dim, beam_dim
    else:
        def _pad_blanks(tokens, max_len):
            if len(tokens) < max_len:
                # print("We had to pad blanks")
                tokens = torch.cat([tokens, torch.tensor([model.blank_idx] * (max_len - len(tokens)))])
            return tokens
        
        def _pad_lists(t, max_len, max_len2):
            if t.shape[0] < max_len2:
                print("We had to pad the list")
                t = torch.cat([t, torch.tensor([[model.blank_idx] * max_len] * (max_len2 - t.shape[0]))])
            return t
        
        def _pad_scores(l, max_len):
            l = torch.tensor(l)
            if len(l) < max_len:
                print("We had to pad scores")
                l = torch.cat([l, torch.tensor([-1000000.0] * (max_len - len(l)))])
            return l
            
        max_length = int(enc_spatial_dim_torch.max())
        hyps = [torch.stack([_pad_blanks(l2.tokens, max_length) for l2 in l1]) for l1 in decoder_results]
        max_length_2 = max([l.shape[0] for l in hyps])
        hyps = [_pad_lists(t, max_length, max_length_2) for t in hyps]
        hyps = torch.stack(hyps)
        beam_dim = rtf.TorchBackend.get_new_dim_raw(hyps, 1, name="beam_dim")
        dims = [batch_dim, beam_dim, enc_spatial_dim]
        hyps = rtf.TorchBackend.convert_to_tensor(hyps, dims = dims, sparse_dim=model.wb_target_dim, dtype = "int64", name="hyps")
        
        scores = [[l2.score for l2 in l1] for l1 in decoder_results]
        max_length_3 = max([len(l) for l in scores])
        scores = torch.stack([_pad_scores(l, max_length_3) for l in scores])
        dims = [batch_dim, beam_dim]
        scores = Tensor("scores", dims = dims, dtype = "float32", raw_tensor = scores)
        
        # print(f"CUSTOM seq_targets: {hyps} \n{hyps.raw_tensor.cpu()},\nscores: {scores} \n{scores.raw_tensor.cpu()}n {scores.raw_tensor.cpu()[0][0]},\nspatial_dim: {enc_spatial_dim.dyn_size_ext.raw_tensor.cpu()},\n beam_size: {beam_dim}")
        
        return hyps, scores, enc_spatial_dim, beam_dim

# RecogDef API
model_recog_lm: RecogDef[Model]
model_recog_lm.output_with_beam = True
model_recog_lm.output_blank_label = OUT_BLANK_LABEL
model_recog_lm.batch_size_dependent = False  # not totally correct, but we treat it as such...