"""Param Import
"""

from __future__ import annotations

import os.path
from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection
import tree
import math
import numpy as np
import hashlib
import contextlib
import functools
import types

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.zeyer.model_interfaces.supports_label_scorer_torch import RFModelWithMakeLabelScorer
from .configs import *
from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.configs import (
    _batch_size_factor,
    _cfg_lrlin1e_5_295k,
    _get_cfg_lrlin_oclr_by_bs_nep,
)

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, ModelWithCheckpoint

from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg
from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset
from i6_experiments.users.zeyer.train_v3 import train

# From Mohammad, 2023-06-29
# dev-clean  2.27
# dev-other  5.39
# test-clean  2.41
# test-other  5.51
# _returnn_tf_config_filename = (
#   "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.1oORPHJTAcW0/output/returnn.config")
# E.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
# /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb
# Main train (2035 subepochs): /work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.SAh74CLCNJQi
# 15k batch size, accum grad 2 (1350 steps per epoch?)
# (With batch size 40k (here default), I have usually 495 steps/epoch. Same accum grad 2.)
# peak_lr = 0.9e-3 (1e-3 should also be fine), with Adam, optimizer_epsilon = 1e-08
# phase1: peak_lr / 10 -> peak_lr (45%)
# phase2: peak_lr -> peak_lr / 10 (45%)
# phase3: peak_lr / 10 -> 1e-6 (10%)
# all linear decay and step-based
# specaugment like my orig (same here, specaugorig), speed perturb same here.
# weight decay: L2 1e-4 in some layers (not all): FF, depthwise conv, self-att, output, LSTM, readout
# Final from learning_rates file:
# 2035: EpochData(learningRate=<misleading>, error={
# 'dev_error_ctc': 0.0520755184693418,
# 'dev_error_output/output_prob': 0.035661241551042944,
# 'dev_score_ctc': 0.2796084385705723,
# 'dev_score_output/output_prob': 0.1718613621694714,
# 'devtrain_error_ctc': 0.005757552549708462,
# 'devtrain_error_output/output_prob': 0.005408351877314902,
# 'devtrain_score_ctc': 0.022935187616968285,
# 'devtrain_score_output/output_prob': 0.05237826015574962,
# 'train_error_ctc': 0.05592114304093772,
# 'train_error_output/output_prob': 0.041970552995693494,
# 'train_score_ctc': 0.21249712733341475,
# 'train_score_output/output_prob': 0.20816428663741796,
# }),
# Retrain RETURNN training job (600 subepochs): /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.ZhtaEElHqWlr
# Epoch-wise LR:
# Fixed for 20 subepochs: 1e-4
# Linear(?) decay for remaining (?): 1e-4 to 1e-6
# Final from learning_rates file:
# 600: EpochData(learningRate=1e-06, error={
# 'dev_error_ctc': 0.04999311020358747,
# 'dev_error_output/output_prob': 0.03406881170076022,
# 'dev_score_ctc': 0.2881619431223589,
# 'dev_score_output/output_prob': 0.16851828029171323,
# 'devtrain_error_ctc': 0.003611245977923651,
# 'devtrain_error_output/output_prob': 0.004028583366881808,
# 'devtrain_score_ctc': 0.014762402644778178,
# 'devtrain_score_output/output_prob': 0.0458638666428664,
# 'train_error_ctc': 0.051649620746772214,
# 'train_error_output/output_prob': 0.03977601830532325,
# 'train_score_ctc': 0.19722012590584306,
# 'train_score_output/output_prob': 0.19768974342596793,
# }),


# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    _sis_setup_global_prefix(os.path.basename(__file__)[:-3])

    # Moh:      dev-clean  2.27, dev-other  5.39, test-clean  2.41,  test-other  5.51
    # RF recog: {"dev-clean": 2.25, "dev-other": 5.34, "test-clean": 2.42, "test-other": 5.56}
    _recog_imported()

    lbs_bpe10k_dataset = get_librispeech_lm_dataset(vocab="bpe10k")

    # monkey patching to use only transcription text data for training
    def get_train_trans_dataset(self):
        return self.get_dataset("transcriptions-train", training=True)

    lbs_bpe10k_dataset.get_train_dataset = types.MethodType(get_train_trans_dataset, lbs_bpe10k_dataset)

    preload_from_files = {
        "aed": {"filename": get_tf_to_rf_converted_ckpt_path(), "init_for_train": True, "ignore_missing": True}
    }

    # train Mini-LSTM ILM
    ilm_with_checkpoints = train(
        f"ilm/mini-lstm-d50",
        config=dict_update_deep(
            config_11gb_ilm_v1,
            {"batch_size": 10_000, "preload_from_files": preload_from_files, "__num_epochs": 40},
        ),
        train_dataset=lbs_bpe10k_dataset,
        model_def=ModelDefWithCfg(
            mini_lstm_ilm_def,
            {"_model_def_dict": rf.build_dict(MiniLstmIlm)},
        ),
        train_def=mini_lstm_ilm_train_def,
    )

    from . import trafo_lm_kazuki_import

    for lm_scale in [0.0, 0.42]:
        lm_recog_config = {
            "beam_search_opts": {"beam_size": 32, "lm_scale": lm_scale},
            "external_language_model": {"class": "TransformerDecoder", **trafo_lm_kazuki_import.TrafoLmOpts},
            "preload_from_files": {
                "trafo_lm": {"prefix": "language_model.", "filename": trafo_lm_kazuki_import.get_pt_checkpoint_path()}
            },
            "batch_size": 4000 * 160,
        }
        _recog_imported(name=f"trafo_lm_{lm_scale}_beam32", recog_config=lm_recog_config)

    for lm_scale in [0.54]:
        for ilm_scale in [0.4]:
            ilm_recog_config = {
                "beam_search_opts": {
                    "beam_size": 70,
                    "lm_scale": lm_scale,
                    "ilm_scale": ilm_scale,
                },
                "external_language_model": {"class": "TransformerDecoder", **trafo_lm_kazuki_import.TrafoLmOpts},
                "internal_language_model": {"class": "MiniLstmIlm"},
                "preload_from_files": {
                    "trafo_lm": {
                        "prefix": "language_model.",
                        "filename": trafo_lm_kazuki_import.get_pt_checkpoint_path(),
                    },
                    "ilm": {
                        "prefix": "internal_language_model.",
                        "filename": ilm_with_checkpoints.get_last_fixed_epoch().checkpoint,  # TODO: not optimal!
                    },
                },
                "batch_size": 1000 * 160,
                "max_seqs": 1,
            }
            _recog_imported(name=f"trafo_lm_{lm_scale}_ilm_{ilm_scale}_beam70", recog_config=ilm_recog_config)


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.sis_setup import get_prefix_for_config

        prefix_name = get_prefix_for_config(__file__)
    global _sis_prefix
    _sis_prefix = prefix_name


def get_tf_to_rf_converted_ckpt_path() -> tk.Path:
    from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import ConvertTfCheckpointToRfPtJob
    from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf._moh_att_2023_06_30_import import map_param_func_v2
    from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
    from i6_core.returnn.training import Checkpoint as TfCheckpoint, PtCheckpoint

    task = _get_ls_task()
    extern_data_dict = task.train_dataset.get_extern_data()
    default_target_key = task.train_dataset.get_default_target()
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    target_dim = targets.feature_dim_or_sparse_dim

    new_chkpt_path = ConvertTfCheckpointToRfPtJob(
        checkpoint=TfCheckpoint(index_path=generic_job_output(_returnn_tf_ckpt_filename)),
        make_model_func=MakeModel(
            in_dim=_log_mel_feature_dim,
            target_dim=target_dim.dimension,
            eos_label=_get_eos_idx(target_dim),
        ),
        map_func=map_param_func_v2,
    ).out_checkpoint

    return new_chkpt_path


def _recog_imported(name: str = "recog_resutls", recog_config: Optional[Dict[str, Any]] = None):
    from i6_core.returnn.training import PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint

    new_chkpt_path = get_tf_to_rf_converted_ckpt_path()
    new_chkpt = PtCheckpoint(new_chkpt_path)

    model_with_checkpoint = ModelWithCheckpoint(definition=from_scratch_model_def, checkpoint=new_chkpt)

    _recog(name, model_with_checkpoint, recog_config=recog_config)


def _recog(
    name: str,
    model_with_checkpoint: ModelWithCheckpoint,
    recog_def: RecogDef = None,
    recog_config: Optional[Dict[str, Any]] = None,
    *,
    search_rqmt: Optional[Dict[str, Any]] = None,
    dev_sets: Optional[Collection[str]] = None,
):
    from sisyphus import tk
    from i6_experiments.users.zeyer.recog import recog_model

    if recog_def is None:
        recog_def = model_recog

    task = _get_ls_task()

    res = recog_model(
        task,
        model_with_checkpoint,
        recog_def=recog_def,
        config=recog_config,
        search_rqmt=search_rqmt,
        dev_sets=dev_sets,
    )
    tk.register_output(_sis_prefix + "/" + name, res.output)


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    *,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    fine_tune: Optional[Union[int, List[Tuple[int, Dict[str, Any]]]]] = None,
    time_rqmt: Optional[int] = None,
    model_avg: bool = False,
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    from .train import train
    from i6_experiments.users.zeyer.recog import recog_training_exp

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    task = _get_ls_task()
    config = config.copy()
    config = dict_update_deep(config, config_updates, config_deletes)
    if "__num_epochs" in config:
        num_epochs = config.pop("__num_epochs")
    if "__gpu_mem" in config:
        gpu_mem = config.pop("__gpu_mem")
    if "__num_processes" in config:
        num_processes = config.pop("__num_processes")

    model_with_checkpoint = train(
        prefix,
        task=task,
        config=config,
        post_config=dict_update_deep(post_config, post_config_updates),
        model_def=from_scratch_model_def,
        train_def=from_scratch_training,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        distributed_launch_cmd="torchrun" if num_processes else "mpirun",
        time_rqmt=time_rqmt,
    )
    recog_training_exp(prefix, task, model_with_checkpoint, recog_def=model_recog, model_avg=model_avg)

    if fine_tune:
        if isinstance(fine_tune, int):
            fine_tune = [(fine_tune, {})]
        for ep, opts in fine_tune:
            assert isinstance(ep, int) and isinstance(opts, dict)
            suffix = f"/finetune/{ep}"
            opts = opts.copy()
            if opts:
                for k, v in sorted(opts.items()):
                    k: str
                    suffix += "-" + k.lstrip("_")
                    v = str(v).replace("-", "_")
                    if len(v) > 16 and not k.startswith("_"):
                        suffix += "_" + hashlib.md5(v.encode("utf8")).hexdigest()[:8]
                    else:
                        suffix += v
            num_epochs_ = opts.pop("num_epochs", 50)
            config_ = config.copy()
            config_["import_model_train_epoch1"] = model_with_checkpoint.get_epoch(ep).checkpoint
            config_.pop("dynamic_learning_rate")
            lrs = opts.pop("learning_rates", None)
            if lrs is None:
                lr_decay_type = opts.pop("lr_decay_type", "geomspace")  # geomspace or linspace
                lr_decay_func = getattr(np, lr_decay_type)
                lr = config_["learning_rate"]
                final_lr = opts.pop("final_lr", 1e-7)
                lrs = list(lr_decay_func(lr, final_lr, num=num_epochs_))
            else:
                assert isinstance(lrs, (list, tuple))
                assert len(lrs) == num_epochs_
            config_["learning_rates"] = lrs
            config_["learning_rate"] = float(lrs[-1])
            config_["specaugment_steps"] = (0, 0, 0)
            config_.update({k: v for k, v in opts.items() if not k.startswith("_")})

            finetune_model_with_ckpt = train(
                prefix + suffix,
                task=task,
                config=config_,
                post_config=post_config,
                model_def=from_scratch_model_def,
                train_def=from_scratch_training,
                num_epochs=num_epochs_,
                gpu_mem=gpu_mem,
            )
            # _recog(name + suffix + "/recog/last", finetune_model_with_ckpt.get_last_fixed_epoch())
            recog_training_exp(prefix + suffix, task, finetune_model_with_ckpt, recog_def=model_recog)

    return model_with_checkpoint


_ls_task = None


def _get_ls_task():
    global _ls_task
    if _ls_task:
        return _ls_task

    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_bpe10k_raw

    _ls_task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)
    return _ls_task


py = sis_run_with_prefix  # if run directly via `sis m ...`


class MakeModel:
    """for import"""

    def __init__(self, in_dim: int, target_dim: int, *, eos_label: int = 0, num_enc_layers: int = 12):
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.eos_label = eos_label
        self.num_enc_layers = num_enc_layers

    def __call__(self) -> Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(name="target", dimension=self.target_dim, kind=Dim.Types.Feature)
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        return self.make_model(in_dim, target_dim, num_enc_layers=self.num_enc_layers)

    @classmethod
    def make_model(
        cls,
        in_dim: Dim,
        target_dim: Dim,
        *,
        num_enc_layers: int = 12,
        pos_emb_dropout: float = 0.0,
        language_model: Optional[Dict[str, Any]] = None,
        internal_language_model: Optional[Dict[str, Any]] = None,
        **extra,
    ) -> Model:
        """make"""

        target_embed_dim = Dim(name="target_embed", dimension=640)
        att_num_heads = Dim(name="att_num_heads", dimension=1)
        enc_model_dim = Dim(name="enc", dimension=512, kind=Dim.Types.Feature)

        lm = None
        if language_model:
            assert isinstance(language_model, dict)
            language_model = language_model.copy()
            cls_name = language_model.pop("class")
            assert cls_name == "TransformerDecoder"
            language_model.pop("vocab_dim", None)  # will just overwrite

            from . import trafo_lm

            lm = trafo_lm.MakeModel(vocab_dim=target_dim, **language_model)()
            lm = (lm, functools.partial(trafo_lm.make_label_scorer_torch, model=lm))

        ilm = None
        if internal_language_model:
            assert isinstance(internal_language_model, dict)
            internal_language_model = internal_language_model.copy()
            cls_name = internal_language_model.pop("class")
            assert cls_name == "MiniLstmIlm"
            internal_language_model.pop("vocab_dim", None)  # will just overwrite
            ilm = MiniLstmIlm(
                target_dim=target_dim,
                target_embed_dim=target_embed_dim,
                att_context_dim=enc_model_dim,
                att_num_heads=att_num_heads,
                **internal_language_model,
            )

        return Model(
            in_dim,
            num_enc_layers=num_enc_layers,
            enc_model_dim=enc_model_dim,
            enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
            enc_att_num_heads=8,
            enc_conformer_layer_opts=dict(
                conv_norm_opts=dict(use_mask=True),
                self_att_opts=dict(
                    # Shawn et al 2018 style, old RETURNN way.
                    with_bias=False,
                    with_linear_pos=False,
                    with_pos_bias=False,
                    learnable_pos_emb=True,
                    separate_pos_emb_per_head=False,
                    pos_emb_dropout=pos_emb_dropout,
                ),
                ff_activation=lambda x: rf.relu(x) ** 2.0,
            ),
            att_num_heads=att_num_heads,
            target_dim=target_dim,
            target_embed_dim=target_embed_dim,
            blank_idx=target_dim.dimension,
            bos_idx=_get_bos_idx(target_dim),
            eos_idx=_get_eos_idx(target_dim),
            language_model=lm,
            internal_language_model=ilm,
            **extra,
        )


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        target_dim: Dim,
        target_embed_dim: Dim,
        wb_target_dim: Optional[Dim] = None,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_model_dim: Dim,
        att_num_heads: Dim,
        enc_aux_logits: Sequence[int] = (),  # layers
        enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
        enc_att_num_heads: int = 4,
        enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        att_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
        l2: float = 0.0001,
        language_model: Optional[RFModelWithMakeLabelScorer] = None,
        internal_language_model: Optional[MiniLstmIlm] = None,
    ):
        super(Model, self).__init__()

        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        self.in_dim = in_dim
        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
        )

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

        # https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/base2.conv2l.specaug4a.ctc.devtrain.config

        self.enc_ctx = rf.Linear(self.encoder.out_dim, enc_key_total_dim)
        self.enc_ctx_dropout = 0.2
        self.enc_win_dim = Dim(name="enc_win_dim", dimension=5)

        self.inv_fertility = rf.Linear(self.encoder.out_dim, att_num_heads, with_bias=False)

        self.target_embed = rf.Embedding(target_dim, target_embed_dim)
        if config.float("embed_init_stddev", None):
            self.target_embed.weight.initial = rf.init.Normal(stddev=config.float("embed_init_stddev", 0.0))

        self.s = rf.ZoneoutLSTM(
            self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
            Dim(name="lstm", dimension=1024),
            zoneout_factor_cell=0.15,
            zoneout_factor_output=0.05,
            use_zoneout_output=False,  # like RETURNN/TF ZoneoutLSTM old default
            # parts_order="icfo",  # like RETURNN/TF ZoneoutLSTM
            # parts_order="ifco",
            parts_order="jifo",  # NativeLSTM (the code above converts it...)
            forget_bias=0.0,  # the code above already adds it during conversion
        )

        self.weight_feedback = rf.Linear(att_num_heads, enc_key_total_dim, with_bias=False)
        self.s_transformed = rf.Linear(self.s.out_dim, enc_key_total_dim, with_bias=False)
        self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)
        self.readout_in = rf.Linear(
            self.s.out_dim + self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
            Dim(name="readout", dimension=1024),
        )
        self.output_prob = rf.Linear(self.readout_in.out_dim // 2, target_dim)

        for p in self.parameters():
            p.weight_decay = l2

        if enc_aux_logits:
            if not wb_target_dim:
                wb_target_dim = target_dim + 1
        for i in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
            "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
            or (_log_mel_feature_dim // 5),
            "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
        }

        self._pretrain_opts: Optional[Dict[str, Any]] = config.typed_value("pretrain_opts")

        self._mixup = None
        if config.typed_value("mixup", None) is not None:
            from i6_experiments.users.zeyer.returnn.models.rf_mixup import Mixup, MixupOpts

            self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))

        # Note: Even though we have this here, it is not used in loop_step or decode_logits.
        # Instead, it is intended to make a separate label scorer for it.
        self.language_model = None
        self.language_model_make_label_scorer = None
        if language_model:
            self.language_model, self.language_model_make_label_scorer = language_model

        self.internal_language_model = None
        if internal_language_model:
            self.internal_language_model = internal_language_model

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        # log mel filterbank features
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source,
            in_spatial_dim=in_spatial_dim,
            out_dim=self.in_dim,
            sampling_rate=16_000,
            log_base=math.exp(2.3026),  # almost 10.0 but not exactly...
        )
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
        with _opt_apply_pretrain_to_encoder(self.encoder, collected_outputs, self._pretrain_opts):
            enc, enc_spatial_dim = self.encoder(
                source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
            )
        enc_ctx = self.enc_ctx(enc)
        inv_fertility = rf.sigmoid(self.inv_fertility(enc))
        return dict(enc=enc, enc_ctx=enc_ctx, inv_fertility=inv_fertility), enc_spatial_dim

    def decoder_default_initial_state(self, *, batch_dims: Sequence[Dim], enc_spatial_dim: Dim) -> rf.State:
        """Default initial state"""
        state = rf.State(
            s=self.s.default_initial_state(batch_dims=batch_dims),
            att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.encoder.out_dim]),
            accum_att_weights=rf.zeros(
                list(batch_dims) + [enc_spatial_dim, self.att_num_heads], feature_dim=self.att_num_heads
            ),
        )
        state.att.feature_dim_axis = len(state.att.dims) - 1
        return state

    def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
        """loop step out"""
        return {
            "s": Tensor(
                "s", dims=batch_dims + [self.s.out_dim], dtype=rf.get_default_float_dtype(), feature_dim_axis=-1
            ),
            "att": Tensor(
                "att",
                dims=batch_dims + [self.att_num_heads * self.encoder.out_dim],
                dtype=rf.get_default_float_dtype(),
                feature_dim_axis=-1,
            ),
        }

    def loop_step(
        self,
        *,
        enc: rf.Tensor,
        enc_ctx: rf.Tensor,
        inv_fertility: rf.Tensor,
        enc_spatial_dim: Dim,
        input_embed: rf.Tensor,
        state: Optional[rf.State] = None,
    ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
        """step of the inner loop"""
        if state is None:
            batch_dims = enc.remaining_dims(
                remove=(enc.feature_dim, enc_spatial_dim) if enc_spatial_dim != single_step_dim else (enc.feature_dim,)
            )
            state = self.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim)
        state_ = rf.State()

        prev_att = state.att

        s, state_.s = self.s(rf.concat_features(input_embed, prev_att), state=state.s, spatial_dim=single_step_dim)

        weight_feedback = self.weight_feedback(state.accum_att_weights)
        s_transformed = self.s_transformed(s)
        energy_in = enc_ctx + weight_feedback + s_transformed
        energy = self.energy(rf.tanh(energy_in))
        att_weights = rf.softmax(energy, axis=enc_spatial_dim)
        state_.accum_att_weights = state.accum_att_weights + att_weights * inv_fertility * 0.5
        att0 = rf.dot(att_weights, enc, reduce=enc_spatial_dim, use_mask=False)
        att0.feature_dim = self.encoder.out_dim
        att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.encoder.out_dim))
        state_.att = att

        return {"s": s, "att": att}, state_

    def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
        """logits for the decoder"""
        readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
        readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
        readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
        logits = self.output_prob(readout)
        return logits


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


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    lm_opts = config.typed_value("external_language_model")
    ilm_opts = config.typed_value("internal_language_model")
    return MakeModel.make_model(
        in_dim,
        target_dim,
        enc_aux_logits=enc_aux_logits or (),
        pos_emb_dropout=pos_emb_dropout,
        language_model=lm_opts,
        internal_language_model=ilm_opts,
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


def from_scratch_training(
    *, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim
):
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

    collected_outputs = {}
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_loss = rf.ctc_loss(
                logits=aux_logits,
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
            # decoded, decoded_spatial_dim = rf.ctc_greedy_decode(aux_logits, in_spatial_dim=enc_spatial_dim)
            # error = rf.edit_distance(
            #     a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_spatial_dim
            # )
            # error.mark_as_loss("label", as_error=True, custom_inv_norm_factor=targets_spatial_dim.get_size_tensor())

    batch_dims = data.remaining_dims(data_spatial_dim)
    input_embeddings = model.target_embed(targets)
    input_embeddings = rf.shift_right(input_embeddings, axis=targets_spatial_dim, pad_value=0.0)

    def _body(input_embed: Tensor, state: rf.State):
        new_state = rf.State()
        loop_out_, new_state.decoder = model.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=state.decoder,
        )
        return loop_out_, new_state

    loop_out, _, _ = rf.scan(
        spatial_dim=targets_spatial_dim,
        xs=input_embeddings,
        ys=model.loop_step_output_templates(batch_dims=batch_dims),
        initial=rf.State(
            decoder=model.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim),
        ),
        body=_body,
    )

    logits = model.decode_logits(input_embed=input_embeddings, **loop_out)
    logits_packed, pack_dim = rf.pack_padded(logits, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False)
    targets_packed, _ = rf.pack_padded(
        targets, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
    )

    log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
    log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
    loss = rf.cross_entropy(
        target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
    )
    loss.mark_as_loss("ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

    best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
    frame_error = best != targets_packed
    frame_error.mark_as_loss(name="fer", as_error=True)


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"


@contextlib.contextmanager
def _opt_apply_pretrain_to_encoder(
    encoder: ConformerEncoder, collected_outputs: Optional[Dict[str, Tensor]], pretrain_opts: Optional[Dict[str, Any]]
):
    """Function is run within RETURNN."""
    if not pretrain_opts:
        yield
        return
    step = rf.get_run_ctx().step
    steps: Union[Sequence[Tuple[int, Dict[str, Any]]], Dict[int, Dict[str, Any]]] = pretrain_opts["steps"]
    if isinstance(steps, (list, tuple)):
        steps_ = {}
        step_bound = 0
        for step_bound_rel, opts in steps:
            step_bound += step_bound_rel
            steps_[step_bound] = opts
        steps = steps_
    assert isinstance(steps, dict)
    for step_bound, opts in sorted(steps.items()):
        if step < step_bound:
            assert isinstance(opts, dict)
            opts_ = opts.copy()
            # somewhat hacky but that is still the easiest way I can think of, without touching a lot of other code
            pretrain_num_layers = opts_.pop("num_layers")
            assert not opts_, f"unhandled opts: {opts_} in opts {opts} for step bound {step_bound}"
            orig_layers = encoder.layers[:]
            del encoder.layers[pretrain_num_layers:]
            yield
            encoder.layers[:] = orig_layers
            if collected_outputs is not None:
                assert len(collected_outputs) == pretrain_num_layers
                for i in range(pretrain_num_layers, len(orig_layers)):
                    collected_outputs[str(i)] = collected_outputs[str(pretrain_num_layers - 1)]
            return
    yield
    return


def model_recog(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    max_seq_len: Optional[int] = None,
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

    from returnn.config import get_global_config

    config = get_global_config()
    beam_search_opts = config.typed_value("beam_search_opts")

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = beam_search_opts.get("beam_size", 12)
    length_normalization_exponent = beam_search_opts.get("length_normalization_exponent", 1.0)
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    decoder_state = model.decoder_default_initial_state(batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)
    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    ended = rf.constant(False, dims=batch_dims_)
    out_seq_len = rf.constant(0, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    lm_scale = beam_search_opts.get("lm_scale", None)
    lm_state = None
    if lm_scale:
        lm_state = model.language_model.default_initial_state(batch_dims=batch_dims_)

    ilm_scale = beam_search_opts.get("ilm_scale", None)
    ilm_state = None
    if ilm_scale:
        ilm_state = model.internal_language_model.decoder_default_initial_state(batch_dims=batch_dims_)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        if i == 0:
            input_embed = rf.zeros(batch_dims_ + [model.target_embed.out_dim], feature_dim=model.target_embed.out_dim)
        else:
            input_embed = model.target_embed(target)

        step_out, decoder_state = model.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=decoder_state,
        )
        logits = model.decode_logits(input_embed=input_embed, **step_out)
        label_log_prob = rf.log_softmax(logits, axis=model.target_dim)

        if lm_scale:
            lm_log_prob, lm_state = model.language_model(target, spatial_dim=single_step_dim, state=lm_state)
            label_log_prob += lm_scale * lm_log_prob

        if ilm_state:
            ilm_step_out, ilm_state = model.internal_language_model.loop_step(input_embed=input_embed, state=ilm_state)
            ilm_logits = model.internal_language_model.decode_logits(input_embed=input_embed, **ilm_step_out)
            ilm_log_prob = rf.log_softmax(ilm_logits, axis=model.target_dim)
            label_log_prob -= ilm_scale * ilm_log_prob

        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=-1.0e30),
            label_log_prob,
        )
        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{i}-beam"), axis=[beam_dim, model.target_dim]
        )  # seq_log_prob, backrefs, target: Batch, Beam
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        decoder_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), decoder_state)

        if lm_state:
            lm_state = rf.nested.gather_nested(lm_state, indices=backrefs)

        if ilm_state:
            ilm_state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), ilm_state)

        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        i += 1

        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

        if i > 1 and length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob *= rf.where(
                ended,
                (i / (i - 1)) ** length_normalization_exponent,
                1.0,
            )

    if i > 0 and length_normalization_exponent != 0:
        # WARNING: small error here, should be (1/(i-1))^(length_normalization_exponent).
        #  But not changing this now to not change behavior.
        seq_log_prob *= (1 / i) ** length_normalization_exponent

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
    out_spatial_dim = Dim(out_seq_len, name="out-spatial")
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
# output_blank_label=blank is actually wrong for AED, but now we don't change it anymore
# because it would change all recog hashes.
# Also, it does not matter too much -- it will just cause an extra SearchRemoveLabelJob,
# which will not have any effect here.
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False


def model_recog_pure_torch(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    targets: Optional[Tensor] = None,
    targets_spatial_dim: Optional[Dim] = None,
    max_seq_len: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor], Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        recog results info: key -> {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import torch
    import time
    from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_v5 import BeamSearchOptsV5, beam_search_v5
    from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_sep_ended import (
        BeamSearchDynBeamOpts,
        beam_search_sep_ended,
    )
    from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_sep_ended_keep_v6 import (
        BeamSearchSepEndedKeepOpts,
        beam_search_sep_ended_keep_v6,
    )
    from i6_experiments.users.zeyer.decoding.beam_search_torch.scorers.length_reward import LengthRewardScorer
    from i6_experiments.users.zeyer.decoding.beam_search_torch.scorers.shallow_fusion import ShallowFusedLabelScorers
    from returnn.config import get_global_config

    config = get_global_config()

    torch.cuda.set_sync_debug_mode(1)  # debug CUDA sync. does not hurt too much to leave this always in?
    start_time = time.perf_counter_ns()

    data_concat_zeros = config.float("data_concat_zeros", 0)
    if data_concat_zeros:
        data_concat_zeros_dim = Dim(int(data_concat_zeros * _batch_size_factor * 100), name="data_concat_zeros")
        data, data_spatial_dim = rf.concat(
            (data, data_spatial_dim), (rf.zeros([data_concat_zeros_dim]), data_concat_zeros_dim), allow_broadcast=True
        )

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    assert len(batch_dims) == 1, batch_dims  # not implemented otherwise, simple to add...
    batch_dim = batch_dims[0]
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")

    if data.raw_tensor.device.type == "cuda":
        # Just so that timing of encoder is correct.
        torch.cuda.synchronize(data.raw_tensor.device)

    enc_end_time = time.perf_counter_ns()

    beam_search_version = config.typed_value("beam_search_version", 1)
    beam_search_func = {
        5: beam_search_v5,
        "sep_ended": beam_search_sep_ended,
        "sep_ended_keep_v6": beam_search_sep_ended_keep_v6,
    }[beam_search_version]
    if beam_search_version == "sep_ended":
        beam_search_opts_cls = BeamSearchDynBeamOpts
    elif isinstance(beam_search_version, str) and beam_search_version.startswith("sep_ended_keep"):
        beam_search_opts_cls = BeamSearchSepEndedKeepOpts
    elif isinstance(beam_search_version, int) and beam_search_version >= 5:
        beam_search_opts_cls = BeamSearchOptsV5
    else:
        raise ValueError(f"unexpected {beam_search_version=}")
    beam_search_opts = (config.typed_value("beam_search_opts", None) or {}).copy()
    if beam_search_opts.get("beam_size") is None:
        beam_search_opts["beam_size"] = config.int("beam_size", 12)
    if beam_search_opts.get("length_normalization_exponent") is None:
        beam_search_opts["length_normalization_exponent"] = config.float("length_normalization_exponent", 1.0)
    if beam_search_opts.get("length_reward") is None:
        beam_search_opts["length_reward"] = config.float("length_reward", 0.0)
    extra = {}
    out_individual_seq_scores = None
    if config.bool("beam_search_collect_individual_seq_scores", False):
        out_individual_seq_scores = {}
        extra["out_individual_seq_scores"] = out_individual_seq_scores
    cheating = config.bool("cheating", False)
    if cheating:
        assert targets and targets_spatial_dim
        extra["cheating_targets"] = targets.copy_compatible_to_dims_raw([batch_dim, targets_spatial_dim])
        extra["cheating_targets_seq_len"] = targets_spatial_dim.dyn_size_ext.copy_compatible_to_dims_raw([batch_dim])
    coverage_scale = beam_search_opts.pop("attention_coverage_scale", 0.0)
    coverage_opts = beam_search_opts.pop("attention_coverage_opts", {})
    neg_coverage_scale = beam_search_opts.pop("neg_attention_coverage_scale", 0.0)
    neg_coverage_opts = beam_search_opts.pop("neg_attention_coverage_opts", {})
    monotonicity_scale = beam_search_opts.pop("attention_monotonicity_scale", 0.0)
    monotonicity_opts = beam_search_opts.pop("attention_monotonicity_opts", {})
    max_seq_len_factor = beam_search_opts.pop("max_seq_len_factor", 1)
    if max_seq_len_factor != 1:
        max_seq_len = rf.cast(max_seq_len * max_seq_len_factor, max_seq_len.dtype)
    label_scorer = ShallowFusedLabelScorers()
    if coverage_scale or neg_coverage_scale or cheating:
        label_scorer.label_scorers.update(
            get_label_scorer_and_coverage_scorer_pure_torch(
                model=model,
                batch_dim=batch_dim,
                enc=enc,
                enc_spatial_dim=enc_spatial_dim,
                coverage_opts=coverage_opts,
                coverage_scale=coverage_scale,
                neg_coverage_scale=neg_coverage_scale,
                neg_coverage_opts=neg_coverage_opts,
                monotonicity_scale=monotonicity_scale,
                monotonicity_opts=monotonicity_opts,
                always_add_scorers=cheating,
            )
        )
    else:
        label_scorer.label_scorers["decoder"] = (
            get_label_scorer_pure_torch(model=model, batch_dim=batch_dim, enc=enc, enc_spatial_dim=enc_spatial_dim),
            1.0,
        )
    if isinstance(beam_search_version, str) or beam_search_version >= 5:
        len_reward = beam_search_opts.pop("length_reward", 0.0)
        if len_reward or cheating:
            label_scorer.label_scorers["length_reward"] = (LengthRewardScorer(), len_reward)
    if model.language_model:
        lm_scale = beam_search_opts.pop("lm_scale")  # must be defined with LM
        label_scorer.label_scorers["lm"] = (model.language_model_make_label_scorer(), lm_scale)

    print("** max seq len:", max_seq_len.raw_tensor)

    # Beam search happening here:
    (
        seq_targets,  # [Batch,FinalBeam,OutSeqLen]
        seq_log_prob,  # [Batch,FinalBeam]
        out_seq_len,  # [Batch,FinalBeam]
    ) = beam_search_func(
        label_scorer,
        batch_size=int(batch_dim.get_dim_value()),
        max_seq_len=max_seq_len.copy_compatible_to_dims_raw([batch_dim]),
        device=data.raw_tensor.device,
        opts=beam_search_opts_cls(
            **beam_search_opts,
            bos_label=model.bos_idx,
            eos_label=model.eos_idx,
            num_labels=model.target_dim.dimension,
        ),
        **extra,
    )

    beam_dim = Dim(seq_log_prob.shape[1], name="beam")
    out_spatial_dim = Dim(rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim], name="out_spatial"))
    seq_targets_t = rf.convert_to_tensor(
        seq_targets, dims=[batch_dim, beam_dim, out_spatial_dim], sparse_dim=model.target_dim
    )
    seq_log_prob_t = rf.convert_to_tensor(seq_log_prob, dims=[batch_dim, beam_dim])

    search_end_time = time.perf_counter_ns()
    data_seq_len_sum = rf.reduce_sum(data_spatial_dim.dyn_size_ext, axis=data_spatial_dim.dyn_size_ext.dims)
    data_seq_len_sum_secs = data_seq_len_sum.raw_tensor / _batch_size_factor / 100.0
    data_seq_len_max_seqs = data_spatial_dim.get_dim_value() / _batch_size_factor / 100.0
    out_len_longest_sum = rf.reduce_sum(rf.reduce_max(out_spatial_dim.dyn_size_ext, axis=beam_dim), axis=batch_dim)
    print(
        "TIMINGS:",
        ", ".join(
            (
                f"batch size {data.get_batch_dim_tag().get_dim_value()}",
                f"data len max {data_spatial_dim.get_dim_value()} ({data_seq_len_max_seqs:.2f} secs)",
                f"data len sum {data_seq_len_sum.raw_tensor} ({data_seq_len_sum_secs:.2f} secs)",
                f"enc {enc_end_time - start_time} ns",
                f"enc len max {enc_spatial_dim.get_dim_value()}",
                f"dec {search_end_time - enc_end_time} ns",
                f"out len max {out_spatial_dim.get_dim_value()}",
                f"out len longest sum {out_len_longest_sum.raw_tensor}",
            )
        ),
    )

    extra_recog_results = {}
    if out_individual_seq_scores:
        for k, v in out_individual_seq_scores.items():
            extra_recog_results[f"score:{k}"] = rf.convert_to_tensor(
                v.expand(batch_dim.get_dim_value(), beam_dim.get_dim_value()), dims=[batch_dim, beam_dim]
            )

    return seq_targets_t, seq_log_prob_t, extra_recog_results, out_spatial_dim, beam_dim


def get_label_scorer_pure_torch(
    *,
    model: Model,
    batch_dim: Dim,
    enc: Dict[str, Tensor],
    enc_spatial_dim: Dim,
):
    import torch
    import functools
    from i6_experiments.users.zeyer.decoding.beam_search_torch.interface import (
        LabelScorerIntf,
        StateObjTensorExt,
        StateObjIgnored,
    )

    class LabelScorer(LabelScorerIntf):
        """label scorer"""

        def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
            """Initial state."""
            beam_dim = Dim(1, name="initial-beam")
            batch_dims_ = [batch_dim, beam_dim]
            decoder_state = model.decoder_default_initial_state(batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)
            return tree.map_structure(functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim), decoder_state)

        def max_remaining_seq_score(
            self, *, state: Any, max_remaining_steps: torch.Tensor, device: torch.device
        ) -> torch.Tensor:
            """max remaining"""
            return torch.zeros((1, 1), device=device)

        def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
        ) -> Tuple[torch.Tensor, Any]:
            """update state"""
            beam_dim = Dim(prev_label.shape[1], name="beam")

            def _map_raw_to_tensor(v):
                if isinstance(v, StateObjTensorExt):
                    tensor: Tensor = v.extra
                    tensor = tensor.copy_template_new_dim_tags(
                        (batch_dim, beam_dim) + tensor.dims[2:], keep_special_axes=True
                    )
                    tensor.raw_tensor = v.tensor
                    return tensor
                elif isinstance(v, StateObjIgnored):
                    return v.content
                else:
                    raise TypeError(f"_map_raw_to_tensor: unexpected {v} ({type(v).__name__})")

            input_embed = model.target_embed(
                rf.convert_to_tensor(prev_label, dims=[batch_dim, beam_dim], sparse_dim=model.target_dim)
            )
            decode_out, decoder_state = model.loop_step(
                **enc,
                enc_spatial_dim=enc_spatial_dim,
                input_embed=input_embed,
                state=tree.map_structure(_map_raw_to_tensor, prev_state),
            )
            logits = model.decode_logits(input_embed=input_embed, **decode_out)
            label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
            assert set(label_log_prob.dims) == {batch_dim, beam_dim, model.target_dim}

            return (
                self._map_tensor_to_raw(label_log_prob, beam_dim=beam_dim).tensor,
                tree.map_structure(functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim), decoder_state),
            )

        @staticmethod
        def _map_tensor_to_raw(v, *, beam_dim: Dim):
            if isinstance(v, Tensor):
                if beam_dim not in v.dims:
                    return StateObjIgnored(v)
                batch_dims_ = [batch_dim, beam_dim]
                v = v.copy_transpose(batch_dims_ + [dim for dim in v.dims if dim not in batch_dims_])
                raw = v.raw_tensor
                return StateObjTensorExt(raw, v.copy_template())
            elif isinstance(v, Dim):
                return StateObjIgnored(v)
            else:
                raise TypeError(f"_map_tensor_to_raw: unexpected {v} ({type(v).__name__})")

    return LabelScorer()


# RecogDef API
model_recog_pure_torch: RecogDef[Model]
model_recog_pure_torch.output_with_beam = True
model_recog_pure_torch.output_blank_label = None
model_recog_pure_torch.batch_size_dependent = False


def get_label_scorer_and_coverage_scorer_pure_torch(
    *,
    model: Model,
    batch_dim: Dim,
    enc: Dict[str, Tensor],
    enc_spatial_dim: Dim,
    coverage_scale: float = 0.0,
    coverage_opts: Optional[Dict[str, Any]] = None,
    neg_coverage_scale: float = 0.0,
    neg_coverage_opts: Optional[Dict[str, Any]] = None,
    monotonicity_scale: float = 0.0,
    monotonicity_opts: Optional[Dict[str, Any]] = None,
    always_add_scorers: bool = False,
):
    import torch
    import functools
    from returnn.frontend.decoder.transformer import TransformerDecoderLayer
    from i6_experiments.users.zeyer.decoding.beam_search_torch.interface import (
        LabelScorerIntf,
        StateObjTensorExt,
        StateObjIgnored,
    )

    accum_att_weights = rf.zeros(())  # [Batch,Beam,kv_axis]
    att_weights_dec_frame: Tensor  # [Batch,Beam,kv_axis]
    beam_dim: Dim

    raise NotImplementedError("need more work here")  # TODO...

    model_att_reduce_type = coverage_opts.get("model_att_reduce_type", "max")

    def hooked_cross_att(self: rf.CrossAttention, q: Tensor, k: Tensor, v: Tensor, *, kv_axis: Dim) -> Tensor:
        """apply attention"""
        nonlocal att_weights_dec_frame
        # Standard dot attention, inline rf.dot_attention.
        q *= self.key_dim_per_head.dimension**-0.5
        energy = rf.matmul(q, k, reduce=self.key_dim_per_head)
        att_weights = rf.softmax(energy, axis=kv_axis)
        if model_att_reduce_type == "max":
            att_weights_dec_frame = rf.maximum(att_weights_dec_frame, rf.reduce_max(att_weights, axis=self.num_heads))
        elif model_att_reduce_type == "avg":
            att_weights_dec_frame += rf.reduce_mean(att_weights, axis=self.num_heads) * (1 / len(model.decoder.layers))
        else:
            raise ValueError(f"invalid model_att_reduce_type {model_att_reduce_type!r}")
        # Masking not needed because softmax should already have masked,
        # so we have 0.0 att weights for padded frames.
        att = rf.matmul(att_weights, v, reduce=kv_axis, use_mask=False)
        if v.feature_dim in att.dims:
            att.feature_dim = v.feature_dim
        output, _ = rf.merge_dims(att, dims=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total)
        if self.proj:
            output = self.proj(output)
        return output

    for layer in model.decoder.layers:
        layer: TransformerDecoderLayer
        layer.cross_att.attention = functools.partial(hooked_cross_att, self=layer.cross_att)

    class LabelScorer(LabelScorerIntf):
        """label scorer"""

        def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
            """Initial state."""
            beam_dim = Dim(1, name="initial-beam")
            batch_dims_ = [batch_dim, beam_dim]
            decoder_state = model.decoder_default_initial_state(batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)
            if coverage_scale or neg_coverage_scale or always_add_scorers:
                decoder_state["accum_att_weights"] = rf.zeros(batch_dims_)
            return tree.map_structure(functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim), decoder_state)

        def max_remaining_seq_score(
            self, *, state: Any, max_remaining_steps: torch.Tensor, device: torch.device
        ) -> torch.Tensor:
            """max remaining"""
            return torch.zeros((1, 1), device=device)

        def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
        ) -> Tuple[torch.Tensor, Any]:
            """update state"""
            nonlocal beam_dim
            beam_dim = Dim(prev_label.shape[1], name="beam")

            def _map_raw_to_tensor(v):
                if isinstance(v, StateObjTensorExt):
                    tensor: Tensor = v.extra
                    tensor = tensor.copy_template_new_dim_tags(
                        (batch_dim, beam_dim) + tensor.dims[2:], keep_special_axes=True
                    )
                    tensor.raw_tensor = v.tensor
                    return tensor
                elif isinstance(v, StateObjIgnored):
                    return v.content
                else:
                    raise TypeError(f"_map_raw_to_tensor: unexpected {v} ({type(v).__name__})")

            prev_state = tree.map_structure(_map_raw_to_tensor, prev_state)

            nonlocal accum_att_weights, att_weights_dec_frame
            accum_att_weights = prev_state["accum_att_weights"]
            att_weights_dec_frame = rf.zeros(())
            logits, decoder_state = model.decoder(
                rf.convert_to_tensor(prev_label, dims=[batch_dim, beam_dim], sparse_dim=model.target_dim),
                spatial_dim=single_step_dim,
                encoder=enc,
                state=prev_state,
            )
            accum_att_weights += att_weights_dec_frame
            if coverage_scale or neg_coverage_scale or always_add_scorers:
                decoder_state["accum_att_weights"] = accum_att_weights
            label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
            assert set(label_log_prob.dims) == {batch_dim, beam_dim, model.target_dim}

            return (
                self._map_tensor_to_raw(label_log_prob, beam_dim=beam_dim).tensor,
                tree.map_structure(functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim), decoder_state),
            )

        @staticmethod
        def _map_tensor_to_raw(v, *, beam_dim: Dim):
            if isinstance(v, Tensor):
                if beam_dim not in v.dims:
                    return StateObjIgnored(v)
                batch_dims_ = [batch_dim, beam_dim]
                v = v.copy_transpose(batch_dims_ + [dim for dim in v.dims if dim not in batch_dims_])
                raw = v.raw_tensor
                return StateObjTensorExt(raw, v.copy_template())
            elif isinstance(v, Dim):
                return StateObjIgnored(v)
            else:
                raise TypeError(f"_map_tensor_to_raw: unexpected {v} ({type(v).__name__})")

    class CoverageScorer(LabelScorerIntf):
        """coverage

        Google NMT: https://arxiv.org/pdf/1609.08144.pdf
        Alternative: https://arxiv.org/abs/1612.02695
        Another alternative: https://arxiv.org/pdf/2105.00982.pdf
        """

        def __init__(self, opts: Dict[str, Any]):
            self.opts = opts

        def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
            """Initial state."""
            return {"prev_score": torch.zeros([batch_size, 1], device=device)}

        def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
        ) -> Tuple[torch.Tensor, Any]:
            """update state"""
            prev_label  # noqa  # unused
            # We assume the label scorer has run before us (make sure by right ordering).
            accum_att_weights_ = accum_att_weights
            assert set(accum_att_weights_.dims) == {batch_dim, beam_dim, enc_spatial_dim}
            cov_type = self.opts.get("type", "log1p")
            if self.opts.get("rescale", False):
                accum_att_weights_ /= rf.maximum(rf.reduce_max(accum_att_weights_, axis=enc_spatial_dim), 1.0)
            if cov_type == "log1p":  # log1p, to avoid having lots of negative numbers. So this starts more around 0.0.
                coverage_score = rf.log1p(rf.minimum(accum_att_weights_, 1.0))
            elif cov_type == "log":  # orig Google NMT: https://arxiv.org/pdf/1609.08144.pdf, but clipped
                eps = self.opts.get("eps", 0.0)
                clip_min = self.opts.get("clip_min", 0.01)
                coverage_score = rf.log(rf.clip_by_value(accum_att_weights_, clip_min, 1.0) + eps)
            elif cov_type == "indicator":
                threshold = self.opts.get("threshold", 0.5)
                coverage_score = rf.where(accum_att_weights_ >= threshold, 1.0, 0.0)
            elif cov_type == "relu_upper":
                threshold = self.opts.get("threshold", 0.5)
                coverage_score = rf.where(accum_att_weights_ >= threshold, accum_att_weights_ - threshold, 0.0)
            else:
                raise ValueError(f"invalid coverage opts type {cov_type!r}")
            coverage_score = rf.reduce_sum(coverage_score, axis=enc_spatial_dim)
            coverage_score_raw = coverage_score.copy_compatible_to_dims_raw((batch_dim, beam_dim))
            state = {"prev_score": coverage_score_raw}
            return (coverage_score_raw - prev_state["prev_score"])[:, :, None], state

    class MonotonicityScorer(LabelScorerIntf):
        """score monotonicity"""

        def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
            """Initial state."""
            return {"att_pos": torch.zeros([batch_size, 1], device=device)}

        def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
        ) -> Tuple[torch.Tensor, Any]:
            """update state"""
            prev_label  # noqa  # unused
            # We assume the label scorer has run before us (make sure by right ordering).
            assert set(att_weights_dec_frame.dims) == {batch_dim, beam_dim, enc_spatial_dim}
            att_pos = rf.matmul(
                att_weights_dec_frame,
                rf.range_over_dim(enc_spatial_dim, dtype=att_weights_dec_frame.dtype),
                reduce=enc_spatial_dim,
                use_mask=False,  # not needed, att weights already 0 outside
            )  # [Batch,Beam]
            att_pos_raw = att_pos.copy_compatible_to_dims_raw((batch_dim, beam_dim))
            delta_raw = prev_state["att_pos"] - att_pos_raw
            threshold = monotonicity_opts.get("threshold", 1.0)
            # Penalize when below threshold. The more it is below (or even negative), the more.
            score_raw = torch.where(delta_raw < threshold, delta_raw - threshold, 0.0)  # [Batch,Beam]
            return score_raw[:, :, None], {"att_pos": att_pos_raw}

    # Note: insertion order matters here, we want that decoder is scored first.
    res = {"decoder": (LabelScorer(), 1.0)}
    if coverage_scale or always_add_scorers:
        res["attention_coverage"] = (CoverageScorer(coverage_opts or {}), coverage_scale)
    if neg_coverage_scale or (neg_coverage_opts and always_add_scorers):
        # Idea: Too much attention on some frames (e.g. repetitions) is scored negatively.
        res["attention_neg_coverage"] = (CoverageScorer(neg_coverage_opts or {}), -neg_coverage_scale)
    if monotonicity_scale or always_add_scorers:
        res["attention_monotonicity"] = (MonotonicityScorer(), monotonicity_scale)
    return res


def model_warmup(*, model: Model, **_kwargs):
    """warmup, for more reliable timing measures"""
    import torch
    import time
    from returnn.config import get_global_config
    from returnn.tensor import Dim
    import returnn.frontend as rf

    config = get_global_config()
    start_time = time.monotonic()
    limit = start_time + config.float("model_warmup_time", 10.0)

    print("*** warming up...")
    while time.monotonic() < limit:
        batch_dim = Dim(10, name="dummy_batch")
        time_dim = Dim(rf.full(dims=[batch_dim], fill_value=16_000), name="dummy_time")
        feat_dim = Dim(1, name="dummy_feat")
        source = rf.zeros([batch_dim, time_dim, feat_dim])
        res = model.encode(source=source, in_spatial_dim=time_dim)
        if source.raw_tensor.device.type == "cuda":
            torch.cuda.synchronize(source.raw_tensor.device)
        res  # noqa  # keep ref to make sure it is calculated


class MiniLstmIlm(rf.Module):
    """
    Represents a Mini-LSTM ILM decoder (text only input).
    This is based on our standard AED LSTM decoder.
    Note that during training, all params are frozen except for the parameters of Mini-LSTM and
    the attention context linear projection.
    """

    def __init__(
        self,
        *,
        target_dim: Dim,
        target_embed_dim: Dim,
        att_context_dim: Dim,
        att_num_heads: Dim,
        hidden_dim: int = 50,
    ):
        """
        :param target_dim:
        :param target_embed_dim:
        :param att_context_dim:
        :param hidden_dim:
        :param att_num_heads:
        """

        frozen_modules = []

        self.target_dim = target_dim

        self.target_embed = rf.Embedding(target_dim, target_embed_dim)
        frozen_modules.append(self.target_embed)

        self.att_num_heads = att_num_heads
        self.att_context_dim = att_context_dim

        self.s = rf.ZoneoutLSTM(
            self.target_embed.out_dim + self.att_num_heads * self.att_context_dim,
            Dim(name="lstm", dimension=1024),
            zoneout_factor_cell=0.15,
            zoneout_factor_output=0.05,
            use_zoneout_output=False,  # like RETURNN/TF ZoneoutLSTM old default
            # parts_order="icfo",  # like RETURNN/TF ZoneoutLSTM
            # parts_order="ifco",
            parts_order="jifo",  # NativeLSTM (the code above converts it...)
            forget_bias=0.0,  # the code above already adds it during conversion
        )
        frozen_modules.append(self.s)

        mini_lstm_dim = Dim(name="mini_lstm_dim", dimension=hidden_dim)

        self.mini_lstm = rf.LSTM(in_dim=target_embed_dim, out_dim=mini_lstm_dim)

        # this is used instead of original attention context vector
        self.att_context_proj = rf.Linear(mini_lstm_dim, self.att_num_heads * self.att_context_dim)

        self.readout_in = rf.Linear(
            self.s.out_dim + self.target_embed.out_dim + self.att_num_heads * self.att_context_dim,
            Dim(name="readout", dimension=1024),
        )
        frozen_modules.append(self.readout_in)

        self.output_prob = rf.Linear(self.readout_in.out_dim // 2, target_dim)
        frozen_modules.append(self.output_prob)

        self.dropout_broadcast = rf.dropout_broadcast_default()

        for module in frozen_modules:
            for param in module.parameters():
                param.trainable = False

    def decoder_default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        """Default initial state"""
        state = rf.State(
            s=self.s.default_initial_state(batch_dims=batch_dims),
            mini_lstm=self.mini_lstm.default_initial_state(batch_dims=batch_dims),
            att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.att_context_dim]),
        )
        state.att.feature_dim_axis = len(state.att.dims) - 1
        return state

    def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
        """loop step out"""
        return {
            "s": Tensor(
                "s", dims=batch_dims + [self.s.out_dim], dtype=rf.get_default_float_dtype(), feature_dim_axis=-1
            ),
            "att": Tensor(
                "att",
                dims=batch_dims + [self.att_num_heads * self.att_context_dim],
                dtype=rf.get_default_float_dtype(),
                feature_dim_axis=-1,
            ),
        }

    def loop_step(
        self,
        *,
        input_embed: rf.Tensor,
        state: Optional[rf.State] = None,
    ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
        """step of the inner loop"""
        if state is None:
            batch_dims = input_embed.remaining_dims(input_embed.feature_dim)
            state = self.decoder_default_initial_state(batch_dims=batch_dims)
        state_ = rf.State()

        prev_att = state.att
        s, state_.s = self.s(rf.concat_features(input_embed, prev_att), state=state.s, spatial_dim=single_step_dim)

        # compute attention context vector via mini-lstm
        mini_lstm_out, state_.mini_lstm = self.mini_lstm(
            input_embed, state=state.mini_lstm, spatial_dim=single_step_dim
        )

        att = self.att_context_proj(mini_lstm_out)
        state_.att = att

        return {"s": s, "att": att}, state_

    def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
        """logits for the decoder"""
        readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
        readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
        readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
        logits = self.output_prob(readout)
        return logits


def mini_lstm_ilm_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> MiniLstmIlm:
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    assert target_dim
    config = get_global_config()  # noqa

    model = rf.build_from_dict(config.typed_value("_model_def_dict"), target_dim=target_dim)
    return model


mini_lstm_ilm_def: ModelDef
mini_lstm_ilm_def.behavior_version = 21
mini_lstm_ilm_def.backend = "torch"
mini_lstm_ilm_def.batch_size_factor = _batch_size_factor


def mini_lstm_ilm_train_def(
    *, model: MiniLstmIlm, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim
):
    from returnn.config import get_global_config

    config = get_global_config()
    use_normalized_loss = config.bool("use_normalized_loss", True)

    batch_dims = data.remaining_dims(data_spatial_dim)
    input_embeddings = model.target_embed(targets)
    input_embeddings = rf.shift_right(input_embeddings, axis=targets_spatial_dim, pad_value=0.0)

    def _body(input_embed: Tensor, state: rf.State):
        new_state = rf.State()
        loop_out_, new_state.decoder = model.loop_step(
            input_embed=input_embed,
            state=state.decoder,
        )
        return loop_out_, new_state

    loop_out, _, _ = rf.scan(
        spatial_dim=targets_spatial_dim,
        xs=input_embeddings,
        ys=model.loop_step_output_templates(batch_dims=batch_dims),
        initial=rf.State(
            decoder=model.decoder_default_initial_state(batch_dims=batch_dims),
        ),
        body=_body,
    )

    logits = model.decode_logits(input_embed=input_embeddings, **loop_out)
    logits_packed, pack_dim = rf.pack_padded(logits, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False)
    targets_packed, _ = rf.pack_padded(
        targets, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
    )

    log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
    log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
    loss = rf.cross_entropy(
        target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
    )
    loss.mark_as_loss("ce", use_normalized_loss=use_normalized_loss)

    best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
    frame_error = best != targets_packed
    frame_error.mark_as_loss(name="fer", as_error=True)


mini_lstm_ilm_train_def: TrainDef[MiniLstmIlm]
mini_lstm_ilm_train_def.learning_rate_control_error_measure = "ce"


ilm_config_params = dict(
    # torch_amp="bfloat16",
    # grad_scaler=None,
    batching="laplace:.1000",
    max_seqs=200,
    max_seq_length_default_target=None,
    gradient_clip_global_norm=5.0,
    optimizer={
        "class": "adamw",
        "epsilon": 1e-16,
        "weight_decay": 1e-6,
        "weight_decay_modules_blacklist": [
            "rf.Embedding",
            "rf.LearnedRelativePositionalEncoding",
        ],
    },
    accum_grad_multiple_step=2,
    learning_rate=1e-5,
    pos_emb_dropout=0.1,  # WARNING: when the self-att or conformer opts are custom, this is ignored! also for CTC!
    rf_att_dropout_broadcast=False,
)

config_11gb_ilm_v1 = dict_update_deep(
    ilm_config_params,
    {
        "optimizer.weight_decay": 1e-2,
        "calculate_exp_loss": True,
    },
)
