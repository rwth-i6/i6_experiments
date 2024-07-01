from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List
from itertools import product

from i6_experiments.users.gaudino.models.asr.rf.nn_lm.lm_import_2023_11_09 import (
    Trafo_LM_Model,
)
from sisyphus import tk

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_conformer_ctc import Model as ModelCTC
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import Model as ModelAED

from i6_experiments.users.gaudino.models.asr.rf.nn_lm.lm_import_2023_09_03 import (
    LSTM_LM_Model,
    # MakeModel,
)
from i6_experiments.users.gaudino.models.asr.rf.ilm_import_2024_04_17 import (
    MiniAtt_ILM_Model,
)
from i6_experiments.users.gaudino.model_interfaces.model_interfaces import ModelDef, TrainDef


import torch
import numpy


class MakeModel:
    """for import"""

    def __init__(
        self,
        in_dim: int,
        target_dim: int,
        *,
        eos_label: int = 0,
        num_enc_layers_1: int = 12,
        num_enc_layers_2: int = 12,

        model_args_1: Optional[Dict[str, Any]] = {},
        search_args_1: Optional[Dict[str, Any]] = {},
    ):
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.eos_label = eos_label
        self.num_enc_layers_1 = num_enc_layers_1
        self.num_enc_layers_2 = num_enc_layers_2
        self.search_args_1 = search_args_1
        self.model_args_1 = model_args_1

    def __call__(self) -> Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(
            name="target", dimension=self.target_dim, kind=Dim.Types.Feature
        )
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        return self.make_model(
            in_dim,
            target_dim,
            num_enc_layers_1=self.num_enc_layers_1,
            num_enc_layers_2=self.num_enc_layers_2,
            model_args=self.model_args_1,
            search_args=self.search_args_1,
        )

    @classmethod
    def make_model(
        cls,
        in_dim: Dim,
        target_dim: Dim,
        *,
        model_args: Optional[Dict[str, Any]] = {},
        search_args: Optional[Dict[str, Any]] = {},
        num_enc_layers_1: int = 12,
        num_enc_layers_2: int = 12,
        pos_emb_dropout: float = 0.0,
        lm_opts: Optional[Dict[str, Any]] = None,
        ilm_opts: Optional[Dict[str, Any]] = None,
        **extra,
    ) -> Model:
        """make"""

        target_embed_dim = Dim(name="target_embed", dimension=model_args.get("target_embed_dim", 640))
        lm = None
        ilm = None
        if lm_opts:
            assert isinstance(lm_opts, dict)
            lm_opts = lm_opts.copy()
            cls_name = lm_opts.pop("class")
            assert cls_name == "Trafo_LM_Model" or cls_name == "LSTM_LM_Model"
            lm_opts.pop("vocab_dim", None)  # will just overwrite

            if cls_name == "Trafo_LM_Model":
                lm = Trafo_LM_Model(target_dim, target_dim, **lm_opts)

            elif cls_name == "LSTM_LM_Model":
                lm = LSTM_LM_Model(target_dim, target_dim, **lm_opts)

        if ilm_opts:
            assert isinstance(ilm_opts, dict)
            ilm_opts = ilm_opts.copy()
            cls_name = ilm_opts.pop("class")

            ilm = MiniAtt_ILM_Model(target_embed_dim, target_dim, **ilm_opts)

        # lm = (lm, functools.partial(trafo_lm.make_label_scorer_torch, model=lm))

        model_aed = ModelAED(
            in_dim,
            num_enc_layers=num_enc_layers_1,
            enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
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
                ),
                ff_activation=lambda x: rf.relu(x) ** 2.0,
            ),
            target_embed_dim=target_embed_dim,
            target_dim=target_dim,
            blank_idx=target_dim.dimension,
            bos_idx=_get_bos_idx(target_dim),
            eos_idx=_get_eos_idx(target_dim),
            model_args=model_args,
            search_args=search_args,
            language_model=None,
            ilm=ilm,
            **extra,
        )

        model_ctc = ModelCTC(
            in_dim,
            num_enc_layers=num_enc_layers_2,
            enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
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
            target_dim=target_dim,
            blank_idx=target_dim.dimension,
            bos_idx=_get_bos_idx(target_dim),
            eos_idx=_get_eos_idx(target_dim),
            language_model=None,
            **extra,
        )

        return Model(
            in_dim,
            target_dim=target_dim,
            blank_idx=target_dim.dimension,
            eos_idx=_get_eos_idx(target_dim),
            bos_idx=_get_bos_idx(target_dim),
            model_aed=model_aed,
            model_ctc=model_ctc,
            language_model=lm,
            search_args=search_args,
        )


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        target_dim: Dim,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        model_aed: ModelAED,
        model_ctc: ModelCTC,
        language_model: Optional[rf.Module] = None,
        # model_args: Optional[Dict[str, Any]] = None,
        search_args: Optional[Dict[str, Any]] = None,
        # ilm: Optional[rf.Module] = None,
    ):
        super(Model, self).__init__()
        self.model_aed = model_aed
        self.model_ctc = model_ctc

        self.in_dim = in_dim
        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx

        # self.model_args = model_args
        self.search_args = search_args

        self.language_model = None
        if language_model:
            self.language_model = language_model



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

_log_mel_feature_dim = 80

def from_scratch_model_def(
    *,
    epoch: int,
    in_dim: Dim,
    target_dim: Dim,
    model_args: Optional[Dict[str, Any]],
    search_args: Optional[Dict[str, Any]],
) -> Model:
    """Function is run within RETURNN."""
    in_dim, epoch  # noqa
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    lm_opts = model_args.get("external_language_model")
    ilm_opts = model_args.get("internal_language_model")
    return MakeModel.make_model(
        in_dim, target_dim, model_args=model_args, search_args=search_args, lm_opts=lm_opts, ilm_opts=ilm_opts
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = (  # bsf * 20000
    20  # change batch size here - 20 for att_window - 40 for ctc_prefix
)
from_scratch_model_def.max_seqs = 200  # 1
