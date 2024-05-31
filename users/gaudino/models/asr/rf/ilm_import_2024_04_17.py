"""Import Tedlium2 LM from TF checkpoint to RETURNN frontend model with PT backend."""
from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree
from returnn.tensor import Tensor, Dim, single_step_dim, batch_dim
import returnn.frontend as rf
from typing import Optional, Union, Any, Tuple, List, Dict, Callable
import copy as _copy
from returnn.util.basic import NotSpecified

import functools

# from .base import ISeqDownsamplingEncoder


class MiniAtt_ILM_Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        target_dim: Dim,
        *,
        mini_att_lstm_dim: int = 50,
        mini_att_out_dim: int = 512,
        prior_dim: int = 1024,
        s_use_zoneout_output: bool = True, # for ted2, for ls960 set to false
        # layer_out_dim: int = 768, # default values for ted2 trafo lm
        # layer_ff_dim: int = 4096,
        # embed_dim: int = 128,
        # num_layers: int = 30,
        # att_num_heads: int = 12,
        # ff_activation: str = "gelu",
        # use_pos_enc: bool = False, # False for Tedlium2 LM
        search_args: Optional[Dict[str, Any]] = None,
    ):
        super(MiniAtt_ILM_Model, self).__init__()

        self.in_dim = in_dim
        self.target_dim = target_dim

        self.mini_att_lstm_dim = Dim(mini_att_lstm_dim, name="mini-att-lstm-dim")
        self.mini_att_out_dim = Dim(mini_att_out_dim, name="mini-att-out-dim")
        self.prior_dim = Dim(prior_dim, name="prior-dim")

        self.mini_att_lstm = rf.LSTM(in_dim, self.mini_att_lstm_dim, with_bias=True)

        self.mini_att = rf.Linear(
            self.mini_att_lstm_dim, self.mini_att_out_dim, with_bias=True
        )

        self.s_use_zoneout_output = s_use_zoneout_output

        self.prior_s = rf.ZoneoutLSTM(
            in_dim + self.mini_att_out_dim,
            self.prior_dim,
            zoneout_factor_cell=0.15,
            zoneout_factor_output=0.05,
            use_zoneout_output=self.s_use_zoneout_output,
            # parts_order="icfo",  # like RETURNN/TF ZoneoutLSTM
            # parts_order="ifco",
            parts_order="jifo",  # NativeLSTM (the code above converts it...)
        )
        self.prior_readout_in = rf.Linear(
            self.prior_dim + in_dim + self.mini_att_out_dim,
            self.prior_dim,
            with_bias=True,
        )
        self.prior_output_prob = rf.Linear(
            self.prior_dim / 2, target_dim, with_bias=True
        )

    def default_initial_state(
        self, *, batch_dims: Sequence[Dim], use_batch_dims_for_pos: bool = False
    ) -> rf.State:
        """default initial state"""
        state = rf.State(
            mini_att_lstm=self.mini_att_lstm.default_initial_state(
                batch_dims=batch_dims
            ),
            prior_s=self.prior_s.default_initial_state(batch_dims=batch_dims),
            mini_att_out=rf.zeros(
                list(batch_dims) + [self.mini_att_out_dim],
                feature_dim=self.mini_att_out_dim,
            ),
        )
        return state

    def select_state(self, state: rf.State, backrefs) -> rf.State:
        state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), state)
        return state

    def __call__(
        self,
        prev_target_emb,
        *,
        state: rf.State,
        spatial_dim: Dim,
        batch_dims=None,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ):
        """loop step"""

        #
        # "mini_att_lstm": {
        #     "class": "rec",
        #     "unit": "nativelstm2",
        #     "n_out": 50,
        #     "direction": 1,
        #     "from": "prev:target_embed",
        # },
        # "mini_att": {
        #     "class": "linear",
        #     "activation": None,
        #     "with_bias": True,
        #     "from": "mini_att_lstm",
        #     "n_out": 512,
        # },
        # "prior_s": {
        #     "class": "rnn_cell",
        #     "unit": "zoneoutlstm",
        #     "n_out": 1024,
        #     "from": ["prev:target_embed", "prev:mini_att"],
        #     "L2": 0.0001,
        #     "unit_opts": {
        #         "zoneout_factor_cell": 0.15,
        #         "zoneout_factor_output": 0.05,
        #         "use_zoneout_output": True,
        #     },
        # },
        # "prior_readout_in": {
        #     "class": "linear",
        #     "activation": None,
        #     "with_bias": True,
        #     "from": ["prior_s", "prev:target_embed", "mini_att"],
        #     "n_out": 1024,
        #     "L2": 0.0001,
        # },
        # "prior_readout": {
        #     "class": "reduce_out",
        #     "from": ["prior_readout_in"],
        #     "num_pieces": 2,
        #     "mode": "max",
        # },
        # "prior_output_prob": {
        #     "class": "softmax",
        #     "from": ["prior_readout"],
        #     "dropout": 0.3,
        #     "target": "bpe_labels",
        #     "loss": "ce",
        #     "loss_opts": {"label_smoothing": 0.1},
        #     "L2": 0.0001,
        # },

        new_state = rf.State()

        mini_att_lstm, mini_att_lstm_state = self.mini_att_lstm(
            prev_target_emb, state=state.mini_att_lstm, spatial_dim=single_step_dim
        )
        new_state.mini_att_lstm = mini_att_lstm_state

        mini_att = self.mini_att(mini_att_lstm)
        new_state.mini_att_out = mini_att

        prior_s, prior_s_state = self.prior_s(
            rf.concat_features(prev_target_emb, state.mini_att_out),
            state=state.prior_s,
            spatial_dim=single_step_dim,
        )
        new_state.prior_s = prior_s_state

        readout_in = self.prior_readout_in(
            rf.concat_features(prior_s, prev_target_emb, mini_att)
        )
        readout = rf.reduce_out(
            readout_in, mode="max", num_pieces=2, out_dim=self.prior_output_prob.in_dim
        )
        readout_lin = self.prior_output_prob(readout)
        # prior_output_prob = rf.softmax(readout_lin, axis=self.target_dim)

        return {"output": readout_lin, "state": new_state}


class MakeModel:
    """for import"""

    def __init__(
        self,
        in_dim: int,
        target_dim: int,
        *,
        eos_label: int = 0,
        # num_enc_layers: int = 12,
    ):
        self.in_dim = in_dim
        self.target_dim = target_dim

        self.eos_label = eos_label

    def __call__(self) -> MiniAtt_ILM_Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(
            name="target", dimension=self.target_dim, kind=Dim.Types.Feature
        )
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        return self.make_model(in_dim, target_dim)

    @classmethod
    def make_model(
        cls,
        in_dim: Dim,
        target_dim: Dim,
        # *,
        # search_args: Optional[Dict[str, Any]],
        # num_enc_layers: int = 12,
    ) -> MiniAtt_ILM_Model:
        """make"""
        return MiniAtt_ILM_Model(
            in_dim=in_dim,
            target_dim=target_dim,
        )
