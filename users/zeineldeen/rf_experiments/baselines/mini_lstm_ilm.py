from typing import Sequence, List, Dict, Optional, Tuple

import types

import returnn.frontend as rf

from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, TrainDef
from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset
from i6_experiments.users.zeyer.train_v3 import train
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep

from .aed_lstm import get_tf_to_rf_converted_ckpt_path

from returnn.tensor import Tensor, Dim, single_step_dim

from .configs import _batch_size_factor


def py():
    lbs_bpe10k_dataset = get_librispeech_lm_dataset(vocab="bpe10k")

    def get_train_trans_dataset(self):
        return self.get_dataset("transcriptions-train", training=True)

    lbs_bpe10k_dataset.get_train_dataset = types.MethodType(get_train_trans_dataset, lbs_bpe10k_dataset)

    preload_from_files = {
        "aed": {"filename": get_tf_to_rf_converted_ckpt_path(), "init_for_train": True, "ignore_missing": True}
    }

    train(
        f"ilm/mini-lstm",
        config=dict_update_deep(
            config_11gb_ilm_v1,
            {"batch_size": 10_000, "preload_from_files": preload_from_files},
        ),
        train_dataset=lbs_bpe10k_dataset,
        model_def=ModelDefWithCfg(
            mini_lstm_ilm_def,
            {"_model_def_dict": rf.build_dict(MiniLstmIlm)},
        ),
        train_def=mini_lstm_ilm_train_def,
    )


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
        target_embed_dim: int = 640,
        att_context_dim: int = 512,
        hidden_dim: int = 50,
        att_num_heads: int = 1,
    ):
        """
        :param target_dim:
        :param target_embed_dim:
        :param att_context_dim:
        :param hidden_dim:
        :param att_num_heads:
        """

        frozen_modules = []

        target_embed_dim = Dim(name="target_embed", dimension=target_embed_dim)
        self.target_embed = rf.Embedding(target_dim, target_embed_dim)
        frozen_modules.append(self.target_embed)

        self.att_num_heads = att_num_heads

        self.s = rf.ZoneoutLSTM(
            self.target_embed.out_dim + att_num_heads * att_context_dim,
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
        self.att_context_proj = rf.Linear(mini_lstm_dim, Dim(name="att_context_proj", dimension=att_context_dim))

        self.readout_in = rf.Linear(
            self.s.out_dim + self.target_embed.out_dim + att_num_heads * att_context_dim,
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
            att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.encoder.out_dim]),
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
        mini_lstm_out = self.mini_lstm(input_embed)
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


config_params = dict(
    torch_amp="bfloat16",
    grad_scaler=None,
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
    config_params,
    {
        "optimizer.weight_decay": 1e-2,
        "calculate_exp_loss": True,
    },
)
