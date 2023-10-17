"""Param Import
"""

from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree

from sisyphus import tk

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.lm_import_2023_09_03 import (
    LSTM_LM_Model,
    MakeModel,
)
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog import (
    model_recog,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_time_sync import (
    model_recog_time_sync,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_dump import (
    model_recog_dump,
)


import torch
import numpy

# from functools import partial


# From Mohammad, 2023-06-29
# dev-clean  2.27
# dev-other  5.39
# test-clean  2.41
# test-other  5.51
# _returnn_tf_config_filename = "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.1oORPHJTAcW0/output/returnn.config"
# E.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
_torch_ckpt_filename_w_lstm_lm = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/full_w_lm_import_2023_09_07/average.pt"
# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def sis_run_with_prefix(prefix_name: str = None):
    """run the exp"""
    from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
    from ._moh_att_2023_06_30_import import map_param_func_v3
    from .sis_setup import get_prefix_for_config
    from i6_core.returnn.training import Checkpoint as TfCheckpoint, PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
    from i6_experiments.users.gaudino.recog import recog_model
    from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import (
        ConvertTfCheckpointToRfPtJob,
    )
    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_bpe10k_raw,
    )

    if not prefix_name:
        prefix_name = get_prefix_for_config(__file__)

    task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)

    extern_data_dict = task.train_dataset.get_extern_data()
    default_target_key = task.train_dataset.get_default_target()
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    target_dim = targets.feature_dim_or_sparse_dim

    new_chkpt_path = ConvertTfCheckpointToRfPtJob(
        checkpoint=TfCheckpoint(
            index_path=generic_job_output(_returnn_tf_ckpt_filename)
        ),
        make_model_func=MakeModel(
            in_dim=_log_mel_feature_dim,
            target_dim=target_dim.dimension,
            eos_label=_get_eos_idx(target_dim),
        ),
        map_func=map_param_func_v3,
    ).out_checkpoint

    # att + ctc decoding

    new_chkpt_path = tk.Path(
        _torch_ckpt_filename_w_lstm_lm, hash_overwrite="torch_ckpt_w_lstm_lm"
    )
    new_chkpt = PtCheckpoint(new_chkpt_path)
    model_with_checkpoint = ModelWithCheckpoint(
        definition=from_scratch_model_def, checkpoint=new_chkpt
    )

    if True:
        search_args = {
            "beam_size": 12,
            # att decoder args
            "att_scale": 0.7,
            "ctc_scale": 0.3,
            "use_ctc": True,
            "mask_eos": True,
            "add_lstm_lm": False,
            "lstm_scale": 0.33,
            "prior_corr": True,
            "prior_scale": 0.2,
            "length_normalization_exponent": 1.0,  # 0.0 for disabled
            # "window_margin": 10,
            "rescore_w_ctc": False,
        }
        # dev_sets = ["dev-other"]  # only dev-other for testing
        dev_sets = None  # all
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=dev_sets,
            search_args=search_args,
        )
        tk.register_output(
            prefix_name
            + f"/espnet_att{search_args['att_scale']}_ctc{search_args['ctc_scale']}_beam{search_args['beam_size']}_bsf40"
            # + f"/att{search_args['att_scale']}_ctc{search_args['ctc_scale']}_beam{search_args['beam_size']}_two_pass_maskeos"
            # + f"/att{search_args['att_scale']}_lstm_lm{search_args['lstm_scale']}_beam{search_args['beam_size']}"
            # + f"/att{search_args['att_scale']}_beam{search_args['beam_size']}_bsf_40"
            + f"/recog_results",
            res.output,
        )

    for prior_scale in []:
        search_args["prior_scale"] = prior_scale
        search_args["length_normalization_exponent"] = 1.0
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog,
            dev_sets=dev_sets,
            search_args=search_args,
        )
        tk.register_output(
            prefix_name
            # + f"/espnet_att{search_args['att_scale']}_ctc{search_args['ctc_scale']}_beam{search_args['beam_size']}_maskEos"
            + f"/att{search_args['att_scale']}_ctc{search_args['ctc_scale']}_beam{search_args['beam_size']}_prior{search_args['prior_scale']}"
            + f"/recog_results",
            res.output,
        )

    # ctc only decoding
    if False:
        search_args = {
            "beam_size": 12,
            "add_lstm_lm": False,
        }

        dev_sets = ["dev-other"]  # only dev-other for testing
        # dev_sets = None  # all
        res = recog_model(
            task,
            model_with_checkpoint,
            model_recog_ctc,
            dev_sets=dev_sets,
            search_args=search_args,
        )
        tk.register_output(
            prefix_name
            # + f"/espnet_att{search_args['att_scale']}_ctc{search_args['ctc_scale']}_beam{search_args['beam_size']}_maskEos"
            + f"/ctc_greedy" + f"/recog_results",
            res.output,
        )

    # time sync decoding
    search_args = {
        "beam_size": 32,
        "add_lstm_lm": False,
        "length_normalization_exponent": 1.0,  # 0.0 for disabled
        "mask_eos": True,
        "att_scale": 0.65,
        "ctc_scale": 0.35,
        "rescore_w_ctc": False,
        "prior_corr": True,
        "prior_scale": 0.3,
    }

    dev_sets = ["dev-other"]  # only dev-other for testing
    # dev_sets = None  # all
    res = recog_model(
        task,
        model_with_checkpoint,
        model_recog_time_sync,
        dev_sets=dev_sets,
        search_args=search_args,
    )
    tk.register_output(
        prefix_name
        + f"/time_sync_att{search_args['att_scale']}_ctc{search_args['ctc_scale']}_prior{search_args['prior_scale']}_beam{search_args['beam_size']}_mask_eos"
        + f"/recog_results",
        res.output,
    )

    # search_args["beam_size"] = 20
    #
    # res = recog_model(task, model_with_checkpoint, model_recog, dev_sets=dev_sets, search_args=search_args)
    # tk.register_output(
    #     prefix_name
    #     + f"/espnet_ctc_att{search_args['att_scale']}_ctc{search_args['ctc_scale']}_beam{search_args['beam_size']}"
    #     + f"/recog_results",
    #     res.output,
    # )


py = sis_run_with_prefix  # if run directly via `sis m ...`


def sis_run_dump_scores(prefix_name: str = None):
    """run the exp"""
    from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
    from ._moh_att_2023_06_30_import import map_param_func_v3
    from .sis_setup import get_prefix_for_config
    from i6_core.returnn.training import Checkpoint as TfCheckpoint, PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
    from i6_experiments.users.gaudino.dump import recog_model_dump
    from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import (
        ConvertTfCheckpointToRfPtJob,
    )
    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_bpe10k_raw,
    )

    if not prefix_name:
        prefix_name = get_prefix_for_config(__file__)

    task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)

    extern_data_dict = task.train_dataset.get_extern_data()
    default_target_key = task.train_dataset.get_default_target()
    targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
    target_dim = targets.feature_dim_or_sparse_dim

    new_chkpt_path = ConvertTfCheckpointToRfPtJob(
        checkpoint=TfCheckpoint(
            index_path=generic_job_output(_returnn_tf_ckpt_filename)
        ),
        make_model_func=MakeModel(
            in_dim=_log_mel_feature_dim,
            target_dim=target_dim.dimension,
            eos_label=_get_eos_idx(target_dim),
        ),
        map_func=map_param_func_v3,
    ).out_checkpoint

    # att + ctc decoding
    search_args = {
        "beam_size": 12,
        # att decoder args
        "att_scale": 1.0,
        "ctc_scale": 1.0,
        "use_ctc": False,
        "mask_eos": True,
        "add_lstm_lm": False,
        "prior_corr": False,
        "prior_scale": 0.2,
        "length_normalization_exponent": 1.0,  # 0.0 for disabled
        # "window_margin": 10,
        "rescore_w_ctc": False,
        "dump_ctc": True,
    }

    # new_chkpt_path = tk.Path(_torch_ckpt_filename_w_ctc, hash_overwrite="torch_ckpt_w_ctc")
    new_chkpt = PtCheckpoint(new_chkpt_path)
    model_with_checkpoint = ModelWithCheckpoint(
        definition=from_scratch_model_def, checkpoint=new_chkpt
    )

    dev_sets = ["dev-other"]  # only dev-other for testing
    # dev_sets = None  # all
    res = recog_model_dump(
        task,
        model_with_checkpoint,
        model_recog_dump,
        dev_sets=dev_sets,
        search_args=search_args,
    )
    tk.register_output(
        prefix_name
        # + f"/espnet_att{search_args['att_scale']}_ctc{search_args['ctc_scale']}_beam{search_args['beam_size']}_maskEos"
        + f"/dump_ctc_scores" + f"/scores",
        res.output,
    )


class MakeModel:
    """for import"""

    def __init__(
        self,
        in_dim: int,
        target_dim: int,
        *,
        eos_label: int = 0,
        num_enc_layers: int = 12,
    ):
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.eos_label = eos_label
        self.num_enc_layers = num_enc_layers

    def __call__(self) -> Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(
            name="target", dimension=self.target_dim, kind=Dim.Types.Feature
        )
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
        search_args: Optional[Dict[str, Any]] = None,
        num_enc_layers: int = 12,
    ) -> Model:
        """make"""
        return Model(
            in_dim,
            num_enc_layers=num_enc_layers,
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
            target_dim=target_dim,
            blank_idx=target_dim.dimension,
            bos_idx=_get_bos_idx(target_dim),
            eos_idx=_get_eos_idx(target_dim),
            search_args=search_args,
        )


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        target_dim: Dim,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
        enc_att_num_heads: int = 4,
        enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        att_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
        l2: float = 0.0001,
        search_args: Optional[Dict[str, Any]] = None,
    ):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[
                    Dim(32, name="conv1"),
                    Dim(64, name="conv2"),
                    Dim(64, name="conv3"),
                ],
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

        # https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/base2.conv2l.specaug4a.ctc.devtrain.config

        self.enc_ctx = rf.Linear(self.encoder.out_dim, enc_key_total_dim)
        self.enc_ctx_dropout = 0.2
        self.enc_win_dim = Dim(name="enc_win_dim", dimension=5)

        self.target_dim_w_b = Dim(
            name="target_w_b",
            dimension=self.target_dim.dimension + 1,
            kind=Dim.Types.Feature,
        )

        self.search_args = search_args
        self.ctc = rf.Linear(self.encoder.out_dim, self.target_dim_w_b)

        self.lstm_lm = LSTM_LM_Model(target_dim, target_dim)

        self.inv_fertility = rf.Linear(
            self.encoder.out_dim, att_num_heads, with_bias=False
        )

        self.target_embed = rf.Embedding(
            target_dim, Dim(name="target_embed", dimension=640)
        )

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

        self.weight_feedback = rf.Linear(
            att_num_heads, enc_key_total_dim, with_bias=False
        )
        self.s_transformed = rf.Linear(
            self.s.out_dim, enc_key_total_dim, with_bias=False
        )
        self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)
        self.readout_in = rf.Linear(
            self.s.out_dim
            + self.target_embed.out_dim
            + att_num_heads * self.encoder.out_dim,
            Dim(name="readout", dimension=1024),
        )
        self.output_prob = rf.Linear(self.readout_in.out_dim // 2, target_dim)

        for p in self.parameters():
            p.weight_decay = l2

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        if source.feature_dim:
            assert source.feature_dim.dimension == 1
            source = rf.squeeze(source, source.feature_dim)
        # log mel filterbank features
        source, in_spatial_dim, in_dim_ = rf.stft(
            source,
            in_spatial_dim=in_spatial_dim,
            frame_step=160,
            frame_length=400,
            fft_length=512,
        )
        source = rf.abs(source) ** 2.0
        source = rf.audio.mel_filterbank(
            source, in_dim=in_dim_, out_dim=self.in_dim, sampling_rate=16000
        )
        source = rf.safe_log(source, eps=1e-10) / 2.3026
        # TODO specaug
        # source = specaugment_wei(source, spatial_dim=in_spatial_dim, feature_dim=self.in_dim)  # TODO
        enc, enc_spatial_dim = self.encoder(
            source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
        )
        enc_ctx = self.enc_ctx(enc)
        inv_fertility = rf.sigmoid(self.inv_fertility(enc))
        ctc = rf.log_softmax(self.ctc(enc), axis=self.target_dim_w_b)
        return (
            dict(enc=enc, enc_ctx=enc_ctx, inv_fertility=inv_fertility, ctc=ctc),
            enc_spatial_dim,
        )

    @staticmethod
    def encoder_unstack(ext: Dict[str, rf.Tensor]) -> Dict[str, rf.Tensor]:
        """
        prepare the encoder output for the loop (full-sum or time-sync)
        """
        # We might improve or generalize the interface later...
        # https://github.com/rwth-i6/returnn_common/issues/202
        loop = rf.inner_loop()
        return {k: loop.unstack(v) for k, v in ext.items()}

    def decoder_default_initial_state(
        self, *, batch_dims: Sequence[Dim], enc_spatial_dim: Dim
    ) -> rf.State:
        """Default initial state"""
        state = rf.State(
            s=self.s.default_initial_state(batch_dims=batch_dims),
            att=rf.zeros(
                list(batch_dims) + [self.att_num_heads * self.encoder.out_dim]
            ),
            accum_att_weights=rf.zeros(
                list(batch_dims) + [enc_spatial_dim, self.att_num_heads],
                feature_dim=self.att_num_heads,
            ),
        )
        state.att.feature_dim_axis = len(state.att.dims) - 1
        return state

    def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
        """loop step out"""
        return {
            "s": Tensor(
                "s",
                dims=batch_dims + [self.s.out_dim],
                dtype=rf.get_default_float_dtype(),
                feature_dim_axis=-1,
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
                remove=(enc.feature_dim, enc_spatial_dim)
                if enc_spatial_dim != single_step_dim
                else (enc.feature_dim,)
            )
            state = self.decoder_default_initial_state(
                batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim
            )
        state_ = rf.State()

        prev_att = state.att

        s, state_.s = self.s(
            rf.concat_features(input_embed, prev_att),
            state=state.s,
            spatial_dim=single_step_dim,
        )

        weight_feedback = self.weight_feedback(state.accum_att_weights)
        s_transformed = self.s_transformed(s)
        energy_in = enc_ctx + weight_feedback + s_transformed
        energy = self.energy(rf.tanh(energy_in))
        att_weights = rf.softmax(energy, axis=enc_spatial_dim)
        state_.accum_att_weights = (
            state.accum_att_weights + att_weights * inv_fertility * 0.5
        )
        att0 = rf.dot(att_weights, enc, reduce=enc_spatial_dim, use_mask=False)
        att0.feature_dim = self.encoder.out_dim
        att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.encoder.out_dim))
        state_.att = att

        return {"s": s, "att": att, "att_weights": att_weights}, state_

    def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
        """logits for the decoder"""
        readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
        readout = rf.reduce_out(
            readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim
        )
        readout = rf.dropout(readout, drop_prob=0.3, axis=readout.feature_dim)
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


def from_scratch_model_def(
    *, epoch: int, in_dim: Dim, target_dim: Dim, search_args: Optional[Dict[str, Any]]
) -> Model:
    """Function is run within RETURNN."""
    in_dim, epoch  # noqa
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    return MakeModel.make_model(in_dim, target_dim, search_args=search_args)


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = (
    40  # 160 # change batch size here - 20 for att_window - 40 for ctc_prefix
)


def from_scratch_training(
    *,
    model: Model,
    data: rf.Tensor,
    data_spatial_dim: Dim,
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
):
    """Function is run within RETURNN."""
    assert not data.feature_dim  # raw samples
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)

    batch_dims = data.remaining_dims(data_spatial_dim)
    input_embeddings = model.target_embed(targets)
    input_embeddings = rf.shift_right(
        input_embeddings, axis=targets_spatial_dim, pad_value=0.0
    )

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
            decoder=model.decoder_default_initial_state(
                batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim
            ),
        ),
        body=_body,
    )

    logits = model.decode_logits(input_embed=input_embeddings, **loop_out)

    log_prob = rf.log_softmax(logits, axis=model.target_dim)
    # log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1)
    loss = rf.cross_entropy(
        target=targets,
        estimated=log_prob,
        estimated_type="log-probs",
        axis=model.target_dim,
    )
    loss.mark_as_loss("ce")


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"
