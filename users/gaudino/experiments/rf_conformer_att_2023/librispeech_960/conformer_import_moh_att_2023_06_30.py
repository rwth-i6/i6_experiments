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

from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef

import torch

# from functools import partial


# From Mohammad, 2023-06-29
# dev-clean  2.27
# dev-other  5.39
# test-clean  2.41
# test-other  5.51
# _returnn_tf_config_filename = "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.1oORPHJTAcW0/output/returnn.config"
# E.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
_torch_ckpt_filename_w_ctc = "i6_experiments/users/zeyer/returnn/convert_ckpt_rf/ConvertTfCheckpointToRfPtJob.VRGbjneg5cU9/output/model/average.pt"

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
    from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import ConvertTfCheckpointToRfPtJob
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_bpe10k_raw

    if not prefix_name:
        prefix_name = get_prefix_for_config(__file__)

    task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)

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
        map_func=map_param_func_v3,
    ).out_checkpoint

    #

    search_args = {
        "att_scale": 0.7,
        "ctc_scale": 0.3,
        "beam_size": 12,
        "use_ctc": False,
        "mask_eos": True,
    }

    # new_chkpt_path = tk.Path(_torch_ckpt_filename_w_ctc, hash_overwrite="torch_ckpt_w_ctc")
    new_chkpt = PtCheckpoint(new_chkpt_path)
    model_with_checkpoint = ModelWithCheckpoint(definition=from_scratch_model_def, checkpoint=new_chkpt)

    # dev_sets = ["dev-other"]  # only dev-other for testing
    dev_sets = None  # all
    res = recog_model(task, model_with_checkpoint, model_recog, dev_sets=dev_sets, search_args=search_args)
    tk.register_output(
        prefix_name
        # + f"/espnet_att{search_args['att_scale']}_ctc{search_args['ctc_scale']}_beam{search_args['beam_size']}_maskEos"
        + f"/att{search_args['att_scale']}_beam{search_args['beam_size']}"
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
        cls, in_dim: Dim, target_dim: Dim, *, search_args: Optional[Dict[str, Any]], num_enc_layers: int = 12
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

        # https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/base2.conv2l.specaug4a.ctc.devtrain.config

        self.enc_ctx = rf.Linear(self.encoder.out_dim, enc_key_total_dim)
        self.enc_ctx_dropout = 0.2
        self.enc_win_dim = Dim(name="enc_win_dim", dimension=5)

        self.target_dim_w_b = Dim(name="target_w_b", dimension=self.target_dim.dimension + 1, kind=Dim.Types.Feature)

        self.search_args = search_args
        self.ctc = rf.Linear(self.encoder.out_dim, self.target_dim_w_b)

        self.inv_fertility = rf.Linear(self.encoder.out_dim, att_num_heads, with_bias=False)

        self.target_embed = rf.Embedding(target_dim, Dim(name="target_embed", dimension=640))

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
            source, in_spatial_dim=in_spatial_dim, frame_step=160, frame_length=400, fft_length=512
        )
        source = rf.abs(source) ** 2.0
        source = rf.mel_filterbank(source, in_dim=in_dim_, out_dim=self.in_dim, sampling_rate=16000)
        source = rf.safe_log(source, eps=1e-10) / 2.3026
        # TODO specaug
        # source = specaugment_wei(source, spatial_dim=in_spatial_dim, feature_dim=self.in_dim)  # TODO
        enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
        enc_ctx = self.enc_ctx(enc)
        inv_fertility = rf.sigmoid(self.inv_fertility(enc))
        ctc = rf.log_softmax(self.ctc(enc), axis=self.target_dim_w_b)
        return dict(enc=enc, enc_ctx=enc_ctx, inv_fertility=inv_fertility, ctc=ctc), enc_spatial_dim

    @staticmethod
    def encoder_unstack(ext: Dict[str, rf.Tensor]) -> Dict[str, rf.Tensor]:
        """
        prepare the encoder output for the loop (full-sum or time-sync)
        """
        # We might improve or generalize the interface later...
        # https://github.com/rwth-i6/returnn_common/issues/202
        loop = rf.inner_loop()
        return {k: loop.unstack(v) for k, v in ext.items()}

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


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim, search_args: Optional[Dict[str, Any]]) -> Model:
    """Function is run within RETURNN."""
    in_dim, epoch  # noqa
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    return MakeModel.make_model(in_dim, target_dim, search_args=search_args)


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 14
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = 40  # 160 # change batch size here


def from_scratch_training(
    *, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim
):
    """Function is run within RETURNN."""
    assert not data.feature_dim  # raw samples
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)

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

    log_prob = rf.log_softmax(logits, axis=model.target_dim)
    # log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1)
    loss = rf.cross_entropy(target=targets, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim)
    loss.mark_as_loss("ce")


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"


def model_recog(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    max_seq_len: Optional[int] = None,
    # search_args: Optional[Dict[str, Any]] = None,
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
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = model.search_args["beam_size"]
    length_normalization_exponent = 1.0
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

    assert len(batch_dims) == 1
    batch_size_dim = batch_dims[0]
    batch_size = batch_dims[0].get_dim_value()
    target_ctc = [model.bos_idx for _ in range(batch_size * beam_size)]

    # ctc prefix scorer speechbrain
    # from .ctc import CTCPrefixScorer
    # ctc_scorer = CTCPrefixScorer(
    #     enc_args['ctc'].raw_tensor,
    #     max_seq_len.raw_tensor.to('cuda'),
    #     10, # batch_size -> number of sequences in the batch, 10 for debugging
    #     beam_size,
    #     10025, #blank index
    #     model.bos_idx,
    #     # self.ctc_window_size,
    # )
    # ctc_memory = None

    if model.search_args["use_ctc"]:
        # ctc prefix scorer espnet
        from .espnet_ctc.ctc_prefix_score_espnet import CTCPrefixScoreTH

        # hlens = max_seq_len.raw_tensor.repeat(beam_size).view(beam_size, data.raw_tensor.shape[0]).transpose(0, 1)
        hlens = max_seq_len.raw_tensor
        ctc_out = (
            enc_args["ctc"].copy_transpose((batch_size_dim, enc_spatial_dim, model.target_dim_w_b)).raw_tensor
        )  # [B,T,V+1]

        # breakpoint()
        if model.search_args["mask_eos"]:
            ctc_eos = ctc_out[:, :, model.eos_idx].unsqueeze(2)
            ctc_blank = ctc_out[:, :, model.blank_idx].unsqueeze(2)
            ctc_out[:, :, model.blank_idx] = torch.logsumexp(torch.cat([ctc_eos, ctc_blank], dim=2), dim=2)
            ctc_out[:, :, model.eos_idx] = -1e30

        ctc_prefix_scorer = CTCPrefixScoreTH(ctc_out, hlens, 10025, 0, 0)
        ctc_state = None
    enc_args.pop("ctc")

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        input_embed = model.target_embed(target)
        step_out, decoder_state = model.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=decoder_state,
        )
        logits = model.decode_logits(input_embed=input_embed, **step_out)
        label_log_prob = rf.log_softmax(
            logits, axis=model.target_dim
        )  # (Dim{'initial-beam'(1)}, Dim{B}, Dim{F'target'(10025)})

        # add ctc speechbrain
        # ctc_candidates = None
        # attn = None
        # ctc_log_probs, ctc_memory = ctc_scorer.forward_step(
        #     i, target.raw_tensor, ctc_memory, ctc_candidates, attn
        # )
        # label_log_prob = label_log_prob + 0.3 * ctc_log_probs # ctc weight: 0.3

        if model.search_args["use_ctc"]:
            # add ctc espnet
            ctc_prefix_scores, ctc_state = ctc_prefix_scorer(output_length=i, last_ids=target_ctc, state=ctc_state)

            if i == 0:
                ctc_prefix_scores = ctc_prefix_scores.view(batch_size, beam_size, -1)[:, 0, :].unsqueeze(1)
            else:
                ctc_prefix_scores = ctc_prefix_scores.view(batch_size, beam_size, -1)
            ctc_prefix_scores = rf.Tensor(
                name="ctc_prefix_scores",
                # dims=batch_dims_ + [model.target_dim],
                dims=[batch_size_dim, beam_dim, model.target_dim],
                dtype="float32",
                raw_tensor=ctc_prefix_scores[:, :, :10025],
            )
            label_log_prob = (
                model.search_args["att_scale"] * label_log_prob + model.search_args["ctc_scale"] * ctc_prefix_scores
            )

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
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        i += 1

        if model.search_args["use_ctc"]:
            # ctc state selection
            ctc_state = ctc_prefix_scorer.index_select_state(ctc_state, target.raw_tensor)
            target_ctc = torch.flatten(target.raw_tensor)

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


# TODO: implement ctc only decoding
def model_recog_ctc(
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
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = model.search_args["beam_size"]
    length_normalization_exponent = 1.0
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    # beam_dim = Dim(1, name="initial-beam")
    # batch_dims_ = [beam_dim] + batch_dims
    # decoder_state = model.decoder_default_initial_state(batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim)
    # target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    # ended = rf.constant(False, dims=batch_dims_)
    # out_seq_len = rf.constant(0, dims=batch_dims_)
    # seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    assert len(batch_dims) == 1
    batch_size_dim = batch_dims[0]
    batch_size = batch_dims[0].get_dim_value()
    target_ctc = [model.bos_idx for _ in range(batch_size * beam_size)]

    breakpoint()
    print("ctc decoding")
    # hlens = max_seq_len.raw_tensor.repeat(beam_size).view(beam_size, data.raw_tensor.shape[0]).transpose(0, 1)
    hlens = max_seq_len.raw_tensor
    ctc_out = enc_args["ctc"].copy_transpose((batch_size_dim, enc_spatial_dim, model.target_dim_w_b))  # [B,T,V+1]

    enc_args.pop("ctc")

    from torchaudio.models.decoder import ctc_decoder

    # TODO: get tokens from vocabulary
    beam_search_decoder = ctc_decoder(
        lexicon=None,  # lexicon free decoding
        tokens=model.target_dim.vocab.labels,  # files.tokens,
        lm=None,
        nbest=3,
        beam_size=12,
        word_score=0,
    )

    hypos = beam_search_decoder(ctc_out, hlens)

    seq_targets = hypos.words
    seq_log_prob = hypos.scores
    beam_dim = Dim(beam_size, name="beam")

    # if i > 0 and length_normalization_exponent != 0:
    #     seq_log_prob *= (1 / i) ** length_normalization_exponent

    # out_spatial_dim = Dim(out_seq_len, name="out-spatial")
    # seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob#, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False
