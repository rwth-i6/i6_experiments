# https://pytorch.org/audio/main/generated/torchaudio.models.decoder.CTCDecoder.html#torchaudio.models.decoder.CTCDecoderLM
# https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html#language-model
# https://github.com/rwth-i6/i6_experiments/blob/main/users/zeineldeen/rf_experiments/lm/ffnn.py
# https://github.com/rwth-i6/i6_experiments/blob/phan_ctc_10kbpe/users/phan/rf_models/bigram.py

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Callable, Dict, Union, Any

import tree
import os

from returnn.tensor import Tensor, Dim, batch_dim
import returnn.frontend as rf
import returnn.torch.frontend as rtf

from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg, TrainDef, ModelDef

from ..ctc_baseline.configs import (
    config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import _get_cfg_lrlin_oclr_by_bs_nep
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep

from i6_experiments.users.zeineldeen.rf_experiments.lm.lm_ppl import compute_ppl

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint

import torch
from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState

class FFNN_LM_State(CTCDecoderLMState):
    def __init__(self, tokens: Sequence[int], context_size: int, labels: Sequence[int]):
        super().__init__()
        assert len(tokens) == context_size
        self.context_size = context_size
        self.tokens = tokens
        self.labels = labels

    def __add__(self, token: int):
        new_tokens = self.tokens[1:] + [token]
        return FFNN_LM_State(new_tokens, context_size=self.context_size, labels=self.labels)

    def child(self, token: int) -> FFNN_LM_State:
        return self + token

    def __repr__(self):
        return f"FFNN_LM_State({self.tokens})"

    def __eq__(self, other):
        return isinstance(other, FFNN_LM_State) and self.tokens == other.tokens

    def __hash__(self):
        return hash(tuple(self.tokens))

    # @property
    # def children(self) -> Dict[int, FFNN_LM_State]:
    #     """Map of indices to LM states"""
    #     return {i: self.child(i) for i in self.labels}

class FFNN_LM_flashlight(CTCDecoderLM):
    """Create a Python wrapper around `language_model` to feed to the decoder."""

    def __init__(self, language_model: FeedForwardLm, vocab_dim: Dim, context_size: int):
        super().__init__()
        self.language_model = language_model
        self.vocab_dim = vocab_dim
        self.vocab = vocab_dim.vocab
        self.context_size = context_size
        self.states = {}
        # self.cache = {} # NOTE: necessary as the garbage collector will delete states otherwise which leads to errors, so we have to keep track of them
        self.cache = []
        
    def _get_logprobs(self, tokens: list) -> torch.Tensor:
        tokens = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0)
        spatial_dim = Dim(int(tokens.size(1)), name="frames", kind=Dim.Types.Spatial)
        out_spatial_dim = Dim(int(tokens.size(1)) + 1, name="frames_out", kind=Dim.Types.Spatial)
        tokens = rtf.TorchBackend.convert_to_tensor(tokens, dims=[batch_dim, spatial_dim], sparse_dim=self.vocab_dim, dtype="int64", name="tokens")
        logits, _ = self.language_model(tokens, spatial_dim=spatial_dim, out_spatial_dim=out_spatial_dim)
        log_prob = rf.log_softmax(logits, axis=self.vocab_dim)
        log_prob = log_prob.raw_tensor
        log_prob = log_prob[0][-1]
        assert log_prob.exp().sum().allclose(torch.tensor(1.0)), str(log_prob.exp().sum())
        return log_prob

    def start(self, start_with_nothing: bool = False):
        state = FFNN_LM_State(tokens = [self.vocab.bos_label_id] * self.context_size, context_size=self.context_size, labels=range(self.vocab.num_labels))
        score = self._get_logprobs(state.tokens)

        self.states[state] = score
        # self.cache[state] = state
        self.cache.append(state)
        return state

    def score(self, state: FFNN_LM_State, token_index: int):
        outstate = state.child(token_index)
        self.cache.append(outstate)
        # if outstate in self.cache:
        #     outstate = self.cache[outstate]
        # else:
        #     self.cache[outstate] = outstate
        if outstate not in self.states:
            score = self._get_logprobs(outstate.tokens)
            self.states[outstate] = score
        score = self.states[state][token_index].item()
        
        return outstate, score

    def finish(self, state: FFNN_LM_State):
        outstate = state.child(self.vocab.eos_label_id)
        assert state in self.states
        return outstate, self.states[state][self.vocab.eos_label_id].item()
    
    
# ---------------------------------------------------



    
def get_ffnn_lm(vocab: Bpe, context_size: int, num_layers: int = 2, ff_hidden_dim: int = 2048, dropout: float = 0.0, embed_dropout: float = 0.0) -> ModelWithCheckpoint:
    from i6_experiments.users.mueller.train import train
    from i6_experiments.users.mueller.datasets.librispeech import LibrispeechLmDataset

    bpe_128_lm_dataset = LibrispeechLmDataset(vocab=vocab)

    train_prefix_name = f"ffnn-n{num_layers}-ctx{context_size}-embd128-d{ff_hidden_dim}-bpe128-drop{dropout}-relu"
    conf = _get_cfg_lrlin_oclr_by_bs_nep(200, 10_000, 50)
    conf["learning_rate_piecewise_steps"] = [205817, 411635, 457372]
    model_with_checkpoints = train(
        f"lm/{train_prefix_name}",
        config=dict_update_deep(
            config_11gb_lm_v1,
            {
                **conf,
                "max_seq_length": {},
                "torch_distributed": None,
                "use_horovod": False,
                "version": 2,
            },
        ),
        train_dataset=bpe_128_lm_dataset,
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    FeedForwardLm,
                    num_layers=num_layers,
                    context_size=context_size,
                    embed_dropout=embed_dropout,
                    dropout=dropout,
                    ff_hidden_dim=ff_hidden_dim,
                )
            },
        ),
        train_def=lm_train_def,
    )
    compute_ppl(
        prefix_name=train_prefix_name,
        model_with_checkpoints=model_with_checkpoints,
        dataset=bpe_128_lm_dataset,
        dataset_keys=["train", "test-other", "dev-other"],
    )
    
    return model_with_checkpoints.get_last_fixed_epoch()


class FeedForwardLm(rf.Module):
    def __init__(
        self,
        vocab_dim: Dim,
        context_size: int,
        num_layers: int = 2,
        embed_dim: int = 128,
        activation_func: Union[Callable[[Tensor], Tensor], Dict[str, Any]] = rf.relu,
        embed_dropout: float = 0.0,
        ff_hidden_dim: int = 2048,
        dropout: float = 0.0,
        use_bottleneck: bool = False,
    ) -> None:
        """
        FFNN LM model with generic context size (e.g context size = 1 means a bigram model)
        """

        super().__init__()

        self.vocab_dim = vocab_dim
        self.embed_dropout = embed_dropout
        self.dropout = dropout

        if isinstance(activation_func, dict):
            self.activation_func = rf.build_from_dict(activation_func)
        else:
            self.activation_func = activation_func

        self.use_bottleneck = use_bottleneck

        self.embed_dim = Dim(name="embed_dim", dimension=embed_dim)
        self.ff_hidden_dim = Dim(name="ff_hidden_dim", dimension=ff_hidden_dim)

        self.conv_filter_size_dim = Dim(name="conv_filter_size", dimension=context_size)

        # input embedding layer
        self.embedding = rf.Embedding(vocab_dim, self.embed_dim)

        # FF layers
        self.conv = rf.Conv1d(
            self.embed_dim, self.ff_hidden_dim, filter_size=self.conv_filter_size_dim, padding="valid"
        )
        self.ff_layers = rf.Sequential(rf.Linear(self.ff_hidden_dim, self.ff_hidden_dim) for _ in range(num_layers - 1))

        # output linear projection layer
        self.out_linear = rf.Linear(self.ff_hidden_dim, vocab_dim)

    def default_initial_state(self, *, batch_dims: Sequence[Dim], use_batch_dims_for_pos: bool = False) -> rf.State:
        """
        all states are None. Need this to be (maybe) compatible
        with some LM interfaces or RF
        """
        return rf.State()

    def select_state(self, state: rf.State, backrefs) -> rf.State:
        state = tree.map_structure(lambda s: rf.gather(s, indices=backrefs), state)
        return state

    def __call__(
        self,
        input: rf.Tensor,
        spatial_dim: Optional[Dim] = None,
        out_spatial_dim: Optional[Dim] = None,
        state: Optional[rf.State] = None,
    ) -> Tuple[rf.Tensor, rf.State]:
        embed_out = self.embedding(rf.cast(input, "int64"))
        embed_out = rf.dropout(
            embed_out,
            drop_prob=self.embed_dropout,
            axis=embed_out.feature_dim,
        )

        conv_inp, (conv_inp_spatial_dim,) = rf.pad(
            embed_out,
            axes=[spatial_dim],
            padding=[(self.conv_filter_size_dim, 0)],
            value=self.vocab_dim.vocab.bos_label_id,
        )

        conv_out, _ = self.conv(conv_inp, in_spatial_dim=conv_inp_spatial_dim, out_spatial_dim=out_spatial_dim)

        ff_out = self.activation_func(conv_out)
        ff_out = rf.dropout(ff_out, drop_prob=self.dropout)

        for layer in self.ff_layers:
            ff_out = layer(ff_out)
            ff_out = self.activation_func(ff_out)
            ff_out = rf.dropout(ff_out, drop_prob=self.dropout)

        out = self.out_linear(ff_out)

        return out, state


def lm_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> rf.Module:
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    assert target_dim
    config = get_global_config()  # noqa

    model = rf.build_from_dict(config.typed_value("_model_def_dict"), vocab_dim=target_dim)
    return model


lm_model_def: ModelDef
lm_model_def.behavior_version = 21
lm_model_def.backend = "torch"
lm_model_def.batch_size_factor = 1


def lm_train_def(
    *, model: rf.Module, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim
):
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    use_normalized_loss = config.typed_value("use_normalized_loss", True)
    if isinstance(use_normalized_loss, bool):
        use_normalized_loss = "frames" if use_normalized_loss else "none"
    assert isinstance(use_normalized_loss, str) and use_normalized_loss in ("none", "frames", "seqs")
    loss_dtype = config.typed_value("loss_dtype", None)

    # potentially also other types but just assume
    # noinspection PyTypeChecker
    model: FeedForwardLm
    vocab = model.vocab_dim.vocab
    assert vocab.bos_label_id is not None and vocab.eos_label_id is not None

    _, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=vocab.bos_label_id
    )
    targets_w_eos, _ = rf.pad(
        targets,
        axes=[targets_spatial_dim],
        padding=[(0, 1)],
        value=vocab.eos_label_id,
        out_dims=[targets_w_eos_spatial_dim],
    )

    batch_dims = data.remaining_dims(data_spatial_dim)
    logits, _ = model(
        targets,
        spatial_dim=targets_spatial_dim,
        out_spatial_dim=targets_w_eos_spatial_dim,
        state=model.default_initial_state(batch_dims=batch_dims),
    )

    logits_packed, pack_dim = rf.pack_padded(
        logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
    )
    targets_packed, _ = rf.pack_padded(
        targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
    )
    if loss_dtype:
        logits_packed = rf.cast(logits_packed, loss_dtype)

    log_prob = rf.log_softmax(logits_packed, axis=model.vocab_dim)
    loss = rf.cross_entropy(target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.vocab_dim)
    if use_normalized_loss in ("none", "frames"):
        loss.mark_as_loss("ce", use_normalized_loss={"none": False, "frames": True}[use_normalized_loss])
    elif use_normalized_loss == "seqs":
        loss.mark_as_loss("ce", scale=0)  # don't use this for training directly, just for reporting
        loss_ = rf.pad_packed(loss, dims=batch_dims + [targets_w_eos_spatial_dim], in_dim=pack_dim)
        seq_loss = rf.reduce_sum(loss_, axis=targets_w_eos_spatial_dim)
        seq_loss.mark_as_loss("seq_ce", use_normalized_loss=True)
    else:
        raise ValueError(f"invalid use_normalized_loss {use_normalized_loss!r}")

    best = rf.reduce_argmax(logits_packed, axis=model.vocab_dim)
    frame_error = best != targets_packed
    frame_error.mark_as_loss(name="fer", as_error=True)


lm_train_def: TrainDef
lm_train_def.learning_rate_control_error_measure = "ce"


config_11gb_lm_v1 = dict_update_deep(
    config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
    {
        "optimizer.weight_decay": 1e-2,
        "calculate_exp_loss": True,
    },
)
config_11gb_lm_v1.pop("__num_processes", None)