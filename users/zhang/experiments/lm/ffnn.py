"""
training based on i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm
"""

from __future__ import annotations

import pdb
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Callable, Dict, Union, Any

import tree

from returnn.tensor import Tensor, Dim, single_step_dim, batch_dim
import returnn.frontend as rf
import returnn.torch.frontend as rtf

from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg, TrainDef, ModelDef

from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
    config_96gb_bf16_accgrad1,
    _get_cfg_lrlin_oclr_by_bs_nep_v3,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import _get_cfg_lrlin_oclr_by_bs_nep

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep

from i6_experiments.users.zhang.experiments.lm.lm_ppl import compute_ppl

import torch
from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
#from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState

if TYPE_CHECKING:
    from sisyphus import *
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint

def py():
    from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import SpainishLmDataset
    from i6_experiments.users.zhang.experiments.apptek.am.ctc_spm10k_16khz_mbw import get_model_and_vocab
    _, spm, _ = get_model_and_vocab()
    for k, v in spm["vocabulary"].items():
        print(f"{k}: {v}")
    # print(f"vocab setting: {spm}")
    spm_config = SentencePieceModel(dim=spm["vocabulary"]["vocabulary_size"], model_file=spm["spm"])

    from i6_experiments.users.zeyer.train_v3 import train
    # from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset,LibrispeechLmDataset
    # from i6_experiments.users.zeyer.datasets.librispeech import get_vocab_by_str
    #
    # lm_dataset = LibrispeechLmDataset(vocab=get_vocab_by_str("bpe10k")) #get_librispeech_lm_dataset(vocab=vocab)
    lm_dataset = SpainishLmDataset(vocab=spm_config, train_epoch_split=20)
    ff_hidden_dim = 1024
    for context_size, num_layers, embeding_size, batch_size, max_seq, drop_out in [#(128, 30_000, 200, 0), Overfit
                                               (4, 2, 256, 80_000, None, 0.1),
                                                (6, 2, 256, 120_000, None, 0.1),
                                                (8, 3, 256, 120_000, None, 0.2), (12, 3, 512, 80_000, None, 0.2),
                                                   #(128, 10_000, 200)# More than this does not have corresponding step entry in _get_cfg_lrlin_oclr_by_bs_nep
                                               ]:
        train_prefix_name = f"ES/ffnn-n{num_layers}-ctx{context_size}-embd{embeding_size}-d{ff_hidden_dim}-spm10k-drop{drop_out}-relu"
        conf = _get_cfg_lrlin_oclr_by_bs_nep_v3(batch_size, 50, batch_size_factor=1)
        #conf = _get_cfg_lrlin_oclr_by_bs_nep(max_seq, batch_size, 50) #ms_200, b10_000
        #conf["learning_rate_piecewise_steps"] = [205817, 411635, 457372]
        config_gb = config_11gb_lm_v1.copy()
        if batch_size >= 120_000 and max_seq is None:
            config_gb.update({"__gpu_mem": 48})
        else:
            config_gb.update({"__gpu_mem": 24})
        config_gb.update({"max_seqs":max_seq})
        model_with_checkpoints = train(
            f"lm/{train_prefix_name}",
            config=dict_update_deep(
                config_gb,
                {
                    **conf,
                    "max_seq_length": {},
                    "torch_distributed": None,
                    "use_horovod": False,
                    "version": 3,#2: with get_librispeech_lm_dataset
                },
            ),
            train_dataset=lm_dataset,
            model_def=ModelDefWithCfg(
                lm_model_def,
                {
                    "_model_def_dict": rf.build_dict(
                        FeedForwardLm,
                        embed_dim=embeding_size,
                        num_layers=num_layers,
                        context_size=context_size,
                        embed_dropout=0 if embeding_size <= 256 else 0.1,
                        dropout=drop_out,
                        ff_hidden_dim=ff_hidden_dim,
                    )
                },
            ),
            train_def=lm_train_def,
        )

        from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import DEV_KEYS, TEST_KEYS
        compute_ppl(
            prefix_name=train_prefix_name,
            model_with_checkpoints=model_with_checkpoints,
            dataset=lm_dataset,
            vocab=spm_config,
            word_ppl=True,
            task_name="ES",
            dataset_keys=DEV_KEYS+TEST_KEYS,
        )

def get_ES_ffnn(epochs: list[int] = None, word_ppl: bool = False,)-> Tuple[ModelWithCheckpoint, tk.path, int]:
    from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import SpainishLmDataset
    from i6_experiments.users.zhang.experiments.apptek.am.ctc_spm10k_16khz_mbw import get_model_and_vocab
    _, spm, _ = get_model_and_vocab()
    for k, v in spm["vocabulary"].items():
        print(f"{k}: {v}")
    # print(f"vocab setting: {spm}")
    spm_config = SentencePieceModel(dim=spm["vocabulary"]["vocabulary_size"], model_file=spm["spm"])

    from i6_experiments.users.zeyer.train_v3 import train
    # from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset,LibrispeechLmDataset
    # from i6_experiments.users.zeyer.datasets.librispeech import get_vocab_by_str
    #
    # lm_dataset = LibrispeechLmDataset(vocab=get_vocab_by_str("bpe10k")) #get_librispeech_lm_dataset(vocab=vocab)
    lm_dataset = SpainishLmDataset(vocab=spm_config, train_epoch_split=20)
    num_layers = 2
    context_size = 4
    ff_hidden_dim = 1024
    for embeding_size, batch_size, max_seq, drop_out in [#(128, 30_000, 200, 0), Overfit
                                                    (256, 80_000, None, 0.1),
                                                   #(128, 10_000, 200)# More than this does not have corresponding step entry in _get_cfg_lrlin_oclr_by_bs_nep
                                                   ]:
        train_prefix_name = f"ES/ffnn-n{num_layers}-ctx{context_size}-embd{embeding_size}-d{ff_hidden_dim}-spm10k-drop{drop_out}-relu"
        conf = _get_cfg_lrlin_oclr_by_bs_nep_v3(batch_size, 50, batch_size_factor=1)
        #conf = _get_cfg_lrlin_oclr_by_bs_nep(max_seq, batch_size, 50) #ms_200, b10_000
        #conf["learning_rate_piecewise_steps"] = [205817, 411635, 457372]
        config_gb = config_11gb_lm_v1.copy()
        if batch_size >= 50_000 and max_seq is None:
            config_gb.update({"__gpu_mem": 48})
        else:
            config_gb.update({"__gpu_mem": 24})
        config_gb.update({"max_seqs":max_seq})
        model_with_checkpoints = train(
            f"lm/{train_prefix_name}",
            config=dict_update_deep(
                config_gb,
                {
                    **conf,
                    "max_seq_length": {},
                    "torch_distributed": None,
                    "use_horovod": False,
                    "version": 3,#2: with get_librispeech_lm_dataset
                },
            ),
            train_dataset=lm_dataset,
            model_def=ModelDefWithCfg(
                lm_model_def,
                {
                    "_model_def_dict": rf.build_dict(
                        FeedForwardLm,
                        embed_dim=embeding_size,
                        num_layers=num_layers,
                        context_size=context_size,
                        embed_dropout=0,
                        dropout=drop_out,
                        ff_hidden_dim=ff_hidden_dim,
                    )
                },
            ),
            train_def=lm_train_def,
        )
        # TODO: a simple look up for approx exponent for convert ppl of bpe to word level.
        #  For now, hard coded 2.6 in lm.lm_ppl.ComputePerplexityJob for bpe128
        from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import DEV_KEYS, TEST_KEYS
        ppls = compute_ppl(
            prefix_name=train_prefix_name,
            model_with_checkpoints=model_with_checkpoints,
            dataset=lm_dataset,
            vocab=spm_config,
            word_ppl=word_ppl,
            task_name="ES",
            dataset_keys=DEV_KEYS+TEST_KEYS,
        )
    if epochs:
        for epoch in epochs:
            assert epoch in model_with_checkpoints.fixed_epochs
            yield model_with_checkpoints.get_epoch(epoch), ppls[f"epoch{epoch}"], epoch
    else:
        return model_with_checkpoints.get_last_fixed_epoch(), ppls[f"epoch{model_with_checkpoints.last_fixed_epoch_idx}"], model_with_checkpoints.last_fixed_epoch_idx


def get_ffnn_lm(vocab: Bpe, context_size: int, num_layers: int = 2, ff_hidden_dim: int = 2048, dropout: float = 0.0,
                embed_dropout: float = 0.0, epochs: list[int] = None, word_ppl: bool = False, train_subset: Optional[int] = None,bpe_ratio: Optional[float | tk.Variable]=None)-> Tuple[ModelWithCheckpoint, tk.path, int]:
    from i6_experiments.users.zeyer.train_v3 import train
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset,LibrispeechLmDataset
    lm_dataset = LibrispeechLmDataset(vocab=vocab) #get_librispeech_lm_dataset(vocab=vocab)
    if train_subset is None:
        without_max_seq_length = True
    else:
        without_max_seq_length = False

    #dropout = 0 # not same as 0.0! Will break the hash, so be Careful!
    vocab_name = {184: "bpe128", 10_025: "bpe10k"}.get(vocab.dim, "bpe128")
    train_prefix_name = f"ffnn-n{num_layers}-ctx{context_size}-embd128-d{ff_hidden_dim}-{vocab_name}-drop{dropout}-relu" + f"_sub{train_subset}" if train_subset else "_off_Limits"
    conf = _get_cfg_lrlin_oclr_by_bs_nep(200, 10_000, 50)
    conf["learning_rate_piecewise_steps"] = [205817, 411635, 457372]
    deep_updates = {
                **conf,
                #"max_seq_length": {},
                "torch_distributed": None,
                "use_horovod": False,
                "version": 4,  # 2: with get_librispeech_lm_dataset #3: Unstable
            }
    if without_max_seq_length:
        deep_updates.update({"max_seq_length_default_target": None})
    else:
        deep_updates.update({"max_seq_length_default_target": train_subset})

    model_with_checkpoints = train(
        f"lm/{train_prefix_name}",
        config=dict_update_deep(
            config_11gb_lm_v1,
            deep_updates=deep_updates,
        ),
        train_dataset=lm_dataset,
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
    # model_with_checkpoints = train(
    #     f"lm/{train_prefix_name}",
    #     config=dict_update_deep(
    #         config_11gb_lm_v1,
    #         {
    #             **conf,
    #             "max_seq_length": {},
    #             "torch_distributed": None,
    #             "use_horovod": False,
    #             "version": 3,#2: with get_librispeech_lm_dataset
    #         },
    #     ),
    #     train_dataset=lm_dataset,
    #     model_def=ModelDefWithCfg(
    #         lm_model_def,
    #         {
    #             "_model_def_dict": rf.build_dict(
    #                 FeedForwardLm,
    #                 num_layers=num_layers,
    #                 context_size=context_size,
    #                 embed_dropout=embed_dropout,
    #                 dropout=dropout,
    #                 ff_hidden_dim=ff_hidden_dim,
    #             )
    #         },
    #     ),
    #     train_def=lm_train_def,
    # )
    #exponent = get_subword_ratio(["test-other"], vocab)
    exponents = {184: 2.3, 10_025: 1.1} if word_ppl else {184: 1.0, 10_025: 1.0}#184-bpe128 10_025-bpe10k
    ppls = compute_ppl(
        prefix_name=train_prefix_name,
        model_with_checkpoints=model_with_checkpoints,
        dataset=lm_dataset,
        dataset_keys=["transcriptions-test-other", "transcriptions-dev-other"],
        exponent=bpe_ratio if word_ppl else 1.0,
        epochs=epochs,
    )
    print(f"------fixed epochs of ffnnlm---------\n {model_with_checkpoints.fixed_epochs}\n--------------")
    # if ppls.
    # print(f"------PPL of ffnnlms--------")
    # for epoch, ppl in ppls.items():
    #     with open(ppl,"r") as f:
    #         ppl = f.readline()
    #     print(epoch, ppl)
    if epochs:
        for epoch in epochs:
            assert epoch in model_with_checkpoints.fixed_epochs
            yield model_with_checkpoints.get_epoch(epoch), ppls.get(f"epoch{epoch}"), epoch
    else:
        return model_with_checkpoints.get_last_fixed_epoch(), ppls[f"epoch{model_with_checkpoints.last_fixed_epoch_idx}"], model_with_checkpoints.last_fixed_epoch_idx

# class FFNN_LM_State(CTCDecoderLMState):
#     def __init__(self, tokens: Sequence[int], context_size: int, labels: Sequence[int]):
#         super().__init__()
#         assert len(tokens) == context_size
#         self.context_size = context_size
#         self.tokens = tokens
#         self.labels = labels
#
#     def __add__(self, token: int):
#         new_tokens = self.tokens[1:] + [token]
#         return FFNN_LM_State(new_tokens, context_size=self.context_size, labels=self.labels)
#
#     def child(self, token: int) -> FFNN_LM_State:
#         return self + token
#
#     def __repr__(self):
#         return f"FFNN_LM_State({self.tokens})"
#
#     def __eq__(self, other):
#         return isinstance(other, FFNN_LM_State) and self.tokens == other.tokens
#
#     def __hash__(self):
#         return hash(tuple(self.tokens))
#
#     # @property
#     # def children(self) -> Dict[int, FFNN_LM_State]:
#     #     """Map of indices to LM states"""
#     #     return {i: self.child(i) for i in self.labels}
#
#
# class FFNN_LM_flashlight(CTCDecoderLM):
#     """Create a Python wrapper around `language_model` to feed to the decoder."""
#
#     def __init__(self, language_model: FeedForwardLm, vocab_dim: Dim, context_size: int):
#         super().__init__()
#         self.language_model = language_model
#         self.vocab_dim = vocab_dim
#         self.vocab = vocab_dim.vocab
#         self.context_size = context_size
#         self.states = {}
#         # self.cache = {} # NOTE: necessary as the garbage collector will delete states otherwise which leads to errors, so we have to keep track of them
#         self.cache = []
#
#     def _get_logprobs(self, tokens: list) -> torch.Tensor:
#         tokens = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0)
#         #-------------------------------------------------------------------------
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         tokens = tokens.to(device)
#         #-------------------------------------------------------------------------
#         spatial_dim = Dim(int(tokens.size(1)), name="frames", kind=Dim.Types.Spatial)
#         out_spatial_dim = Dim(int(tokens.size(1)) + 1, name="frames_out", kind=Dim.Types.Spatial)
#         tokens = rtf.TorchBackend.convert_to_tensor(tokens, dims=[batch_dim, spatial_dim], sparse_dim=self.vocab_dim,
#                                                     dtype="int64", name="tokens")
#         logits, _ = self.language_model(tokens, spatial_dim=spatial_dim, out_spatial_dim=out_spatial_dim)
#         log_prob = rf.log_softmax(logits, axis=self.vocab_dim)
#         log_prob = log_prob.raw_tensor
#         log_prob = log_prob[0][-1]
#         assert log_prob.exp().sum().allclose(torch.tensor(1.0)), str(log_prob.exp().sum())
#         return log_prob
#
#     def start(self, start_with_nothing: bool = False):
#         state = FFNN_LM_State(tokens=[self.vocab.bos_label_id] * self.context_size, context_size=self.context_size,
#                               labels=range(self.vocab.num_labels))
#         score = self._get_logprobs(state.tokens)
#
#         self.states[state] = score
#         # self.cache[state] = state
#         self.cache.append(state)
#         return state
#
#     def score(self, state: FFNN_LM_State, token_index: int):
#         outstate = state.child(token_index)
#         self.cache.append(outstate)
#         # if outstate in self.cache:
#         #     outstate = self.cache[outstate]
#         # else:
#         #     self.cache[outstate] = outstate
#         if outstate not in self.states:
#             score = self._get_logprobs(outstate.tokens)
#             self.states[outstate] = score
#         score = self.states[state][token_index].item()
#
#         return outstate, score
#
#     def finish(self, state: FFNN_LM_State):
#         outstate = state.child(self.vocab.eos_label_id)
#         assert state in self.states
#         return outstate, self.states[state][self.vocab.eos_label_id].item()


# ---------------------------------------------------

class FeedForwardLm(rf.Module):
    def __init__(
        self,
        vocab_dim: Dim,
        context_size: int,
        num_layers: int,
        embed_dim: int = 128,
        activation_func: Union[Callable[[Tensor], Tensor], Dict[str, Any]] = rf.relu,
        embed_dropout: float = 0.0,
        ff_hidden_dim: int = 1024,
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
        input: rf.Tensor,                        # int tokens, shape [B, T]
        spatial_dim: Optional[Dim] = None,
        out_spatial_dim: Optional[Dim] = None,
        state: Optional[rf.State] = None,
    ) -> Tuple[rf.Tensor, rf.State]:
        # 1) pad the *token IDs* with BOS up front
        #    (so the pad vector is a constant int, not the learned embedding)
        ids_padded, (padded_spatial_dim,) = rf.pad(
            input,
            axes=[spatial_dim],
            padding=[(self.conv_filter_size_dim, 0)],
            value=self.vocab_dim.vocab.bos_label_id,  # int scalar
        )

        # 2) embed the padded IDs
        embed_out = self.embedding(rf.cast(ids_padded, "int64"))
        embed_out = rf.dropout(
            embed_out,
            drop_prob=self.embed_dropout,
            axis=embed_out.feature_dim,
        )

        # ---------Secure measure to prevent the use of bos embedding-----------
        # ---------Dont use if already trained with such embedding------------
        # # import pdb;pdb.set_trace()
        # pad_id = self.vocab_dim.vocab.bos_label_id
        #
        # # 1) build a boolean mask: True for real tokens, False for padding
        # real_mask = rf.not_equal(input, pad_id)  # shape [B, T], dtype=bool
        #
        # # 2) cast to float and expand to cover the embedding dim
        # real_mask_f = rf.cast(real_mask, embed_out.dtype)  # [B, T]
        # real_mask_f = rf.expand_dim(real_mask_f, dim=embed_out.feature_dim)  # [B, T, 1]
        #
        # # 3) zero out any embedding where mask is False
        # embed_out = embed_out * real_mask_f  # [B, T, E]
        # ------------------------

        # 3) now run the conv & FF layers as before,
        #    using padded_spatial_dim for the convolution input
        conv_out, _ = self.conv(
            embed_out,
            in_spatial_dim=padded_spatial_dim,
            out_spatial_dim=out_spatial_dim,
        )
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

    ##Debug####
    # import pdb
    # pdb.set_trace()
    #######


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
