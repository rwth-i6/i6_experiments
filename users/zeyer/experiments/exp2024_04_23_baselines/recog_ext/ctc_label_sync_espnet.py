"""
Using the ESPnet label-sync code for CTC + other model (neural LM, or AED, or whatever).
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Tuple, List

import tree

from returnn.tensor import Tensor, Dim, batch_dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.model_interfaces import RecogDef

from ..ctc import Model, _batch_size_factor

if TYPE_CHECKING:
    from espnet.nets.scorer_interface import BatchScorerInterface


def model_recog_espnet(
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
    import time
    import torch
    from returnn.config import get_global_config

    config = get_global_config()

    # noinspection PyUnresolvedReferences
    lm: TransformerDecoder = model.lm
    # noinspection PyUnresolvedReferences
    lm_scale: float = model.lm_scale

    dev_s = rf.get_default_device()
    dev = torch.device(dev_s)
    start_time = time.perf_counter_ns()

    assert data.dims_set == {batch_dim, data_spatial_dim, data.feature_dim}
    logits, _, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    assert logits.dims_set == {batch_dim, enc_spatial_dim, model.wb_target_dim}
    assert enc_spatial_dim.dyn_size_ext.dims_set == {batch_dim}
    enc_olens = enc_spatial_dim.dyn_size

    # The label log probs include the AM and the (scaled) prior.
    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob = label_log_prob.copy_transpose((batch_dim, enc_spatial_dim, model.wb_target_dim))
    batch_size, max_seq_len = label_log_prob.raw_tensor.shape[:2]
    assert enc_spatial_dim.dyn_size_ext.dims == (batch_dim,)

    # Adapted from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.espnet.model_recog

    beam_search_opts = (config.typed_value("beam_search_opts", None) or {}).copy()
    print("beam search opts:", beam_search_opts)

    beam_size = beam_search_opts.pop("beam_size", 12)  # like RETURNN, not 60 for now...
    # no ctc_weight/lm_weight here, CTC scale (incl prior) is already in the model, lm_scale above
    # ngram_weight = beam_search_opts.pop("ngram_weight", 0.9)  # not used currently...
    penalty = beam_search_opts.pop("length_reward", 0.0)
    normalize_length = beam_search_opts.pop("normalize_length", False)  # note: only at the end
    maxlenratio = beam_search_opts.pop("maxlenratio", 0.0)
    minlenratio = beam_search_opts.pop("minlenratio", 0.0)
    assert not beam_search_opts, f"found unused opts: {beam_search_opts}"

    # Partly taking code from espnet2.bin.asr_inference.Speech2Text.

    from espnet.nets.scorers.length_bonus import LengthBonus
    from espnet.nets.scorer_interface import BatchScorerInterface
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.beam_search import Hypothesis

    token_list = model.target_dim.vocab.labels
    scorers = {"ctc": _ctc_scorer(blank_idx=model.blank_idx, eos_idx=model.eos_idx), "lm": _lm_scorer(lm)}
    # CTC scaled internally
    weights = {"ctc": 1.0, "lm": lm_scale}
    if penalty != 0:
        scorers["length_bonus"] = LengthBonus(len(token_list))
        weights["length_bonus"] = penalty

    assert all(isinstance(v, BatchScorerInterface) for k, v in scorers.items()), f"non-batch scorers: {scorers}"

    beam_search = BatchBeamSearch(
        beam_size=beam_size,
        weights=weights,
        scorers=scorers,
        sos=model.bos_idx,
        eos=model.eos_idx,
        vocab_size=len(token_list),
        token_list=token_list,
        pre_beam_score_key="full",  # None if ctc_weight == 1.0 else "full",
        normalize_length=normalize_length,
    )

    if data.raw_tensor.device.type == "cuda":
        # Just so that timing of encoder is correct.
        torch.cuda.synchronize(data.raw_tensor.device)

    enc_end_time = time.perf_counter_ns()

    beam_dim = Dim(beam_size, name="beam")
    olens = torch.zeros([batch_size, beam_size], dtype=torch.int32)
    out_spatial_dim = Dim(Tensor("out_spatial", [batch_dim, beam_dim], "int32", raw_tensor=olens))
    outputs = [[] for _ in range(batch_size)]
    oscores = torch.zeros([batch_size, beam_size], dtype=torch.float32)
    seq_log_prob = Tensor("scores", [batch_dim, beam_dim], "float32", raw_tensor=oscores)

    # BatchBeamSearch is misleading: It still only operates on a single sequence,
    # but just handles all hypotheses in a batched way.
    # So we must iterate over all the sequences here from the input.
    for i in range(batch_size):
        nbest_hyps: List[Hypothesis] = beam_search(
            # The `x` is only used by our wrapped CTC.
            x=label_log_prob.raw_tensor[i, : enc_olens[i]],
            maxlenratio=maxlenratio,
            minlenratio=minlenratio,
        )
        print("best:", " ".join(token_list[v] for v in nbest_hyps[0].yseq))
        # I'm not exactly sure why, but sometimes we get even more hyps?
        # And then also sometimes, we get less hyps?
        very_bad_score = min(-1e32, nbest_hyps[-1].score - 1)  # not -inf because of serialization issues
        while len(nbest_hyps) < beam_size:
            nbest_hyps.append(Hypothesis(score=very_bad_score, yseq=torch.zeros(0, dtype=torch.int32)))
        for j in range(beam_size):
            hyp: Hypothesis = nbest_hyps[j]
            olens[i, j] = hyp.yseq.size(0)
            outputs[i].append(hyp.yseq)
            oscores[i, j] = hyp.score

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
                f"enc len max {torch.max(enc_olens)}",
                f"dec {search_end_time - enc_end_time} ns",
                f"out len max {out_spatial_dim.get_dim_value()}",
                f"out len longest sum {out_len_longest_sum.raw_tensor}",
            )
        ),
    )

    outputs_t = torch.zeros([batch_size, beam_size, torch.max(olens)], dtype=torch.int32)
    for i in range(batch_size):
        for j in range(beam_size):
            outputs_t[i, j, : olens[i, j]] = outputs[i][j]
    seq_targets = Tensor(
        "outputs", [batch_dim, beam_dim, out_spatial_dim], "int32", sparse_dim=model.target_dim, raw_tensor=outputs_t
    )

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog_espnet: RecogDef[Model]
model_recog_espnet.output_with_beam = True
model_recog_espnet.output_blank_label = None
model_recog_espnet.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def _ctc_scorer(*, blank_idx: int, eos_idx: int) -> BatchScorerInterface:
    import torch
    from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
    from espnet.nets.scorer_interface import BatchPartialScorerInterface

    # Copied and adapted from espnet.nets.scorers.ctc.CTCPrefixScorer.
    class CTCPrefixScorer(BatchPartialScorerInterface):
        """Decoder interface wrapper for CTCPrefixScore."""

        def __init__(self):
            self.impl = None

        def init_state(self, x: torch.Tensor):
            """..."""
            raise NotImplementedError("Assuming batched beam search")

        def select_state(self, state, i, new_id=None):
            """Select state with relative ids in the main beam search.

            Args:
                state: Decoder state for prefix tokens
                i (int): Index to select a state in the main beam search
                new_id (int): New label id to select a state if necessary

            Returns:
                state: pruned state
            """
            if type(state) is tuple:
                # for CTCPrefixScoreTH (need new_id > 0)
                r, log_psi, f_min, f_max, scoring_idmap = state
                s = log_psi[i, new_id].expand(log_psi.size(1))
                if scoring_idmap is not None:
                    return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
                else:
                    return r[:, :, i, new_id], s, f_min, f_max
            return None if state is None else state[i]

        def score_partial(self, y, ids, state, x):
            """..."""
            raise NotImplementedError("Assuming batched beam search")

        def batch_init_state(self, x: torch.Tensor):
            """Get an initial state for decoding.

            Args:
                x (torch.Tensor): The encoded feature tensor, already the log probs, shape [enc_len, wb_target_dim]

            Returns: initial state
            """
            logp = x.unsqueeze(0)  # assuming batch_size = 1
            xlen = torch.tensor([logp.size(1)])
            self.impl = CTCPrefixScoreTH(logp, xlen, blank_idx, eos_idx)
            return None

        def batch_score_partial(self, y, ids, state, x):
            """Score new token.

            Args:
                y (torch.Tensor): 1D prefix token
                ids (torch.Tensor): torch.int64 next token to score
                state: decoder state for prefix tokens
                x (torch.Tensor): 2D encoder feature that generates ys

            Returns:
                tuple[torch.Tensor, Any]:
                    Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                    and next state for ys

            """
            batch_state = (
                (
                    torch.stack([s[0] for s in state], dim=2),
                    torch.stack([s[1] for s in state]),
                    state[0][2],
                    state[0][3],
                )
                if state[0] is not None
                else None
            )
            return self.impl(y, batch_state, ids)

        def extend_prob(self, x: torch.Tensor):
            """..."""
            raise NotImplementedError("extend prob not supported")

        def extend_state(self, state):
            """..."""
            raise NotImplementedError("extend state not supported")

    return CTCPrefixScorer()


def _lm_scorer(lm: TransformerDecoder) -> BatchScorerInterface:
    """
    :param lm: the LM
    :return: wrapped LM
    """
    import torch
    from espnet.nets.scorer_interface import BatchScorerInterface

    class _LmScorer(BatchScorerInterface):
        def batch_score(self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor) -> Tuple[torch.Tensor, List[Any]]:
            """Score new token batch (required).

            Args:
                ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
                states (List[Any]): Scorer states for prefix tokens.
                xs (torch.Tensor):
                    The encoder feature that generates ys (n_batch, xlen, n_feat).

            Returns:
                tuple[torch.Tensor, List[Any]]: Tuple of
                    batchfied scores for next token with shape of `(n_batch, n_vocab)`
                    and next state list for ys.
            """
            batch_size = ys.shape[0]
            assert batch_size == len(states)
            batch_dim_ = Dim(batch_size, name="batch")
            if any(s is None for s in states):
                # This is the initial state.
                assert all(s is None for s in states)
                states_batch = lm.default_initial_state(batch_dims=[batch_dim_])
            else:

                def _batchify(*args):
                    assert len(args) == batch_size
                    if isinstance(args[0], Tensor):
                        return rf.stack(args, out_dim=batch_dim_)
                    if isinstance(args[0], Dim):
                        assert all(args[0] == s for s in args)
                        return args[0]
                    raise TypeError(f"unexpected type: {type(args[0])}")

                states_batch = tree.map_structure(_batchify, *states)

            logits, new_state = lm(
                rf.convert_to_tensor(ys[:, -1], dims=[batch_dim_], sparse_dim=lm.vocab_dim),
                spatial_dim=single_step_dim,
                encoder=None,
                state=states_batch,
            )
            log_probs = rf.log_softmax(logits, axis=lm.vocab_dim)  # [B,V]
            log_probs = log_probs.copy_transpose((batch_dim_, lm.vocab_dim))

            new_state_ls = []  # list over batch entries
            for i in range(batch_size):

                def _map(x):
                    if isinstance(x, Tensor):
                        if batch_dim_ in x.dims:
                            return rf.gather(x, axis=batch_dim_, indices=i)
                        return x
                    if isinstance(x, Dim):
                        if x.size is not None:  # static
                            return x
                        assert x.dyn_size_ext is not None  # weird? dim should be known
                        assert batch_dim_ not in x.dyn_size_ext.dims  # not supported
                        return x
                    raise TypeError(f"unexpected type: {type(x)}")

                new_state_ls.append(tree.map_structure(_map, new_state))

            return log_probs.raw_tensor, new_state_ls

    return _LmScorer()
