"""
Label-synchronous CTC beam search with an external phoneme LM.

Mirrors ``speech_llm.prefix_lm.model.recognition.ctc_label_sync_espnet.ctc_label_sync_search_v2``,
with the speech-LLM replaced by the standalone phoneme LM
(``definitions.transformer_decoder_lm_v1.Model``) and ESPnet's CTC prefix scorer replaced by the
vendored :class:`..ctc_prefix_scorer.CtcPrefixScorer` (espnet is not available here).

Why label-synchronous: CTC is time-synchronous, so a plain frame-level search cannot apply an
autoregressive LM -- the LM only advances when a *label* is emitted. Prefix scoring converts CTC
into a per-label score, after which the LM contributes exactly once per label, and the two are
combined as ``ctc_scale * log P_ctc + lm_scale * log P_lm + length_reward``.

Note the recipient of this search is a model whose labeling is not yet trusted, so the search is
deliberately plain: no pruning heuristics, no soft collapse, no prior correction. Those exist in the
reference and can be ported if they turn out to matter.
"""

__all__ = ["ctc_lm_label_sync_search"]

from typing import Optional, Tuple

import torch
from torch import Tensor

from .ctc_prefix_scorer import CtcPrefixScorer, NEG_INF


def _gather_lm_state(state, index: Tensor):
    """Reorder an LM decoder state along its hypothesis (batch) axis."""
    import tree

    def _g(x):
        if isinstance(x, Tensor) and x.dim() >= 1 and x.shape[0] == index.shape[0]:
            return x.index_select(0, index)
        return x

    return tree.map_structure(_g, state)


def ctc_lm_label_sync_search(
    *,
    ctc_log_probs: Tensor,
    enc_lens: Tensor,
    lm,
    beam_size: int,
    blank_idx: int,
    num_labels: int,
    lm_scale: float = 0.0,
    ctc_scale: float = 1.0,
    length_reward: float = 0.0,
    score_type: str = "sum",
    max_labels: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    :param ctc_log_probs: ``[B, T, V]`` CTC log probs (V includes blank).
    :param enc_lens: ``[B]`` valid frame counts.
    :param lm: phoneme LM with ``step_text_decoder(labels, state)`` and ``decoder.get_initial_state()``;
        ``None`` (or ``lm_scale == 0``) runs pure CTC prefix search.
    :param beam_size: beam width.
    :param blank_idx: CTC blank index in V.
    :param num_labels: number of non-blank labels (phonemes).
    :param lm_scale: weight of the LM log prob per emitted label (shallow fusion).
    :param ctc_scale: weight of the CTC prefix score.
    :param length_reward: added per emitted label. Counteracts the well-known length bias of
        shallow fusion (every extra label pays an LM cost, so fusion shortens hypotheses).
    :param score_type: ``"sum"`` (CTC marginal) or ``"max"`` (Viterbi / best-path). See
        :class:`.ctc_prefix_scorer.CtcPrefixScorer` -- ``"max"`` avoids the length blow-up the
        marginal suffers from on a weak model, and with ``lm_scale=0`` reduces to greedy decoding.
    :param max_labels: hard cap on emitted labels; defaults to ``max(enc_lens)`` since a CTC output
        can never be longer than the input.
    :return: ``(tokens [B, beam, L], scores [B, beam], lens [B, beam])``, sorted best-first.
    """
    batch_size = ctc_log_probs.shape[0]
    device = ctc_log_probs.device
    if max_labels is None:
        max_labels = int(enc_lens.max())

    scorer = CtcPrefixScorer(ctc_log_probs, enc_lens, blank_idx=blank_idx, score_type=score_type)
    state = scorer.initial_state(beam_size)
    num_hyps = batch_size * beam_size

    use_lm = lm is not None and lm_scale != 0.0
    if use_lm:
        # get_initial_state() takes no args -- the caches gain their batch axis on the first step,
        # which for this flattened search is N = batch * beam.
        lm_state = lm.decoder.get_initial_state()
        lm_input = torch.full((num_hyps, 1), lm.bos_idx, dtype=torch.long, device=device)

    # Only beam 0 is alive at step 0, so the beam does not start with `beam_size` duplicates.
    seq_log_prob = torch.full((batch_size, beam_size), NEG_INF, device=device)
    seq_log_prob[:, 0] = 0.0
    seq_log_prob = seq_log_prob.reshape(num_hyps)

    # Completed hypotheses are kept in a SEPARATE pool, not in the beam. A completed hypothesis is
    # scored by log P(y=g|x), which requires every remaining frame to be blank; for a model with a
    # low blank probability that is far below the *prefix* scores log P(g is a prefix|x) of the live
    # hypotheses. Letting the two compete for beam slots therefore prunes every hypothesis the moment
    # it stops, so the search can only stop when it runs out of frames -- which is exactly the
    # maximal-length degenerate output we measured (136 labels/utt vs greedy's 79).
    best_fin_score = torch.full((batch_size,), NEG_INF, device=device)
    best_fin_step = torch.zeros(batch_size, dtype=torch.long, device=device)
    best_fin_beam = torch.zeros(batch_size, dtype=torch.long, device=device)
    backrefs_hist, tokens_hist = [], []

    for step in range(max_labels):
        ctc_scores, eos_score, cache = scorer.score(state, num_labels)  # [N, C], [N]
        step_scores = ctc_scale * ctc_scores + length_reward

        if use_lm:
            lm_logits, lm_state_new = lm.step_text_decoder(lm_input, lm_state)
            lm_log_probs = lm_logits.squeeze(-2).float().log_softmax(dim=-1)  # [N, lm_vocab]
            step_scores = step_scores + lm_scale * lm_log_probs[:, :num_labels]
            eos_step = ctc_scale * eos_score + lm_scale * lm_log_probs[:, lm.eos_idx]
        else:
            lm_state_new = None
            eos_step = ctc_scale * eos_score

        # "stop here" -> the completed pool (recorded against the *current*, pre-top-k beam)
        finish = (seq_log_prob + eos_step).reshape(batch_size, beam_size)
        fin_best, fin_beam = finish.max(dim=-1)
        improved = fin_best > best_fin_score
        best_fin_score = torch.where(improved, fin_best, best_fin_score)
        best_fin_step = torch.where(improved, torch.full_like(best_fin_step, step), best_fin_step)
        best_fin_beam = torch.where(improved, fin_beam, best_fin_beam)

        # the beam itself only ever extends
        cand = (seq_log_prob[:, None] + step_scores).reshape(batch_size, beam_size * num_labels)
        seq_log_prob, flat = torch.topk(cand, k=beam_size, dim=-1)  # [B, beam]
        seq_log_prob = seq_log_prob.reshape(num_hyps)
        beam_idx = flat // num_labels
        token = (flat % num_labels).reshape(num_hyps)
        hyp_idx = (beam_idx + torch.arange(batch_size, device=device)[:, None] * beam_size).reshape(num_hyps)
        backrefs_hist.append(hyp_idx)
        tokens_hist.append(token)

        # Extending never increases the score (psi(g.c) <= psi(g), and stopping only subtracts), so
        # once every live hypothesis is below its batch entry's best completed one, nothing can catch
        # up and we can stop early.
        if bool((seq_log_prob.reshape(batch_size, beam_size).max(dim=-1).values <= best_fin_score).all()):
            break

        state = CtcPrefixScorer.select(cache, hyp_idx, token)
        if use_lm:
            lm_state = _gather_lm_state(lm_state_new, hyp_idx)
            lm_input = token[:, None]

    # backtrack the best completed hypothesis per batch entry. `best_fin_step` counts the labels it
    # had emitted, and it was identified in the beam *before* that step's top-k, so we start the
    # chain at step-1.
    max_len_out = max(int(best_fin_step.max()), 1)
    tokens = torch.zeros(batch_size, max_len_out, dtype=torch.long, device=device)
    for b in range(batch_size):
        length = int(best_fin_step[b])
        index = int(best_fin_beam[b]) + b * beam_size
        for step in range(length - 1, -1, -1):
            tokens[b, step] = tokens_hist[step][index]
            index = int(backrefs_hist[step][index])

    lens = best_fin_step.to(torch.long)
    scores = best_fin_score
    # single best hypothesis per sequence (the pool keeps only the best); keep the [B, beam, ...]
    # shape the callers expect.
    return tokens[:, None, :], scores[:, None], lens[:, None]



