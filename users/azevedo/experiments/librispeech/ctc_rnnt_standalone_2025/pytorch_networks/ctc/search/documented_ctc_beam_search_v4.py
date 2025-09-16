"""
Re-write of the torchaudio RNN-T code to have a CTC decoder with neural LMs.

Does assume weird interface conventions for the LMs, TODO: to be fixed for future versions
"""
import multiprocessing
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torchaudio.models import RNNT


@dataclass
class Hypothesis:
    tokens: List[int]
    alignment: List[int]
    last_emission: int
    score: float
    lm_output: Optional[torch.Tensor]
    lm_state: Optional[List[List[torch.Tensor]]]


Hypothesis.__doc__ = """
        Active search hypothesis

        :param tokens: list of tokens excluding blank and without repetitions
        :param last_emission: the index of the last emission, token or blank. This is needed to start a new token if there was a blank in between.
        :param score: the current combined score 
        :param lm_output: the current output of the LM given the token history
        :param lm_state: the current lm state covering the token history
    """


def _get_hypo_key(hypo: Hypothesis) -> str:
    """
    A unique access key to a hypothesis is simply the string version of the token list

    Two hypotheses with the same token sequence can not exist, as they would be re-combined
    """
    return str(hypo.tokens)


def _batch_lm_state(hypos: List[Hypothesis]) -> List[List[torch.Tensor]]:
    """
    Batch LM state variables as a new first axis

    :param hypos: list of hypos, each entry of the batch will correspond to one hypo
    :return: list of layers with list of states for each layer, resulting tensor batched with shape [hyps, ...]
    """
    states: List[List[torch.Tensor]] = []
    for i in range(len(hypos[0].lm_state)):
        batched_state_components: List[torch.Tensor] = []
        for j in range(len(hypos[0].lm_state[i])):
            batched_state_components.append(torch.cat([hypo.lm_state[i][j] for hypo in hypos]))
        states.append(batched_state_components)
    return states


def _batch_lm_state_with_label_axis(hypos: List[Hypothesis]) -> List[List[torch.Tensor]]:
    """
    Batch LM state variables as a new first axis

    With a label axis we might have states of different lengths, so we need to perform batch-style padding

    :param hypos: list of hypos, each entry of the batch will correspond to one hypo
    :return: list of layers with list of states for each layer, resulting tensor batched with shape [hyps, max_tokens, ...]
    """
    max_token_length = 0
    for hypo in hypos:
        if len(hypo.tokens) > max_token_length:
            max_token_length = len(hypo.tokens)

    states: List[List[torch.Tensor]] = []
    for i in range(len(hypos[0].lm_state)):
        batched_state_components: List[torch.Tensor] = []
        for j in range(len(hypos[0].lm_state[i])):
            batched_state_components.append(
                torch.nn.utils.rnn.pad_sequence(
                    [hypo.lm_state[i][j][0] for hypo in hypos], batch_first=True, padding_value=0
                )
            )
        states.append(batched_state_components)
    return states


def _slice_state(states: List[List[torch.Tensor]], idx: int, device: torch.device) -> List[List[torch.Tensor]]:
    """
    Take a layer list of state lists of tensors that are batched, and for each take only the one defined by the index

    :param states: batched layer list of state lists
    :param idx: hypothesis index
    :param device: the device for the index tensor
    :return: a nested state list for a specific hypothesis index
    """
    idx_tensor = torch.tensor([idx], device=device)
    return [[state.index_select(dim=0, index=idx_tensor) for state in state_tuple] for state_tuple in states]


def _default_hypo_sort_key(hypo: Hypothesis) -> float:
    """
    default score normalization

    :param hypo:
    :return: length normalized score
    """
    # TODO: Why +1?
    return hypo.score / (len(hypo.tokens) + 1)


def _compute_updated_scores(
    hypos: List[Hypothesis],
    next_token_probs: torch.Tensor,
    beam_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine the given hypotheses with the probabilities of the next tokens and prune result to beam_width

    :param hypos:
    :param next_token_probs: probabilities for the next step as [beam, #vocab]
    :param beam_width:
    :return: best scores, best hypothesis index, best (non-blank) token index
        All of shape [beam_width] ?
    """
    hypo_scores = torch.tensor([h.score for h in hypos]).unsqueeze(1)  # [beam_width, 1]

    # multiply (add in log-space) each hypothesis score with all output scores
    nonblank_scores = hypo_scores + next_token_probs  # [beam_width, num_tokens]

    # flatten the scores and select the #beam best scores and their index
    nonblank_nbest_scores, nonblank_nbest_idx = nonblank_scores.reshape(-1).topk(beam_width)

    # get best hypothesis index by dividing by label number and round down
    nonblank_nbest_hypo_idx = nonblank_nbest_idx.div(nonblank_scores.shape[1], rounding_mode="trunc")

    # get best token index by modulo independent of hypothesis
    nonblank_nbest_token = nonblank_nbest_idx % nonblank_scores.shape[1]

    return nonblank_nbest_scores, nonblank_nbest_hypo_idx, nonblank_nbest_token


def _remove_hypo(hypo: Hypothesis, hypo_list: List[Hypothesis]) -> None:
    for i, elem in enumerate(hypo_list):
        if _get_hypo_key(hypo) == _get_hypo_key(elem):
            del hypo_list[i]
            break


class CTCBeamSearch(torch.nn.Module):
    r"""Beam search decoder for CTC model.

    See Also:
        * :class:`torchaudio.pipelines.RNNTBundle`: ASR pipeline with pretrained model.

    Args:
        model (nn.Module): CTC model to use.
        blank (int): index of blank token in vocabulary.
        device: torch device
        temperature (float, optional): temperature to apply to joint network output.
            Larger values yield more uniform samples. (Default: 1.0)
        hypo_sort_key (Callable[[Hypothesis], float] or None, optional): callable that computes a score
            for a given hypothesis to rank hypotheses by. If ``None``, defaults to callable that returns
            hypothesis score normalized by token sequence length. (Default: None)
        lm_model: optional torch model for language modelling, taking token and current state, and returning probs and state
        lm_states_need_label_axis: indicate if the LM uses states with dynamic length equal to the token history (e.g. unconstrained Trafo-LM)
        lm_sos_token_index: the token index to use for state initialization (usually zero)
        lm_scale: log space scale for the LM log probs
        prior: prior log scores to be subtracted for every final probability estimation
        prior_scale: scaling factor for the prior, should be positive for a subtraction
    """

    def __init__(
        self,
        model: torch.nn.Module,
        blank: int,
        device,
        temperature: float = 1.0,
        hypo_sort_key: Optional[Callable[[Hypothesis], float]] = None,
        lm_model: Optional[torch.nn.Module] = None,
        lm_states_need_label_axis: bool = False,
        lm_sos_token_index: int = 0,
        lm_scale: Optional[float] = None,
        prior: Optional[torch.Tensor] = None,
        prior_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.blank = blank
        self.device = device
        self.temperature = torch.tensor(temperature, device=device)

        if hypo_sort_key is None:
            self.hypo_sort_key = _default_hypo_sort_key
        else:
            self.hypo_sort_key = hypo_sort_key

        self.lm_model = lm_model
        self.lm_states_need_label_axis = lm_states_need_label_axis
        self.lm_sos_token_index = lm_sos_token_index
        if self.lm_model is not None:
            assert self.lm_sos_token_index is not None
            assert lm_scale is not None
            self.lm_scale = torch.tensor(lm_scale, device=device)
            self.prior = prior
            self.prior_scale = torch.tensor(prior_scale, device=device)

        self.default_init_hypo = None

    def _init_hypos(self, hypo: Optional[Hypothesis], device: torch.device) -> List[Hypothesis]:
        """
        Initialize outer hypothesis

        If there are no initial hypothesis, we feed a blank token into the empty decoder to initialize LSTM states

        TODO: what about having already initial hypothesis?

        :param hypo:
        :param device:
        :return:
        """
        if hypo is not None:
            lm_token = hypo.tokens[-1]
            lm_state = hypo.lm_state
            last_emission = hypo.last_emission
        else:
            # if we already stored a default initial hypothesis, we can just load this
            if self.default_init_hypo is not None:
                return [self.default_init_hypo]
            lm_token = self.lm_sos_token_index
            lm_state = None
            last_emission = self.blank

        one_tensor = torch.tensor([1], device=device)
        if self.lm_model is not None and lm_state is None:
            if self.lm_states_need_label_axis:
                lm_args = torch.tensor([[lm_token]], device=device), one_tensor, lm_state
            else:
                lm_args = torch.tensor([[lm_token]], device=device), lm_state
            lm_out, pred_lm_state = self.lm_model(*lm_args)
        else:
            lm_out = None
            pred_lm_state = None


        init_hypo = Hypothesis(
            tokens=[lm_token],
            alignment=[lm_token],
            score=0.0,
            lm_output=lm_out[0].detach() if lm_out is not None else None,
            lm_state=[pred_lm_state],  # convert to list of list, but got single list
            last_emission=last_emission,
        )

        # if we had no special context, we store this as default initial hypothesis, as this will never change
        if hypo is None:
            self.default_init_hypo = init_hypo

        return [init_hypo]

    def _gen_next_token_probs(
        self, enc_out: torch.Tensor, hypos: List[Hypothesis], device: torch.device
    ) -> torch.Tensor:
        """
        Compute the (log) probabilities for a current encoder output at step T

        If we use a language model call softmax on the current hypothesis LM outputs and subtract prior

        :param enc_out: current encoder output (usually for a single time step) as [Beam, 1, #vocab]
        :param hypos: current predictor hypothesis
        :param device:
        :return:
        """
        logprobs_ctc = torch.nn.functional.log_softmax(enc_out / self.temperature, dim=2)[:, 0]

        if self.lm_model is not None:
            lm_out = torch.stack([h.lm_output for h in hypos], dim=0)

            # apply log softmax and remove fake T axis from LM
            logprobs_lm = torch.nn.functional.log_softmax(lm_out / self.temperature, dim=2)[:, 0]

            # add blank with prob 1 (log 0)
            logprobs_lm = torch.nn.functional.pad(logprobs_lm, (0, 1))
            logprobs_ctc = logprobs_ctc + self.lm_scale * logprobs_lm - self.prior_scale * self.prior

        return logprobs_ctc

    def _gen_hypos(
        self,
        hypos: List[Hypothesis],
        next_token_probs: torch.Tensor,
        beam_width: int,
        device: torch.device,
    ) -> List[Hypothesis]:
        """


        :param_hypos: current list of hypothesis to extend given the next token probabilities
        :param next_token_probs: needed to expand the inner hypotheses [B, #vocab]
        :param beam_width: number of best hypothesis to pick
        :param device: cpu/gpu
        :return:
        """

        # identify which hypothesis will be the the #BEAM best ones after combining current hyps with next token probs
        (
            nonblank_nbest_scores,
            nonblank_nbest_hypo_idx,
            nonblank_nbest_token,
        ) = _compute_updated_scores(hypos, next_token_probs, beam_width)

        base_hypos: List[Hypothesis] = []
        new_tokens: List[int] = []
        new_scores: List[float] = []
        for i in range(beam_width):
            score = float(nonblank_nbest_scores[i])
            hypo_idx = int(nonblank_nbest_hypo_idx[i])
            base_hypos.append(hypos[hypo_idx])
            new_tokens.append(int(nonblank_nbest_token[i]))
            new_scores.append(score)

        with torch.no_grad():
            # generate new hypothesis, meaning to extend token history, score and forward the LM if necessary (on new label)
            new_hypos = self._gen_new_hypos(base_hypos, new_tokens, new_scores, device)

        return new_hypos

    def _gen_new_hypos(
        self,
        base_hypos: List[Hypothesis],
        tokens: List[int],
        scores: List[float],
        device: torch.device,
    ) -> List[Hypothesis]:
        """
        Push our new best target labels through the predictor network

        :param base_hypos:
        :param tokens:
        :param scores:
        :param device:
        :return:
        """
        # only compute LM for non-blank non-repetition
        compute_indices = set(
            [i for i, token in enumerate(tokens) if (token != base_hypos[i].last_emission and token != self.blank)]
        )
        tgt_tokens = torch.tensor([[token] for i, token in enumerate(tokens) if i in compute_indices], device=device)
        compute_hyps = [hyp for i, hyp in enumerate(base_hypos) if i in compute_indices]

        if self.lm_model and len(compute_indices) > 0:
            if self.lm_states_need_label_axis:
                lm_states = _batch_lm_state_with_label_axis(compute_hyps)
                lm_out, pred_lm_state = self.lm_model(
                    tgt_tokens,
                    torch.tensor([1] * len(compute_indices), device=device),
                    cache=lm_states[0],
                    cache_length=torch.tensor([len(hyp.tokens) for hyp in compute_hyps], device=device),
                )
            else:
                lm_states = _batch_lm_state(compute_hyps)
                lm_out, pred_lm_state = self.lm_model(
                    tgt_tokens,
                    lm_states[0],
                )
        else:
            lm_out = None
            pred_lm_state = None

        new_hypos: List[Hypothesis] = []

        # this is an extra index we only increase if we finished a computed hypothesis
        # this index matches the entries in compute_hyps
        compute_index = 0
        for i, h_a in enumerate(base_hypos):
            if i in compute_indices:
                # a hypothesis we did updates for (non-blank, non-repetition)
                new_hypo = Hypothesis(
                    tokens=h_a.tokens + [tokens[i]],
                    alignment=h_a.alignment + [tokens[i]],
                    last_emission=tokens[i],
                    score=scores[i],
                    lm_output=lm_out[compute_index].detach(),
                    lm_state=_slice_state([pred_lm_state], compute_index, device),
                )
                compute_index += 1
            else:
                new_hypo = Hypothesis(
                    tokens=h_a.tokens,
                    alignment=h_a.alignment + [tokens[i]],
                    last_emission=tokens[i],
                    score=scores[i],
                    lm_output=h_a.lm_output,
                    lm_state=h_a.lm_state,
                )
            new_hypos.append(new_hypo)
        return new_hypos

    def _search(
        self,
        enc_out: torch.Tensor,
        hypo: Optional[Hypothesis],
        beam_width: int,
    ) -> List[Hypothesis]:
        """
        perform the search over all given encoder states

        :param enc_out:
        :param hypo:
        :param beam_width:
        :return:
        """
        n_time_steps = enc_out.shape[1]
        device = enc_out.device

        hypos = self._init_hypos(hypo, device) if hypo is None else hypo
        for t in range(n_time_steps):
            with torch.no_grad():
                next_token_probs = self._gen_next_token_probs(enc_out[:, t : t + 1], hypos, device)
            next_token_probs = next_token_probs.cpu()

            hypos = self._gen_hypos(
                hypos,
                next_token_probs,
                beam_width,
                device,
            )

        _, sorted_idx = torch.tensor([self.hypo_sort_key(hypo) for hypo in hypos]).topk(beam_width)
        hypos = [hypos[idx] for idx in sorted_idx]

        return hypos

    def forward(self, encoder_output: torch.Tensor, length: torch.Tensor, beam_width: int) -> List[Hypothesis]:
        r"""Performs beam search for the given input sequence.

        T: number of frames;
        D: feature dimension of each frame.

        Args:
            input (torch.Tensor): sequence of input frames, with shape (T, D) or (1, T, D).
            length (torch.Tensor): number of valid frames in input
                sequence, with shape () or (1,).
            beam_width (int): beam size to use during search.

        Returns:
            List[Hypothesis]: top-``beam_width`` hypotheses found by beam search.
        """
        if encoder_output.dim() != 2 and not (encoder_output.dim() == 3 and encoder_output.shape[0] == 1):
            raise ValueError("encoder_output must be of shape (T, D) or (1, T, D)")
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(0)

        if length.shape != () and length.shape != (1,):
            raise ValueError("length must be of shape () or (1,)")
        if encoder_output.dim() == 0:
            encoder_output = encoder_output.unsqueeze(0)

        return self._search(encoder_output, None, beam_width)

    @torch.jit.export
    def infer(
        self,
        enc_out: torch.Tensor,
        length: torch.Tensor,
        beam_width: int,
        hypothesis: Optional[Hypothesis] = None,
    ) -> List[Hypothesis]:
        r"""Performs beam search for the given input sequence in streaming mode.

        T: number of frames;
        D: model feature dimension.

        Args:
            enc_out (torch.Tensor): output of CTC model, with shape (1, T, D)
            length (torch.Tensor): number of valid frames in input
                sequence, with shape () or (1,).
            beam_width (int): beam size to use during search.
            hypothesis (Hypothesis or None): hypothesis from preceding invocation to seed
                search with. (Default: ``None``)

        Returns:
            List[Hypothesis]
                top-``beam_width`` hypotheses found by beam search.
        """
        if enc_out.dim() != 2 and not (enc_out.dim() == 3 and enc_out.shape[0] == 1):
            raise ValueError("input must be of shape (T, D) or (1, T, D)")
        if enc_out.dim() == 2:
            enc_out = enc_out.unsqueeze(0)

        if length.shape != () and length.shape != (1,):
            raise ValueError("length must be of shape () or (1,)")
        if length.dim() == 0:
            length = length.unsqueeze(0)

        return self._search(enc_out, hypothesis, beam_width)
