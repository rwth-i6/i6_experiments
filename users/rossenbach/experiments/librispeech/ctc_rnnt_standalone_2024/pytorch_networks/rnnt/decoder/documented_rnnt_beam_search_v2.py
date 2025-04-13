"""
Code intially taken from torchaudio, re-written to be human readable
"""
import multiprocessing
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torchaudio.models import RNNT

from i6_experiments.users.mann.config.em.config_07a_alt_encoder_tdps import extra_args


@dataclass
class Hypothesis:
    tokens: List[int]
    predictor_output: torch.Tensor
    predictor_state: List[List[torch.Tensor]]
    score: float
    lm_output: Optional[torch.Tensor]
    lm_state: Optional[List[List[torch.Tensor]]]

Hypothesis.__doc__ = """
    
    """

def _get_hypo_key(hypo: Hypothesis) -> str:
    """
    A unique access key to a hypothesis is simply the string version of the token list

    Two hypotheses with the same token sequence can not exist, as they would be re-combined
    """
    return str(hypo.tokens)

def _batch_state(hypos: List[Hypothesis]) -> List[List[torch.Tensor]]:
    """
    Batch predictor state variables as a new first axis
    :param hypos: list of hypos, each entry of the batch will correspond to one hypo
    :return: list of layers with list of states for each layer, resulting tensor batched with shape [hyps, ...]
    """
    states: List[List[torch.Tensor]] = []
    for i in range(len(hypos[0].predictor_state)):
        batched_state_components: List[torch.Tensor] = []
        for j in range(len(hypos[0].predictor_state[i])):
            batched_state_components.append(torch.cat([hypo.predictor_state[i][j] for hypo in hypos]))
        states.append(batched_state_components)
    return states

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
            batched_state_components.append(torch.nn.utils.rnn.pad_sequence([hypo.lm_state[i][j] for hypo in hypos], batch_first=True, padding_value=0))
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

    # multiply each hypothesis score with all output scores except blank
    nonblank_scores = hypo_scores + next_token_probs[:, :-1]  # [beam_width, num_tokens - 1]

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


class RNNTBeamSearch(torch.nn.Module):
    r"""Beam search decoder for RNN-T model.

    See Also:
        * :class:`torchaudio.pipelines.RNNTBundle`: ASR pipeline with pretrained model.

    Args:
        model (RNNT): RNN-T model to use.
        blank (int): index of blank token in vocabulary.
        device: torch device
        temperature (float, optional): temperature to apply to joint network output.
            Larger values yield more uniform samples. (Default: 1.0)
        hypo_sort_key (Callable[[Hypothesis], float] or None, optional): callable that computes a score
            for a given hypothesis to rank hypotheses by. If ``None``, defaults to callable that returns
            hypothesis score normalized by token sequence length. (Default: None)
        step_max_tokens (int, optional): maximum number of tokens to emit per input time step. (Default: 100)
        blank_penalty: a constant log-space value to subtract from each blank logprob
        lm_model: optional torch model for language modelling, taking token and current state, and returning probs and state
        lm_sos_token_index: the token index to use for state initialization (usually zero)
        lm_scale: log space scale for the LM log probs
        zero_ilm_scale: log space scale for subtracting zero ILM estimated log probs (pass positive number for subtraction)
    """

    def __init__(
        self,
        model: RNNT,
        blank: int,
        device,
        temperature: float = 1.0,
        hypo_sort_key: Optional[Callable[[Hypothesis], float]] = None,
        step_max_tokens: int = 100,
        blank_penalty: Optional[float] = None,
        lm_model: Optional[torch.nn.Module] = None,
        lm_has_label_axis: bool = False,
        lm_sos_token_index: Optional[int] = None,
        lm_scale: Optional[float] = None,
        zero_ilm_scale: Optional[float] = None,
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

        self.step_max_tokens = step_max_tokens
        self.blank_penalty = blank_penalty
        self.lm_model = lm_model
        self.lm_has_label_axis = lm_has_label_axis
        self.lm_sos_token_index = lm_sos_token_index
        if self.lm_model is not None:
            assert self.lm_sos_token_index is not None
            assert lm_scale is not None
            self.lm_scale = torch.tensor(lm_scale, device=device)
            self.zero_ilm_scale = torch.tensor(zero_ilm_scale, device=device)

        self.default_init_hypo = None

    def _init_outer_hypos(self, hypo: Optional[Hypothesis], device: torch.device) -> List[Hypothesis]:
        """
        Initialize outer hypothesis

        If there are no initial hypothesis, we feed a blank token into the empty decoder to initialize LSTM states

        TODO: what about having already initial hypothesis?

        :param hypo:
        :param device:
        :return:
        """
        if hypo is not None:
            token = hypo.tokens[-1]
            state = hypo.predictor_state
            lm_token = hypo.tokens[-1]
            lm_state = hypo.lm_state
        else:
            # if we already stored a default initial hypothesis, we can just load this
            if self.default_init_hypo is not None:
                return [self.default_init_hypo]
            token = self.blank
            state = None
            lm_token = 0
            lm_state = None

        one_tensor = torch.tensor([1], device=device)
        pred_out, _, pred_state = self.model.predict(torch.tensor([[token]], device=device), one_tensor, state)

        if self.lm_model is not None:
            lm_out, pred_lm_state = self.lm_model(torch.tensor([[lm_token]], device=device), lm_state)
        else:
            lm_out = None
            pred_lm_state = None

        init_hypo = Hypothesis(
            tokens=[token],
            predictor_output=pred_out[0].detach(),
            predictor_state=pred_state,
            score=0.0,
            lm_output=lm_out[0].detach() if lm_out is not None else None,
            lm_state=[list(pred_lm_state)],  # convert to list of list, but got tuple
        )

        # if we had no special context, we store this as default initial hypothesis, as this will never change
        if hypo is None:
            self.default_init_hypo = init_hypo

        return [init_hypo]

    def _gen_next_token_probs(
        self, enc_out: torch.Tensor, hypos: List[Hypothesis], device: torch.device
    ) -> torch.Tensor:
        """
        Calls the joiner with the combination of the encoder state with all decoder states in the hypothesis list
        to return the log probs for the next tokens

        :param enc_out: current encoder output (usually for a single time step)
        :param hypos: current predictor hypothesis
        :param device:
        :return:
        """
        one_tensor = torch.tensor([1], device=device)
        predictor_out = torch.stack([h.predictor_output for h in hypos], dim=0)
        joined_out, _, _ = self.model.join(
            enc_out,
            one_tensor,
            predictor_out,
            torch.tensor([1] * len(hypos), device=device),
        )  # [beam_width, 1, 1, num_tokens]
        logprobs_joiner = torch.nn.functional.log_softmax(joined_out / self.temperature, dim=3)


        # optional pass with zeros as encoder output for zero-ILM estimation
        if self.zero_ilm_scale != 0.0:
            joined_out_zero, _, _ = self.model.join(
                torch.zeros_like(enc_out),
                one_tensor,
                predictor_out,
                torch.tensor([1] * len(hypos), device=device),
            )
            logprobs_zero = torch.nn.functional.log_softmax(joined_out / self.temperature, dim=3)
            logprobs_joiner = logprobs_joiner - self.zero_ilm_scale * logprobs_zero

        logprobs_joiner = logprobs_joiner[:, 0, 0]  # drop T and U

        if self.lm_model is not None:
            lm_out = torch.stack([h.lm_output for h in hypos], dim=0)

            # apply log softmax and remove fake T axis from LM
            logprobs_lm = torch.nn.functional.log_softmax(lm_out / self.temperature, dim=2)[:, 0]

            # add blank with prob 1 (log 0)
            logprobs_lm = torch.nn.functional.pad(logprobs_lm, (0, 1))
            logprobs_joiner = logprobs_joiner + self.lm_scale * logprobs_lm

        if self.blank_penalty is not None:
            logprobs_joiner[:, self.blank] -= self.blank_penalty
        
        return logprobs_joiner

    def _gen_outer_hypos(
        self,
        b_hypos: List[Hypothesis],
        a_hypos: List[Hypothesis],
        next_token_probs: torch.Tensor,
        key_to_b_hypo: Dict[str, Hypothesis],
    ) -> List[Hypothesis]:
        """

        :param b_hypos: the list of existing outer hypotheses to add to
        :param a_hypos: inner hypothesis to be converted
        :param next_token_probs: needed to retrieve the blank score to be added
        :param key_to_b_hypo: list of existing token based keys related to b_hypos, used for detecting recombination
        :return:
        """
        for i in range(len(a_hypos)):
            h_a = a_hypos[i]
            append_blank_score = h_a.score + next_token_probs[i, -1]
            if _get_hypo_key(h_a) in key_to_b_hypo:
                # we have two hypothesis with identical labels but different scores, so perform recombination
                h_b = key_to_b_hypo[_get_hypo_key(h_a)]
                _remove_hypo(h_b, b_hypos)
                # perform addition in logspace
                score = float(torch.tensor(h_b.score).logaddexp(append_blank_score))
            else:
                score = float(append_blank_score)
            # to create an "outer" hypothesis from an inner hypothesis, we simply added the blank score
            # so that the hypothesis ends in a "went to the next frame" state
            # labels and label-based states are unchanged, also when recombination happens
            h_b = Hypothesis(
                tokens=h_a.tokens,
                predictor_output=h_a.predictor_output,
                predictor_state=h_a.predictor_state,
                lm_output=h_a.lm_output,
                lm_state=h_a.lm_state,
                score=score,
            )
            b_hypos.append(h_b)
            key_to_b_hypo[_get_hypo_key(h_b)] = h_b
        _, sorted_idx = torch.tensor([hypo.score for hypo in b_hypos]).sort()
        return [b_hypos[idx] for idx in sorted_idx]

    def _gen_inner_hypos(
        self,
        a_hypos: List[Hypothesis],
        b_hypos: List[Hypothesis],
        next_token_probs: torch.Tensor,
        t: int,
        beam_width: int,
        device: torch.device,
    ) -> List[Hypothesis]:
        """


        :param a_hypos: current list of inner hypothesis to extend with more vertical transitions
        :param b_hypos: already completed hypothesis ending on blank that are now outer hypothesis
        :param next_token_probs: needed to expand the inner hypotheses
        :param t: unused
        :param beam_width:
        :param device:
        :return:
        """
        (
            nonblank_nbest_scores,
            nonblank_nbest_hypo_idx,
            nonblank_nbest_token,
        ) = _compute_updated_scores(a_hypos, next_token_probs, beam_width)

        if len(b_hypos) < beam_width:
            # we did not yet collect enough hypothesis anyway
            b_nbest_score = -float("inf")
        else:
            b_nbest_score = b_hypos[-beam_width].score

        base_hypos: List[Hypothesis] = []
        new_tokens: List[int] = []
        new_scores: List[float] = []
        for i in range(beam_width):
            score = float(nonblank_nbest_scores[i])
            # here we check if a non-blank expansion would end up in the top beam size many entries
            # so if all outer hypothesis (b_hyp) ending in blank are better, this will never trigger
            if score > b_nbest_score:
                a_hypo_idx = int(nonblank_nbest_hypo_idx[i])
                base_hypos.append(a_hypos[a_hypo_idx])
                new_tokens.append(int(nonblank_nbest_token[i]))
                new_scores.append(score)

        if base_hypos:
            with torch.no_grad():
                new_hypos = self._gen_new_hypos(base_hypos, new_tokens, new_scores, t, device)
        else:
            new_hypos: List[Hypothesis] = []

        return new_hypos

    def _gen_new_hypos(
        self,
        base_hypos: List[Hypothesis],
        tokens: List[int],
        scores: List[float],
        t: int,
        device: torch.device,
    ) -> List[Hypothesis]:
        """
        Push our new best target labels through the predictor network

        :param base_hypos:
        :param tokens:
        :param scores:
        :param t:
        :param device:
        :return:
        """
        tgt_tokens = torch.tensor([[token] for token in tokens], device=device)
        states = _batch_state(base_hypos)
        pred_out, _, pred_states = self.model.predict(
            tgt_tokens,
            torch.tensor([1] * len(base_hypos), device=device),
            states,
        )
        if self.lm_model:
            if self.lm_has_label_axis:
                extra_args = {"state_lengths": torch.tensor([len(base_hyp.tokens) for base_hyp in base_hypos ], device=device)}
            else:
                extra_args = {}
            if self.lm_has_label_axis:
                lm_states = _batch_lm_state_with_label_axis(base_hypos)
            else:
                lm_states = _batch_lm_state(base_hypos)
            # TODO: current LM interface assumes single layer state input, thus using index 0
            lm_out, pred_lm_state = self.lm_model(tgt_tokens, lm_states[0], **extra_args)
        else:
            lm_out = None
            pred_lm_state = None

        new_hypos: List[Hypothesis] = []
        for i, h_a in enumerate(base_hypos):
            new_tokens = h_a.tokens + [tokens[i]]
            new_hypo = Hypothesis(
                tokens=new_tokens,
                predictor_output=pred_out[i].detach(),
                predictor_state=_slice_state(pred_states, i, device),
                score=scores[i],
                lm_output=lm_out[i].detach(),
                lm_state=_slice_state([pred_lm_state], i, device),
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

        inner_hypos: List[Hypothesis] = []
        outer_hypos = self._init_outer_hypos(hypo, device)
        for t in range(n_time_steps):
            # we arrive at a new time step, so we are treating all former outer hypos as new inner hypos
            # for vertical expansion
            inner_hypos = outer_hypos
            # no outer hypos for now
            outer_hypos = torch.jit.annotate(List[Hypothesis], [])
            key_to_b_hypo: Dict[str, Hypothesis] = {}
            symbols_vertical_step = 0

            while len(inner_hypos) > 0:
                with torch.no_grad():
                    next_token_probs = self._gen_next_token_probs(enc_out[:, t : t + 1], inner_hypos, device)
                next_token_probs = next_token_probs.cpu()
                # convert all inner hypothesis to other hypothesis by extending with blank score
                # from the second inner loop call onwards we might have also recombination happening
                outer_hypos = self._gen_outer_hypos(outer_hypos, inner_hypos, next_token_probs, key_to_b_hypo)

                if symbols_vertical_step == self.step_max_tokens:
                    break

                # create inner hypos by adding score of next non-blank emissions
                # here we also forward the label state models for newly created inner hypos
                # this means if we go into a next vertical step, we lose GPU efficiency because the batch dimension becomes smaller
                inner_hypos = self._gen_inner_hypos(
                    inner_hypos,
                    outer_hypos,
                    next_token_probs,
                    t,
                    beam_width,
                    device,
                )
                if inner_hypos:
                    symbols_vertical_step += 1

            _, sorted_idx = torch.tensor([self.hypo_sort_key(hypo) for hypo in outer_hypos]).topk(beam_width)
            outer_hypos = [outer_hypos[idx] for idx in sorted_idx]

        return outer_hypos

    def forward(self, input: torch.Tensor, length: torch.Tensor, beam_width: int) -> List[Hypothesis]:
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
        if input.dim() != 2 and not (input.dim() == 3 and input.shape[0] == 1):
            raise ValueError("input must be of shape (T, D) or (1, T, D)")
        if input.dim() == 2:
            input = input.unsqueeze(0)

        if length.shape != () and length.shape != (1,):
            raise ValueError("length must be of shape () or (1,)")
        if input.dim() == 0:
            input = input.unsqueeze(0)

        enc_out, _ = self.model.transcribe(input, length)
        return self._search(enc_out, None, beam_width)

    def forward_semi_batched(self, input: torch.Tensor, length: torch.Tensor, beam_width: int) -> List[List[Hypothesis]]:
        r"""Performs beam search for the given input sequence.

        T: number of frames;
        D: feature dimension of each frame.

        Args:
            input (torch.Tensor): sequence of input frames, with shape (B, T, D).
            length (torch.Tensor): number of valid frames in input
                sequence, (B,).
            beam_width (int): beam size to use during search.

        Returns:
            List[Hypothesis]: top-``beam_width`` hypotheses found by beam search.
        """
        if input.dim() != 3:
            raise ValueError("input must be of shape (B, T, D)")

        if length.dim() != 1:
            raise ValueError("length must be of shape (B,)")

        enc_out_batched, _ = self.model.transcribe(input, length)

        search_outputs = []
        for enc_out in enc_out_batched:
            search_outputs.append(self._search(enc_out.unsqueeze(0), None, beam_width))

        return search_outputs

    @torch.jit.export
    def infer(
        self,
        input: torch.Tensor,
        length: torch.Tensor,
        beam_width: int,
        state: Optional[List[List[torch.Tensor]]] = None,
        hypothesis: Optional[Hypothesis] = None,
    ) -> Tuple[List[Hypothesis], List[List[torch.Tensor]]]:
        r"""Performs beam search for the given input sequence in streaming mode.

        T: number of frames;
        D: feature dimension of each frame.

        Args:
            input (torch.Tensor): sequence of input frames, with shape (T, D) or (1, T, D).
            length (torch.Tensor): number of valid frames in input
                sequence, with shape () or (1,).
            beam_width (int): beam size to use during search.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing transcription network internal state generated in preceding
                invocation. (Default: ``None``)
            hypothesis (Hypothesis or None): hypothesis from preceding invocation to seed
                search with. (Default: ``None``)

        Returns:
            (List[Hypothesis], List[List[torch.Tensor]]):
                List[Hypothesis]
                    top-``beam_width`` hypotheses found by beam search.
                List[List[torch.Tensor]]
                    list of lists of tensors representing transcription network
                    internal state generated in current invocation.
        """
        if input.dim() != 2 and not (input.dim() == 3 and input.shape[0] == 1):
            raise ValueError("input must be of shape (T, D) or (1, T, D)")
        if input.dim() == 2:
            input = input.unsqueeze(0)

        if length.shape != () and length.shape != (1,):
            raise ValueError("length must be of shape () or (1,)")
        if length.dim() == 0:
            length = length.unsqueeze(0)

        enc_out, _, state = self.model.transcribe_streaming(input, length, state)
        return self._search(enc_out, hypothesis, beam_width), state
