import torch
from typing import Optional, Callable, List, Dict, Tuple
from torchaudio.models.rnnt_decoder import (
    RNNTBeamSearch, RNNT, Hypothesis, 
    _get_hypo_predictor_out, _get_hypo_tokens, _get_hypo_key, _get_hypo_score, _get_hypo_state, _remove_hypo,
    _batch_state, _slice_state
)


class ModifiedRNNTBeamSearch(RNNTBeamSearch):
    r"""Beam search decoder for RNN-T model.

    Modified with blank penalty and semi-batched search where encoder is run in batch mode

    See Also:
        * :class:`torchaudio.pipelines.RNNTBundle`: ASR pipeline with pretrained model.

    Args:
        model (RNNT): RNN-T model to use.
        blank (int): index of blank token in vocabulary.
        temperature (float, optional): temperature to apply to joint network output.
            Larger values yield more uniform samples. (Default: 1.0)
        hypo_sort_key (Callable[[Hypothesis], float] or None, optional): callable that computes a score
            for a given hypothesis to rank hypotheses by. If ``None``, defaults to callable that returns
            hypothesis score normalized by token sequence length. (Default: None)
        step_max_tokens (int, optional): maximum number of tokens to emit per input time step. (Default: 100)
        blank_penalty: blank penalty in log space
    """

    def __init__(
        self,
        model: RNNT,
        blank: int,
        temperature: float = 1.0,
        hypo_sort_key: Optional[Callable[[Hypothesis], float]] = None,
        step_max_tokens: int = 100,
        blank_penalty: Optional[float] = None,
        eos_penalty: Optional[float] = None,
    ) -> None:
        super().__init__(
            model=model,
            blank=blank,
            temperature=temperature,
            hypo_sort_key=hypo_sort_key,
            step_max_tokens=step_max_tokens,
        )
        self.blank_penalty = blank_penalty
        self.eos_penalty = eos_penalty

    def _gen_next_token_probs(
        self, enc_out: torch.Tensor, hypos: List[Hypothesis], device: torch.device
    ) -> torch.Tensor:
        one_tensor = torch.tensor([1], device=device)
        predictor_out = torch.stack([_get_hypo_predictor_out(h) for h in hypos], dim=0)
        joined_out, _, _ = self.model.join(
            enc_out,
            one_tensor,
            predictor_out,
            torch.tensor([1] * len(hypos), device=device),
        )  # [beam_width, 1, 1, num_tokens]
        joined_out = torch.nn.functional.log_softmax(joined_out / self.temperature, dim=3)

        if self.blank_penalty is not None:
            # assumes blank is last
            joined_out[:, :, :, self.blank] -= self.blank_penalty
        
        if self.eos_penalty is not None:
            # assumes eos is first
            joined_out[:, :, :, 0] -= self.eos_penalty
            self.eos_penalty *= 0.997

        return joined_out[:, 0, 0]

    def forward_semi_batched(
        self, input: torch.Tensor, length: torch.Tensor, beam_width: int
        ) -> List[List[Hypothesis]]:
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
 


HypothesisV2 = Tuple[List[int], torch.Tensor, List[List[torch.Tensor]], float, List[int]]

def _get_hypo_alignment(hypo: HypothesisV2) -> List[int]:
    return hypo[4]

class ModifiedRNNTBeamSearchV2(ModifiedRNNTBeamSearch):
    r"""Beam search decoder for RNN-T model.

    Modified with blank penalty and semi-batched search where encoder is run in batch mode

    See Also:
        * :class:`torchaudio.pipelines.RNNTBundle`: ASR pipeline with pretrained model.

    Args:
        model (RNNT): RNN-T model to use.
        blank (int): index of blank token in vocabulary.
        temperature (float, optional): temperature to apply to joint network output.
            Larger values yield more uniform samples. (Default: 1.0)
        hypo_sort_key (Callable[[Hypothesis], float] or None, optional): callable that computes a score
            for a given hypothesis to rank hypotheses by. If ``None``, defaults to callable that returns
            hypothesis score normalized by token sequence length. (Default: None)
        step_max_tokens (int, optional): maximum number of tokens to emit per input time step. (Default: 100)
        blank_penalty: blank penalty in log space
    """

    def __init__(
        self,
        model: RNNT,
        blank: int,
        temperature: float = 1.0,
        hypo_sort_key: Optional[Callable[[Hypothesis], float]] = None,
        step_max_tokens: int = 100,
        blank_penalty: Optional[float] = None,
        eos_penalty: Optional[float] = None,
    ) -> None:
        super().__init__(
            model=model,
            blank=blank,
            temperature=temperature,
            hypo_sort_key=hypo_sort_key,
            step_max_tokens=step_max_tokens,
            blank_penalty=blank_penalty,
            eos_penalty=eos_penalty
        )

    def _init_b_hypos(self, device: torch.device) -> List[HypothesisV2]:
        token = self.blank
        state = None

        one_tensor = torch.tensor([1], device=device)
        pred_out, _, pred_state = self.model.predict(torch.tensor([[token]], device=device), one_tensor, state)
        init_hypo = (
            [token],
            pred_out[0].detach(),
            pred_state,
            0.0,
            [token]
        )
        return [init_hypo]

    def _gen_b_hypos(
        self,
        b_hypos: List[HypothesisV2],
        a_hypos: List[HypothesisV2],
        next_token_probs: torch.Tensor,
        key_to_b_hypo: Dict[str, HypothesisV2],
    ) -> List[HypothesisV2]:
        for i in range(len(a_hypos)):
            h_a = a_hypos[i]
            append_blank_score = _get_hypo_score(h_a) + next_token_probs[i, -1]
            if _get_hypo_key(h_a) in key_to_b_hypo:
                h_b = key_to_b_hypo[_get_hypo_key(h_a)]
                _remove_hypo(h_b, b_hypos)
                score = float(torch.tensor(_get_hypo_score(h_b)).logaddexp(append_blank_score))
            else:
                score = float(append_blank_score)
            h_b = (
                _get_hypo_tokens(h_a),
                _get_hypo_predictor_out(h_a),
                _get_hypo_state(h_a),
                score,
                _get_hypo_alignment(h_a),
            )
            b_hypos.append(h_b)
            key_to_b_hypo[_get_hypo_key(h_b)] = h_b
        _, sorted_idx = torch.tensor([_get_hypo_score(hypo) for hypo in b_hypos]).sort()
        return [b_hypos[idx] for idx in sorted_idx]
    
    def _gen_new_hypos(
        self,
        base_hypos: List[HypothesisV2],
        tokens: List[int],
        scores: List[float],
        t: int,
        device: torch.device,
    ) -> List[Hypothesis]:
        tgt_tokens = torch.tensor([[token] for token in tokens], device=device)
        states = _batch_state(base_hypos)
        pred_out, _, pred_states = self.model.predict(
            tgt_tokens,
            torch.tensor([1] * len(base_hypos), device=device),
            states,
        )
        new_hypos: List[Hypothesis] = []
        for i, h_a in enumerate(base_hypos):
            new_tokens = _get_hypo_tokens(h_a) + [tokens[i]]
            new_hypos.append((
                new_tokens, 
                pred_out[i].detach(), 
                _slice_state(pred_states, i, device), 
                scores[i],
                _get_hypo_alignment(h_a) + [tokens[i]]
            ))
        return new_hypos

    def _search(
        self,
        enc_out: torch.Tensor,
        hypo: Optional[List[HypothesisV2]],
        beam_width: int,
    ) -> List[HypothesisV2]:
        print(f"> {enc_out.shape = }")

        n_time_steps = enc_out.shape[1]
        device = enc_out.device

        a_hypos: List[HypothesisV2] = []
        b_hypos = self._init_b_hypos(device) if hypo is None else hypo
        for t in range(n_time_steps):
            a_hypos = b_hypos
            b_hypos = torch.jit.annotate(List[HypothesisV2], [])
            key_to_b_hypo: Dict[str, HypothesisV2] = {}
            symbols_current_t = 0

            while a_hypos:
                next_token_probs = self._gen_next_token_probs(enc_out[:, t : t + 1], a_hypos, device)
                next_token_probs = next_token_probs.cpu()
                b_hypos = self._gen_b_hypos(b_hypos, a_hypos, next_token_probs, key_to_b_hypo)

                if symbols_current_t == self.step_max_tokens:
                    break

                a_hypos = self._gen_a_hypos(
                    a_hypos,
                    b_hypos,
                    next_token_probs,
                    t,
                    beam_width,
                    device,
                )
                if a_hypos:
                    symbols_current_t += 1

            _, sorted_idx = torch.tensor([self.hypo_sort_key(hyp) for hyp in b_hypos]).topk(beam_width)

            new_hypos = []
            for idx in sorted_idx:
                # try:
                #     al = _get_hypo_alignment(b_hypos[idx])
                # except IndexError:
                #     al = _get_hypo_tokens(b_hypos[idx])
                new_hypos.append((
                    _get_hypo_tokens(b_hypos[idx]),
                    _get_hypo_predictor_out(b_hypos[idx]),
                    _get_hypo_state(b_hypos[idx]),
                    _get_hypo_score(b_hypos[idx]),
                    _get_hypo_alignment(b_hypos[idx]) + [self.blank],
                ))
            b_hypos = new_hypos

        return b_hypos
 