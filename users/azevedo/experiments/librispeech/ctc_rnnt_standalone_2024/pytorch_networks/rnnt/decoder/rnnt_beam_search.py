import torch
from typing import Optional, Callable, List, Tuple
from torchaudio.models.rnnt_decoder import RNNTBeamSearch, RNNT, Hypothesis, _get_hypo_predictor_out


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
 
    """
    # TODO: maybe use this function to figure out the actual alignment path and analyze right chunk-boundary
    #       non-blank emmision confidence
    def _gen_next_token_probs(
        self, enc_out: torch.Tensor, hypos: List[Hypothesis], device: torch.device
        ) -> torch.Tensor:
        
        res =  super()._gen_next_token_probs(enc_out, hypos, device)
        #print("_gen_next_token_probs:", res[:, -1])
        #print(f"{res.shape = }")
        return res

    def _gen_a_hypos(
        self,
        a_hypos: List[Hypothesis],
        b_hypos: List[Hypothesis],
        next_token_probs: torch.Tensor,
        t: int,
        beam_width: int,
        device: torch.device,
        ) -> List[Hypothesis]:

        res = super()._gen_a_hypos(a_hypos, b_hypos, next_token_probs, t, beam_width, device)
        #print(f"> _gen_a_hypos at {t=}:", [r[0] for r in res] if res else res)
        #print(f"> _gen_a_hypos b_hypos:", [b[0] for b in b_hypos] if b_hypos else b_hypos)
        #best_probs, best_tokens = next_token_probs.reshape(-1).topk(beam_width)
        #best_tokens_idc = best_tokens % next_token_probs.shape[1]

        #print("> _gen_a_hypos:", f"{best_tokens_idc = }\n{best_probs = }\n")

        return res
    """