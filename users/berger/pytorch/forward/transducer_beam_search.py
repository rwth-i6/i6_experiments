from functools import lru_cache
from typing import List
from dataclasses import dataclass

import torch
from torchaudio.models.rnnt import RNNT

from i6_experiments.users.berger.pytorch.models.conformer_transducer import FFNNTransducer


@dataclass
class Hypothesis:
    tokens: List[int]
    score: float
    timestep: int

    @property
    def avg_score(self) -> float:
        return self.score / len(self.tokens)


def extended_hypothesis(base_hyp: Hypothesis, token: int, is_blank: bool, score: float) -> Hypothesis:
    return Hypothesis(
        base_hyp.tokens + [token],
        base_hyp.score + score,
        base_hyp.timestep + int(is_blank),
    )


def beam_search(*, model: RNNT, features: torch.Tensor, features_len: torch.Tensor, beam_size: int = 100) -> List[int]:
    # Some dimension checks
    assert features.dim() == 2 or (features.dim() == 3 and features.size(0) == 1)  # [T, F] or [1, T, F]
    if features.dim() == 2:
        features = features.unsqueeze(0)  # [1, T, F]

    assert features_len.dim() == 0 or (features_len.dim() == 1 and features_len.size(0) == 1)  # [] or [1]
    if features_len.dim() == 0:
        features_len = features_len.unsqueeze(0)  # [1]

    # Compute encoder once
    enc, enc_lens = model.forward_encoder(features, features_len)  # [1, T, C], [1]
    T = enc_lens[0].cpu().item()

    # Function to get scores for time and history to avoid duplicate computation
    @lru_cache
    def cached_forward(timestep: int, context_tensor: torch.Tensor) -> torch.Tensor:
        enc_state = enc[:, timestep]  # [1, C]
        log_probs = model.forward_single(enc_state, context_tensor.unsqueeze(0))[0]  # [C]
        scores = -log_probs  # [C]
        return scores.cpu()

    # Initial hypothesis contains all-blank history with 0 score
    hypotheses = [Hypothesis([model.blank_idx] * model.context_history_size, 0, 0)]

    # all_finished indicated if all hypotheses have reached the end of the time axis
    all_finished = False
    while not all_finished:
        next_hypotheses = []  # accumulate extended hypotheses in here
        all_finished = True

        # iterate over all existing hypotheses and extend them
        for hypothesis in hypotheses:
            if hypothesis.timestep == T:
                # hypothesis has already reached end of time axis and is not extended
                next_hypotheses.append(hypothesis)
                continue

            # hypothesis gets extended, so not all are finished
            all_finished = False

            # form limited context of hypothesis into a tensor and compute log probs with it
            context_tensor = torch.tensor(hypothesis.tokens[-model.context_history_size :], device=enc.device)  # [H]
            scores = cached_forward(hypothesis.timestep, context_tensor)  # [C]

            # extend hypothesis with all possible next classes
            for c in range(scores.size(0)):
                next_hypotheses.append(extended_hypothesis(hypothesis, c, c == model.blank_idx, scores[c].item()))

        # Pruning
        hypotheses = sorted(next_hypotheses, key=lambda hyp: hyp.avg_score)[:beam_size]

    best_hypothesis = hypotheses[0]  # list has been sorted at the end of the loop so the best is first in the list
    assert best_hypothesis.timestep == T
    non_blanks = list(filter(lambda c: c != model.blank_idx, best_hypothesis.tokens))

    return non_blanks
