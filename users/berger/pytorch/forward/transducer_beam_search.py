from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
from torchaudio.models.rnnt import RNNT


@dataclass
class Hypothesis:
    tokens: List[int]
    pred_state: torch.Tensor
    pred_history_state: Optional[List[List[torch.Tensor]]]
    score: float
    timestep: int


def extended_hypothesis(
    base_hyp: Hypothesis,
    new_pred_state: torch.Tensor,
    new_pred_history_state: List[List[torch.Tensor]],
    token: int,
    score: float,
) -> Hypothesis:
    return Hypothesis(
        tokens=base_hyp.tokens + [token],
        pred_state=new_pred_state,
        pred_history_state=new_pred_history_state,
        score=base_hyp.score + score,
        timestep=base_hyp.timestep + 1,
    )


def monotonic_timesync_beam_search(
    *, model: RNNT, features: torch.Tensor, feature_lengths: torch.Tensor, blank_id: int, beam_size: int = 10
) -> Tuple[List[int], float]:
    # Some dimension checks
    if features.dim() == 2:  # [T, F]
        features = features.unsqueeze(0)  # [1, T, F]
    assert features.dim() == 3 and features.size(0) == 1  # [1, T, F]
    assert feature_lengths.dim() == 1 and feature_lengths.size(0) == 1  # [1]

    # Compute encoder once
    enc, enc_lengths = model.transcribe(features, feature_lengths)  # [1, T, E], [1]
    T = int(enc_lengths[0].cpu().item())

    def predict_next(
        token: int, history_state: Optional[List[List[torch.Tensor]]]
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        new_pred_state, _, new_pred_history_state = model.predict(  # [1, P]
            targets=torch.tensor([[token]], device=enc.device),
            target_lengths=torch.tensor([1], device=enc.device),
            state=history_state,
        )

        return new_pred_state, new_pred_history_state

    # Initial hypothesis contains all-blank history with 0 score
    initial_pred_state, initial_pred_history_state = predict_next(blank_id, None)  # [1, P]
    hypotheses = [
        Hypothesis(
            tokens=[],
            pred_state=initial_pred_state,
            pred_history_state=initial_pred_history_state,
            score=0.0,
            timestep=0,
        )
    ]

    # all_finished indicated if all hypotheses have reached the end of the time axis
    for t in range(T):
        enc_state = enc[:, t : t + 1, :]  # [1, 1, E]

        next_hypotheses = []  # accumulate extended hypotheses in here

        # iterate over all existing hypotheses and extend them
        for hypothesis in hypotheses:
            recent_token = hypothesis.tokens[-1] if len(hypothesis.tokens) > 0 else blank_id
            if recent_token != blank_id:
                new_pred_state, new_pred_history_state = predict_next(
                    recent_token, hypothesis.pred_history_state
                )  # [1, P]
            else:
                new_pred_state = hypothesis.pred_state  # [1, P]
                new_pred_history_state = hypothesis.pred_history_state

            assert new_pred_history_state is not None

            log_probs, _, _ = model.join(  # [1, C] (packed) or [1, 1, 1, C] (not packed))
                source_encodings=enc_state,
                source_lengths=torch.tensor([1], device=enc.device),
                target_encodings=new_pred_state,
                target_lengths=torch.tensor([1], device=enc.device),
            )
            log_probs = log_probs.squeeze()  # [C]

            # extend hypothesis with all possible next classes
            for c in range(log_probs.size(0)):
                next_hypotheses.append(
                    extended_hypothesis(
                        base_hyp=hypothesis,
                        new_pred_state=new_pred_state,
                        new_pred_history_state=new_pred_history_state,
                        token=c,
                        score=log_probs[c].item(),
                    )
                )

        # Pruning
        hypotheses = sorted(next_hypotheses, key=lambda hyp: hyp.score, reverse=True)[:beam_size]

    best_hypothesis = hypotheses[0]  # list has been sorted at the end of the loop so the best is first in the list
    non_blanks = list(filter(lambda c: c != blank_id, best_hypothesis.tokens))

    return non_blanks, best_hypothesis.score
