"""
Learned filter
"""

from typing import Tuple, Dict
import torch
from torch import nn
from .filter_base import LearnedDataFilterBase


class LearnedDataFilterViaLoss(LearnedDataFilterBase):
    """
    learned filter

    Usage:

    - Put somewhere inside your model, maybe early.
    - Call this module (:func:`forward`). It will filter the seqs based on estimated scores.
    - Calculate loss of your model (whatever it is, CE or so).
    - Call :func:`score_estimator_loss`, pass it the model loss, get a loss for the score estimation,
      to be able to train that part.
    """

    def score_estimator_loss(self, model_loss: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the score estimation.

        :param model_loss: e.g. [B'] or [B',T_] (T_ = max(seq_lens) for seq_lens after filter)
        :return: loss [B']
        """
        assert self._estimated_scores is not None, "forward not called?"
        est_scores, est_scores_seq_lens = self._get_filtered_estimated_scores()  # [B',T']
        est_scores_time_size = est_scores.shape[1]
        model_loss = model_loss.detach()  # no gradient flow to the model loss
        if model_loss.ndim == 1:
            est_scores = est_scores.mean(dim=1)  # [B']
            est_scores = est_scores * (est_scores_time_size / est_scores_seq_lens)
            assert model_loss.shape == est_scores.shape
        elif model_loss.ndim == 2:
            model_loss = model_loss[:, None]  # [B',1,T]
            model_loss = torch.nn.functional.avg_pool1d(
                model_loss, self.score_estimator.total_stride, padding=(self.score_estimator.total_stride - 1) // 2
            )  # [B',1,T']
            model_loss = model_loss.squeeze(dim=1)  # [B',T'], like est_scores
            assert model_loss.shape == est_scores.shape
        else:
            raise Exception(f"unexpected model loss shape {model_loss.shape} ndim")
        real_loss = -model_loss  # e.g. positive log prob
        loss = torch.square(real_loss - est_scores)  # [B'] or [B',T']
        if loss.ndim == 2:
            loss = loss.mean(dim=1)  # [B']
            loss = loss * (est_scores_time_size / est_scores_seq_lens)
        assert loss.shape == est_scores.shape[:1]  # [B']
        return loss


def demo():
    """demo"""

    class Model(nn.Module):
        def __init__(self, in_features: int, out_features: int, *, hidden_size: int = 10):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.layer1 = nn.Linear(in_features, hidden_size)
            self.data_filter = LearnedDataFilterViaLoss(hidden_size, hidden_size // 2)
            self.layer2 = nn.Linear(hidden_size, hidden_size)
            self.layer3 = nn.Linear(hidden_size, out_features)

        # noinspection PyShadowingNames
        def forward(
            self, x: torch.Tensor, seq_lens: torch.Tensor, *, targets: torch.Tensor
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            """
            :param x: [B,T,D]
            :param seq_lens: [B]
            :param targets: [B,T]
            :return: y [B',T,D], loss [B']
            """
            x = self.layer1(x).relu()  # [B,T,D]
            x, seq_lens = self.data_filter(x, seq_lens=seq_lens, btd_axes=(0, 1, 2))  # [B',T_,D]
            targets = self.data_filter.filter_batch(targets, time_axis=1)  # [B',T_]
            x = self.layer2(x).relu()  # [B',T,D]
            x = self.layer3(x)  # [B',T_,D]
            model_loss = torch.nn.functional.cross_entropy(x.permute(0, 2, 1), targets, reduction="none")  # [B',T_]
            return x, {
                "model": model_loss.mean(),
                "data_filter": self.data_filter.score_estimator_loss(model_loss).mean(),
            }

    model = Model(5, 5)
    opt = torch.optim.Adam(model.parameters())
    while True:
        opt.zero_grad()
        b, t = 5, 7
        x = torch.randn((b, t, model.in_features))
        targets = torch.randint(0, model.out_features, (b, t))
        seq_lens = torch.tensor([t, t - 1, t - 2, t - 3, t - 4])
        assert seq_lens.shape == x.shape[:1]
        _, loss = model(x, seq_lens=seq_lens, targets=targets)
        print("loss:", loss)
        total_loss: torch.Tensor = sum(v for v in loss.values())
        total_loss.backward()
        opt.step()


if __name__ == "__main__":
    demo()
