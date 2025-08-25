from dataclasses import dataclass
import torch

from i6_models.config import ModelConfiguration


@dataclass
class SelfAttRelPosEncodingV1Config(ModelConfiguration):
    out_dim: int
    clipping: int
    dropout: float


class SelfAttRelPosEncodingV1(torch.nn.Module):
    def __init__(self, cfg: SelfAttRelPosEncodingV1Config) -> None:
        super().__init__()

        self.dropout = torch.nn.Dropout(p=cfg.dropout)
        self.out_dim = cfg.out_dim
        self.encoding_matrix = torch.nn.Parameter(torch.empty(size=(2 * cfg.clipping + 1, cfg.out_dim)))
        torch.nn.init.xavier_uniform_(self.encoding_matrix)

        self.clipping = cfg.clipping

        # self.pos_enc = None
        # self.forward(torch.zeros(size=(1, cfg.clipping + 1), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]

        # if self.pos_enc is not None and self.pos_enc.size(0) >= x.size(1):
        #     if self.pos_enc.device != x.device:
        #         self.pos_enc.to(device=x.device)
        #     # Encoding matrix has already been computed and is large enough
        #     return self.pos_enc[: x.size(1), : x.size(1)]  # [T, T, out_dim]

        # Example: T = 4, clipping = 2
        # Position range [0, 1, 2, 3]
        # [0, 1, 2]
        position = torch.arange(start=0, end=x.size(1), dtype=torch.int64)  # [T]

        # Position difference matrix
        # [[0, 1, 2, 3], [-1, 0, 1, 2], [-2, -1, 0, 1], [-3, -2, -1, 0]]
        # [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
        distance_mat = position.unsqueeze(0) - position.unsqueeze(1)  # [T, T]

        # [[0, 1, 2, 2], [-1, 0, 1, 2], [-2, -1, 0, 1], [-2, -2, -1, 0]]
        # [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
        distance_mat_clipped = torch.clip(distance_mat, min=-self.clipping, max=self.clipping)  # [T, T]

        # All values shifted to be >= 0
        # [[2, 3, 4, 4], [1, 2, 3, 4], [0, 1, 2, 3], [0, 0, 1, 2]]
        # [[2, 3, 4], [1, 2, 3], [0, 1, 2]]
        pos_distance_mat_clipped = distance_mat_clipped + self.clipping  # [T, T]

        # Index such that self.pos_enc[a, b, :] = self.encoding_matrix[pos_distance_mat_clipped[a, b], :]
        # self.pos_enc = self.encoding_matrix[pos_distance_mat_clipped].to(device=x.device)  # [T, T, out_dim]
        return self.encoding_matrix[pos_distance_mat_clipped].to(device=x.device)  # [T, T, out_dim]
