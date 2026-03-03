import os
import torch
import sisyphus.toolkit as tk
from sisyphus import Job, Task


class ComputePriorWithoutBlank(Job):
    """
    Loads a tensor stored in a Sisyphus path, removes a given index along a
    specified dimension, renormalizes the remaining weights so they sum to 1
    along that dimension, and saves the result.

    Args:
        tensor_file:   tk.Path to the .pt file (saved with torch.save).
        remove_index:  The index to remove along `dim`.
        dim:           Dimension along which to remove and renormalize
                       (default: -1, i.e. the vocab/label dimension).
    """

    def __init__(self, tensor_file: tk.Path, blank_idx: int, dim: int = -1):
        self.tensor_file = tensor_file
        self.blank_idx = blank_idx
        self.dim = dim

        self.out_tensor = self.output_path("prior_without_blank.pt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # ── load ──────────────────────────────────────────────────────────────
        with open(self.tensor_file.get_path(), "r") as f:
            values = [float(line.strip()) for line in f if line.strip()]

        tensor = torch.tensor(values, dtype=torch.float64)
        print(f"Loaded tensor shape: {tensor.shape}, dtype: {tensor.dtype}")

        dim = self.dim
        n = tensor.shape[dim]

        if not (-n <= self.blank_idx < n):
            raise ValueError(
                f"remove_index={self.blank_idx} is out of range for "
                f"dimension {dim} of size {n}."
            )

        # ── remove index ──────────────────────────────────────────────────────
        keep_indices = [i for i in range(n) if i != (self.blank_idx % n)]
        keep_tensor = torch.index_select(
            tensor,
            dim=dim,
            index=torch.tensor(keep_indices, dtype=torch.long),
        )
        print(f"Removed index {self.blank_idx} → new shape: {keep_tensor.shape}")

        # ── renormalize ───────────────────────────────────────────────────────
        # Assumes tensor holds probabilities (sum=1 along `dim`).
        # For log-probs: exponentiate first, renormalize, then log again.
        weight_sum = keep_tensor.sum(dim=dim, keepdim=True).clamp(min=1e-12)
        renormalized = keep_tensor / weight_sum

        print(
            f"Normalized along dim {dim}. "
            f"Sum check (≈1): {renormalized.sum(dim=dim).mean().item():.6f}"
        )

        # ── save ──────────────────────────────────────────────────────────────
        out_path = self.out_tensor.get_path()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(renormalized, out_path)
        print(f"Saved to: {out_path}")