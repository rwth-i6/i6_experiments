from typing import Any, Callable, Dict, Optional
from sisyphus import Job, Task, tk
import logging
from i6_core.util import uopen
import json
from returnn.datasets.basic import Vocabulary
import math
import numpy as np


def compute_ece(confidences, accuracies, n_bins=10, strategy="equal_width"):
    """
    Computes Expected Calibration Error (ECE).

    Parameters:
        confidences (np.ndarray): Confidence scores for predictions.
        accuracies (np.ndarray): Binary accuracy values (1 for correct, 0 for incorrect).
        n_bins (int): Number of bins to divide [0,1] confidence range into.

    Returns:
        float: Expected Calibration Error (ECE)
    """
    real_ece = 0.0
    ece = 0.0
    signed_ece = 0.0
    N = len(confidences)

    if strategy == "equal_width":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "equal_size":
        # Compute quantile-based edges
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(confidences, quantiles)
    else:
        raise ValueError("strategy must be 'equal_width' or 'equal_size'")

    bins = []

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        # Get indices of confidences in the bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        cur_bin = {
            "index": i,
            "lower": float(bin_lower),
            "upper": float(bin_upper),
            "size": int(bin_size),
        }

        print(f"Bin {i}: [{bin_lower:.2f}, {bin_upper:.2f}] - Size: {bin_size}")

        if bin_size > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            print(f"  Avg Confidence: {avg_confidence:.2f}, Avg Accuracy: {avg_accuracy:.2f}")
            ece += (bin_size / N) * np.abs(avg_confidence - avg_accuracy)
            signed_ece += (bin_size / N) * (avg_confidence - avg_accuracy)
            cur_bin["avg_confidence"] = float(avg_confidence)
            cur_bin["avg_accuracy"] = float(avg_accuracy)
        bins.append(cur_bin)
    real_ece = np.mean(confidences - accuracies)
    print(f"Total ECE: {ece:.4f}")
    print(f"Total Real ECE: {real_ece:.4f}")
    print(f"Total Signed ECE: {signed_ece:.4f}")

    return bins, float(ece), float(signed_ece), float(real_ece)


class CalculateECE(Job):
    """
    Takes a dataset and computes the Expected Calibration Error (ECE).
    """

    def __init__(
        self,
        *,
        returnn_dataset: Dict[str, Any],
        returnn_root: Optional[tk.Path] = None,
        confidence_key: str,
        accuracy_key: str,
        confidence_logspace: bool = True,
        num_bins: int = 10,
    ):
        self.returnn_dataset = returnn_dataset
        self.returnn_root = returnn_root
        self.confidence_key = confidence_key
        self.accuracy_key = accuracy_key
        self.entropy_key = "entropy"
        self.confidence_logspace = confidence_logspace
        self.num_bins = num_bins

        self.out_stats = self.output_path("ece_stats.json")
        self.out_stats_equalsize = self.output_path("ece_stats_equalsize.json")
        self.out_ece = self.output_var("ece")
        self.out_signed_ece = self.output_var("signed_ece")
        self.out_mean_entropy = self.output_var("mean_entropy")
        self.out_real_ece = self.output_var("real_ece")

    def tasks(self):
        yield Task("run", mini_task=True)

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 10
        return super().hash(d)

    def run(self):
        """run"""
        import sys

        if self.returnn_root is not None:
            sys.path.insert(0, self.returnn_root.get_path())
        from returnn.config import set_global_config, Config
        from returnn.log import log

        config = Config()
        set_global_config(config)

        if not config.has("log_verbosity"):
            config.typed_dict["log_verbosity"] = 5
        log.init_by_config(config)
        from returnn.datasets import init_dataset
        import tree

        dataset_dict = self.returnn_dataset
        dataset_dict = tree.map_structure(lambda x: x.get_path() if isinstance(x, tk.Path) else x, dataset_dict)
        logging.debug("RETURNN dataset dict:", dataset_dict)
        assert isinstance(dataset_dict, dict)
        dataset = init_dataset(dataset_dict)
        dataset.init_seq_order(epoch=1)

        assert self.confidence_key in dataset.get_data_keys()
        assert self.accuracy_key in dataset.get_data_keys()
        assert self.entropy_key in dataset.get_data_keys()

        confidences = []
        accuracies = []
        entropies = []

        seq_idx = 0
        while dataset.is_less_than_num_seqs(seq_idx):
            dataset.load_seqs(seq_idx, seq_idx + 1)
            if seq_idx % 10000 == 0:
                logging.info(f"seq_idx {seq_idx}")
            # key = dataset.get_tag(seq_idx)
            confidence = dataset.get_data(seq_idx, self.confidence_key)
            assert isinstance(confidence, np.ndarray), "Confidence should be a numpy array."
            if self.confidence_logspace:
                confidence = np.exp(confidence)

            assert np.all(confidence >= 0.0) and np.all(confidence <= 1.0), (
                "Confidence values should be in [0, 1] range."
            )
            accuracy = dataset.get_data(seq_idx, self.accuracy_key)
            assert isinstance(accuracy, np.ndarray), "Accuracy should be a numpy array."
            assert np.all(np.isin(accuracy, [0, 1])), "Accuracy should be binary (0 or 1)."

            entropy = dataset.get_data(seq_idx, self.entropy_key)
            assert isinstance(entropy, np.ndarray), "Entropy should be a numpy array."
            entropies.append(entropy)

            confidences.append(confidence)
            accuracies.append(accuracy)

            seq_idx += 1

        bins, ece, signed_ece, real_ece = compute_ece(
            confidences=np.concatenate(confidences),
            accuracies=np.concatenate(accuracies),
            n_bins=self.num_bins,
            strategy="equal_width",
        )

        entropy_avg = float(np.concatenate(entropies).mean())
        logging.info(f"Mean entropy: {entropy_avg:.4f}")

        with uopen(self.out_stats, "wt") as out:
            json.dump({"ece": ece, "signed_ece": signed_ece, "bins": bins}, out, indent=2, sort_keys=True)
        bins_equalsize, ece_equalsize, signed_ece_equalsize, realece_equalsize = compute_ece(
            confidences=np.concatenate(confidences),
            accuracies=np.concatenate(accuracies),
            n_bins=self.num_bins,
            strategy="equal_size",
        )
        with uopen(self.out_stats_equalsize, "wt") as out:
            json.dump(
                {"ece": ece_equalsize, "signed_ece": signed_ece_equalsize, "bins": bins_equalsize},
                out,
                indent=2,
                sort_keys=True,
            )
        self.out_ece.set(ece)
        self.out_real_ece.set(real_ece)
        self.out_signed_ece.set(signed_ece)
        self.out_mean_entropy.set(entropy_avg)
