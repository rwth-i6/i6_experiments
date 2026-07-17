__all__ = ["ClusteringDecodeCallback"]
import os
import time

import numpy as np
import torch
from scipy.spatial.distance import cdist

from .util import PoolingRegistry
from .model import GaussianModelNumpy
from .parallel_recognizer import ParallelSegmentRecognizer

from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import TensorDict


class ClusteringDecodeCallback(ForwardCallbackIface):
    def __init__(
        self,
        centroids_file: str | None,
        recognition_config: str,
        gaussian_model: GaussianModelNumpy | None = None,
        distance_scale: float = 1.0,
        subsampling: int | None = None,
        pooling_function: str = "maxpool_time_np",
        verbosity: int = 1,
        exclude_lemmata=["[SILENCE]"],
        num_workers: int | None = 7,
        task_timeout: float | None = 1800.0,
    ):
        self.centroids_file = centroids_file
        self.recognition_config = recognition_config

        self.gaussian_model = gaussian_model

        self.subsampling = subsampling
        self.distance_scale = distance_scale

        self.config = None

        self.pool = PoolingRegistry.select(
            pooling_function,
            stride=subsampling,
            kernel_size=2 * subsampling if subsampling else None,
        )

        self.verbosity = verbosity

        self.exclude_lemmata = exclude_lemmata

        self.hyp_buffer = []

        self.recognizer = ParallelSegmentRecognizer(
            recognition_config, num_workers=num_workers, task_timeout=task_timeout
        )

        # --- debug timing (see finish() for the breakdown this feeds) ---
        self._t_init: float | None = None

    def init(self, *, model: torch.nn.Module | None = None):
        self._t_init = time.time()

        if not self.gaussian_model:
            self.centroids = self.load_centroids()

        self.recognizer.start()

    def load_centroids(self) -> bool:
        """
        Tries to load previously saved centroids.

        Returns whether any previous centroids were found and adjusts the epoch accordingly.

        :return: whether any previous centroids were found
        """
        assert self.centroids_file is not None
        assert os.path.exists(self.centroids_file)

        # do the actual loading
        return np.load(self.centroids_file)

    def compute_squared_distances(
        self,
        *,
        seq_tag: str,
        hidden_states: np.ndarray,
    ) -> np.ndarray:
        centroids = self.centroids
        # assert centroids, "Centroids must be initialized before computing distances."
        assert centroids is not None
        assert isinstance(hidden_states, np.ndarray)
        dist = cdist(hidden_states, centroids, metric="sqeuclidean")
        return dist

    def process_seq(self, *, seq_tag: str, outputs: TensorDict, last_seq: bool = False):
        hidden_state_tensor = outputs["output"].raw_tensor
        assert isinstance(hidden_state_tensor, np.ndarray)

        if self.subsampling and self.subsampling > 1:
            hidden_state_tensor = self.pool(
                hidden_state_tensor,
            )

        if self.gaussian_model is None:
            distances = self.compute_squared_distances(
                seq_tag=seq_tag,
                hidden_states=hidden_state_tensor,
            )
        else:
            distances = self.gaussian_model.forward(hidden_state_tensor)
        scaled_distances = distances * self.distance_scale

        if self.verbosity >= 2:
            print(f"Submitting sequence {seq_tag}.")

        self.recognizer.submit(seq_tag, scaled_distances)

    def finish(self):
        """Collect all pending recognitions and write the hypothesis file."""
        results = self.recognizer.drain()

        for seq_tag, items in results:
            if self.verbosity >= 2:
                print(f"Finished sequence {seq_tag}.")
            hyp = " ".join(filter(lambda lem: lem not in self.exclude_lemmata, (item.lemma for item in items)))
            self.hyp_buffer.append(hyp)

        self.recognizer.shutdown()

        print(f"[TIMING] total (init->finish): {time.time() - self._t_init:.3f}s", flush=True)

        # print(self.hyp_buffer)
        with open("hyp.txt", "w+") as fp:
            fp.write("\n".join(self.hyp_buffer))
