__all__ = ["ClusteringDecodeCallback"]
import os

import numpy as np
import torch
from scipy.spatial.distance import cdist

from .util import PoolingRegistry

from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import TensorDict


class ClusteringDecodeCallback(ForwardCallbackIface):
    def __init__(
        self,
        centroids_file: str,
        recognition_config: str,
        distance_scale: float = 1.0,
        subsampling: int | None = None,
        pooling_function: str = "maxpool_time_np",
        verbosity: int = 1,
        exclude_lemmata=["[SILENCE]"]
    ):
        self.centroids_file = centroids_file
        self.recognition_config = recognition_config

        self.subsampling = subsampling
        self.distance_scale = distance_scale

        self.config = None
        self.search_algo = None

        self.pool = PoolingRegistry.select(
            pooling_function,
            stride=subsampling,
            kernel_size=2 * subsampling if subsampling else None,
        )

        self.verbosity = verbosity

        self.exclude_lemmata = exclude_lemmata 

        self.hyp_buffer = []
    
    def init(self, *, model: torch.nn.Module | None = None):
        self.search_algo = self.init_search_algorithm()
        self.centroids = self.load_centroids()

    def init_search_algorithm(self):
        from librasr import Configuration, SearchAlgorithm

        config = Configuration()
        config.set_from_file(self.recognition_config)

        return SearchAlgorithm(config=config)
    
    def load_centroids(self) -> bool:
        """
        Tries to load previously saved centroids.
        
        Returns whether any previous centroids were found and adjusts the epoch accordingly.
        
        :return: whether any previous centroids were found
        """
        assert os.path.exists(self.centroids_file)

        # do the actual loading
        return np.load(self.centroids_file)
    
    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["search_algo"] = None

        return d

    def __setstate__(self, d) -> None:
        self.__dict__ = d
        self.search_algo = self.init_search_algorithm()
    
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
        assert self.search_algo is not None

       # if self.verbosity >= 2:
       #     print(f"Processing sequence {seq_tag}.")

        if self.subsampling:
            hidden_state_tensor = self.pool(
                hidden_state_tensor,
            )

        distances = self.compute_squared_distances(
            seq_tag=seq_tag,
            hidden_states=hidden_state_tensor,
        )
        scaled_distances = distances * self.distance_scale

        traceback = self.search_algo.recognize_segment(scaled_distances)

       # print(traceback)

        hyp = " ".join(filter(
            lambda lem: lem not in self.exclude_lemmata,
            (item.lemma for item in traceback)
        ))

        self.hyp_buffer.append((seq_tag, hyp))

    def finish(self):
        """Close file buffer."""
       # print(self.hyp_buffer)
        with open("hyp.txt", "w+") as fp:
            fp.write("\n".join(f"{tag}\t{hyp}" for tag, hyp in self.hyp_buffer))
