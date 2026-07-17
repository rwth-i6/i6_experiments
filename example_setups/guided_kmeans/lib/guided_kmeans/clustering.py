__all__ = [
    "BatchwiseUpdater",
    "KMeansPlusPlusInitializer",
    "BatchwiseKMeansUpdater",
    "NnOutputClusteringCallback",
]

import os
import glob
import re
import xml.etree.ElementTree as ET
from enum import Enum
from collections import UserDict, Counter
import random
import time

from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import TensorDict
from returnn.datasets.hdf import SimpleHDFWriter, NextGenHDFDataset

from abc import ABC, abstractmethod
import timeit
import numpy as np
from scipy.spatial.distance import cdist
import torch
import pickle
from typing import Optional, Any, Callable, List, Iterable

from i6_core.lib.lexicon import Lexicon
from i6_core.util import uopen

from .util import segments_to_array, ProgressLogger, TracebackLogger, PoolingRegistry
from .pca import StreamingPCA
from .running_update import RunningAverageUpdater, RelativeFrequencyUpdater
from .statistics import (
    get_default_logger
)

from .model import GaussianModelNumpy
from .parallel_recognizer import ParallelSegmentRecognizer, PlainTracebackItem

class EvalReservoir:
    def __init__(self, cap=50000, l2_normalize=True, seed=0, dtype=np.float32):
        self.cap = cap
        self.l2 = l2_normalize
        self.rng = np.random.default_rng(seed)
        self.buf = None
        self.filled = 0
        self.seen = 0
        self.dtype = dtype

    def _norm(self, X):
        if not self.l2: return X
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n, 1e-12)

    def add(self, X):
        if X.ndim == 1: X = X[None, :]
        X = self._norm(X.astype(self.dtype, copy=False))
        n, d = X.shape
        if self.buf is None:
            self.buf = np.empty((self.cap, d), dtype=self.dtype)

        # Fill phase (contiguous copy, no Python loops)
        if self.filled < self.cap:
            take = min(n, self.cap - self.filled)
            self.buf[self.filled:self.filled+take] = X[:take]
            self.filled += take
            self.seen  += take
            X = X[take:]
            n -= take
            if n == 0:
                return

        # Reservoir replacement (simple, in-place)
        # This loop runs only for overflow items (usually small vs total data)
        for i in range(n):
            self.seen += 1
            j = self.rng.integers(0, self.seen)
            if j < self.cap:
                self.buf[j] = X[i]

    def eval_sse(self, centroids):
        if self.filled == 0: return None
        # Use sqeuclidean via fast BLAS path
        C = centroids.astype(self.dtype, copy=False)
        X = self.buf[:self.filled]
        x2 = np.sum(X*X, axis=1, keepdims=True)
        c2 = np.sum(C*C, axis=1, keepdims=True).T
        # dist2 = ||x||^2 + ||c||^2 - 2 x.c
        dist2 = x2 + c2 - 2.0 * (X @ C.T)
        return float(np.mean(np.min(dist2, axis=1)))


class BatchwiseUpdater:
    # make this class pickable, so it can be used in the config
    __slots__ = ("batch_size", "data_index", "data_batch", "l2_normalize", "centroids")

    def __init__(self, batch_size: int = 1024, l2_normalize: bool = False):
        self.batch_size = batch_size
        self.l2_normalize = l2_normalize
        self.data_index = 0
        self.data_batch = []

        self.centroids = None

    # make this class picklable by defining __getstate__ and __setstate__
    def __getstate__(self):
        return {
            "batch_size": self.batch_size,
            "data_index": self.data_index,
            "data_batch": self.data_batch,
            "l2_normalize": self.l2_normalize,
        }
    
    def __setstate__(self, state):
        self.batch_size = state["batch_size"]
        self.data_index = state["data_index"]
        self.data_batch = state["data_batch"]
    
    @property
    def batch_idx(self) -> int:
        """
        Returns the current batch index based on the data index and batch size.
        """
        return self.data_index // self.batch_size
    
    @property
    def inner_idx(self) -> int:
        """
        Returns the index of the current data point within the current batch.
        """
        return self.data_index % self.batch_size

    def _norm(self, X):
        if not self.l2_normalize:
            return X
        # avoid div by 0
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n, 1e-12)

    def on_batch_collected(self, data_batch: np.ndarray) -> bool:
        """
        Process a batch of data points to update centroids or perform other operations.
        This can include normalization or other preprocessing steps.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def process_data_point(self, data_point: np.ndarray) -> bool:
        """
        Process a single data point to prepare it for clustering.
        This can include normalization or other preprocessing steps.
        """
        self.data_batch.append(data_point)
        has_ended = False
        
        if self.inner_idx == self.batch_size - 1:
            # Process the batch of data points
            data_batch_np = np.array(self.data_batch)
            self.data_batch = []

            # Update centroids or perform other operations on the batch
            has_ended = self.on_batch_collected(data_batch_np)

        self.data_index += 1
        return has_ended
    
    def flush(self) -> None:
        """
        Flush any remaining data points in the current batch.
        This is useful for processing the last batch if it is not full.
        """
        if self.data_batch:
            data_batch_np = np.array(self.data_batch)
            self.data_batch = []
            self.on_batch_collected(data_batch_np)
            self.data_index += len(data_batch_np)

class BaseInitializer(ABC):
    def __init__(self):
        self.centroids: Optional[np.ndarray] = None

    @abstractmethod
    def process_seq(self, data: np.ndarray, last_seq: bool = False):
        ...
    
    def finalize(self) -> np.ndarray:
        assert self.centroids is not None
        return self.centroids

class KMeansPlusPlusInitializer(BatchwiseUpdater, BaseInitializer):
    def __init__(self, num_clusters: int, batch_size: int = 1024):
        super().__init__(batch_size=batch_size)
        self.num_clusters = num_clusters

        self.centroids = []
    
    def on_batch_collected(self, data_tensors: np.ndarray) -> bool:
        # choose first centroid uniformly at random from the data points
        if len(self.centroids) == 0:
            random_idx = np.random.randint(0, data_tensors.shape[0])
            self.centroids.append(data_tensors[random_idx])
            # print(f"Selected initial centroid: {self.centroids[-1]}")
            return False
        
        # compute squared distances from the existing centroids
        # and use as weights for the next centroid selection
        if len(self.centroids) < self.num_clusters:
            distances = cdist(data_tensors, self.centroids).min(axis=1) ** 2
            probabilities = distances / distances.sum()
            
            # select next centroid based on the computed probabilities
            next_centroid_idx = np.random.choice(
                np.arange(data_tensors.shape[0]), p=probabilities
            )
            self.centroids.append(data_tensors[next_centroid_idx])
            # print(f"Selected new centroid: {self.centroids[-1]}")
            return False
        
        # If we have enough centroids, we can stop
        return True

class StandardInitializer(BaseInitializer):
    def __init__(self, num_clusters: int):
        self.num_clusters = num_clusters
        self.data: Optional[np.ndarray] = None
        self.centroids: Optional[np.ndarray] = None
    
    def process_seq(self, data: np.ndarray, last_seq: bool = False):
        if self.data is None:
            self.data = data
        else:
            self.data = np.concatenate((self.data, data), axis=0)

        if last_seq:
            assert self.data is not None
            self.centroids = np.random.permutation(self.data)[:self.num_clusters]

class StreamingStandardInitializer(BaseInitializer):
    """
    Implements reservoir sampling to use random encoder states as initial centroids.
    See also 'Algorithm R'.
    """
    def __init__(self, num_clusters: int, seed: int = 0):
        self.num_clusters = num_clusters
        self.samples = []
        self.counter = 0
        self.rng = np.random.RandomState(seed)
    
    def process_seq(self, data: np.ndarray, last_seq: bool = False):
        for x in data:
            self.counter += 1
            if len(self.samples) < self.num_clusters:
                self.samples.append(x)
                continue

            assert self.counter > self.num_clusters
            j = self.rng.randint(0, self.counter - 1)
            if j < self.num_clusters:
                print("resample")
                self.samples[j] = x
    
    def finalize(self) -> np.ndarray:
        assert len(self.samples) == self.num_clusters
        return np.array(self.samples)

class PreloadCentroidsInitializer(BaseInitializer):
    def __init__(
        self,
        num_clusters: int,
        centroids_path: str,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.centroids_path = centroids_path
        self._loaded = False
    
    def process_seq(self, data: np.ndarray, last_seq: bool = False):
        if self._loaded:
            return

        centroids = np.load(self.centroids_path)
        if centroids.shape[0] != self.num_clusters:
            raise ValueError(
                f"Expected {self.num_clusters} centroids, got {centroids.shape[0]}"
            )
        self.centroids = centroids

        self._loaded = True

class PickleCentroidRandomMapInitializer(BaseInitializer):
    def __init__(self, kmeans_path: str="/u/jxu/setups/unsupervised/2025-05-30--marten-unsupervised/output/w2v_kmeans_41.pkl", num_clusters: int=41):
        super().__init__()
        self.path = kmeans_path
        self.num_clusters = num_clusters
        self._loaded = False

    def process_seq(self, data: np.ndarray, last_seq: bool = False):
        if self._loaded:
            return
        with open(self.path, "rb") as f:
            payload = pickle.load(f)
        centroids = payload["cluster_centers"]
        if centroids.shape[0] != self.num_clusters:
            raise ValueError(
                f"Expected {self.num_clusters} centroids, got {centroids.shape[0]}"
            )
        self.centroids = centroids.astype(np.float64, copy=False)
        self._loaded = True


class PickleCentroidFrequencyOrderMapInitializer(BaseInitializer):
    def __init__(
        self,
        num_clusters: int,
        kmeans_path: str = "/u/jxu/setups/unsupervised/2025-05-30--marten-unsupervised/output/w2v_kmeans_41.pkl",
        alignments_path: str = "/u/jxu/setups/unsupervised/2025-05-30--marten-unsupervised/output/sampled_alignments.pkl",
        lexicon_path: str = "/u/mann/experiments/clones/2025-05-30--marten-unsupervised/test/librasr_recog/lex.xml",
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.kmeans_path = kmeans_path
        self.alignments_path = alignments_path
        self.lexicon_path = lexicon_path
        self._loaded = False

    @staticmethod
    def _load_lexicon_phonemes(path: str) -> list[str]:
        with uopen(path) as fp:
            root = ET.parse(fp).getroot()
            return [elem.findtext("symbol") for elem in root.findall(".//phoneme-inventory/phoneme")]

    def process_seq(self, data: np.ndarray, last_seq: bool = False):
        if self._loaded:
            return

        with open(self.kmeans_path, "rb") as f:
            kmeans_payload = pickle.load(f)
        with open(self.alignments_path, "rb") as f:
            alignments_payload = pickle.load(f)

        cluster_centers = kmeans_payload["cluster_centers"]
        cluster_counts = kmeans_payload["cluster_assignment_counts"]
        phoneme_counts = alignments_payload["phoneme_symbol_counts"]
        lexicon_phonemes = self._load_lexicon_phonemes(self.lexicon_path)

        if cluster_centers.shape[0] != self.num_clusters:
            raise ValueError(
                f"Expected {self.num_clusters} centroids, got {cluster_centers.shape[0]}"
            )
        if len(cluster_counts) != self.num_clusters:
            raise ValueError(
                f"Expected {self.num_clusters} cluster counts, got {len(cluster_counts)}"
            )
        if len(lexicon_phonemes) != self.num_clusters:
            raise ValueError(
                f"Expected {self.num_clusters} lexicon phonemes, got {len(lexicon_phonemes)}"
            )

        sorted_cluster_indices = [
            idx for idx, _ in sorted(enumerate(cluster_counts), key=lambda kv: (-kv[1], kv[0]))
        ]
        sorted_phonemes = sorted(
            lexicon_phonemes,
            key=lambda phon: (-phoneme_counts.get(phon, 0), lexicon_phonemes.index(phon)),
        )
        phoneme_to_index = {phon: idx for idx, phon in enumerate(lexicon_phonemes)}

        centroids = np.empty_like(cluster_centers, dtype=np.float64)
        remaining_cluster_indices = iter(sorted_cluster_indices)
        for phoneme in sorted_phonemes:
            out_idx = phoneme_to_index[phoneme]
            cluster_idx = next(remaining_cluster_indices)
            centroids[out_idx] = cluster_centers[cluster_idx]

        self.centroids = centroids
        self._loaded = True


class PickleCheatingCentroidInitializer(BaseInitializer):
    def __init__(
        self,
        num_clusters: int,
        centroids_path: str = "/u/jxu/setups/unsupervised/2025-05-30--marten-unsupervised/output/cheating_centroids.pkl",
        lexicon_path: str = "/u/mann/experiments/clones/2025-05-30--marten-unsupervised/test/librasr_recog/lex.xml",
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.centroids_path = centroids_path
        self.lexicon_path = lexicon_path
        self.unknown_fill_value = 1e6
        self._loaded = False

    def process_seq(self, data: np.ndarray, last_seq: bool = False):
        if self._loaded:
            return

        with open(self.centroids_path, "rb") as f:
            payload = pickle.load(f)

        centroid_map = payload["centroids"]
        lexicon_phonemes = PickleCentroidFrequencyOrderMapInitializer._load_lexicon_phonemes(self.lexicon_path)
        if len(lexicon_phonemes) != self.num_clusters:
            raise ValueError(
                f"Expected {self.num_clusters} lexicon phonemes, got {len(lexicon_phonemes)}"
            )

        first_centroid = next(iter(centroid_map.values()))
        feature_dim = int(first_centroid.shape[0])
        centroids = np.empty((self.num_clusters, feature_dim), dtype=np.float64)

        for idx, phoneme in enumerate(lexicon_phonemes):
            if phoneme == "[UNKNOWN]":
                centroids[idx] = np.full(feature_dim, self.unknown_fill_value, dtype=np.float64)
                continue
            if phoneme not in centroid_map:
                raise KeyError(f"Missing centroid for phoneme {phoneme!r}")
            centroids[idx] = np.asarray(centroid_map[phoneme], dtype=np.float64)

        self.centroids = centroids
        self._loaded = True


class BatchwiseKMeansUpdater(BatchwiseUpdater):
    def __init__(self, num_clusters: int, batch_size: int = 1024, l2_normalize: bool = True):
        super().__init__(batch_size=batch_size, l2_normalize=l2_normalize)
        self.num_clusters = num_clusters
        self.centroids = None
        self.objective_value = None
    
    def init(self, centroids: np.ndarray):
        """
        Initialize the centroids with the provided array.
        """
        if centroids.shape[0] != self.num_clusters:
            raise ValueError(
                f"Expected {self.num_clusters} centroids, got {centroids.shape[0]}"
            )
        self.centroids = centroids.astype(np.float64, copy=True)
        self.cluster_counts = np.zeros(centroids.shape[0], dtype=np.float32)
    
    def on_batch_collected(self, data_batch: np.ndarray) -> bool:
        """
        Update centroids based on the current batch of data points.
        """
        assert self.centroids is not None, "Centroids must be initialized before updating."
        X = self._norm(data_batch.astype(np.float64))
        dist = cdist(X, self.centroids)
        closest = np.argmin(dist, axis=1)

        for cidx in range(self.num_clusters):
            mask = closest == cidx
            count = int(mask.sum())
            if count == 0:
                # If no points are assigned to this centroid, keep the old centroid
                continue
            # update the centroid for this cluster
            # sum of new cluster points
            new_sum = X[mask].sum(axis=0)
            prev = self.cluster_counts[cidx]
            self.cluster_counts[cidx] = prev + count

            # weighted update of the cluster centroid
            self.centroids[cidx, :] = (
                self.centroids[cidx, :] * prev + new_sum
            ) / (
                self.cluster_counts[cidx]
            )
        
        # compute value of k-means objective
        dist_new = cdist(X, self.centroids)
        kmeans_objective = float(np.mean(np.min(dist_new, axis=1)**2))
        self.objective_value = kmeans_objective
        print(f"Current k-means objective value: {self.objective_value}")

        # # print(f"Updated centroids: {self.centroids}")
        # print(f"Current k-means objective value: {self.objective_value}")
        # print("Δbatch:", float((
        #     (dist_new.min(axis=1)**2).mean() - (dist.min(axis=1)**2).mean()
        # )))
        
        # We never provide an ending signal here
        return False

class EMAUpdater(BatchwiseUpdater):
    def __init__(
            self, num_clusters: int, batch_size: int = 1024,
            l2_normalize: bool = True,
            eta: float = 0.05):
        super().__init__(batch_size=batch_size, l2_normalize=l2_normalize)
        self.num_clusters = num_clusters
        self.eta = eta
        self.centroids = None
    
    def init(self, centroids: np.ndarray):
        """
        Initialize the centroids with the provided array.
        """
        if centroids.shape[0] != self.num_clusters:
            raise ValueError(
                f"Expected {self.num_clusters} centroids, got {centroids.shape[0]}"
            )
        self.centroids = centroids.astype(np.float64, copy=True)
    
    def on_batch_collected(self, data_batch: np.ndarray) -> bool:
        """
        Update centroids using exponential moving average based on the current batch of data points.
        """
        assert self.centroids is not None, "Centroids must be initialized before updating."
        X = self._norm(data_batch.astype(np.float64))
        dist = cdist(X, self.centroids)
        closest = np.argmin(dist, axis=1)

        for cidx in range(self.num_clusters):
            mask = closest == cidx
            count = int(mask.sum())
            if count == 0:
                # If no points are assigned to this centroid, keep the old centroid
                continue
            # update the centroid for this cluster
            # sum of new cluster points
            batch_mean = X[mask].mean(axis=0)

            # weighted update of the cluster centroid
            self.centroids[cidx, :] = self.centroids[cidx, :] * (1 - self.eta) + batch_mean * self.eta
        
        # compute value of k-means objective
        dist_new = cdist(X, self.centroids)
        kmeans_objective = float(np.mean(np.min(dist_new, axis=1)**2))
        self.objective_value = kmeans_objective

        # print(f"Updated centroids: {self.centroids}")
        print(f"Current k-means objective value: {self.objective_value}")
        print("Δbatch:", float((
            (dist_new.min(axis=1)**2).mean() - (dist.min(axis=1)**2).mean()
        )))
        
        # We never provide an ending signal here
        return False

class NnOutputClusteringCallback(ForwardCallbackIface):
    K_MEANS_LOSS_FNAME = "k_means_loss.txt"

    def __init__(
        self,
        num_clusters: int,
        batch_size: int = 1024,
        initializer: Optional[BaseInitializer] = None,
        kmeans_updater: Optional[BatchwiseUpdater] = None,
        writer_batch_size: int = 1024,
    ):
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        # self.initializer = initializer if initializer is not None else KMeansPlusPlusInitializer(
        #     num_clusters=num_clusters, batch_size=batch_size
        # )
        if initializer is None:
            initializer = StreamingStandardInitializer(num_clusters)
        self.initializer = initializer
        self.kmeans_updater = kmeans_updater if kmeans_updater is not None else BatchwiseKMeansUpdater(
            num_clusters=num_clusters, batch_size=batch_size
        )
        self._write_batch_size = writer_batch_size  # how many batches to write to file at once
        self._write_buffer = []

    def init(self, *, model: Optional[torch.nn.Module] = None):
        print(f"Starting k-means clustering with {self.num_clusters} clusters and batch size {self.batch_size}.")
        self.n = 1
        # self.avg_probs = None
        self.init_data_tensors = []
        self.collect_data = []
        self.centroids = None
        self.is_initialized = False

        # self.initializer = SlidingWindowKMeansPPInitializer(
        #     self.num_clusters,
        #     self.batch_size,
        # )

        # self.kmeans_updater = BatchwiseKMeansUpdater(
        #     num_clusters=self.num_clusters,
        #     batch_size=self.batch_size,
        # )

        self.eval_pool = EvalReservoir(cap=50000, l2_normalize=True, seed=0)
    
    def _write_buffer_to_file(self):
        """
        Write the contents of the write buffer to a file.
        This is called when the buffer reaches a certain size.
        """
        with open(self.K_MEANS_LOSS_FNAME, "a+") as f:
            for value in self._write_buffer:
                f.write(f"{value}\n")
            self._write_buffer.clear()
    
    def process_point(self, x: np.ndarray):
        if not self.is_initialized:
            has_ended = self.initializer.process_data_point(x)
            if has_ended:
                self.is_initialized = True
                self.centroids = np.array(self.initializer.centroids)
                self.kmeans_updater.init(self.centroids)
            return
        
        self.eval_pool.add(np.expand_dims(x, axis=0))

        self.kmeans_updater.process_data_point(x)

        if self.kmeans_updater.inner_idx != 0:
            return

        print(f"Processing batch {self.kmeans_updater.batch_idx}.")
        if self.kmeans_updater.batch_idx % self._write_batch_size == 0:
            eval_sse = self.eval_pool.eval_sse(self.kmeans_updater.centroids)
            # log the k-means objective value every _write_batch_size batches
            if eval_sse is not None:
                print("Current evaluation SSE (fixed pool):", eval_sse)
                self._write_buffer.append(eval_sse)
                if len(self._write_buffer) >= self._write_batch_size:
                    self._write_buffer_to_file()

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        hidden_state_tensor = outputs["output"].raw_tensor
        assert isinstance(hidden_state_tensor, np.ndarray)

        self.eval_pool.add(hidden_state_tensor)

        for x in hidden_state_tensor:
            self.process_point(x)

    def finish(self):
        # flush any remaining data points in the k-means updater
        self.kmeans_updater.flush()

        # write buffer again case it has remaining values
        if self._write_buffer:
            self._write_buffer_to_file()

        # Save the final centroids to a file
        np.savetxt("centroids.txt", self.kmeans_updater.centroids)

class RepositoryState(Enum):
    READ = "read"
    WRITE = "write"

class TracebackRepository:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.state = RepositoryState.READ

        self.hdf_writer = SimpleHDFWriter(filename="traceback.hdf", dim=1)
    
    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["hdf_writer"] = None
        return d

    def __setstate__(self, d) -> None:
        self.__dict__ = d
        self.hdf_writer = SimpleHDFWriter(filename="traceback.hdf", dim=1)
    
    def store(self, seq_tag: str, traceback: Any):
        assert self.state is RepositoryState.READ

    def get(self, seq_tag: str):
        pass

class BaseRepository(ABC):
    def start_write(self):
        pass
    
    def end_write(self):
        pass
    
    @abstractmethod
    def get(self, key: str):
        ...
    
    @abstractmethod
    def store(self, key: str, traceback: Any):
        ...
    
class SimpleDictRepository(BaseRepository):
    def __init__(self):
        self.data = {}
    
    def get(self, key):
        return self.data[key]
    
    def store(self, key, traceback):
        self.data[key] = traceback

class PlyvelTracebackRepository(BaseRepository):
    def __init__(self, batch_size, db_path="./traceback_db"):
        self.batch_size = batch_size
        self._state = RepositoryState.READ
        self.db_path = db_path

        self._seq_counter = 0

        self._db = None
        # self.load_db()
    
    # def load_db(self):
    #     import plyvel
    #     self.db = plyvel.DB(self.db_path, create_if_missing=True)
    
    @property
    def db(self):
        # if self._db is None:
        _db = getattr(self, "_db", None)
        if _db is None or _db.closed:
            import plyvel
            self._db = plyvel.DB(self.db_path, create_if_missing=True)
        return self._db
    
    def __getstate__(self) -> dict:
        print(f"{self}.__getstate__")
        d = dict(self.__dict__)
        # self.db.close()
        d["_db"] = None
        d["_wb"] = None
        # print(self._db)
        return d

    def __setstate__(self, state) -> None:
        print(f"{self}.__setstate__")
        print(state)
        self.__dict__.update(state)
        # if self._state == RepositoryState.WRITE:
        #     print("Setting db and write_batch")
        #     self._wb = self.db.write_batch()
        #     assert not self.db.closed
    
    def start_write(self):
        self._state = RepositoryState.WRITE
        self._wb = self.db.write_batch()
    
    def store(self, seq_tag: str, traceback: Any):
        assert self._state is RepositoryState.WRITE
        key = bytes(seq_tag, encoding="utf-8")
        obj = pickle.dumps(traceback)
        try:
            self._wb.put(key, obj)
        except RuntimeError as re:
            raise RuntimeError(f"{self}._db.closed == {self._db.closed}")
        self._seq_counter += 1

        if self._seq_counter >= self.batch_size:
            # flush and reset counter
            self._wb.write()
            self._wb.clear()
            self._seq_counter = 0
    
    def end_write(self):
        self._state = RepositoryState.READ
        # flush remaining data in batch
        self._wb.write()
        self._wb.clear()
        self._seq_counter = 0
        del self._wb
        self._wb = None

    def get(self, seq_tag: str) -> Any:
        assert self._state is RepositoryState.READ
        key = bytes(seq_tag, encoding="utf-8")
        data = self.db.get(key)
        if data is None:
            raise KeyError("No entry for given sequence tag in database.")
        obj = pickle.loads(data)
        return obj


class GuidedClusteringPhase(Enum):
    INITIALIZATION = 0
    RECOGNITION = 1
    CLUSTERING = 2

    def transition(self, callback: Optional[Callable] = None):
        edges = {
            GuidedClusteringPhase.INITIALIZATION: GuidedClusteringPhase.RECOGNITION,
            GuidedClusteringPhase.RECOGNITION: GuidedClusteringPhase.CLUSTERING,
            GuidedClusteringPhase.CLUSTERING: GuidedClusteringPhase.RECOGNITION
        }
        out = edges[self]
        if callback is not None:
            callback(out, old_phase=self)
        
        return out

class Saver:
    def __init__(self):
        self._idx = 0
    
    def save(self, tensor, name):
        fname = f"{name}.{self._idx}.pt"
        torch.save(tensor, fname)

saver = Saver()

class PhonemeIdxMap(UserDict):
    def __init__(self, lexicon_path):
        self.data = self.load_lexicon_map(lexicon_path)

    def load_lexicon_map(self, lexicon_path: str):
        lex = Lexicon()
        lex.load(lexicon_path)

        return {phon: i for i, phon in enumerate(lex.phonemes)}
    
    def apply(self, it: Iterable[str]) -> List[int]:
        return [self[phon] for phon in it] 
    
    def inverse(self) -> dict:
        return {idx: phon for phon, idx in self.data.items()}

def hyp_from_traceback(traceback: List[Any]) -> str:
    return " ".join(item.lemma for item in traceback)

def _traceback_to_score(traceback: List[Any]) -> float:
    if len(traceback) > 0:
        return traceback[-1].am_score + traceback[-1].lm_score
    else:
        return float("inf")

def one_hot_numpy(labels: np.ndarray, num_labels: int):
    return np.eye(num_labels)[labels]

class GuidedKMeansClusteringCallback(NnOutputClusteringCallback):
    CENTROIDS_FPATTERN_RAW = "centroids.{}.npy"
    CENTROIDS_REGEX = re.compile(CENTROIDS_FPATTERN_RAW.format("(\\d+)"))

    def __init__(
        self,
        num_clusters: int,
        recognition_config: str,
        lexicon_path: str,
        num_seqs: int,
        distance_scale: float = 1.0,
        batch_size: int = 1024,
        initializer: Optional[BaseInitializer] = None,
        kmeans_updater: Optional[BatchwiseUpdater] = None,
        writer_batch_size: int = 1024,
        traceback_write_chunk_size: int = 128,
        load_last_epoch: int | None = None,
        subsampling: Optional[int] = None,
        pooling_function: str = "maxpool_time_np",
        pool_for_init: bool = True,
        gaussian_model: GaussianModelNumpy | None = None,
        verbosity: int = 1,
        num_workers: int | None = 7,
        task_timeout: float | None = 1800.0,
    ):
        super().__init__(
            num_clusters=num_clusters,
            batch_size=batch_size,
            initializer=initializer,
            kmeans_updater=kmeans_updater,
            writer_batch_size=writer_batch_size,
        )
        self.recognition_config = recognition_config

        self.phoneme_map = PhonemeIdxMap(lexicon_path)

        self.traceback_write_chunk_size = traceback_write_chunk_size
        self.traceback_repository = None

        self.seq_tags_seen = set()
        self.seq_count = 0
        self.current_epoch = 0
        self.current_seq = 0
        self.initialization_offset = None
        self.num_seqs = num_seqs
        self._last_epoch = load_last_epoch
        self.subsampling = subsampling

        self.distance_scale = distance_scale

        self.config = None
        self.recognizer = ParallelSegmentRecognizer(
            recognition_config, num_workers=num_workers, task_timeout=task_timeout
        )
        self.gaussian_model = gaussian_model

        self.centroid_updater = RunningAverageUpdater((num_clusters, 1024))
        self.score_updater = RunningAverageUpdater(())
        self.score_history = []

        self.statistics_logger = get_default_logger(self.phoneme_map)

        self.pool = PoolingRegistry.select(
            pooling_function,
            stride=subsampling,
            kernel_size=2 * subsampling if subsampling else None,
        )
        self.pool_for_init = pool_for_init

        self.tracebacks = {}
        self.verbosity = verbosity
    
    def init(self, *, model: Optional[torch.nn.Module] = None):
        super().init(model=model)
        self.recognizer.start()

        self.traceback_repository = PlyvelTracebackRepository(self.traceback_write_chunk_size)
        # self.traceback_repository = SimpleDictRepository()

        if self._last_epoch is not None:
            self.load_last_epoch(self._last_epoch)
            self.phase = GuidedClusteringPhase.RECOGNITION
            self._on_phase_transition(self.phase)
            return

        has_loaded = self.maybe_load_centroids()

        if not has_loaded:
            self.phase = GuidedClusteringPhase.INITIALIZATION
        else:
            self.phase = GuidedClusteringPhase.RECOGNITION
        self._on_phase_transition(self.phase)

    def maybe_load_centroids(self) -> bool:
        """
        Tries to load previously saved centroids.
        
        Returns whether any previous centroids were found and adjusts the epoch accordingly.
        
        :return: whether any previous centroids were found
        """
        centroid_files = glob.glob(self.CENTROIDS_FPATTERN_RAW.format("*"))

        if not centroid_files:
            return False

        def get_epoch(fname: str) -> int:
            m = self.CENTROIDS_REGEX.match(fname)
            if m:
                return int(m.group(1))
            raise ValueError("Did not match pattern")

        last_epoch = max(map(get_epoch, centroid_files))

        last_epoch_centroids_file = self.CENTROIDS_FPATTERN_RAW.format(last_epoch)
        assert os.path.exists(last_epoch_centroids_file)

        print(f"Loading centroids from epoch {last_epoch}.")

        # do the actual loading
        self.centroids = np.load(last_epoch_centroids_file)

        self.current_epoch = last_epoch * 2 + 1

        return True
    
    def load_last_epoch(self, last_epoch: int):
        last_epoch_centroids_file = self.CENTROIDS_FPATTERN_RAW.format(last_epoch)
        assert os.path.exists(last_epoch_centroids_file)

        print(f"Loading centroids from epoch {last_epoch}.")

        # do the actual loading
        self.centroids = np.load(last_epoch_centroids_file)

        self.current_epoch = last_epoch * 2 + 1

    def _on_phase_transition(self, new_phase: GuidedClusteringPhase, old_phase: Optional[GuidedClusteringPhase] = None):
        # transition out of initialization
        if old_phase is GuidedClusteringPhase.INITIALIZATION:
            self.centroids = self.initializer.finalize()
            np.save("centroids.0.npy", self.centroids)
            print(f"Finished Initialization: centroids = {self.centroids}")
        
        # transition in and out of recognition
        if new_phase is GuidedClusteringPhase.RECOGNITION:
            self.traceback_repository.start_write()
            self.score_updater = RunningAverageUpdater(())
            self.unigram_counter = RelativeFrequencyUpdater((len(self.phoneme_map),))

            self.statistics_logger.start_epoch()
        
        if old_phase is GuidedClusteringPhase.RECOGNITION:
            self.traceback_repository.end_write()
            current_score = self.score_updater.value
            self.score_history.append(current_score)
            with open("scores.txt", "w+") as sf:
                sf.write("\n".join(map(str, self.score_history)))
            print(f"Finished Recognition with score {current_score}")

            self.statistics_logger.end_epoch(self.current_epoch // 2)

        # transition into and out of clustering
        if new_phase is GuidedClusteringPhase.CLUSTERING:
            self.centroid_updater = RunningAverageUpdater((self.num_clusters, 1024))

        if old_phase is GuidedClusteringPhase.CLUSTERING:
            new_centroids = self.centroid_updater.value
            # Phonemes never assigned this pass would become zero vectors; keep their old centroid
            # so they remain viable candidates in the next recognition pass rather than dying out.
            dead_mask = self.centroid_updater.counts == 0
            new_centroids[dead_mask] = self.centroids[dead_mask]
            self.centroids = new_centroids
            centroids_file = f"centroids.{self.current_epoch // 2}.npy"
            print(f"Saving centroids to {centroids_file}")
            np.save(centroids_file, self.centroids)
        
        # handle process
        self.progress_logger = ProgressLogger(
            self.num_seqs,
            bar_length=40,
            logging_step=32,
        )
        self.progress_logger.start()
    
    def maybe_transition_phase(self, last_seq):
        """
        Determine the current phase of processing based on the sequence tag.
        Phase 1: Generating pseudo-labels via recognition.
        Phase 2: Using pseudo-labels as targets for clustering.
        """
        if self.current_seq == self.initialization_offset or last_seq:
            if self.phase == GuidedClusteringPhase.RECOGNITION:
                # _on_phase_transition's "old_phase is RECOGNITION" branch
                # reads self.score_updater.value and calls
                # traceback_repository.end_write() - every recognition
                # submitted this phase must be applied (repository stored,
                # score/statistics updated) before that runs, and before the
                # following CLUSTERING phase starts reading the repository.
                self._drain_recognition()
            self.phase = self.phase.transition(self._on_phase_transition)
            print(f"Starting Phase {self.phase} in epoch {self.current_epoch}")

    def increase_epoch_and_seq(self, last_seq: bool = False):
        if last_seq:
            self.current_epoch += 1
            self.current_seq = 0
            return
        self.current_seq += 1

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
    
    def update_centroids(self, hidden_states: np.ndarray, centroid_idxs: np.ndarray):
        assert self.centroids is not None
        assert len(hidden_states) == len(centroid_idxs), \
            f"Shape of hidden states {hidden_states.shape} does not match shape of centroid indices {centroid_idxs.shape}"
        idx_matrix = one_hot_numpy(centroid_idxs, self.num_clusters) # [T, C]
        idx_counts = idx_matrix.sum(0) # [C]
        feature_sums = idx_matrix.T @ hidden_states # [C, T] x [T, F] = [C, F]
        self.centroid_updater.update(feature_sums, idx_counts)

    def _apply_recognition_result(self, seq_tag: str, traceback: List[PlainTracebackItem]):
        """
        Everything that used to run inline in process_seq() right after
        self.search_algo.recognize_segment() returned, now applied once the
        (asynchronously submitted) result is available - see _drain_recognition().
        """
        self.statistics_logger.read_traceback(traceback)

        if self.verbosity >= 2:
            print(f"{traceback=}")

        segments = np.asarray([
            (self.phoneme_map[item.lemma], item.start_time, item.end_time)
            for item in traceback
        ])
        idx_list = segments_to_array(segments).tolist()
        self.traceback_repository.store(seq_tag, idx_list)
        self.tracebacks[seq_tag] = idx_list
        score = _traceback_to_score(traceback)
        self.score_updater.update_single(score)

    def _drain_recognition(self):
        """
        Block until every recognition submitted since the last drain is
        done, and apply results in submission order (some downstream state,
        e.g. SampledTracebackPrinter's reservoir sample, is keyed on call
        order, not completion order).
        """
        for seq_tag, traceback in self.recognizer.drain():
            self._apply_recognition_result(seq_tag, traceback)

    def process_seq(self, *, seq_tag: str, outputs: TensorDict, last_seq: bool = False):
        # processing happens in three phases:
        # 0. initialization of centroids via k-means++
        # 1. in the first phase we use the potentially random cluster centroids to
        #    generate pseudo-labels via recognition
        # 2. in the second phase we use the pseudo-labels as targets for clustering
        # After 2. we can go back to 1. with updated centroids.
        # We decide in which phase we are by observing whether we have had a full path through
        # all the seq_tags, i.e. if a seq_tag appears again, we are in phase 2

        assert isinstance(self.phase, GuidedClusteringPhase)

        last_seq = self.current_seq + 1 == self.num_seqs

        hidden_state_tensor = outputs["output"].raw_tensor
        assert isinstance(hidden_state_tensor, np.ndarray)
        assert self.traceback_repository is not None

        if self.verbosity >= 2:
            print(f"Processing sequence {seq_tag} in epoch {self.current_epoch}, phase {self.phase}.")
        hidden_state_tensor_pre_subsample = hidden_state_tensor.copy()
        if self.subsampling:
            hidden_state_tensor = self.pool(
                hidden_state_tensor,
                # stride=self.subsampling,
                # kernel_size=2 * self.subsampling,
            )

        if self.phase == GuidedClusteringPhase.INITIALIZATION:
            assert isinstance(hidden_state_tensor, np.ndarray)
            self.initializer.process_seq(
                data=(
                    hidden_state_tensor
                    if self.pool_for_init
                    else hidden_state_tensor_pre_subsample
                ),
                # data=hidden_state_tensor,
                last_seq=last_seq
            )
        
        if self.phase == GuidedClusteringPhase.RECOGNITION:
            # assert self.is_initialized, "Not initialized before recognition phase"
            distances = self.compute_squared_distances(
                seq_tag=seq_tag,
                hidden_states=hidden_state_tensor,
            )
            scaled_distances = distances * self.distance_scale
            # Non-blocking: the result (and everything that used to happen
            # here immediately - statistics, repository store, score update)
            # is applied later in _drain_recognition(), called from
            # maybe_transition_phase() before this phase ends.
            self.recognizer.submit(seq_tag, scaled_distances)

        if self.phase == GuidedClusteringPhase.CLUSTERING:
            try:
                idx_list = self.traceback_repository.get(seq_tag)
            except KeyError:
                print(f"{seq_tag=}")
                print(f"{len(self.tracebacks)=}")
                print(f"{self.tracebacks=}")
                raise
            idx_array = np.array(idx_list)
            try:
                self.update_centroids(hidden_state_tensor, idx_array)
            except AssertionError as ae:
                print(f"{hidden_state_tensor.shape=}")
                print(f"{idx_array.shape=}")
                print(f"{seq_tag=}")
                raise

        self.progress_logger.progress(self.current_seq)
        
        self.maybe_transition_phase(last_seq)
        self.increase_epoch_and_seq(last_seq)

    def finish(self):
        self.recognizer.shutdown()
        # flush any remaining data points in the centroid updater
        # last_seq = self.current_seq == self.num_seqs
        # if not last_seq:
        #     self.maybe_transition_phase(last_seq=True)


''' Not used at the moment, might need some fixes when enabled again
class HierarchicalGuidedKMeansClusteringCallback(NnOutputClusteringCallback):
    CENTROIDS_FPATTERN_RAW = "centroids.{}.npy"
    CENTROIDS_REGEX = re.compile(CENTROIDS_FPATTERN_RAW.format("(\\d+)"))

    def __init__(
        self,
        num_clusters: int,
        recognition_config: str,
        lexicon_path: str,
        num_seqs: int,
        phoneme_frequency_path: str,
        batch_size: int = 1024,
        # initializer: Optional[BaseInitializer] = None,
        kmeans_updater: Optional[BatchwiseUpdater] = None,
        writer_batch_size: int = 1024,
        traceback_write_chunk_size: int = 128,
        load_last_epoch: int | None = None,
        split_direction: Optional[np.ndarray] = None,
        split_eps: float = 1.0,
        subsampling: Optional[int] = None,
    ):
        super().__init__(
            num_clusters=num_clusters,
            batch_size=batch_size,
            # initializer=initializer,
            kmeans_updater=kmeans_updater,
            writer_batch_size=writer_batch_size,
        )
        self.recognition_config = recognition_config

        self.phoneme_map = PhonemeIdxMap(lexicon_path)

        self.traceback_write_chunk_size = traceback_write_chunk_size
        self.traceback_repository = None

        self.load_init_phoneme_split(phoneme_frequency_path)

        self.seq_tags_seen = set()
        self.seq_count = 0
        self.current_epoch = 0
        self.current_seq = 0
        self.initialization_offset = None
        self.num_seqs = num_seqs
        self._last_epoch = load_last_epoch

        self.config = None
        self.search_algo = None

        self.single_centroid_initializer = RunningAverageUpdater((1, 1024))
        self.centroid_updater = RunningAverageUpdater((num_clusters, 1024))
        self.score_updater = RunningAverageUpdater(())
        self.score_history = []

        self.pca = StreamingPCA(n_components=1)
        self.subsampling = subsampling

        self.tracebacks = {}

        self.traceback_logger = TracebackLogger()
    
    def init(self, *, model: Optional[torch.nn.Module] = None):
        super().init(model=model)
        self.search_algo = self.init_search_algorithm()

        self.traceback_repository = PlyvelTracebackRepository(self.traceback_write_chunk_size)
        # self.traceback_repository = SimpleDictRepository()

        if self._last_epoch is not None:
            self.load_last_epoch(self._last_epoch)
            self.phase = GuidedClusteringPhase.RECOGNITION
            self._on_phase_transition(self.phase)
            return
    
        has_loaded = self.maybe_load_centroids()        

        # has_loaded = False

        if not has_loaded:
            self.phase = GuidedClusteringPhase.INITIALIZATION
        else:
            self.phase = GuidedClusteringPhase.RECOGNITION
        self._on_phase_transition(self.phase)

        # self.maybe_transition_phase(False)

    def load_init_phoneme_split(self, phoneme_frequency_path):
        print("Loading inital phoneme split...", end=" ")
        line_re = re.compile(r"(\d+)[\s\t]+(\w+)")
        silence_phon = "[SILENCE]"
        assert silence_phon in self.phoneme_map
        phon_split = ([silence_phon], [])
        with open(phoneme_frequency_path, "r") as fp:
            freq_dict = {}
            prev_freq = float("inf")
            for i, line in enumerate(fp, start=1):
                # phon, freq_str = line.strip("\n").split(" ")
                m = line_re.match(line)
                assert m is not None, "Could not match line"
                freq_str, phon = m.group(1), m.group(2)
                freq = int(freq_str)
                assert freq <= prev_freq
                prev_freq = freq
                freq_dict[phon] = freq
                phon_split[i % 2].append(phon)
        
        rem_phons = set(self.phoneme_map) - {silence_phon} - set(freq_dict)
        for j, p in enumerate(rem_phons, start=i+1):
            phon_split[j % 2].append(p)
        
        self.phoneme_split = phon_split

        print(f"Found: {','.join(phon_split[0])} | {','.join(phon_split[1])}")

    def init_search_algorithm(self):
        from librasr import Configuration, SearchAlgorithm

        config = Configuration()
        config.set_from_file(self.recognition_config)

        return SearchAlgorithm(config=config)
    
    def maybe_load_centroids(self) -> bool:
        """
        Tries to load previously saved centroids.
        
        Returns whether any previous centroids were found and adjusts the epoch accordingly.
        
        :return: whether any previous centroids were found
        """
        centroid_files = glob.glob(self.CENTROIDS_FPATTERN_RAW.format("*"))

        if not centroid_files:
            return False

        def get_epoch(fname: str) -> int:
            m = self.CENTROIDS_REGEX.match(fname)
            if m:
                return int(m.group(1))
            raise ValueError("Did not match pattern")

        last_epoch = max(map(get_epoch, centroid_files))

        last_epoch_centroids_file = self.CENTROIDS_FPATTERN_RAW.format(last_epoch)
        assert os.path.exists(last_epoch_centroids_file)

        print(f"Loading centroids from epoch {last_epoch}.")

        # do the actual loading
        self.centroids = np.load(last_epoch_centroids_file)

        self.current_epoch = last_epoch * 2 + 1

        return True
    
    def load_last_epoch(self, last_epoch: int):
        last_epoch_centroids_file = self.CENTROIDS_FPATTERN_RAW.format(last_epoch)
        assert os.path.exists(last_epoch_centroids_file)

        print(f"Loading centroids from epoch {last_epoch}.")

        # do the actual loading
        self.centroids = np.load(last_epoch_centroids_file)

        self.current_epoch = last_epoch * 2 + 1

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["search_algo"] = None

        return d

    def __setstate__(self, d) -> None:
        self.__dict__ = d
        self.search_algo = self.init_search_algorithm()

    def _on_phase_transition(self, new_phase: GuidedClusteringPhase, old_phase: Optional[GuidedClusteringPhase] = None):
        print("=" * 10)
        print(f"Starting phase {new_phase}")
        print("=" * 10)
        # transition out of initialization
        if old_phase is GuidedClusteringPhase.INITIALIZATION:
            # self.centroids = self.initializer.finalize()
            # self.single_centroid = self.single_centroid_initializer.value[0,:]
            pca_res = self.pca.finalize()
            print(f"PCA Output = {pca_res}")
            self.single_centroid = pca_res.mean
            np.save("single_centroid.npy", self.single_centroid)
            print(f"Finished Initialization: centroids = {self.single_centroid}")

            centroid_splits = self.split_centroid(self.single_centroid, pca_res.components[0, :])

            self.assign_centroid_split(centroid_splits)
            np.save(f"centroids.{self.current_epoch}.npy", self.centroids)

            input("Awaiting input")
        
        # transition in and out of recognition
        if new_phase is GuidedClusteringPhase.RECOGNITION:
            self.phoneme_counts = Counter()
            self.traceback_repository.start_write()
            self.score_updater = RunningAverageUpdater(())

        
        if old_phase is GuidedClusteringPhase.RECOGNITION:
            print(f"Phoneme Counts: {self.phoneme_counts}")
            self.traceback_repository.end_write()
            current_score = self.score_updater.value
            self.score_history.append(current_score)
            with open("scores.txt", "w+") as sf:
                sf.write("\n".join(map(str, self.score_history)))
            print(f"Finished Recognition with score {current_score}")

            self.traceback_logger.finalize()
            print("Quitting program since further phases are not implemented yet.")
            quit()

        # transition into and out of clustering
        if new_phase is GuidedClusteringPhase.CLUSTERING:
            self.centroid_updater = RunningAverageUpdater((self.num_clusters, 1024))

        if old_phase is GuidedClusteringPhase.CLUSTERING:
            self.centroids = self.centroid_updater.value
            np.save(f"centroids.{self.current_epoch}.npy", self.centroids)
        
        # handle process
        self.progress_logger = ProgressLogger(
            self.num_seqs,
            bar_length=40,
            logging_step=32,
        )
        self.progress_logger.start()
    
    def maybe_transition_phase(self, last_seq):
        """
        Determine the current phase of processing based on the sequence tag.
        Phase 1: Generating pseudo-labels via recognition.
        Phase 2: Using pseudo-labels as targets for clustering.
        """
        if self.current_seq == self.initialization_offset or last_seq:
            self.phase = self.phase.transition(self._on_phase_transition)
            print(f"Starting Phase {self.phase} in epoch {self.current_epoch}")
    
    def increase_epoch_and_seq(self, last_seq: bool = False):
        if last_seq:
            self.current_epoch += 1
            self.current_seq = 0
            return
        self.current_seq += 1
    
    def compute_distances(
        self,
        *,
        seq_tag: str,
        outputs: TensorDict,
    ) -> np.ndarray:
        centroids = self.centroids
        # assert centroids, "Centroids must be initialized before computing distances."
        hidden_states = outputs["output"].raw_tensor
        assert centroids is not None
        assert isinstance(hidden_states, np.ndarray)
        dist = cdist(hidden_states, centroids)
        return dist
    
    def update_centroids(self, hidden_states: np.ndarray, centroid_idxs: np.ndarray):
        assert self.centroids is not None
        assert len(hidden_states) == len(centroid_idxs)
        idx_matrix = one_hot_numpy(centroid_idxs, self.num_clusters) # [T, C]
        idx_counts = idx_matrix.sum(0) # [C]
        feature_sums = idx_matrix.T @ hidden_states # [C, T] x [T, F] = [C, F]
        self.centroid_updater.update(feature_sums, idx_counts)
    
    def initialize_single_centroid(self, data, last_seq: bool):
        pass
    
    def split_centroid(self, centroid, split_axis, split_scale=1.0):
        split_vec = split_scale * split_axis
        return centroid - split_vec, centroid + split_vec
    
    def assign_centroid_split(self, centroids):
        assert len(centroids) == len(self.phoneme_split)

        d = centroids[0].shape[0]
        n = len(self.phoneme_map)
        out_centroids = np.empty((n, d))

        for centroid, phonemes in zip(centroids, self.phoneme_split):
            for phon in phonemes:
                phon_idx = self.phoneme_map[phon]
                out_centroids[phon_idx, :] = centroid
        
        self.centroids = out_centroids
    
    def process_seq(self, *, seq_tag: str, outputs: TensorDict, last_seq: bool = False):
        # processing happens in three phases:
        # 0. initialization of centroids via k-means++
        # 1. in the first phase we use the potentially random cluster centroids to
        #    generate pseudo-labels via recognition
        # 2. in the second phase we use the pseudo-labels as targets for clustering
        # After 2. we can go back to 1. with updated centroids.
        # We decide in which phase we are by observing whether we have had a full path through
        # all the seq_tags, i.e. if a seq_tag appears again, we are in phase 2

        assert isinstance(self.phase, GuidedClusteringPhase)

        last_seq = self.current_seq + 1 == self.num_seqs

        hidden_state_tensor = outputs["output"].raw_tensor

        if self.subsampling:
            hidden_state_tensor = maxpool_time_np(
                hidden_state_tensor,
                stride=self.subsampling,
                kernel_size=2 * self.subsampling,
            )

        if self.phase == GuidedClusteringPhase.INITIALIZATION:
            self.pca.process_sequence(hidden_state_tensor)
        
        if self.phase == GuidedClusteringPhase.RECOGNITION:
            # assert self.is_initialized, "Not initialized before recognition phase"
            distances = self.compute_distances(
                seq_tag=seq_tag,
                outputs=outputs,
            )
            traceback = self.search_algo.recognize_segment(distances)
            # phon_seq = [item.lemma for item in traceback]
            self.phoneme_counts.update([item.lemma for item in traceback])
            idx_list = self.phoneme_map.apply(item.lemma for item in traceback)

            # print(f"traceback: {[item.lemma for item in traceback]}")

            print(f"{traceback=}")

            segments = np.asarray([
                (self.phoneme_map[item.lemma], item.start_time, item.end_time)
                for item in traceback
            ])
            idx_list = segments_to_array(segments).tolist()
            self.traceback_repository.store(seq_tag, idx_list)
            self.traceback_logger.feed(traceback)
            self.tracebacks[seq_tag] = idx_list
            score = _traceback_to_score(traceback)
            self.score_updater.update_single(score)
        
        if self.phase == GuidedClusteringPhase.CLUSTERING:
            try:
                idx_list = self.traceback_repository.get(seq_tag)
            except KeyError:
                print(f"{seq_tag=}")
                print(f"{len(self.tracebacks)=}")
                print(f"{self.tracebacks=}")
                raise
            idx_array = np.array(idx_list)
            self.update_centroids(hidden_state_tensor, idx_array)

        self.progress_logger.progress(self.current_seq)
        
        self.maybe_transition_phase(last_seq)
        self.increase_epoch_and_seq(last_seq)

        # input()
'''