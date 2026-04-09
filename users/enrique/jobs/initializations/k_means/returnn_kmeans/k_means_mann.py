__all__ = [
    "BatchwiseUpdater",
    "KMeansPlusPlusInitializer",
    "BatchwiseKMeansUpdater",
    "NnOutputClusteringCallback",
]

from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import TensorDict

import numpy as np
from scipy.spatial.distance import cdist
import torch
from typing import Optional


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
    __slots__ = ("batch_size", "data_index", "data_batch", "l2_normalize")

    def __init__(self, batch_size: int = 1024, l2_normalize: bool = False):
        self.batch_size = batch_size
        self.l2_normalize = l2_normalize
        self.data_index = 0
        self.data_batch = []

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


class SlidingWindowKMeansPPInitializer(BatchwiseUpdater):
    def __init__(
        self, num_clusters: int, batch_size: int = 1024,
        pool_size: int = 50000, min_pool_factor: int = 50,
        picks_per_batch: int | None = None,
        l2_normalize: bool = True, rng: np.random.Generator | None = None
    ):
        super().__init__(batch_size=batch_size, l2_normalize=l2_normalize)
        self.num_clusters = num_clusters
        self.pool_size = pool_size                 # max points kept
        self.min_pool = max(num_clusters * min_pool_factor, num_clusters * 10)
        self.picks_per_batch = picks_per_batch     # if None, fill all remaining in first eligible batch
        self.rng = rng or np.random.default_rng()

        self._pool = None
        self.centroids = []

    def _append_to_pool(self, X):
        if self._pool is None:
            self._pool = X
        else:
            self._pool = np.concatenate([self._pool, X], axis=0)
        # drop oldest if too large
        if self._pool.shape[0] > self.pool_size:
            overflow = self._pool.shape[0] - self.pool_size
            self._pool = self._pool[overflow:, :]

    def _kpp_select_from_pool(self, m):
        """Select m new centers via k++ from the current pool."""
        assert self._pool is not None and self._pool.shape[0] >= self.num_clusters
        X = self._pool

        # If no centroids yet, pick the first one uniformly at random
        if len(self.centroids) == 0:
            idx0 = self.rng.integers(0, X.shape[0])
            self.centroids.append(X[idx0])

        need = min(m, self.num_clusters - len(self.centroids))
        if need <= 0:
            return

        # distances wrt existing centroids
        d2 = np.min(cdist(X, np.stack(self.centroids, axis=0)), axis=1)**2
        s = float(d2.sum())

        if s == 0.0:
            # all points identical to existing centroid(s); just sample uniformly
            new_idx = self.rng.choice(X.shape[0], size=need, replace=False)
            for i in new_idx:
                self.centroids.append(X[i])
            return

        # sample 'need' without replacement with probs ∝ d2
        p = d2 / s
        # Avoid numerical issues (can happen if p has tiny residual negatives)
        p = np.clip(p, 0, 1)
        p = p / p.sum()

        # If pool is smaller than needed unique picks, fall back to replace=True
        replace = need > X.shape[0]
        new_idx = self.rng.choice(X.shape[0], size=need, replace=replace, p=p)
        for i in (np.unique(new_idx) if not replace else new_idx):
            self.centroids.append(X[i])

    def on_batch_collected(self, data_batch: np.ndarray) -> bool:
        X = self._norm(data_batch.astype(np.float64))
        self._append_to_pool(X)

        if len(self.centroids) >= self.num_clusters:
            return True  # done

        if self._pool.shape[0] < self.min_pool:
            return False  # wait until pool is large enough

        picks = (self.num_clusters - len(self.centroids)) if self.picks_per_batch is None \
                else min(self.picks_per_batch, self.num_clusters - len(self.centroids))

        self._kpp_select_from_pool(picks)
        return len(self.centroids) >= self.num_clusters


class KMeansPlusPlusInitializer(BatchwiseUpdater):
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
        initializer: Optional[BatchwiseUpdater] = None,
        kmeans_updater: Optional[BatchwiseUpdater] = None,
        writer_batch_size: int = 1024,
    ):
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.initializer = initializer if initializer is not None else KMeansPlusPlusInitializer(
            num_clusters=num_clusters, batch_size=batch_size
        )
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

