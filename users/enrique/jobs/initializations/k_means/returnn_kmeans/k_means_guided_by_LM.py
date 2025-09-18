__all__ = [
    "NnOutputClusteringCallback",
]

import sys
import os

from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import TensorDict

import numpy as np
from scipy.spatial.distance import cdist
import torch
from typing import Optional

from i6_experiments.users.enrique.jobs.initializations.k_means.models import KMeansModel, Wav2VecModel
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.recog_rasr_config import get_tree_timesync_recog_config
import json
import matplotlib.pyplot as plt


try:
    import faiss 
except ImportError:
    raise ImportError("faiss is not installed: Run 'pip install faiss-cpu'")

CENTROIDS_PATH = "centroids_and_scores.json"
VAL_LOSS_GRAPH_PATH = "validation_loss_graph.png"


PCA_A_PATH = "pca/A.pt"
PCA_B_PATH = "pca/b.pt"

DEFAULT_BATCH_SIZE = 1024

def print_report(batch, centroids, objective_value):
    entry = {
        "batch": batch,
        "centroids": centroids,
        "objective_value": objective_value,
    }
    # Try to read existing entries
    try:
        with open(CENTROIDS_PATH, "r") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            entries = [entries]
    except (FileNotFoundError, json.JSONDecodeError):
        entries = []
    entries.append(entry)
    # Save updated list
    with open(CENTROIDS_PATH, "w") as f:
        json.dump(entries, f, indent=2)
    abs_path = os.path.abspath(CENTROIDS_PATH)
    print(f"Appended centroids and scores to {abs_path}")

def plot_loss_graph(
    json_path: str = CENTROIDS_PATH, 
    output_filename: str = VAL_LOSS_GRAPH_PATH
):
    with open(json_path, "r") as f:
        entries = json.load(f)

    if not isinstance(entries, list) or not entries:
        print("No data available to plot.")
        return

    losses = [entry.get("objective_value") for entry in entries]
    batches = [entry.get("batch") for entry in entries]


    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(batches, losses, marker='o', linestyle='-', color='royalblue')

    # Add titles and labels for clarity
    plt.title("Validation Loss vs. Seen Batches")
    plt.xlabel("seen_batches")
    plt.ylabel("Loss on Valid Set")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the plot to the specified file
    plt.savefig(output_filename)
    print(f"Graph successfully saved as '{output_filename}'")


class PointsBufferPCA:
    def __init__(self, max_points: int = 10000, dtype=np.float32, out_dim: int = None, eigen_power: float = 0.0):
        self.max_points = max_points
        self.points = []
        self.dtype = dtype
        self.finish = False
        self.out_dim = out_dim  # target reduced dimension
        self.eigen_power = eigen_power
        self.pca_transform = None
        self.pca_bias = None

    def add(self, x):
        if self.finish:
            return

        if len(self.points) < self.max_points:
            self.points.append(x)
        else:
            self.pca()
            
    def pca(self):
        X = np.array(self.points, dtype=self.dtype)

        in_dim = X.shape[-1]
        out_dim = self.out_dim if self.out_dim is not None else in_dim

        print(f"Training PCA on {len(X)} points of dimension {in_dim} → {out_dim}")

        pca = faiss.PCAMatrix(in_dim, out_dim, self.eigen_power)
        pca.train(X)

        b = faiss.vector_to_array(pca.b)
        A = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)

        self.pca_bias = torch.from_numpy(b).float()
        self.pca_transform = torch.from_numpy(A).float()
        self.pca_out_dim = out_dim

        self.finish = True
        print("PCA training complete.")

        self._safe_pca_matrixes()
    
    def _safe_pca_matrixes(self):
        if self.pca_transform is None or self.pca_bias is None:
            print("PCA has not been computed. Skipping save operation.")
            return

        try:
            output_dir = os.path.dirname(PCA_A_PATH)
            os.makedirs(output_dir, exist_ok=True)
            print(f"Directory '{output_dir}' is ready.")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return

        print(f"Saving PCA transform matrix to {PCA_A_PATH}")
        torch.save(self.pca_transform, PCA_A_PATH)

        print(f"Saving PCA bias vector to {PCA_B_PATH}")
        torch.save(self.pca_bias, PCA_B_PATH)





class SequenceBatchwiseUpdater:
    # make this class pickable, so it can be used in the config
    __slots__ = ("batch_size", "data_index", "data_batch")

    def __init__(self, kmeans_model: KMeansModel, batch_size: int = DEFAULT_BATCH_SIZE):
        self.batch_size = batch_size
        self.data_index = 0
        self.data_batch = []
        self.sequences_lengths = []
        self.kmeans_model = kmeans_model

    # make this class picklable by defining __getstate__ and __setstate__
    def __getstate__(self):
        return {
            "batch_size": self.batch_size,
            "data_index": self.data_index,
            "data_batch": self.data_batch,
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

    # def _norm(self, X):
    #     if not self.l2_normalize:
    #         return X
    #     # avoid div by 0
    #     n = np.linalg.norm(X, axis=1, keepdims=True)
    #     return X / np.maximum(n, 1e-12)

    def on_batch_collected(self, data_batch: np.ndarray) -> bool:
        """
        Process a batch of data points to update centroids or perform other operations.
        This can include normalization or other preprocessing steps.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def process_sequence(self, seq: np.ndarray):
        seq_length = seq.shape[0]
        if not (self.batch_size - self.data_index) > seq_length:
            self.on_batch_collected(np.array(self.data_batch))
            self.reset()
 
        self.sequences_lengths.append(seq_length)
        self.data_index += seq_length
        for data_point in seq:
            self.data_batch.append(data_point)
    
    def reset(self):
        self.data_index = 0
        self.data_batch = []
        self.sequences_lengths = []

class EvalBatches(SequenceBatchwiseUpdater):
    def __init__(
            self,kmeans_model: KMeansModel, n_batches_size: int = 5, batch_size: int = DEFAULT_BATCH_SIZE,):
        super().__init__(batch_size=batch_size, kmeans_model=kmeans_model)
        self.n_batches_size = n_batches_size
        self.ready = False
        self.batches = []
    
    def loss(self, X):
        dist = cdist(X, self.kmeans_model.centroids)
        return float(np.mean(np.min(dist, axis=1)**2))

    def eval_and_report(self, batch_idx: int = None):
        lossses_sum = 0.0
        for i, batch in enumerate(self.batches):
            lossses_sum += self.loss(batch)

        avg_loss = lossses_sum / len(self.batches)
        print_report(batch_idx,self.kmeans_model.centroids.tolist(), avg_loss)

    def on_batch_collected(self, data_batch: np.ndarray) -> bool:
        self.batches.append(data_batch)
        if len(self.batches) >= self.n_batches_size:
            self.ready = True
            return True
        return False



        



class TraditionalBatchUpdater(SequenceBatchwiseUpdater):
    """ Traditional k-means updater: update centroids as the mean of assigned points every n batches"""
    def __init__(
            self, kmeans_model: KMeansModel, dim: int = None, batch_size: int = DEFAULT_BATCH_SIZE):
        super().__init__(kmeans_model=kmeans_model, batch_size=batch_size)
        self.seen_batches = 0
        self.update_every_n_batches = 200
        self.objective_value = float('inf')

        self.centroid_sum = torch.zeros((kmeans_model.num_clusters, dim))


    def on_batch_collected(self, data_batch: np.ndarray) -> bool:
        """
        Update centroids as the mean of the points assigned to it (traditional k-means).
        """
        self.seen_batches += 1
        assert self.kmeans_model.centroids is not None, "Centroids must be initialized before updating."

        X = data_batch
        # Assign each point to the closest centroid
        dist = cdist(X, self.kmeans_model.centroids)
        closest_centroid_indices = np.argmin(dist, axis=1)

        # Update centroids
        new_centroids_for_batch = self.kmeans_model.centroids.clone().numpy()
        for cidx in range(self.kmeans_model.num_clusters):
            # Find all points assigned to this centroid
            mask = (closest_centroid_indices == cidx)
            points_in_cluster = X[mask]

            # If the cluster is not empty, update the centroid to the mean of its points
            if points_in_cluster.shape[0] > 0:
                new_centroids_for_batch[cidx, :] = torch.from_numpy(points_in_cluster.mean(axis=0))

        self.centroid_sum += torch.from_numpy(new_centroids_for_batch)

        if self.seen_batches > 0 and self.seen_batches % self.update_every_n_batches == 0:
            # Calculate the average
            averaged_centroids = self.centroid_sum / self.update_every_n_batches
            
            # Update the model's centroids
            self.kmeans_model.update_centroids(averaged_centroids.clone())
            print(f"Updated centroids after batch {self.seen_batches} by averaging over the last {self.update_every_n_batches} batches.")

            # Reset the sum for the next cycle
            self.centroid_sum.zero_()

        return self.seen_batches % self.update_every_n_batches == 0

    # def save_centroids_and_scores(self):
    #     # Prepare entry to save
    #     entry = {
    #         "centroids": self.kmeans_model.centroids.tolist(),
    #         "objective_value": self.objective_value,
    #     }
    #     # Try to read existing entries
    #     try:
    #         with open(CENTROIDS_PATH, "r") as f:
    #             entries = json.load(f)
    #         if not isinstance(entries, list):
    #             entries = [entries]
    #     except (FileNotFoundError, json.JSONDecodeError):
    #         entries = []
    #     entries.append(entry)
    #     # Save updated list
    #     with open(CENTROIDS_PATH, "w") as f:
    #         json.dump(entries, f, indent=2)
    #     abs_path = os.path.abspath(CENTROIDS_PATH)
    #     print(f"Appended centroids and scores to {abs_path}")



class EMAUpdater(SequenceBatchwiseUpdater):
    def __init__(
            self, kmeans_model: KMeansModel, batch_size: int = DEFAULT_BATCH_SIZE,
            eta: float = 0.08):
        super().__init__(kmeans_model=kmeans_model, batch_size=batch_size)
        self.seen_batches = 0
        self.eta = eta
    
    def on_batch_collected(self, data_batch: np.ndarray) -> bool:
        """
        Update centroids using exponential moving average based on the current batch of data points.
        """
        self.seen_batches += 1
        assert self.kmeans_model.centroids is not None, "Centroids must be initialized before updating."
        X = data_batch
        dist = cdist(X, self.kmeans_model.centroids)
        closest = np.argmin(dist, axis=1)

        for cidx in range(self.kmeans_model.num_clusters):
            mask = closest == cidx
            count = int(mask.sum())
            if count == 0:
                # If no points are assigned to this centroid, keep the old centroid
                continue
            # update the centroid for this cluster
            # sum of new cluster points
            batch_mean = X[mask].mean(axis=0)

            # weighted update of the cluster centroid
            self.kmeans_model.centroids[cidx, :] = self.kmeans_model.centroids[cidx, :] * (1 - self.eta) + batch_mean * self.eta
        
        # compute value of k-means objective
        dist_new = cdist(X, self.kmeans_model.centroids)
        kmeans_objective = float(np.mean(np.min(dist_new, axis=1)**2))
        self.objective_value = kmeans_objective
        
        # if self.seen_batches % self.print_report_every_n_batches == 1:
        #     self.save_centroids_and_scores()

        return dist_new, kmeans_objective

    # def save_centroids_and_scores(self):
    #     # Prepare entry to save
    #     entry = {
    #         "centroids": self.kmeans_model.centroids.tolist(),
    #         "objective_value": self.objective_value,
    #     }
    #     # Try to read existing entries
    #     try:
    #         with open(CENTROIDS_PATH, "r") as f:
    #             entries = json.load(f)
    #         if not isinstance(entries, list):
    #             entries = [entries]
    #     except (FileNotFoundError, json.JSONDecodeError):
    #         entries = []
    #     entries.append(entry)
    #     # Save updated list
    #     with open(CENTROIDS_PATH, "w") as f:
    #         json.dump(entries, f, indent=2)
    #     abs_path = os.path.abspath(CENTROIDS_PATH)
    #     print(f"Appended centroids and scores to {abs_path}")








class KMeansPPInitializer:
    def __init__(self, num_clusters: int, max_points: int = 10000, rng: np.random.Generator = None):
        self.num_clusters = num_clusters
        self.max_points = max_points
        self.data_points = []
        self.centroids = []
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def add_point(self, point: np.ndarray) -> bool:
        """
        Add a data point. Returns True if enough points collected.
        Checks that all points have the same dimension.
        """



        if len(self.data_points) > 0:
            expected_dim = self.data_points[0].shape
            if point.shape != expected_dim:
                print(f"Dimension mismatch detected!")
                print(f"Expected dimension: {expected_dim}")
                print(f"Received point dimension: {point.shape}")
                print(f"Problematic point: {point}")
                print(f"All collected points so far:")
                for idx, p in enumerate(self.data_points):
                    print(f"Index {idx}: shape {p.shape}, point {p}")
                raise ValueError(
                    f"Dimension mismatch in KMeansPPInitializer: expected {expected_dim}, got {point.shape}. "
                    f"Object info: num_clusters={self.num_clusters}, max_points={self.max_points}, "
                    f"current_points={len(self.data_points)}, rng={self.rng}"
                )
            

        if len(self.data_points) < self.max_points:
            self.data_points.append(point)
        
        return len(self.data_points) < self.max_points

    def initialize_centroids(self) -> np.ndarray:
        """
        Apply K-Means++ initialization on collected points.
        """
        if len(self.data_points) < self.num_clusters:
            raise ValueError(f"Need at least {self.num_clusters} points, got {len(self.data_points)}")
        
        X = np.array(self.data_points)
        
        # Step 1: Choose first centroid uniformly at random
        idx = self.rng.randint(0, X.shape[0])
        self.centroids = [X[idx]]
        
        # Step 2: Choose remaining centroids using K-Means++
        for _ in range(1, self.num_clusters):
            # Compute squared distances to nearest centroid
            distances = cdist(X, np.array(self.centroids))
            min_distances = np.min(distances, axis=1) ** 2
            
            # Convert to probabilities
            probabilities = min_distances / np.sum(min_distances)
            
            # Choose next centroid
            next_idx = self.rng.choices(range(X.shape[0]), weights=probabilities, k=1)[0]
            self.centroids.append(X[next_idx])
        
        return np.array(self.centroids)
    def reset(self):
        """Clear collected data."""
        self.data_points = []
        self.centroids = []




class NnOutputClusteringCallback(ForwardCallbackIface):
    def __init__(self):
        self.seen_sequences = 0
        self.seen_points = 0

        pass

    def init(self, *, model: Optional[torch.nn.Module] = None):
        self.kmeans_model = model.kmeans_model
        self.kmeans_model: KMeansModel
        self.wav2vec_model = model.wav2vec_model
        self.wav2vec_model: Wav2VecModel
        self.training_lm = model.wav2vec_model.train_language_model
        self.rng = model.rng
        self.kmeans_initializer = KMeansPPInitializer(num_clusters=self.kmeans_model.num_clusters, rng=self.rng)
        self.eval_every_n_batches = 500
        self.eval_batches = EvalBatches(model.kmeans_model, n_batches_size=10)

        # PCA stuff
        if (
            self.wav2vec_model.n_points_to_calculate_pca
            and self.wav2vec_model.pca_enabled
            and not self.wav2vec_model.pca_ready
        ):
            self.points_buffer_pca = PointsBufferPCA(
                max_points=self.wav2vec_model.n_points_to_calculate_pca,
                out_dim=self.wav2vec_model.pca_dim,
                eigen_power=0.5)
            
        self.centroid_updater = TraditionalBatchUpdater(self.kmeans_model, dim=self.wav2vec_model.pca_dim)

    def _write_buffer_to_file(self):
        pass

    def process_seq_for_calculating_pca_matrixes(self, *, seq_tag: str, outputs: TensorDict):
        points = outputs["output"].raw_tensor
        for x in points:
            self.points_buffer_pca.add(x)

    def process_seq_for_kmeans(self, *, seq_tag: str, outputs: TensorDict):
        points = outputs["output"].raw_tensor
    
        if not self.kmeans_model.is_initialized:
            initializer_is_full = False
            for x in points:
                if not self.kmeans_initializer.add_point(x):
                    initializer_is_full = True
                    break
            if initializer_is_full:
                self.kmeans_model.update_centroids(torch.from_numpy(self.kmeans_initializer.initialize_centroids())) 
            return
        
        if not self.eval_batches.ready:
            self.eval_batches.process_sequence(points)
            return

        # If kmeans model is initialized and pca in wav2vec model is ready, train kmeans:
        self.centroid_updater.process_sequence(points)

        if self.eval_every_n_batches > 0 and self.centroid_updater.seen_batches % self.eval_every_n_batches == 0:
            self.eval_batches.eval_and_report(batch_idx=self.centroid_updater.seen_batches)

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        self.seen_sequences += 1
        self.seen_points += outputs["output"].raw_tensor.shape[0]

        if self.wav2vec_model.pca_enabled:
            if not self.wav2vec_model.pca_ready:
                self.process_seq_for_calculating_pca_matrixes(seq_tag=seq_tag, outputs=outputs)
                
                if self.points_buffer_pca.finish:
                    # finalize PCA and set the model's PCA parameters
                    self.wav2vec_model.pca_transform = self.points_buffer_pca.pca_transform
                    self.wav2vec_model.pca_bias = self.points_buffer_pca.pca_bias
                    self.wav2vec_model.pca_ready = True
                    print("PCA is now ready and set in the wav2vec2 model.")
                    print(f"PCA transform shape: {self.wav2vec_model.pca_transform.shape}")
                    print(f"PCA bias shape: {self.wav2vec_model.pca_bias.shape}")
                
                return

        if outputs["output"].dims[1].size == self.wav2vec_model.pca_dim:
            self.process_seq_for_kmeans(seq_tag=seq_tag, outputs=outputs)


    def finish(self):
        plot_loss_graph()
        pass
