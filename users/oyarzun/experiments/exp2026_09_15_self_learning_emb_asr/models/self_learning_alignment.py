"""
Self-Learning Bilingual/Bimodal Embedding Alignment (Artetxe et al., 2017)
Adapted for unsupervised / semi-supervised ASR aligning audio cluster embeddings with text phoneme embeddings.
"""

from typing import Dict, Optional, Tuple, Union, Iterator
import os
import json
import logging
import numpy as np

from sisyphus import Job, Task, tk


def normalize_and_center(embeddings: np.ndarray) -> np.ndarray:
    """
    Length-normalizes (unit L2 norm) and mean-centers an embedding matrix.
    :param embeddings: Matrix of shape (V, d)
    :return: Normalized and centered matrix of shape (V, d)
    """
    # 1. Length normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    norm_emb = embeddings / norms

    # 2. Mean centering
    mean = np.mean(norm_emb, axis=0, keepdims=True)
    centered_emb = norm_emb - mean

    # Re-normalize to unit length for cosine similarity via dot product
    final_norms = np.linalg.norm(centered_emb, axis=1, keepdims=True)
    final_norms[final_norms == 0] = 1e-12
    return centered_emb / final_norms


def compute_optimal_orthogonal_mapping(
    X: np.ndarray,
    Z: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """
    Solves the orthogonal Procrustes problem:
      W* = argmax_W Tr(X W Z^T D^T)  s.t.  W W^T = I
    using SVD:
      X^T D Z = U Sigma V^T  =>  W* = U V^T
    :param X: Source embeddings (K x d), e.g. Audio clusters
    :param Z: Target embeddings (P x d), e.g. Text phonemes
    :param D: Alignment matrix (K x P), binary or weighted dictionary
    :return: Orthogonal transformation matrix W* (d x d)
    """
    M = X.T @ D @ Z  # (d x d)
    U, _, Vt = np.linalg.svd(M)
    W = U @ Vt
    return W


def induce_dictionary_nn(
    X: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Induces a new dictionary D via nearest neighbor retrieval using dot-product similarity:
      j = argmax_k (X_i W) . Z_k
    :param X: Source embeddings (K x d)
    :param W: Transformation matrix (d x d)
    :param Z: Target embeddings (P x d)
    :return: (D, similarities) where D is binary (K x P) and similarities is (K,)
    """
    K = X.shape[0]
    P = Z.shape[0]

    # Transformed source embeddings
    X_mapped = X @ W  # (K x d)
    sim_matrix = X_mapped @ Z.T  # (K x P)

    best_targets = np.argmax(sim_matrix, axis=1)  # (K,)
    best_sims = np.max(sim_matrix, axis=1)  # (K,)

    D = np.zeros((K, P), dtype=np.float32)
    D[np.arange(K), best_targets] = 1.0

    return D, best_sims


def initialize_seed_dictionary(
    init_method: str,
    K: int,
    P: int,
    d: int,
    source_freqs: Optional[np.ndarray] = None,
    target_freqs: Optional[np.ndarray] = None,
    top_n_freq: Optional[int] = None,
    seed: int = 42,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Creates the initial state: either an initial mapping W0 or initial seed dictionary D0.
    """
    rng = np.random.RandomState(seed)

    if init_method == "random":
        # Random orthogonal matrix via QR decomposition
        H = rng.randn(d, d)
        Q, R = np.linalg.qr(H)
        d_diag = np.diagonal(R)
        ph = d_diag / np.abs(d_diag)
        W0 = Q * ph
        return W0, None

    elif init_method == "frequency":
        # Align top-N most frequent source cluster IDs with top-N most frequent target phonemes
        D0 = np.zeros((K, P), dtype=np.float32)
        if source_freqs is not None and target_freqs is not None:
            sorted_source = np.argsort(-source_freqs)
            sorted_target = np.argsort(-target_freqs)
        else:
            sorted_source = np.arange(K)
            sorted_target = np.arange(P)

        num_pairs = min(K, P)
        if top_n_freq is not None:
            num_pairs = min(num_pairs, top_n_freq)

        for i in range(num_pairs):
            s_idx = sorted_source[i]
            t_idx = sorted_target[i]
            D0[s_idx, t_idx] = 1.0

        return None, D0

    else:
        raise ValueError(f"Unknown initialization method: {init_method}")


def run_self_learning_alignment(
    X_raw: np.ndarray,
    Z_raw: np.ndarray,
    init_method: str = "frequency",
    top_n_freq: Optional[int] = None,
    source_freqs: Optional[np.ndarray] = None,
    target_freqs: Optional[np.ndarray] = None,
    max_iters: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
) -> Dict:
    """
    Runs the full self-learning alignment loop (Artetxe et al. 2017).
    """
    K, d_x = X_raw.shape
    P, d_z = Z_raw.shape
    assert d_x == d_z, f"Dimension mismatch: {d_x} vs {d_z}"
    d = d_x

    # 1. Preprocessing: length-norm + mean-center
    X = normalize_and_center(X_raw)
    Z = normalize_and_center(Z_raw)

    # 2. Initialization
    W0, D0 = initialize_seed_dictionary(
        init_method=init_method,
        K=K,
        P=P,
        d=d,
        source_freqs=source_freqs,
        target_freqs=target_freqs,
        top_n_freq=top_n_freq,
        seed=seed,
    )

    if W0 is not None:
        W = W0
        D, sims = induce_dictionary_nn(X, W, Z)
        prev_avg_sim = np.mean(sims)
    else:
        D = D0
        W = compute_optimal_orthogonal_mapping(X, Z, D)
        D, sims = induce_dictionary_nn(X, W, Z)
        prev_avg_sim = np.mean(sims)

    history = [float(prev_avg_sim)]
    converged = False
    iteration = 0

    for iteration in range(1, max_iters + 1):
        # Step 1: Mapping step
        W = compute_optimal_orthogonal_mapping(X, Z, D)

        # Step 2: Dictionary induction step
        D, sims = induce_dictionary_nn(X, W, Z)
        curr_avg_sim = float(np.mean(sims))
        delta = curr_avg_sim - prev_avg_sim
        history.append(curr_avg_sim)

        logging.info(
            f"Iteration {iteration:03d}: avg_similarity = {curr_avg_sim:.6f}, delta = {delta:.8f}"
        )

        if abs(delta) < tol:
            converged = True
            break

        prev_avg_sim = curr_avg_sim

    # Final mappings
    best_targets = np.argmax(D, axis=1).tolist()
    mapped_X = X @ W

    return {
        "W": W,
        "D": D,
        "mapped_X": mapped_X,
        "cluster_to_phoneme_map": best_targets,
        "final_avg_similarity": float(history[-1]),
        "iterations": iteration,
        "converged": converged,
        "objective_history": history,
    }


class SelfLearningEmbeddingAlignmentJob(Job):
    """
    Sisyphus job executing the Artetxe et al. (2017) self-learning embedding alignment.
    Aligns source audio cluster embeddings with target text phoneme embeddings.
    """

    def __init__(
        self,
        audio_embeddings_npy: tk.Path,
        text_embeddings_npy: tk.Path,
        audio_vocab_file: Optional[tk.Path] = None,
        text_vocab_file: Optional[tk.Path] = None,
        audio_counts_file: Optional[tk.Path] = None,
        text_counts_file: Optional[tk.Path] = None,
        init_method: str = "frequency",
        top_n_freq: Optional[int] = None,
        max_iters: int = 100,
        tol: float = 1e-6,
        seed: int = 42,
    ):
        super().__init__()
        self.audio_embeddings_npy = audio_embeddings_npy
        self.text_embeddings_npy = text_embeddings_npy
        self.audio_vocab_file = audio_vocab_file
        self.text_vocab_file = text_vocab_file
        self.audio_counts_file = audio_counts_file
        self.text_counts_file = text_counts_file
        self.init_method = init_method
        self.top_n_freq = top_n_freq
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed

        # Outputs
        self.out_mapping_matrix_npy = self.output_path("W_star.npy")
        self.out_mapped_audio_emb_npy = self.output_path("mapped_audio_embeddings.npy")
        self.out_dictionary_json = self.output_path("cluster_to_phoneme_dict.json")
        self.out_alignment_report = self.output_path("alignment_report.json")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        logging.basicConfig(level=logging.INFO)
        X_raw = np.load(self.audio_embeddings_npy.get_path())
        Z_raw = np.load(self.text_embeddings_npy.get_path())

        source_freqs = None
        target_freqs = None

        if self.audio_counts_file is not None and os.path.exists(self.audio_counts_file.get_path()):
            try:
                with open(self.audio_counts_file.get_path(), "r") as f:
                    counts = [float(line.strip().split()[-1]) for line in f if line.strip()]
                    source_freqs = np.array(counts)
            except Exception as e:
                logging.warning(f"Could not load audio counts: {e}")

        if self.text_counts_file is not None and os.path.exists(self.text_counts_file.get_path()):
            try:
                with open(self.text_counts_file.get_path(), "r") as f:
                    counts = [float(line.strip().split()[-1]) for line in f if line.strip()]
                    target_freqs = np.array(counts)
            except Exception as e:
                logging.warning(f"Could not load text counts: {e}")

        res = run_self_learning_alignment(
            X_raw=X_raw,
            Z_raw=Z_raw,
            init_method=self.init_method,
            top_n_freq=self.top_n_freq,
            source_freqs=source_freqs,
            target_freqs=target_freqs,
            max_iters=self.max_iters,
            tol=self.tol,
            seed=self.seed,
        )

        np.save(self.out_mapping_matrix_npy.get_path(), res["W"])
        np.save(self.out_mapped_audio_emb_npy.get_path(), res["mapped_X"])

        # Load vocabularies if provided to save human-readable dictionary
        audio_vocab = {}
        if self.audio_vocab_file is not None and os.path.exists(self.audio_vocab_file.get_path()):
            with open(self.audio_vocab_file.get_path(), "r") as f:
                for idx, line in enumerate(f):
                    tok = line.strip().split()[0]
                    audio_vocab[idx] = tok

        text_vocab = {}
        if self.text_vocab_file is not None and os.path.exists(self.text_vocab_file.get_path()):
            with open(self.text_vocab_file.get_path(), "r") as f:
                for idx, line in enumerate(f):
                    tok = line.strip().split()[0]
                    text_vocab[idx] = tok

        mapping_named = {}
        for s_idx, t_idx in enumerate(res["cluster_to_phoneme_map"]):
            s_name = audio_vocab.get(s_idx, f"cluster_{s_idx}")
            t_name = text_vocab.get(t_idx, f"phoneme_{t_idx}")
            mapping_named[s_name] = t_name

        with open(self.out_dictionary_json.get_path(), "w") as f:
            json.dump({
                "cluster_index_to_phoneme_index": res["cluster_to_phoneme_map"],
                "cluster_token_to_phoneme_token": mapping_named,
            }, f, indent=2)

        with open(self.out_alignment_report.get_path(), "w") as f:
            json.dump({
                "init_method": self.init_method,
                "top_n_freq": self.top_n_freq,
                "converged": res["converged"],
                "iterations": res["iterations"],
                "final_avg_similarity": res["final_avg_similarity"],
                "objective_history": res["objective_history"],
            }, f, indent=2)
