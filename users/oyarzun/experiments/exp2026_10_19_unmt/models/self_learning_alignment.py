"""
Self-Learning Cross-Lingual Embedding Alignment (Artetxe et al., ACL 2017)
Adapted for Unsupervised Neural Machine Translation (UNMT) with numeral initialization
(matching ^[0-9]+$ tokens across source and target vocabularies), orthogonal Procrustes
mapping, and nearest-neighbor / CSLS dictionary induction.
"""

from typing import Dict, Optional, Tuple, Union, Iterator
import os
import re
import json
import logging
import numpy as np

from sisyphus import Job, Task, tk


def normalize_and_center(embeddings: np.ndarray) -> np.ndarray:
    """
    Length-normalizes (unit L2 norm) and mean-centers an embedding matrix,
    followed by re-normalization to unit L2 length (Artetxe et al., 2017).
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

    # 3. Re-normalize to unit length for cosine similarity via dot product
    final_norms = np.linalg.norm(centered_emb, axis=1, keepdims=True)
    final_norms[final_norms == 0] = 1e-12
    return (centered_emb / final_norms).astype(np.float32)


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
    :param X: Source embeddings (V_src x d)
    :param Z: Target embeddings (V_trg x d)
    :param D: Alignment matrix (V_src x V_trg), sparse or dense binary/weighted dictionary
    :return: Orthogonal transformation matrix W* (d x d)
    """
    M = X.T @ (D @ Z)  # (d x d)
    U, _, Vt = np.linalg.svd(M)
    W = U @ Vt
    return W.astype(np.float32)


def compute_csls_similarity(
    X_mapped: np.ndarray,
    Z: np.ndarray,
    k: int = 10,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Cross-Domain Similarity Local Scaling (CSLS) (Conneau et al., 2017):
      CSLS(x, z) = 2 cos(x, z) - r_T(x) - r_S(z)
    in memory-efficient batches and returns (best_targets, best_sims).
    """
    K = X_mapped.shape[0]
    P = Z.shape[0]
    k_T = min(k, P)
    k_S = min(k, K)

    # Compute r_T(x) in batches and r_S(z) by transposing
    r_T = np.zeros(K, dtype=np.float32)
    for i in range(0, K, batch_size):
        sim_batch = X_mapped[i : i + batch_size] @ Z.T
        topk_T = np.partition(sim_batch, -k_T, axis=1)[:, -k_T:]
        r_T[i : i + batch_size] = np.mean(topk_T, axis=1)

    r_S = np.zeros(P, dtype=np.float32)
    for j in range(0, P, batch_size):
        sim_batch = Z[j : j + batch_size] @ X_mapped.T
        topk_S = np.partition(sim_batch, -k_S, axis=1)[:, -k_S:]
        r_S[j : j + batch_size] = np.mean(topk_S, axis=1)

    best_targets = np.zeros(K, dtype=np.int64)
    best_sims = np.zeros(K, dtype=np.float32)
    for i in range(0, K, batch_size):
        sim_batch = 2.0 * (X_mapped[i : i + batch_size] @ Z.T) - r_T[i : i + batch_size, None] - r_S[None, :]
        best_targets[i : i + batch_size] = np.argmax(sim_batch, axis=1)
        best_sims[i : i + batch_size] = np.max(sim_batch, axis=1)

    return best_targets, best_sims


def induce_dictionary_nn(
    X: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    retrieval_metric: str = "cosine",
    csls_k: int = 10,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Induces a new dictionary D via nearest neighbor retrieval using dot-product similarity
    or CSLS (Cross-Domain Similarity Local Scaling):
      j = argmax_k Sim(X_i W, Z_k)
    Returns (best_targets, best_sims) of shape (K,) each to avoid allocating a huge dense KxP matrix
    when K, P = 50,000.
    """
    K = X.shape[0]
    X_mapped = X @ W  # (K x d)

    if retrieval_metric == "csls":
        X_norm = X_mapped / np.maximum(np.linalg.norm(X_mapped, axis=1, keepdims=True), 1e-12)
        Z_norm = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        best_targets, best_sims = compute_csls_similarity(X_norm, Z_norm, k=csls_k, batch_size=batch_size)
    else:
        best_targets = np.zeros(K, dtype=np.int64)
        best_sims = np.zeros(K, dtype=np.float32)
        for i in range(0, K, batch_size):
            sim_batch = X_mapped[i : i + batch_size] @ Z.T
            best_targets[i : i + batch_size] = np.argmax(sim_batch, axis=1)
            best_sims[i : i + batch_size] = np.max(sim_batch, axis=1)

    return best_targets, best_sims


def compute_orthogonal_mapping_from_pairs(
    X: np.ndarray,
    Z: np.ndarray,
    src_indices: np.ndarray,
    trg_indices: np.ndarray,
) -> np.ndarray:
    """
    Memory-efficient orthogonal Procrustes SVD directly from paired indices (src_indices, trg_indices):
      M = sum_i X[src_indices[i]]^T Z[trg_indices[i]] = X[src_indices]^T @ Z[trg_indices]
      U, _, Vt = svd(M)
      W = U @ Vt
    """
    M = X[src_indices].T @ Z[trg_indices]  # (d x d)
    U, _, Vt = np.linalg.svd(M)
    W = U @ Vt
    return W.astype(np.float32)


def initialize_seed_dictionary(
    init_method: str,
    K: int,
    P: int,
    d: int,
    source_vocab: Optional[Dict[int, str]] = None,
    target_vocab: Optional[Dict[int, str]] = None,
    source_freqs: Optional[np.ndarray] = None,
    target_freqs: Optional[np.ndarray] = None,
    top_n_freq: Optional[int] = None,
    seed: int = 42,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Creates the initial state: either an initial mapping W0 or initial seed pairs (src_indices, trg_indices).
    Supports:
      - 'numerals': Matches identical numeral strings matching ^[0-9]+$ in source_vocab and target_vocab
                    (Artetxe et al., ACL 2017 / VecMap --init_numerals).
      - 'frequency': Aligns top-N most frequent source tokens with top-N most frequent target tokens.
      - 'random': Random orthogonal matrix via QR decomposition.
    """
    rng = np.random.RandomState(seed)

    if init_method == "numerals":
        if source_vocab and target_vocab:
            numeral_regex = re.compile(r"^[0-9]+$")
            src_numerals = {
                word: idx for idx, word in source_vocab.items()
                if idx < K and numeral_regex.match(word) is not None
            }
            trg_numerals = {
                word: idx for idx, word in target_vocab.items()
                if idx < P and numeral_regex.match(word) is not None
            }
            common_numerals = sorted(set(src_numerals.keys()).intersection(set(trg_numerals.keys())))
            if len(common_numerals) > 0:
                logging.info(f"Numeral initialization: found {len(common_numerals)} shared numerals in vocabularies.")
                src_idx = np.array([src_numerals[w] for w in common_numerals], dtype=np.int64)
                trg_idx = np.array([trg_numerals[w] for w in common_numerals], dtype=np.int64)
                return None, (src_idx, trg_idx)
            else:
                logging.warning("No shared numerals found in vocabularies; falling back to frequency initialization.")
        else:
            logging.warning("Vocabularies not provided for numeral initialization; falling back to frequency initialization.")
        init_method = "frequency"

    if init_method == "random":
        H = rng.randn(d, d)
        Q, R = np.linalg.qr(H)
        d_diag = np.diagonal(R)
        ph = d_diag / np.abs(d_diag)
        W0 = (Q * ph).astype(np.float32)
        return W0, None

    elif init_method == "frequency":
        if source_freqs is not None and target_freqs is not None:
            sorted_source = np.argsort(-source_freqs)
            sorted_target = np.argsort(-target_freqs)
        else:
            sorted_source = np.arange(K)
            sorted_target = np.arange(P)

        num_pairs = min(K, P)
        if top_n_freq is not None:
            num_pairs = min(num_pairs, top_n_freq)

        src_idx = sorted_source[:num_pairs].astype(np.int64)
        trg_idx = sorted_target[:num_pairs].astype(np.int64)
        return None, (src_idx, trg_idx)

    else:
        raise ValueError(f"Unknown initialization method: {init_method}")


def run_self_learning_alignment(
    X_raw: np.ndarray,
    Z_raw: np.ndarray,
    init_method: str = "numerals",
    source_vocab: Optional[Dict[int, str]] = None,
    target_vocab: Optional[Dict[int, str]] = None,
    top_n_freq: Optional[int] = None,
    source_freqs: Optional[np.ndarray] = None,
    target_freqs: Optional[np.ndarray] = None,
    max_iters: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
    retrieval_metric: str = "cosine",
    csls_k: int = 10,
) -> Dict:
    """
    Runs the full self-learning alignment loop (Artetxe et al., ACL 2017) with numeral initialization
    and optional CSLS retrieval.
    """
    K, d_x = X_raw.shape
    P, d_z = Z_raw.shape
    assert d_x == d_z, f"Dimension mismatch: {d_x} vs {d_z}"
    d = d_x

    # 1. Preprocessing: length-norm + mean-center + length-norm
    X = normalize_and_center(X_raw)
    Z = normalize_and_center(Z_raw)

    # 2. Initialization
    W0, seed_pairs = initialize_seed_dictionary(
        init_method=init_method,
        K=K,
        P=P,
        d=d,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        source_freqs=source_freqs,
        target_freqs=target_freqs,
        top_n_freq=top_n_freq,
        seed=seed,
    )

    all_src_indices = np.arange(K, dtype=np.int64)

    if W0 is not None:
        W = W0
        best_targets, sims = induce_dictionary_nn(X, W, Z, retrieval_metric=retrieval_metric, csls_k=csls_k)
        prev_avg_sim = float(np.mean(sims))
    else:
        src_idx, trg_idx = seed_pairs
        W = compute_orthogonal_mapping_from_pairs(X, Z, src_idx, trg_idx)
        best_targets, sims = induce_dictionary_nn(X, W, Z, retrieval_metric=retrieval_metric, csls_k=csls_k)
        prev_avg_sim = float(np.mean(sims))

    history = [float(prev_avg_sim)]
    converged = False
    iteration = 0

    for iteration in range(1, max_iters + 1):
        # Step 1: Orthogonal Procrustes mapping step
        W = compute_orthogonal_mapping_from_pairs(X, Z, all_src_indices, best_targets)

        # Step 2: Dictionary induction step
        best_targets, sims = induce_dictionary_nn(X, W, Z, retrieval_metric=retrieval_metric, csls_k=csls_k)
        curr_avg_sim = float(np.mean(sims))
        delta = curr_avg_sim - prev_avg_sim
        history.append(curr_avg_sim)

        logging.info(
            f"Iteration {iteration:03d} ({retrieval_metric}): avg_similarity = {curr_avg_sim:.6f}, delta = {delta:.8f}"
        )

        if abs(delta) < tol:
            converged = True
            break

        prev_avg_sim = curr_avg_sim

    mapped_X = (X @ W).astype(np.float32)

    return {
        "W": W,
        "mapped_X": mapped_X,
        "normalized_Z": Z,
        "src_to_trg_map": best_targets.tolist(),
        "cluster_to_phoneme_map": best_targets.tolist(),
        "final_avg_similarity": float(history[-1]),
        "iterations": iteration,
        "converged": converged,
        "objective_history": history,
        "retrieval_metric": retrieval_metric,
        "csls_k": csls_k,
    }


def load_vocab_dict(vocab_path: Optional[str]) -> Dict[int, str]:
    """
    Loads vocabulary from either a Python dict string file (e.g. {'word': 0, ...})
    or a standard whitespace/newline-separated token file.
    """
    if vocab_path is None or not os.path.exists(vocab_path):
        return {}

    with open(vocab_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("{") and content.endswith("}"):
        import ast
        try:
            vocab_dict = ast.literal_eval(content)
            return {int(idx): str(tok) for tok, idx in vocab_dict.items()}
        except Exception:
            pass

    res = {}
    for idx, line in enumerate(content.splitlines()):
        line = line.strip()
        if line:
            res[idx] = line.split()[0]
    return res


def write_vecmap_text_embeddings(filepath: str, embeddings: np.ndarray, vocab: Dict[int, str]):
    """
    Writes embeddings in standard Word2Vec / VecMap / UNdreaMT text format:
      <num_words> <dim>
      <word> <v_1> <v_2> ... <v_d>
    """
    V, d = embeddings.shape
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{V} {d}\n")
        for idx in range(V):
            word = vocab.get(idx, f"word_{idx}")
            vec_str = " ".join(f"{val:.6f}" for val in embeddings[idx])
            f.write(f"{word} {vec_str}\n")


class SelfLearningEmbeddingAlignmentJob(Job):
    """
    Sisyphus job executing the Artetxe et al. (ACL 2017) self-learning cross-lingual embedding alignment.
    Supports numeral initialization (matching ^[0-9]+$ tokens across source and target vocabularies),
    frequency initialization, and cosine/CSLS nearest-neighbor dictionary induction.
    """
    _version = 4

    def __init__(
        self,
        src_embeddings_npy: Optional[tk.Path] = None,
        trg_embeddings_npy: Optional[tk.Path] = None,
        src_vocab_file: Optional[tk.Path] = None,
        trg_vocab_file: Optional[tk.Path] = None,
        src_counts_file: Optional[tk.Path] = None,
        trg_counts_file: Optional[tk.Path] = None,
        init_method: str = "numerals",
        top_n_freq: Optional[int] = None,
        max_iters: int = 100,
        tol: float = 1e-6,
        seed: int = 42,
        retrieval_metric: str = "cosine",
        csls_k: int = 10,
        # Legacy aliases for backward compatibility
        audio_embeddings_npy: Optional[tk.Path] = None,
        text_embeddings_npy: Optional[tk.Path] = None,
        audio_vocab_file: Optional[tk.Path] = None,
        text_vocab_file: Optional[tk.Path] = None,
        audio_counts_file: Optional[tk.Path] = None,
        text_counts_file: Optional[tk.Path] = None,
        _version: Optional[int] = None,
    ):
        super().__init__()
        if _version is not None:
            self._version = _version

        self.src_embeddings_npy = src_embeddings_npy or audio_embeddings_npy
        self.trg_embeddings_npy = trg_embeddings_npy or text_embeddings_npy
        self.src_vocab_file = src_vocab_file or audio_vocab_file
        self.trg_vocab_file = trg_vocab_file or text_vocab_file
        self.src_counts_file = src_counts_file or audio_counts_file
        self.trg_counts_file = trg_counts_file or text_counts_file

        self.init_method = init_method
        self.top_n_freq = top_n_freq
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        self.retrieval_metric = retrieval_metric
        self.csls_k = csls_k

        # Outputs
        self.out_mapping_matrix_npy = self.output_path("W_star.npy")
        self.out_mapped_src_emb_npy = self.output_path("mapped_src_embeddings.npy")
        self.out_normalized_trg_emb_npy = self.output_path("normalized_trg_embeddings.npy")
        self.out_mapped_src_vec_txt = self.output_path("mapped_src_embeddings.vec")
        self.out_normalized_trg_vec_txt = self.output_path("normalized_trg_embeddings.vec")
        self.out_dictionary_json = self.output_path("src_to_trg_dict.json")
        self.out_alignment_report = self.output_path("alignment_report.json")

        # Backward-compatible aliases
        self.out_mapped_audio_emb_npy = self.out_mapped_src_emb_npy

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 2})

    def run(self):
        logging.basicConfig(level=logging.INFO)
        X_raw = np.load(self.src_embeddings_npy.get_path())
        Z_raw = np.load(self.trg_embeddings_npy.get_path())

        src_vocab_path = self.src_vocab_file.get_path() if self.src_vocab_file is not None else None
        trg_vocab_path = self.trg_vocab_file.get_path() if self.trg_vocab_file is not None else None
        src_vocab = load_vocab_dict(src_vocab_path)
        trg_vocab = load_vocab_dict(trg_vocab_path)

        source_freqs = None
        target_freqs = None

        if self.src_counts_file is not None and os.path.exists(self.src_counts_file.get_path()):
            try:
                with open(self.src_counts_file.get_path(), "r", encoding="utf-8") as f:
                    counts = [float(line.strip().split()[-1]) for line in f if line.strip()]
                    source_freqs = np.array(counts)
            except Exception as e:
                logging.warning(f"Could not load source counts: {e}")

        if self.trg_counts_file is not None and os.path.exists(self.trg_counts_file.get_path()):
            try:
                with open(self.trg_counts_file.get_path(), "r", encoding="utf-8") as f:
                    counts = [float(line.strip().split()[-1]) for line in f if line.strip()]
                    target_freqs = np.array(counts)
            except Exception as e:
                logging.warning(f"Could not load target counts: {e}")

        retrieval_metric = getattr(self, "retrieval_metric", "cosine")
        csls_k = getattr(self, "csls_k", 10)

        res = run_self_learning_alignment(
            X_raw=X_raw,
            Z_raw=Z_raw,
            init_method=self.init_method,
            source_vocab=src_vocab,
            target_vocab=trg_vocab,
            top_n_freq=self.top_n_freq,
            source_freqs=source_freqs,
            target_freqs=target_freqs,
            max_iters=self.max_iters,
            tol=self.tol,
            seed=self.seed,
            retrieval_metric=retrieval_metric,
            csls_k=csls_k,
        )

        np.save(self.out_mapping_matrix_npy.get_path(), res["W"])
        np.save(self.out_mapped_src_emb_npy.get_path(), res["mapped_X"])
        np.save(self.out_normalized_trg_emb_npy.get_path(), res["normalized_Z"])

        write_vecmap_text_embeddings(self.out_mapped_src_vec_txt.get_path(), res["mapped_X"], src_vocab)
        write_vecmap_text_embeddings(self.out_normalized_trg_vec_txt.get_path(), res["normalized_Z"], trg_vocab)

        mapping_named = {}
        for s_idx, t_idx in enumerate(res["src_to_trg_map"]):
            s_name = src_vocab.get(s_idx, f"src_{s_idx}")
            t_name = trg_vocab.get(t_idx, f"trg_{t_idx}")
            mapping_named[s_name] = t_name

        with open(self.out_dictionary_json.get_path(), "w", encoding="utf-8") as f:
            json.dump({
                "src_index_to_trg_index": res["src_to_trg_map"],
                "src_token_to_trg_token": mapping_named,
                "retrieval_metric": retrieval_metric,
                "csls_k": csls_k,
            }, f, indent=2, ensure_ascii=False)

        with open(self.out_alignment_report.get_path(), "w", encoding="utf-8") as f:
            json.dump({
                "init_method": self.init_method,
                "top_n_freq": self.top_n_freq,
                "retrieval_metric": retrieval_metric,
                "csls_k": csls_k,
                "converged": res["converged"],
                "iterations": res["iterations"],
                "final_avg_similarity": res["final_avg_similarity"],
                "objective_history": res["objective_history"],
            }, f, indent=2)
