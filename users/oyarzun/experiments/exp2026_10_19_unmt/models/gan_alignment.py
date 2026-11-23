"""
Domain-Adversarial (GAN) Cross-Modal / Cross-Lingual Embedding Alignment (Chung et al., 2018; Conneau et al., 2018)
Adapted for Unsupervised Neural Machine Translation (UNMT) and Cross-Modal Phoneme Alignment (e.g. EN+Blank <-> EN/DE).

Reference:
  Chung et al. (2018), "Unsupervised Cross-Modal Alignment of Speech and Text Embedding Spaces" (arXiv:1805.07467)
  - Discriminator: Two-layer neural network of size 512 with ReLU activation function.
  - Optimization: Both Discriminator and linear mapping W trained by SGD with a fixed learning rate of 1e-3.
  - Refinement: Mutual nearest neighbors / CSLS (Conneau et al., 2018) to build synthetic parallel dictionary,
    followed by orthogonal Procrustes closed-form SVD update (W* = U V^T).
"""

from typing import Dict, List, Optional, Tuple, Iterator
import os
import json
import logging
import numpy as np

from sisyphus import Job, Task, tk
from .self_learning_alignment import (
    normalize_and_center,
     load_vocab_dict,
    write_vecmap_text_embeddings,
)


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
    Supports both 2D alignment matrix D (K x P) and 1D index array D (K,) where D[i] >= 0 is the target index.
    """
    if D.ndim == 1:
        valid = D >= 0
        if not np.any(valid):
            return np.eye(X.shape[1], dtype=np.float32)
        M = X[valid].T @ Z[D[valid]]
    else:
        M = X.T @ (D @ Z)
    U, _, Vt = np.linalg.svd(M.astype(np.float64), full_matrices=False)
    W = (U @ Vt).astype(np.float32)
    return W


def compute_csls_similarity_matrix(
    X_mapped: np.ndarray,
    Z: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """
    Computes Cross-Domain Similarity Local Scaling (CSLS) similarity matrix (K x P):
      CSLS(x, z) = 2 cos(x, z) - r_T(x) - r_S(z)
    """
    K = X_mapped.shape[0]
    P = Z.shape[0]
    X_norm = (X_mapped / np.maximum(np.linalg.norm(X_mapped, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    Z_norm = (Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)).astype(np.float32)

    sim_matrix = X_norm @ Z_norm.T  # (K x P)
    k_T = min(k, P)
    k_S = min(k, K)

    if k_T > 0:
        topk_T = np.partition(sim_matrix, -k_T, axis=1)[:, -k_T:]
        r_T = np.mean(topk_T, axis=1, keepdims=True)
    else:
        r_T = np.zeros((K, 1), dtype=np.float32)

    if k_S > 0:
        topk_S = np.partition(sim_matrix, -k_S, axis=0)[-k_S:, :]
        r_S = np.mean(topk_S, axis=0, keepdims=True)
    else:
        r_S = np.zeros((1, P), dtype=np.float32)

    return (2.0 * sim_matrix - r_T - r_S).astype(np.float32)


def compute_nn_and_col_argmax(
    X_mapped: np.ndarray,
    Z: np.ndarray,
    retrieval_metric: str = "cosine",
    csls_k: int = 10,
    need_col_argmax: bool = False,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Memory-efficient nearest-neighbor retrieval (cosine or CSLS) in row chunks.
    """
    K = X_mapped.shape[0]
    P = Z.shape[0]

    if retrieval_metric == "csls":
        X_use = (X_mapped / np.maximum(np.linalg.norm(X_mapped, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        Z_use = (Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        k_T = min(csls_k, P)
        k_S = min(csls_k, K)

        r_T = np.zeros(K, dtype=np.float32)
        running_top_S: Optional[np.ndarray] = None

        for start in range(0, K, chunk_size):
            end = min(start + chunk_size, K)
            sim_chunk = X_use[start:end] @ Z_use.T
            if k_T > 0:
                topk_T = np.partition(sim_chunk, -k_T, axis=1)[:, -k_T:]
                r_T[start:end] = np.mean(topk_T, axis=1)
            if k_S > 0:
                b_len = end - start
                k_chunk = min(k_S, b_len)
                chunk_top = np.partition(sim_chunk, -k_chunk, axis=0)[-k_chunk:, :]
                if running_top_S is None:
                    running_top_S = chunk_top.copy()
                else:
                    combined = np.concatenate([running_top_S, chunk_top], axis=0)
                    if combined.shape[0] > k_S:
                        running_top_S = np.partition(combined, -k_S, axis=0)[-k_S:, :].copy()
                    else:
                        running_top_S = combined

        r_S = np.mean(running_top_S, axis=0, keepdims=True) if (k_S > 0 and running_top_S is not None) else np.zeros((1, P), dtype=np.float32)
    else:
        X_use = X_mapped.astype(np.float32, copy=False)
        Z_use = Z.astype(np.float32, copy=False)
        r_T = None
        r_S = None

    best_t_for_s = np.empty(K, dtype=np.int64)
    best_sim_for_s = np.empty(K, dtype=np.float32)
    best_s_for_t = np.zeros(P, dtype=np.int64) if need_col_argmax else None
    best_col_sim = np.full(P, -np.inf, dtype=np.float32) if need_col_argmax else None

    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        sim_chunk = X_use[start:end] @ Z_use.T
        if retrieval_metric == "csls":
            sim_chunk = 2.0 * sim_chunk - r_T[start:end, None] - r_S

        row_best_t = np.argmax(sim_chunk, axis=1)
        best_t_for_s[start:end] = row_best_t
        best_sim_for_s[start:end] = sim_chunk[np.arange(end - start), row_best_t]

        if need_col_argmax:
            col_argmax_local = np.argmax(sim_chunk, axis=0)
            col_max_local = sim_chunk[col_argmax_local, np.arange(P)]
            better = col_max_local > best_col_sim
            best_col_sim[better] = col_max_local[better]
            best_s_for_t[better] = start + col_argmax_local[better]

    return best_t_for_s, best_sim_for_s, best_s_for_t


def induce_dictionary_nn(
    X: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    retrieval_metric: str = "csls",
    csls_k: int = 10,
    return_dense: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Induces a dictionary D via nearest-neighbor retrieval using cosine or CSLS similarity.
    """
    K = X.shape[0]
    P = Z.shape[0]
    X_mapped = X @ W
    best_targets, best_sims, _ = compute_nn_and_col_argmax(
        X_mapped,
        Z,
        retrieval_metric=retrieval_metric,
        csls_k=csls_k,
        need_col_argmax=False,
    )
    if not return_dense:
        return best_targets, best_sims
    D = np.zeros((K, P), dtype=np.float32)
    D[np.arange(K), best_targets] = 1.0
    return D, best_sims


def induce_mutual_nn_dictionary(
    X: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    retrieval_metric: str = "csls",
    csls_k: int = 10,
    source_freqs: Optional[np.ndarray] = None,
    target_freqs: Optional[np.ndarray] = None,
    top_freq_ratio: float = 1.0,
    return_dense: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Builds a synthetic parallel dictionary retaining mutual nearest neighbors (MNN)
    under CSLS or cosine similarity.
    """
    K = X.shape[0]
    P = Z.shape[0]
    X_mapped = X @ W

    best_t_for_s, _, best_s_for_t = compute_nn_and_col_argmax(
        X_mapped,
        Z,
        retrieval_metric=retrieval_metric,
        csls_k=csls_k,
        need_col_argmax=True,
    )

    valid_s = np.ones(K, dtype=bool)
    valid_t = np.ones(P, dtype=bool)
    if top_freq_ratio < 1.0:
        if source_freqs is not None and len(source_freqs) == K:
            n_s = max(10, int(K * top_freq_ratio))
            top_s = np.argsort(-source_freqs)[:n_s]
            valid_s[:] = False
            valid_s[top_s] = True
        if target_freqs is not None and len(target_freqs) == P:
            n_t = max(10, int(P * top_freq_ratio))
            top_t = np.argsort(-target_freqs)[:n_t]
            valid_t[:] = False
            valid_t[top_t] = True

    if not return_dense:
        D_mnn_1d = np.full(K, -1, dtype=np.int64)
        mnn_count = 0
        for s_idx in range(K):
            if not valid_s[s_idx]:
                continue
            t_idx = int(best_t_for_s[s_idx])
            if not valid_t[t_idx]:
                continue
            if int(best_s_for_t[t_idx]) == s_idx:
                D_mnn_1d[s_idx] = t_idx
                mnn_count += 1
        return D_mnn_1d, mnn_count

    D_mnn = np.zeros((K, P), dtype=np.float32)
    mnn_count = 0
    for s_idx in range(K):
        if not valid_s[s_idx]:
            continue
        t_idx = int(best_t_for_s[s_idx])
        if not valid_t[t_idx]:
            continue
        if int(best_s_for_t[t_idx]) == s_idx:
            D_mnn[s_idx, t_idx] = 1.0
            mnn_count += 1
    return D_mnn, mnn_count


def plot_gan_training_curves(
    history: Dict,
    out_png_path: str,
    title: str = "GAN Cross-Modal Alignment Training & CSLS Refinement",
):
    """
    Plots Discriminator / Generator adversarial loss curves and the unsupervised CSLS
    validation criterion across adversarial epochs and Procrustes refinement iterations.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["disc_loss"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200)

    ax1.plot(epochs, history["disc_loss"], label="Discriminator Loss ($\\mathcal{L}_D$)", color="#d9534f", linewidth=1.8)
    ax1.plot(epochs, history["gen_loss"], label="Generator / Mapping Loss ($\\mathcal{L}_W$)", color="#2b5c8f", linewidth=1.8)
    ax1.axhline(np.log(2.0), color="gray", linestyle="--", alpha=0.6, label="Nash Equilibrium ($\\ln 2 \\approx 0.693$)")
    ax1.set_xlabel("Adversarial Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=10, fontweight="bold")
    ax1.set_title("Domain-Adversarial Training Losses", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="best", fontsize=9)

    ax2.plot(epochs, history["adv_csls_criterion"], label="Adversarial Mean CSLS", color="#2b5c8f", linewidth=1.8)
    best_ep = history.get("best_adv_epoch", 1)
    best_adv_val = history.get("best_adv_csls", 0.0)
    ax2.scatter([best_ep], [best_adv_val], color="#d95f02", s=60, zorder=5, label=f"Best Adv Epoch ({best_ep}: {best_adv_val:.4f})")

    if history.get("refinement_history"):
        ref_vals = history["refinement_history"]
        ax2.axhline(ref_vals[-1], color="#1b9e77", linestyle="-.", linewidth=1.8, label=f"Post-Refinement CSLS ({ref_vals[-1]:.4f})")

    ax2.set_xlabel("Adversarial Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean CSLS Similarity", fontsize=10, fontweight="bold")
    ax2.set_title("Unsupervised Alignment Criterion (CSLS)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)


def run_gan_alignment(
    X_raw: np.ndarray,
    Z_raw: np.ndarray,
    source_freqs: Optional[np.ndarray] = None,
    target_freqs: Optional[np.ndarray] = None,
    disc_hidden_dim: int = 512,
    disc_num_layers: int = 2,
    disc_input_dropout: float = 0.1,
    disc_hidden_dropout: float = 0.0,
    label_smoothing: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 32,
    n_epochs: int = 50,
    epoch_size: int = 1000,
    dis_steps: int = 5,
    ortho_beta: float = 0.01,
    most_frequent_sampling_ratio: float = 1.0,
    refinement_iters: int = 50,
    refinement_tol: float = 1e-6,
    use_mutual_nn_refinement: bool = True,
    retrieval_metric: str = "csls",
    csls_k: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Executes Domain-Adversarial Training (GAN) followed by CSLS Procrustes Refinement
    following Chung et al. (2018) (arXiv:1805.07467) and MUSE (Conneau et al., 2018).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    K, d_x = X_raw.shape
    P, d_z = Z_raw.shape
    assert d_x == d_z, f"Embedding dimension mismatch: src={d_x} vs trg={d_z}"
    d = d_x

    # 1. Preprocessing: length-norm + mean-center + unit norm
    X = normalize_and_center(X_raw).astype(np.float32)
    Z = normalize_and_center(Z_raw).astype(np.float32)

    X_tensor = torch.from_numpy(X).to(device)
    Z_tensor = torch.from_numpy(Z).to(device)

    if most_frequent_sampling_ratio < 1.0 and source_freqs is not None and len(source_freqs) == K:
        n_pool_s = max(min(K, batch_size), int(K * most_frequent_sampling_ratio))
        pool_s = np.argsort(-source_freqs)[:n_pool_s]
    else:
        pool_s = np.arange(K)

    if most_frequent_sampling_ratio < 1.0 and target_freqs is not None and len(target_freqs) == P:
        n_pool_t = max(min(P, batch_size), int(P * most_frequent_sampling_ratio))
        pool_t = np.argsort(-target_freqs)[:n_pool_t]
    else:
        pool_t = np.arange(P)

    # 2. Define Discriminator (2-layer NN of size 512 with ReLU activations)
    layers = [nn.Dropout(disc_input_dropout)]
    in_dim = d
    for _ in range(disc_num_layers):
        layers.append(nn.Linear(in_dim, disc_hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if disc_hidden_dropout > 0.0:
            layers.append(nn.Dropout(disc_hidden_dropout))
        in_dim = disc_hidden_dim
    layers.append(nn.Linear(in_dim, 1))
    discriminator = nn.Sequential(*layers).to(device)

    # 3. Define Mapping W (d x d linear transformation without bias, initialized to Identity)
    mapping = nn.Linear(d, d, bias=False).to(device)
    with torch.no_grad():
        mapping.weight.copy_(torch.eye(d, dtype=torch.float32, device=device))

    opt_disc = torch.optim.SGD(discriminator.parameters(), lr=lr)
    opt_map = torch.optim.SGD(mapping.parameters(), lr=lr)

    def _get_W_numpy() -> np.ndarray:
        return mapping.weight.detach().cpu().numpy().T.copy()

    def _orthogonalize_mapping():
        if ortho_beta > 0.0:
            with torch.no_grad():
                W_mat = mapping.weight.data
                mapping.weight.data.copy_(
                    (1.0 + ortho_beta) * W_mat - ortho_beta * W_mat.mm(W_mat.transpose(0, 1)).mm(W_mat)
                )

    W_init = _get_W_numpy()
    _, init_sims = induce_dictionary_nn(
        X, W_init, Z,
        retrieval_metric=retrieval_metric,
        csls_k=csls_k,
        return_dense=False,
    )
    best_adv_csls = float(np.mean(init_sims))
    best_W_adv = W_init.copy()
    best_adv_epoch = 0

    disc_loss_history = []
    gen_loss_history = []
    adv_csls_history = []

    steps_per_epoch = max(1, epoch_size // batch_size)
    logging.info(
        f"Starting GAN Cross-Modal Alignment on {device}: K={K}, P={P}, d={d}, "
        f"epochs={n_epochs}, steps/epoch={steps_per_epoch}, dis_steps={dis_steps}, lr={lr}, "
        f"disc={disc_num_layers}x{disc_hidden_dim} ReLU"
    )

    for epoch in range(1, n_epochs + 1):
        epoch_d_losses = []
        epoch_g_losses = []

        for _ in range(steps_per_epoch):
            # (a) Discriminator steps
            discriminator.train()
            mapping.eval()
            d_step_loss = 0.0
            for _ in range(dis_steps):
                s_idx = rng.choice(pool_s, size=batch_size, replace=(len(pool_s) < batch_size))
                t_idx = rng.choice(pool_t, size=batch_size, replace=(len(pool_t) < batch_size))

                with torch.no_grad():
                    s_mapped = mapping(X_tensor[s_idx])
                t_emb = Z_tensor[t_idx]

                disc_in = torch.cat([s_mapped, t_emb], dim=0)
                y_disc = torch.empty(2 * batch_size, 1, dtype=torch.float32, device=device)
                y_disc[:batch_size] = 1.0 - label_smoothing
                y_disc[batch_size:] = label_smoothing

                opt_disc.zero_grad()
                logits = discriminator(disc_in)
                loss_d = F.binary_cross_entropy_with_logits(logits, y_disc)
                loss_d.backward()
                opt_disc.step()
                d_step_loss += float(loss_d.item())

            epoch_d_losses.append(d_step_loss / dis_steps)

            # (b) Generator / Mapping W step
            discriminator.eval()
            mapping.train()

            s_idx = rng.choice(pool_s, size=batch_size, replace=(len(pool_s) < batch_size))
            t_idx = rng.choice(pool_t, size=batch_size, replace=(len(pool_t) < batch_size))

            s_mapped = mapping(X_tensor[s_idx])
            t_emb = Z_tensor[t_idx]
            gen_in = torch.cat([s_mapped, t_emb], dim=0)

            y_gen = torch.empty(2 * batch_size, 1, dtype=torch.float32, device=device)
            y_gen[:batch_size] = label_smoothing
            y_gen[batch_size:] = 1.0 - label_smoothing

            opt_map.zero_grad()
            logits_g = discriminator(gen_in)
            loss_g = F.binary_cross_entropy_with_logits(logits_g, y_gen)
            loss_g.backward()
            opt_map.step()

            _orthogonalize_mapping()
            epoch_g_losses.append(float(loss_g.item()))

        mean_d_loss = float(np.mean(epoch_d_losses))
        mean_g_loss = float(np.mean(epoch_g_losses))
        disc_loss_history.append(mean_d_loss)
        gen_loss_history.append(mean_g_loss)

        W_curr = _get_W_numpy()
        _, sims_curr = induce_dictionary_nn(
            X, W_curr, Z,
            retrieval_metric=retrieval_metric,
            csls_k=csls_k,
            return_dense=False,
        )
        curr_csls = float(np.mean(sims_curr))
        adv_csls_history.append(curr_csls)

        if curr_csls > best_adv_csls:
            best_adv_csls = curr_csls
            best_W_adv = W_curr.copy()
            best_adv_epoch = epoch

        if epoch % 5 == 0 or epoch == 1 or epoch == n_epochs:
            logging.info(
                f"GAN Epoch {epoch:03d}/{n_epochs:03d} | L_D: {mean_d_loss:.4f} | L_W: {mean_g_loss:.4f} | "
                f"Mean {retrieval_metric.upper()}: {curr_csls:.6f} (Best: {best_adv_csls:.6f} @ ep {best_adv_epoch})"
            )

    # Project best adversarial W onto the exact orthogonal manifold O(d) via SVD before refinement
    U_0, _, Vt_0 = np.linalg.svd(best_W_adv.astype(np.float64), full_matrices=False)
    W = (U_0 @ Vt_0).astype(np.float32)

    # 4. Refinement Procedure (Mutual Nearest Neighbors + Procrustes SVD with CSLS)
    D, sims = induce_dictionary_nn(
        X, W, Z,
        retrieval_metric=retrieval_metric,
        csls_k=csls_k,
        return_dense=False,
    )
    prev_avg_sim = float(np.mean(sims))
    refinement_history = [prev_avg_sim]
    mnn_counts_history = []
    converged = False
    ref_iter = 0

    for ref_iter in range(1, refinement_iters + 1):
        if use_mutual_nn_refinement:
            D_mnn, mnn_cnt = induce_mutual_nn_dictionary(
                X, W, Z,
                retrieval_metric=retrieval_metric,
                csls_k=csls_k,
                source_freqs=source_freqs,
                target_freqs=target_freqs,
                return_dense=False,
            )
            mnn_counts_history.append(int(mnn_cnt))
            D_procrustes = D_mnn if mnn_cnt >= min(25, max(2, min(K, P) // 4)) else D
        else:
            D_procrustes = D

        W_cand = compute_optimal_orthogonal_mapping(X, Z, D_procrustes)
        D_cand, sims_cand = induce_dictionary_nn(
            X, W_cand, Z,
            retrieval_metric=retrieval_metric,
            csls_k=csls_k,
            return_dense=False,
        )
        curr_avg_sim = float(np.mean(sims_cand))
        delta = curr_avg_sim - prev_avg_sim

        W = W_cand
        D = D_cand
        refinement_history.append(curr_avg_sim)

        logging.info(
            f"Refinement Iter {ref_iter:03d} ({retrieval_metric}): avg_similarity = {curr_avg_sim:.6f}, "
            f"delta = {delta:.8f}" + (f", MNN pairs = {mnn_counts_history[-1]}" if mnn_counts_history else "")
        )

        if abs(delta) < refinement_tol:
            converged = True
            break
        prev_avg_sim = curr_avg_sim

    best_targets = D.tolist() if D.ndim == 1 else np.argmax(D, axis=1).tolist()
    mapped_X = (X @ W).astype(np.float32)

    return {
        "W_adv": best_W_adv,
        "W": W,
        "D": D,
        "mapped_X": mapped_X,
        "normalized_Z": Z,
        "src_to_trg_map": best_targets,
        "cluster_to_phoneme_map": best_targets,
        "best_adv_epoch": best_adv_epoch,
        "best_adv_csls": best_adv_csls,
        "final_avg_similarity": float(refinement_history[-1]),
        "refinement_iterations": ref_iter,
        "converged": converged,
        "disc_loss_history": disc_loss_history,
        "gen_loss_history": gen_loss_history,
        "adv_csls_criterion": adv_csls_history,
        "objective_history": refinement_history,
        "mnn_counts_history": mnn_counts_history,
        "retrieval_metric": retrieval_metric,
        "csls_k": csls_k,
    }


class GANEmbeddingAlignmentJob(Job):
    """
    Sisyphus job executing Domain-Adversarial (GAN) Cross-Modal / Cross-Lingual Embedding Alignment
    followed by CSLS Procrustes Refinement (Chung et al., 2018, arXiv:1805.07467).
    Drop-in compatible with SelfLearningEmbeddingAlignmentJob in UNMT and ASR pipelines.
    """
    _version = 1

    def __init__(
        self,
        src_embeddings_npy: Optional[tk.Path] = None,
        trg_embeddings_npy: Optional[tk.Path] = None,
        src_vocab_file: Optional[tk.Path] = None,
        trg_vocab_file: Optional[tk.Path] = None,
        src_counts_file: Optional[tk.Path] = None,
        trg_counts_file: Optional[tk.Path] = None,
        disc_hidden_dim: int = 512,
        disc_num_layers: int = 2,
        disc_input_dropout: float = 0.1,
        label_smoothing: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 32,
        n_epochs: int = 50,
        epoch_size: int = 1000,
        dis_steps: int = 5,
        ortho_beta: float = 0.01,
        refinement_iters: int = 50,
        refinement_tol: float = 1e-6,
        use_mutual_nn_refinement: bool = True,
        retrieval_metric: str = "csls",
        csls_k: int = 5,
        seed: int = 42,
        # Legacy aliases
        audio_embeddings_npy: Optional[tk.Path] = None,
        text_embeddings_npy: Optional[tk.Path] = None,
        audio_vocab_file: Optional[tk.Path] = None,
        text_vocab_file: Optional[tk.Path] = None,
        audio_counts_file: Optional[tk.Path] = None,
        text_counts_file: Optional[tk.Path] = None,
        _version: int = 1,
    ):
        super().__init__()
        self._version = _version
        self.src_embeddings_npy = src_embeddings_npy or audio_embeddings_npy
        self.trg_embeddings_npy = trg_embeddings_npy or text_embeddings_npy
        self.src_vocab_file = src_vocab_file or audio_vocab_file
        self.trg_vocab_file = trg_vocab_file or text_vocab_file
        self.src_counts_file = src_counts_file or audio_counts_file
        self.trg_counts_file = trg_counts_file or text_counts_file

        self.disc_hidden_dim = disc_hidden_dim
        self.disc_num_layers = disc_num_layers
        self.disc_input_dropout = disc_input_dropout
        self.label_smoothing = label_smoothing
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_size = epoch_size
        self.dis_steps = dis_steps
        self.ortho_beta = ortho_beta
        self.refinement_iters = refinement_iters
        self.refinement_tol = refinement_tol
        self.use_mutual_nn_refinement = use_mutual_nn_refinement
        self.retrieval_metric = retrieval_metric
        self.csls_k = csls_k
        self.seed = seed

        # Outputs (identical interface to SelfLearningEmbeddingAlignmentJob + GAN artifacts)
        self.out_mapping_matrix_npy = self.output_path("W_star.npy")
        self.out_adv_mapping_matrix_npy = self.output_path("W_adv.npy")
        self.out_mapped_src_emb_npy = self.output_path("mapped_src_embeddings.npy")
        self.out_normalized_trg_emb_npy = self.output_path("normalized_trg_embeddings.npy")
        self.out_mapped_src_vec_txt = self.output_path("mapped_src_embeddings.vec")
        self.out_normalized_trg_vec_txt = self.output_path("normalized_trg_embeddings.vec")
        self.out_dictionary_json = self.output_path("src_to_trg_dict.json")
        self.out_alignment_report = self.output_path("alignment_report.json")
        self.out_gan_loss_plot = self.output_path("gan_training_curves.png")

        # Backward-compatible aliases
        self.out_mapped_audio_emb_npy = self.out_mapped_src_emb_npy

    def tasks(self) -> Iterator[Task]:
        yield Task("run", resume="run", rqmt={"cpu": 4, "mem": 16, "time": 2})

    def run(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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

        res = run_gan_alignment(
            X_raw=X_raw,
            Z_raw=Z_raw,
            source_freqs=source_freqs,
            target_freqs=target_freqs,
            disc_hidden_dim=self.disc_hidden_dim,
            disc_num_layers=self.disc_num_layers,
            disc_input_dropout=self.disc_input_dropout,
            label_smoothing=self.label_smoothing,
            lr=self.lr,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            epoch_size=self.epoch_size,
            dis_steps=self.dis_steps,
            ortho_beta=self.ortho_beta,
            refinement_iters=self.refinement_iters,
            refinement_tol=self.refinement_tol,
            use_mutual_nn_refinement=self.use_mutual_nn_refinement,
            retrieval_metric=self.retrieval_metric,
            csls_k=self.csls_k,
            seed=self.seed,
        )

        np.save(self.out_mapping_matrix_npy.get_path(), res["W"])
        np.save(self.out_adv_mapping_matrix_npy.get_path(), res["W_adv"])
        np.save(self.out_mapped_src_emb_npy.get_path(), res["mapped_X"])
        np.save(self.out_normalized_trg_emb_npy.get_path(), res["normalized_Z"])

        write_vecmap_text_embeddings(self.out_mapped_src_vec_txt.get_path(), res["mapped_X"], src_vocab)
        write_vecmap_text_embeddings(self.out_normalized_trg_vec_txt.get_path(), res["normalized_Z"], trg_vocab)

        try:
            plot_gan_training_curves(
                history={
                    "disc_loss": res["disc_loss_history"],
                    "gen_loss": res["gen_loss_history"],
                    "adv_csls_criterion": res["adv_csls_criterion"],
                    "best_adv_epoch": res["best_adv_epoch"],
                    "best_adv_csls": res["best_adv_csls"],
                    "refinement_history": res["objective_history"],
                },
                out_png_path=self.out_gan_loss_plot.get_path(),
                title=f"GAN Alignment (K={X_raw.shape[0]} -> P={Z_raw.shape[0]}, d={X_raw.shape[1]})",
            )
        except Exception as e:
            logging.warning(f"Could not render GAN training curves plot: {e}")

        mapping_named = {}
        for s_idx, t_idx in enumerate(res["src_to_trg_map"]):
            s_name = src_vocab.get(s_idx, f"src_{s_idx}")
            t_name = trg_vocab.get(t_idx, f"trg_{t_idx}")
            mapping_named[s_name] = t_name

        with open(self.out_dictionary_json.get_path(), "w", encoding="utf-8") as f:
            json.dump({
                "src_index_to_trg_index": res["src_to_trg_map"],
                "src_token_to_trg_token": mapping_named,
                "cluster_index_to_phoneme_index": res["src_to_trg_map"],
                "cluster_token_to_phoneme_token": mapping_named,
                "alignment_method": "gan",
                "retrieval_metric": self.retrieval_metric,
                "csls_k": self.csls_k,
            }, f, indent=2, ensure_ascii=False)

        with open(self.out_alignment_report.get_path(), "w", encoding="utf-8") as f:
            json.dump({
                "init_method": "gan_domain_adversarial",
                "alignment_method": "gan",
                "paper_reference": "Chung et al. (2018) arXiv:1805.07467",
                "disc_hidden_dim": self.disc_hidden_dim,
                "disc_num_layers": self.disc_num_layers,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "epoch_size": self.epoch_size,
                "dis_steps": self.dis_steps,
                "ortho_beta": self.ortho_beta,
                "retrieval_metric": self.retrieval_metric,
                "csls_k": self.csls_k,
                "best_adv_epoch": res["best_adv_epoch"],
                "best_adv_csls": res["best_adv_csls"],
                "converged": res["converged"],
                "iterations": res["refinement_iterations"],
                "final_avg_similarity": res["final_avg_similarity"],
                "objective_history": res["objective_history"],
                "disc_loss_history": res["disc_loss_history"],
                "gen_loss_history": res["gen_loss_history"],
                "adv_csls_criterion": res["adv_csls_criterion"],
                "mnn_counts_history": res["mnn_counts_history"],
            }, f, indent=2)
