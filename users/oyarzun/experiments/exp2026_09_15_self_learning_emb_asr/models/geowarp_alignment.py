"""
GeoWarP: Coarse Geometric Initialization + Wasserstein Procrustes Cross-Modal Embedding Alignment
(Schnaus, Dagès, Cremers, Wang, & Isola, 2026, arXiv:2610.09411)

Reference:
  Schnaus et al. (2026), "Shared Geometry As A Rosetta Stone: Cross-Modal Alignment Without Paired Data" (arXiv:2610.09411)
  - Stage 1 (GeoWarP Initialization): Multi-scale coarse K-Means clustering on disjoint embedding spaces,
    relational distance / Gram structure matching via Gromov-Wasserstein & Hungarian assignment, and
    coarse orthogonal Procrustes initialization (W_0 in O(d)).
  - Stage 2 (Wasserstein Procrustes Optimization): Alternating temperature-annealed log-domain Sinkhorn
    Optimal Transport (on CSLS-debiased similarity cost) and exact Orthogonal Procrustes SVD updates (W = U V^T).
  - Stage 3 (CSLS MNN Refinement): Mutual nearest-neighbor dictionary induction and final Procrustes polish.
"""

from typing import Dict, List, Optional, Tuple, Iterator
import os
import json
import logging
import numpy as np

from sisyphus import Job, Task, tk
from .self_learning_alignment import (
    normalize_and_center,
    compute_optimal_orthogonal_mapping,
    compute_csls_similarity,
    induce_dictionary_nn,
    load_vocab_dict,
)
from .gan_alignment import (
    compute_structural_compatibility_mask,
    induce_mutual_nn_dictionary,
)


def sinkhorn_log_domain_torch(
    cost_matrix: "torch.Tensor",
    mu_s: "torch.Tensor",
    mu_t: "torch.Tensor",
    reg: float = 0.05,
    num_iters: int = 60,
    tol: float = 1e-5,
) -> "torch.Tensor":
    """
    Numerically stable log-domain Sinkhorn-Knopp optimal transport solver on GPU/CPU.

    Solves:
      min_{P in Pi(mu_s, mu_t)} <P, C> + reg * KL(P || mu_s x mu_t)

    :param cost_matrix: (K x P) tensor of pairwise transport costs
    :param mu_s: (K,) source marginal distribution (sums to 1)
    :param mu_t: (P,) target marginal distribution (sums to 1)
    :param reg: Entropic regularization temperature epsilon > 0
    :param num_iters: Maximum Sinkhorn iterations
    :return: Transport plan P of shape (K x P)
    """
    import torch

    # Normalize cost scale for consistent temperature behavior across iterations
    c_min = torch.min(cost_matrix)
    c_shifted = cost_matrix - c_min
    c_std = torch.std(c_shifted)
    if float(c_std) > 1e-6:
        c_scaled = c_shifted / c_std
    else:
        c_scaled = c_shifted

    log_K = -c_scaled / max(reg, 1e-4)  # (K x P)
    log_mu_s = torch.log(torch.clamp(mu_s, min=1e-12))
    log_mu_t = torch.log(torch.clamp(mu_t, min=1e-12))

    u = torch.zeros_like(mu_s)
    v = torch.zeros_like(mu_t)

    for it in range(num_iters):
        u_prev = u
        u = log_mu_s - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
        v = log_mu_t - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)
        if it % 10 == 0 and float(torch.max(torch.abs(u - u_prev))) < tol:
            break

    log_P = log_K + u.unsqueeze(1) + v.unsqueeze(0)
    P = torch.exp(log_P)
    return P


def solve_orthogonal_procrustes_svd(M: np.ndarray) -> np.ndarray:
    """
    Given cross-covariance matrix M = X^T P Z (d x d), computes the exact orthogonal
    transformation W* = argmax_{W in O(d)} Tr(W^T M) = U V^T.
    """
    U, _, Vt = np.linalg.svd(M.astype(np.float64), full_matrices=False)
    W = (U @ Vt).astype(np.float32)
    return W


def _gromov_wasserstein_coarse_match(
    C_X: np.ndarray,
    C_Z: np.ndarray,
    p_X: np.ndarray,
    p_Z: np.ndarray,
    gw_iters: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matches two sets of coarse centroids C_X (k_c x d) and C_Z (k_c x d) using relational
    intra-space similarity structures (Gromov-Wasserstein + Hungarian permutation).

    :return: (W_soft, W_hard) two candidate orthogonal matrices in O(d)
    """
    from scipy.optimize import linear_sum_assignment
    import torch

    k_c = C_X.shape[0]
    # Normalize centroids to unit sphere
    C_X_norm = C_X / np.maximum(np.linalg.norm(C_X, axis=1, keepdims=True), 1e-12)
    C_Z_norm = C_Z / np.maximum(np.linalg.norm(C_Z, axis=1, keepdims=True), 1e-12)

    # Relational Cosine Gram Matrices
    G_X = C_X_norm @ C_X_norm.T  # (k_c x k_c)
    G_Z = C_Z_norm @ C_Z_norm.T  # (k_c x k_c)

    # 1. Initial coordinate-free relational descriptor: sorted row similarity profiles + weighted moments
    sort_X = np.sort(G_X, axis=1)
    sort_Z = np.sort(G_Z, axis=1)

    # Pairwise L2 distance between sorted relational profiles
    diff = sort_X[:, None, :] - sort_Z[None, :, :]
    M_init = np.sum(diff ** 2, axis=-1)  # (k_c x k_c)

    # Add rank-frequency prior if cluster weights are non-uniform
    freq_diff = (p_X[:, None] - p_Z[None, :]) ** 2
    if np.max(freq_diff) > 1e-12:
        M_init = M_init + 0.25 * (freq_diff / np.max(freq_diff)) * np.mean(M_init)

    row_ind, col_ind = linear_sum_assignment(M_init)
    Gamma = np.zeros((k_c, k_c), dtype=np.float64)
    Gamma[row_ind, col_ind] = 1.0 / k_c
    # Smooth initial coupling
    Gamma = 0.8 * Gamma + 0.2 * np.outer(p_X, p_Z)

    # 2. Entropic Gromov-Wasserstein refinement on relational structures G_X and G_Z
    const_X = (G_X ** 2) @ p_X  # (k_c,)
    const_Z = (G_Z ** 2) @ p_Z  # (k_c,)
    const_mat = const_X[:, None] + const_Z[None, :]

    p_X_t = torch.from_numpy(p_X.astype(np.float32))
    p_Z_t = torch.from_numpy(p_Z.astype(np.float32))

    for it in range(gw_iters):
        # Gradient of GW quadratic loss: L = const - 2 * G_X @ Gamma @ G_Z^T
        L_gw = const_mat - 2.0 * (G_X @ Gamma @ G_Z.T)
        L_gw_t = torch.from_numpy(L_gw.astype(np.float32))
        reg_gw = max(0.03, 0.15 * (0.88 ** it))
        Gamma_next = sinkhorn_log_domain_torch(L_gw_t, p_X_t, p_Z_t, reg=reg_gw, num_iters=40).numpy().astype(np.float64)
        Gamma = 0.5 * Gamma + 0.5 * Gamma_next

    # Soft GW coupling Procrustes candidate
    W_soft = solve_orthogonal_procrustes_svd(C_X.T @ Gamma @ C_Z)

    # Hard Hungarian projection Procrustes candidate
    L_final = const_mat - 2.0 * (G_X @ Gamma @ G_Z.T)
    r_idx, c_idx = linear_sum_assignment(L_final)
    Pi_hard = np.zeros((k_c, k_c), dtype=np.float64)
    Pi_hard[r_idx, c_idx] = 1.0
    W_hard = solve_orthogonal_procrustes_svd(C_X.T @ Pi_hard @ C_Z)

    return W_soft, W_hard


def compute_geowarp_initialization(
    X: np.ndarray,
    Z: np.ndarray,
    source_freqs: Optional[np.ndarray] = None,
    target_freqs: Optional[np.ndarray] = None,
    coarse_cluster_Grid: Optional[List[int]] = None,
    num_seeds_per_k: int = 3,
    retrieval_metric: str = "csls",
    csls_k: int = 10,
    compatibility_mask: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    Stage 1: GeoWarP Coarse Geometric Initialization (Schnaus et al., 2026, arXiv:2610.09411).
    Builds a set of candidate orthogonal initializations via:
      1. Multi-scale coarse K-Means clustering + Gromov-Wasserstein relational matching.
      2. Principal axis / SVD subspace alignment with sign disambiguation on top components.
      3. Frequency-sorted relational profile matching on top anchor tokens.
    Evaluates each candidate with a 2-step lookahead using the unsupervised CSLS criterion,
    returning the best initial orthogonal matrix W_0 in O(d).
    """
    from sklearn.cluster import MiniBatchKMeans, KMeans

    K, d = X.shape
    P, _ = Z.shape
    rng = np.random.RandomState(seed)

    if coarse_cluster_Grid is None:
        max_k = min(K, P)
        coarse_cluster_Grid = [k for k in (16, 32, 48, 64, 96) if k <= max_k]
        if not coarse_cluster_Grid:
            coarse_cluster_Grid = [max(2, max_k // 2)]

    # Normalize frequency weights
    if source_freqs is not None and len(source_freqs) == K and np.sum(source_freqs) > 0:
        w_X = (source_freqs / np.sum(source_freqs)).astype(np.float64)
    else:
        w_X = np.full(K, 1.0 / K, dtype=np.float64)

    if target_freqs is not None and len(target_freqs) == P and np.sum(target_freqs) > 0:
        w_Z = (target_freqs / np.sum(target_freqs)).astype(np.float64)
    else:
        w_Z = np.full(P, 1.0 / P, dtype=np.float64)

    candidates: List[Tuple[str, np.ndarray]] = []

    # Candidate 0: Identity baseline
    candidates.append(("identity", np.eye(d, dtype=np.float32)))

    # Candidate family 1: Multi-scale coarse K-Means + Gromov-Wasserstein relational matching
    for k_c in coarse_cluster_Grid:
        for s_offset in range(num_seeds_per_k):
            cur_seed = seed + 100 * k_c + s_offset
            km_x = KMeans(n_clusters=k_c, random_state=cur_seed, n_init=3, max_iter=100)
            km_z = KMeans(n_clusters=k_c, random_state=cur_seed + 17, n_init=3, max_iter=100)

            km_x.fit(X, sample_weight=w_X)
            km_z.fit(Z, sample_weight=w_Z)

            C_X = km_x.cluster_centers_.astype(np.float64)
            C_Z = km_z.cluster_centers_.astype(np.float64)

            # Compute cluster masses
            p_X = np.bincount(km_x.labels_, weights=w_X, minlength=k_c).astype(np.float64)
            p_Z = np.bincount(km_z.labels_, weights=w_Z, minlength=k_c).astype(np.float64)
            p_X = np.maximum(p_X, 1e-8) / np.sum(np.maximum(p_X, 1e-8))
            p_Z = np.maximum(p_Z, 1e-8) / np.sum(np.maximum(p_Z, 1e-8))

            W_soft, W_hard = _gromov_wasserstein_coarse_match(C_X, C_Z, p_X, p_Z, gw_iters=15)
            candidates.append((f"geowarp_gw_soft_k{k_c}_s{s_offset}", W_soft))
            candidates.append((f"geowarp_gw_hard_k{k_c}_s{s_offset}", W_hard))

    # Candidate family 2: Direct top-anchor relational GW matching (top M most frequent tokens)
    for m_anchors in (64, 128, 256):
        if m_anchors <= min(K, P):
            top_s_idx = np.argsort(-w_X)[:m_anchors]
            top_t_idx = np.argsort(-w_Z)[:m_anchors]
            p_s_sub = w_X[top_s_idx] / np.sum(w_X[top_s_idx])
            p_t_sub = w_Z[top_t_idx] / np.sum(w_Z[top_t_idx])
            W_a_soft, W_a_hard = _gromov_wasserstein_coarse_match(
                X[top_s_idx].astype(np.float64),
                Z[top_t_idx].astype(np.float64),
                p_s_sub,
                p_t_sub,
                gw_iters=20,
            )
            candidates.append((f"geowarp_anchors_soft_m{m_anchors}", W_a_soft))
            candidates.append((f"geowarp_anchors_hard_m{m_anchors}", W_a_hard))

    # Candidate family 3: Principal Axis / Spectral Basis Alignment with Sign Disambiguation
    _, _, Vt_X = np.linalg.svd(X.astype(np.float64), full_matrices=False)
    _, _, Vt_Z = np.linalg.svd(Z.astype(np.float64), full_matrices=False)
    # Explore all 2^4 = 16 sign combinations on the top 4 principal components
    for mask_bits in range(16):
        signs = np.ones(d, dtype=np.float64)
        for bit_idx in range(min(4, d)):
            if (mask_bits >> bit_idx) & 1:
                signs[bit_idx] = -1.0
        W_pca = (Vt_X.T * signs[None, :]) @ Vt_Z
        candidates.append((f"spectral_pca_signs_{mask_bits}", W_pca.astype(np.float32)))

    # Evaluate all candidates with a fast 2-step Procrustes lookahead
    best_score = -1e9
    best_name = "identity"
    best_W0 = np.eye(d, dtype=np.float32)
    candidate_scores = {}

    logging.info(f"GeoWarP Initialization: Evaluating {len(candidates)} geometric candidates...")
    for name, W_cand in candidates:
        W_eval = W_cand.copy()
        # 2-step fast lookahead
        for _ in range(2):
            D_tmp, _ = induce_dictionary_nn(
                X, W_eval, Z,
                retrieval_metric=retrieval_metric,
                csls_k=csls_k,
                compatibility_mask=compatibility_mask,
            )
            W_eval = compute_optimal_orthogonal_mapping(X, Z, D_tmp)

        _, sims = induce_dictionary_nn(
            X, W_eval, Z,
            retrieval_metric=retrieval_metric,
            csls_k=csls_k,
            compatibility_mask=compatibility_mask,
        )
        score = float(np.mean(sims))
        candidate_scores[name] = score
        if score > best_score:
            best_score = score
            best_name = name
            best_W0 = W_cand.copy()

    logging.info(
        f"GeoWarP Selected Initialization: '{best_name}' with lookahead {retrieval_metric.upper()} = {best_score:.6f}"
    )

    return best_W0, {
        "best_candidate_name": best_name,
        "best_candidate_lookahead_csls": best_score,
        "num_candidates_evaluated": len(candidates),
        "top_5_candidates": sorted(candidate_scores.items(), key=lambda kv: -kv[1])[:5],
    }


def plot_geowarp_training_curves(
    history: Dict,
    out_png_path: str,
    title: str = "GeoWarP & Wasserstein Procrustes Cross-Modal Alignment",
):
    """
    Plots Wasserstein Optimal Transport cost and Unsupervised CSLS alignment criterion
    across Wasserstein Procrustes iterations and MNN Procrustes refinement.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wp_iters = list(range(1, len(history["wp_transport_cost"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200)

    # Panel 1: Wasserstein Transport Cost
    ax1.plot(
        wp_iters,
        history["wp_transport_cost"],
        label="Sinkhorn Wasserstein Transport Cost",
        color="#d9534f",
        linewidth=2.0,
        marker="o",
        markersize=3,
    )
    ax1.set_xlabel("Wasserstein Procrustes Iteration", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Transport Cost $\\langle P, C \\rangle$", fontsize=10, fontweight="bold")
    ax1.set_title("Wasserstein Procrustes Transport Cost", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="best", fontsize=9)

    # Panel 2: Unsupervised Alignment Criterion (Mean CSLS Similarity)
    ax2.plot(
        wp_iters,
        history["wp_csls_criterion"],
        label="Wasserstein Procrustes Mean CSLS",
        color="#2b5c8f",
        linewidth=2.0,
    )
    best_it = history.get("best_wp_iter", 1)
    best_wp_val = history.get("best_wp_csls", 0.0)
    ax2.scatter(
        [best_it],
        [best_wp_val],
        color="#d95f02",
        s=65,
        zorder=5,
        label=f"Best WP Iter ({best_it}: {best_wp_val:.4f})",
    )

    if history.get("refinement_history"):
        ref_vals = history["refinement_history"]
        ax2.axhline(
            ref_vals[-1],
            color="#1b9e77",
            linestyle="-.",
            linewidth=1.8,
            label=f"Post-Refinement CSLS ({ref_vals[-1]:.4f})",
        )

    ax2.set_xlabel("Wasserstein Procrustes Iteration", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean CSLS Similarity", fontsize=10, fontweight="bold")
    ax2.set_title("Unsupervised Alignment Criterion (CSLS)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)


def run_geowarp_alignment(
    X_raw: np.ndarray,
    Z_raw: np.ndarray,
    source_freqs: Optional[np.ndarray] = None,
    target_freqs: Optional[np.ndarray] = None,
    coarse_cluster_grid: Optional[List[int]] = None,
    num_seeds_per_k: int = 3,
    wp_iters: int = 40,
    sinkhorn_iters: int = 60,
    reg_start: float = 0.10,
    reg_end: float = 0.015,
    top_k_coupling: int = 15,
    uniform_marginals: bool = True,
    refinement_iters: int = 50,
    refinement_tol: float = 1e-6,
    use_mutual_nn_refinement: bool = True,
    retrieval_metric: str = "csls",
    csls_k: int = 10,
    seed: int = 42,
    compatibility_mask: Optional[np.ndarray] = None,
    structural_constraint: Optional[str] = None,
) -> Dict:
    """
    Executes GeoWarP Coarse Geometric Initialization followed by Wasserstein Procrustes
    Alternating Optimization and CSLS Procrustes Refinement (arXiv:2610.09411).
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    K, d_x = X_raw.shape
    P, d_z = Z_raw.shape
    assert d_x == d_z, f"Embedding dimension mismatch: audio={d_x} vs text={d_z}"
    d = d_x

    # 1. Preprocessing: length-norm + mean-center + unit norm
    X = normalize_and_center(X_raw).astype(np.float32)
    Z = normalize_and_center(Z_raw).astype(np.float32)

    apply_mask = (
        compatibility_mask is not None
        and structural_constraint in ("masked_csls", "reg_gan_and_masked_csls")
    )
    active_mask = compatibility_mask if apply_mask else None

    # 2. Stage 1: GeoWarP Coarse Geometric Initialization
    W_0, init_info = compute_geowarp_initialization(
        X=X,
        Z=Z,
        source_freqs=source_freqs,
        target_freqs=target_freqs,
        coarse_cluster_Grid=coarse_cluster_grid,
        num_seeds_per_k=num_seeds_per_k,
        retrieval_metric=retrieval_metric,
        csls_k=csls_k,
        compatibility_mask=active_mask,
        seed=seed,
    )

    # Evaluate initial W_0
    _, init_sims = induce_dictionary_nn(
        X, W_0, Z,
        retrieval_metric=retrieval_metric,
        csls_k=csls_k,
        compatibility_mask=active_mask,
    )
    best_wp_csls = float(np.mean(init_sims))
    best_W_wp = W_0.copy()
    best_wp_iter = 0

    # Prepare Optimal Transport marginals
    if not uniform_marginals and source_freqs is not None and len(source_freqs) == K and np.sum(source_freqs) > 0:
        # Smooth frequency power (alpha = 0.5) so rare n-grams still participate in transport
        s_weights = np.power(np.maximum(source_freqs, 1.0), 0.5)
        mu_s_np = (s_weights / np.sum(s_weights)).astype(np.float32)
    else:
        mu_s_np = np.full(K, 1.0 / K, dtype=np.float32)

    if not uniform_marginals and target_freqs is not None and len(target_freqs) == P and np.sum(target_freqs) > 0:
        t_weights = np.power(np.maximum(target_freqs, 1.0), 0.5)
        mu_t_np = (t_weights / np.sum(t_weights)).astype(np.float32)
    else:
        mu_t_np = np.full(P, 1.0 / P, dtype=np.float32)

    mu_s_t = torch.from_numpy(mu_s_np).to(device)
    mu_t_t = torch.from_numpy(mu_t_np).to(device)

    wp_transport_cost_history = []
    wp_csls_history = []

    W_curr = W_0.copy()
    logging.info(
        f"Starting Wasserstein Procrustes Optimization on {device}: K={K}, P={P}, d={d}, "
        f"wp_iters={wp_iters}, reg={reg_start}->{reg_end}, top_k_coupling={top_k_coupling}"
    )

    # 3. Stage 2: Wasserstein Procrustes Alternating Optimization
    for it in range(1, wp_iters + 1):
        # Geometric temperature annealing
        frac = (it - 1) / max(1, wp_iters - 1)
        reg_curr = reg_start * ((reg_end / max(reg_start, 1e-5)) ** frac)

        X_mapped = X @ W_curr
        X_norm = X_mapped / np.maximum(np.linalg.norm(X_mapped, axis=1, keepdims=True), 1e-12)
        Z_norm = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

        # Compute similarity & transport cost matrix
        if retrieval_metric == "csls":
            sim_matrix = compute_csls_similarity(X_norm, Z_norm, k=csls_k)
        else:
            sim_matrix = X_norm @ Z_norm.T

        if active_mask is not None:
            sim_matrix = np.where(active_mask, sim_matrix, -1e3)

        # Cost is negative similarity
        cost_np = (-sim_matrix).astype(np.float32)
        cost_t = torch.from_numpy(cost_np).to(device)

        # E-step: Sinkhorn Optimal Transport Plan P
        P_t = sinkhorn_log_domain_torch(
            cost_matrix=cost_t,
            mu_s=mu_s_t,
            mu_t=mu_t_t,
            reg=reg_curr,
            num_iters=sinkhorn_iters,
        )
        transport_cost = float(torch.sum(P_t * cost_t).item())
        wp_transport_cost_history.append(transport_cost)

        P_np = P_t.detach().cpu().numpy()

        # Optional sparse top-k coupling trimming per source row to suppress diffuse background noise
        if top_k_coupling is not None and 0 < top_k_coupling < P:
            P_trimmed = np.zeros_like(P_np)
            top_indices = np.argpartition(-P_np, kth=top_k_coupling - 1, axis=1)[:, :top_k_coupling]
            row_idx = np.arange(K)[:, None]
            P_trimmed[row_idx, top_indices] = P_np[row_idx, top_indices]
            P_for_svd = P_trimmed
        else:
            P_for_svd = P_np

        # M-step: Orthogonal Procrustes SVD update W = U V^T
        M_cross = X.T @ P_for_svd @ Z
        W_curr = solve_orthogonal_procrustes_svd(M_cross)

        # Unsupervised validation criterion: Mean CSLS similarity of induced dictionary
        _, sims_curr = induce_dictionary_nn(
            X, W_curr, Z,
            retrieval_metric=retrieval_metric,
            csls_k=csls_k,
            compatibility_mask=active_mask,
        )
        curr_csls = float(np.mean(sims_curr))
        wp_csls_history.append(curr_csls)

        if curr_csls > best_wp_csls:
            best_wp_csls = curr_csls
            best_W_wp = W_curr.copy()
            best_wp_iter = it

        if it % 5 == 0 or it == 1 or it == wp_iters:
            logging.info(
                f"WP Iter {it:03d}/{wp_iters:03d} (reg={reg_curr:.4f}) | OT Cost: {transport_cost:.6f} | "
                f"Mean {retrieval_metric.upper()}: {curr_csls:.6f} (Best: {best_wp_csls:.6f} @ iter {best_wp_iter})"
            )

    # 4. Stage 3: Mutual Nearest-Neighbor (MNN) + CSLS Procrustes Polish
    W = best_W_wp.copy()
    D, sims = induce_dictionary_nn(
        X, W, Z,
        retrieval_metric=retrieval_metric,
        csls_k=csls_k,
        compatibility_mask=active_mask,
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
                compatibility_mask=active_mask,
            )
            mnn_counts_history.append(int(mnn_cnt))
            D_procrustes = D_mnn if mnn_cnt >= min(25, min(K, P) // 4) else D
        else:
            D_procrustes = D

        W_cand = compute_optimal_orthogonal_mapping(X, Z, D_procrustes)
        D_cand, sims_cand = induce_dictionary_nn(
            X, W_cand, Z,
            retrieval_metric=retrieval_metric,
            csls_k=csls_k,
            compatibility_mask=active_mask,
        )
        curr_avg_sim = float(np.mean(sims_cand))
        delta = curr_avg_sim - prev_avg_sim

        # Monotonic safeguard: only accept refinement update if similarity improves or stays non-decreasing
        if delta >= -1e-7:
            W = W_cand
            D = D_cand
            refinement_history.append(curr_avg_sim)
        else:
            refinement_history.append(prev_avg_sim)

        logging.info(
            f"Refinement Iter {ref_iter:03d} ({retrieval_metric}): avg_similarity = {curr_avg_sim:.6f}, "
            f"delta = {delta:.8f}" + (f", MNN pairs = {mnn_counts_history[-1]}" if mnn_counts_history else "")
        )

        if abs(delta) < refinement_tol or delta < 0:
            converged = True
            break
        prev_avg_sim = curr_avg_sim

    best_targets = np.argmax(D, axis=1).tolist()
    mapped_X = X @ W

    return {
        "W_init": W_0,
        "W_wp": best_W_wp,
        "W": W,
        "D": D,
        "mapped_X": mapped_X,
        "cluster_to_phoneme_map": best_targets,
        "init_info": init_info,
        "best_wp_iter": best_wp_iter,
        "best_wp_csls": best_wp_csls,
        "final_avg_similarity": float(refinement_history[-1]),
        "refinement_iterations": ref_iter,
        "converged": converged,
        "wp_transport_cost_history": wp_transport_cost_history,
        "wp_csls_criterion": wp_csls_history,
        "objective_history": refinement_history,
        "mnn_counts_history": mnn_counts_history,
        "retrieval_metric": retrieval_metric,
        "csls_k": csls_k,
    }


class GeoWarPEmbeddingAlignmentJob(Job):
    """
    Sisyphus job executing GeoWarP Coarse Geometric Initialization + Wasserstein Procrustes
    Cross-Modal Embedding Alignment (Schnaus et al., 2026, arXiv:2610.09411).
    Drop-in compatible with GANEmbeddingAlignmentJob and SelfLearningEmbeddingAlignmentJob.
    """
    _version = 1

    def __init__(
        self,
        audio_embeddings_npy: tk.Path,
        text_embeddings_npy: tk.Path,
        audio_vocab_file: Optional[tk.Path] = None,
        text_vocab_file: Optional[tk.Path] = None,
        audio_counts_file: Optional[tk.Path] = None,
        text_counts_file: Optional[tk.Path] = None,
        coarse_cluster_grid: Optional[List[int]] = None,
        num_seeds_per_k: int = 3,
        wp_iters: int = 40,
        sinkhorn_iters: int = 60,
        reg_start: float = 0.10,
        reg_end: float = 0.015,
        top_k_coupling: int = 15,
        uniform_marginals: bool = True,
        refinement_iters: int = 50,
        refinement_tol: float = 1e-6,
        use_mutual_nn_refinement: bool = True,
        retrieval_metric: str = "csls",
        csls_k: int = 10,
        seed: int = 42,
        structural_constraint: Optional[str] = None,
        compatibility_mode: str = "collapsed_length",
        _version: int = 1,
    ):
        super().__init__()
        self._version = _version
        self.audio_embeddings_npy = audio_embeddings_npy
        self.text_embeddings_npy = text_embeddings_npy
        self.audio_vocab_file = audio_vocab_file
        self.text_vocab_file = text_vocab_file
        self.audio_counts_file = audio_counts_file
        self.text_counts_file = text_counts_file
        self.coarse_cluster_grid = coarse_cluster_grid
        self.num_seeds_per_k = num_seeds_per_k
        self.wp_iters = wp_iters
        self.sinkhorn_iters = sinkhorn_iters
        self.reg_start = reg_start
        self.reg_end = reg_end
        self.top_k_coupling = top_k_coupling
        self.uniform_marginals = uniform_marginals
        self.refinement_iters = refinement_iters
        self.refinement_tol = refinement_tol
        self.use_mutual_nn_refinement = use_mutual_nn_refinement
        self.retrieval_metric = retrieval_metric
        self.csls_k = csls_k
        self.seed = seed
        self.structural_constraint = structural_constraint
        self.compatibility_mode = compatibility_mode

        # Outputs (drop-in compatible with GANEmbeddingAlignmentJob)
        self.out_mapping_matrix_npy = self.output_path("W_star.npy")
        self.out_init_mapping_matrix_npy = self.output_path("W_init.npy")
        self.out_wp_mapping_matrix_npy = self.output_path("W_wp.npy")
        self.out_adv_mapping_matrix_npy = self.out_wp_mapping_matrix_npy
        self.out_mapped_audio_emb_npy = self.output_path("mapped_audio_embeddings.npy")
        self.out_dictionary_json = self.output_path("cluster_to_phoneme_dict.json")
        self.out_alignment_report = self.output_path("alignment_report.json")
        self.out_geowarp_loss_plot = self.output_path("geowarp_training_curves.png")
        self.out_gan_loss_plot = self.out_geowarp_loss_plot

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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

        audio_vocab_path = self.audio_vocab_file.get_path() if self.audio_vocab_file is not None else None
        text_vocab_path = self.text_vocab_file.get_path() if self.text_vocab_file is not None else None
        audio_vocab = load_vocab_dict(audio_vocab_path)
        text_vocab = load_vocab_dict(text_vocab_path)

        compatibility_mask = None
        if self.structural_constraint in ("masked_csls", "reg_gan_and_masked_csls"):
            compatibility_mask = compute_structural_compatibility_mask(
                audio_vocab=audio_vocab,
                text_vocab=text_vocab,
                K=X_raw.shape[0],
                P=Z_raw.shape[0],
                mode=self.compatibility_mode,
            )

        res = run_geowarp_alignment(
            X_raw=X_raw,
            Z_raw=Z_raw,
            source_freqs=source_freqs,
            target_freqs=target_freqs,
            coarse_cluster_grid=self.coarse_cluster_grid,
            num_seeds_per_k=self.num_seeds_per_k,
            wp_iters=self.wp_iters,
            sinkhorn_iters=self.sinkhorn_iters,
            reg_start=self.reg_start,
            reg_end=self.reg_end,
            top_k_coupling=self.top_k_coupling,
            uniform_marginals=self.uniform_marginals,
            refinement_iters=self.refinement_iters,
            refinement_tol=self.refinement_tol,
            use_mutual_nn_refinement=self.use_mutual_nn_refinement,
            retrieval_metric=self.retrieval_metric,
            csls_k=self.csls_k,
            seed=self.seed,
            compatibility_mask=compatibility_mask,
            structural_constraint=self.structural_constraint,
        )

        np.save(self.out_mapping_matrix_npy.get_path(), res["W"])
        np.save(self.out_init_mapping_matrix_npy.get_path(), res["W_init"])
        np.save(self.out_wp_mapping_matrix_npy.get_path(), res["W_wp"])
        np.save(self.out_mapped_audio_emb_npy.get_path(), res["mapped_X"])

        try:
            plot_geowarp_training_curves(
                history={
                    "wp_transport_cost": res["wp_transport_cost_history"],
                    "wp_csls_criterion": res["wp_csls_criterion"],
                    "best_wp_iter": res["best_wp_iter"],
                    "best_wp_csls": res["best_wp_csls"],
                    "refinement_history": res["objective_history"],
                },
                out_png_path=self.out_geowarp_loss_plot.get_path(),
                title=f"GeoWarP & Wasserstein Procrustes Alignment (K={X_raw.shape[0]} -> P={Z_raw.shape[0]}, d={X_raw.shape[1]})",
            )
        except Exception as e:
            logging.warning(f"Could not render GeoWarP training curves plot: {e}")

        mapping_named = {}
        for s_idx, t_idx in enumerate(res["cluster_to_phoneme_map"]):
            s_name = audio_vocab.get(s_idx, f"cluster_{s_idx}")
            t_name = text_vocab.get(t_idx, f"phoneme_{t_idx}")
            mapping_named[s_name] = t_name

        with open(self.out_dictionary_json.get_path(), "w") as f:
            json.dump({
                "cluster_index_to_phoneme_index": res["cluster_to_phoneme_map"],
                "cluster_token_to_phoneme_token": mapping_named,
                "alignment_method": "geowarp",
                "retrieval_metric": self.retrieval_metric,
                "csls_k": self.csls_k,
                "structural_constraint": self.structural_constraint,
                "compatibility_mode": self.compatibility_mode,
            }, f, indent=2)

        with open(self.out_alignment_report.get_path(), "w") as f:
            json.dump({
                "init_method": "geowarp_coarse_geometric",
                "optimization_method": "wasserstein_procrustes",
                "paper_reference": "Schnaus et al. (2026) arXiv:2610.09411",
                "init_info": res["init_info"],
                "structural_constraint": self.structural_constraint,
                "compatibility_mode": self.compatibility_mode,
                "wp_iters": self.wp_iters,
                "sinkhorn_iters": self.sinkhorn_iters,
                "reg_start": self.reg_start,
                "reg_end": self.reg_end,
                "top_k_coupling": self.top_k_coupling,
                "uniform_marginals": self.uniform_marginals,
                "retrieval_metric": self.retrieval_metric,
                "csls_k": self.csls_k,
                "best_wp_iter": res["best_wp_iter"],
                "best_wp_csls": res["best_wp_csls"],
                "converged": res["converged"],
                "iterations": res["refinement_iterations"],
                "final_avg_similarity": res["final_avg_similarity"],
                "objective_history": res["objective_history"],
                "wp_transport_cost_history": res["wp_transport_cost_history"],
                "wp_csls_criterion": res["wp_csls_criterion"],
                "mnn_counts_history": res["mnn_counts_history"],
            }, f, indent=2)
