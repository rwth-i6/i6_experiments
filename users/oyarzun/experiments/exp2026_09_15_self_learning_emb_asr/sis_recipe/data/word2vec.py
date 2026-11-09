"""
Word2Vec training jobs for text phonemes and audio cluster token sequences.
"""

from typing import Optional, Iterator
import os
import logging
import numpy as np

from sisyphus import Job, Task, tk


class TrainWord2VecJob(Job):
    """
    Trains a Word2Vec model on token sequence files (e.g. phoneme sequences or cluster ID sequences)
    and exports the embedding matrix (V x d), vocabulary, and token counts.
    """
    _version = 2

    def __init__(
        self,
        token_text_file: tk.Path,
        vector_size: int = 300,
        window: int = 10,
        min_count: int = 1,
        sg: int = 1,  # 1 for Skip-gram, 0 for CBOW
        negative: int = 10,
        epochs: int = 20,
        workers: int = 4,
        seed: int = 42,
        fixed_vocab_file: Optional[tk.Path] = None,
        _version: int = 2,
    ):
        super().__init__()
        self._version = _version
        self.token_text_file = token_text_file
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.negative = negative
        self.epochs = epochs
        self.workers = workers
        self.seed = seed
        self.fixed_vocab_file = fixed_vocab_file

        # Outputs
        self.out_embeddings_npy = self.output_path("embeddings.npy")
        self.out_vocab_txt = self.output_path("vocab.txt")
        self.out_counts_txt = self.output_path("counts.txt")
        self.out_model = self.output_path("word2vec.model")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": self.workers, "mem": 16, "time": 4})

    def run(self):
        logging.basicConfig(level=logging.INFO)
        import sys
        venv_pkgs = "/work/smt4/benjamin.oyarzun/venv/lib/python3.10/site-packages"
        if venv_pkgs not in sys.path:
            sys.path.append(venv_pkgs)
        from gensim.models import Word2Vec

        class SentenceCorpus:
            def __init__(self, filepath):
                self.filepath = filepath

            def __iter__(self):
                with open(self.filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        tokens = line.strip().split()
                        if tokens:
                            yield tokens

        corpus = SentenceCorpus(self.token_text_file.get_path())

        logging.info(f"Training Word2Vec model on {self.token_text_file.get_path()}...")
        model = Word2Vec(
            sentences=corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            negative=self.negative,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
        )

        model.save(self.out_model.get_path())

        # Determine vocabulary ordering
        if self.fixed_vocab_file is not None and os.path.exists(self.fixed_vocab_file.get_path()):
            with open(self.fixed_vocab_file.get_path(), "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith("{") and content.endswith("}"):
                import ast
                try:
                    vocab_dict = ast.literal_eval(content)
                    vocab_list = [k for k, _ in sorted(vocab_dict.items(), key=lambda item: item[1])]
                except Exception:
                    vocab_list = [line.strip().split()[0] for line in content.splitlines() if line.strip()]
            else:
                vocab_list = [line.strip().split()[0] for line in content.splitlines() if line.strip()]
        else:
            # Sort by frequency / natural index if integer tokens
            try:
                vocab_list = sorted(model.wv.index_to_key, key=lambda x: int(x))
            except ValueError:
                vocab_list = model.wv.index_to_key

        V = len(vocab_list)
        embedding_matrix = np.zeros((V, self.vector_size), dtype=np.float32)
        counts = []

        with open(self.out_vocab_txt.get_path(), "w") as f_vocab, open(self.out_counts_txt.get_path(), "w") as f_counts:
            for idx, word in enumerate(vocab_list):
                if word in model.wv:
                    embedding_matrix[idx] = model.wv[word]
                    cnt = model.wv.get_vecattr(word, "count")
                else:
                    # Random vector if token not in corpus (e.g. rare or unseen)
                    rng = np.random.RandomState(self.seed + idx)
                    embedding_matrix[idx] = rng.randn(self.vector_size).astype(np.float32) * 0.01
                    cnt = 0

                f_vocab.write(f"{word}\n")
                f_counts.write(f"{word} {cnt}\n")

        np.save(self.out_embeddings_npy.get_path(), embedding_matrix)
        logging.info(f"Saved {V} embeddings to {self.out_embeddings_npy.get_path()}")


class TrainStructuredSkipGramJob(Job):
    """
    Trains a Structured Skip-Gram (SSG / position-dependent Word2Vec) model (Ling et al., 2015).
    Unlike standard Word2Vec which treats context symmetrically in a bag-of-words,
    Structured Skip-Gram maintains distinct context projection matrices for each relative offset
    r in [-window, ..., -1, +1, ..., +window].
    Exports the target embedding matrix (V x d), vocabulary, and token counts,
    exactly compatible with TrainWord2VecJob.
    """
    _version = 2

    def __init__(
        self,
        token_text_file: tk.Path,
        vector_size: int = 128,
        window: int = 2,
        min_count: int = 1,
        negative: int = 5,
        epochs: int = 20,
        lr: float = 0.005,
        batch_size: int = 16384,
        workers: int = 4,
        seed: int = 42,
        fixed_vocab_file: Optional[tk.Path] = None,
        _version: int = 2,
    ):
        super().__init__()
        self._version = _version
        self.token_text_file = token_text_file
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.workers = workers
        self.seed = seed
        self.fixed_vocab_file = fixed_vocab_file

        # Outputs (drop-in identical to TrainWord2VecJob + loss curve)
        self.out_embeddings_npy = self.output_path("embeddings.npy")
        self.out_vocab_txt = self.output_path("vocab.txt")
        self.out_counts_txt = self.output_path("counts.txt")
        self.out_loss_plot = self.output_path("ssg_loss.png")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": self.workers, "mem": 16, "time": 4})

    def run(self):
        os.environ["OMP_NUM_THREADS"] = str(self.workers)
        import collections
        import time
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device} (workers: {self.workers})")

        # 1. Read corpus and count token frequencies
        corpus_path = self.token_text_file.get_path()
        logging.info(f"Reading corpus from {corpus_path}...")
        token_counts = collections.Counter()
        sentences = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    sentences.append(tokens)
                    token_counts.update(tokens)

        logging.info(f"Corpus loaded: {len(sentences)} sentences, {sum(token_counts.values())} total tokens, {len(token_counts)} unique tokens.")

        # 2. Determine vocabulary ordering
        if self.fixed_vocab_file is not None and os.path.exists(self.fixed_vocab_file.get_path()):
            fixed_path = self.fixed_vocab_file.get_path()
            logging.info(f"Loading fixed vocabulary from {fixed_path}...")
            with open(fixed_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith("{") and content.endswith("}"):
                import ast
                try:
                    vocab_dict = ast.literal_eval(content)
                    vocab_list = [k for k, _ in sorted(vocab_dict.items(), key=lambda item: item[1])]
                except Exception:
                    vocab_list = [line.strip().split()[0] for line in content.splitlines() if line.strip()]
            else:
                vocab_list = [line.strip().split()[0] for line in content.splitlines() if line.strip()]
        else:
            try:
                vocab_list = sorted([w for w, c in token_counts.items() if c >= self.min_count], key=lambda x: int(x))
            except ValueError:
                vocab_list = [w for w, _ in token_counts.most_common() if token_counts[w] >= self.min_count]

        V = len(vocab_list)
        word2idx = {w: idx for idx, w in enumerate(vocab_list)}
        logging.info(f"Vocabulary size: {V}")

        # 3. Build Structured Skip-Gram pairs
        offsets = [r for r in range(-self.window, self.window + 1) if r != 0]
        num_offsets = len(offsets)
        logging.info(f"Extracting context pairs for window={self.window} (offsets: {offsets})...")

        targets = []
        contexts = []
        pos_ids = []

        for sent in sentences:
            token_ids = [word2idx[w] for w in sent if w in word2idx]
            L = len(token_ids)
            for i in range(L):
                t = token_ids[i]
                for p_idx, r in enumerate(offsets):
                    ctx_i = i + r
                    if 0 <= ctx_i < L:
                        targets.append(t)
                        contexts.append(token_ids[ctx_i])
                        pos_ids.append(p_idx)

        targets = np.array(targets, dtype=np.int32)
        contexts = np.array(contexts, dtype=np.int32)
        pos_ids = np.array(pos_ids, dtype=np.int16)
        total_pairs = len(targets)
        logging.info(f"Total Structured Skip-Gram pairs: {total_pairs:,}")

        # 4. Precompute unigram negative sampling table (Mikolov / word2vec standard)
        # Avoids PyTorch multinomial OpenMP multithreading bug on CPU Slurm nodes
        raw_counts = np.array([max(token_counts.get(w, 0), 1) for w in vocab_list], dtype=np.float32)
        unigram_probs = raw_counts ** 0.75
        unigram_probs /= unigram_probs.sum()

        table_size = 2_000_000
        table_counts = np.maximum(np.round(unigram_probs * table_size).astype(np.int32), 1)
        neg_table = np.repeat(np.arange(V, dtype=np.int32), table_counts)
        neg_table_t = torch.from_numpy(neg_table).long().to(device)
        neg_table_len = len(neg_table_t)
        logging.info(f"Negative sampling table built with {neg_table_len:,} entries.")

        # 5. Model definition: Target embeddings + Position-specific Context embeddings
        target_emb = nn.Embedding(V, self.vector_size).to(device)
        context_emb = nn.Embedding(num_offsets * V, self.vector_size).to(device)
        nn.init.uniform_(target_emb.weight, -0.5 / self.vector_size, 0.5 / self.vector_size)
        nn.init.zeros_(context_emb.weight)

        optimizer = torch.optim.Adam(
            list(target_emb.parameters()) + list(context_emb.parameters()),
            lr=self.lr,
        )

        epoch_losses = []
        start_time = time.time()
        logging.info(f"Starting SSG training: {self.epochs} epochs, batch_size={self.batch_size}, negative={self.negative}...")

        for epoch in range(1, self.epochs + 1):
            perm = np.random.permutation(total_pairs)
            total_loss = 0.0
            num_batches = 0

            # Learning rate linear decay
            current_lr = max(self.lr * (1.0 - (epoch - 1) / self.epochs), 1e-4)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            for start in range(0, total_pairs, self.batch_size):
                end = min(start + self.batch_size, total_pairs)
                b_idx = perm[start:end]
                B = len(b_idx)

                t_b = torch.from_numpy(targets[b_idx]).long().to(device)
                c_b = torch.from_numpy(contexts[b_idx]).long().to(device)
                p_b = torch.from_numpy(pos_ids[b_idx]).long().to(device)

                neg_idx = torch.randint(0, neg_table_len, (B, self.negative), device=device)
                neg_b = neg_table_t[neg_idx]

                u = target_emb(t_b)  # (B, d)
                pos_ctx = p_b * V + c_b  # (B,)
                v_pos = context_emb(pos_ctx)  # (B, d)

                neg_ctx = p_b.unsqueeze(1) * V + neg_b  # (B, K)
                v_neg = context_emb(neg_ctx)  # (B, K, d)

                pos_dot = (u * v_pos).sum(dim=-1)  # (B,)
                neg_dot = (u.unsqueeze(1) * v_neg).sum(dim=-1)  # (B, K)

                loss = -F.logsigmoid(pos_dot).mean() - F.logsigmoid(-neg_dot).sum(dim=-1).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)
            elapsed = time.time() - start_time
            logging.info(f"Epoch {epoch:2d}/{self.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Elapsed: {elapsed:.1f}s")

        # 6. Save learned target embeddings
        embedding_matrix = target_emb.weight.detach().cpu().numpy()
        np.save(self.out_embeddings_npy.get_path(), embedding_matrix)
        logging.info(f"Saved {V} Structured Skip-Gram embeddings to {self.out_embeddings_npy.get_path()}")

        with open(self.out_vocab_txt.get_path(), "w", encoding="utf-8") as f_vocab, open(self.out_counts_txt.get_path(), "w", encoding="utf-8") as f_counts:
            for word in vocab_list:
                cnt = token_counts.get(word, 0)
                f_vocab.write(f"{word}\n")
                f_counts.write(f"{word} {cnt}\n")

        # 7. Plot loss curve
        try:
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", color="#1f77b4", label="SSG Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Structured Skip-Gram Loss (dim={self.vector_size}, win={self.window})")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.out_loss_plot.get_path(), dpi=150)
            plt.close()
        except Exception as e:
            logging.warning(f"Could not save loss plot: {e}")


def compute_isotropy_measure(embeddings: np.ndarray) -> dict:
    """
    Computes the partition-function isotropy measure from Mu, Bhat, & Viswanath (2018),
    "All-but-the-Top: Simple and Effective Postprocessing for Word Representations"
    (https://arxiv.org/pdf/1702.01417):

      Z(c) = sum_{w in V} exp(c^T v(w))
      I({v(w)}) = (min_{c in C} Z(c)) / (max_{c in C} Z(c))

    where C = {u_1, ..., u_d} is the set of principal directions / unit eigenvectors of V^T V.
    Computed via logsumexp in float64 for exact numerical stability:
      I({v(w)}) = exp(min_{c in C} log Z(c) - max_{c in C} log Z(c))
    """
    from scipy.special import logsumexp

    X = np.asarray(embeddings, dtype=np.float64)
    V_size, d = X.shape

    # Mean vector and norms
    mu = np.mean(X, axis=0)
    mean_vec_norm = float(np.linalg.norm(mu))
    vec_norms = np.linalg.norm(X, axis=1)
    avg_vec_norm = float(np.mean(vec_norms))
    std_vec_norm = float(np.std(vec_norms))
    mean_to_avg_norm_ratio = float(mean_vec_norm / max(avg_vec_norm, 1e-12))

    # 1. Eigenvectors C of uncentered second-moment matrix X^T X (right singular vectors of X)
    _, s_uncentered, vt_uncentered = np.linalg.svd(X, full_matrices=False)
    proj_uncentered = X @ vt_uncentered.T  # (V_size, d)
    log_Z = logsumexp(proj_uncentered, axis=0)  # (d,)
    min_log_Z = float(np.min(log_Z))
    max_log_Z = float(np.max(log_Z))
    isotropy_eig = float(np.exp(min_log_Z - max_log_Z))

    # Symmetric +/- eigenvectors {+u_i, -u_i}
    log_Z_neg = logsumexp(-proj_uncentered, axis=0)
    log_Z_sym = np.concatenate([log_Z, log_Z_neg])
    isotropy_sym = float(np.exp(np.min(log_Z_sym) - np.max(log_Z_sym)))

    # 2. Eigenvectors of centered covariance matrix (X - mu)^T (X - mu)
    X_centered = X - mu[None, :]
    _, s_centered, vt_centered = np.linalg.svd(X_centered, full_matrices=False)
    proj_on_centered_pcs = X @ vt_centered.T
    log_Z_centered_pcs = logsumexp(proj_on_centered_pcs, axis=0)
    isotropy_centered_pcs = float(np.exp(np.min(log_Z_centered_pcs) - np.max(log_Z_centered_pcs)))

    total_var = float(np.sum(s_centered ** 2))
    evr = (s_centered ** 2) / max(total_var, 1e-12)

    return {
        "vocab_size": int(V_size),
        "embedding_dim": int(d),
        "isotropy_measure": isotropy_eig,
        "isotropy_measure_symmetric": isotropy_sym,
        "isotropy_measure_centered_pca_dirs": isotropy_centered_pcs,
        "min_log_Z": min_log_Z,
        "max_log_Z": max_log_Z,
        "mean_log_Z": float(np.mean(log_Z)),
        "std_log_Z": float(np.std(log_Z)),
        "log_vocab_size": float(np.log(max(V_size, 1))),
        "mean_vector_norm": mean_vec_norm,
        "avg_embedding_norm": avg_vec_norm,
        "std_embedding_norm": std_vec_norm,
        "mean_to_avg_norm_ratio": mean_to_avg_norm_ratio,
        "top_10_singular_values_uncentered": [float(v) for v in s_uncentered[:10]],
        "top_10_singular_values_centered": [float(v) for v in s_centered[:10]],
        "top_10_explained_variance_ratios": [float(v) for v in evr[:10]],
        "top_2d_explained_variance_ratio": float(np.sum(evr[:2])),
        "log_Z_per_component": [float(v) for v in log_Z],
    }


def apply_all_but_the_top(
    embeddings: np.ndarray,
    d_remove: Optional[int] = None,
) -> tuple:
    """
    Applies Algorithm 1 ("All-but-the-Top" postprocessing for word representations) from
    Mu, Bhat, & Viswanath (2018), ICLR 2018 (https://arxiv.org/pdf/1702.01417):

      1. Subtract mean:
           mu = (1 / |V|) sum_{w in V} v(w)
           v_tilde(w) = v(w) - mu
      2. Compute top D PCA components u_1, ..., u_D of {v_tilde(w), w in V}
         (default rule of thumb from the paper: D = max(1, d // 100)).
      3. Project out top D components:
           v'(w) = v_tilde(w) - sum_{i=1}^{D} (u_i^T v_tilde(w)) u_i

    :param embeddings: Input array of shape (V, d)
    :param d_remove: Number of top principal components D to remove (default: max(1, d // 100))
    :return: Tuple of (postprocessed_embeddings_float32, metadata_dict)
    """
    X = np.asarray(embeddings, dtype=np.float64)
    V_size, d = X.shape
    if d_remove is None:
        d_remove = max(1, d // 100)
    d_remove = int(d_remove)

    # Step 1: Center by subtracting common mean vector mu
    mu = np.mean(X, axis=0, keepdims=True)
    X_tilde = X - mu

    # Step 2: Compute PCA components via SVD of centered representations
    _, s, vt = np.linalg.svd(X_tilde, full_matrices=False)
    total_var = float(np.sum(s ** 2))
    evr = (s ** 2) / max(total_var, 1e-12)

    # Step 3: Remove top D principal directions u_1, ..., u_D
    if d_remove > 0:
        u_top = vt[:d_remove]  # (D, d)
        X_prime = X_tilde - (X_tilde @ u_top.T) @ u_top
        removed_var_ratio = float(np.sum(evr[:d_remove]))
    else:
        X_prime = X_tilde
        removed_var_ratio = 0.0

    metadata = {
        "d_remove": d_remove,
        "removed_variance_ratio": removed_var_ratio,
        "mean_vector_norm_removed": float(np.linalg.norm(mu)),
        "top_d_singular_values_removed": [float(v) for v in s[:d_remove]],
        "top_d_explained_variance_ratios_removed": [float(v) for v in evr[:d_remove]],
    }
    return X_prime.astype(np.float32), metadata


class PostprocessWord2VecAllButTheTopJob(Job):
    """
    Applies the "All-but-the-Top" isotropic postprocessing transformation (Mu, Bhat, & Viswanath,
    ICLR 2018, https://arxiv.org/pdf/1702.01417) to trained Word2Vec / Structured Skip-Gram embeddings:
      1. Mean-centering: v_tilde(w) = v(w) - mu
      2. Removing the top D principal components: v'(w) = v_tilde(w) - sum_{i=1}^D (u_i^T v_tilde(w)) u_i

    Exposes the exact same output interface as TrainWord2VecJob / TrainStructuredSkipGramJob:
      - out_embeddings_npy: (V, d) float32 numpy array of postprocessed representations v'(w)
      - out_embeddings_vec: standard word2vec text format (V d\\nword v1 ... vd)
      - out_vocab_txt: newline-separated vocabulary list (0..V-1)
      - out_counts_txt: token frequency counts
      - out_isotropy_report_json: JSON report with isotropy measures I({v(w)}) before and after
      - out_isotropy_report_txt: human-readable text report comparing isotropy before and after
    """
    _version = 1

    def __init__(
        self,
        embeddings_npy: tk.Path,
        vocab_txt: tk.Path,
        counts_txt: tk.Path,
        num_components_to_remove: Optional[int] = None,
        returnn_vocab: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.embeddings_npy = embeddings_npy
        self.vocab_txt = vocab_txt
        self.counts_txt = counts_txt
        self.num_components_to_remove = num_components_to_remove
        self.returnn_vocab = returnn_vocab

        self.out_embeddings_npy = self.output_path("embeddings.npy")
        self.out_embeddings_vec = self.output_path("embeddings.vec")
        self.out_vocab_txt = vocab_txt
        self.out_counts_txt = counts_txt
        if returnn_vocab is not None:
            self.out_returnn_vocab = returnn_vocab
        self.out_isotropy_report_json = self.output_path("isotropy_report.json")
        self.out_isotropy_json = self.out_isotropy_report_json
        self.out_isotropy_report_txt = self.output_path("isotropy_report.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        import json

        logging.basicConfig(level=logging.INFO)

        raw_embs = np.load(self.embeddings_npy.get_path())
        V, dim = raw_embs.shape
        d_remove = self.num_components_to_remove if self.num_components_to_remove is not None else max(1, dim // 100)

        logging.info(
            f"Applying All-but-the-Top postprocessing (Mu et al., 2018, arXiv:1702.01417) "
            f"to embeddings of shape ({V}, {dim}) with D={d_remove}"
        )

        # Compute isotropy before transformation
        isotropy_before = compute_isotropy_measure(raw_embs)

        # Compute isotropy after mean centering only (D=0)
        embs_centered, _ = apply_all_but_the_top(raw_embs, d_remove=0)
        isotropy_centered_only = compute_isotropy_measure(embs_centered)

        # Apply All-but-the-Top transformation with target D
        post_embs, abtt_meta = apply_all_but_the_top(raw_embs, d_remove=d_remove)
        isotropy_after = compute_isotropy_measure(post_embs)

        # Also sweep D in [1, 2, 3, 5, 7, 10] for reference
        sweep_results = {}
        for d_val in (1, 2, 3, 5, 7, 10):
            if d_val < dim:
                embs_d, meta_d = apply_all_but_the_top(raw_embs, d_remove=d_val)
                iso_d = compute_isotropy_measure(embs_d)
                sweep_results[f"D={d_val}"] = {
                    "d_remove": d_val,
                    "isotropy_measure": iso_d["isotropy_measure"],
                    "isotropy_measure_symmetric": iso_d["isotropy_measure_symmetric"],
                    "min_log_Z": iso_d["min_log_Z"],
                    "max_log_Z": iso_d["max_log_Z"],
                    "removed_variance_ratio": meta_d["removed_variance_ratio"],
                    "top_2d_explained_variance_ratio_after": iso_d["top_2d_explained_variance_ratio"],
                }

        # Save postprocessed embeddings (.npy and .vec)
        np.save(self.out_embeddings_npy.get_path(), post_embs)

        with open(self.vocab_txt.get_path(), "r", encoding="utf-8") as f_v:
            words = [line.strip().split()[0] for line in f_v if line.strip()]

        with open(self.out_embeddings_vec.get_path(), "w", encoding="utf-8") as f_vec:
            f_vec.write(f"{V} {dim}\n")
            for idx, word in enumerate(words[:V]):
                vec_str = " ".join(f"{val:.6f}" for val in post_embs[idx])
                f_vec.write(f"{word} {vec_str}\n")

        # Save JSON and text isotropy reports
        report_data = {
            "paper_reference": "Mu, Bhat, & Viswanath (2018), All-but-the-Top: Simple and Effective Postprocessing for Word Representations (https://arxiv.org/pdf/1702.01417)",
            "vocab_size": int(V),
            "embedding_dim": int(dim),
            "num_components_removed_D": int(d_remove),
            "abtt_transformation_metadata": abtt_meta,
            "before_transformation": {
                k: v for k, v in isotropy_before.items() if k != "log_Z_per_component"
            },
            "mean_centered_only_D0": {
                k: v for k, v in isotropy_centered_only.items() if k != "log_Z_per_component"
            },
            "after_transformation": {
                k: v for k, v in isotropy_after.items() if k != "log_Z_per_component"
            },
            "isotropy_improvement_delta": float(isotropy_after["isotropy_measure"] - isotropy_before["isotropy_measure"]),
            "isotropy_improvement_ratio": float(
                isotropy_after["isotropy_measure"] / max(isotropy_before["isotropy_measure"], 1e-12)
            ),
            "d_sweep_comparison": sweep_results,
        }

        with open(self.out_isotropy_report_json.get_path(), "w", encoding="utf-8") as f_json:
            json.dump(report_data, f_json, indent=2)

        lines = [
            "=" * 88,
            "WORD2VEC / SSG ISOTROPY ANALYSIS (Mu, Bhat, & Viswanath 2018 — arXiv:1702.01417)",
            "=" * 88,
            f"Vocabulary Size |V|          : {V}",
            f"Embedding Dimension d        : {dim}",
            f"Removed Top PCA Components D : {d_remove} (rule of thumb: d / 100 = {max(1, dim // 100)})",
            f"Removed Variance Ratio       : {abtt_meta['removed_variance_ratio'] * 100:.2f}%",
            "-" * 88,
            "Isotropy Measure: I({v(w)}) = min_{c in C} Z(c) / max_{c in C} Z(c),  Z(c) = sum_{w in V} exp(c^T v(w))",
            "-" * 88,
            f"{'Stage':<34} | {'I({v(w)})':>12} | {'I_sym(+/-c)':>12} | {'min log Z':>10} | {'max log Z':>10} | {'||mu||_2':>9}",
            "-" * 88,
            (
                f"{'1. Raw Embeddings (Before)':<34} | "
                f"{isotropy_before['isotropy_measure']:>12.6f} | "
                f"{isotropy_before['isotropy_measure_symmetric']:>12.6f} | "
                f"{isotropy_before['min_log_Z']:>10.2f} | "
                f"{isotropy_before['max_log_Z']:>10.2f} | "
                f"{isotropy_before['mean_vector_norm']:>9.4f}"
            ),
            (
                f"{'2. Mean-Centered Only (D=0)':<34} | "
                f"{isotropy_centered_only['isotropy_measure']:>12.6f} | "
                f"{isotropy_centered_only['isotropy_measure_symmetric']:>12.6f} | "
                f"{isotropy_centered_only['min_log_Z']:>10.2f} | "
                f"{isotropy_centered_only['max_log_Z']:>10.2f} | "
                f"{isotropy_centered_only['mean_vector_norm']:>9.4f}"
            ),
            (
                f"{f'3. All-but-the-Top (D={d_remove}) [Active]':<34} | "
                f"{isotropy_after['isotropy_measure']:>12.6f} | "
                f"{isotropy_after['isotropy_measure_symmetric']:>12.6f} | "
                f"{isotropy_after['min_log_Z']:>10.2f} | "
                f"{isotropy_after['max_log_Z']:>10.2f} | "
                f"{isotropy_after['mean_vector_norm']:>9.4f}"
            ),
            "-" * 88,
            "Sweep across number of removed PCA components D:",
        ]
        for d_key, d_info in sweep_results.items():
            lines.append(
                f"  {d_key:<8}: I({{v(w)}}) = {d_info['isotropy_measure']:.6f} "
                f"(I_sym = {d_info['isotropy_measure_symmetric']:.6f}, "
                f"min_logZ = {d_info['min_log_Z']:.2f}, max_logZ = {d_info['max_log_Z']:.2f}, "
                f"removed_var = {d_info['removed_variance_ratio']*100:.2f}%)"
            )
        lines.append("=" * 88)
        with open(self.out_isotropy_report_txt.get_path(), "w", encoding="utf-8") as f_txt:
            f_txt.write("\n".join(lines) + "\n")

