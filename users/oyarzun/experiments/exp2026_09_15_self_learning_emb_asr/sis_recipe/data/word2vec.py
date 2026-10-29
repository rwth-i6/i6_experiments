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
