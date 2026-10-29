"""
Word2Vec training jobs for monolingual text corpora (German and English) in UNMT.
Uses Skip-gram (sg=1), 10 negative samples (negative=10), context window of 10 words (window=10),
and 300 dimensions (vector_size=300), following Artetxe et al. (2018) / UNdreaMT.
"""

from typing import Optional, Iterator
import os
import gzip
import logging
import numpy as np

from sisyphus import Job, Task, tk


class TrainWord2VecJob(Job):
    """
    Trains a Word2Vec model independently on a monolingual text file using Skip-gram (sg=1),
    negative=10, window=10, vector_size=300, and exports:
      - out_embeddings_npy: (V, vector_size) float32 numpy array
      - out_embeddings_vec: standard word2vec text format (V vector_size\\nword v1 ... vd)
      - out_vocab_txt: newline-separated vocabulary list (0..V-1)
      - out_returnn_vocab: RETURNN dictionary file ({word: idx})
      - out_counts_txt: word frequency counts
    """
    _version = 3
    __sis_hash_exclude__ = {"sample": None}

    def __init__(
        self,
        token_text_file: tk.Path,
        vector_size: int = 300,
        window: int = 10,
        min_count: int = 5,
        max_vocab_size: Optional[int] = 50000,
        sg: int = 1,  # 1 for Skip-gram, 0 for CBOW
        negative: int = 10,
        epochs: int = 5,
        sample: Optional[float] = None,
        workers: int = 8,
        seed: int = 42,
        fixed_vocab_file: Optional[tk.Path] = None,
        _version: int = 3,
    ):
        super().__init__()
        self._version = _version
        self.token_text_file = token_text_file
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sg = sg
        self.negative = negative
        self.epochs = epochs
        self.sample = sample
        self.workers = workers
        self.seed = seed
        self.fixed_vocab_file = fixed_vocab_file

        # Outputs
        self.out_embeddings_npy = self.output_path("embeddings.npy")
        self.out_embeddings_vec = self.output_path("embeddings.vec")
        self.out_vocab_txt = self.output_path("vocab.txt")
        self.out_returnn_vocab = self.output_path("returnn_vocab.py")
        self.out_counts_txt = self.output_path("counts.txt")
        self.out_model = self.output_path("word2vec.model")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": self.workers, "mem": 32, "time": 8})

    def run(self):
        logging.basicConfig(level=logging.INFO)
        import sys
        extra_paths = [
            "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/container_libs",
            "/work/smt4/benjamin.oyarzun/venv/lib/python3.10/site-packages",
        ]
        for p in extra_paths:
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)

        from gensim.models import Word2Vec

        class SentenceCorpus:
            def __init__(self, filepath: str):
                self.filepath = filepath

            def __iter__(self):
                open_fn = gzip.open if self.filepath.endswith(".gz") else open
                with open_fn(self.filepath, "rt", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        tokens = line.strip().split()
                        if tokens:
                            yield tokens

        corpus = SentenceCorpus(self.token_text_file.get_path())

        logging.info(
            f"Training Word2Vec (sg={self.sg}, negative={self.negative}, window={self.window}, "
            f"dim={self.vector_size}, max_vocab={self.max_vocab_size}) on {self.token_text_file.get_path()}..."
        )
        w2v_kwargs = {
            "sentences": corpus,
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "max_final_vocab": self.max_vocab_size,
            "sg": self.sg,
            "negative": self.negative,
            "epochs": self.epochs,
            "workers": self.workers,
            "seed": self.seed,
        }
        if self.sample is not None:
            w2v_kwargs["sample"] = self.sample

        model = Word2Vec(**w2v_kwargs)

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
            vocab_list = list(model.wv.index_to_key)
            if self.max_vocab_size is not None:
                vocab_list = vocab_list[: self.max_vocab_size]

        V = len(vocab_list)
        embedding_matrix = np.zeros((V, self.vector_size), dtype=np.float32)

        returnn_vocab_dict = {}
        with (
            open(self.out_vocab_txt.get_path(), "w", encoding="utf-8") as f_vocab,
            open(self.out_counts_txt.get_path(), "w", encoding="utf-8") as f_counts,
            open(self.out_embeddings_vec.get_path(), "w", encoding="utf-8") as f_vec,
        ):
            f_vec.write(f"{V} {self.vector_size}\n")
            for idx, word in enumerate(vocab_list):
                if word in model.wv:
                    embedding_matrix[idx] = model.wv[word]
                    cnt = model.wv.get_vecattr(word, "count")
                else:
                    rng = np.random.RandomState(self.seed + idx)
                    embedding_matrix[idx] = rng.randn(self.vector_size).astype(np.float32) * 0.01
                    cnt = 0

                f_vocab.write(f"{word}\n")
                f_counts.write(f"{word} {cnt}\n")
                vec_str = " ".join(f"{val:.6f}" for val in embedding_matrix[idx])
                f_vec.write(f"{word} {vec_str}\n")
                # Offset by 4 special symbols (PAD=0, OOV=1, EOS=2, SOS=3) in RETURNN vocab dict
                returnn_vocab_dict[word] = idx + 4

        # Add special symbols to RETURNN vocab dict
        full_returnn_vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "</s>": 2,
            "<s>": 3,
        }
        for w, i in returnn_vocab_dict.items():
            if w not in full_returnn_vocab:
                full_returnn_vocab[w] = i

        with open(self.out_returnn_vocab.get_path(), "w", encoding="utf-8") as f_ret:
            f_ret.write("{\n")
            for w, i in full_returnn_vocab.items():
                f_ret.write(f"  {w!r}: {i},\n")
            f_ret.write("}\n")

        np.save(self.out_embeddings_npy.get_path(), embedding_matrix)
        logging.info(f"Saved {V} embeddings to {self.out_embeddings_npy.get_path()}")


class GetVocabSizeJob(Job):
    """
    Extracts the exact number of words and total vocabulary dimension (words + 4 special tokens)
    from a RETURNN vocab file or Word2Vec vocab text file.
    """

    def __init__(self, vocab_file: tk.Path):
        super().__init__()
        self.vocab_file = vocab_file
        self.out_vocab_size = self.output_var("vocab_size")
        self.out_total_dim = self.output_var("total_dim")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        import ast

        path = self.vocab_file.get_path()
        if path.endswith(".py"):
            with open(path, "r", encoding="utf-8") as f:
                v = ast.literal_eval(f.read())
            total_dim = len(v)
            num_words = max(0, total_dim - 4)
        else:
            with open(path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            num_words = len(lines)
            total_dim = num_words + 4

        self.out_vocab_size.set(num_words)
        self.out_total_dim.set(total_dim)

