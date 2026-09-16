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
    ):
        super().__init__()
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
            vocab_list = []
            with open(self.fixed_vocab_file.get_path(), "r") as f:
                for line in f:
                    tok = line.strip().split()[0]
                    vocab_list.append(tok)
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
