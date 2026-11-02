"""
Word2Vec training jobs for monolingual text corpora (German and English) in UNMT.
Uses Skip-gram (sg=1), 10 negative samples (negative=10), context window of 10 words (window=10),
and 300 dimensions (vector_size=300), following Artetxe et al. (2018) / UNdreaMT.
"""

from typing import Optional, Iterator, Dict, List
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
        from ..cluster_adapter import setup_extra_python_paths
        setup_extra_python_paths()

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


class AverageWordNgramEmbeddingsJob(Job):
    """
    Constructs word-level embeddings for the top N (e.g. 50k) most frequent words in a text corpus
    by decomposing each word into its constituent phoneme n-grams (diphones or triphones) and
    averaging their trained n-gram embeddings.

    Exports outputs drop-in compatible with TrainWord2VecJob:
      - out_embeddings_npy: (V, vector_size) float32 numpy array
      - out_embeddings_vec: standard word2vec text format (V vector_size\\nword v1 ... vd)
      - out_vocab_txt: newline-separated vocabulary list (0..V-1)
      - out_returnn_vocab: RETURNN dictionary file ({word: idx})
      - out_counts_txt: word frequency counts
    """
    _version = 1

    def __init__(
        self,
        word_text_file: tk.Path,
        ngram_embeddings_npy: tk.Path,
        ngram_vocab_file: tk.Path,
        ngram_order: int = 2,
        delimiter: str = "_",
        max_vocab_size: int = 50000,
        min_count: int = 5,
        vector_size: int = 300,
        normalize: bool = False,
        seed: int = 42,
        fixed_vocab_file: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.word_text_file = word_text_file
        self.ngram_embeddings_npy = ngram_embeddings_npy
        self.ngram_vocab_file = ngram_vocab_file
        self.ngram_order = ngram_order
        self.delimiter = delimiter
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.vector_size = vector_size
        self.normalize = normalize
        self.seed = seed
        self.fixed_vocab_file = fixed_vocab_file

        self.out_embeddings_npy = self.output_path("embeddings.npy")
        self.out_embeddings_vec = self.output_path("embeddings.vec")
        self.out_vocab_txt = self.output_path("vocab.txt")
        self.out_returnn_vocab = self.output_path("returnn_vocab.py")
        self.out_counts_txt = self.output_path("counts.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 4})

    def run(self):
        import re
        from collections import Counter
        from ..cluster_adapter import get_nltk_data_paths, setup_extra_python_paths

        logging.basicConfig(level=logging.INFO)
        setup_extra_python_paths()

        import nltk
        for nltk_data_path in get_nltk_data_paths():
            if os.path.exists(nltk_data_path) and nltk_data_path not in nltk.data.path:
                nltk.data.path.insert(0, nltk_data_path)

        from nltk.corpus import cmudict
        cmu = cmudict.dict()
        from g2p_en import G2p
        g2p = G2p()

        # 1. Load N-gram embeddings and vocabulary
        ngram_embs = np.load(self.ngram_embeddings_npy.get_path())
        with open(self.ngram_vocab_file.get_path(), "r", encoding="utf-8") as f:
            ngram_vocab_list = [line.strip().split()[0] for line in f if line.strip()]
        ngram_to_idx = {ng: i for i, ng in enumerate(ngram_vocab_list)}
        logging.info(f"Loaded {len(ngram_to_idx)} n-gram embeddings of dim {ngram_embs.shape[1]}")

        # 2. Extract word vocabulary and counts
        open_fn = gzip.open if self.word_text_file.get_path().endswith(".gz") else open
        word_counts = Counter()
        with open_fn(self.word_text_file.get_path(), "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    word_counts.update(tokens)

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
            candidates = [w for w, c in word_counts.most_common() if c >= self.min_count]
            vocab_list = candidates[: self.max_vocab_size]

        V = len(vocab_list)
        logging.info(f"Building averaged embeddings for {V} words (order={self.ngram_order})...")

        # 3. Phonemization cache and word->phones helper
        cache: Dict[str, List[str]] = {}

        def word_to_phones(word: str) -> List[str]:
            if word in cache:
                return cache[word]
            w_clean = re.sub(r"[^a-zA-Z0-9\']", "", word).lower()
            if not w_clean:
                cache[word] = []
                return []
            if w_clean in cmu:
                phones = [re.sub(r"\d+", "", p) for p in cmu[w_clean][0]]
            else:
                try:
                    res = g2p(w_clean)
                    phones = [re.sub(r"\d+", "", p) for p in res if p.strip() and p.isalnum()]
                except Exception:
                    phones = []
            cache[word] = phones
            return phones

        n = self.ngram_order
        delim = self.delimiter
        embedding_matrix = np.zeros((V, self.vector_size), dtype=np.float32)
        returnn_vocab_dict = {}

        num_with_ngrams = 0
        with (
            open(self.out_vocab_txt.get_path(), "w", encoding="utf-8") as f_vocab,
            open(self.out_counts_txt.get_path(), "w", encoding="utf-8") as f_counts,
            open(self.out_embeddings_vec.get_path(), "w", encoding="utf-8") as f_vec,
        ):
            f_vec.write(f"{V} {self.vector_size}\n")
            for idx, word in enumerate(vocab_list):
                phones = word_to_phones(word)
                if len(phones) == 0:
                    word_ngrams = []
                elif len(phones) < n:
                    word_ngrams = [delim.join(phones)]
                else:
                    word_ngrams = [delim.join(phones[i : i + n]) for i in range(len(phones) - n + 1)]

                valid_vecs = [ngram_embs[ngram_to_idx[ng]] for ng in word_ngrams if ng in ngram_to_idx]
                if valid_vecs:
                    avg_vec = np.mean(valid_vecs, axis=0)
                    if self.normalize:
                        norm = np.linalg.norm(avg_vec)
                        if norm > 1e-12:
                            avg_vec = avg_vec / norm
                    embedding_matrix[idx] = avg_vec
                    num_with_ngrams += 1
                else:
                    rng = np.random.RandomState(self.seed + idx)
                    embedding_matrix[idx] = rng.randn(self.vector_size).astype(np.float32) * 0.01

                cnt = word_counts.get(word, 0)
                f_vocab.write(f"{word}\n")
                f_counts.write(f"{word} {cnt}\n")
                vec_str = " ".join(f"{val:.6f}" for val in embedding_matrix[idx])
                f_vec.write(f"{word} {vec_str}\n")
                returnn_vocab_dict[word] = idx + 4

        # Add special symbols for RETURNN vocab dict
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
        logging.info(
            f"Successfully saved {V} averaged embeddings ({num_with_ngrams}/{V} words matched n-grams) "
            f"to {self.out_embeddings_npy.get_path()}"
        )


class ClusterWordAveragedEmbeddingsJob(Job):
    """
    Averages constituent phonetic monophone embeddings for English words, fits MiniBatchKMeans
    clustering (K=50k or 10k), and outputs:
      - out_cluster_text: English text corpus with words mapped to cluster index tokens (c0, c1, ...)
      - out_cluster_vocab_txt: vocabulary list of cluster tokens (c0, c1, ...)
      - out_cluster_counts_txt: frequency counts of each cluster token
      - out_word_to_cluster_json: mapping from word string to cluster token
      - out_cluster_to_word_json: mapping from cluster token to the most frequent word in that cluster
      - out_centroids_npy: K-means centroid vectors [K, vector_size]
    """
    _version = 1

    def __init__(
        self,
        word_text_file: tk.Path,
        ngram_embeddings_npy: tk.Path,
        ngram_vocab_file: tk.Path,
        num_clusters: int,
        max_input_words: Optional[int] = None,
        min_count: int = 2,
        ngram_order: int = 1,
        vector_size: int = 300,
        delimiter: str = "_",
        normalize: bool = True,
        batch_size: int = 8192,
        max_iter: int = 50,
        seed: int = 42,
    ):
        super().__init__()
        self.word_text_file = word_text_file
        self.ngram_embeddings_npy = ngram_embeddings_npy
        self.ngram_vocab_file = ngram_vocab_file
        self.num_clusters = num_clusters
        self.max_input_words = max_input_words if max_input_words is not None else max(150_000, num_clusters * 3)
        self.min_count = min_count
        self.ngram_order = ngram_order
        self.vector_size = vector_size
        self.delimiter = delimiter
        self.normalize = normalize
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.seed = seed

        self.out_cluster_text = self.output_path("news_crawl.en.cluster.txt")
        self.out_cluster_vocab_txt = self.output_path("cluster_vocab.txt")
        self.out_cluster_counts_txt = self.output_path("cluster_counts.txt")
        self.out_word_to_cluster_json = self.output_path("word_to_cluster.json")
        self.out_cluster_to_word_json = self.output_path("cluster_to_word.json")
        self.out_centroids_npy = self.output_path("centroids.npy")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 8, "mem": 32, "time": 4})

    def run(self):
        import re
        import json
        from collections import Counter
        from sklearn.cluster import MiniBatchKMeans
        from ..cluster_adapter import get_nltk_data_paths, setup_extra_python_paths

        logging.basicConfig(level=logging.INFO)
        setup_extra_python_paths()

        import nltk
        for nltk_data_path in get_nltk_data_paths():
            if os.path.exists(nltk_data_path) and nltk_data_path not in nltk.data.path:
                nltk.data.path.insert(0, nltk_data_path)

        from nltk.corpus import cmudict
        cmu = cmudict.dict()
        from g2p_en import G2p
        g2p = G2p()

        # 1. Load N-gram (monophone) embeddings and vocabulary
        ngram_embs = np.load(self.ngram_embeddings_npy.get_path())
        with open(self.ngram_vocab_file.get_path(), "r", encoding="utf-8") as f:
            ngram_vocab_list = [line.strip().split()[0] for line in f if line.strip()]
        ngram_to_idx = {ng: i for i, ng in enumerate(ngram_vocab_list)}
        logging.info(f"Loaded {len(ngram_to_idx)} n-gram embeddings of dim {ngram_embs.shape[1]}")

        # 2. Extract word vocabulary and counts
        open_fn = gzip.open if self.word_text_file.get_path().endswith(".gz") else open
        word_counts = Counter()
        with open_fn(self.word_text_file.get_path(), "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    word_counts.update(tokens)

        # Candidate words to cluster: top max_input_words with count >= min_count
        candidates = [w for w, c in word_counts.most_common() if c >= self.min_count]
        if len(candidates) < self.num_clusters:
            candidates = [w for w, _ in word_counts.most_common(self.num_clusters)]
        vocab_list = candidates[: self.max_input_words]
        V = len(vocab_list)
        logging.info(f"Extracted {V} candidate words for clustering into K={self.num_clusters}...")

        # 3. Phonemize words and compute averaged monophone representations
        cache: Dict[str, List[str]] = {}

        def word_to_phones(word: str) -> List[str]:
            if word in cache:
                return cache[word]
            w_clean = re.sub(r"[^a-zA-Z0-9\']", "", word).lower()
            if not w_clean:
                cache[word] = []
                return []
            if w_clean in cmu:
                phones = [re.sub(r"\d+", "", p) for p in cmu[w_clean][0]]
            else:
                try:
                    res = g2p(w_clean)
                    phones = [re.sub(r"\d+", "", p) for p in res if p.strip() and p.isalnum()]
                except Exception:
                    phones = []
            cache[word] = phones
            return phones

        n = self.ngram_order
        delim = self.delimiter
        embedding_matrix = np.zeros((V, self.vector_size), dtype=np.float32)

        for idx, word in enumerate(vocab_list):
            phones = word_to_phones(word)
            if len(phones) == 0:
                word_ngrams = []
            elif len(phones) < n:
                word_ngrams = [delim.join(phones)]
            else:
                word_ngrams = [delim.join(phones[i : i + n]) for i in range(len(phones) - n + 1)]

            valid_vecs = [ngram_embs[ngram_to_idx[ng]] for ng in word_ngrams if ng in ngram_to_idx]
            if valid_vecs:
                avg_vec = np.mean(valid_vecs, axis=0)
                if self.normalize:
                    norm = np.linalg.norm(avg_vec)
                    if norm > 1e-12:
                        avg_vec = avg_vec / norm
                embedding_matrix[idx] = avg_vec
            else:
                rng = np.random.RandomState(self.seed + idx)
                embedding_matrix[idx] = rng.randn(self.vector_size).astype(np.float32) * 0.01

        # 4. Fit MiniBatchKMeans
        effective_k = min(self.num_clusters, V)
        logging.info(f"Fitting MiniBatchKMeans with K={effective_k} on {V} word representations...")
        kmeans = MiniBatchKMeans(
            n_clusters=effective_k,
            random_state=self.seed,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            n_init=1,
        )
        cluster_labels = kmeans.fit_predict(embedding_matrix)
        centroids = kmeans.cluster_centers_.astype(np.float32)
        np.save(self.out_centroids_npy.get_path(), centroids)

        # 5. Build word_to_cluster and cluster_to_word mappings
        word_to_cluster = {}
        cluster_word_candidates = {k: [] for k in range(effective_k)}
        for idx, word in enumerate(vocab_list):
            c_id = int(cluster_labels[idx])
            c_tok = f"c{c_id}"
            word_to_cluster[word] = c_tok
            cluster_word_candidates[c_id].append((word, word_counts.get(word, 0)))

        # Canonical representative word for each cluster (highest frequency count)
        cluster_to_word = {}
        for c_id in range(effective_k):
            c_tok = f"c{c_id}"
            c_words = cluster_word_candidates[c_id]
            if c_words:
                c_words.sort(key=lambda item: item[1], reverse=True)
                cluster_to_word[c_tok] = c_words[0][0]
            else:
                cluster_to_word[c_tok] = f"<clus_{c_id}>"

        with open(self.out_word_to_cluster_json.get_path(), "w", encoding="utf-8") as f_w2c:
            json.dump(word_to_cluster, f_w2c, ensure_ascii=False)
        with open(self.out_cluster_to_word_json.get_path(), "w", encoding="utf-8") as f_c2w:
            json.dump(cluster_to_word, f_c2w, ensure_ascii=False)

        # 6. Quantize the text corpus into cluster sequences
        cluster_token_counts = Counter()
        fallback_cluster = "c0"

        with open_fn(self.word_text_file.get_path(), "rt", encoding="utf-8", errors="replace") as f_in, \
             open(self.out_cluster_text.get_path(), "w", encoding="utf-8") as f_out:
            for line in f_in:
                tokens = line.strip().split()
                c_seq = [word_to_cluster.get(t, fallback_cluster) for t in tokens]
                f_out.write(" ".join(c_seq) + "\n")
                cluster_token_counts.update(c_seq)

        # 7. Write cluster vocabulary and counts files
        with open(self.out_cluster_vocab_txt.get_path(), "w", encoding="utf-8") as f_vocab, \
             open(self.out_cluster_counts_txt.get_path(), "w", encoding="utf-8") as f_counts:
            for c_id in range(effective_k):
                c_tok = f"c{c_id}"
                f_vocab.write(f"{c_tok}\n")
                cnt = cluster_token_counts.get(c_tok, 0)
                f_counts.write(f"{c_tok} {cnt}\n")

        logging.info(
            f"Clustering complete. Saved quantized corpus with {len(cluster_token_counts)} unique clusters "
            f"to {self.out_cluster_text.get_path()}"
        )


class ApplyWordToClusterJob(Job):
    """
    Quantizes an evaluation text file of words into cluster tokens using word_to_cluster.json.
    """
    _version = 1

    def __init__(
        self,
        input_text_file: tk.Path,
        word_to_cluster_json: tk.Path,
        fallback_cluster: str = "c0",
    ):
        super().__init__()
        self.input_text_file = input_text_file
        self.word_to_cluster_json = word_to_cluster_json
        self.fallback_cluster = fallback_cluster
        self.out_cluster_text = self.output_path("quantized_clusters.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 2, "mem": 4, "time": 1})

    def run(self):
        import json
        with open(self.word_to_cluster_json.get_path(), "r", encoding="utf-8") as f:
            w2c = json.load(f)

        with open(self.input_text_file.get_path(), "r", encoding="utf-8", errors="replace") as f_in, \
             open(self.out_cluster_text.get_path(), "w", encoding="utf-8") as f_out:
            for line in f_in:
                tokens = line.strip().split()
                c_seq = [w2c.get(t, self.fallback_cluster) for t in tokens]
                f_out.write(" ".join(c_seq) + "\n")


class MapClustersToWordsJob(Job):
    """
    Maps cluster tokens back to English words using cluster_to_word.json for BLEU evaluation.
    Exposes both out_word_text and out_debpe_text for seamless integration with BLEU scoring.
    """
    _version = 1

    def __init__(
        self,
        hyp_file: tk.Path,
        cluster_to_word_json: tk.Path,
        fallback_word: str = "<unk>",
    ):
        super().__init__()
        self.hyp_file = hyp_file
        self.cluster_to_word_json = cluster_to_word_json
        self.fallback_word = fallback_word
        self.out_word_text = self.output_path("hypotheses.word_level.txt")
        self.out_debpe_text = self.out_word_text

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 1, "mem": 2, "time": 1})

    def run(self):
        import json
        with open(self.cluster_to_word_json.get_path(), "r", encoding="utf-8") as f:
            c2w = json.load(f)

        with open(self.hyp_file.get_path(), "r", encoding="utf-8", errors="replace") as f_in, \
             open(self.out_word_text.get_path(), "w", encoding="utf-8") as f_out:
            for line in f_in:
                tokens = line.strip().split()
                w_seq = [c2w.get(t, self.fallback_word) for t in tokens]
                f_out.write(" ".join(w_seq) + "\n")

