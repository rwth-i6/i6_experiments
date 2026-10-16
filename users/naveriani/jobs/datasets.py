import fnmatch
import glob
import gzip
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
from multiprocessing import Pool
from urllib.parse import urlparse

from sisyphus import Job, Task, gs, tk

from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir


def _remote_size(url):
    r = subprocess.run(["curl", "-sIL", "--max-time", "60", url], stdout=subprocess.PIPE, check=False)
    n = [
        int(ln.split(":", 1)[1])
        for ln in r.stdout.decode("utf8", "replace").splitlines()
        if ln.lower().startswith("content-length:")
    ]
    return n[-1] if n else None


def _link_dir(out, root, name):
    # job output dir as a symlink to root/name (root read from gs at run time, so not hashed), a plain dir without root
    if os.path.lexists(out):
        return
    if root:
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        os.symlink(d, out)
    else:
        os.makedirs(out)


def _scratch_link(job):
    # intermediate text data: gs.SCRATCH_DIR/text-corpora/<job name>
    root = getattr(gs, "SCRATCH_DIR", None)
    _link_dir(job.out_dir.get_path(), root and f"{root}/text-corpora", os.path.basename(job.job_id()))


class DownloadUrlsJob(Job):
    # urls: list of urls, tk.Path to a url list file, or (hub_cache_dir, rel path) for a list inside an HF repo
    # name: folder under gs.RAW_DATA_DIR, include: fnmatch globs on the url path, strip: leading path parts to drop
    def __init__(self, urls, *, name, include=None, strip=0, num_shards=8):
        super().__init__()
        self.urls = urls
        self.name = name
        self.include = include
        self.strip = strip
        self.num_shards = num_shards
        self.out_dir = self.output_path("data")  # a symlink into gs.RAW_DATA_DIR if set
        self.out_shard_stats = [self.output_path(f"stats-{i:03d}.json") for i in range(num_shards)]
        self.out_stats = self.output_path("stats.json")
        self.rqmt = {"cpu": 1, "mem": 2, "time": 24}

    def tasks(self):
        yield Task("link", mini_task=True)
        yield Task("run", rqmt=self.rqmt, args=[[i] for i in range(self.num_shards)])
        yield Task("stats", mini_task=True)

    def link(self):
        _link_dir(self.out_dir.get_path(), getattr(gs, "RAW_DATA_DIR", None), self.name)

    def _urls(self):
        if isinstance(self.urls, tuple):
            fn = os.path.join(get_content_dir_from_hub_cache_dir(self.urls[0]), self.urls[1])
        elif isinstance(self.urls, tk.Path):
            fn = self.urls.get_path()
        else:
            return list(self.urls)
        urls = [ln.strip() for ln in open(fn) if ln.strip()]
        if self.include:
            for p in self.include:
                assert any(fnmatch.fnmatch(urlparse(u).path, p) for u in urls), f"no url matches {p!r}"
            urls = [u for u in urls if any(fnmatch.fnmatch(urlparse(u).path, p) for p in self.include)]
        return urls

    def _rel(self, url):
        return "/".join(urlparse(url).path.lstrip("/").split("/")[self.strip :])

    def run(self, shard):
        stats = {}
        for u in self._urls()[shard :: self.num_shards]:
            rel = self._rel(u)
            dst, n = os.path.join(self.out_dir.get_path(), rel), _remote_size(u)
            if not (os.path.exists(dst) and (n is None or os.path.getsize(dst) == n)):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                curl = "curl -sS -L -C - --retry 20 --retry-all-errors --retry-delay 10".split()
                subprocess.run(curl + ["-o", dst + ".part", u], check=True)
                assert n is None or os.path.getsize(dst + ".part") == n, f"{u}: size != remote {n}"
                os.replace(dst + ".part", dst)
            # stats per top-level folder, e.g. the dolma source
            s = stats.setdefault(rel.split("/")[0] if "/" in rel else ".", {"files": 0, "bytes": 0})
            s["files"] += 1
            s["bytes"] += os.path.getsize(dst)
        with open(self.out_shard_stats[shard].get_path(), "w") as f:
            json.dump(stats, f, indent=1)

    def stats(self):
        tot = {}
        for p in self.out_shard_stats:
            for k, s in json.load(open(p.get_path())).items():
                t = tot.setdefault(k, {"files": 0, "bytes": 0})
                t["files"] += s["files"]
                t["bytes"] += s["bytes"]
        files, size = sum(t["files"] for t in tot.values()), sum(t["bytes"] for t in tot.values())
        with open(self.out_stats.get_path(), "w") as f:
            json.dump({"name": self.name, "files": files, "bytes": size, "folders": tot}, f, indent=1)


def _est_words(fn, num_docs=1000):
    # words of the whole file, from the words per compressed byte of its first documents
    with open(fn, "rb") as raw, gzip.open(raw, "rt") as f:
        n = 0
        for i, ln in enumerate(f):
            n += len(json.loads(ln)["text"].split())
            if i + 1 == num_docs:
                break
        return round(n * os.path.getsize(fn) / raw.tell())


class RandomFileSubsetJob(Job):
    # num_files random .json.gz files of a jsonl dataset dir, drawn with probability ~ file size (Efraimidis-Spirakis
    # keys), and the document keep fraction for num_words words, estimated from the first 1000 documents of each file
    def __init__(self, data_dir, *, num_words, num_files=128, seed=1):
        super().__init__()
        self.data_dir = data_dir
        self.num_words = num_words
        self.num_files = num_files
        self.seed = seed
        self.out_dir = self.output_path("data")  # a symlink into gs.SCRATCH_DIR if set
        self.out_files = self.out_dir.join_right("files.txt")
        self.out_stats = self.out_dir.join_right("stats.json")
        self.rqmt = {"cpu": 1, "mem": 2, "time": 2}

    def tasks(self):
        yield Task("link", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def link(self):
        _scratch_link(self)

    def run(self):
        root = self.data_dir.get_path()
        files = sorted(glob.glob(f"{root}/**/*.json.gz", recursive=True))
        rng = random.Random(self.seed)
        files.sort(key=lambda fn: -math.log(rng.random()) / os.path.getsize(fn))
        est = {os.path.relpath(fn, root): _est_words(fn) for fn in files[: self.num_files]}
        keep = min(1.0, self.num_words / sum(est.values()))
        with open(self.out_files.get_path(), "w") as f:
            f.write("".join(fn + "\n" for fn in est))
        with open(self.out_stats.get_path(), "w") as f:
            json.dump({"files": len(est), "est_words": sum(est.values()), "keep": keep, "per_file": est}, f, indent=1)


# sentence ends inside a paragraph, before an (optionally quoted) capital or digit
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(\[]?[A-Z0-9])")
_norm = None


def _sentences(text, max_words):
    # paragraphs at blank lines; single line breaks are hard wraps (books) and get joined
    for para in re.split(r"\n\s*\n", text):
        for s in _SENT_END.split(" ".join(para.split())):
            w = s.split()
            for i in range(0, len(w), max_words):
                yield " ".join(w[i : i + max_words])


def _recipe_on_path():
    # sis finds recipe modules via its RecipeFinder and runs tasks in the job dir: fresh imports of i6_core (also in
    # pool workers) need the recipe dir on sys.path as an absolute path
    import i6_experiments

    sys.path.insert(0, os.path.dirname(os.path.abspath(list(i6_experiments.__path__)[0])))


def _init_norm():
    # one TextNormaliser per worker, its per-sentence drop warnings muted (the counts are in the stats)
    global _norm
    from i6_experiments.users.zeyer.datasets.loquacious import TextNormaliser

    _norm = TextNormaliser()
    logging.getLogger("i6_experiments.users.zeyer.datasets.loquacious").setLevel(logging.ERROR)


def _normalize_doc(args):
    text, max_words = args
    sents = list(_sentences(text, max_words))
    out = [o for o in map(_norm, sents) if o]
    return out, len(sents), sum(len(s.split()) for s in sents)


_grams = None


def _init_grams(grams):
    global _grams
    _grams = grams


def _contaminated(text, max_hits, n=10):
    # >= max_hits word n-grams shared with the eval transcripts (cheap-normalised on both sides)
    w, hits = _cheap_norm(text).split(), 0
    for i in range(len(w) - n + 1):
        if tuple(w[i : i + n]) in _grams:
            hits += 1
            if hits >= max_hits:
                return True
    return False


def _check_doc(args):
    rel, text, max_hits = args
    return rel, text, bool(_grams) and _contaminated(text, max_hits)


class ExtractDocsJob(Job):
    # documents of a jsonl dataset dir: the given files (list, or files.txt + keep fraction of a RandomFileSubsetJob),
    # minus documents sharing >= max_hits 10-grams with the eval transcripts (decontamination), round-robin into
    # num_shards jsonl.gz shards, so the normalisation can run on another cluster
    def __init__(self, data_dir, files, *, subset_stats=None, eval_text=None, max_hits=5, num_shards=32, seed=1):
        super().__init__()
        self.data_dir = data_dir
        self.files = files
        self.subset_stats = subset_stats
        self.eval_text = eval_text
        self.max_hits = max_hits
        self.num_shards = num_shards
        self.seed = seed
        self.out_dir = self.output_path("data")  # a symlink into gs.SCRATCH_DIR if set
        self.out_docs = [self.out_dir.join_right(f"docs-{i:03d}.jsonl.gz") for i in range(num_shards)]
        self.out_stats = self.out_dir.join_right("stats.json")
        self.rqmt = {"cpu": 3, "mem": 16, "time": 6}

    def tasks(self):
        yield Task("link", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def link(self):
        _scratch_link(self)

    def _eval_grams(self, n=10):
        # word n-grams of the dev + test transcripts of an HF text dataset (get_hf_text_only)
        import datasets

        grams = set()
        for split in ["dev", "test"]:
            for t in datasets.load_from_disk(self.eval_text.get_path())[split]["text"]:
                w = _cheap_norm(t).split()
                grams.update(tuple(w[i : i + n]) for i in range(len(w) - n + 1))
        return grams

    def _docs(self, files, keep):
        # every file's documents with probability keep, the same draw on every rerun
        for rel in files:
            rng = random.Random(f"{self.seed}/{rel}")
            with gzip.open(os.path.join(self.data_dir.get_path(), rel), "rt") as f:
                for ln in f:
                    if rng.random() < keep:
                        yield rel, json.loads(ln)["text"], self.max_hits

    def run(self):
        files = self.files
        if isinstance(files, tk.Path):
            files = open(files.get_path()).read().split()
        keep = json.load(open(self.subset_stats.get_path()))["keep"] if self.subset_stats else 1.0
        grams = self._eval_grams() if self.eval_text else None
        outs = [gzip.open(p.get_path(), "wt") for p in self.out_docs]
        st, i = {}, 0
        with Pool(self.rqmt["cpu"], initializer=_init_grams, initargs=(grams,)) as pool:
            for rel, t, bad in pool.imap(_check_doc, self._docs(files, keep), chunksize=16):
                s = st.setdefault(rel.split("/")[0], {"docs": 0, "words": 0, "dropped": 0})
                if bad:
                    s["dropped"] += 1
                    continue
                outs[i % self.num_shards].write(json.dumps({"text": t}) + "\n")
                s["docs"] += 1
                s["words"] += len(t.split())
                i += 1
        for f in outs:
            f.close()
        tot = {"docs": sum(s["docs"] for s in st.values()), "words": sum(s["words"] for s in st.values())}
        with open(self.out_stats.get_path(), "w") as f:
            json.dump({**tot, "sources": st}, f, indent=1)


class NormalizeTextJob(Job):
    # jsonl documents -> sentences (split at max_words) -> TextNormaliser (NeMo), one line per sentence;
    # one task per docs shard on a full CPU node (RZ c23ms, ~1.3 h for 74M words)
    def __init__(self, docs, *, max_words=40):
        super().__init__()
        self.docs = docs
        self.max_words = max_words
        self.out_dir = self.output_path("data")  # a symlink into gs.SCRATCH_DIR if set
        self.out_texts = [self.out_dir.join_right(f"text-{i:03d}.txt.gz") for i in range(len(docs))]
        self.out_task_stats = [self.out_dir.join_right(f"stats-{i:03d}.json") for i in range(len(docs))]
        self.out_stats = self.out_dir.join_right("stats.json")
        self.rqmt = {"cpu": 96, "mem": 96, "time": 4}  # ~0.45 GB per NeMo worker

    def tasks(self):
        yield Task("link", mini_task=True)
        yield Task("run", rqmt=self.rqmt, args=[[i] for i in range(len(self.docs))])
        yield Task("stats", mini_task=True)

    def link(self):
        _scratch_link(self)

    def _docs(self, task):
        with gzip.open(self.docs[task].get_path(), "rt") as f:
            for ln in f:
                yield json.loads(ln)["text"], self.max_words

    def run(self, task):
        # the normaliser must import before the pool starts, else every pool worker dies and is restarted forever
        _recipe_on_path()
        from i6_experiments.users.zeyer.datasets.loquacious import TextNormaliser  # noqa: F401

        st = {"docs": 0, "sents_in": 0, "words_in": 0, "sents_out": 0, "words_out": 0}
        with Pool(self.rqmt["cpu"], initializer=_init_norm) as pool:
            with gzip.open(self.out_texts[task].get_path(), "wt") as f:
                for out, n_sents, n_words in pool.imap(_normalize_doc, self._docs(task), chunksize=8):
                    f.write("".join(o + "\n" for o in out))
                    st["docs"] += 1
                    st["sents_in"] += n_sents
                    st["words_in"] += n_words
                    st["sents_out"] += len(out)
                    st["words_out"] += sum(len(o.split()) for o in out)
                    if st["docs"] % 100_000 == 0:
                        print(st, flush=True)
        with open(self.out_task_stats[task].get_path(), "w") as f:
            json.dump(st, f, indent=1)

    def stats(self):
        tot = {}
        for p in self.out_task_stats:
            for k, v in json.load(open(p.get_path())).items():
                tot[k] = tot.get(k, 0) + v
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(tot, f, indent=1)


# hard filters: no URLs / code / formulas, a sentence end, >= 95 % letters or digits, <= 20 % digits
_URL = re.compile(r"https?://|www\.|\b\w+\.(com|org|net|html?|php)\b|@\w+\.\w+", re.I)
_CODE = re.compile(r"[{}<>|\\=]|;\s*$")
_END = re.compile(r"[.!?][\"'”’)\]]*$")
_BULLET = re.compile(r"^\s*(?:[#*•\-–]+|\d+[.)])\s+")
_PUNCT = set(".,!?;:'\"-–—()’‘“”")


def _keep_sentence(s):
    if _URL.search(s) or _CODE.search(s) or not _END.search(s):
        return False
    chars = [c for c in s if not c.isspace() and c not in _PUNCT]
    letters = sum(c.isascii() and c.isalpha() for c in chars)
    digits = sum(c.isdigit() for c in chars)
    return bool(chars) and (letters + digits) / len(chars) >= 0.95 and digits / len(chars) <= 0.2


def _cheap_norm(s):
    # transcript-like surface for scoring (uppercase, A-Z and '), not the TextNormaliser
    s = s.upper().replace("’", "'").replace("‘", "'")
    return " ".join(re.sub(r"[^A-Z' ]", " ", s).split())


def _doc_sents(text, min_words=8, max_words=40):
    # a document's filtered, cheap-normalised sentences, long ones cut at max_words
    out = []
    for para in re.split(r"\n\s*\n", text):
        for s in _SENT_END.split(" ".join(para.split())):
            s = _BULLET.sub("", s)
            if len(s.split()) >= min_words and _keep_sentence(s):
                w = _cheap_norm(s).split()
                out += [" ".join(w[i : i + max_words]) for i in range(0, len(w), max_words) if len(w[i:]) >= min_words]
    return out


def _auc(scores, labels):
    ranks = [r for r, (_, y) in enumerate(sorted(zip(scores, labels))) if y]
    n_pos, n_neg = len(ranks), len(scores) - len(ranks)
    return (sum(ranks) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)


class TrainFastTextJob(Job):
    # fastText classifier: Loquacious transcripts of one source (positives) vs random Dolma sentences (unlabelled
    # background, books and papers excluded), both cheap-normalised; the positives are recut to the background's
    # sentence lengths, so length is no cue. stats: held-out accuracy / AUC, mean P(pos) per Loquacious source
    def __init__(self, loq_text, data_dir, files, *, source, num=650_000, docs_per_file=4000, num_files=16, seed=1):
        super().__init__()
        self.loq_text = loq_text
        self.data_dir = data_dir
        self.files = files
        self.source = source
        self.num = num
        self.docs_per_file = docs_per_file
        self.num_files = num_files
        self.seed = seed
        self.out_dir = self.output_path("data")  # a symlink into gs.SCRATCH_DIR if set
        self.out_model = self.out_dir.join_right("model.bin")
        self.out_stats = self.output_path("stats.json")
        self.rqmt = {"cpu": 3, "mem": 32, "time": 2}

    def tasks(self):
        yield Task("link", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def link(self):
        _scratch_link(self)

    def run(self):
        import datasets
        import fasttext

        _recipe_on_path()
        from i6_experiments.users.zeyer.datasets.loquacious import loquacious_source_from_id

        rng = random.Random(self.seed)
        root, d = self.data_dir.get_path(), self.out_dir.get_path()
        files = [f for f in open(self.files.get_path()).read().split() if not f.startswith(("books/", "pes2o/"))]
        rng.shuffle(files)
        neg = []
        for rel in files[: self.num_files]:
            with gzip.open(os.path.join(root, rel), "rt") as f:
                for i, ln in enumerate(f):
                    if i == self.docs_per_file:
                        break
                    neg += _doc_sents(json.loads(ln)["text"])
        rng.shuffle(neg)
        neg = neg[: self.num]
        lens = [len(s.split()) for s in neg]

        # transcripts by source, from a random 2M sample of the train split
        tr = datasets.load_from_disk(self.loq_text.get_path())["train"]
        rows = tr.select(rng.sample(range(len(tr)), 2_000_000))
        by_src = {}
        for sid, text in zip(rows["id"], rows["text"]):
            by_src.setdefault(loquacious_source_from_id(sid), []).append(text)

        def recut(lines):
            for t in lines:
                w = t.split()
                while len(w) >= 8:
                    n = rng.choice(lens)
                    yield " ".join(w[:n])
                    w = w[n:]

        held = {src: lines[-20_000:] for src, lines in by_src.items()}  # never trained on
        pos = list(recut(by_src[self.source][:-20_000]))
        rng.shuffle(pos)
        data = [f"__label__pos {s}" for s in pos[: self.num]] + [f"__label__neg {s}" for s in neg]
        rng.shuffle(data)
        n_valid = min(100_000, len(data) // 10)
        valid, train = data[:n_valid], data[n_valid:]
        with open(f"{d}/train.txt", "w") as f:
            f.write("\n".join(train) + "\n")
        m = fasttext.train_supervised(
            f"{d}/train.txt", epoch=5, lr=0.5, wordNgrams=2, dim=64, minCount=2, thread=self.rqmt["cpu"], seed=self.seed
        )
        m.save_model(self.out_model.get_path())

        def p_pos(texts):
            labels, probs = m.predict(texts, k=2)
            return [float(dict(zip(lb, p)).get("__label__pos", 0.0)) for lb, p in zip(labels, probs)]

        vt = [ln.split(" ", 1) for ln in valid]
        vs = p_pos([t for _, t in vt])
        vy = [lb == "__label__pos" for lb, _ in vt]
        mean_p = {}
        for src, lines in held.items():
            sc = p_pos(list(recut(lines)))
            mean_p[src] = sum(sc) / max(1, len(sc))
        st = {
            "source": self.source,
            "pos": len(pos[: self.num]),
            "neg": len(neg),
            "valid_acc": sum((x > 0.5) == y for x, y in zip(vs, vy)) / len(vy),
            "valid_auc": _auc(vs, vy),
            "mean_p_pos_loquacious": mean_p,
        }
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(st, f, indent=1)
