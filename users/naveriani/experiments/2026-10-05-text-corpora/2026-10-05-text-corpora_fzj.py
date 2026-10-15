import json
import os
import subprocess
from urllib.parse import urlparse

from sisyphus import Job, Task, tk

from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJobV2,
    get_content_dir_from_hub_cache_dir,
)
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module


def py():
    prefix = get_setup_prefix_for_module(__name__)
    _data(prefix)


HF_REPOS = {  # key -> (repo id, repo type, include globs)
    "dolma": ("allenai/dolma", "dataset", ["urls/*"]),  # the repo has only url lists, the data is on olmo-data.org
}
DOLMA_VERSION = "v1_7"
DOLMA_SOURCES = (  # natural-language text only (4.20 TB), no code, math or LaTeX sources
    "books",
    "c4-filtered",
    "cc_en_head",
    "cc_en_middle",
    "cc_en_tail",
    "cc_news_head",
    "cc_news_middle",
    "cc_news_tail",
    "falcon-refinedweb-filtered",
    "pes2o",
    "reddit",
    "wiki",
    "wikiref_megawika",
)
DOLMA_OUT = "/e/scratch/spell/common_raw/dolma"  # shared raw downloads, scratch is wiped after 30 days
NUM_SHARDS = 8  # parallel download tasks on the login node


def _data(prefix, *, register=True):
    # CPU-only tasks run on the login node, which has internet
    hub = {}
    for k, (rid, rt, include) in HF_REPOS.items():
        hub[k] = DownloadHuggingFaceRepoJobV2(repo_id=rid, repo_type=rt, include=include)
        hub[k].add_alias(f"{prefix}/data/{k}/download")
    dolma = DolmaDownloadJob(hub["dolma"].out_hub_cache_dir, version=DOLMA_VERSION, sources=DOLMA_SOURCES,
                             out_root=DOLMA_OUT)
    dolma.add_alias(f"{prefix}/data/dolma/{DOLMA_VERSION}")
    if register:
        tk.register_output(f"{prefix}/data/dolma/{DOLMA_VERSION}/stats.json", dolma.out_stats)
    return {"dolma": dolma.out_dir}


def _source(url):
    # https://olmo-data.org/dolma-v1_7/<source>/<file>.json.gz
    return urlparse(url).path.split("/")[-2]


def _remote_size(url):
    r = subprocess.run(["curl", "-sIL", "--max-time", "60", url], stdout=subprocess.PIPE, check=False)
    n = [int(l.split(":", 1)[1]) for l in r.stdout.decode("utf8", "replace").splitlines()
         if l.lower().startswith("content-length:")]
    return n[-1] if n else None


class DolmaDownloadJob(Job):
    # downloads the files of the url list to out_root, a rerun resumes .part files and skips finished ones
    def __init__(self, hub_cache_dir, *, version=DOLMA_VERSION, sources=None, num_shards=NUM_SHARDS, out_root=None):
        super().__init__()
        self.hub_cache_dir = hub_cache_dir
        self.version = version
        self.sources = tuple(sources) if sources else None
        self.num_shards = num_shards
        self.out_root = out_root
        self.out_dir = self.output_path("data")  # a symlink when out_root is set
        self.out_shard_stats = [self.output_path(f"stats-{i:03d}.json") for i in range(num_shards)]
        self.out_stats = self.output_path("stats.json")
        self.rqmt = {"cpu": 1, "mem": 2, "time": 24}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, args=[[i] for i in range(self.num_shards)])
        yield Task("stats", mini_task=True)

    def _urls(self):
        fn = os.path.join(get_content_dir_from_hub_cache_dir(self.hub_cache_dir), "urls", f"{self.version}.txt")
        urls = [l.strip() for l in open(fn) if l.strip()]
        if self.sources:
            missing = set(self.sources) - {_source(u) for u in urls}
            assert not missing, f"unknown sources {missing}"
            urls = [u for u in urls if _source(u) in self.sources]
        return urls

    def run(self, shard):
        root = self.out_root or self.out_dir.get_path()
        stats = {}
        for u in self._urls()[shard :: self.num_shards]:
            dst, n = os.path.join(root, urlparse(u).path.lstrip("/")), _remote_size(u)
            if not (os.path.exists(dst) and (n is None or os.path.getsize(dst) == n)):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                subprocess.run(["curl", "-sS", "-L", "-C", "-", "--retry", "20", "--retry-all-errors",
                                "--retry-delay", "10", "-o", dst + ".part", u], check=True)
                assert n is None or os.path.getsize(dst + ".part") == n, f"{u}: size != remote {n}"
                os.replace(dst + ".part", dst)
            s = stats.setdefault(_source(u), {"files": 0, "bytes": 0})
            s["files"] += 1
            s["bytes"] += os.path.getsize(dst)
        with open(self.out_shard_stats[shard].get_path(), "w") as f:
            json.dump(stats, f, indent=1)

    def stats(self):
        # link out_root into the job output
        if self.out_root and not os.path.exists(self.out_dir.get_path()):
            os.symlink(self.out_root, self.out_dir.get_path())
        tot = {}
        for p in self.out_shard_stats:
            for src, s in json.load(open(p.get_path())).items():
                t = tot.setdefault(src, {"files": 0, "bytes": 0})
                t["files"] += s["files"]
                t["bytes"] += s["bytes"]
        with open(self.out_stats.get_path(), "w") as f:
            json.dump({"version": self.version, "files": sum(t["files"] for t in tot.values()),
                       "bytes": sum(t["bytes"] for t in tot.values()), "sources": tot}, f, indent=1)
