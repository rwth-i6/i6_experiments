from sisyphus import tk

from i6_experiments.users.naveriani.jobs.datasets import (
    DownloadUrlsJob,
    ExtractDocsJob,
    LinkFilesJob,
    NormalizeTextJob,
    RandomFileSubsetJob,
    ScoreDocsJob,
    SelectTextJob,
    TrainFastTextJob,
)
from i6_experiments.users.zeyer.datasets.loquacious import get_hf_text_only
from i6_experiments.users.zeyer.external_models.huggingface import DownloadHuggingFaceRepoJobV2
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module


def py():
    # RZ: the CPU-heavy steps; their inputs are finished FZJ job folders copied here (docs, pool files, selectors).
    # No outputs of the upstream jobs are registered, so sis never starts the Dolma download on RZ
    prefix = get_setup_prefix_for_module(__name__)
    dolma_ = dolma(prefix)
    subset, docs = dolma_random(prefix, dolma_)
    books = gutenberg(prefix, dolma_)
    for name, d in {"dolma-random-10x": docs, "gutenberg": books}.items():
        norm = normalize(prefix, name, d)
        tk.register_output(f"{prefix}/data/{name}/stats.json", norm.out_stats)
    sel = select(prefix, *dolma_pool(prefix, dolma_), selectors(prefix, dolma_, subset), books, docs)
    tk.register_output(f"{prefix}/data/dolma-selected-10x/stats.json", sel.out_stats)
    norm = normalize(prefix, "dolma-selected-10x", sel)
    tk.register_output(f"{prefix}/data/dolma-selected-10x/norm-stats.json", norm.out_stats)


def dolma(prefix):
    # the HF repo has only the url lists, the data is on olmo-data.org
    urls = DownloadHuggingFaceRepoJobV2(repo_id="allenai/dolma", repo_type="dataset", include=["urls/*"])
    urls.add_alias(f"{prefix}/data/dolma/urls")
    # natural-language text only (~4.2 TB gz), no code, math or LaTeX
    srcs = [
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
    ]
    # https://olmo-data.org/dolma-v1_7/<source>/<file>.json.gz -> dolma/v1_7/<source>/<file>.json.gz
    dolma = DownloadUrlsJob(
        (urls.out_hub_cache_dir, "urls/v1_7.txt"), name="dolma/v1_7", include=[f"*/{s}/*" for s in srcs], strip=1
    )
    dolma.add_alias(f"{prefix}/data/dolma/v1_7")
    return dolma


def dolma_random(prefix, dolma):
    # random Dolma documents: a random fraction of 128 files (weighted by size) with 10x the transcript words.
    # All document sets drop documents sharing >= 5 10-grams with the Loquacious dev/test transcripts
    subset = RandomFileSubsetJob(dolma.out_dir, num_words=10 * 235_900_000)
    subset.add_alias(f"{prefix}/data/dolma-random-10x/files")
    docs = ExtractDocsJob(dolma.out_dir, subset.out_files, subset_stats=subset.out_stats, eval_text=get_hf_text_only())
    docs.add_alias(f"{prefix}/data/dolma-random-10x/docs")
    return subset, docs


def gutenberg(prefix, dolma):
    # all Dolma books = Project Gutenberg (56k books, ~3.5B words); LibriSpeech dev/test are chapters of such books
    docs = ExtractDocsJob(
        dolma.out_dir, [f"books/books-{i:04d}.json.gz" for i in range(3)], eval_text=get_hf_text_only(), num_shards=48
    )
    docs.add_alias(f"{prefix}/data/gutenberg/docs")
    return docs


def selectors(prefix, dolma, subset):
    # source selectors: the transcripts of one Loquacious source vs random Dolma sentences (no books, no papers).
    # LibriHeavy (book detector) probe 2026-10-16: valid AUC 0.981, mean P(book) LibriHeavy 0.94 vs 0.17-0.29 for the
    # other sources; held-out Gutenberg books: 97.5 % in the top 5 % of web docs (doc level)
    sel = {}
    for src in ["libriheavy", "yodas", "peoplesspeech", "commonvoice", "voxpopuli"]:
        sel[src] = TrainFastTextJob(get_hf_text_only(), dolma.out_dir, subset.out_files, source=src)
        sel[src].add_alias(f"{prefix}/data/fasttext/{src}")
    return sel


def dolma_pool(prefix, dolma):
    # ~150B words of random web files (no books) to select from; the folder of links is what gets copied to RZ
    pool = RandomFileSubsetJob(dolma.out_dir, num_words=150_000_000_000, num_files=200, seed=2, exclude=["books/"])
    pool.add_alias(f"{prefix}/data/dolma-pool/files")
    files = LinkFilesJob(dolma.out_dir, pool.out_files)
    files.add_alias(f"{prefix}/data/dolma-pool/link")
    return pool, files


def select(prefix, pool, files, sel, books, random_docs):
    # Probe 2026-10-16: single sentences are too noisy, 5-sentence windows pick the right text (hearings for
    # People's Speech, video transcripts for YODAS, wiki-like sentences for CommonVoice, EU politics for VoxPopuli);
    # books whole documents. Same word count as the random set, split like the Loquacious sources (M words)
    score = ScoreDocsJob(
        files.out_dir,
        pool.out_files,
        {k: j.out_model for k, j in sel.items()},
        window=5,
        doc_min_p=0.6,
        num_tasks=20,
    )
    score.rqmt.update({"cpu": 96, "mem": 120, "time": 4})
    score.add_alias(f"{prefix}/data/dolma-pool/scores")
    shares = {"libriheavy": 104.0, "peoplesspeech": 58.4, "yodas": 57.9, "commonvoice": 10.9, "voxpopuli": 4.7}
    sel_job = SelectTextJob(
        score.out_scores, shares, random_docs.out_stats, fill_docs=books.out_docs, eval_text=get_hf_text_only()
    )
    sel_job.add_alias(f"{prefix}/data/dolma-selected-10x/select")
    return sel_job


def normalize(prefix, name, docs):
    # Albert's TextNormaliser (NeMo), like the Loquacious transcripts; 171 words/s/core, so on RZ CPU nodes
    norm = NormalizeTextJob(docs.out_docs)
    norm.add_alias(f"{prefix}/data/{name}/normalize")
    return norm
