from sisyphus import tk

from i6_experiments.users.naveriani.jobs.datasets import (
    DownloadUrlsJob,
    ExtractDocsJob,
    NormalizeTextJob,
    RandomFileSubsetJob,
)
from i6_experiments.users.zeyer.datasets.loquacious import get_hf_text_only
from i6_experiments.users.zeyer.external_models.huggingface import DownloadHuggingFaceRepoJobV2
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module


def py():
    # RZ: only the NeMo normalisation of the random Dolma documents, the docs job is a finished FZJ folder copied here.
    # No outputs of the upstream jobs are registered, so sis never starts the Dolma download on RZ
    prefix = get_setup_prefix_for_module(__name__)
    dolma_ = dolma(prefix)
    _, docs = dolma_random(prefix, dolma_)
    sets = {"dolma-random-10x": docs, "gutenberg": gutenberg(prefix, dolma_)}
    for name, docs in sets.items():
        norm = normalize(prefix, name, docs)
        tk.register_output(f"{prefix}/data/{name}/stats.json", norm.out_stats)


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


def normalize(prefix, name, docs):
    # Albert's TextNormaliser (NeMo), like the Loquacious transcripts; 171 words/s/core, so on RZ CPU nodes
    norm = NormalizeTextJob(docs.out_docs)
    norm.add_alias(f"{prefix}/data/{name}/normalize")
    return norm
