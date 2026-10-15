from sisyphus import tk

from i6_experiments.users.naveriani.jobs.datasets import DownloadUrlsJob
from i6_experiments.users.zeyer.external_models.huggingface import DownloadHuggingFaceRepoJobV2
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module


def py():
    prefix = get_setup_prefix_for_module(__name__)
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
    tk.register_output(f"{prefix}/data/dolma/v1_7/stats.json", dolma.out_stats)
