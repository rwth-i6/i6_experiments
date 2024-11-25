import dataclasses

from sisyphus import tk

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import DecodingTensorMap

# used for hardcoded alignments later

TRI_GMM_ALIGNMENT = tk.Path(
    "/work/common/asr/tedliumv2/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.1cl0zXQN6iBG/output/alignment.cache.bundle",
    cached=True,
)
GMM_ALLOPHONES = tk.Path(
    "/work/common/asr/tedliumv2/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.GP1idt9FdeWe/output/allophones",
    cached=True,
)
