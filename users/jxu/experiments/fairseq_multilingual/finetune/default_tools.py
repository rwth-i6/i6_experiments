from i6_core.tools.git import CloneGitRepositoryJob
from sisyphus import tk

PACKAGE = __package__

FAIRSEQ = CloneGitRepositoryJob(
    "https://github.com/facebookresearch/fairseq",
    commit="100cd91db19bb27277a06a25eb4154c805b10189",
    checkout_folder_name="fairseq",
).out_repository
FAIRSEQ.hash_overwrite = "LIBRISPEECH_DEFAULT_FAIRSEQ"

tk.register_output("fairseq_repo", FAIRSEQ)
