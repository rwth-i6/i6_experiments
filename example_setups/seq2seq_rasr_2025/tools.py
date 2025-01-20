from i6_core.lm.kenlm import CompileKenLMJob
from i6_core.tools.compile import MakeJob
from i6_core.tools.git import CloneGitRepositoryJob
from sisyphus import tk

from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
from i6_experiments.common.tools.sctk import compile_sctk

returnn_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn.git", checkout_folder_name="returnn"
).out_repository
returnn_root.hash_overwrite = "RETURNN_ROOT"

minireturnn_root = CloneGitRepositoryJob(
    "https://github.com/JackTemaki/MiniReturnn.git", checkout_folder_name="returnn"
).out_repository
minireturnn_root.hash_overwrite = "MINIRETURNN_ROOT"

returnn_python_exe = tk.Path("/usr/bin/python3")

rasr_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/rasr.git",
    branch="seq2seq-revamp",
    checkout_folder_name="rasr",
).out_repository
rasr_root.hash_overwrite = "RASR_ROOT"

rasr_binary_path: tk.Path = MakeJob(
    folder=rasr_root,
    make_sequence=["build", "install"],
    configure_opts=["--apptainer-setup=2024-11-06_onnx-1.16_v1"],
    num_processes=8,
    link_outputs={"binaries": "arch/linux-x86_64-standard/"},
).out_links["binaries"]
rasr_binary_path.hash_overwrite = "RASR_BINARY_PATH"

sctk_binary_path = compile_sctk(branch="v2.4.12")
sctk_binary_path.hash_overwrite = "SCTK_BINARY_PATH"

ken_lm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm.git").out_repository
ken_lm_repo.hash_overwrite = "KENLM"
ken_lm_binaries = CompileKenLMJob(repository=ken_lm_repo).out_binaries

subword_nmt_repo = get_returnn_subword_nmt(commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf")
subword_nmt_repo.hash_overwrite = "I6_SUBWORD_NMT_V2"
