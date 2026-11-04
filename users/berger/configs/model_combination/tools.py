from i6_core.lm.kenlm import CompileKenLMJob
from i6_core.tools.compile import CMakeJob
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

# TODO: When everything necessary is merged, switch to master branch, update apptainer-setup
# to torch-only image with CUDA enabled, update unhashed_package_roots
# to version with no variations
# And make all export returnn dims explicit

rasr_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/rasr.git",
    branch="berger_dev",
    checkout_folder_name="rasr",
).out_repository
rasr_root.hash_overwrite = "RASR_ROOT"

# rasr_make_job = MakeJob(
#     folder=rasr_root,
#     make_sequence=["build", "install"],
#     configure_opts=[
#         "--apptainer-setup=2026-04-28_torch-2.8_onnx-1.22",
#     ],
#     num_processes=8,
#     link_outputs={"binaries": "arch/linux-x86_64-standard/"},
# )
# rasr_make_job.rqmt["gpu"] = 1
# rasr_make_job.rqmt["gpu_mem"] = 24
rasr_make_job = CMakeJob(
    source_folder=rasr_root,
    cmake_opts=["-DMODULE_TENSORFLOW=Off", "-DMODULE_LM_TFRNN=Off", "-DMARCH=x86-64-v3"],
    num_processes=8,
    mem_rqmt=8,
)
# Needed to have CUDA headers available
# rasr_make_job.rqmt["gpu"] = 1
# rasr_make_job.rqmt["gpu_mem"] = 24

rasr_binary_path: tk.Path = rasr_make_job.out_install_dir
rasr_binary_path.hash_overwrite = "RASR_BINARY_PATH"


# Isolated prototype build: do not overwrite the hashes of the production paths.
# This master-based decoder deliberately has no berger_dev LM look-ahead.
OOV_PROTOTYPE_RASR_COMMIT = "5083eddece2982cb3b55c09daf1368e6d460f467"


def get_oov_prototype_rasr_binary_path() -> tk.Path:
    """Build the pinned OOV prototype, independently of the production decoder."""
    from pathlib import Path

    # Approved compatibility patch: XML cannot otherwise express membership of
    # one separator lemma in both special groups. Content enters the clone hash.
    separator_patch = Path(__file__).with_name("rasr_oov_separator.patch").read_text()
    source = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/rasr.git",
        commit=OOV_PROTOTYPE_RASR_COMMIT,
        checkout_folder_name="rasr",
        patches=[separator_patch],
    )
    source.add_alias("tools/rasr-oov-prototype/source")
    build = CMakeJob(
        source_folder=source.out_repository,
        cmake_opts=["-DMODULE_TENSORFLOW=Off", "-DMODULE_LM_TFRNN=Off", "-DMARCH=x86-64-v3"],
        num_processes=8,
        mem_rqmt=8,
    )
    build.add_alias("tools/rasr-oov-prototype/cmake")
    tk.register_output("tools/rasr-oov-prototype/binaries", build.out_install_dir)
    return build.out_install_dir


def get_oov_length_rasr_binary_path() -> tk.Path:
    """Second, separately hashed prototype; keep the 5083edde experiments intact."""
    from pathlib import Path

    source = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/rasr.git",
        commit="101ed636a8250e1cb6981f3d885f108c0e5fa3e0",
        checkout_folder_name="rasr",
        patches=[Path(__file__).with_name("rasr_oov_separator.patch").read_text()],
    )
    source.add_alias("tools/rasr-oov-101ed636/source")
    build = CMakeJob(
        source_folder=source.out_repository,
        cmake_opts=["-DMODULE_TENSORFLOW=Off", "-DMODULE_LM_TFRNN=Off", "-DMARCH=x86-64-v3"],
        num_processes=8,
        mem_rqmt=8,
    )
    build.add_alias("tools/rasr-oov-101ed636/cmake")
    tk.register_output("tools/rasr-oov-101ed636/binaries", build.out_install_dir)
    return build.out_install_dir

def get_oov_upfront_rasr_binary_path() -> tk.Path:
    """Upfront-beta fix, with independent source/build hashes and output aliases."""
    from pathlib import Path

    source = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/rasr.git",
        commit="3ed6d27b7ca628d3a6b46ff2ac65033f16d16810",
        checkout_folder_name="rasr",
        patches=[Path(__file__).with_name("rasr_oov_separator.patch").read_text()],
    )
    source.add_alias("tools/rasr-oov-3ed6d27b/source")
    build = CMakeJob(
        source_folder=source.out_repository,
        cmake_opts=["-DMODULE_TENSORFLOW=Off", "-DMODULE_LM_TFRNN=Off", "-DMARCH=x86-64-v3"],
        num_processes=8,
        mem_rqmt=8,
    )
    build.add_alias("tools/rasr-oov-3ed6d27b/cmake")
    tk.register_output("tools/rasr-oov-3ed6d27b/binaries", build.out_install_dir)
    return build.out_install_dir


def get_oov_lookahead_rasr_binary_path() -> tk.Path:
    """Beta look-ahead before within-word pruning; keep earlier builds intact."""
    from pathlib import Path

    source = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/rasr.git",
        commit="6e07a2e503fb18cc4fcddc89fa01e2d9fbd4a5d9",
        checkout_folder_name="rasr",
        patches=[Path(__file__).with_name("rasr_oov_separator.patch").read_text()],
    )
    source.add_alias("tools/rasr-oov-6e07a2e5/source")
    build = CMakeJob(
        source_folder=source.out_repository,
        cmake_opts=["-DMODULE_TENSORFLOW=Off", "-DMODULE_LM_TFRNN=Off", "-DMARCH=x86-64-v3"],
        num_processes=8,
        mem_rqmt=8,
    )
    build.add_alias("tools/rasr-oov-6e07a2e5/cmake")
    tk.register_output("tools/rasr-oov-6e07a2e5/binaries", build.out_install_dir)
    return build.out_install_dir


def get_llm_search_rasr_binary_path() -> tk.Path:
    """Lexicon-free dynamic LLM prototype; do not change earlier decoder jobs."""
    source = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/rasr.git",
        commit="9804f5aa545d1fcf69df4843382a41242eb9f722",
        checkout_folder_name="rasr",
    )
    source.add_alias("tools/rasr-llm-9804f5aa/source")
    build = CMakeJob(
        source_folder=source.out_repository,
        cmake_opts=["-DMODULE_TENSORFLOW=Off", "-DMODULE_LM_TFRNN=Off", "-DMARCH=x86-64-v3"],
        num_processes=8,
        mem_rqmt=8,
    )
    build.add_alias("tools/rasr-llm-9804f5aa/cmake")
    tk.register_output("tools/rasr-llm-9804f5aa/binaries", build.out_install_dir)
    return build.out_install_dir


def get_llm_rnnt_search_rasr_binary_path() -> tk.Path:
    """Pinned RNN-T dynamic-word LLM search, isolated from completed CTC jobs."""
    source = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/rasr.git",
        commit="d000386c08a9b9213238e3cafdeb4291dc0b1775",
        checkout_folder_name="rasr",
    )
    source.add_alias("tools/rasr-llm-rnnt-d000386c/source")
    build = CMakeJob(
        source_folder=source.out_repository,
        cmake_opts=["-DMODULE_TENSORFLOW=Off", "-DMODULE_LM_TFRNN=Off", "-DMARCH=x86-64-v3"],
        num_processes=8,
        mem_rqmt=8,
    )
    build.add_alias("tools/rasr-llm-rnnt-d000386c/cmake")
    tk.register_output("tools/rasr-llm-rnnt-d000386c/binaries", build.out_install_dir)
    return build.out_install_dir


def get_nbest_fixed_rasr_binary_path() -> tk.Path:
    """OOV tree plus corrected native n-best scores, separate from other grids."""
    from pathlib import Path

    source = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/rasr.git",
        commit="ff206f90c64fa5c44fa1acb7059f23722bfefc8c",
        checkout_folder_name="rasr",
        patches=[Path(__file__).with_name("rasr_oov_separator.patch").read_text()],
    )
    source.add_alias("tools/rasr-nbest-ff206f90/source")
    build = CMakeJob(
        source_folder=source.out_repository,
        cmake_opts=["-DMODULE_TENSORFLOW=Off", "-DMODULE_LM_TFRNN=Off", "-DMARCH=x86-64-v3"],
        num_processes=8,
        mem_rqmt=8,
    )
    build.add_alias("tools/rasr-nbest-ff206f90/cmake")
    tk.register_output("tools/rasr-nbest-ff206f90/binaries", build.out_install_dir)
    return build.out_install_dir


def get_oov_multitoken_rasr_binary_path() -> tk.Path:
    """Multi-token LM merge, isolated from the completed 6e07a2e5 comparisons."""
    from pathlib import Path

    source = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/rasr.git",
        commit="8c1ac793675fb7aba0198cc165dca7a2e624375d",
        checkout_folder_name="rasr",
        patches=[Path(__file__).with_name("rasr_oov_separator.patch").read_text()],
    )
    source.add_alias("tools/rasr-oov-8c1ac793/source")
    build = CMakeJob(
        source_folder=source.out_repository,
        cmake_opts=["-DMODULE_TENSORFLOW=Off", "-DMODULE_LM_TFRNN=Off", "-DMARCH=x86-64-v3"],
        num_processes=8,
        mem_rqmt=8,
    )
    build.add_alias("tools/rasr-oov-8c1ac793/cmake")
    tk.register_output("tools/rasr-oov-8c1ac793/binaries", build.out_install_dir)
    return build.out_install_dir


# rasr_root = tk.Path(
#     "/work/asr4/berger/rasr_dev/label_scorer/rasr/",
#     hash_overwrite="RASR_ROOT",
# )
#
# rasr_binary_path = tk.Path(
#     "/work/asr4/berger/rasr_dev/label_scorer/rasr/arch/linux-x86_64-standard",
#     hash_overwrite="RASR_BINARY_PATH",
# )

sctk_binary_path = compile_sctk(branch="v2.4.12")
sctk_binary_path.hash_overwrite = "SCTK_BINARY_PATH"

ken_lm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm.git").out_repository
ken_lm_repo.hash_overwrite = "KENLM"
ken_lm_binaries = CompileKenLMJob(repository=ken_lm_repo).out_binaries

subword_nmt_repo = get_returnn_subword_nmt(commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf")
subword_nmt_repo.hash_overwrite = "I6_SUBWORD_NMT_V2"
