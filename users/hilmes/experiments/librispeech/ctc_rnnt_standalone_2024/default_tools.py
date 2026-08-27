"""
Defines the external software to be used for the Experiments
"""
from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lm.kenlm import CompileKenLMJob

from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
from i6_experiments.common.tools.sctk import compile_sctk

# python from apptainer/singularity/docker
RETURNN_EXE = tk.Path("/usr/bin/python3", hash_overwrite="GENERIC_RETURNN_LAUNCHER")

MINI_RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/JackTemaki/MiniReturnn", commit="0dc69329b21ce0acade4fcb2bf1be0dc8cc0b121"
).out_repository.copy()
MINI_RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

I6_MODELS_REPO_PATH = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/i6_models",
    commit="5aa74f878cc0d8d7bbc623a3ced681dcb31955ec",
    checkout_folder_name="i6_models",
).out_repository.copy()
I6_MODELS_REPO_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_I6_MODELS"

I6_NATIVE_OPS_REPO_PATH = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/i6_native_ops",
    commit="9ea83b59d23d631fb0388c76164fece2e5ae7fb3",
    checkout_folder_name="i6_native_ops",
).out_repository.copy()
I6_NATIVE_OPS_REPO_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_I6_NATIVE_OPS"

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12").copy()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"

kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"

SUBWORD_NMT_REPO = get_returnn_subword_nmt(
    commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf",
).copy()
SUBWORD_NMT_REPO.hash_overwrite = "I6_SUBWORD_NMT_V2"

# TORCH_MEMRISTOR_PATH = CloneGitRepositoryJob(
#     url="https://github.com/rwth-i6/SynaptogenML",
#     commit="7f5320d9331d4f27a0a7a5a58c2b697e608f0272",
#     checkout_folder_name="SynaptogenML",
# ).out_repository.copy()
# TORCH_MEMRISTOR_PATH = CloneGitRepositoryJob(
#     url="git@git.rwth-aachen.de:mlhlt/torch-memristor.git",
#     commit="88af8c663fa8ce55ac3b559581081653da3e1610",
#     checkout_folder_name="torch_memristor",
#     branch="bene_cycle",
# ).out_repository.copy()
from i6_core.tools.git import CloneGitRepositoryJob
# 2026-08-10 silent switch: a049b99 -> 601680e (== TORCH_MEMRISTOR_PATH_v5,
# branch bene_programming_speedup). hash_overwrite below keeps all downstream
# hashes stable; finished jobs keep their results, only newly created jobs pick
# up the new clone. Content delta vs a049b99 audited 2026-08-08: QAT layers
# (quant_modules.py), default_params.json, config.py, util.py byte-identical;
# everything new is opt-in/default-off (fast inference + fast programming) plus
# bit-exact applyVoltage speedups. Validated by the 70-point
# _newsynap_progfast_ A/B (grand mean dWER -0.004, conversions x6.3-9.7).
TORCH_MEMRISTOR_PATH = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/SynaptogenML",
    commit="601680e6cec45e8a2eae958071e286c36f375c3e",
    checkout_folder_name="SynaptogenML",
).out_repository.copy()
# TORCH_MEMRISTOR_PATH = TORCH_MEMRISTOR_PATH + "/.."
TORCH_MEMRISTOR_PATH.hash_overwrite = "LIBRISPEECH_STANDALONE_DEFAULT_TORCH_MEMRISTOR"

TORCH_MEMRISTOR_PATH_v2 = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/SynaptogenML",
    commit="bebab63f4232d50bbd3b7097212266be3d25e742",
    checkout_folder_name="SynaptogenML",
).out_repository.copy()
# TORCH_MEMRISTOR_PATH = TORCH_MEMRISTOR_PATH + "/.."

# identical to v2, plus a device-adaptive fast-inference backend (TorchScript/NNC
# fallback on CUDA capability < 7.0, e.g. GTX 1080 / Pascal, instead of crashing
# under Triton); used only by the one-off _newsynap_jitfallback_ test run so it
# doesn't disturb any already-cached v2-pinned job.
TORCH_MEMRISTOR_PATH_v3 = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/SynaptogenML",
    commit="6f89b529e034d0bf17f8f5c211867be5f1358e9e",
    checkout_folder_name="SynaptogenML",
).out_repository.copy()

# v3 plus corrected readout-noise constants (e = elementary charge 1.602e-19,
# BW = 1e8 Hz, matching upstream synaptogen.py; previously Euler's number and
# 1e-8 — a port typo that zeroed the Johnson term and inflated shot-noise sigma
# ~40x) and an opt-in noise-free readout flag (set_readout_noise). Used by the
# _newsynap_noisefix_ / _newsynap_noisefree_ seeded A/B recognitions.
TORCH_MEMRISTOR_PATH_v4 = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/SynaptogenML",
    commit="72f981338674c2a64e5eec04a34186e041acb012",
    checkout_folder_name="SynaptogenML",
).out_repository.copy()

# v3 lineage plus the cell-programming speedups (branch bene_programming_speedup):
# bit-exact applyVoltage fixes (ndarray.any, np.full, write-verify early-exit +
# hoisted buffer) and the opt-in parallel programming path
# (synaptogen_ml.set_fast_programming / SYN_FAST_PROG env; distributions
# unchanged, individual draws differ -- see benchmarks/check_programming*.py).
# Used by the _newsynap_progfast_ conversion-speed A/B recognitions.
# Since 2026-08-10 the default TORCH_MEMRISTOR_PATH points at this same commit;
# this pin stays for the hash continuity of the finished progfast A/B jobs.
TORCH_MEMRISTOR_PATH_v5 = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/SynaptogenML",
    commit="601680e6cec45e8a2eae958071e286c36f375c3e",
    checkout_folder_name="SynaptogenML",
).out_repository.copy()

# identical to TORCH_MEMRISTOR_PATH except for the no-op current_capture_hook in
# MemristorArray.forward (branch bene_add_energy_measure); used only by the energy
# measurement recognitions via import_memristor="energy"
TORCH_MEMRISTOR_PATH_ENERGY = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/SynaptogenML",
    commit="a8ec88669b929fded40115d711cce42826eed426",
    checkout_folder_name="SynaptogenML",
).out_repository.copy()
TORCH_MEMRISTOR_PATH_ENERGY.hash_overwrite = "LIBRISPEECH_STANDALONE_TORCH_MEMRISTOR_ENERGY"

rasr_path = "/work/asr4/hilmes/dev/rasr_librasr_19_09_25/"
rasr_root = tk.Path(rasr_path, hash_overwrite="LIBRISPEECH_STANDALONE_DEFAULT_RASR_ROOT").copy()
rasr_binary_path = tk.Path(
    f"{rasr_path}arch/linux-x86_64-standard",
    hash_overwrite="RASR_BINARY_PATH",
).copy()

I6_CORE_REPO_PATH = CloneGitRepositoryJob(
    url="https://github.com/rwth-i6/i6_core",
    commit="fd5b07bd902e4027d5ca9b15e9d88687699ea274",
    checkout_folder_name="i6_core",
).out_repository.copy()
I6_CORE_REPO_PATH.hash_overwrite = "LIBRISPEECH_STANDALONE_DEFAULT_I6_CORE"
