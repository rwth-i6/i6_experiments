import getpass

from sisyphus import tk

_ASR3_ROOT = "/work/asr3/michel/mann/"
_ASR4_ROOT = "/work/asr4/mann/"


def moved_from_asr3(path: str) -> tk.Path:
    """
    Path to a tool that was copied 1:1 from /work/asr3/michel/mann to /work/asr4/mann
    (2026-09-17). It is hashed as its old asr3 location, so every job computed before
    the move keeps its hash. Only use this for byte-identical copies: a rebuilt tool
    under the same path would not trigger any recomputation.
    """
    assert path.startswith(_ASR4_ROOT), f"not an asr4 path: {path}"
    return tk.Path(path, hash_overwrite=_ASR3_ROOT + path[len(_ASR4_ROOT):])


# Change these path such that they match your environment
cur_user = getpass.getuser()

if cur_user == "lkleppel":
    RETURNN_PYTHON_EXE = tk.Path("/usr/bin/python3")
    RETURNN_ROOT = tk.Path("/u/mann/src/returnn")

    RASR_PATH = tk.Path("/work/asr4/lkleppel/rasr_dev/ngram_linear_search/rasr/arch/linux-x86_64-standard")    # for linear search
    RASR_PATH_FORWARD_BACKWARD = tk.Path("/work/asr4/lkleppel/rasr_dev/forward-backward/rasr/arch/linux-x86_64-standard")    # for forward-backward
else:
    RETURNN_PYTHON_EXE = moved_from_asr3("/work/asr4/mann/virtualenv/2025-04-23_tensorflow-2.17_onnx-1.20_v1/bin/python3.11")
    RETURNN_ROOT = tk.Path("/u/mann/src/returnn")

    RASR_PATH = moved_from_asr3("/work/asr4/mann/tools/rasr/librasr_recog2/arch/linux-x86_64-standard")
    # Local forward-backward build. lkleppel's build cannot be used from this setup:
    # it is compiled in the Ubuntu-24.04 image and ships librasr.cpython-312-*.so,
    # which RETURNN_PYTHON_EXE (3.11) does not even recognize as a module, so the FB
    # worker pool dies with "ModuleNotFoundError: No module named 'librasr'".
    # This build is the same source compiled against python 3.11 (MODULE_CUDA=OFF,
    # MODULE_TENSORFLOW=OFF); see build_with_python.sh next to it.
    RASR_PATH_FORWARD_BACKWARD = moved_from_asr3("/work/asr4/mann/tools/rasr/fwd_bwd/arch/linux-x86_64-standard")


#: The backoff n-gram HMM forward-backward CUDA op (see its SPEC.md and README).
#: Not per-user: it lives under /work/asr4, which every job container binds, and
#: it is read-only to everyone but its owner. /u/mann would *not* work -- the
#: binds in settings.py:worker_wrapper do not include it.
#:
#: Hashed like the RASR paths are, so pointing at a different checkout is a
#: different tool and recomputes. Rebuilding in place does not, which is the
#: same bargain RASR_PATH makes.
BACKOFF_FB = tk.Path("/work/asr4/mann/tools/backoff_fb")
