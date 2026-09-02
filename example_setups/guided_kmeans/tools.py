import getpass

from sisyphus import tk

# Change these path such that they match your environment
cur_user = getpass.getuser()

if cur_user == "lkleppel":
    RETURNN_PYTHON_EXE = tk.Path("/usr/bin/python3")
    RETURNN_ROOT = tk.Path("/u/mann/src/returnn")

    RASR_PATH = tk.Path("/work/asr4/lkleppel/rasr_dev/ngram_linear_search/rasr/arch/linux-x86_64-standard")    # for linear search
    RASR_PATH_FORWARD_BACKWARD = tk.Path("/work/asr4/lkleppel/rasr_dev/forward-backward/rasr/arch/linux-x86_64-standard")    # for forward-backward
else:
    RETURNN_PYTHON_EXE = tk.Path("/work/asr3/michel/mann/virtualenv/2025-04-23_tensorflow-2.17_onnx-1.20_v1/bin/python3.11")
    RETURNN_ROOT = tk.Path("/u/mann/src/returnn")

    RASR_PATH = tk.Path("/work/asr3/michel/mann/tools/rasr/librasr_recog2/arch/linux-x86_64-standard")
    # Local forward-backward build. lkleppel's build cannot be used from this setup:
    # it is compiled in the Ubuntu-24.04 image and ships librasr.cpython-312-*.so,
    # which RETURNN_PYTHON_EXE (3.11) does not even recognize as a module, so the FB
    # worker pool dies with "ModuleNotFoundError: No module named 'librasr'".
    # This build is the same source compiled against python 3.11 (MODULE_CUDA=OFF,
    # MODULE_TENSORFLOW=OFF); see build_with_python.sh next to it.
    RASR_PATH_FORWARD_BACKWARD = tk.Path("/work/asr3/michel/mann/tools/rasr/fwd_bwd/arch/linux-x86_64-standard")
