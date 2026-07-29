import getpass

from sisyphus import tk

# Change these path such that they match your environment
cur_user = getpass.getuser()

if cur_user == "lkleppel":
    RETURNN_PYTHON_EXE = tk.Path("/usr/bin/python3")
    RETURNN_ROOT = tk.Path("/u/mann/src/returnn")

    RASR_PATH = tk.Path("/work/asr4/lkleppel/rasr_dev/ngram_linear_search/rasr2/arch/linux-x86_64-standard")    # for linear search
    RASR_PATH_FORWARD_BACKWARD = tk.Path("/work/asr4/lkleppel/rasr_dev/forward-backward/rasr/arch/linux-x86_64-standard")    # for forward-backward
else:
    RETURNN_PYTHON_EXE = tk.Path("/work/asr3/michel/mann/virtualenv/2025-04-23_tensorflow-2.17_onnx-1.20_v1/bin/python3.11")
    RETURNN_ROOT = tk.Path("/u/mann/src/returnn")

    RASR_PATH = tk.Path("/work/asr3/michel/mann/tools/rasr/librasr_recog2/arch/linux-x86_64-standard")
