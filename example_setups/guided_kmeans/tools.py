from sisyphus import tk

# Change these path such that they match your environment

RETURNN_PYTHON_EXE = tk.Path("/usr/bin/python3")
RETURNN_ROOT = tk.Path("/u/mann/src/returnn")

RASR_PATH = tk.Path("/work/asr4/lkleppel/rasr_dev/ngram_linear_search/rasr/arch/linux-x86_64-standard")    # for linear search
RASR_PATH_FORWARD_BACKWARD = tk.Path("/work/asr4/lkleppel/rasr_dev/forward-backward/rasr/arch/linux-x86_64-standard")    # for forward-backward