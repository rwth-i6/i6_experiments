from sisyphus import tk

# Different LMs for sequence discriminative training
BIGRAM_LM = tk.Path(
    "/work/asr3/raissi/shared_workspaces/gunz/dependencies/lm/bigram.seq_train.gz",
    cached=True,
    hash_overwrite="BIGRAM_LM_LBS",
)
