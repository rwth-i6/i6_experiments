from sisyphus import tk

# Different LMs for sequence discriminative training

#LBS
UNIGRAM_LM_LBS = tk.Path(
    "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/discr_lms/word/kn1.no_pruning.gz",
    cached=True,
    hash_overwrite="UNIIGRAM_LM_LBS",
)
BIGRAM_LM_LBS = tk.Path(
    "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/discr_lms/word/kn2.no_pruning.gz",
    cached=True,
    hash_overwrite="BIGRAM_LM_LBS",
)
FOURGRAM_LM = tk.Path(
    "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/discr_lms/word/kn4.no_pruning.gz",
    cached=True,
    hash_overwrite="FOURGRAM_LM_LBS",
)

#SWB
FOURGRAM_LM_SWB = tk.Path(
    "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/tuske-files/swb.fsh.4gr.voc30k.LM.gz",
    cached=True,
    hash_overwrite="FOURGRAM_LM_SWB",
)
