__all__ = ["concurrent", "corpora", "lexica", "stm_path", "glm_path", "feature_bundles"]

from sisyphus import tk

Path = tk.setup_path(__package__)

SWB_ROOT = "/u/corpora/speech/switchboard-1/"
HUB500_ROOT = "/u/corpora/speech/hub-5-00/"
HUB501_ROOT = "/u/corpora/speech/hub-5-01/"
HUB5E00_ROOT = "/u/corpora/speech/hub5e_00/"

concurrent = {"train": 200, "dev": 20, "eval": 20}

corpora = {
    "train": {k: tk.Object() for k in ["full", "100k", "2000h"]},
    "dev": {k: tk.Object() for k in ["hub5-00", "hub5-01", "hub5e-00", "dev_zoltan"]},
    "eval": {k: tk.Object() for k in ["hub5-00", "hub5-01", "hub5e-00", "dev_zoltan"]},
}

corpora["train"]["full"].corpus_file = Path(SWB_ROOT + "xml/swb1-all/swb1-all.corpus.gz")
corpora["train"]["full"].audio_dir = Path(SWB_ROOT + "audio/")
corpora["train"]["full"].audio_format = "wav"
corpora["train"]["full"].duration = 311.78

corpora["train"]["100k"].corpus_file = Path(SWB_ROOT + "xml/swb1-100k/swb1-100k.corpus.gz")
corpora["train"]["100k"].audio_dir = Path(SWB_ROOT + "audio/")
corpora["train"]["100k"].audio_format = "wav"
corpora["train"]["100k"].duration = 123.60

corpora["train"]["2000h"].corpus_file = Path(SWB_ROOT + "xml/swb-2000/train.2000.xml.gz")
corpora["train"]["2000h"].audio_dir = Path("/work/speechcorpora/fisher/en/audio/")
corpora["train"]["2000h"].audio_format = "wav"
corpora["train"]["2000h"].duration = 2033.0

for c in ["dev", "eval"]:
    corpora[c]["hub5-00"].corpus_file = Path(HUB500_ROOT + "xml/hub-5-00-all.corpus.gz")
    corpora[c]["hub5-00"].audio_dir = Path(HUB500_ROOT + "audio/")
    corpora[c]["hub5-00"].audio_format = "wav"
    corpora[c]["hub5-00"].duration = 3.65

    corpora[c]["hub5e-00"].corpus_file = Path(HUB5E00_ROOT + "xml/hub5e_00.corpus.gz")
    corpora[c]["hub5e-00"].audio_dir = Path(HUB5E00_ROOT + "english/")
    corpora[c]["hub5e-00"].audio_format = "wav"
    corpora[c]["hub5e-00"].duration = 3.65

    corpora[c]["hub5-01"].corpus_file = Path(
        "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/tuske-files/hub5e_01.corpus.gz"
    )
    corpora[c]["hub5-01"].audio_dir = Path("/u/corpora/speech/hub-5-01/audio/")
    corpora[c]["hub5-01"].audio_format = "nist"
    corpora[c]["hub5-01"].duration = 6.2

    corpora[c]["dev_zoltan"].corpus_file = Path(
        "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/tuske-files/dev.corpus.gz"
    )
    corpora[c]["dev_zoltan"].audio_dir = Path(HUB500_ROOT + "audio/")
    corpora[c]["dev_zoltan"].audio_format = "wav"
    corpora[c]["dev_zoltan"].duration = 3.65


lexica = {
    "train": Path(SWB_ROOT + "lexicon/train.lex.v1_0_3.ci.gz"),
    "train-2000": Path(SWB_ROOT + "xml/swb-2000/train.merge.g2p.lex.gz"),
    "eval": Path(SWB_ROOT + "lexicon/train.lex.v1_0_4.ci.gz"),
}

lms = {
    "train": Path(SWB_ROOT + "lm/switchboardFisher.lm.gz"),
    "eval": Path("/u/vieting/setups/swb/20230406_feat/dependencies/swb.fsh.4gr.voc30k.LM.gz"),
}  # TODO Check if correct lms are used /home/tuske/work/ASR/switchboard/corpus/lm/data/mylm/swb.fsh.4gr.voc30k.LM.gz got deleted

stm_path = {
    "hub5-00": Path(HUB500_ROOT + "scoring/hub-5-00-all.stm"),
    "hub5e-00": Path(HUB5E00_ROOT + "xml/hub5e_00.stm"),
    "hub5-01": Path(HUB501_ROOT + "raw/hub5e_01/data/transcr/hub5e01.english.20010402.stm"),
    "dev_zoltan": Path("/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/tuske-files/hub5e_00.2.stm"),
}
glm_path = {
    "hub5-00": Path(HUB500_ROOT + "scoring/en20000405_hub5.glm"),
    "hub5e-00": Path(HUB5E00_ROOT + "xml/glm"),
    "hub5-01": Path(HUB500_ROOT + "scoring/en20000405_hub5.glm"),
    "dev_zoltan": Path(HUB500_ROOT + "scoring/en20000405_hub5.glm"),
}


##########################
# Comparability with Zhou
##########################
prepath_dependencies = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies"
prepath_work_folder = "/work/asr3/raissi/master-thesis/raissi/work"
feature_bundles = {
    "train": Path(("/").join(
        [
            prepath_work_folder,
            "features/extraction/FeatureExtraction.Gammatone.Jlfrg2riiRX3/output/gt.cache.bundle",
        ]
    ), hash_overwrite="legacy_train_features"),
    "hub500": Path(("/").join(
        [
            prepath_work_folder,
            "features/extraction/FeatureExtraction.Gammatone.dVkMNkHYPXb4/output/gt.cache.bundle",
        ]
    ), hash_overwrite="legacy_dev_features"),
    "hub501": Path(("/").join(
        [
            prepath_dependencies,
            "hub5-01/gammatone/FeatureExtraction.Gammatone.osrT6JyBKDB2/output/gt.cache.bundle",
        ]
    ), hash_overwrite="legacy_eval_features"),
}



merged_train_cv = {
    "pre_path": ("/").join([prepath_dependencies, "cv-from-hub5-00"]),
    "merged_corpus_path": ("/").join(["merged_corpora", "train-dev.corpus.gz"]),
    "merged_corpus_segment": ("/").join(["merged_corpora", "segments"]),
    "cleaned_dev_corpus_path": ("/").join(["zhou-files-dev", "hub5_00.corpus.cleaned.gz"]),
    "cleaned_dev_segment_path": ("/").join(["zhou-files-dev", "segments"]),
    "features_path": ("/").join(["features", "gammatones", "FeatureExtraction.Gammatone.pp9W8m2Z8mHU"]),
}
