import re
from typing import List

from sisyphus import tk, Job, Task

from i6_core.audio.ffmpeg import BlissFfmpegJob

import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.users.berger.recipe.lexicon.modification import (
    EnsureSilenceFirstJob,
    DeleteEmptyOrthJob,
)
from i6_core.meta.system import CorpusObject
from i6_core.lib.corpus import Corpus
from i6_core.lib.lexicon import Lexicon
from i6_core.util import write_xml


dep_dir = "/work/asr4/berger/dependencies/sms_wsj"


def process_string(s: str):
    s = re.sub(r"[\!\"\%\,\/\:\;\?\{\}\&]", "", s)
    s = re.sub("`", "'", s)
    s = re.sub(r"\.(\w)", r"\g<1>", s)
    s = re.sub(r"(\s|\A)\'", r"\g<1>", s)
    s = re.sub(r"(\s|\A)\(", r"\g<1>", s)
    s = re.sub(r"(\s|\A)\)", r"\g<1>", s)
    s = re.sub(r"\(\S*\)", "", s)
    s = re.sub(r"\[\S*\]", "", s)
    s = re.sub("-HYPHEN", "HYPHEN", s)
    s = re.sub("--DASH", "DASH", s)
    s = " ".join(s.split())
    return s


class PreprocessWSJTranscriptionsJob(Job):
    def __init__(self, corpus_file: tk.Path):
        self.corpus_file = corpus_file

        self.out_corpus_file = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        corpus = Corpus()
        corpus.load(self.corpus_file.get_path())

        for segment in corpus.segments():
            segment.orth = process_string(segment.orth)

        corpus.dump(self.out_corpus_file.get_path())


class PreprocessWSJLexiconJob(Job):
    def __init__(self, lexicon_file: tk.Path):
        self.lexicon_file = lexicon_file

        self.out_lexicon_file = self.output_path("lexicon.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lexicon = Lexicon()
        lexicon.load(self.lexicon_file)
        for lemma in lexicon.lemmata:
            if lemma.orth:
                lemma.orth = [process_string(o) for o in lemma.orth]
        write_xml(self.out_lexicon_file, lexicon.to_xml())


def get_corpus_object_dict():

    corpus_object_dict = {}

    for name, freq, duration, audio_dir, audio_format in [
        (
            "train_si284",
            8,
            81.41,
            "/work/asr3/converse/data/sms_wsj_original_dump/wsj_8k_zeromean/",
            "wav",
        ),
        ("train_si284", 16, 81.41, "/u/corpora/speech/wsj/audio/", "nist"),
        (
            "cv_dev93",
            8,
            1.09,
            "/work/asr3/converse/data/sms_wsj_original_dump/wsj_8k_zeromean/",
            "wav",
        ),
        ("cv_dev93", 16, 1.09, "/u/corpora/speech/wsj/audio/", "nist"),
        (
            "test_eval92",
            8,
            1.54,
            "/work/asr3/converse/data/sms_wsj_original_dump/wsj_8k_zeromean/",
            "wav",
        ),
        ("test_eval92", 16, 1.54, "/u/corpora/speech/wsj/audio/", "nist"),
        (
            "sms_train_si284",
            8,
            144.77,
            "/work/asr3/converse/data/sms_wsj/audio/",
            "wav",
        ),
        (
            "sms_train_si284",
            16,
            144.77,
            "/work/asr3/berger/sms_wsj_16kHz/audio/",
            "wav",
        ),
        ("sms_cv_dev93", 8, 4.21, "/work/asr3/converse/data/sms_wsj/audio/", "wav"),
        ("sms_cv_dev93", 16, 4.21, "/work/asr3/berger/sms_wsj_16kHz/audio/", "wav"),
        ("sms_test_eval92", 8, 5.64, "/work/asr3/converse/data/sms_wsj/audio/", "wav"),
        ("sms_test_eval92", 16, 5.64, "/work/asr3/berger/sms_wsj_16kHz/audio/", "wav"),
    ]:
        corpus_object = CorpusObject()
        corpus_object.corpus_file = tk.Path(
            f"{dep_dir}/corpus/{freq}kHz/{name}.corpus.gz"
        )
        corpus_object.audio_dir = audio_dir
        corpus_object.audio_format = audio_format
        corpus_object.duration = duration
        corpus_object_dict[f"{name}_{freq}kHz"] = corpus_object

    for name in ["train_si284", "cv_dev93", "test_eval92"]:
        corpus_object = corpus_object_dict[f"{name}_8kHz"]
        j = BlissFfmpegJob(
            corpus_object.corpus_file,
            ffmpeg_options=["-map_channel", "0.0.0"],
            recover_duration=False,
            output_format="wav",
        )
        corpus_object.corpus_file = j.out_corpus
        corpus_object.audio_dir = j.out_audio_folder
        corpus_object.audio_format = j.output_format

    return corpus_object_dict


def get_data_inputs(
    train_key: str = "train_si284",
    dev_keys: List[str] = ["cv_dev93"],
    test_keys: List[str] = ["test_eval92"],
    freq: int = 16,
    lm_name: str = "5k_3gram",
    recog_lex_name: str = "nab-64k",
    delete_empty_orth: bool = False,
):
    corpus_object_dict = get_corpus_object_dict()

    filename = {
        "5k_3gram": "lm-5k.lm.gz",
        "20k_3gram": "lm-20k.lm.gz",
        "64k_3gram": "lm-64k.lm.gz",
    }[lm_name]

    lm = {
        "filename": tk.Path(f"/work/speech/wsj/lm/recognition/{filename}"),
        "type": "ARPA",
        "scale": 10,
    }

    train_lexicon_path = tk.Path(f"{dep_dir}/lexicon/wsj01-train.lexicon.gz")
    train_lexicon_path = EnsureSilenceFirstJob(train_lexicon_path).out_lexicon
    if delete_empty_orth:
        train_lexicon_path = DeleteEmptyOrthJob(train_lexicon_path).out_lexicon
    train_bliss_lexicon = {
        "filename": train_lexicon_path,
        "normalize_pronunciation": False,
        "add_all": True,
        "add_from_lexicon": False,
    }
    recog_lexicon_path = tk.Path(f"{dep_dir}/lexicon/{recog_lex_name}.lexicon.gz")
    recog_lexicon_path = EnsureSilenceFirstJob(recog_lexicon_path).out_lexicon
    if delete_empty_orth:
        recog_lexicon_path = DeleteEmptyOrthJob(recog_lexicon_path).out_lexicon
    recog_bliss_lexicon = {
        "filename": recog_lexicon_path,
        "normalize_pronunciation": False,
        "add_all": True,
        "add_from_lexicon": False,
    }

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs[train_key] = rasr_util.RasrDataInput(
        corpus_object=corpus_object_dict[f"{train_key}_{freq}kHz"],
        concurrent=50,
        lexicon=train_bliss_lexicon,
    )

    for dev_key in dev_keys:
        dev_data_inputs[dev_key] = rasr_util.RasrDataInput(
            corpus_object=corpus_object_dict[f"{dev_key}_{freq}kHz"],
            concurrent=10,
            lexicon=recog_bliss_lexicon,
            lm=lm,
        )

    for test_key in test_keys:
        test_data_inputs[test_key] = rasr_util.RasrDataInput(
            corpus_object=corpus_object_dict[f"{test_key}_{freq}kHz"],
            concurrent=10,
            lexicon=recog_bliss_lexicon,
            lm=lm,
        )

    return train_data_inputs, dev_data_inputs, test_data_inputs


def get_final_gmm_output():
    output_args = rasr_util.OutputArgs("final")

    for train_key in ["train_si284", "sms_train_si284"]:
        output_args.define_corpus_type(train_key, "train")

    for dev_key in ["cv_dev93", "sms_cv_dev93"]:
        output_args.define_corpus_type(dev_key, "dev")

    for test_key in ["test_eval92", "sms_test_eval92"]:
        output_args.define_corpus_type(test_key, "test")

    output_args.add_feature_to_extract("gt")

    return output_args
