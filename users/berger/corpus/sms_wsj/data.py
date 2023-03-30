import re
from typing import Dict, Generator, Iterable, List, Optional, Tuple
from i6_core.text.label.subword_nmt.train import ReturnnTrainBpeJob
from i6_core.tools.git import CloneGitRepositoryJob

from sisyphus import tk, Job, Task

from i6_core.audio.ffmpeg import BlissFfmpegJob

import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.users.berger.recipe.lexicon.modification import (
    EnsureSilenceFirstJob,
    DeleteEmptyOrthJob,
    EnsureUnknownPronunciationOrthJob,
)
from i6_core.meta.system import CorpusObject
from i6_core.lib.corpus import Corpus
from i6_core.lib.lexicon import Lexicon
from i6_core.util import uopen, write_xml

from returnn.datasets.lm import english_cleaners


dep_dir = "/work/asr4/berger/dependencies/sms_wsj"

lm_corpus = tk.Path("/u/corpora/language/wsj/NAB-training-corpus.gz", cached=True)


def process_string(s: str) -> str:
    s = re.sub(r"[\!\"\%\,\/\:\;\?\{\}\&\*]", "", s)
    s = re.sub("`", "'", s)
    s = re.sub(r"\.+(\w)", r"\g<1>", s)
    s = re.sub(r"(\s|\A)\'", r"\g<1>", s)
    s = re.sub(r"(\s|\A)\(", r"\g<1>", s)
    s = re.sub(r"(\s|\A)\)", r"\g<1>", s)
    s = re.sub(r"\(\S*\)", "", s)
    s = re.sub(r"\[\S*\]", "", s)
    s = re.sub(r"<\S*>", "", s)
    s = re.sub("-HYPHEN", "HYPHEN", s)
    s = re.sub("--DASH", "DASH", s)
    s = " ".join(s.split())
    return s


def lm_cleaning(s: str) -> str:
    remove_regexes = [
        re.compile(expr)
        for expr in [
            r" *</s>",
            r"<s> *",
            r" *<.*> *",
            r" *< *",
            r" *> *",
            r" *\* *",
            r" *, *",
            r" *\^ *",
            r" *\\ *",
            r" *\| *",
            r" *~ *",
            r" *\[.*\] *",
            r" *\[ *",
            r" *\] *",
            r" *\. *",
            r" *# *",
        ]
    ]
    replace_regexes = [
        (re.compile(r"\$"), "dollars"),
        (r"(.)\1+", r"\1\1"),
    ]
    sentence_clean = english_cleaners(s)
    for expr in remove_regexes:
        sentence_clean = re.sub(expr, "", sentence_clean)
    for expr, repl in replace_regexes:
        sentence_clean = re.sub(expr, repl, sentence_clean)
    return sentence_clean


class PreprocessWSJTranscriptionsJob(Job):
    def __init__(self, corpus_file: tk.Path, lm_cleaning: bool = False) -> None:
        self.corpus_file = corpus_file
        self.lm_cleaning = lm_cleaning

        self.out_corpus_file = self.output_path("corpus.xml.gz")

    def tasks(self) -> Generator[Task, None, None]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        corpus = Corpus()
        corpus.load(self.corpus_file.get_path())

        for segment in corpus.segments():
            segment.orth = process_string(segment.orth)
            if self.lm_cleaning:
                segment.orth = lm_cleaning(segment.orth)

        corpus.dump(self.out_corpus_file.get_path())


class PreprocessWSJLexiconJob(Job):
    def __init__(self, lexicon_file: tk.Path, lm_cleaning: bool = False) -> None:
        self.lexicon_file = lexicon_file
        self.lm_cleaning = lm_cleaning

        self.out_lexicon_file = self.output_path("lexicon.xml.gz")

    def tasks(self) -> Generator[Task, None, None]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        lexicon = Lexicon()
        lexicon.load(self.lexicon_file)
        removed_lemmata = []
        for lemma in lexicon.lemmata:
            if lemma.orth and not lemma.special:
                lemma.orth = [process_string(o) for o in lemma.orth]
                if self.lm_cleaning:
                    lemma.orth = [lm_cleaning(o) for o in lemma.orth]
                lemma.orth = [o for o in lemma.orth if o != ""]
                if len(lemma.orth) == 0:
                    removed_lemmata.append(lemma)
        lexicon.lemmata = [
            lemma for lemma in lexicon.lemmata if lemma not in removed_lemmata
        ]
        write_xml(self.out_lexicon_file, lexicon.to_xml())


class PreprocessLmFileJob(Job):
    def __init__(self, lm_file: tk.Path, lm_cleaning: bool = False) -> None:
        self.lm_file = lm_file
        self.lm_cleaning = lm_cleaning

        self.out_txt = self.output_path("corpus.txt")

    def tasks(self) -> Generator[Task, None, None]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        with uopen(self.lm_file, "rt") as in_file:
            with uopen(self.out_txt, "wt") as out_file:
                for line in in_file:
                    processed_line = process_string(line)
                    if self.lm_cleaning:
                        processed_line = lm_cleaning(processed_line)
                    out_file.write(processed_line + "\n")


def get_corpus_object_dict():

    corpus_object_dict = {}

    for name, freq, duration, audio_dir, audio_format, concurrency in [
        (
            "train_si284",
            8,
            81.41,
            "/work/asr3/converse/data/sms_wsj_original_dump/wsj_8k_zeromean/",
            "wav",
            50,
        ),
        ("train_si284", 16, 81.41, "/u/corpora/speech/wsj/audio/", "nist", 50),
        (
            "cv_dev93",
            8,
            1.09,
            "/work/asr3/converse/data/sms_wsj_original_dump/wsj_8k_zeromean/",
            "wav",
            10,
        ),
        ("cv_dev93", 16, 1.09, "/u/corpora/speech/wsj/audio/", "nist", 10),
        (
            "test_eval92",
            8,
            1.54,
            "/work/asr3/converse/data/sms_wsj_original_dump/wsj_8k_zeromean/",
            "wav",
            10,
        ),
        ("test_eval92", 16, 1.54, "/u/corpora/speech/wsj/audio/", "nist", 10),
        (
            "sms_train_si284",
            8,
            144.77,
            "/work/asr3/converse/data/sms_wsj/audio/",
            "wav",
            50,
        ),
        (
            "sms_train_si284",
            16,
            144.77,
            "/work/asr3/berger/sms_wsj_16kHz/audio/",
            "wav",
            50,
        ),
        (
            "sms_train_si284_mixed",
            8,
            87.37,
            "/work/asr3/converse/data/sms_wsj_original_dump/observation/train_si284/",
            "wav",
            50,
        ),
        (
            "sms_train_si284_speechsource",
            8,
            144.77,
            "/work/asr3/converse/data/sms_wsj_original_dump/speech_source/train_si284/",
            "wav",
            50,
        ),
        ("sms_cv_dev93", 8, 4.21, "/work/asr3/converse/data/sms_wsj/audio/", "wav", 10),
        ("sms_cv_dev93", 16, 4.21, "/work/asr3/berger/sms_wsj_16kHz/audio/", "wav", 10),
        (
            "sms_cv_dev93_mixed",
            8,
            2.53,
            "/work/asr3/converse/data/sms_wsj_original_dump/observation/cv_dev93/",
            "wav",
            10,
        ),
        (
            "sms_cv_dev93_speechsource",
            8,
            4.21,
            "/work/asr3/converse/data/sms_wsj_original_dump/speech_source/cv_dev93/",
            "wav",
            10,
        ),
        (
            "sms_test_eval92",
            8,
            5.64,
            "/work/asr3/converse/data/sms_wsj/audio/",
            "wav",
            10,
        ),
        (
            "sms_test_eval92",
            16,
            5.64,
            "/work/asr3/berger/sms_wsj_16kHz/audio/",
            "wav",
            10,
        ),
        (
            "sms_test_eval92_mixed",
            8,
            3.36,
            "/work/asr3/converse/data/sms_wsj_original_dump/observation/test_eval92/",
            "wav",
            10,
        ),
        (
            "sms_test_eval92_speechsource",
            8,
            5.64,
            "/work/asr3/converse/data/sms_wsj_original_dump/speech_source/test_eval92/",
            "wav",
            10,
        ),
    ]:
        corpus_object = CorpusObject()
        corpus_object.corpus_file = tk.Path(
            f"{dep_dir}/corpus/{freq}kHz/{name}.corpus.gz"
        )
        corpus_object.audio_dir = audio_dir
        corpus_object.audio_format = audio_format
        corpus_object.duration = duration
        corpus_object_dict[f"{name}_{freq}kHz"] = (corpus_object, concurrency)

    for name in [
        "train_si284",
        "cv_dev93",
        "test_eval92",
        "sms_train_si284_mixed",
        "sms_cv_dev93_mixed",
        "sms_test_eval92_mixed",
        "sms_train_si284_speechsource",
        "sms_cv_dev93_speechsource",
        "sms_test_eval92_speechsource",
    ]:
        corpus_object, _ = corpus_object_dict[f"{name}_8kHz"]
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
    train_keys: List[str] = ["train_si284"],
    dev_keys: List[str] = ["cv_dev93"],
    test_keys: List[str] = ["test_eval92"],
    align_keys: List[str] = [],
    freq: int = 16,
    lm_name: str = "5k_3gram",
    recog_lex_name: str = "nab-64k",
    delete_empty_orth: bool = False,
    preprocessing: bool = True,
    lm_cleaning: bool = False,
    add_all_allophones: bool = True,
) -> Tuple[Dict[str, rasr_util.RasrDataInput], ...]:
    corpus_object_dict = get_corpus_object_dict()

    for name, (corpus_object, _) in corpus_object_dict.items():
        if preprocessing or "train" in name:
            corpus_object.corpus_file = PreprocessWSJTranscriptionsJob(
                corpus_object.corpus_file,
                lm_cleaning=lm_cleaning,
            ).out_corpus_file

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
    train_lexicon_path = PreprocessWSJLexiconJob(
        train_lexicon_path, lm_cleaning
    ).out_lexicon_file
    train_lexicon_path = EnsureSilenceFirstJob(train_lexicon_path).out_lexicon
    train_lexicon_path = EnsureUnknownPronunciationOrthJob(
        train_lexicon_path
    ).out_lexicon
    if delete_empty_orth:
        train_lexicon_path = DeleteEmptyOrthJob(train_lexicon_path).out_lexicon

    train_bliss_lexicon = {
        "filename": train_lexicon_path,
        "normalize_pronunciation": False,
        "add_all": add_all_allophones,
        "add_from_lexicon": not add_all_allophones,
    }
    recog_lexicon_path = tk.Path(f"{dep_dir}/lexicon/{recog_lex_name}.lexicon.gz")
    recog_lexicon_path = EnsureSilenceFirstJob(recog_lexicon_path).out_lexicon
    if delete_empty_orth:
        recog_lexicon_path = DeleteEmptyOrthJob(recog_lexicon_path).out_lexicon
    recog_bliss_lexicon = {
        "filename": recog_lexicon_path,
        "normalize_pronunciation": False,
        "add_all": add_all_allophones,
        "add_from_lexicon": not add_all_allophones,
    }

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}
    align_data_inputs = {}

    for train_key in train_keys:
        corpus_object, concurrency = corpus_object_dict[f"{train_key}_{freq}kHz"]
        train_data_inputs[train_key] = rasr_util.RasrDataInput(
            corpus_object=corpus_object,
            concurrent=concurrency,
            lexicon=train_bliss_lexicon,
        )

    for dev_key in dev_keys:
        corpus_object, concurrency = corpus_object_dict[f"{dev_key}_{freq}kHz"]
        dev_data_inputs[dev_key] = rasr_util.RasrDataInput(
            corpus_object=corpus_object,
            concurrent=concurrency,
            lexicon=recog_bliss_lexicon,
            lm=lm,
        )

    for test_key in test_keys:
        corpus_object, concurrency = corpus_object_dict[f"{test_key}_{freq}kHz"]
        test_data_inputs[test_key] = rasr_util.RasrDataInput(
            corpus_object=corpus_object,
            concurrent=concurrency,
            lexicon=recog_bliss_lexicon,
            lm=lm,
        )

    for align_key in align_keys:
        corpus_object, concurrency = corpus_object_dict[f"{align_key}_{freq}kHz"]
        align_data_inputs[f"{align_key}_align"] = rasr_util.RasrDataInput(
            corpus_object=corpus_object,
            concurrent=concurrency,
            lexicon=train_bliss_lexicon,
        )

    return train_data_inputs, dev_data_inputs, test_data_inputs, align_data_inputs


def get_final_gmm_output(
    train_keys: Optional[Iterable[str]] = None,
    dev_keys: Optional[Iterable[str]] = None,
    test_keys: Optional[Iterable[str]] = None,
    align_keys: Optional[Iterable[str]] = None,
):
    output_args = rasr_util.OutputArgs("final")

    for train_key in train_keys or ["train_si284", "sms_train_si284"]:
        output_args.define_corpus_type(train_key, "train")

    for dev_key in dev_keys or ["cv_dev93", "sms_cv_dev93"]:
        output_args.define_corpus_type(dev_key, "dev")

    for test_key in test_keys or ["test_eval92", "sms_test_eval92"]:
        output_args.define_corpus_type(test_key, "test")

    for align_key in align_keys or [
        "sms_train_si284_speechsource",
        "sms_cv_dev93_speechsource",
        "sms_test_eval92_speechsource",
    ]:
        output_args.define_corpus_type(align_key, "align")

    output_args.add_feature_to_extract("gt")

    return output_args


def get_lm_corpus(**kwargs) -> tk.Path:
    return PreprocessLmFileJob(lm_file=lm_corpus, **kwargs).out_txt


def get_bpe(size: int, **kwargs) -> ReturnnTrainBpeJob:
    clean_lm_corpus = get_lm_corpus(**kwargs)

    subword_nmt_repo = CloneGitRepositoryJob(
        "https://github.com/albertz/subword-nmt.git"
    ).out_repository

    return ReturnnTrainBpeJob(
        clean_lm_corpus,
        size,
        subword_nmt_repo=subword_nmt_repo,
    )
