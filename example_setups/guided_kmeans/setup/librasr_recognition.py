from sisyphus import tk, Job, Task

from dataclasses import dataclass, asdict
import math

from i6_core.rasr import RasrConfig, WriteRasrConfigJob
from i6_core.am.config import acoustic_model_config
from i6_core.util import uopen, write_xml
from i6_core.text import PipelineJob
from i6_core.lib import lexicon

from .external.recog_rasr_config import get_tree_timesync_recog_config, get_linear_search_recog_config

phonetic_vocabulary_path = tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups"
                                   "/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec"
                                   "/wav2vec_data_utils/PrepareWav2VecTextDataJob.RZfllsI3R2Pd/output/text/phones/dict.txt")

# phonetic_lm_path = tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups"
#                            "/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec"
#                            "/wav2vec_data_utils/PrepareWav2VecTextDataJob.RZfllsI3R2Pd/output/text/phones/lm.phones.filtered.04.arpa")

phonetic_lm_path = tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/data/2gram/output/phones/lm.phones.filtered.02.arpa")

def neg_log(x):
    return -math.log(x) if x > 0.0 else float("inf")

class PhoneticLexiconFromPhonemeListJob(Job):
    def __init__(self, phoneme_list_file, add_unknown=True, add_noise=False, silence_orth="[SILENCE]", add_empty_silence=False):
        self.add_unknown = add_unknown
        self.add_noise = add_noise
        self.phoneme_list_file = phoneme_list_file
        self.silence_orth = silence_orth
        self.add_empty_silence = add_empty_silence

        self.out_bliss_lexicon = self.output_path("phoneme.lex.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(tk.uncached_path(self.phoneme_list_file), "rt") as f:
            phonemes = [str(lem.strip()) for lem in f]

        # phonemes = set()
        # for w in phonemes:
        #     phonemes.update(w)
        # phonemes.discard(" ")  # just in case

        lex = lexicon.Lexicon()
        lex.add_phoneme("[SILENCE]", variation="none")
        for p in sorted(phonemes):
            lex.add_phoneme(p, "none")
        if self.add_unknown:
            lex.add_phoneme("[UNKNOWN]", "none")
        if self.add_noise:
            lex.add_phoneme("noise", "none")

        # TODO: figure out requirements on synt/eval element for differnt types of lemmata
        # silence lemma, needs synt/eval element with empty token sequence
        lex.add_lemma(
            lexicon.Lemma(
                orth=[self.silence_orth, ""],
                phon=["[SILENCE]"],
                synt=["<SIL>"],
                special="blank",
                eval=[[]],
            )
        )
        if self.add_empty_silence:
            lex.add_lemma(
                lexicon.Lemma(
                    orth=[""],
                    phon=["[SILENCE]"],
                    synt=[],
                    special="silence",
                    eval=[[]],
                )
            )
        # sentence border lemmata, needs no eval element
        lex.add_lemma(lexicon.Lemma(orth=["[SENTENCE_BEGIN]"], synt=["<s>"], special="sentence-begin"))
        lex.add_lemma(lexicon.Lemma(orth=["[SENTENCE_END]"], synt=["</s>"], special="sentence-end"))
        # unknown lemma, needs no synt/eval element
        if self.add_unknown:
            lex.add_lemma(lexicon.Lemma(orth=["[UNKNOWN]"], phon=["[UNKNOWN]"], synt=["<unk>"], special="unknown"))
            # TODO: synt = ["<UNK>"] ???
        # noise lemma, needs empty synt token sequence but no eval element?
        if self.add_noise:
            lex.add_lemma(
                lexicon.Lemma(
                    orth=["[NOISE]"],
                    phon=["noise"],
                    synt=[],
                    special="unknown",
                )
            )

        for phoneme in phonemes:
            phonetic_lemma = lexicon.Lemma([phoneme], [phoneme])
            lex.add_lemma(phonetic_lemma)
        
        write_xml(self.out_bliss_lexicon.get_path(), lex.to_xml())


def get_acoustic_model_config() -> RasrConfig:
    return acoustic_model_config(
        states_per_phone=1,
    )

def create_lexicon() -> tk.Path:
    get_first_column = PipelineJob(phonetic_vocabulary_path, ["cut -f1 -d ' '", "grep -vE \"'|<SIL>\""], mini_task=True)
    lexicon_job = PhoneticLexiconFromPhonemeListJob(get_first_column.out)
    tk.register_output("lexicon/phoneme.lex.xml.gz", lexicon_job.out_bliss_lexicon)
    return lexicon_job.out_bliss_lexicon

def get_lm_config(lm_path, scale=3.0) -> RasrConfig:
    config = RasrConfig()
    config.type = "ARPA"
    config.file = lm_path
    config.scale = scale

    return config

def get_label_scorer_config(emission_scale = 1.0, transition_scale = 0.5, loop_probability = 0.5, silence_loop_probability = 0.75) -> RasrConfig:
    config = RasrConfig()
    config.type = "combine"
    config.num_scorers = 2

    config.scorer_1 = RasrConfig()
    config.scorer_1.scale = emission_scale
    config.scorer_1.type = "no-op"

    transition_config = RasrConfig()
    transition_config.type = "transition"
    transition_config.scale = transition_scale

    transition_config.label_to_label_score = neg_log(1.0 - loop_probability)
    transition_config.label_to_blank_score = neg_log(1.0 - loop_probability)
    transition_config.label_loop_score = neg_log(loop_probability)

    transition_config.blank_to_label_score = neg_log(1.0 - silence_loop_probability)
    transition_config.blank_loop_score = neg_log(silence_loop_probability)

    config.scorer_2 = transition_config

    return config

def get_lexicon_config() -> RasrConfig:
    return RasrConfig()

@dataclass
class RecogConfig:
    lm_scale: float = 3.0
    emission_scale: float = 1.0
    transition_scale: float | None = None
    loop_probability: float = 0.1
    silence_loop_probability: float = 0.1
    max_beam_size: int = 100_000
    lm_path: str | tk.Path = phonetic_lm_path

def create_recog_rasr_config(
    lm_scale=3.0,
    emission_scale=1.0,
    transition_scale=None,
    loop_probability=0.1,
    silence_loop_probability=0.1,
    max_beam_size=100_000,
    lm_path=phonetic_lm_path,
    use_tree_search=False
):
    if transition_scale is None:
        transition_scale = lm_scale
    if use_tree_search:
        recog_config = get_tree_timesync_recog_config(
            lexicon_file=create_lexicon(),
            collapse_repeated_labels=True,
            label_scorer_config=get_label_scorer_config(
                emission_scale=emission_scale,
                transition_scale=transition_scale,
                loop_probability=loop_probability,
                silence_loop_probability=silence_loop_probability
            ),
            lm_config=get_lm_config(lm_path, lm_scale),
            blank_index=0,
            max_beam_size=max_beam_size,
            intermediate_max_beam_size=None,
            score_threshold=None,
            word_end_score_threshold=None,
            max_word_end_beam_size=None,
            sentence_end_fallback=False,
            log_stepwise_statistics=False,
        )
    else:
        recog_config = get_linear_search_recog_config(
            lexicon_file=create_lexicon(),
            collapse_repeated_labels=True,
            label_scorer_config=get_label_scorer_config(
                emission_scale=emission_scale,
                transition_scale=transition_scale,
                loop_probability=loop_probability,
                silence_loop_probability=silence_loop_probability
            ),
            lm_config=get_lm_config(lm_path, lm_scale),
            blank_index=0,
            max_beam_size=max_beam_size,
            score_threshold=None,
            log_statistics=False,
            log_stepwise_statistics=False,
        )
    return recog_config

def create_rasr_config(recog_config: RecogConfig):
    return create_recog_rasr_config(
        **asdict(recog_config)
    )

def create_recog_rasr_file(
    lm_scale=3.0,
    emission_scale=1.0,
    transition_scale=None,
    loop_probability=0.1,
):
    recog_config = create_recog_rasr_config(
        lm_scale=lm_scale,
        emission_scale=emission_scale,
        transition_scale=transition_scale,
        loop_probability=loop_probability,
    )
    write_job = WriteRasrConfigJob(recog_config, post_config=RasrConfig())
    return write_job.out_config

def py():
    config = create_recog_rasr_config()
    tk.register_output("config/librasr_recognition.config", config)
