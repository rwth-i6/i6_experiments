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

# order -> path
phonetic_lm_dict = {
    2: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm/phon_count_2gram.arpa.gz"),
    3: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm/phon_count_3gram.arpa.gz"),
    4: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm/phon_count_4gram.arpa.gz"),
    5: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm/phon_count_5gram.arpa.gz"),
    6: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm/phon_count_6gram.arpa.gz"),
    7: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm/phon_count_7gram.arpa.gz"),
    8: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm/phon_count_8gram.arpa.gz"),
    9: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm/phon_count_9gram.arpa.gz"),
}

# LMs including EOW phonemes
phonetic_eow_lm_dict = {
    2: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm_eow/phon_count_2gram_eow.arpa.gz"),
    3: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm_eow/phon_count_3gram_eow.arpa.gz"),
    4: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm_eow/phon_count_4gram_eow.arpa.gz"),
    5: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm_eow/phon_count_5gram_eow.arpa.gz"),
    6: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm_eow/phon_count_6gram_eow.arpa.gz"),
    7: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm_eow/phon_count_7gram_eow.arpa.gz"),
    8: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm_eow/phon_count_8gram_eow.arpa.gz"),
    9: tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm_eow/phon_count_9gram_eow.arpa.gz"),
}

def neg_log(x):
    return -math.log(x) if x > 0.0 else float("inf")

class PhoneticLexiconFromPhonemeListJob(Job):
    def __init__(self, phoneme_list_file, add_unknown=True, add_unknown_phoneme=True, add_noise=False, silence_orth="[SILENCE]", add_empty_silence=False):
        self.add_unknown = add_unknown
        self.add_unknown_phoneme = add_unknown_phoneme
        self.add_noise = add_noise
        self.phoneme_list_file = phoneme_list_file
        self.silence_orth = silence_orth
        self.add_empty_silence = add_empty_silence

        self.out_bliss_lexicon = self.output_path("phoneme.lex.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        #with uopen(tk.uncached_path(self.phoneme_list_file), "rt") as f:
        #    phonemes = [str(lem.strip()) for lem in f]

        # First write normal phonemes in alphabetical order, then EOW-phonemes
        with uopen(tk.uncached_path(self.phoneme_list_file), "rt") as f:
            phonemes = [str(lem.strip()) for lem in f if lem.strip()]

        phonemes_ordered = sorted(
            phonemes,
            key=lambda p: (p.endswith("#"), p[:-1] if p.endswith("#") else p),
        )

        # phonemes = set()
        # for w in phonemes:
        #     phonemes.update(w)
        # phonemes.discard(" ")  # just in case

        lex = lexicon.Lexicon()
        lex.add_phoneme("[SILENCE]", variation="none")
        for p in phonemes_ordered: #sorted(phonemes):
            lex.add_phoneme(p, "none")
        if self.add_unknown and self.add_unknown_phoneme:
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
            if self.add_unknown_phoneme:
                lex.add_lemma(lexicon.Lemma(orth=["[UNKNOWN]"], phon=["[UNKNOWN]"], synt=["<unk>"], special="unknown"))
            else:
                lex.add_lemma(lexicon.Lemma(orth=["[UNKNOWN]"], synt=["<unk>"], special="unknown"))
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

        for phoneme in phonemes_ordered: #phonemes:
            phonetic_lemma = lexicon.Lemma([phoneme], [phoneme])
            lex.add_lemma(phonetic_lemma)
        
        write_xml(self.out_bliss_lexicon.get_path(), lex.to_xml())

# Take list of phonemes and append EOW-augmented phonemes
class AddEOWPhonemesToVocabListJob(Job):
    def __init__(self, phoneme_list_file):
        self.phoneme_list_file = phoneme_list_file
        self.out_phoneme_list = self.output_path("phonemes.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.phoneme_list_file, "r") as infile:
            lines = [line.rstrip("\n") for line in infile]

        with open(self.out_phoneme_list, "w") as outfile:
            # Write original lines
            for line in lines:
                outfile.write(line + "\n")

            # Write copied lines with # appended
            for line in lines:
                outfile.write(line + "#\n")


def get_acoustic_model_config() -> RasrConfig:
    return acoustic_model_config(
        states_per_phone=1,
    )

def create_lexicon(use_eow_phonemes: bool = False, add_unknown_phoneme: bool = True) -> tk.Path:
    phoneme_list = PipelineJob(phonetic_vocabulary_path, ["cut -f1 -d ' '", "grep -vE \"'|<SIL>\""], mini_task=True).out
    if use_eow_phonemes:
        phoneme_list = AddEOWPhonemesToVocabListJob(phoneme_list).out_phoneme_list
    lexicon_job = PhoneticLexiconFromPhonemeListJob(phoneme_list, add_unknown_phoneme=add_unknown_phoneme)
    output_name = "lexicon/phoneme.lex.xml.gz"
    if use_eow_phonemes:
        output_name = "lexicon/eow_phoneme.lex.xml.gz"
    tk.register_output(output_name, lexicon_job.out_bliss_lexicon)
    return lexicon_job.out_bliss_lexicon

def get_lm_config(lm_path, scale=3.0) -> RasrConfig:
    config = RasrConfig()
    config.scale = scale
    if scale > 0.0:
        config.type = "ARPA"
        config.file = lm_path

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
    lm_path: str | tk.Path = None

def create_recog_rasr_config(
    lm_scale=3.0,
    emission_scale=1.0,
    transition_scale=None,
    loop_probability=0.1,
    silence_loop_probability=0.1,
    max_beam_size=100_000,
    score_threshold=None,
    lm_order=2,
    use_eow_phonemes=False,
    use_tree_search=False
):
    if transition_scale is None:
        transition_scale = lm_scale

    if use_eow_phonemes:
        lm_path = phonetic_eow_lm_dict[lm_order]
    else:
        lm_path = phonetic_lm_dict[lm_order]

    if use_tree_search:
        recog_config = get_tree_timesync_recog_config(
            lexicon_file=create_lexicon(use_eow_phonemes, add_unknown_phoneme=False),
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
            lexicon_file=create_lexicon(use_eow_phonemes, add_unknown_phoneme=False),
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
            score_threshold=score_threshold,
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
