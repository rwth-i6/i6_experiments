"""
RASR utilities for posterior HMM training:
- CreateLibrasrVenvJob: creates a Python venv with librasr installed from a wheel
- build_phmm_am_config: creates a default acoustic model config for pHMM
- build_fsa_exporter_config: builds and writes a RASR FSA exporter config
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
import subprocess as sp
import sys
from typing import Optional, Tuple

from sisyphus import Job, Task, tk

from i6_core.lib.lexicon import Lemma, Lexicon
import i6_core.util as util
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob, build_config_from_mapping
from i6_core.rasr.crp import CommonRasrParameters, crp_add_default_output
from i6_core.util import write_xml


@dataclass(frozen=True)
class BpeLexiconRasrConfig:
    """Experiment-side config for corpus-driven BPE lexicon creation and FSA export."""

    topology: str
    special_phoneme: str
    special_orths: Tuple[str, ...]
    special_lemma: str
    allow_silence_repetitions: bool = False
    normalize_lemma_sequence_scores: bool = False
    unknown_orths: Tuple[str, ...] = ("[UNKNOWN]",)
    unknown_synt: Tuple[str, ...] = ("<UNK>",)


DEFAULT_PHMM_BPE_RASR_CONFIG = BpeLexiconRasrConfig(
    topology="hmm",
    special_phoneme="[SILENCE]",
    special_orths=("[SILENCE]", "[silence]", ""),
    special_lemma="silence",
)


DEFAULT_CTC_BPE_RASR_CONFIG = BpeLexiconRasrConfig(
    topology="ctc",
    special_phoneme="[BLANK]",
    special_orths=("[BLANK]", "[blank]", ""),
    special_lemma="blank",
)


class CreateLibrasrVenvJob(Job):
    """
    Create a Python virtual environment with librasr installed from a wheel file.
    The resulting venv python binary can be used as the RETURNN python executable
    for posterior HMM training (which requires librasr for FSA building and i6_native_ops for fbw2_loss).
    """

    def __init__(
        self,
        python_exe: tk.Path,
        librasr_wheel: tk.Path,
        extra_pip_packages: Optional[list] = None,
        python_wrapper_name: str = "python_with_path",
        track_wheel_contents: bool = False,
    ):
        """
        :param python_exe: path to the base python binary
        :param librasr_wheel: path to the librasr .whl file
        :param extra_pip_packages: optional list of additional pip packages to install (str or tk.Path)
        :param python_wrapper_name: name of the exported launcher script inside venv/bin
        """
        super().__init__()
        self.python_exe = python_exe
        self.librasr_wheel = librasr_wheel
        self.extra_pip_packages = extra_pip_packages or []
        self.python_wrapper_name = python_wrapper_name
        self.track_wheel_contents = track_wheel_contents

        self.out_venv_dir = self.output_path("venv")
        self.out_python_bin = self.out_venv_dir.join_right(f"bin/{self.python_wrapper_name}")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import subprocess

        python_path = tk.uncached_path(self.python_exe)
        venv_dir = self.out_venv_dir.get_path()

        try:
            subprocess.check_call([python_path, "-m", "venv", "--system-site-packages", venv_dir])
        except subprocess.CalledProcessError:
            # Some container images ship stdlib venv but no ensurepip. In that case,
            # create the environment without seeding a local pip and rely on the
            # system pip made visible via --system-site-packages.
            subprocess.check_call([
                python_path,
                "-m",
                "venv",
                "--system-site-packages",
                "--without-pip",
                venv_dir,
            ])

        venv_python = self.out_venv_dir.join_right("bin/python").get_path()
        wheel_path = tk.uncached_path(self.librasr_wheel)
        subprocess.check_call([venv_python, "-m", "pip", "install", wheel_path])

        for pkg in self.extra_pip_packages:
            pkg_str = tk.uncached_path(pkg) if isinstance(pkg, tk.Path) else pkg
            subprocess.check_call([venv_python, "-m", "pip", "install", pkg_str])

        wrapper_path = self.out_python_bin.get_path()
        venv_bin_dir = self.out_venv_dir.join_right("bin").get_path()
        venv_python_exec = self.out_venv_dir.join_right("bin/python3").get_path()
        with open(wrapper_path, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write(f'export PATH="{venv_bin_dir}:$PATH"\n')
            f.write(f'exec "{venv_python_exec}" "$@"\n')
        subprocess.check_call(["chmod", "+x", wrapper_path])

    @classmethod
    def hash(cls, kwargs):
        if not kwargs.get("track_wheel_contents", False):
            kwargs = {k: v for k, v in kwargs.items() if k != "track_wheel_contents"}
            return super().hash(kwargs)

        wheel_path = tk.uncached_path(kwargs["librasr_wheel"])
        wheel_md5 = hashlib.md5()
        with open(wheel_path, "rb") as wheel_file:
            for chunk in iter(lambda: wheel_file.read(1 << 20), b""):
                wheel_md5.update(chunk)
        return super().hash(
            {
                "python_exe": kwargs["python_exe"],
                "librasr_wheel_path": wheel_path,
                "librasr_wheel_md5": wheel_md5.hexdigest(),
                "extra_pip_packages": kwargs.get("extra_pip_packages") or [],
                "python_wrapper_name": kwargs.get("python_wrapper_name", "python_with_path"),
            }
        )


class CreateCorpusBpeFsaLexiconJob(Job):
    """Create a corpus-complete BPE Bliss lexicon for fbw2-based training."""

    def __init__(
        self,
        corpus_text: tk.Path,
        bpe_codes: tk.Path,
        bpe_vocab: tk.Path,
        subword_nmt_repo: tk.Path,
        rasr_config: Optional[BpeLexiconRasrConfig] = None,
        unk_label: str = "<unk>",
    ):
        super().__init__()
        self.corpus_text = corpus_text
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab
        self.subword_nmt_repo = subword_nmt_repo
        self.rasr_config = rasr_config or DEFAULT_PHMM_BPE_RASR_CONFIG
        self.unk_label = unk_label

        self.out_lexicon = self.output_path("lexicon.xml.gz")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 2})

    def run(self):
        lex = Lexicon()
        lex.phonemes = OrderedDict()

        vocab = set()
        lex.add_phoneme(self.rasr_config.special_phoneme, variation="none")
        with util.uopen(self.bpe_vocab.get_path(), "rt") as f, util.uopen("fake_count_vocab.txt", "wt") as vocab_file:
            for line in f:
                line = line.strip()
                if line in {"{", "}"}:
                    continue
                symbol = line.split(":")[0][1:-1]
                if symbol in {"<s>", "</s>", self.unk_label}:
                    continue
                vocab_file.write(symbol + " -1\n")
                symbol = symbol.replace(".", "_")
                vocab.add(symbol)
                lex.add_phoneme(symbol, variation="none")

        lex.add_lemma(
            Lemma(
                orth=list(self.rasr_config.special_orths),
                phon=[self.rasr_config.special_phoneme],
                synt=[],
                eval=[[]],
                special=self.rasr_config.special_lemma,
            )
        )
        lex.add_lemma(
            Lemma(
                orth=list(self.rasr_config.unknown_orths),
                phon=[self.rasr_config.special_phoneme],
                synt=list(self.rasr_config.unknown_synt),
                eval=[],
                special="unknown",
            )
        )

        words = set()
        with util.uopen(self.corpus_text.get_path(), "rt") as f:
            for line in f:
                for word in line.strip().split():
                    if word:
                        words.add(word)
        words = sorted(words)

        with util.uopen("words", "wt") as f:
            for word in words:
                f.write(word + "\n")

        apply_binary = os.path.join(tk.uncached_path(self.subword_nmt_repo), "apply_bpe.py")
        sp.run(
            [
                sys.executable,
                apply_binary,
                "--input",
                "words",
                "--codes",
                self.bpe_codes.get_path(),
                "--vocabulary",
                "fake_count_vocab.txt",
                "--output",
                "bpes",
            ],
            check=True,
        )

        with util.uopen("bpes", "rt") as f:
            bpe_tokens = [line.strip() for line in f]

        if len(words) != len(bpe_tokens):
            raise ValueError(f"BPE output line count mismatch: {len(words)} words vs {len(bpe_tokens)} lines")

        for word, bpe in zip(words, bpe_tokens):
            pron = [token.replace(".", "_") for token in bpe.split()]
            if any(token not in vocab for token in pron):
                raise ValueError(f"BPE token outside vocabulary for word {word!r}: {bpe!r}")
            lex.add_lemma(Lemma(orth=[word], phon=[" ".join(pron)]))

        write_xml(self.out_lexicon.get_path(), lex.to_xml())


class CreateCorpusBpePhmmLexiconJob(CreateCorpusBpeFsaLexiconJob):
    """Backward-compatible pHMM lexicon job wrapper."""

    def __init__(
        self,
        corpus_text: tk.Path,
        bpe_codes: tk.Path,
        bpe_vocab: tk.Path,
        subword_nmt_repo: tk.Path,
        silence_phoneme: str = "[SILENCE]",
        unk_label: str = "<unk>",
    ):
        super().__init__(
            corpus_text=corpus_text,
            bpe_codes=bpe_codes,
            bpe_vocab=bpe_vocab,
            subword_nmt_repo=subword_nmt_repo,
            rasr_config=BpeLexiconRasrConfig(
                topology="hmm",
                special_phoneme=silence_phoneme,
                special_orths=(silence_phoneme, "[silence]", ""),
                special_lemma="silence",
            ),
            unk_label=unk_label,
        )


class NormalizeBpeForPhmmLexiconJob(Job):
    """Post-process a BPE Bliss lexicon for posterior HMM training."""

    def __init__(self, bliss_lexicon: tk.Path, silence_phoneme: str = "[SILENCE]"):
        super().__init__()
        self.bliss_lexicon = bliss_lexicon
        self.silence_phoneme = silence_phoneme

        self.out_lexicon = self.output_path("lexicon.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = Lexicon()
        lex.load(tk.uncached_path(self.bliss_lexicon))

        removed_phonemes = {"<s>", "</s>", "<unk>"}
        removed_specials = {"sentence-begin", "sentence-end"}

        out_lex = Lexicon()
        out_lex.phonemes = OrderedDict()
        out_lex.add_phoneme(self.silence_phoneme, variation="none")
        for symbol in lex.phonemes.keys():
            if symbol in removed_phonemes or symbol == self.silence_phoneme:
                continue
            out_lex.add_phoneme(symbol, variation="none")

        out_lex.add_lemma(
            Lemma(
                orth=[self.silence_phoneme, "[silence]", ""],
                phon=[self.silence_phoneme],
                synt=[],
                eval=[[]],
                special="silence",
            )
        )

        for lemma in lex.lemmata:
            if lemma.special in removed_specials:
                continue
            if lemma.special == "unknown":
                out_lex.add_lemma(
                    Lemma(
                        orth=list(lemma.orth),
                        phon=[self.silence_phoneme],
                        synt=lemma.synt,
                        eval=lemma.eval,
                        special="unknown",
                    )
                )
                continue
            if any(phon in removed_phonemes for pron in lemma.phon for phon in pron.split()):
                continue
            out_lex.add_lemma(lemma)

        write_xml(self.out_lexicon.get_path(), out_lex.to_xml())


class NormalizePhonForPhmmLexiconJob(Job):
    """Post-process an EOW phoneme Bliss lexicon for posterior HMM training.

    Takes an EOW phoneme lexicon (without silence) and produces a PHMM-ready
    lexicon with [SILENCE] as the first phoneme (index 0), a silence lemma
    with empty orth, and without sentence-begin/end lemmata.
    """

    def __init__(self, bliss_lexicon: tk.Path, silence_phoneme: str = "[SILENCE]"):
        super().__init__()
        self.bliss_lexicon = bliss_lexicon
        self.silence_phoneme = silence_phoneme

        self.out_lexicon = self.output_path("lexicon.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = Lexicon()
        lex.load(tk.uncached_path(self.bliss_lexicon))

        removed_specials = {"sentence-begin", "sentence-end"}

        out_lex = Lexicon()
        out_lex.phonemes = OrderedDict()

        # [SILENCE] must be the first phoneme (index 0)
        out_lex.add_phoneme(self.silence_phoneme, variation="none")

        # Copy all existing phonemes, skipping [SILENCE] and [UNKNOWN]
        for symbol in lex.phonemes.keys():
            if symbol in (self.silence_phoneme, "[UNKNOWN]"):
                continue
            out_lex.add_phoneme(symbol, variation=lex.phonemes[symbol])

        # Silence lemma with empty orth
        out_lex.add_lemma(
            Lemma(
                orth=[self.silence_phoneme, ""],
                phon=[self.silence_phoneme],
                synt=[],
                eval=[[]],
                special="silence",
            )
        )

        for lemma in lex.lemmata:
            if lemma.special in removed_specials:
                continue
            if lemma.special == "silence":
                continue  # already added above
            if lemma.special == "unknown":
                # Map unknown to silence phoneme
                out_lex.add_lemma(
                    Lemma(
                        orth=list(lemma.orth),
                        phon=[self.silence_phoneme],
                        synt=lemma.synt,
                        eval=lemma.eval,
                        special="unknown",
                    )
                )
                continue
            out_lex.add_lemma(lemma)

        write_xml(self.out_lexicon.get_path(), out_lex.to_xml())


class AddSentenceBoundaryLemmataToPhmmLexiconJob(Job):
    """Add sentence-begin and sentence-end special lemmata to a PHMM lexicon for LM-based recognition."""

    def __init__(self, bliss_lexicon: tk.Path):
        super().__init__()
        self.bliss_lexicon = bliss_lexicon
        self.out_lexicon = self.output_path("lexicon.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = Lexicon()
        lex.load(tk.uncached_path(self.bliss_lexicon))

        if not any(lemma.special == "sentence-begin" for lemma in lex.lemmata):
            lex.add_lemma(Lemma(orth=["[SENTENCE-BEGIN]"], synt=["<s>"], eval=[[]], special="sentence-begin"))
        if not any(lemma.special == "sentence-end" for lemma in lex.lemmata):
            lex.add_lemma(Lemma(orth=["[SENTENCE-END]"], synt=["</s>"], eval=[[]], special="sentence-end"))

        write_xml(self.out_lexicon.get_path(), lex.to_xml())


def build_phmm_am_config(
    states_per_phone: int = 1,
    state_repetitions: int = 1,
    across_word_model: bool = True,
    early_recombination: bool = False,
    fix_allophone_context_at_word_boundaries: bool = True,
    transducer_builder_filter_out_invalid_allophones: bool = True,
    state_tying: str = "monophone",
    allophones_add_all: bool = False,
    allophones_add_from_lexicon: bool = True,
    tdp_scale: float = 1.0,
    tdp_applicator_type: str = "corrected",
    tdp_loop: float = 0.0,
    tdp_forward: float = 0.0,
    tdp_skip: str = "infinity",
    tdp_exit: float = 0.0,
    silence_loop: float = 0.0,
    silence_forward: float = 0.0,
    silence_skip: str = "infinity",
    silence_exit: float = 0.0,
    tdp_entry_m1_loop: str = "infinity",
    tdp_entry_m2_loop: str = "infinity",
) -> RasrConfig:
    """
    Build a default acoustic model RasrConfig for posterior HMM.

    :returns: RasrConfig for the acoustic model section
    """
    am = RasrConfig()
    am.fix_allophone_context_at_word_boundaries = fix_allophone_context_at_word_boundaries
    am.transducer_builder_filter_out_invalid_allophones = transducer_builder_filter_out_invalid_allophones
    am.state_tying.type = state_tying
    am.allophones.add_all = allophones_add_all
    am.allophones.add_from_lexicon = allophones_add_from_lexicon
    am.hmm.states_per_phone = states_per_phone
    am.hmm.state_repetitions = state_repetitions
    am.hmm.across_word_model = across_word_model
    am.hmm.early_recombination = early_recombination
    am.tdp.scale = tdp_scale
    am.tdp.applicator_type = tdp_applicator_type
    am.tdp["entry-m1"].loop = tdp_entry_m1_loop
    am.tdp["entry-m2"].loop = tdp_entry_m2_loop
    am.tdp["*"].loop = tdp_loop
    am.tdp["*"].forward = tdp_forward
    am.tdp["*"].skip = tdp_skip
    am.tdp["*"].exit = tdp_exit
    am.tdp.silence.loop = silence_loop
    am.tdp.silence.forward = silence_forward
    am.tdp.silence.skip = silence_skip
    am.tdp.silence.exit = silence_exit
    return am


def build_fsa_exporter_config(
    lexicon_path: tk.Path,
    am_config: Optional[RasrConfig] = None,
    am_post_config: Optional[RasrConfig] = None,
    corpus_path: Optional[tk.Path] = None,
    fsa_config: Optional[BpeLexiconRasrConfig] = None,
    topology: Optional[str] = None,
    allow_silence_repetitions: Optional[bool] = None,
    normalize_lemma_sequence_scores: Optional[bool] = None,
    extra_config: Optional[RasrConfig] = None,
    extra_post_config: Optional[RasrConfig] = None,
) -> tk.Path:
    """
    Build a RASR FSA exporter config file for use with librasr.AllophoneStateFsaBuilder.

    :param lexicon_path: path to the BLISS lexicon file
    :param am_config: acoustic model config; if None, uses build_phmm_am_config() defaults
    :param am_post_config: acoustic model post config (unhashed)
    :param corpus_path: optional path to the BLISS corpus file
    :param fsa_config: experiment-side lexicon/FSA config; defaults to posterior HMM
    :param topology: explicit FSA topology override
    :param allow_silence_repetitions: orthographic parser setting override
    :param normalize_lemma_sequence_scores: orthographic parser setting override
    :param extra_config: additional RASR config entries (hashed)
    :param extra_post_config: additional RASR post config entries (unhashed)
    :returns: tk.Path to the written RASR config file
    """
    fsa_config = fsa_config or DEFAULT_PHMM_BPE_RASR_CONFIG
    topology = topology or fsa_config.topology
    if allow_silence_repetitions is None:
        allow_silence_repetitions = fsa_config.allow_silence_repetitions
    if normalize_lemma_sequence_scores is None:
        normalize_lemma_sequence_scores = fsa_config.normalize_lemma_sequence_scores

    crp = CommonRasrParameters()
    crp_add_default_output(crp)

    # Lexicon
    crp.lexicon_config = RasrConfig()
    crp.lexicon_config.file = tk.uncached_path(lexicon_path)
    crp.lexicon_config.normalize_pronunciation = False

    # Acoustic model
    if am_config is None:
        am_config = build_phmm_am_config()
    crp.acoustic_model_config = am_config
    crp.acoustic_model_post_config = am_post_config

    # Corpus (optional, not needed for build_by_orthography)
    if corpus_path is not None:
        crp.corpus_config = RasrConfig()
        crp.corpus_config.file = tk.uncached_path(corpus_path)
        crp.corpus_config.warn_about_unexpected_elements = False
        crp.corpus_config.capitalize_transcriptions = False
        crp.corpus_config.progress_indication = "global"

    mapping = {
        "lexicon": "lib-rasr.alignment-fsa-exporter.model-combination.lexicon",
        "acoustic_model": "lib-rasr.alignment-fsa-exporter.model-combination.acoustic-model",
    }
    if corpus_path is not None:
        mapping["corpus"] = "lib-rasr.corpus"

    config, post_config = build_config_from_mapping(crp, mapping)

    # Allophone state graph builder settings
    graph_builder = config.lib_rasr.alignment_fsa_exporter.allophone_state_graph_builder
    graph_builder.topology = topology
    graph_builder.orthographic_parser.allow_for_silence_repetitions = allow_silence_repetitions
    graph_builder.orthographic_parser.normalize_lemma_sequence_scores = normalize_lemma_sequence_scores

    post_config["*"].output_channel.file = "fastbw.log"

    if extra_config is not None:
        config._update(extra_config)
    if extra_post_config is not None:
        post_config._update(extra_post_config)

    write_job = WriteRasrConfigJob(config, post_config)
    return write_job.out_config


def build_librasr_phmm_recognition_config(
    lexicon_path: tk.Path,
    lm_config: Optional[RasrConfig] = None,
    am_config: Optional[RasrConfig] = None,
    max_beam_size: int = 1024,
    max_word_end_beam_size: Optional[int] = None,
    intermediate_max_beam_size: int = 1024,
    score_threshold: float = 18.0,
    word_end_score_threshold: Optional[float] = None,
    intermediate_score_threshold: float = 18.0,
    sentence_end_fallback: bool = True,
    maximum_stable_delay: Optional[int] = None,
    log_stepwise_statistics: bool = True,
    logfile_suffix: str = "recog",
) -> tk.Path:
    """
    Build a LibRASR recognition config for phoneme pHMM lexical search.

    This uses the HMM tree builder and keeps repeated-label collapse disabled
    to match the current pHMM prototype assumptions.
    """
    config = RasrConfig()
    post_config = RasrConfig()

    logfile_name = f"rasr.{logfile_suffix}.log"
    config["*.log.channel"] = logfile_name
    config["*.warning.channel"] = logfile_name
    config["*.error.channel"] = logfile_name
    config["*.statistics.channel"] = logfile_name
    config["*.unbuffered"] = False
    post_config["*.encoding"] = "UTF-8"

    config.lib_rasr = RasrConfig()
    config.lib_rasr.global_cache = RasrConfig()
    config.lib_rasr.global_cache.file = "global.cache"
    config.lib_rasr.global_cache.read_only = False
    config.lib_rasr.lexicon = RasrConfig()
    config.lib_rasr.lexicon.file = lexicon_path
    config.lib_rasr.lexicon.normalize_pronunciation = False

    if lm_config is not None:
        config.lib_rasr.lm = lm_config
    else:
        config.lib_rasr.lm = RasrConfig()
        config.lib_rasr.lm.scale = 0.0

    config.lib_rasr.acoustic_model = am_config if am_config is not None else build_phmm_am_config()
    config.lib_rasr.label_scorer = RasrConfig()
    config.lib_rasr.label_scorer.type = "no-op"

    search_algorithm = RasrConfig()
    search_algorithm.type = "tree-timesync-beam-search"
    search_algorithm.tree_builder_type = "hmm"
    search_algorithm.collapse_repeated_labels = False
    search_algorithm.force_blank_between_repeated_labels = False
    search_algorithm.max_beam_size = max_beam_size
    search_algorithm.intermediate_max_beam_size = intermediate_max_beam_size
    search_algorithm.score_threshold = score_threshold
    search_algorithm.intermediate_score_threshold = intermediate_score_threshold
    search_algorithm.sentence_end_fall_back = sentence_end_fallback
    search_algorithm.log_stepwise_statistics = log_stepwise_statistics
    if max_word_end_beam_size is not None:
        search_algorithm.max_word_end_beam_size = max_word_end_beam_size
    if word_end_score_threshold is not None:
        search_algorithm.word_end_score_threshold = word_end_score_threshold
    if maximum_stable_delay is not None:
        search_algorithm.maximum_stable_delay = maximum_stable_delay
    config.lib_rasr.search_algorithm = search_algorithm

    write_job = WriteRasrConfigJob(config, post_config)
    return write_job.out_config
