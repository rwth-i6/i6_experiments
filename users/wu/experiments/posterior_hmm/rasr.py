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


class CreateLibrasrVenvWithKenLMJob(CreateLibrasrVenvJob):
    """
    Like CreateLibrasrVenvJob, but additionally builds+installs the python `kenlm` module
    from a kenlm repo checkout with ``MAX_ORDER`` set (KenLM defaults to 6, which can't LOAD
    an order>6 model). The recog-time n-gram label scorer imports this `kenlm`.

    Built from source rather than PyPI because CreateLibrasrVenvJob's plain `pip install`
    can't pass the build-time max order; kenlm's setup.py reads it from the MAX_ORDER env var.
    """

    def __init__(self, *, kenlm_repository: tk.Path, kenlm_max_order: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.kenlm_repository = kenlm_repository
        self.kenlm_max_order = kenlm_max_order

    def run(self):
        super().run()
        import tempfile
        import shutil
        from sisyphus import gs

        venv_python = self.out_venv_dir.join_right("bin/python").get_path()
        env = dict(os.environ, MAX_ORDER=str(self.kenlm_max_order))
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as td:
            # copy to a writable dir (pip may build in-tree; the repo output is read-only)
            repo = os.path.join(td, "kenlm")
            shutil.copytree(tk.uncached_path(self.kenlm_repository), repo)
            sp.check_call([venv_python, "-m", "pip", "install", "--no-build-isolation", repo], env=env)

    @classmethod
    def hash(cls, kwargs):
        kwargs = dict(kwargs)
        kenlm_repository = kwargs.pop("kenlm_repository")
        kenlm_max_order = kwargs.pop("kenlm_max_order")
        # Reuse the parent's venv-input hashing, then fold in the kenlm build params.
        base = CreateLibrasrVenvJob.hash(kwargs)
        return Job.hash(
            {"venv": base, "kenlm_repository": kenlm_repository, "kenlm_max_order": kenlm_max_order}
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


class BuildPhonLexiconfreeLexiconJob(Job):
    """
    Build a Bliss lexicon for lexicon-free pHMM recognition.

    The output has exactly one lemma per output label, in label-index order:
    `[SILENCE]` at index 0 (marked `special="blank"` so the search treats silence
    as a blank-style skip), then the EOW phonemes from the source pHMM lexicon,
    then `<s>` and `</s>` (latter marked `special="sentence-end"`).
    """

    def __init__(
        self,
        bliss_lexicon: tk.Path,
        silence_phoneme: str = "[SILENCE]",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
    ):
        super().__init__()
        self.bliss_lexicon = bliss_lexicon
        self.silence_phoneme = silence_phoneme
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.out_lexicon = self.output_path("lexicon.xml.gz")
        self.out_vocab_size = self.output_var("vocab_size")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        src = Lexicon()
        src.load(tk.uncached_path(self.bliss_lexicon))

        ordered_phonemes = list(src.phonemes.keys())
        assert ordered_phonemes and ordered_phonemes[0] == self.silence_phoneme, (
            "Source pHMM lexicon must have %r as the first phoneme, got %r"
            % (self.silence_phoneme, ordered_phonemes[:3])
        )
        assert self.bos_token not in ordered_phonemes and self.eos_token not in ordered_phonemes

        out = Lexicon()
        out.phonemes = OrderedDict()
        for phon in ordered_phonemes:
            out.add_phoneme(phon, variation="none")
        out.add_phoneme(self.bos_token, variation="none")
        out.add_phoneme(self.eos_token, variation="none")

        # Lemma order = label index order. Bliss assigns lemma->id() in add-order.
        out.add_lemma(
            Lemma(
                orth=[self.silence_phoneme],
                phon=[self.silence_phoneme],
                synt=[],
                eval=[[]],
                special="blank",
            )
        )
        for phon in ordered_phonemes[1:]:
            out.add_lemma(Lemma(orth=[phon], phon=[phon]))
        out.add_lemma(Lemma(orth=[self.bos_token], phon=[self.bos_token]))
        out.add_lemma(
            Lemma(
                orth=[self.eos_token],
                phon=[self.eos_token],
                synt=[],
                eval=[[]],
                special="sentence-end",
            )
        )

        write_xml(self.out_lexicon.get_path(), out.to_xml())
        self.out_vocab_size.set(len(ordered_phonemes) + 2)


class BuildEowPhonCtcLexiconJob(Job):
    """Build an EOW phoneme Bliss lexicon for **CTC** training (FSA, ``topology="ctc"``) and
    LibRASR recognition.

    Mirrors :class:`NormalizePhonForPhmmLexiconJob` but with the CTC blank in place of silence:
    ``[BLANK]`` is the **first phoneme (index 0)**, marked ``special="blank"`` (so both the RASR
    CTC FSA builder and the LibRASR search infer the blank emission index), followed by the source
    EOW phonemes. Any ``unknown`` lemma is remapped to ``[BLANK]``; sentence-begin/end lemmata are
    dropped (re-added for LM recognition by :class:`AddSentenceBoundaryLemmataToPhmmLexiconJob`).

    Index layout matches the pHMM lexicon ([SILENCE] at 0): the AM softmax has
    ``label_target_size = #EOW_phonemes + 1`` outputs with blank at index 0.
    """

    def __init__(self, bliss_lexicon: tk.Path, blank_phoneme: str = "[BLANK]"):
        super().__init__()
        self.bliss_lexicon = bliss_lexicon
        self.blank_phoneme = blank_phoneme

        self.out_lexicon = self.output_path("lexicon.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = Lexicon()
        lex.load(tk.uncached_path(self.bliss_lexicon))

        removed_specials = {"sentence-begin", "sentence-end"}

        out_lex = Lexicon()
        out_lex.phonemes = OrderedDict()

        # [BLANK] must be the first phoneme (index 0), like [SILENCE] in the pHMM lexicon.
        out_lex.add_phoneme(self.blank_phoneme, variation="none")
        for symbol in lex.phonemes.keys():
            if symbol in (self.blank_phoneme, "[UNKNOWN]"):
                continue
            out_lex.add_phoneme(symbol, variation=lex.phonemes[symbol])

        # Blank lemma (empty orth so it can be emitted without consuming a word).
        out_lex.add_lemma(
            Lemma(
                orth=[self.blank_phoneme, ""],
                phon=[self.blank_phoneme],
                synt=[],
                eval=[[]],
                special="blank",
            )
        )

        for lemma in lex.lemmata:
            if lemma.special in removed_specials:
                continue
            if lemma.special == "blank":
                continue  # already added above
            if lemma.special == "unknown":
                out_lex.add_lemma(
                    Lemma(
                        orth=list(lemma.orth),
                        phon=[self.blank_phoneme],
                        synt=lemma.synt,
                        eval=lemma.eval,
                        special="unknown",
                    )
                )
                continue
            out_lex.add_lemma(lemma)

        write_xml(self.out_lexicon.get_path(), out_lex.to_xml())


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


def build_lexiconfree_phmm_recognition_config(
    lexicon_path: tk.Path,
    onnx_state_initializer: tk.Path,
    onnx_state_updater: tk.Path,
    onnx_scorer: tk.Path,
    *,
    lm_scale: float = 0.5,
    am_scale: float = 1.0,
    max_beam_size: int = 32,
    score_threshold: float = 14.0,
    loop_updates_history: bool = False,
    blank_updates_history: bool = False,
    log_stepwise_statistics: bool = True,
    logfile_suffix: str = "lexfree_recog",
) -> tk.Path:
    """
    Build a LibRASR recognition config for lexicon-free phoneme search with a
    stateful-ONNX neural LM as the second sub-scorer of a `combine` LabelScorer.

    The AM is supplied externally via the `no-op` sub-scorer (the Python side
    pushes `-log p(token | frame)` log-probs, padded with `-inf` at `<s>`/`</s>`
    indices). The LM operates in pure-LM mode: none of the three ONNX models
    declare an `encoder-states` mapping, so RASR does not attempt to feed
    acoustic features into the LM scorer.
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

    # Combined label scorer: scorer-1 = AM (no-op), scorer-2 = NN LM (stateful-onnx).
    # NOTE: the combined scorer's enabled-transition set is the *intersection* of
    # its sub-scorers' transition presets (no-op defaults to CTC preset; LM scorer
    # defaults to LM preset). SENTENCE_END is dropped by that intersection, so the
    # LM's `</s>` log-prob is NOT added to hypotheses at sentence-end finalization.
    # The search still appends `</s>` to the traceback unconditionally. To recover
    # the LM EOS contribution we would need to either (a) pad one extra AM frame
    # with `0` only at the `</s>` index, or (b) override the no-op preset to "lm".
    label_scorer = RasrConfig()
    label_scorer.type = "combine"
    label_scorer.num_scorers = 2

    am_scorer = RasrConfig()
    am_scorer.type = "no-op"
    am_scorer.scale = am_scale
    label_scorer["scorer-1"] = am_scorer

    lm_scorer = RasrConfig()
    lm_scorer.type = "stateful-onnx"
    lm_scorer.scale = lm_scale
    lm_scorer.loop_updates_history = loop_updates_history
    lm_scorer.blank_updates_history = blank_updates_history
    lm_scorer.scorer_model = RasrConfig()
    lm_scorer.scorer_model.file = onnx_scorer
    lm_scorer.state_initializer_model = RasrConfig()
    lm_scorer.state_initializer_model.file = onnx_state_initializer
    lm_scorer.state_updater_model = RasrConfig()
    lm_scorer.state_updater_model.file = onnx_state_updater
    # io-map: link RASR's IOSpec name "token" to the actual ONNX input name
    # produced by ExportStatefulOnnxLMJob (see stateful_onnx_v1.StateUpdater).
    lm_scorer.state_updater_model.io_map = RasrConfig()
    lm_scorer.state_updater_model.io_map["token"] = "token"
    label_scorer["scorer-2"] = lm_scorer

    config.lib_rasr.label_scorer = label_scorer

    search_algorithm = RasrConfig()
    search_algorithm.type = "lexiconfree-timesync-beam-search"
    search_algorithm.max_beam_size = max_beam_size
    search_algorithm.score_threshold = score_threshold
    search_algorithm.collapse_repeated_labels = False
    search_algorithm.log_stepwise_statistics = log_stepwise_statistics
    # blank-label-index and sentence-end-label-index are inferred from the
    # `special="blank"` / `special="sentence-end"` lemmata in the lexicon.
    config.lib_rasr.search_algorithm = search_algorithm

    write_job = WriteRasrConfigJob(config, post_config)
    return write_job.out_config


def build_lexiconfree_count_recognition_config(
    lexicon_path: tk.Path,
    *,
    scorer_name: str = "ngram-phon",
    lm_scale: float = 0.5,
    am_scale: float = 1.0,
    max_beam_size: int = 32,
    score_threshold: float = 14.0,
    log_stepwise_statistics: bool = True,
    logfile_suffix: str = "lexfree_count_recog",
) -> tk.Path:
    """
    Like :func:`build_lexiconfree_phmm_recognition_config`, but the LM sub-scorer is a
    **count** phoneme n-gram exposed as a custom Python label scorer (KenLM), instead of
    the stateful-ONNX neural LM.

    The scorer is registered at runtime under ``scorer_name`` by the decoder
    (``rasr_phmm_lexfree_ngram_v1.ForwardCallback._setup_lexicon`` ->
    ``ngram_label_scorer.register_ngram_label_scorer``); here we only reference that name as
    ``scorer-2.type``. The Python scorer takes its KenLM path / vocab from the decoder config
    (via closure), so no scorer parameters are written into the RASR config.

    The AM is still supplied via the ``no-op`` sub-scorer (``-log p(token|frame)``, padded
    with ``-inf`` at the ``<s>``/``</s>`` indices), exactly as in the neural lexfree path.
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

    # Combined label scorer: scorer-1 = AM (no-op), scorer-2 = count phoneme n-gram (Python).
    # See the note in build_lexiconfree_phmm_recognition_config about the transition-set
    # intersection dropping SENTENCE_END (the search appends </s> to the traceback anyway).
    label_scorer = RasrConfig()
    label_scorer.type = "combine"
    label_scorer.num_scorers = 2

    am_scorer = RasrConfig()
    am_scorer.type = "no-op"
    am_scorer.scale = am_scale
    label_scorer["scorer-1"] = am_scorer

    lm_scorer = RasrConfig()
    lm_scorer.type = scorer_name  # custom Python KenLM label scorer, registered at runtime
    lm_scorer.scale = lm_scale
    label_scorer["scorer-2"] = lm_scorer

    config.lib_rasr.label_scorer = label_scorer

    search_algorithm = RasrConfig()
    search_algorithm.type = "lexiconfree-timesync-beam-search"
    search_algorithm.max_beam_size = max_beam_size
    search_algorithm.score_threshold = score_threshold
    search_algorithm.collapse_repeated_labels = False
    search_algorithm.log_stepwise_statistics = log_stepwise_statistics
    config.lib_rasr.search_algorithm = search_algorithm

    write_job = WriteRasrConfigJob(config, post_config)
    return write_job.out_config


def build_ctc_am_config(
    state_tying: str = "monophone",
    tdp_scale: float = 1.0,
    tdp_applicator_type: str = "corrected",
) -> RasrConfig:
    """
    Acoustic model config for CTC LibRASR recognition.

    CTC uses a single emitting state per label with free self-loops/forwards (the blank label
    absorbs the "stay" mass), so the transition penalties are flat (all zero) and skips are
    forbidden -- this is the same monophone, 1-state-per-phone setup as ``build_phmm_am_config``
    with neutral TDPs. The blank label is identified by the ``special="blank"`` lemma in the
    lexicon, not by the AM config.
    """
    return build_phmm_am_config(
        states_per_phone=1,
        state_repetitions=1,
        state_tying=state_tying,
        tdp_scale=tdp_scale,
        tdp_applicator_type=tdp_applicator_type,
        tdp_loop=0.0,
        tdp_forward=0.0,
        tdp_skip="infinity",
        tdp_exit=0.0,
        silence_loop=0.0,
        silence_forward=0.0,
        silence_skip="infinity",
        silence_exit=0.0,
    )


def build_librasr_ctc_recognition_config(
    lexicon_path: tk.Path,
    lm_config: Optional[RasrConfig] = None,
    am_config: Optional[RasrConfig] = None,
    max_beam_size: int = 1024,
    intermediate_max_beam_size: int = 1024,
    score_threshold: float = 18.0,
    intermediate_score_threshold: float = 18.0,
    sentence_end_fallback: bool = True,
    log_stepwise_statistics: bool = True,
    logfile_suffix: str = "ctc_recog",
) -> tk.Path:
    """
    LibRASR recognition config for **CTC** lexical search.

    Same tree-timesync beam search as the pHMM, but with CTC label semantics enabled:
    ``collapse_repeated_labels`` and ``force_blank_between_repeated_labels`` are turned on, and
    the blank label index is inferred from the ``special="blank"`` lemma of the CTC lexicon
    (see :class:`BuildEowPhonCtcLexiconJob`). The AM posteriors (incl. the blank dimension) are
    fed through the ``no-op`` label scorer exactly as for the pHMM.
    """
    return build_librasr_phmm_recognition_config(
        lexicon_path=lexicon_path,
        lm_config=lm_config,
        am_config=am_config if am_config is not None else build_ctc_am_config(),
        max_beam_size=max_beam_size,
        intermediate_max_beam_size=intermediate_max_beam_size,
        score_threshold=score_threshold,
        intermediate_score_threshold=intermediate_score_threshold,
        sentence_end_fallback=sentence_end_fallback,
        log_stepwise_statistics=log_stepwise_statistics,
        tree_builder_type="hmm",
        collapse_repeated_labels=True,
        force_blank_between_repeated_labels=True,
        logfile_suffix=logfile_suffix,
    )


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
    tree_builder_type: str = "hmm",
    collapse_repeated_labels: bool = False,
    force_blank_between_repeated_labels: bool = False,
    logfile_suffix: str = "recog",
) -> tk.Path:
    """
    Build a LibRASR recognition config for phoneme lexical (tree-timesync) search.

    Defaults reproduce the pHMM prototype (HMM tree builder, no repeated-label collapse).
    For CTC recognition use :func:`build_librasr_ctc_recognition_config`, which flips
    ``collapse_repeated_labels`` / ``force_blank_between_repeated_labels`` on so the search
    applies CTC label collapsing around the (lexicon-inferred) blank label.
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
    search_algorithm.tree_builder_type = tree_builder_type
    search_algorithm.collapse_repeated_labels = collapse_repeated_labels
    search_algorithm.force_blank_between_repeated_labels = force_blank_between_repeated_labels
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
