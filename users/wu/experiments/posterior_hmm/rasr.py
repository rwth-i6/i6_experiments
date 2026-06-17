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
    # NO empty "" orth (and no lowercase "[blank]" variant): an empty <orth/> on the blank lemma makes
    # RASR's orthographic parser (Bliss/Orthography.cc) insert [BLANK] as the optional word-boundary
    # filler, which collides with the topology=ctc addBlank pass (blank-on-blank) and makes the fbw2
    # full-sum over-count alignments (loss < true CTC NLL). This matches the proven BuildEowPhonCtcLexiconJob
    # blank lemma (orth=[blank] only); see its note and [[ctc-fsa-overcounts-vs-torch]]. The sibling
    # DEFAULT_PHMM_BPE_RASR_CONFIG keeps "" on purpose -- benign for the [SILENCE]/hmm topology.
    special_orths=("[BLANK]",),
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
            raw_pron = bpe.split()
            # The lexicon phoneme names rewrite "." -> "_" (below), but any count/neural LM trained on
            # the BPE text (data/bpe_lm.ApplyBPEToTextJob) keeps the RAW subword tokens (no rewrite). A
            # literal "." in a BPE token would make the lexicon symbol ("_"-form) diverge from the LM
            # token ("."-form) and silently score as <unk> in the lexicon-free n-gram scorer. Normalized
            # LibriSpeech (A-Z + apostrophe) never produces a "." token, so fail loudly here instead of
            # corrupting LM scores downstream. (No-op for the existing pHMM-BPE lexicon; run() is not hashed.)
            assert all("." not in token for token in raw_pron), (
                f"BPE token contains '.', which breaks lexicon<->LM token matching for word {word!r}: {bpe!r}"
            )
            pron = [token.replace(".", "_") for token in raw_pron]
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


class MakeLexiconContextIndependentJob(Job):
    """Rewrite a Bliss lexicon so that **all phonemes are context-independent** (``variation="none"``).

    Required for LibRASR CTC lexical search: the ``ctc`` tree builder
    (:cpp:class:`Search::CtcTreeBuilder`, ``search-algorithm.tree-builder-type=ctc``) asserts that
    no phoneme is context-dependent, whereas the EOW phoneme lexicon inherits ``variation="context"``
    from the source pHMM lexicon (see :class:`BuildEowPhonCtcLexiconJob`).

    Because the CTC AM uses single-state monophone state tying, flipping the variation flag does not
    change any label's emission index -- the phoneme order is preserved, so blank stays at index 0
    and every EOW phoneme keeps its column in the AM softmax. Only the search-tree construction is
    affected. Apply this to the **recognition** lexicon only; the training lexicon (which feeds the
    ``topology="ctc"`` FSA and hence the trained model) must stay untouched to avoid invalidating it.
    """

    def __init__(self, bliss_lexicon: tk.Path):
        super().__init__()
        self.bliss_lexicon = bliss_lexicon
        self.out_lexicon = self.output_path("lexicon.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = Lexicon()
        lex.load(tk.uncached_path(self.bliss_lexicon))
        lex.phonemes = OrderedDict((symbol, "none") for symbol in lex.phonemes)
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

    # 2026-06-08: dropped the empty "" orth on the blank lemma. The empty orth made RASR's
    # orthographic parser insert [BLANK] as the optional word-boundary filler, which collided with
    # the topology=ctc addBlank pass (blank-on-blank) and made the fbw2 full-sum over-count
    # alignments (loss below true CTC NLL). Bump to invalidate the old lexicon hash and force retrain.
    __sis_version__ = 1

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

        # Blank lemma. NOTE: no empty "" orth here (unlike the pHMM silence lemma). An empty orth
        # makes RASR's orthographic parser (Bliss/Orthography.cc) insert this lemma as the optional
        # word-boundary filler; for CTC that filler is [BLANK], so the base FSA already carries a
        # boundary blank that then collides with the topology=ctc addBlank pass (blank-on-blank),
        # making the fbw2 full-sum over-count alignments (loss < true CTC NLL, negative when
        # confident). The CTC blank mechanism handles inter-word gaps on its own; no filler wanted.
        out_lex.add_lemma(
            Lemma(
                orth=[self.blank_phoneme],
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
    collapse_repeated_labels: bool = False,
    max_cached_score_vectors: int = 256,
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

    # Two TOP-LEVEL label scorers applied as a CASCADE (NOT a `combine`).
    #
    # Why not `combine`: a `combine` scorer's enabled-transition set is the INTERSECTION of its
    # sub-scorers' presets (CombineLabelScorer.cc:27,32). The AM `no-op` uses the CTC preset (all
    # label/blank/loop transitions) and the LM uses the LM preset (label-emitting only), so
    # combine = CTC ∩ LM = {label-to-label, blank-to-label, initial-label} -- this DROPS every
    # blank/loop transition. The search only ever queries the top-level scorer's transition gate
    # (LexiconfreeTimesyncBeamSearch.cc:421) and the combined accessor sums sub-scores
    # unconditionally (ScoreAccessor.cc:66), so the AM's acoustic cost on blank/loop frames is
    # NEVER added. An all-blank path then accrues zero cost and always wins -> the search returns
    # an EMPTY hypothesis (observed: 2703/2703 empty, 100% WER).
    #
    # A top-level cascade fixes this: ModelCombination reads `[label-scorer-1]`/`[label-scorer-2]`
    # when `num-label-scorers > 1` (ModelCombination.cc:75-83), and the search applies the scorers
    # one after another, each with ITS OWN transition gate, accumulating into the hyp score
    # (LexiconfreeTimesyncBeamSearch.cc:401-428, with an intermediate beam prune between stages).
    # So the AM (CTC preset) scores all transitions incl. blank, the LM (LM preset) scores only
    # label-emitting ones, exactly as intended. `put_features` feeds the AM matrix to every scorer
    # (cc:308-311); the LM scorers' `add_inputs` is a no-op, so this is harmless.
    config.lib_rasr.num_label_scorers = 2

    am_scorer = RasrConfig()
    am_scorer.type = "no-op"  # CTC transition preset baked in (NoOpLabelScorer.cc:23) -> scores blank/loop too
    am_scorer.scale = am_scale
    config.lib_rasr["label-scorer-1"] = am_scorer

    lm_scorer = RasrConfig()
    lm_scorer.type = "stateful-onnx"  # LM transition preset baked in (StatefulOnnxLabelScorer.cc)
    lm_scorer.scale = lm_scale
    lm_scorer.loop_updates_history = loop_updates_history
    lm_scorer.blank_updates_history = blank_updates_history
    # Bound the per-segment memoization caches. The scorer keeps two FIFO caches keyed by scoring
    # context (scoreCache_ + stateCache_); stateCache_ is sized to `max-cached-score-vectors` too
    # (StatefulOnnxLabelScorer.cc:55-58,132). Each cached hidden state is the FULL KV cache --
    # num_layers x [1, t_max, hidden] f32 ~= 25 MB at 12x1024x512 -- so the default cap of 1000
    # lets a single long segment grow toward ~25 GB of HOST RAM and get OOM-killed (observed: CPU
    # recog killed at the 16 GB cgroup limit, monotonic climb during the longest segment's search).
    # The caches are pure memoization (a miss just recomputes via the state-updater) and are cleared
    # per segment (recognizeSegment -> reset()), so a smaller cap trades a little recompute for a
    # bounded, predictable footprint: 256 -> <=6.4 GB. Note the LM ONNX always runs on the CPU
    # execution provider, so this host-RAM bound matters even when the AM forward is on GPU.
    lm_scorer.max_cached_score_vectors = max_cached_score_vectors
    # Each of the three sub-models is an `Onnx::Model`, which reads its file path from a NESTED
    # `session` sub-config -- NOT directly from `<model>.file`. `Onnx::Model` does
    # `session(select("session"))` (Onnx/Model.cc:22) and `Onnx::Session` reads `paramFile("file")`
    # at the session scope, default "" (Onnx/Session.cc:16,46). Writing the path at
    # `<model>.file` therefore leaves the session's `file` EMPTY and RASR aborts while building the
    # SearchAlgorithm with `Load model from  failed:Load model  failed. File doesn't exist` (note
    # the empty path). The correct key is `<model>.session.file`.
    lm_scorer.scorer_model = RasrConfig()
    lm_scorer.scorer_model.session = RasrConfig()
    lm_scorer.scorer_model.session.file = onnx_scorer
    # The scorer's `scores` OUTPUT is a NON-optional IOSpec entry (StatefulOnnxLabelScorer.cc:61-68),
    # so it MUST be present in the model's io-map or `IOValidator` aborts construction with
    # "required input/output 'scores' is missing from mapping" (Onnx/IOSpecification.cc:52-58, strict
    # by default). The exported ONNX output is literally named "scores" (export_onnx.py), so this is
    # an identity mapping -- but it must still be declared explicitly (an absent io-map entry is
    # treated as "unmapped", not "same name").
    lm_scorer.scorer_model.io_map = RasrConfig()
    lm_scorer.scorer_model.io_map["scores"] = "scores"
    lm_scorer.state_initializer_model = RasrConfig()
    lm_scorer.state_initializer_model.session = RasrConfig()
    lm_scorer.state_initializer_model.session.file = onnx_state_initializer
    # No io-map for the initializer: its `encoder-states`/`encoder-states-size` IOSpec entries are
    # OPTIONAL (so IOValidator skips them when unmapped) and unused for a pure LM. Crucially, the
    # initializer ONNX has ZERO graph inputs -- `do_constant_folding` dropped the dead
    # `dummy_es_size` input at export -- so RASR runs it input-free (computeInitialHiddenState only
    # feeds encoder-states/size when those names are mapped, StatefulOnnxLabelScorer.cc:368-377).
    # Leaving them unmapped also avoids the feature-deferral branch (cc:271) that would otherwise
    # make the LM wait for all acoustic frames before scoring.
    lm_scorer.state_updater_model = RasrConfig()
    lm_scorer.state_updater_model.session = RasrConfig()
    lm_scorer.state_updater_model.session.file = onnx_state_updater
    # io-map: link RASR's IOSpec name "token" to the actual ONNX input name "token" produced by
    # ExportStatefulOnnxLMJob (see stateful_onnx_v1.StateUpdater). The hidden-state inputs/outputs
    # (CACHE_i, POS, LAST_LOGITS) are threaded via ONNX custom metadata, not the io-map.
    lm_scorer.state_updater_model.io_map = RasrConfig()
    lm_scorer.state_updater_model.io_map["token"] = "token"
    config.lib_rasr["label-scorer-2"] = lm_scorer

    search_algorithm = RasrConfig()
    search_algorithm.type = "lexiconfree-timesync-beam-search"
    # `max-beam-size`/`score-threshold` are vector params (space-separated), one entry per cascade
    # stage; they MUST have >= num-label-scorers entries or the search errors
    # (LexiconfreeTimesyncBeamSearch.cc:207). Stage 1 prunes on the AM score alone (loose, just an
    # acoustic pre-prune), stage 2 prunes on the accumulated AM+LM score.
    search_algorithm.max_beam_size = [max_beam_size, max_beam_size]
    search_algorithm.score_threshold = [score_threshold, score_threshold]
    # CTC collapses repeated emissions of the same label into one output (set True for the CTC AM);
    # the pHMM posterior topology emits every frame, so it stays False.
    search_algorithm.collapse_repeated_labels = collapse_repeated_labels
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
    collapse_repeated_labels: bool = False,
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

    # Two TOP-LEVEL label scorers applied as a CASCADE (NOT a `combine`); see the detailed note in
    # build_lexiconfree_phmm_recognition_config. A `combine` intersects sub-scorer transition sets
    # (CTC ∩ LM), dropping every blank/loop transition so the AM is never scored on blank frames ->
    # the all-blank path is free -> the search emits an EMPTY hypothesis. The count path is doubly
    # affected: the Python n-gram scorer defaults to transition-preset NONE
    # (Python::PythonLabelScorer passes no preset), so combine would even be CTC ∩ NONE = EMPTY.
    # The cascade applies each scorer's own gate, so the AM (CTC) scores all transitions incl.
    # blank and the n-gram LM (LM preset, set explicitly below since the Python default is NONE)
    # scores only label-emitting transitions.
    config.lib_rasr.num_label_scorers = 2

    am_scorer = RasrConfig()
    am_scorer.type = "no-op"  # CTC transition preset baked in (NoOpLabelScorer.cc:23) -> scores blank/loop too
    am_scorer.scale = am_scale
    config.lib_rasr["label-scorer-1"] = am_scorer

    lm_scorer = RasrConfig()
    lm_scorer.type = scorer_name  # custom Python KenLM label scorer, registered at runtime
    lm_scorer.scale = lm_scale
    # Python label scorers default to transition-preset "none" (PythonLabelScorer passes no preset
    # -> Nn::LabelScorer default NONE). Set it to "lm" so this scorer behaves like the C++
    # StatefulOnnxLabelScorer (which bakes in "lm"): score only label-emitting transitions.
    lm_scorer.transition_preset = "lm"
    config.lib_rasr["label-scorer-2"] = lm_scorer

    search_algorithm = RasrConfig()
    search_algorithm.type = "lexiconfree-timesync-beam-search"
    # One (beam, threshold) per cascade stage; >= num-label-scorers entries required
    # (LexiconfreeTimesyncBeamSearch.cc:207). Stage 1 = AM-only pre-prune, stage 2 = AM+LM prune.
    search_algorithm.max_beam_size = [max_beam_size, max_beam_size]
    search_algorithm.score_threshold = [score_threshold, score_threshold]
    # See build_lexiconfree_phmm_recognition_config: True for the CTC AM, False for the pHMM AM.
    search_algorithm.collapse_repeated_labels = collapse_repeated_labels
    search_algorithm.log_stepwise_statistics = log_stepwise_statistics
    config.lib_rasr.search_algorithm = search_algorithm

    write_job = WriteRasrConfigJob(config, post_config)
    return write_job.out_config


def build_lexiconfree_neural_python_recognition_config(
    lexicon_path: tk.Path,
    *,
    scorer_name: str = "neural-phon",
    lm_scale: float = 0.5,
    am_scale: float = 1.0,
    max_beam_size: int = 32,
    score_threshold: float = 14.0,
    collapse_repeated_labels: bool = False,
    log_stepwise_statistics: bool = True,
    logfile_suffix: str = "lexfree_neural_py_recog",
) -> tk.Path:
    """
    Like :func:`build_lexiconfree_count_recognition_config`, but the ``label-scorer-2``
    Python scorer is the **neural Transformer** phoneme LM run directly in torch on the
    GPU (registered at runtime by ``rasr_phmm_lexfree_neural_v1.ForwardCallback._setup_lexicon``
    -> ``neural_label_scorer.register_neural_label_scorer``), instead of the count KenLM
    scorer or the CPU-only ``stateful-onnx`` scorer.

    The RASR-side config is byte-identical to the count path (a Python label scorer with
    ``transition-preset=lm`` as ``scorer-2`` of an AM/LM cascade); only the registered
    ``scorer_name`` and the runtime Python class differ. The torch LM checkpoint / net_args
    reach the scorer through the decoder config (closure), so nothing LM-specific is written
    into the RASR config. See :func:`build_lexiconfree_count_recognition_config` for the full
    rationale on the cascade (vs. ``combine``) and the explicit ``transition-preset``.
    """
    return build_lexiconfree_count_recognition_config(
        lexicon_path=lexicon_path,
        scorer_name=scorer_name,
        lm_scale=lm_scale,
        am_scale=am_scale,
        max_beam_size=max_beam_size,
        score_threshold=score_threshold,
        collapse_repeated_labels=collapse_repeated_labels,
        log_stepwise_statistics=log_stepwise_statistics,
        logfile_suffix=logfile_suffix,
    )


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

    Same tree-timesync beam search as the pHMM, but with the dedicated CTC tree builder
    (``tree-builder-type=ctc``) and CTC label semantics enabled: the ``ctc`` builder interleaves
    blank states (with label/blank self-loops) into the search tree, ``collapse_repeated_labels``
    and ``force_blank_between_repeated_labels`` are turned on, and the blank label index is inferred
    from the ``special="blank"`` lemma of the CTC lexicon (see :class:`BuildEowPhonCtcLexiconJob`).
    The AM posteriors (incl. the blank dimension) are fed through the ``no-op`` label scorer exactly
    as for the pHMM.

    The lexicon passed here must have **context-independent** phonemes (``variation="none"``); the
    ``ctc`` tree builder asserts there are no context-dependent phonemes. Use
    :class:`MakeLexiconContextIndependentJob` on the recognition lexicon (the EOW phoneme lexicon
    inherits ``variation="context"`` from the pHMM source).
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
        tree_builder_type="ctc",
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
