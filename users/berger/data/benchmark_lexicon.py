"""Acoustic inventories and corpus word lists for the isolated Open ASR suite."""

import json
from collections import OrderedDict

from sisyphus import Job, Task
from i6_core.lib.lexicon import Lexicon, Lemma
from i6_core.util import write_xml, uopen

from .huggingface import _get_labels, _load_tokenizer, _sanitize_rasr_label


class BenchmarkInventoryJob(Job):
    """Keep softmax indices intact, including AED special symbols and RNNT blank.

    The shared pronunciation builder temporarily reserves index zero for blank
    when building AED lexica. Only a special tokenizer symbol may occupy it;
    here it is restored, and the sentence-end lemma gets the actual EOS label.
    """

    def __init__(self, repo, family, *, code_version, lexicon=None):
        self.repo, self.family, self.code_version, self.lexicon = repo, family, code_version, lexicon
        self.out_vocab = self.output_path("labels.txt")
        self.out_lexicon = self.output_path("lexicon.xml.gz") if lexicon is not None else None
        self.out_metadata = self.output_path("inventory.json")
        self.out_eos = self.output_var("eos")
        self.out_blank = self.output_var("blank")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 8, "time": 2})

    def run(self):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.repo.get_path())
        tokenizer = _load_tokenizer(self.repo)
        blank = getattr(config, "blank_token_id", 0) if self.family != "aed" else None
        eos = config.eos_token_id if self.family == "aed" else None
        if isinstance(eos, list):
            if len(eos) != 1:
                raise ValueError("Expected one AED EOS")
            eos = eos[0]
        labels = _get_labels(self.repo, blank_index=blank if blank is not None else 0, blank_label="<blank>")
        if self.family == "aed":
            if 0 not in tokenizer.all_special_ids or eos is None:
                raise ValueError("AED inventory requires a special id zero and an explicit EOS")
            labels[0] = _sanitize_rasr_label(tokenizer.convert_ids_to_tokens(0))
        if len(labels) != config.vocab_size or len(set(labels)) != len(labels):
            raise ValueError("Acoustic label inventory must bijectively match model outputs")
        with open(self.out_vocab.get_path(), "w") as f:
            f.write("\n".join(labels) + "\n")
        if self.lexicon is not None:
            lex = Lexicon()
            lex.load(self.lexicon.get_path())
            old_labels = list(lex.phonemes)
            expected = list(labels)
            if self.family == "aed":
                expected[0] = "<blank>"
            if old_labels != expected:
                raise ValueError("Pronunciation builder reordered acoustic labels")
            lex.phonemes = OrderedDict((p, "none") for p in labels)
            for lemma in lex.lemmata:
                if lemma.special == "unknown":
                    lemma.phon = []
                    if lemma.synt != ["<UNK>"]:
                        raise ValueError("Expected LibriSpeech unknown token <UNK>")
                if self.family == "aed" and lemma.special == "sentence-end":
                    lemma.phon = [labels[eos]]
                if self.family == "aed" and lemma.special == "sentence-begin":
                    lemma.phon = []  # prompt lives in the scorer, never an epsilon tree exit
            if self.family == "aed":
                lex.lemmata = [lemma for lemma in lex.lemmata if lemma.special != "blank"]
            for lemma in lex.lemmata:
                for pron in lemma.phon:
                    if any(p not in lex.phonemes for p in pron.split()):
                        raise ValueError(f"Unknown pronunciation label: {pron}")
            write_xml(self.out_lexicon.get_path(), lex.to_xml())
        self.out_eos.set(eos if eos is not None else -1)
        self.out_blank.set(blank if blank is not None else -1)
        with open(self.out_metadata.get_path(), "w") as f:
            json.dump(
                dict(
                    family=self.family,
                    size=len(labels),
                    blank=blank,
                    eos=eos,
                    state_tying="monophone",
                    states_per_phone=1,
                ),
                f,
                indent=2,
            )


class PublicWordListLexiconJob(Job):
    """Retain corpus LM spellings, discarding its *phonetic* pronunciations.

    Downstream the ASR tokenizer supplies new pronunciations. This is a training
    resource, never a word list extracted from benchmark reference transcripts.
    """

    def __init__(self, word_list, *, code_version):
        self.word_list, self.code_version = word_list, code_version
        self.out_lexicon = self.output_path("words.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex, words = Lexicon(), set()
        with uopen(self.word_list, "rt") as f:
            for line in f:
                fields = line.split()
                if fields and not fields[0].startswith(("<", "[", "!")):
                    words.add(fields[0])
        if not words:
            raise ValueError("Public lexicon contains no ordinary words")
        for word in sorted(words):
            lex.add_lemma(Lemma(orth=[word]))
        write_xml(self.out_lexicon.get_path(), lex.to_xml())
