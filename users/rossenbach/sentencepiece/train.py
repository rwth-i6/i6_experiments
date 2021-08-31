from enum import Enum
from typing import *
import subprocess

from sisyphus import *


class SentencePieceType(Enum):
    UNIGRAM = "unigram",
    BPE = "bpe",
    CHAR = "char",
    WORD = "word",


class TrainSentencePiece(Job):
    """

    """

    def __init__(self, training_text, vocab_size, model_type, **opts):
        """

        :param tk.Path training_text: raw text or gzipped text
        :param int vocab_size:
        :param SentencePieceType model_type:
        :param dict opts: trainer options
        """

        self.training_text = training_text
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.opts = opts

        self.out_model = self.output_path("spm_out.model")
        self.out_vocab_size = self.output_var("vocab_size")

        self.rqmt = {'cpu': 1, 'mem': 2, 'time': 4}

    def tasks(self):
        yield Task('run', rqmt=self.rqmt)

    def run(self):
        import sentencepiece

        training_text_path = self.training_text.get_path()
        if training_text_path.endswith(".gz"):
            training_text_path = "unzipped_training_text.txt"
            outfile = open(training_text_path, "wt")
            subprocess.check_call(["gzip", "-dc", training_text_path], stdout=outfile)

        sentencepiece.SentencePieceTrainer.Train(
            input=training_text_path,
            model_prefix="spm_out",
            model_type=self.model_type.value,
            vocab_size=self.vocab_size,
            **self.opts
        )

        #processor = sentencepiece.SentencePieceProcessor(model_file=training_text_path)
        #self.out_vocab_size.set(processor.vocab_size())

        self.out_vocab_size.set(self.vocab_size)



