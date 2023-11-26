import subprocess
import shutil
import os

from recipe.i6_core.util import create_executable

from sisyphus import *

import recipe.i6_private.users.gruev.tools as tools_mod

tools_dir = os.path.dirname(tools_mod.__file__)


class BpeStatisticsJob(Job):
    def __init__(self, bliss_lexicon, transcription, returnn_python_exe=None):
        """
        :param bliss_lexicon: Bliss lexicon with BPE subword units as <phon> sequences
        :param transcription: A text file with utterance transcriptions
        """

        self.bliss_lexicon = bliss_lexicon
        self.transcription = transcription

        if returnn_python_exe is not None:
            self.returnn_python_exe = returnn_python_exe
        else:
            self.returnn_python_exe = gs.RETURNN_PYTHON_EXE

        ## Example:
        # Text: AMAZINGLY COMPLICATED COMPLICATED WANAMAKER
        # Segmentation: AMA@@ ZINGLY COMP@@ LIC@@ ATED COMP@@ LIC@@ ATED WAN@@ AMA@@ KER

        # Num token per sequence: 1 sequence with 11 BPE tokens --> average is 11.
        # Num tokens per word: 4 words with 2 + 3 + 3 + 3 tokens --> average is 11/4 = 2.75
        # Num symbols per token: 7 tokens with 3 + 5 + 3 + 3 + 4 + 3 + 3 symbols --> average is 24/7 = 3.4
        # Token count per vocab: AMA@@ appears two times, COMP@@ and LIC@@ appear one time (per unique word!)
        # Token count per corpus: AMA@@ appears two times, COMP@@ and LIC@@ apear two times (per sequence!)

        # Summary of all other statistics, register as output
        self.out_bpe_statistics = self.output_path("bpe_statistics.txt")

        # Average number of BPE tokens for a sequence (utterance)
        self.out_mean_num_token_per_sequence = self.output_path(
            "mean_num_token_per_sequence.txt"
        )
        self.out_num_token_per_sequence_histogram = self.output_path(
            "num_token_per_sequence_histogram.pdf"
        )

        # Average number of BPE tokens for a single word in the corpus
        self.out_mean_num_token_per_word = self.output_path(
            "mean_num_token_per_word.txt"
        )
        self.out_num_token_per_word_histogram = self.output_path(
            "num_token_per_word_histogram.pdf"
        )

        # Average number of symbols that comprise a BPE token
        self.out_mean_num_symbols_per_token = self.output_path(
            "mean_num_symbols_per_token.txt"
        )
        self.out_num_symbols_per_token_histogram = self.output_path(
            "num_symbols_per_token_histogram.pdf"
        )

        # Number of BPE tokens per vocabulary (all words)
        self.out_mean_token_count_per_vocab = self.output_path(
            "mean_token_count_per_vocab.txt"
        )
        self.out_token_count_per_vocab = self.output_path("token_count_per_vocab.json")
        self.out_token_count_per_vocab_plot = self.output_path(
            "token_count_per_vocab_plot.pdf"
        )

        # Number of BPE tokens per corpus (all sequence)
        self.out_mean_token_count_per_corpus = self.output_path(
            "mean_token_count_per_corpus.txt"
        )
        self.out_token_count_per_corpus = self.output_path(
            "token_count_per_corpus.json"
        )
        self.out_token_count_per_corpus_plot = self.output_path(
            "token_count_per_corpus_plot.pdf"
        )

        # OOV words in corpus
        self.out_oov_words = self.output_path("oov_words.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 2, "time": 2})

    def run(self):
        command = [
            self.returnn_python_exe.get_path(),
            os.path.join(tools_dir, "bpe_statistics.py"),
            self.bliss_lexicon.get_path(),
            self.transcription.get_path(),
        ]

        create_executable("rnn.sh", command)
        subprocess.check_call(["./rnn.sh"])

        # Register
        shutil.move("bpe_statistics.txt", self.out_bpe_statistics.get_path())
        shutil.move("oov_words.json", self.out_oov_words.get_path())

        for (stat, fig) in [
            ("num_token_per_sequence", "histogram"),
            ("num_token_per_word", "histogram"),
            ("num_symbols_per_token", "histogram"),
            ("token_count_per_vocab", "plot"),
            ("token_count_per_corpus", "plot"),
        ]:
            shutil.move(
                f"mean_{stat}.txt", self.__dict__[f"out_mean_{stat}"].get_path()
            )
            shutil.move(
                f"{stat}_{fig}.pdf", self.__dict__[f"out_{stat}_{fig}"].get_path()
            )

            if stat in ["token_count_per_vocab", "token_count_per_corpus"]:
                shutil.move(f"{stat}.json", self.__dict__[f"out_{stat}"].get_path())
