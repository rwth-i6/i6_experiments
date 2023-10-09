from sisyphus import *
import subprocess


class ComputeWordEmitLatencyJob(Job):
    """
    Computes the word emit latency for a given chunked output hdf (e.g output from chunked AED model)
    """

    def __init__(
        self,
        latency_script_path,
        allophones_file,
        alignment_cache,
        bliss_corpus,
        lexicon,
        chunked_output_hdf,
        bpe_vocab,
        left_padding,
        chunk_size,
        chunk_stride,
        python_exec,
    ):
        """
        :param tk.Path latency_script_path: path to latency python script
        :param tk.Path allophones_file: path to allophones file
        :param tk.Path alignment_cache: path to alignment cache
        :param tk.Path bliss_corpus: path to bliss corpus
        :param tk.Path lexicon: path to lexicon
        :param tk.Path chunked_output_hdf: path to chunked output hdf
        :param tk.Path bpe_vocab: bpe vocab size
        :param int left_padding: left padding size
        :param int chunk_size:
        :param int chunk_stride:
        :param tk.Path python_exec: python executable path
        """
        self.latency_script_path = latency_script_path
        self.allphones_file = allophones_file
        self.alignment_cache = alignment_cache
        self.bliss_corpus = bliss_corpus
        self.lexicon = lexicon
        self.chunked_output_hdf = chunked_output_hdf
        self.bpe_vocab = bpe_vocab
        self.left_padding = left_padding
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.python_exec = python_exec

        self.out_dummy_value = self.output_var("latency")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        subprocess.check_call(
            [
                self.python_exec,
                self.latency_script_path,
                "--allophone-file",
                self.allphones_file,
                "--phone-alignments",
                self.alignment_cache,
                "--corpus",
                self.bliss_corpus,
                "--lexicon",
                self.lexicon,
                "--chunk-labels",
                self.chunked_output_hdf,
                "--chunk-bpe-vocab",
                self.bpe_vocab,
                "--chunk-left-padding",
                str(self.left_padding),
                "--chunk-stride",
                str(self.chunk_stride),
                "--chunk-size",
                str(self.chunk_size),
            ]
        )
        self.out_dummy_value.set(1.0)
