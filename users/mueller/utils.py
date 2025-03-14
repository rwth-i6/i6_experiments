import string
import copy

from i6_core.returnn import ReturnnConfig as RF
from i6_core.util import instanciate_delayed
from i6_experiments.common.setups.serialization import PartialImport as PI
from i6_experiments.users.mann.nn.util import DelayedCodeWrapper
from sisyphus.hash import sis_hash_helper
from sisyphus import Job, Task, tk
from i6_core.util import uopen
from i6_experiments.users.mueller.datasets.librispeech import _get_corpus_text_dict, TextDictToTextLinesJob
from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
from i6_experiments.users.mueller.experiments.language_models.n_gram import ApplyBPEToTextJob
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt


class PartialImportCustom(PI):
    def get(self) -> str:
        arguments = {**self.hashed_arguments}
        arguments.update(self.unhashed_arguments)
        print(arguments)
        return string.Template(self.TEMPLATE).substitute(
            {
                "KWARGS": str(instanciate_delayed(arguments)),
                "IMPORT_PATH": self.module,
                "IMPORT_NAME": self.object_name,
                "OBJECT_NAME": self.import_as if self.import_as is not None else self.object_name,
            }
        )
        
class ReturnnConfigCustom(RF):
    def __init__(
        self,
        config,
        post_config=None,
        staged_network_dict=None,
        *,
        python_prolog=None,
        python_prolog_hash=None,
        python_epilog="",
        python_epilog_hash=None,
        hash_full_python_code=False,
        sort_config=True,
        pprint_kwargs=None,
        black_formatting=True,
    ):
        if python_prolog_hash is None and python_prolog is not None:
            python_prolog_hash = []
            
        super().__init__(
            config=config,
            post_config=post_config,
            staged_network_dict=staged_network_dict,
            python_prolog=python_prolog,
            python_prolog_hash=python_prolog_hash,
            python_epilog=python_epilog,
            python_epilog_hash=python_epilog_hash,
            hash_full_python_code=hash_full_python_code,
            sort_config=sort_config,
            pprint_kwargs=pprint_kwargs,
            black_formatting=black_formatting,
        )
        
        if self.python_prolog_hash == []:
            self.python_prolog_hash = None
    
    def _sis_hash(self):
        conf = copy.deepcopy(self.config)
        if "preload_from_files" in conf:
            for v in conf["preload_from_files"].values():
                if "filename" in v and isinstance(v["filename"], DelayedCodeWrapper):
                    v["filename"] = v["filename"].args[0]
        h = {
            "returnn_config": conf,
            "python_epilog_hash": self.python_epilog_hash,
            "python_prolog_hash": self.python_prolog_hash,
        }
        if self.staged_network_dict:
            h["returnn_networks"] = self.staged_network_dict

        return sis_hash_helper(h)
    
    
class DataSetStatsJob(Job):

    def __init__(self, bpe_text: tk.Path):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param output_gzip: gzip the output
        """
        self.bpe_text = bpe_text
        self.out_count_results = self.output_path("count_results.py")

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        f = uopen(self.bpe_text.get_path(), "rt").read()
        if isinstance(f, str):
            d = f.split("\n")
            assert isinstance(d, list)
        else:
            d = eval(f, {"nan": float("nan"), "inf": float("inf")})
            assert isinstance(d, dict)  # seq_tag -> bpe string
            d = d.items()
        with uopen(self.out_count_results, "wt") as out:
            out.write("{\n")
            num_repeated = 0
            num_words_total = 0
            num_seqs_total = len(d)
            nbest = 0
            max_seq_len = 0
            min_seq_len = 10000000000
            for entry in d:
                if isinstance(entry, list):
                    # n-best list as [(score, text), ...]
                    nbest=len(entry)
                    for score, text in entry:
                        prev = len(text.split(" "))
                        if prev > max_seq_len:
                            max_seq_len = prev
                        if prev < min_seq_len:
                            min_seq_len = prev
                        num_words_total += prev
                        new = len(self._filter(text).split(" "))
                        num_repeated += (prev - new)
                else:
                    nbest=1
                    prev = len(entry.split(" "))
                    if prev > max_seq_len:
                        max_seq_len = prev
                    if prev < min_seq_len:
                        min_seq_len = prev
                    num_words_total += prev
                    new = len(self._filter(entry).split(" "))
                    num_repeated += (prev - new)
            out.write("%r: %r,\n" % ("Repeated", num_repeated))
            out.write("%r: %r,\n" % ("Nr of BPEs", num_words_total))
            out.write("%r: %r,\n" % ("Repeated Percentage", (num_repeated / num_words_total) * 100))
            out.write("%r: %r,\n" % ("Nbest", nbest))
            out.write("%r: %r,\n" % ("Nr of Seqs", num_seqs_total))
            out.write("%r: %r,\n" % ("BPEs per Seqs", (num_seqs_total / num_words_total)))
            out.write("%r: %r,\n" % ("Min BPEs for Seqs", min_seq_len))
            out.write("%r: %r,\n" % ("Max BPEs for Seqs", max_seq_len))
            out.write("}\n")

    def _filter(self, txt: str) -> str:
        tokens = txt.split(" ")
        tokens = [t1 for (t1, t2) in zip(tokens, [None] + tokens) if t1 != t2]
        return " ".join(tokens)
    
def calc_stats(vocab: Bpe):
    subword_nmt = get_returnn_subword_nmt()
    
    for corpus in ["train-other-960", "dev-clean", "dev-other", "test-clean", "test-other"]:
        train_corpus_text_dict = _get_corpus_text_dict(corpus)
        t_data = TextDictToTextLinesJob(train_corpus_text_dict, gzip=True).out_text_lines
        t_text = ApplyBPEToTextJob(
            text_file=t_data,
            bpe_codes=vocab.codes,
            bpe_vocab=tk.Path(vocab.vocab.get_path()[:-5] + "dummy_count.vocab"),
            subword_nmt_repo=subword_nmt,
            gzip_output=True,
            mini_task=False,
        ).out_bpe_text
        cnt = DataSetStatsJob(t_text).out_count_results
        tk.register_output(f"datasets/LibriSpeech/stats/{corpus}_ground_truth", cnt)