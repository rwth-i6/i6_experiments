__all__ = ["SriLmSystem"]

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.report as report
import i6_core.lm.srilm as srilm

# -------------------- Init --------------------


@dataclass()
class SriLmModel:
    """
    holds a SRI Language Model with order, vocab and language model itself
    """

    order: int
    vocab: tk.Path
    ngram_lm: tk.Path


@dataclass()
class SriLmData:
    """
    hold data path and the info on text or count mode
    """

    data: tk.Path
    mode: srilm.ComputeNgramLmJob.DataMode


# -------------------- System --------------------


class SriLmSystem:
    def __init__(
        self,
        name: str,
        train_data: Dict[str, tk.Path],
        dev_data: Optional[Dict[str, tk.Path]],
        eval_data: Dict[str, tk.Path],
        ngram_order: [int],
        *,
        vocab: Optional[tk.Path] = None,  # TODO support several vocabs?
        ngram_args: Optional[List[str]] = None,
        perplexity_args: Optional[str] = None,
        srilm_path: Optional[tk.Path] = None,
        ngram_rqmt: Optional[Dict] = None,
        perplexity_rqmt: Optional[Dict] = None,
        mail_address: Optional[str] = None,
        prune_lm: bool = True,
    ):
        """
        :param name: System name used for alias/output and report naming
        :param train_data: Dict with different datasets used for training of individual LMs which will be interpolated
        :param dev_data: Dict with dev datasets used for PPL eval
        When optimize_discounts is true dev_data["dev"] will be used for optimization of the discounts
        :param eval_data: Dict with datasets used for PPL eval
        :param ngram_order: List of orders of LMs to be created
        :param vocab: Vocabulary file, text, one word per line
        :param ngram_args: Additional arguments for the ComputeNgramLmJob
        :param perplexity_args: Additional arguments for the ComputeNgramLmPerplexityJob
        :param srilm_path: Path to the srilm installation
        :param ngram_rqmt: Requirements for the ngram calculation, default values will be used if not set
        :param perplexity_rqmt: Requirements for the PPL calculation, default values will be used if not set
        :param mail_address: Reports will be sent to this E-Mail if set, else no report will be generated
        :param prune_lm: Whether to prune the LM with a Katz LM. Per convention Katz data is called background-data.
        Will use the LM of order n-1 if exists, else will use the LM of order n
        """
        assert "_" not in "-".join(list(train_data.keys()) + list(eval_data.keys())), "symbol not allowed in data name"

        self.name = name
        self.train_data = train_data
        self.dev_data = dev_data
        self.eval_data = eval_data
        self.ngram_order = ngram_order
        self.set_vocab = True if vocab is None else False
        self.vocab = vocab
        self.ngram_args = ngram_args
        self.perplexity_args = perplexity_args
        self.count_exe = srilm_path.join_right("ngram-count") if srilm_path is not None else tk.Path("ngram_count")
        self.ngram_exe = srilm_path.join_right("ngram") if srilm_path is not None else tk.Path("ngram")
        self.compute_best_mix_exe = (
            srilm_path.join_right("compute-best-mix") if srilm_path is not None else tk.Path("compute-best-mix")
        )

        self.ngram_rqmt = (
            ngram_rqmt
            if ngram_rqmt is not None
            else {"cpu_rqmt": 1, "time_rqmt": 24, "mem_rqmt": 16, "fs_rqmt": "100G"}
        )
        self.perplexity_rqmt = (
            perplexity_rqmt
            if perplexity_rqmt is not None
            else {"cpu_rqmt": 1, "time_rqmt": 4, "mem_rqmt": 32, "fs_rqmt": 50}
        )
        self.mail_address = mail_address
        self.prune_lm = prune_lm
        self._perplexity_interpolate_args = "-debug 2"

        self.ngram_lms: Dict[str, Dict[str, SriLmModel]] = defaultdict(dict)
        self.ppl_logs_for_interpolation: Dict[str, Dict[str, tk.Path]] = defaultdict(dict)
        self.ngrams_for_interpolation: Dict[str, Dict[str, SriLmModel]] = defaultdict(dict)
        self.interpolated_lms: Dict[str, Dict[str, SriLmModel]] = defaultdict(dict)
        self.perplexities: Dict[str, Union[str, float, tk.Variable]] = {}

    def compute_ngram_lms(self):
        """
        Compute the LMs for all different training datasets and orders
        """
        for train_corpus_name, train_data in self.train_data.items():
            for n in self.ngram_order:
                exp_name = f"{self.name}/{train_corpus_name}/{n}"
                if isinstance(train_data, SriLmData):
                    train_data = train_data.data
                    data_mode = train_data.mode
                else:
                    train_data = train_data
                    data_mode = srilm.ComputeNgramLmJob.DataMode.TEXT

                n_gram_rqmt = self.ngram_rqmt
                if n == 5:
                    n_gram_rqmt["mem_rqmt"] = 48
                if n == 5 and train_corpus_name in ["news-18pc", "background-data"]:
                    n_gram_rqmt["mem_rqmt"] = 72
                count_job = srilm.CountNgramsJob(
                    ngram_order=n,
                    data=train_data,
                    extra_count_args=["-sort"],
                    count_exe=self.count_exe,
                    **n_gram_rqmt,
                )
                count_job.add_alias(self.name + "/" + train_corpus_name + "/" + f"{n}gram/count_job")
                ngram_job = srilm.ComputeNgramLmJob(
                    ngram_order=n,
                    data=count_job.out_counts,
                    data_mode=srilm.ComputeNgramLmJob.DataMode.COUNT,
                    vocab=self.vocab,
                    extra_ngram_args=self.ngram_args,
                    count_exe=self.count_exe,
                    **n_gram_rqmt,
                )
                ngram_job.add_alias(self.name + "/" + train_corpus_name + "/" + f"{n}gram/compute_lm_job")
                ngram_lm = SriLmModel(order=n, vocab=ngram_job.out_vocab, ngram_lm=ngram_job.out_ngram_lm)

                self.ngram_lms[train_corpus_name][f"{n}gram"] = ngram_lm
                ngram_job.add_alias(f"lm/{exp_name}gram")
                tk.register_output(
                    f"lm/{exp_name}gram.lm.gz",
                    ngram_job.out_ngram_lm,
                )
                tk.register_output(
                    f"lm/{exp_name}gram.vocab",
                    ngram_job.out_vocab,
                )

    def train_katz_lm(self):
        """
        Trains the Katz LM used for pruning of the interpolated LM.
        """
        katz_data = self.train_data["background-data"]
        for dev_name in self.dev_data.keys():
            for n in self.ngram_order:

                ngram_job = srilm.ComputeNgramLmJob(
                    ngram_order=n,
                    data=katz_data,
                    data_mode=srilm.ComputeNgramLmJob.DataMode.TEXT,
                    vocab=self.vocab,
                    extra_ngram_args=[
                        "-gt3min 1",
                        "-gt4min 1",
                        "-gt5min 1",
                        "-gt6min 1",
                        "-interpolate",
                    ],
                    count_exe=self.count_exe,
                )
                ngram_lm = SriLmModel(order=n, vocab=ngram_job.out_vocab, ngram_lm=ngram_job.out_ngram_lm)
                self.ngram_lms["katz"][f"{n}gram"] = ngram_lm

                ppl_job = srilm.ComputeNgramLmPerplexityJob(
                    ngram_order=ngram_lm.order,
                    lm=ngram_lm.ngram_lm,
                    vocab=ngram_lm.vocab,
                    eval_data=self.dev_data[dev_name],
                    extra_ppl_args=self.perplexity_args,
                    ngram_exe=self.ngram_exe,
                    **self.perplexity_rqmt,
                )
                alias_output_name = f"lm/{self.name}/katz/{ngram_lm.order}_{dev_name}"
                self.perplexities[f"katz_{ngram_lm.order}_{dev_name}"] = ppl_job.out_ppl_score
                ppl_job.add_alias(alias_output_name)
                tk.register_output(
                    f"{alias_output_name}.ppl",
                    ppl_job.out_ppl_score,
                )

    def prune_with_katz(self):
        """
        Prune the interpolated LM with the previously trained helper Katz LM (usually of order n-1).
        """
        for dev_name in self.dev_data:
            for n in self.ngram_order:
                katz_order = n - 1 if f"{n-1}gram" in self.ngram_lms["katz"].keys() else n
                prune_job = srilm.PruneLMWithHelperLMJob(
                    ngram_order=n,
                    lm=self.interpolated_lms[dev_name][f"{n}gram"].ngram_lm,
                    prune_thresh=5e-10,
                    helper_lm=self.ngram_lms["katz"][f"{katz_order}gram"].ngram_lm,
                    ngram_exe=self.ngram_exe,
                    **self.perplexity_rqmt,
                )
                if n == 5:
                    prune_job.rqmt_run["mem"] = 64
                self.interpolated_lms[f"{dev_name}-pruned"][f"{n}gram"] = SriLmModel(n, self.vocab, prune_job.out_lm)
                tk.register_output(
                    f"lm/{n}gram.interpolated.pruned.dev.lm.gz",
                    prune_job.out_lm,
                )

    def _compute_ppl(
        self,
        name: str,
        ngram_lms: Dict[str, SriLmModel],
        eval_name: str,
        eval_path: tk.Path,
        ppl_for_interpolation: bool = False,
    ):
        for order_name, lm in ngram_lms.items():
            ppl_job = srilm.ComputeNgramLmPerplexityJob(
                ngram_order=lm.order,
                lm=lm.ngram_lm,
                vocab=lm.vocab,
                eval_data=eval_path,
                extra_ppl_args=self.perplexity_args if not ppl_for_interpolation else self._perplexity_interpolate_args,
                ngram_exe=self.ngram_exe,
                **self.perplexity_rqmt,
            )

            alias_output_name = f"lm/{self.name}/{name}/{order_name}_{eval_name}"

            if ppl_for_interpolation:
                alias_output_name += "_adapt"
                self.ppl_logs_for_interpolation[f"{order_name}_{eval_name}"][name] = ppl_job.out_ppl_log
                self.ngrams_for_interpolation[f"{order_name}_{eval_name}"][name] = lm
                self.perplexities[f"{name}_{order_name}_{eval_name}"] = ppl_job.out_ppl_score
            else:
                self.perplexities[f"{name}_{order_name}_{eval_name}"] = ppl_job.out_ppl_score

            ppl_job.add_alias(alias_output_name)
            tk.register_output(
                f"{alias_output_name}.ppl",
                ppl_job.out_ppl_score,
            )

    def compute_perplexities_of_ngram_lms(self):
        for train_name, v in self.ngram_lms.items():
            for eval_name, eval_path in self.eval_data.items():
                self._compute_ppl(train_name, v, eval_name, eval_path)

    def interpolate_ngram_lms(self):
        """
        Interpolate all LMs for each order separately
        """
        for dev_name, dev_path in self.dev_data.items():
            for train_name, v in self.ngram_lms.items():
                self._compute_ppl(train_name, v, dev_name, dev_path, ppl_for_interpolation=True)

            for n in self.ngram_order:
                order_name = f"{n}gram"
                ppl_log_list = list(self.ppl_logs_for_interpolation[f"{order_name}_{dev_name}"].values())
                lambdas = srilm.ComputeBestMixJob(
                    ppl_log_list,
                    compute_best_mix_exe=self.compute_best_mix_exe,
                ).out_weights
                ngrams = list(self.ngrams_for_interpolation[f"{order_name}_{dev_name}"].values())
                ngrams = [lm.ngram_lm for lm in ngrams]

                interpolate_job = srilm.InterpolateNgramLmJob(ngrams, lambdas, n, ngram_exe=self.ngram_exe)
                if n == 5:
                    interpolate_job.rqmt["mem"] = 64
                interpolate_job.add_alias(f"lms/{order_name}.interpolated.{dev_name}")
                self.interpolated_lms[dev_name][order_name] = SriLmModel(
                    n, self.vocab, interpolate_job.out_interpolated_lm
                )
                tk.register_output(
                    f"lm/{order_name}.interpolated.{dev_name}.lm.gz",
                    interpolate_job.out_interpolated_lm,
                )

    def compute_perplexities_of_interpolated_ngram_lms(self):
        for name, v in self.interpolated_lms.items():
            for eval_name, eval_path in self.eval_data.items():
                self._compute_ppl(name, v, eval_name, eval_path)

    def _format_report_perplexities(self, ppl_dict: Dict[str, Union[str, tk.Variable]]) -> str:
        out = [f"Report - {self.name}", "", "Results:"]

        max_size = max([len(x) for x in self.eval_data.keys()]) + 2
        eval_data = [x.ljust(max_size) for x in self.eval_data.keys()]
        data_header = "  ".join(eval_data)
        order_header = "order"
        col_header = order_header + data_header

        for train_name in self.train_data.keys():
            out.append(f"Train data: {train_name}")
            out.append(col_header)

            for order in self.ngram_order:
                out_str = str(order).ljust(len(order_header))
                for eval_name in self.eval_data.keys():
                    out_str += f'{ppl_dict[f"{train_name}_{order}gram_{eval_name}"].get():.2f}'.ljust(max_size)
                    out_str += " "
                out.append(out_str)

            out.append("")
        for dev_name in self.dev_data.keys():
            out.append(f"Dev data: {dev_name}")
            out.append(col_header)

            for order in self.ngram_order:
                out_str = str(order).ljust(len(order_header))
                for eval_name in self.eval_data.keys():
                    out_str += f'{ppl_dict[f"{dev_name}_{order}gram_{eval_name}"].get():.2f}'.ljust(max_size)
                    out_str += " "
                out.append(out_str)

        return "\n".join(out)

    def report_perplexities(self):
        report_job = report.GenerateReportStringJob(
            report_values=self.perplexities,
            report_template=self._format_report_perplexities,
        )
        tk.register_output(f"lm/reports/{self.name}.txt.gz", report_job.out_report)

        if self.mail_address is not None:
            mail_job = report.MailJob(
                report_job.out_report,
                subject=f"LM PPL: {self.name}",
                mail_address=self.mail_address,
                send_contents=True,
            )
            tk.register_output(f"lm/mail/{self.name}", mail_job.out_status)

    def run_training(self):
        """
        Runs the whole training of the LM.
        First separate LMs per text will be computed, then interpolated and (optionally) pruned with a helper LM.
        """
        self.compute_ngram_lms()

        self.compute_perplexities_of_ngram_lms()

        if len(self.train_data.keys()) > 1 and self.dev_data is not None:
            self.interpolate_ngram_lms()
            if self.prune_lm:
                self.train_katz_lm()
                self.prune_with_katz()
            self.compute_perplexities_of_interpolated_ngram_lms()
        else:
            self.interpolated_lms = {"no_interpol": list(self.ngram_lms.values())[0]}

        self.report_perplexities()
