__all__ = ["CheckSearchErrorsJob"]

from ast import literal_eval
from dataclasses import dataclass
import logging
from typing import Dict, Iterator, Tuple, Union

import i6_core.corpus as corpus_recipe
from i6_core.rasr import CommonRasrParameters, RasrConfig
from i6_core.recognition import LatticeToCtmJob, RescoreLatticeCacheJob, ScliteJob, IntersectStmCtm
from i6_core.text.processing import PipelineJob

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import DecodingJobs


from sisyphus import tk, Task

@dataclass(frozen=True)
class Segment:
    name: str
    text: str
    am_score: float = 0
    lm_score: float = 0
    lines: Tuple[str] = tuple()

    @property
    def total_score(self):
        return self.am_score + self.lm_score


def filter_and_score(stm, ctm, binary):
    ctm_cleaner = PipelineJob(
        ctm,
        pipeline=[
            r"grep -v -e '\[SILENCE\]' -e '!NULL'",  # remove [SILENCE] and NULL!
        ],
    )
    return ScliteJob(ref=stm, hyp=ctm_cleaner.out, sort_files=True, sctk_binary_path=binary)

class CheckSearchErrorsJob(tk.Job):
    def __init__(self, free_ctm: tk.Path, constraint_ctm: tk.Path, reference_text: tk.Path, clip_lm_score: bool = True):
        """Counts the number of search and modelling errors

        :param tk.Path free_ctm: output of a normal recognition in ctm format
        :param tk.Path constraint_ctm: output of a recognition constraint to the reference sentences in ctm format
        :param tk.Path reference_text: reference word sequences in a plaintext one sentence per line file
        """
        self.free_ctm = free_ctm
        self.constraint_ctm = constraint_ctm
        self.reference_text = reference_text
        self.clip_lm_score = clip_lm_score

        self.out_num_correct = self.output_var("correct", backup=1)
        self.out_num_search_errors = self.output_var("search_errors", backup=0)
        self.out_num_model_errors = self.output_var("model_errors", backup=0)
        self.out_num_am_errors = self.output_var("am_errors", backup=0)
        self.out_num_lm_errors = self.output_var("lm_errors", backup=0)
        self.out_num_skipped = self.output_var("skipped", backup=0)
        self.out_avg_search_error_score_diff = self.output_var("avg_search_error_score_diff")

        self.out_search_error_free_ctm = self.output_path("error_free.ctm")
        self.out_search_error_only_ctm = self.output_path("search_error.ctm")
        self.out_report = self.output_path("report.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        free_segments = self._parse_ctm(self.free_ctm.get_path())
        constraint_segments = self._parse_ctm(self.constraint_ctm.get_path())
        reference_segments = self._parse_reference(self.reference_text.get_path())

        num_correct = num_search_errors = num_model_errors = num_am_errors = num_lm_errors = num_skipped = 0
        sum_score_diff = 0.0
        with open(self.out_report, "wt") as out_report, open(self.out_search_error_free_ctm, "wt") as out_serr_free, open(self.out_search_error_only_ctm, "wt") as out_serr_only:
            for segment_name, reference_segment in reference_segments.items():
                free_segment = free_segments.get(segment_name)
                constraint_segment = constraint_segments.get(segment_name)
                if (
                    free_segment is None
                    or constraint_segment is None
                    or constraint_segment.text != reference_segment.text
                ):
                    logging.warning(f"skipping segment {segment_name} {free_segment is None} {constraint_segment is None}")
                    num_skipped += 1
                elif free_segment.text == constraint_segment.text:
                    num_correct += 1
                    out_report.write(f"{segment_name} Correct")
                    if abs((free_segment.lm_score - constraint_segment.lm_score) / free_segment.lm_score) > 0.01:
                        logging.warning(
                            f"LM-score for segment {free_segment.name} differs but they have the same text."
                        )
                    out_report.write("\n")
                    out_serr_free.write("".join(free_segment.lines))
                elif free_segment.total_score > constraint_segment.total_score:
                    num_search_errors += 1
                    sum_score_diff += free_segment.total_score - constraint_segment.total_score
                    out_report.write(
                        f"{segment_name} Search Error:\n"
                        f"{free_segment.total_score:.2f} > {constraint_segment.total_score:.2f} "
                        f"AM: {free_segment.am_score:.2f} {constraint_segment.am_score:.2f} "
                        f"LM: {free_segment.lm_score:.2f} {constraint_segment.lm_score:.2f} "
                        f"Diff: {(free_segment.total_score - constraint_segment.total_score):.2f}\n"
                    )
                    out_serr_free.write("".join(constraint_segment.lines))
                    out_serr_only.write("".join(free_segment.lines))
                else:
                    num_model_errors += 1
                    out_report.write(
                        f"{segment_name} Model Error:\n"
                        f"{free_segment.total_score:.2f} < {constraint_segment.total_score:.2f} "
                        f"AM: {free_segment.am_score:.2f} {constraint_segment.am_score:.2f} "
                        f"LM: {free_segment.lm_score:.2f} {constraint_segment.lm_score:.2f} "
                        f"Diff: {(free_segment.total_score - constraint_segment.total_score):.2f}\n"
                    )
                    am_error = free_segment.am_score < constraint_segment.am_score
                    lm_error = free_segment.lm_score < constraint_segment.lm_score
                    err = 'AM' if am_error else ''
                    err += '&' if am_error and lm_error else ''
                    err += 'LM' if lm_error else ''
                    if am_error:
                        num_am_errors += 1
                        out_report.write(f"- {err} ERROR LM_diff: {free_segment.am_score:.2f} < {constraint_segment.am_score:.2f}\n")
                    if lm_error:
                        num_lm_errors += 1
                        out_report.write(f"- {err} ERROR LM diff: {free_segment.lm_score:.2f} < {constraint_segment.lm_score:.2f}\n")
                    out_report.write("\n")
                    out_serr_free.write("".join(free_segment.lines))

        self.out_num_correct.set(num_correct)
        self.out_num_search_errors.set(num_search_errors)
        self.out_num_model_errors.set(num_model_errors)
        self.out_num_am_errors.set(num_am_errors)
        self.out_num_lm_errors.set(num_lm_errors)
        self.out_num_skipped.set(num_skipped)
        avg_error = sum_score_diff / num_search_errors if num_search_errors > 0 else 0
        self.out_avg_search_error_score_diff.set(avg_error)

    def _parse_reference(self, ref_path: str) -> Dict[str, Segment]:
        with open(ref_path, "rt") as in_text:
            ref: Dict[str, str] = literal_eval(in_text.read())
            res: Dict[str, Segment] = {}
            for k, v in ref.items():
                lines = []
                for idx, word in enumerate(v.split()):
                    rec = k.split("/")[1]  # get recording name assuming it the second name in the segment
                    lines.append(f"{rec} 1 {(idx * 0.1):.3f} {((idx + 1) * 0.1):.3f} {word} 1.0 0.0 0.0\n")
                res[k] = Segment(name=k, text=v, lines=tuple(lines))
            return res

    def _parse_ctm(self, ctm_path: str) -> Dict[str, Segment]:
        res: Dict[str, Segment] = {}
        am_score = 0.0
        lm_score = 0.0
        text: List[str] = []
        lines: List[str] = []
        segment_name = ""
        min_lm_score = 0.0 if self.clip_lm_score else float("-inf")
        with open(ctm_path, "rt") as in_ctm:
            for line in in_ctm:
                fields = line.strip().split()
                if line.startswith(";;"):
                    if text != []:
                        res[segment_name] = Segment(
                            name=segment_name, text=" ".join(text), am_score=am_score, lm_score=lm_score, lines=tuple(lines)
                        )
                    am_score = 0
                    lm_score = 0
                    text = []
                    lines = []
                    segment_name = fields[1]
                    continue
                am_score += float(fields[-2])
                lm_score += max(min_lm_score, float(fields[-1]))
                if ("[" not in fields[4]) and "!NULL" != fields[4]:
                    text += fields[4:-3]
                lines.append(line)
            if text != []:
                res[segment_name] = Segment(
                    name=segment_name, text=" ".join(text), am_score=am_score, lm_score=lm_score, lines=tuple(lines)
                )
        return res

def calculate_search_errors(name: str, recog_jobs: DecodingJobs, cheating_lattices, sctk_binary_path: Union[tk.Path, str]):
    """
    cheating_lattices is the output of a AdvancedTreeSearchJob that ran with LM of type cheating-segment
    """
    tmp_crp = CommonRasrParameters(recog_jobs.search_crp)

    filtered_corpus = corpus_recipe.FilterCorpusRemoveUnknownWordSegmentsJob(
        bliss_corpus=tmp_crp.corpus_config.file,
        bliss_lexicon=tmp_crp.lexicon_config.file,
        case_sensitive=False,
        all_unknown=False,
    )
    tmp_crp.corpus_config.file = filtered_corpus.out_corpus
    #tmp_crp.concurrent = 1

    tk.register_output("corpora/filtered.corpus.xml.gz", filtered_corpus.out_corpus)

    filtered_stm = corpus_recipe.CorpusToStmJob(filtered_corpus.out_corpus).out_stm_path

    full_search = recog_jobs.search
    config = full_search.config
    post_config = full_search.post_config

    tmp_crp.language_model_config = config.flf_lattice_tool.network.recognizer.lm
    tmp_crp.language_model_post_config = post_config.flf_lattice_tool.network.recognizer.lm


    test = RescoreLatticeCacheJob(crp=tmp_crp,
                                  lattice_cache=cheating_lattices)
    tk.register_output(f"cheating/{name}/rescored_lattices.cache.bundle", test.out_lattice_bundle)

    ec = RasrConfig()
    ec.flf_lattice_tool.network.dump_ctm.ctm.scores = "confidence am lm"
    ec.flf_lattice_tool.network.dump_ctm.ctm.dump_non_word = True
    ec.flf_lattice_tool.network.best.links = "apply-scale"
    ec.flf_lattice_tool.network.apply_scale.type = "multiply"
    ec.flf_lattice_tool.network.apply_scale.links = "dump-ctm"
    ec.flf_lattice_tool.network.apply_scale.key = "lm"
    ec.flf_lattice_tool.network.apply_scale.scale = tmp_crp.language_model_config.scale

    convert_cheating = LatticeToCtmJob(tmp_crp, test.out_lattice_bundle, extra_config=ec)
    tk.register_output(f"cheating/{name}/cheating.ctm", convert_cheating.out_ctm_file)



    scorer = filter_and_score(filtered_stm, convert_cheating.out_ctm_file, sctk_binary_path)
    tk.register_output(f"cheating/{name}/cheating.wer", scorer.out_report_dir)

    convert_full = LatticeToCtmJob(tmp_crp, full_search.out_lattice_bundle, extra_config=ec)
    tk.register_output(f"cheating/{name}/free.ctm", convert_full.out_ctm_file)

    reference_text = corpus_recipe.CorpusToTextDictJob(bliss_corpus=tmp_crp.corpus_config.file).out_dictionary
    check_errors = CheckSearchErrorsJob(convert_full.out_ctm_file, convert_cheating.out_ctm_file, reference_text,
                                        True)
    tk.register_output(f"cheating/{name}/error.report", check_errors.out_report)

    scorer = filter_and_score(filtered_stm, convert_full.out_ctm_file, sctk_binary_path)
    tk.register_output(f"cheating/{name}/filtered.wer", scorer.out_report_dir)

    scorer = filter_and_score(filtered_stm, check_errors.out_search_error_free_ctm, sctk_binary_path)
    tk.register_output(f"cheating/{name}/no_search_error.wer", scorer.out_report_dir)

    intersect = IntersectStmCtm(filtered_stm, check_errors.out_search_error_only_ctm)
    scorer = filter_and_score(intersect.out_stm, intersect.out_ctm, sctk_binary_path)
    tk.register_output(f"cheating/{name}/only_search_error.wer", scorer.out_report_dir)